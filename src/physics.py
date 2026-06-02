# physics.py – 物理损失函数
# 包含两个核心物理损失计算器：
#   SVEsPhysicsLoss        → 二维浅水方程 (Saint-Venant Equations) FVM 残差
#   SedimentTransportLoss  → 总输沙输移方程 FVM 残差
# 共用 _CachedMeshTensors 基类实现 GPU 张量缓存。

import numpy as np
import torch
from types import SimpleNamespace

from .config import EPS_DIVISION, EPS_SAFE, EPS_VELOCITY_CLAMP
from .model import FlowPINN, SedimentPINN
from .utils import build_xyt, match_closure, smooth_positive, time_derivative


# ═══════════════════════════════════════════════════════════════
#  GPU 张量缓存基类
# ═══════════════════════════════════════════════════════════════

class _CachedMeshTensors:
    """GPU 张量缓存混入基类。

    各 Loss 类每次 compute_loss 都需要高斯点坐标/法向/权重张量。
    本基类在首次调用时将 NumPy 数组转为 GPU 张量并缓存，
    后续调用直接复用，避免重复创建开销。
    """

    def _init_cache(self, fvm_mesh):
        self._mesh = fvm_mesh
        self._cached_device = None
        self._gauss_coords_t = None
        self._gauss_normals_t = None
        self._gauss_weights_t = None

    def _ensure_tensors(self, device):
        """惰性创建并缓存高斯积分点的 GPU 张量，设备不变时跳过。"""
        if self._cached_device == device:
            return
        self._gauss_coords_t = torch.tensor(
            self._mesh.gauss_coords, dtype=torch.float32, device=device)
        self._gauss_normals_t = torch.tensor(
            self._mesh.gauss_normals, dtype=torch.float32, device=device)
        self._gauss_weights_t = torch.tensor(
            self._mesh.gauss_weights, dtype=torch.float32, device=device).unsqueeze(1)
        self._cached_device = device


# ═══════════════════════════════════════════════════════════════
#  浅水方程 (SVEs) 物理损失
# ═══════════════════════════════════════════════════════════════

class SVEsPhysicsLoss(_CachedMeshTensors):
    """二维浅水方程 FVM 物理损失。

    基于有限体积法在每个单元上积分连续性方程和动量方程，
    计算边界通量积分、床面坡度源项和 Manning 摩擦项的残差。

    参数:
        fvm_mesh: FVMeshPreprocessor 实例
        g: 重力加速度 (m/s²)
        n_manning: Manning 糙率系数
        bounds: 坐标归一化边界字典
        typical_depth: 典型水深，用于物理量反归一化
        typical_velocity: 典型流速，用于物理量反归一化
    """

    def __init__(self, fvm_mesh, g=9.81, n_manning=0.01, bounds=None,
                 typical_depth=10.0, typical_velocity=1.0,
                 simulation_time=1.0, include_time_terms=True, eps=EPS_SAFE):
        self.mesh = fvm_mesh
        self.g = g
        self.n = n_manning
        self.bounds = bounds
        self.typical_h = typical_depth
        self.typical_u = typical_velocity
        self.simulation_time = simulation_time
        self.include_time_terms = include_time_terms
        self.eps = eps
        cell_area = fvm_mesh.cell_area
        cell_perimeter = 4.0 * fvm_mesh.resolution
        time_scale = max(simulation_time, EPS_DIVISION)
        depth_scale = max(typical_depth, EPS_DIVISION)
        velocity_scale = max(typical_velocity, EPS_DIVISION)
        continuity_scale = (
            cell_area * depth_scale / time_scale
            + cell_perimeter * depth_scale * velocity_scale
        )
        momentum_scale = (
            cell_area * depth_scale * velocity_scale / time_scale
            + cell_perimeter * (
                depth_scale * velocity_scale ** 2
                + 0.5 * self.g * depth_scale ** 2
            )
        )
        self.continuity_scale = 1.0 / max(continuity_scale, EPS_DIVISION)
        self.momentum_scale = 1.0 / max(momentum_scale, EPS_DIVISION)
        self._init_cache(fvm_mesh)

    def compute_loss(self, model, t, device):
        """计算 SVEs 物理损失。

        返回:
            total_loss: 连续性 + x动量 + y动量 损失之和
            loss_dict: 各分项损失值字典
        """
        self._ensure_tensors(device)
        gauss_coords = self._gauss_coords_t
        gauss_normals = self._gauss_normals_t
        gauss_weights = self._gauss_weights_t

        # 构建输入并前向推理
        xyt = build_xyt(
            gauss_coords, t, self.bounds, device,
            requires_grad=self.include_time_terms,
        )
        outputs = model(xyt)
        h, u, v = FlowPINN.decode_output(outputs, self.typical_h, self.typical_u)

        # 几何量准备
        nx = gauss_normals[:, 0:1]  # x方向法向量分量，形状 [N, 1]
        ny = gauss_normals[:, 1:2]  # y方向法向量分量
        Nc = self.mesh.n_cells
        npp = self.mesh.n_points_per_cell
        weights_r = gauss_weights.view(Nc, npp)
        # 重塑为单元-点结构
        h_r = h.view(Nc, npp)
        u_r = u.view(Nc, npp)
        v_r = v.view(Nc, npp)
        h_cell = torch.mean(h_r, dim=1, keepdim=True)   # 单元平均水深

        # ① 连续性方程：∂/∂t∫h dA + ∮ (hu·nx + hv·ny) dS = 0
        
        # 平均水深时间导数
        h_t_cell = torch.mean(
            time_derivative(h, xyt, self.simulation_time, self.include_time_terms).view(Nc, npp),
            dim=1,
            keepdim=True,
        )   
        vol_h = h_t_cell * self.mesh.cell_area
        # 边界通量：hu·nx + hv·ny，沿边界积分
        flux_h = (h * u) * nx + (h * v) * ny
        bnd_h = torch.sum(flux_h.view(Nc, npp) * weights_r, dim=1, keepdim=True)
        cont_res = (vol_h + bnd_h) * self.continuity_scale
        loss_continuity = torch.mean(cont_res ** 2)

        # ② 床面坡度源项：S = -g·h·∇zb·A
        dzb_dx_t, dzb_dy_t = self.mesh.get_bed_gradient(device)
        slope_x = -self.g * h_cell * dzb_dx_t.unsqueeze(1) * self.mesh.cell_area
        slope_y = -self.g * h_cell * dzb_dy_t.unsqueeze(1) * self.mesh.cell_area

        # ③ Manning 摩擦项：τ = g·n²·|U|·u / h^(1/3)
        h_safe = torch.clamp(h_r, min=0.05)
        vel_mag = torch.sqrt(u_r ** 2 + v_r ** 2 + self.eps)
        fric_tx = torch.mean(
            self.g * self.n ** 2 * vel_mag * u_r / torch.pow(h_safe, 1. / 3.),
            dim=1, keepdim=True
        ) * self.mesh.cell_area
        fric_ty = torch.mean(
            self.g * self.n ** 2 * vel_mag * v_r / torch.pow(h_safe, 1. / 3.),
            dim=1, keepdim=True
        ) * self.mesh.cell_area

        # ④ x 方向动量方程：
        flux_mx = (h * u * u + 0.5 * self.g * h * h) * nx + (h * u * v) * ny
        bnd_mx = torch.sum(flux_mx.view(Nc, npp) * weights_r, dim=1, keepdim=True)
        hu_t_cell = torch.mean(
            time_derivative(h * u, xyt, self.simulation_time, self.include_time_terms).view(Nc, npp),
            dim=1,
            keepdim=True,
        )
        vol_mx = hu_t_cell * self.mesh.cell_area
        mom_x_res = (vol_mx + bnd_mx - slope_x + fric_tx) * self.momentum_scale
        loss_momentum_x = torch.mean(mom_x_res ** 2)

        # ⑤ y 方向动量方程
        flux_my = (h * v * u) * nx + (h * v * v + 0.5 * self.g * h * h) * ny
        bnd_my = torch.sum(flux_my.view(Nc, npp) * weights_r, dim=1, keepdim=True)
        hv_t_cell = torch.mean(
            time_derivative(h * v, xyt, self.simulation_time, self.include_time_terms).view(Nc, npp),
            dim=1,
            keepdim=True,
        )
        vol_my = hv_t_cell * self.mesh.cell_area
        mom_y_res = (vol_my + bnd_my - slope_y + fric_ty) * self.momentum_scale
        loss_momentum_y = torch.mean(mom_y_res ** 2)

        total_loss = loss_continuity + loss_momentum_x + loss_momentum_y
        loss_dict = {
            'continuity': loss_continuity.item(),
            'momentum_x': loss_momentum_x.item(),
            'momentum_y': loss_momentum_y.item(),
            'total': total_loss.item(),
        }
        return total_loss, loss_dict


# ═══════════════════════════════════════════════════════════════
#  输沙输移方程损失
# ═══════════════════════════════════════════════════════════════

class SedimentTransportLoss(_CachedMeshTensors):
    """总输沙输移方程损失。
    - 总输沙浓度 C_tk 的有限体积对流扩散残差
    - 输沙容量闭合损失（C_tk 趋近平衡浓度 C_capacity）

    核心流程通过 compute_closure() 统一获取所有中间量（h, u, v, C_tk, p_k, E/D），
    供 compute_loss / compute_sediment_loss 复用。

    参数:
        fvm_mesh: FVM 网格
        bounds: 坐标归一化边界
        typical_depth / typical_velocity: 物理量级
        adaptation_length: 输沙适应长度 (m)
        rho_s: 泥沙密度 (kg/m³)
        w_capacity: 容量损失权重
    """

    def __init__(
        self,
        fvm_mesh,
        bounds=None,
        typical_depth=10.0,
        typical_velocity=1.0,
        beta_default=1.0,
        epsilon_default=0.1,
        residual_scale=1.0,
        grain_diameters=None,
        adaptation_length=50.0,
        rho_s=2650.0,
        rho_w=1000.0,
        g=9.81,
        n_manning=0.01,
        kinematic_viscosity=1.0e-6,
        wu_theta_cr=0.03,
        skin_shear_factor=1.0,
        simulation_time=1.0,
        include_time_terms=True,
        alpha_active_layer=10.0,
        w_capacity=0.05,
        w_initial_sediment=1.0,
        initial_sediment_concentration=None,
        w_inlet_sediment=1.0,
        inlet_sediment_concentration=None,
        source_sharpness=EPS_VELOCITY_CLAMP,
    ):
        self.mesh = fvm_mesh
        self.bounds = bounds
        self.typical_h = typical_depth
        self.typical_u = typical_velocity
        self.beta_default = beta_default
        self.epsilon_default = epsilon_default
        self.residual_scale = residual_scale
        self.grain_diameters = grain_diameters
        self.adaptation_length = adaptation_length
        self.rho_s = rho_s
        self.rho_w = rho_w
        self.g = g
        self.n_manning = n_manning
        self.kinematic_viscosity = kinematic_viscosity
        self.wu_theta_cr = wu_theta_cr
        self.skin_shear_factor = skin_shear_factor
        self.simulation_time = simulation_time
        self.include_time_terms = include_time_terms
        self.alpha_active_layer = alpha_active_layer
        self.w_capacity = w_capacity
        self.w_initial_sediment = w_initial_sediment
        self.initial_sediment_concentration = initial_sediment_concentration
        self.w_inlet_sediment = w_inlet_sediment
        self.inlet_sediment_concentration = inlet_sediment_concentration
        self.source_sharpness = source_sharpness
        self.closure_formula = ClosureFormulation(SimpleNamespace(
            rho_w=self.rho_w,
            rho_s=self.rho_s,
            g=self.g,
            n_manning=self.n_manning,
            kinematic_viscosity=self.kinematic_viscosity,
            wu_theta_cr=self.wu_theta_cr,
            skin_shear_factor=self.skin_shear_factor,
        ))
        self._init_cache(fvm_mesh)

    def _dt(self, q, xyt):
        """Return physical-time derivative from normalized network time."""
        return time_derivative(q, xyt, self.simulation_time, self.include_time_terms)

    def _physical_grad(self, q, xyt, dim):
        """Return physical-space derivative for x(dim=0) or y(dim=1)."""
        grad = SedimentPINN._grad(q, xyt, dim)
        if self.bounds is None:
            return grad
        if dim == 0:
            scale = self.bounds['x_max'] - self.bounds['x_min']
        elif dim == 1:
            scale = self.bounds['y_max'] - self.bounds['y_min']
        else:
            raise ValueError("dim must be 0 or 1 for physical-space gradients.")
        return grad / max(scale, EPS_DIVISION)

    def total_load_fvm_loss(self, xyt, h, u, v, C_tk, beta_tk, epsilon_thk, E_tk, D_tk):
        """总输沙方程有限体积残差。

        Integral form:
            d/dt ∫ storage dA + ∮ advective_flux dS
            - ∮ diffusive_flux dS - ∫(E-D)dA = 0
        """
        self._ensure_tensors(C_tk.device)
        Nc = self.mesh.n_cells
        npp = self.mesh.n_points_per_cell
        K = C_tk.shape[1]

        weights = self._gauss_weights_t.view(Nc, npp, 1)
        nx = self._gauss_normals_t[:, 0:1]
        ny = self._gauss_normals_t[:, 1:2]

        beta_safe = torch.clamp(beta_tk, min=EPS_DIVISION)
        # 输沙存储项：storage = h * C_tk / beta_tk，包含输沙修正系数 beta_tk
        storage = h * C_tk / beta_safe
        storage_t = self._dt(storage, xyt)
        volume_storage = torch.mean(storage_t.view(Nc, npp, K), dim=1) * self.mesh.cell_area
        # 对流通量：advective_flux = h * C_tk * (u * nx + v * ny)，包含输沙浓度 C_tk 和水流通量，沿边界积分
        adv_flux = h * C_tk * (u * nx + v * ny)
        boundary_advection = torch.sum(adv_flux.view(Nc, npp, K) * weights, dim=1)
        # 扩散通量：diffusive_flux = ε * h * ∇C_tk，包含扩散系数 ε 和浓度梯度，沿边界积分
        Cx = self._physical_grad(C_tk, xyt, 0)
        Cy = self._physical_grad(C_tk, xyt, 1)
        diff_flux = epsilon_thk * h * (Cx * nx + Cy * ny)
        boundary_diffusion = torch.sum(diff_flux.view(Nc, npp, K) * weights, dim=1)
        # 源汇项：source = E_tk - D_tk，包含侵蚀率 E_tk 和沉积率 D_tk，在单元内积分
        source = E_tk - D_tk
        volume_source = torch.mean(source.view(Nc, npp, K), dim=1) * self.mesh.cell_area
        # 总残差 = 存储项 + 对流项 - 扩散项 - 源汇项，平方后平均得到损失值
        residual = volume_storage + boundary_advection - boundary_diffusion - volume_source
        loss = torch.mean((residual * self.residual_scale) ** 2)
        return loss, residual

    # 统一闭合计算

    def compute_closure(
        self,
        sediment_model,
        flow_model,
        T_norm,
        device,
        p_k_override=None,
        freeze_flow_params=True,
    ):
        """计算所有中间物理量，供输沙 PDE、Exner 方程和级配方程共用。

        返回字典包含：xyt, h, u, v, C_tk, p_k, beta_tk, epsilon_thk,
        E_tk, D_tk, C_capacity, vel_mag。
        """
        self._ensure_tensors(device)
        xyt = build_xyt(
            self._gauss_coords_t, T_norm, self.bounds,
            device, requires_grad=True,
        )

        # 冻结水动力模型参数，防止输沙训练时更新 flow_model
        old_requires_grad = None
        if freeze_flow_params:
            old_requires_grad = [p.requires_grad for p in flow_model.parameters()]
            for p in flow_model.parameters():
                p.requires_grad_(False)

        try:
            flow_out = flow_model(xyt)
            h, u, v = FlowPINN.decode_output(flow_out, self.typical_h, self.typical_u)
        finally:
            if old_requires_grad is not None:
                for p, old_flag in zip(flow_model.parameters(), old_requires_grad):
                    p.requires_grad_(old_flag)

        # 输沙浓度
        C_tk = sediment_model(xyt)
        
        # 级配
        if p_k_override is not None:
            p_k = match_closure(p_k_override, C_tk, 1.0 / C_tk.shape[1])
        else:
            p_k = torch.ones_like(C_tk) / C_tk.shape[1]

        # 闭合关系计算
        beta_tk = torch.ones_like(C_tk) * self.beta_default # 输沙修正系数
        epsilon_thk = torch.ones_like(C_tk) * self.epsilon_default  # 扩散系数
        vel_mag = torch.sqrt(u ** 2 + v ** 2 + EPS_DIVISION)    # 速度模长
        d_k = torch.tensor(self.grain_diameters, dtype=C_tk.dtype, device=C_tk.device)  # 分粒径
        # Wu (2000) 输沙潜力和容量计算，包含床载和悬移分量，以及相关剪应力和沉速诊断量
        q_capacity, C_capacity, wu_diag = self.closure_formula.TransportPotential_Wu(h, u, v, d_k, p_k)
        # 适应时间 = 适应长度 / 速度模长，避免除以零时过大值导致数值不稳定
        adapt_time = self.adaptation_length / torch.clamp(vel_mag, min=EPS_VELOCITY_CLAMP)  # 适应时间
        net_source = h * (C_capacity - C_tk) / adapt_time               # 净源汇项
        # 使用平滑的正负分离函数，确保侵蚀率和沉积率非负且数值稳定
        E_tk = smooth_positive(net_source, sharpness=self.source_sharpness)   # 侵蚀率
        D_tk = smooth_positive(-net_source, sharpness=self.source_sharpness)  # 沉积率

        return {
            'xyt': xyt, 'h': h, 'u': u, 'v': v,
            'C_tk': C_tk, 'p_k': p_k,
            'beta_tk': beta_tk, 'epsilon_thk': epsilon_thk,
            'E_tk': E_tk, 'D_tk': D_tk,
            'C_capacity': C_capacity, 'vel_mag': vel_mag,
            'q_capacity': q_capacity,
            'q_b': wu_diag['q_b'],
            'q_s': wu_diag['q_s'],
            'tau_b': wu_diag['tau_b'],
            'tau_skin': wu_diag['tau_skin'],
            'tau_cr': wu_diag['tau_cr'],
            'fall_velocity': wu_diag['ws'],
        }

    # 仅输沙损失（独立更新 sediment_model）

    def compute_sediment_loss(
        self,
        sediment_model,
        flow_model,
        T_norm,
        device,
        p_k_override=None,
    ):
        """计算仅包含输沙 PDE 残差的损失，用于独立训练 sediment_model 时调用。"""
        closure = self.compute_closure(
            sediment_model, flow_model, T_norm, device,
            p_k_override=p_k_override,
            freeze_flow_params=True,
        )
        xyt     = closure['xyt']
        h, u, v = closure['h'], closure['u'], closure['v']
        C_tk    = closure['C_tk']
        E_tk    = closure['E_tk']
        D_tk    = closure['D_tk']
        C_capacity = closure['C_capacity']
        vel_mag    = closure['vel_mag']

        transport_loss, residual = self.total_load_fvm_loss(
            xyt, h, u, v, C_tk,
            closure['beta_tk'], closure['epsilon_thk'], E_tk, D_tk,
        )

        capacity_loss = torch.mean((C_tk - C_capacity.detach()) ** 2)
        initial_loss = self.initial_condition_loss(C_tk) if abs(float(T_norm)) < 1.0e-8 else torch.zeros_like(capacity_loss)
        inlet_loss = self.inlet_boundary_loss(C_tk)

        loss = (
            transport_loss
            + self.w_capacity * capacity_loss
            + self.w_initial_sediment * initial_loss
            + self.w_inlet_sediment * inlet_loss
        )
        
        return loss, {
            'transport': transport_loss.item(),
            'capacity': capacity_loss.item(),
            'initial': initial_loss.item(),
            'inlet': inlet_loss.item(),
            'residual_mean': torch.mean(torch.abs(residual)).item(),
            'C_min': torch.min(C_tk).item(),
            'C_max': torch.max(C_tk).item(),
            'Ceq_mean': torch.mean(C_capacity).item(),
            'U_mean': torch.mean(vel_mag).item(),
        }

    def initial_condition_loss(self, C_tk):
        if self.initial_sediment_concentration is None:
            return torch.zeros((), dtype=C_tk.dtype, device=C_tk.device)
        target = torch.as_tensor(
            self.initial_sediment_concentration,
            dtype=C_tk.dtype,
            device=C_tk.device,
        ).view(1, -1)
        if target.shape[1] != C_tk.shape[1]:
            raise ValueError("初始泥沙浓度维度必须与泥沙粒径组数量一致。")
        return torch.mean((C_tk - target.expand_as(C_tk)) ** 2)

    def inlet_boundary_loss(self, C_tk):
        if self.inlet_sediment_concentration is None:
            return torch.zeros((), dtype=C_tk.dtype, device=C_tk.device)
        target = torch.as_tensor(
            self.inlet_sediment_concentration,
            dtype=C_tk.dtype,
            device=C_tk.device,
        ).view(1, -1)
        if target.shape[1] != C_tk.shape[1]:
            raise ValueError("入口泥沙浓度维度必须与泥沙粒径组数量一致。")

        coords = self._gauss_coords_t
        neighbor_id = torch.as_tensor(self.mesh.gauss_neighbor_id, dtype=torch.long, device=C_tk.device)
        inlet_x = float(self.mesh.bbox['xmin'])
        inlet_mask = (neighbor_id < 0) & torch.isclose(
            coords[:, 0],
            torch.as_tensor(inlet_x, dtype=coords.dtype, device=coords.device),
            atol=max(float(self.mesh.resolution) * 1.0e-6, 1.0e-6),
            rtol=0.0,
        )
        if not torch.any(inlet_mask):
            return torch.zeros((), dtype=C_tk.dtype, device=C_tk.device)
        inlet_c = C_tk[inlet_mask]
        return torch.mean((inlet_c - target.expand_as(inlet_c)) ** 2)


class MorphodynamicsUpdater:
    """床变和两层级配显式更新。"""

    def __init__(
        self,
        fvm_mesh,
        device,
        sediment_transport_loss_fn,
        porosity=0.4,
        bed_slope_coefficient=0.2,
        history=None,
    ):
        self.mesh = fvm_mesh
        self.device = device
        self.sediment_transport_loss_fn = sediment_transport_loss_fn
        self.porosity = porosity
        self.bed_slope_coefficient = bed_slope_coefficient
        self.rho_bulk = self.sediment_transport_loss_fn.rho_s * (1.0 - porosity)
        self.last_bed = self.mesh.zb.copy()
        self.window_dt = 1.0
        self.window_dt_current = 1.0
        self.history = history

        n_grains = len(self.sediment_transport_loss_fn.grain_diameters)
        self.active_layer_frac = np.full(
            (self.mesh.n_cells, n_grains),
            1.0 / n_grains,
            dtype=np.float32,
        )
        self.second_layer_frac = self.active_layer_frac.copy()
        self.delta1 = self.active_layer_thickness_np(self.active_layer_frac)
        self.delta2 = np.full(self.mesh.n_cells, 1.0, dtype=np.float32)

    def gradation_at_gauss_tensor(self):
        p = self.active_layer_frac[self.mesh.gauss_cell_id]
        return torch.tensor(p, dtype=torch.float32, device=self.device)

    def active_layer_thickness_np(self, fractions):
        d = np.asarray(self.sediment_transport_loss_fn.grain_diameters, dtype=np.float32)
        order = np.argsort(d)
        d_sorted = d[order]
        f_sorted = fractions[:, order]
        cdf = np.cumsum(f_sorted, axis=1)
        idx = np.argmax(cdf >= 0.9, axis=1)
        d90 = d_sorted[idx]
        return np.maximum(
            self.sediment_transport_loss_fn.alpha_active_layer * d90,
            1.0e-4,
        ).astype(np.float32)

    def compute_bed_change_closure(self, sediment_model, flow_model, T):
        if sediment_model is None:
            return None

        flow_model.eval()
        sediment_model.eval()
        p_k_gauss = self.gradation_at_gauss_tensor()
        return self.sediment_transport_loss_fn.compute_closure(
            sediment_model,
            flow_model,
            T,
            self.device,
            p_k_override=p_k_gauss,
            freeze_flow_params=True,
        )

    def exner_dzb_dt_cell(self, sediment_model, flow_model, T, closure=None):
        if sediment_model is None:
            return np.zeros(self.mesh.n_cells, dtype=np.float32), None, None

        if closure is None:
            closure = self.compute_bed_change_closure(sediment_model, flow_model, T)

        npp = self.mesh.n_points_per_cell
        nc = self.mesh.n_cells
        loss_fn = self.sediment_transport_loss_fn
        n_grains = closure['E_tk'].shape[1]

        # E_tk/D_tk are volumetric exchange rates [m/s] because C_tk is a
        # volumetric concentration and h * C_tk / T_adapt has units of m/s.
        net_deposition_rate = (
            closure['D_tk'] - closure['E_tk']
        ).detach().view(nc, npp, n_grains).mean(dim=1)

        weights = loss_fn._gauss_weights_t.view(nc, npp, 1)
        nx = loss_fn._gauss_normals_t[:, 0:1]
        ny = loss_fn._gauss_normals_t[:, 1:2]

        dzb_dx_cell, dzb_dy_cell = self.mesh.get_bed_gradient(self.device)
        gauss_cell_id = torch.as_tensor(self.mesh.gauss_cell_id, dtype=torch.long, device=self.device)
        dzb_dx = dzb_dx_cell[gauss_cell_id].view(nc * npp, 1)
        dzb_dy = dzb_dy_cell[gauss_cell_id].view(nc * npp, 1)
        bed_slope_normal = dzb_dx * nx + dzb_dy * ny

        tau_skin = torch.clamp(closure['tau_skin'], min=1.0e-8)
        tau_cr = torch.clamp(closure['tau_cr'], min=1.0e-8)
        kappa_bk = self.bed_slope_coefficient * torch.sqrt(
            tau_cr / torch.maximum(tau_skin, tau_cr)
        )
        q_bk = closure['p_k'] * closure['q_b']
        slope_flux = kappa_bk * torch.abs(q_bk) * bed_slope_normal
        slope_term = torch.sum(
            slope_flux.view(nc, npp, n_grains) * weights,
            dim=1,
        ) / self.mesh.cell_area

        # Volumetric Exner equation for the non-equilibrium exchange model:
        #   (1 - porosity) dzb/dt = D - E + slope_diffusion
        # E/D are already volumetric rates [m/s], so do not divide by rho_s.
        dzb_dt_k = (
            net_deposition_rate
            + slope_term.detach()
        ) / (1.0 - self.porosity + 1.0e-8)

        if self.history is not None:
            exchange_dzb_dt = torch.sum(net_deposition_rate / (1.0 - self.porosity + 1.0e-8), dim=1)
            self.history['exchange_dzb_dt_min'].append(float(torch.min(exchange_dzb_dt).detach().cpu()))
            self.history['exchange_dzb_dt_max'].append(float(torch.max(exchange_dzb_dt).detach().cpu()))

        dzb_dt = torch.sum(dzb_dt_k, dim=1) # 总床变速率
        dzb_dt_np = dzb_dt.cpu().numpy().astype(np.float32) # 转为 numpy 数组，供显式更新使用
        dzb_dt_k_np = dzb_dt_k.cpu().numpy().astype(np.float32) # 分粒径床变速率，供级配更新使用
        return dzb_dt_np, closure, dzb_dt_k_np

    def update_bed_explicit(
        self,
        sediment_model,
        flow_model,
        T,
        window_dt=None,
        max_bed_change_per_step=None,
    ):
        if window_dt is None:
            window_dt = self.window_dt

        dzb_dt, closure, dzb_dt_k = self.exner_dzb_dt_cell(sediment_model, flow_model, T)
        raw_delta_zb = dzb_dt * window_dt
        max_delta = float(np.max(np.abs(raw_delta_zb))) if raw_delta_zb.size else 0.0
        dt_scale = 1.0
        if (
            max_bed_change_per_step is not None
            and max_bed_change_per_step > 0.0
            and max_delta > max_bed_change_per_step
        ):
            dt_scale = max_bed_change_per_step / max(max_delta, 1.0e-12)

        self.window_dt_current = window_dt * dt_scale
        new_bed = self.mesh.zb.astype(np.float32) + dzb_dt * self.window_dt_current
        self.mesh.update_bed(new_bed)

        if self.history is not None:
            self.history['exner_dzb_dt_min'].append(float(np.min(dzb_dt)))
            self.history['exner_dzb_dt_max'].append(float(np.max(dzb_dt)))
            self.history['bed_dt_scale'].append(float(dt_scale))
            self.history['bed_dt_effective'].append(float(self.window_dt_current))
            self.history['bed_delta_max'].append(float(max_delta * dt_scale))

        return new_bed, closure, dzb_dt_k

    def update_gradation_state(
        self,
        sediment_model,
        flow_model,
        T,
        new_bed,
        closure=None,
        bed_change_rate_k=None,
    ):
        if sediment_model is None:
            return None

        if closure is None:
            closure = self.compute_bed_change_closure(sediment_model, flow_model, T)

        if bed_change_rate_k is None:
            _, _, bed_change_rate_k = self.exner_dzb_dt_cell(
                sediment_model,
                flow_model,
                T,
                closure=closure,
            )

        delta_m_bed = bed_change_rate_k * self.rho_bulk * self.window_dt_current
        rho_bulk = self.rho_bulk
        m1_old = self.active_layer_frac * rho_bulk
        m2_old = self.second_layer_frac * rho_bulk

        delta_zb = np.asarray(new_bed, dtype=np.float32) - self.last_bed.astype(np.float32)
        delta1_old = self.delta1.copy()
        delta2_old = self.delta2.copy()
        delta1_new = self.active_layer_thickness_np(self.active_layer_frac)
        delta2_change = delta_zb - (delta1_new - delta1_old)
        delta2_new = np.maximum(delta2_old + delta2_change, 1.0e-4)

        m_star = np.where(delta2_change[:, None] >= 0.0, m1_old, m2_old)
        m1_new = (
            delta_m_bed
            + m1_old * delta1_old[:, None]
            - m_star * delta2_change[:, None]
        ) / delta1_new[:, None]
        m2_new = (
            m2_old * delta2_old[:, None]
            + m_star * delta2_change[:, None]
        ) / delta2_new[:, None]

        m1_new = np.clip(m1_new, 1.0e-12, None)
        m2_new = np.clip(m2_new, 1.0e-12, None)
        self.active_layer_frac = (m1_new / np.sum(m1_new, axis=1, keepdims=True)).astype(np.float32)
        self.second_layer_frac = (m2_new / np.sum(m2_new, axis=1, keepdims=True)).astype(np.float32)
        self.delta1 = delta1_new.astype(np.float32)
        self.delta2 = delta2_new.astype(np.float32)
        self.last_bed = np.asarray(new_bed, dtype=np.float32).copy()

        if self.history is not None:
            self.history['p_min'].append(float(np.min(self.active_layer_frac)))
            self.history['p_max'].append(float(np.max(self.active_layer_frac)))

        return self.active_layer_frac


class ClosureFormulation:
    """沉速和输沙潜力闭合关系。"""

    def __init__(self, cfg):
        self.cfg = cfg

    # 沉速计算（Soulsby 1997）
    def fall_velocity(self, d_k):
        """
        Soulsby (1997) noncohesive particle fall velocity.
        """
        rho_w = float(getattr(self.cfg, 'rho_w', 1000.0))   # 水密度 (kg/m³)
        rho_s = float(getattr(self.cfg, 'rho_s', 2650.0))   # 泥沙密度 (kg/m³)
        g = float(getattr(self.cfg, 'g', 9.81)) # 重力加速度 (m/s²)
        nu = float(getattr(self.cfg, 'kinematic_viscosity', 1.0e-6))    # 水动力粘度 (m²/s)

        d_k = torch.as_tensor(d_k, dtype=torch.float32) # 分粒径，形状 [K]
        submerged_gravity = rho_s / rho_w - 1.0 # 浮力修正重力加速度
        d_star = d_k * torch.pow(
            torch.as_tensor(submerged_gravity * g / (nu ** 2), dtype=d_k.dtype, device=d_k.device),
            1.0 / 3.0,
        )
        ws = (nu / d_k) * (torch.sqrt(10.36 ** 2 + 1.049 * d_star ** 3) - 10.36)
        return ws

    # 输沙潜力（Wu 2000）
    def TransportPotential_Wu(self, h, u, v, d_k, p_k):
        """计算 Wu et al. (2000) 分粒径总输沙潜力和浓度潜力。

        Returns:
            q_capacity: p_k * (q_b + q_s), shape compatible with p_k/C_tk.
            C_capacity: q_capacity / (h |U|).
            components: diagnostic dictionary with q_b, q_s, tau_b, tau_skin, tau_cr, ws.
        """
        device = h.device
        dtype = h.dtype
        rho_w = float(getattr(self.cfg, 'rho_w', 1000.0))   # 水密度 (kg/m³)
        rho_s = float(getattr(self.cfg, 'rho_s', 2650.0))   # 泥沙密度 (kg/m³)
        g = float(getattr(self.cfg, 'g', 9.81)) # 重力加速度 (m/s²)
        n_manning = float(getattr(self.cfg, 'n_manning', 0.01)) # Manning 糙率系数
        theta_cr = float(getattr(self.cfg, 'wu_theta_cr', 0.03))    # Wu (2000) 临界剪应力参数
        skin_factor = float(getattr(self.cfg, 'skin_shear_factor', 1.0))    # 床面剪应力修正系数

        vel_mag = torch.sqrt(u ** 2 + v ** 2 + EPS_DIVISION)    # 速度模长
        h_safe = torch.clamp(h, min=0.05)   # 安全水深，避免除以零或过小值导致数值不稳定

        tau_b = rho_w * g * n_manning ** 2 * vel_mag ** 2 / torch.pow(h_safe, 1.0 / 3.0)    # 床面剪应力
        tau_skin = torch.clamp(torch.as_tensor(skin_factor, dtype=dtype, device=device), 0.0, 1.0) * tau_b  # 皮肤剪应力

        d_k = torch.as_tensor(d_k, dtype=dtype, device=device)  # 分粒径，形状 [K]
        if d_k.dim() == 1:
            d_k = d_k.view(1, -1)
        d_k = torch.clamp(d_k, min=EPS_DIVISION)

        R = torch.as_tensor(max(rho_s / rho_w - 1.0, EPS_DIVISION), dtype=dtype, device=device)
        tau_cr = theta_cr * (rho_s - rho_w) * g * d_k
        tau_cr = torch.clamp(tau_cr, min=EPS_DIVISION)
        ws = self.fall_velocity(d_k).to(device=device, dtype=dtype)
        transport_scale = torch.sqrt(R * g * d_k ** 3)

        q_b = 0.0053 * transport_scale * torch.relu(tau_skin / tau_cr - 1.0) ** 2.2

        suspension_stage = torch.relu((tau_b / tau_cr - 1.0) * vel_mag / torch.clamp(ws, min=EPS_DIVISION))
        q_s = 2.62e-5 * transport_scale * suspension_stage ** 1.74

        active_bedload = tau_skin > tau_cr
        active_suspended = (tau_b > tau_cr) & active_bedload    

        q_b = torch.where(active_bedload, q_b, torch.zeros_like(q_b))
        q_s = torch.where(active_suspended, q_s, torch.zeros_like(q_s))

        q_capacity = p_k * (q_b + q_s)
        C_capacity = q_capacity / torch.clamp(h * vel_mag, min=EPS_SAFE)
        return q_capacity, C_capacity, {
            'q_b': q_b,
            'q_s': q_s,
            'tau_b': tau_b,
            'tau_skin': tau_skin,
            'tau_cr': tau_cr,
            'ws': ws,
        }
