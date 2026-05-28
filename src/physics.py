# physics.py – 物理损失函数
# 包含三个核心物理损失计算器：
#   SVEsPhysicsLoss        → 二维浅水方程 (Saint-Venant Equations) FVM 残差
#   SedimentTransportLoss  → 总输沙输移方程 FVM 残差
#   ExnerPhysicsLoss       → Exner 床面演化方程残差
# 共用 _CachedMeshTensors 基类实现 GPU 张量缓存。

import torch
import torch.nn.functional as F

from .config import EPS_DIVISION, EPS_SAFE, EPS_VELOCITY_CLAMP
from .losses import build_xyt, match_closure, smooth_positive
from .model import FlowPINN, SedimentPINN


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


def _set_requires_grad(module, requires_grad):
    if module is None:
        return None
    old_flags = [p.requires_grad for p in module.parameters()]
    for p in module.parameters():
        p.requires_grad_(requires_grad)
    return old_flags


def _restore_requires_grad(module, old_flags):
    if module is None or old_flags is None:
        return
    for p, old_flag in zip(module.parameters(), old_flags):
        p.requires_grad_(old_flag)


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
                 T_physical=1.0, include_time_terms=True, eps=EPS_SAFE):
        self.mesh = fvm_mesh
        self.g = g
        self.n = n_manning
        self.bounds = bounds
        self.typical_h = typical_depth
        self.typical_u = typical_velocity
        self.T_physical = T_physical
        self.include_time_terms = include_time_terms
        self.eps = eps
        self.residual_scale = 1.0 / (fvm_mesh.cell_area * typical_depth * typical_velocity)
        self._init_cache(fvm_mesh)

    def compute_loss(self, model, t, device, zb_tensor=None):
        """计算 SVEs 物理损失。

        返回:
            total_loss: 连续性 + x动量 + y动量 损失之和
            loss_dict: 各分项损失值字典
        """
        self._ensure_tensors(device)
        gauss_coords = self._gauss_coords_t
        gauss_normals = self._gauss_normals_t
        gauss_weights = self._gauss_weights_t

        # ── 构建输入并前向推理 ──
        xyt = build_xyt(
            gauss_coords, t, self.bounds, device,
            requires_grad=self.include_time_terms,
        )
        outputs = model(xyt)
        h, u, v = FlowPINN.decode_output(outputs, self.typical_h, self.typical_u)

        def dt(q):
            if not self.include_time_terms:
                return torch.zeros_like(q)
            grad = torch.autograd.grad(
                q, xyt,
                grad_outputs=torch.ones_like(q),
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )[0]
            if grad is None:
                return torch.zeros_like(q)
            return grad[:, 2:3] / max(self.T_physical, EPS_DIVISION)

        # ── 几何量准备 ──
        nx = gauss_normals[:, 0:1]  # x方向法向量分量，形状 [N, 1]
        ny = gauss_normals[:, 1:2]  # y方向法向量分量
        Nc = self.mesh.n_cells
        npp = self.mesh.n_points_per_cell
        weights_r = gauss_weights.view(Nc, npp)

        h_r = h.view(Nc, npp)
        u_r = u.view(Nc, npp)
        v_r = v.view(Nc, npp)
        h_cell = torch.mean(h_r, dim=1, keepdim=True)   # 单元平均水深

        # ── ① 连续性方程：∂/∂t∫h dA + ∮ (hu·nx + hv·ny) dS = 0 ──
        h_t_cell = torch.mean(dt(h).view(Nc, npp), dim=1, keepdim=True)
        vol_h = h_t_cell * self.mesh.cell_area
        flux_h = (h * u) * nx + (h * v) * ny
        bnd_h = torch.sum(flux_h.view(Nc, npp) * weights_r, dim=1, keepdim=True)
        cont_res = (vol_h + bnd_h) * self.residual_scale
        loss_continuity = torch.mean(cont_res ** 2)

        # ── ② 床面坡度源项：S = -g·h·∇zb·A ──
        dzb_dx_t, dzb_dy_t = self.mesh.get_bed_gradient_tensor(zb_tensor, device)
        slope_x = -self.g * h_cell * dzb_dx_t.unsqueeze(1) * self.mesh.cell_area
        slope_y = -self.g * h_cell * dzb_dy_t.unsqueeze(1) * self.mesh.cell_area

        # ── ③ Manning 摩擦项：τ = g·n²·|U|·u / h^(1/3) ──
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

        # ── ④ x 方向动量方程 ──
        flux_mx = (h * u * u + 0.5 * self.g * h * h) * nx + (h * u * v) * ny
        bnd_mx = torch.sum(flux_mx.view(Nc, npp) * weights_r, dim=1, keepdim=True)
        hu_t_cell = torch.mean(dt(h * u).view(Nc, npp), dim=1, keepdim=True)
        vol_mx = hu_t_cell * self.mesh.cell_area
        mom_x_res = (vol_mx + bnd_mx - slope_x + fric_tx) * self.residual_scale
        loss_momentum_x = torch.mean(mom_x_res ** 2)

        # ── ⑤ y 方向动量方程 ──
        flux_my = (h * v * u) * nx + (h * v * v + 0.5 * self.g * h * h) * ny
        bnd_my = torch.sum(flux_my.view(Nc, npp) * weights_r, dim=1, keepdim=True)
        hv_t_cell = torch.mean(dt(h * v).view(Nc, npp), dim=1, keepdim=True)
        vol_my = hv_t_cell * self.mesh.cell_area
        mom_y_res = (vol_my + bnd_my - slope_y + fric_ty) * self.residual_scale
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

    负责计算：
    - 总输沙浓度 C_tk 的有限体积对流扩散残差
    - 输沙容量闭合损失（C_tk 趋近平衡浓度 C_capacity）

    核心流程通过 compute_closure() 统一获取所有中间量（h, u, v, C_tk, p_k, E/D），
    供 compute_loss / compute_sediment_loss 复用。

    参数:
        fvm_mesh: FVM 网格
        bounds: 坐标归一化边界
        typical_depth / typical_velocity: 物理量级
        Ag, m: Grass 输沙公式参数
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
        Ag=0.001,
        m=3,
        adaptation_length=50.0,
        rho_s=2650.0,
        T_physical=1.0,
        alpha_active_layer=10.0,
        w_capacity=0.05,
        source_sharpness=EPS_VELOCITY_CLAMP,    # 用于 smooth_positive 的侵蚀/沉积分离锐度，越大越接近硬分段函数，但可能导致数值不稳定
    ):
        self.mesh = fvm_mesh
        self.bounds = bounds
        self.typical_h = typical_depth
        self.typical_u = typical_velocity
        self.beta_default = beta_default
        self.epsilon_default = epsilon_default
        self.residual_scale = residual_scale
        self.grain_diameters = grain_diameters
        self.Ag = Ag
        self.m = m
        self.adaptation_length = adaptation_length
        self.rho_s = rho_s
        self.T_physical = T_physical
        self.alpha_active_layer = alpha_active_layer
        self.w_capacity = w_capacity
        self.source_sharpness = source_sharpness
        self._init_cache(fvm_mesh)

    def _build_xyt_at_gauss(self, T_norm, device, requires_grad=True):
        """在高斯积分点构建归一化 (x, y, t) 输入张量。"""
        self._ensure_tensors(device)
        return build_xyt(self._gauss_coords_t, T_norm, self.bounds, device, requires_grad=requires_grad)

    def _capacity_closure(self, h, u, v, C_tk, p_k):
        """计算输沙容量闭合：平衡浓度 C_capacity、侵蚀率 E_tk、沉积率 D_tk。

        基于 Grass 公式计算理论输沙量，再通过适应长度将偏离平衡的部分
        拆分为侵蚀（源）和沉积（汇），用 smooth_positive 保证可微。
        """
        vel_mag = torch.sqrt(u ** 2 + v ** 2 + EPS_DIVISION)
        q_capacity = p_k * self.Ag * torch.pow(vel_mag, self.m)         # 理论输沙量
        C_capacity = q_capacity / torch.clamp(h * vel_mag, min=EPS_SAFE)  # 理论输沙浓度
        adapt_time = self.adaptation_length / torch.clamp(vel_mag, min=EPS_VELOCITY_CLAMP)
        net_source = h * (C_capacity - C_tk) / adapt_time               # 净源汇项
        E_tk = smooth_positive(net_source, sharpness=self.source_sharpness)   # 侵蚀率
        D_tk = smooth_positive(-net_source, sharpness=self.source_sharpness)  # 沉积率
        return C_capacity, E_tk, D_tk, vel_mag

    def _dt(self, q, xyt):
        """Return physical-time derivative from normalized network time."""
        def dt_one(q_one):
            grad = torch.autograd.grad(
                q_one, xyt,
                grad_outputs=torch.ones_like(q_one),
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )[0]
            if grad is None:
                return torch.zeros_like(q_one)
            return grad[:, 2:3] / max(self.T_physical, EPS_DIVISION)

        if q.dim() == 1 or q.shape[1] == 1:
            return dt_one(q if q.dim() > 1 else q.unsqueeze(1))
        return torch.cat([dt_one(q[:, k:k + 1]) for k in range(q.shape[1])], dim=1)

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
        storage = h * C_tk / beta_safe
        storage_t = self._dt(storage, xyt)
        volume_storage = torch.mean(storage_t.view(Nc, npp, K), dim=1) * self.mesh.cell_area

        adv_flux = h * C_tk * (u * nx + v * ny)
        boundary_advection = torch.sum(adv_flux.view(Nc, npp, K) * weights, dim=1)

        Cx = self._physical_grad(C_tk, xyt, 0)
        Cy = self._physical_grad(C_tk, xyt, 1)
        diff_flux = epsilon_thk * h * (Cx * nx + Cy * ny)
        boundary_diffusion = torch.sum(diff_flux.view(Nc, npp, K) * weights, dim=1)

        source = E_tk - D_tk
        volume_source = torch.mean(source.view(Nc, npp, K), dim=1) * self.mesh.cell_area

        residual = volume_storage + boundary_advection - boundary_diffusion - volume_source
        loss = torch.mean((residual * self.residual_scale) ** 2)
        return loss, residual

    # ── 统一闭合计算 ──

    def compute_closure(
        self,
        sediment_model,
        flow_model,
        T_norm,
        device,
        gradation_model=None,
        beta_tk=None,
        epsilon_thk=None,
        E_tk=None,
        D_tk=None,
        p_k_override=None,
        freeze_flow_params=True,
    ):
        """计算所有中间物理量，供输沙 PDE、Exner 方程和级配方程共用。

        返回字典包含：xyt, h, u, v, C_tk, p_k, beta_tk, epsilon_thk,
        E_tk, D_tk, C_capacity, vel_mag。
        """
        xyt = self._build_xyt_at_gauss(T_norm, device, requires_grad=True)

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

        # 输沙浓度和级配
        C_tk = sediment_model(xyt)
        if p_k_override is not None:
            p_k = match_closure(p_k_override, C_tk, 1.0 / C_tk.shape[1])
        else:
            p_k = (torch.ones_like(C_tk) / C_tk.shape[1]
                   if gradation_model is None else gradation_model(xyt))

        # 闭合参数整理
        beta_tk = match_closure(beta_tk, C_tk, self.beta_default)
        epsilon_thk = match_closure(epsilon_thk, C_tk, self.epsilon_default)

        if E_tk is None or D_tk is None:
            C_capacity, E_closure, D_closure, vel_mag = self._capacity_closure(h, u, v, C_tk, p_k)
            E_tk = E_closure if E_tk is None else match_closure(E_tk, C_tk, 0.0)
            D_tk = D_closure if D_tk is None else match_closure(D_tk, C_tk, 0.0)
        else:
            C_capacity = C_tk.detach()
            vel_mag = torch.sqrt(u ** 2 + v ** 2 + EPS_DIVISION)
            E_tk = match_closure(E_tk, C_tk, 0.0)
            D_tk = match_closure(D_tk, C_tk, 0.0)

        return {
            'xyt': xyt, 'h': h, 'u': u, 'v': v,
            'C_tk': C_tk, 'p_k': p_k,
            'beta_tk': beta_tk, 'epsilon_thk': epsilon_thk,
            'E_tk': E_tk, 'D_tk': D_tk,
            'C_capacity': C_capacity, 'vel_mag': vel_mag,
        }

    # ── 联合损失（输沙 + 级配）──

    def compute_loss(
        self,
        sediment_model,
        flow_model,
        T_norm,
        device,
        gradation_model=None,
        beta_tk=None,
        epsilon_thk=None,
        E_tk=None,
        D_tk=None,
        p_k_override=None,
        freeze_flow_params=True,
    ):
        """联合损失 = 输沙 PDE 残差 + 容量损失 + 级配损失。"""
        closure = self.compute_closure(
            sediment_model, flow_model, T_norm, device,
            gradation_model=gradation_model, beta_tk=beta_tk,
            epsilon_thk=epsilon_thk, E_tk=E_tk, D_tk=D_tk,
            p_k_override=p_k_override,
            freeze_flow_params=freeze_flow_params,
        )
        xyt     = closure['xyt']
        h, u, v = closure['h'], closure['u'], closure['v']
        C_tk    = closure['C_tk']
        E_tk    = closure['E_tk']
        D_tk    = closure['D_tk']
        C_capacity = closure['C_capacity']
        vel_mag    = closure['vel_mag']

        # 输沙 PDE 有限体积残差
        transport_loss, residual = self.total_load_fvm_loss(
            xyt, h, u, v, C_tk,
            closure['beta_tk'], closure['epsilon_thk'], E_tk, D_tk,
        )
        capacity_loss = torch.mean((C_tk - C_capacity.detach()) ** 2)

        loss = transport_loss + self.w_capacity * capacity_loss
        return loss, {
            'transport': transport_loss.item(),
            'capacity': capacity_loss.item(),
            'total': loss.item(),
            'residual_mean': torch.mean(torch.abs(residual)).item(),
            'C_min': torch.min(C_tk).item(),
            'C_max': torch.max(C_tk).item(),
            'Ceq_mean': torch.mean(C_capacity).item(),
            'p_min': torch.min(p_k).item(),
            'p_max': torch.max(p_k).item(),
            'U_mean': torch.mean(vel_mag).item(),
        }

    # ── 仅输沙损失（独立更新 sediment_model）──

    def compute_sediment_loss(
        self,
        sediment_model,
        flow_model,
        T_norm,
        device,
        gradation_model=None,
        beta_tk=None,
        epsilon_thk=None,
        p_k_override=None,
    ):
        """仅计算输沙 PDE 残差 + 容量损失，用于独立更新 sediment_model。"""
        closure = self.compute_closure(
            sediment_model, flow_model, T_norm, device,
            gradation_model=gradation_model, beta_tk=beta_tk,
            p_k_override=p_k_override,
            epsilon_thk=epsilon_thk, freeze_flow_params=True,
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

        loss = transport_loss + self.w_capacity * capacity_loss
        return loss, {
            'transport': transport_loss.item(),
            'capacity': capacity_loss.item(),
            'residual_mean': torch.mean(torch.abs(residual)).item(),
            'C_min': torch.min(C_tk).item(),
            'C_max': torch.max(C_tk).item(),
            'Ceq_mean': torch.mean(C_capacity).item(),
            'U_mean': torch.mean(vel_mag).item(),
        }

# ═══════════════════════════════════════════════════════════════
#  Exner 床面演化方程损失
# ═══════════════════════════════════════════════════════════════

class ExnerPhysicsLoss(_CachedMeshTensors):
    """Exner 床面演化方程物理损失。

    支持两种闭合模式：
    - 'exchange': 有输沙模型时，用 E_tk/D_tk 净沉积量驱动床面变化
    - 'grass_flux': 无输沙模型时，用 Grass 公式直接计算输沙通量

    损失 = Exner PDE 残差 + w_ic * 初始条件损失

    参数:
        fvm_mesh: FVM 网格
        porosity: 河床孔隙率
        Ag, m: Grass 输沙公式参数
        T_physical: 物理时间尺度 (s)
        typical_zb: 典型河床高程变化量级 (m)
    """

    def __init__(
        self,
        fvm_mesh,
        porosity=0.4,
        Ag=0.001,
        m=3,
        Q=10.0,
        h0=10.0,
        bounds=None,
        typical_zb=1.0,
        T_physical=360000.0,
        eps=EPS_SAFE,
        typical_u=1.0,
    ):
        self.mesh = fvm_mesh
        self.xi = 1.0 / (1.0 - porosity)   # 1/(1-p)，孔隙率修正因子
        self.Ag = Ag
        self.m = m
        self.Q = Q
        self.h0 = h0
        self.bounds = bounds
        self.typical_zb = typical_zb
        self.T_physical = T_physical
        self.eps = eps
        self.typical_u = typical_u
        self.residual_scale = 1.0 / (fvm_mesh.cell_area * typical_zb)
        self._init_cache(fvm_mesh)

    def compute_loss(
        self,
        bed_model,
        flow_model,
        T_norm,
        device,
        w_ic=10.0,
        sediment_model=None,
        gradation_model=None,
        p_k_override=None,
        sediment_transport_loss_fn=None,
    ):
        """计算 Exner 方程损失。

        返回:
            total_loss: Exner 残差 + 初始条件损失
            loss_dict: 各分项损失值字典，含 closure 模式标识
        """
        self._ensure_tensors(device)
        gauss_coords  = self._gauss_coords_t
        gauss_normals = self._gauss_normals_t
        gauss_weights = self._gauss_weights_t

        # ── 计算 ∂zb/∂t（归一化时间域的导数 → 物理时间导数）──
        xyt = build_xyt(gauss_coords, T_norm, self.bounds, device, requires_grad=True)
        zb_pred = bed_model(xyt)
        zb_grad = torch.autograd.grad(
            zb_pred, xyt, torch.ones_like(zb_pred),
            create_graph=True, retain_graph=True,
        )[0]
        zb_t = zb_grad[:, 2:3] / self.T_physical  # ∂zb/∂t_physical

        Nc  = self.mesh.n_cells
        npp = self.mesh.n_points_per_cell
        zb_t_cell = torch.mean(zb_t.view(Nc, npp), dim=1, keepdim=True)

        # ── Exner 残差：两种闭合模式 ──

        if sediment_model is not None and sediment_transport_loss_fn is not None:
            # 模式 1: exchange — 由输沙模型的 E/D 驱动
            sediment_flags = _set_requires_grad(sediment_model, False)
            gradation_flags = _set_requires_grad(gradation_model, False)
            try:
                closure = sediment_transport_loss_fn.compute_closure(
                    sediment_model, flow_model, T_norm, device,
                    gradation_model=gradation_model,
                    p_k_override=p_k_override,
                    freeze_flow_params=True,
                )
            finally:
                _restore_requires_grad(sediment_model, sediment_flags)
                _restore_requires_grad(gradation_model, gradation_flags)
            net_deposition = torch.sum(closure['D_tk'] - closure['E_tk'], dim=1, keepdim=True)
            bed_source = self.xi * net_deposition / (sediment_transport_loss_fn.rho_s + EPS_DIVISION)
            bed_source_cell = torch.mean(bed_source.view(Nc, npp), dim=1, keepdim=True)
            exner_residual = (zb_t_cell - bed_source_cell) / max(self.typical_zb, EPS_DIVISION)
            closure_mode = 'exchange'

        else:
            # 模式 2: grass_flux — 直接用 Grass 公式计算输沙通量边界积分
            xyt_flow = build_xyt(gauss_coords, T_norm, self.bounds, device, requires_grad=False)
            with torch.no_grad():
                flow_out = flow_model(xyt_flow)
                _, u_pred, v_pred = FlowPINN.decode_output(flow_out, 1.0, self.typical_u)

            qx, qy, _ = SedimentPINN.grass_formula(u_pred, v_pred, self.Ag, self.m)
            nx = gauss_normals[:, 0:1]
            ny = gauss_normals[:, 1:2]
            flux_sediment = self.xi * (qx * nx + qy * ny)

            boundary_integral = torch.sum(
                flux_sediment.view(Nc, npp) * gauss_weights.view(Nc, npp),
                dim=1, keepdim=True,
            )
            volume_term = zb_t_cell * self.mesh.cell_area
            exner_residual = (volume_term + boundary_integral) * self.residual_scale
            closure_mode = 'grass_flux'

        loss_exner = torch.mean(exner_residual ** 2)

        # ── 初始条件损失：zb(x, y, t=0) = zb_initial ──
        cx = torch.tensor(self.mesh.cell_centers_x, dtype=torch.float32, device=device)
        cy = torch.tensor(self.mesh.cell_centers_y, dtype=torch.float32, device=device)
        if self.bounds is not None:
            cx = (cx - self.bounds['x_min']) / (self.bounds['x_max'] - self.bounds['x_min'])
            cy = (cy - self.bounds['y_min']) / (self.bounds['y_max'] - self.bounds['y_min'])
        T0 = torch.zeros(self.mesh.n_cells, dtype=torch.float32, device=device)
        xyT0 = torch.stack([cx, cy, T0], dim=1)

        zb_ic_pred = bed_model(xyT0)
        zb_ic_true = torch.tensor(
            self.mesh.zb_initial, dtype=torch.float32, device=device).unsqueeze(1)
        loss_ic = torch.mean((zb_ic_pred - zb_ic_true) ** 2)

        total_loss = loss_exner + w_ic * loss_ic
        return total_loss, {
            'exner': loss_exner.item(),
            'ic': loss_ic.item(),
            'total': total_loss.item(),
            't_physical': T_norm * self.T_physical,
            'closure': closure_mode,
        }
