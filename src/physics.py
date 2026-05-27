import torch
import torch.nn.functional as F
from .losses import match_closure, grain_diameters_like, smooth_positive
from .model import SedimentPINN, GradationPINN


class SVEsPhysicsLoss:
    def __init__(self, fvm_mesh, g=9.81, n_manning=0.01, bounds=None, typical_depth=10.0, typical_velocity=1.0, eps=1e-6):
        self.mesh = fvm_mesh
        self.g = g
        self.n = n_manning
        self.bounds = bounds
        self.typical_h = typical_depth
        self.typical_u = typical_velocity
        self.eps = eps
        self.residual_scale = 1.0 / (fvm_mesh.cell_area * typical_depth * typical_velocity) # 适当缩放残差以平衡损失项

    def _bed_gradient_tensor(self, zb_tensor, device):
        """用张量床面计算坡度；支持 nx=1 或 ny=1 的退化网格。"""
        if zb_tensor is None:
            dzb_dx, dzb_dy = self.mesh.get_bed_gradient()
            return (torch.tensor(dzb_dx, dtype=torch.float32, device=device),
                    torch.tensor(dzb_dy, dtype=torch.float32, device=device))

        zb = zb_tensor.to(dtype=torch.float32, device=device).reshape(self.mesh.ny, self.mesh.nx)
        dzb_dx = torch.zeros_like(zb)
        dzb_dy = torch.zeros_like(zb)
        if self.mesh.nx > 1:
            if self.mesh.nx > 2:
                dzb_dx[:, 1:-1] = (zb[:, 2:] - zb[:, :-2]) / (2 * self.mesh.resolution)
            dzb_dx[:, 0] = (zb[:, 1] - zb[:, 0]) / self.mesh.resolution
            dzb_dx[:, -1] = (zb[:, -1] - zb[:, -2]) / self.mesh.resolution
        if self.mesh.ny > 1:
            if self.mesh.ny > 2:
                dzb_dy[1:-1, :] = (zb[2:, :] - zb[:-2, :]) / (2 * self.mesh.resolution)
            dzb_dy[0, :] = (zb[1, :] - zb[0, :]) / self.mesh.resolution
            dzb_dy[-1, :] = (zb[-1, :] - zb[-2, :]) / self.mesh.resolution
        return dzb_dx.flatten(), dzb_dy.flatten()

    def compute_loss(self, model, t, device, zb_tensor=None):
        n_gauss = self.mesh.n_gauss_total
        gauss_coords = torch.tensor(self.mesh.gauss_coords, dtype=torch.float32, device=device)
        gauss_normals = torch.tensor(self.mesh.gauss_normals, dtype=torch.float32, device=device)
        gauss_weights = torch.tensor(self.mesh.gauss_weights, dtype=torch.float32, device=device).unsqueeze(1)

        if self.bounds is not None:
            x_norm = (gauss_coords[:, 0:1] - self.bounds['x_min']) / (self.bounds['x_max'] - self.bounds['x_min'])
            y_norm = (gauss_coords[:, 1:2] - self.bounds['y_min']) / (self.bounds['y_max'] - self.bounds['y_min'])
        else:
            x_norm = gauss_coords[:, 0:1]
            y_norm = gauss_coords[:, 1:2]

        t_tensor = torch.full((n_gauss, 1), t, device=device, dtype=torch.float32)
        xyt = torch.cat([x_norm, y_norm, t_tensor], dim=1)

        outputs = model(xyt)
        h_norm = outputs[:, 0:1]
        u_norm = outputs[:, 1:2]
        v_norm = outputs[:, 2:3]

        h = h_norm * self.typical_h
        u = (u_norm - 0.5) * 2 * self.typical_u
        v = (v_norm - 0.5) * 2 * self.typical_u

        nx = gauss_normals[:, 0:1]
        ny = gauss_normals[:, 1:2]
        Nc = self.mesh.n_cells
        npp = self.mesh.n_points_per_cell
        weights_r = gauss_weights.view(Nc, npp)

        h_r = h.view(Nc, npp)
        u_r = u.view(Nc, npp)
        v_r = v.view(Nc, npp)
        h_cell = torch.mean(h_r, dim=1, keepdim=True)

        flux_h = (h * u) * nx + (h * v) * ny
        bnd_h = torch.sum(flux_h.view(Nc, npp) * weights_r, dim=1, keepdim=True)
        loss_continuity = torch.mean((bnd_h * self.residual_scale) ** 2)

        dzb_dx_t, dzb_dy_t = self._bed_gradient_tensor(zb_tensor, device)
        slope_x = -self.g * h_cell * dzb_dx_t.unsqueeze(1) * self.mesh.cell_area
        slope_y = -self.g * h_cell * dzb_dy_t.unsqueeze(1) * self.mesh.cell_area

        h_safe = torch.clamp(h_r, min=0.05)
        vel_mag = torch.sqrt(u_r ** 2 + v_r ** 2 + self.eps)
        fric_tx = torch.mean(self.g * self.n ** 2 * vel_mag * u_r / torch.pow(h_safe, 1. / 3.), dim=1, keepdim=True) * self.mesh.cell_area
        fric_ty = torch.mean(self.g * self.n ** 2 * vel_mag * v_r / torch.pow(h_safe, 1. / 3.), dim=1, keepdim=True) * self.mesh.cell_area

        flux_mx = (h * u * u + 0.5 * self.g * h * h) * nx + (h * u * v) * ny
        bnd_mx = torch.sum(flux_mx.view(Nc, npp) * weights_r, dim=1, keepdim=True)
        mom_x_res = (bnd_mx - slope_x + fric_tx) * self.residual_scale
        loss_momentum_x = torch.mean(mom_x_res ** 2)

        flux_my = (h * v * u) * nx + (h * v * v + 0.5 * self.g * h * h) * ny
        bnd_my = torch.sum(flux_my.view(Nc, npp) * weights_r, dim=1, keepdim=True)
        mom_y_res = (bnd_my - slope_y + fric_ty) * self.residual_scale
        loss_momentum_y = torch.mean(mom_y_res ** 2)

        total_loss = loss_continuity + loss_momentum_x + loss_momentum_y
        loss_dict = {
            'continuity': loss_continuity.item(),
            'momentum_x': loss_momentum_x.item(),
            'momentum_y': loss_momentum_y.item(),
            'total': total_loss.item(),
        }

        return total_loss, loss_dict



class SedimentTransportLoss:
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
        alpha_active_layer=10.0,
        w_gradation=1.0,
        w_capacity=0.05,
        source_sharpness=1e-3,
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
        self.alpha_active_layer = alpha_active_layer
        self.w_gradation = w_gradation
        self.w_capacity = w_capacity
        self.source_sharpness = source_sharpness

    def _build_xyt_at_gauss(self, T_norm, device, requires_grad=True):
        gauss_coords = torch.tensor(self.mesh.gauss_coords, dtype=torch.float32, device=device)
        # 对坐标进行归一化，确保输入到神经网络的 x 和 y 都在 [0, 1] 范围内
        if self.bounds is not None:
            x_norm = (gauss_coords[:, 0:1] - self.bounds['x_min']) / (self.bounds['x_max'] - self.bounds['x_min'])
            y_norm = (gauss_coords[:, 1:2] - self.bounds['y_min']) / (self.bounds['y_max'] - self.bounds['y_min'])
        else:
            x_norm = gauss_coords[:, 0:1]
            y_norm = gauss_coords[:, 1:2]
        t_tensor = torch.full((self.mesh.n_gauss_total, 1), T_norm, device=device, dtype=torch.float32)
        xyt = torch.cat([x_norm, y_norm, t_tensor], dim=1)
        return xyt.requires_grad_(requires_grad)

    def _capacity_closure(self, h, u, v, C_tk, p_k):
        vel_mag = torch.sqrt(u ** 2 + v ** 2 + 1e-8) 
        q_capacity = p_k * self.Ag * torch.pow(vel_mag, self.m) # 理论输沙量
        C_capacity = q_capacity / torch.clamp(h * vel_mag, min=1e-6)    # 理论输沙浓度
        adapt_time = self.adaptation_length / torch.clamp(vel_mag, min=1e-3) # 输沙适应长度
        net_source = h * (C_capacity - C_tk) / adapt_time   # 净源汇项
        E_tk = smooth_positive(net_source, sharpness=self.source_sharpness)
        D_tk = smooth_positive(-net_source, sharpness=self.source_sharpness)
        return C_capacity, E_tk, D_tk, vel_mag

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
        freeze_flow_params=True,
    ):
        """计算 C_tk、p_k 和 E/D 闭合量，供输沙 PDE 与 Exner 共用。"""
        xyt = self._build_xyt_at_gauss(T_norm, device, requires_grad=True)  # 构建高斯点的坐标和时间
        
        # 冻结水动力模型参数
        old_requires_grad = None
        if freeze_flow_params:
            old_requires_grad = [p.requires_grad for p in flow_model.parameters()]
            for p in flow_model.parameters():
                p.requires_grad_(False)
        
        try:
            flow_out = flow_model(xyt)
            h = flow_out[:, 0:1] * self.typical_h
            u = (flow_out[:, 1:2] - 0.5) * 2.0 * self.typical_u
            v = (flow_out[:, 2:3] - 0.5) * 2.0 * self.typical_u
        finally:
            if old_requires_grad is not None:
                for p, old_flag in zip(flow_model.parameters(), old_requires_grad):
                    p.requires_grad_(old_flag)

        C_tk = sediment_model(xyt)
        p_k = torch.ones_like(C_tk) / C_tk.shape[1] if gradation_model is None else gradation_model(xyt)
        
        beta_tk = match_closure(beta_tk, C_tk, self.beta_default)
        epsilon_thk = match_closure(epsilon_thk, C_tk, self.epsilon_default)
        if E_tk is None or D_tk is None:
            C_capacity, E_closure, D_closure, vel_mag = self._capacity_closure(h, u, v, C_tk, p_k)
            E_tk = E_closure if E_tk is None else match_closure(E_tk, C_tk, 0.0)
            D_tk = D_closure if D_tk is None else match_closure(D_tk, C_tk, 0.0)
        else:
            C_capacity = C_tk.detach()
            vel_mag = torch.sqrt(u ** 2 + v ** 2 + 1e-8)
            E_tk = match_closure(E_tk, C_tk, 0.0)
            D_tk = match_closure(D_tk, C_tk, 0.0)

        return {
            'xyt': xyt, 'h': h, 'u': u, 'v': v,
            'C_tk': C_tk, 'p_k': p_k,
            'beta_tk': beta_tk, 'epsilon_thk': epsilon_thk,
            'E_tk': E_tk, 'D_tk': D_tk,
            'C_capacity': C_capacity, 'vel_mag': vel_mag,
        }

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
        freeze_flow_params=True,
    ):
        closure = self.compute_closure(
            sediment_model, flow_model, T_norm, device,
            gradation_model=gradation_model, beta_tk=beta_tk,
            epsilon_thk=epsilon_thk, E_tk=E_tk, D_tk=D_tk,
            freeze_flow_params=freeze_flow_params,
        )
        xyt = closure['xyt']
        h = closure['h']
        u = closure['u']
        v = closure['v']
        C_tk = closure['C_tk']
        p_k = closure['p_k']
        beta_tk = closure['beta_tk']
        epsilon_thk = closure['epsilon_thk']
        E_tk = closure['E_tk']
        D_tk = closure['D_tk']
        C_capacity = closure['C_capacity']
        vel_mag = closure['vel_mag']

        _, residual = SedimentPINN.total_load_loss(
            xyt, h, u, v, C_tk, beta_tk, epsilon_thk, E_tk, D_tk
        )
        transport_loss = torch.mean((residual * self.residual_scale) ** 2)
        capacity_loss = torch.mean((C_tk - C_capacity.detach()) ** 2)

        gradation_loss = torch.zeros((), dtype=C_tk.dtype, device=C_tk.device)
        if gradation_model is not None:
            d_k = grain_diameters_like(self.grain_diameters, C_tk)
            L_a = GradationPINN.active_layer_thickness(d_k, p_k=p_k, alpha_a=self.alpha_active_layer)
            gradation_loss, _ = GradationPINN.active_layer_loss(xyt, p_k, L_a, E_tk, D_tk, rho_s=self.rho_s)

        loss = transport_loss + self.w_capacity * capacity_loss + self.w_gradation * gradation_loss
        return loss, {
            'transport': transport_loss.item(),
            'gradation': gradation_loss.item(),
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

    def compute_sediment_loss(
        self,
        sediment_model,
        flow_model,
        T_norm,
        device,
        gradation_model=None,
        beta_tk=None,
        epsilon_thk=None,
    ):
        """仅计算输沙 PDE 残差 + 容量损失，用于独立更新 sediment_model。"""
        closure = self.compute_closure(
            sediment_model, flow_model, T_norm, device,
            gradation_model=gradation_model, beta_tk=beta_tk,
            epsilon_thk=epsilon_thk, freeze_flow_params=True,
        )

        xyt = closure['xyt']
        h = closure['h']
        u = closure['u']
        v = closure['v']
        C_tk = closure['C_tk']
        beta_tk = closure['beta_tk']
        epsilon_thk = closure['epsilon_thk']
        E_tk = closure['E_tk']
        D_tk = closure['D_tk']
        C_capacity = closure['C_capacity']
        vel_mag = closure['vel_mag']

        _, residual = SedimentPINN.total_load_loss(
            xyt, h, u, v, C_tk, beta_tk, epsilon_thk, E_tk, D_tk
        )
        transport_loss = torch.mean((residual * self.residual_scale) ** 2)
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

    def compute_gradation_loss(
        self,
        sediment_model,
        flow_model,
        T_norm,
        device,
        gradation_model,
    ):
        """仅计算级配损失，用于独立更新 gradation_model。"""
        closure = self.compute_closure(
            sediment_model, flow_model, T_norm, device,
            gradation_model=gradation_model, freeze_flow_params=True,
        )
        C_tk = closure['C_tk']
        p_k = closure['p_k']
        E_tk = closure['E_tk']
        D_tk = closure['D_tk']

        # 级配损失
        d_k = grain_diameters_like(self.grain_diameters, C_tk)
        L_a = GradationPINN.active_layer_thickness(d_k, p_k=p_k, alpha_a=self.alpha_active_layer)
        gradation_loss, _ = GradationPINN.active_layer_loss(
            closure['xyt'], p_k, L_a, E_tk, D_tk, rho_s=self.rho_s
        )

        loss = self.w_gradation * gradation_loss
        return loss, {
            'gradation': gradation_loss.item(),
            'p_min': torch.min(p_k).item(),
            'p_max': torch.max(p_k).item(),
        }


class ExnerPhysicsLoss:
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
        eps=1e-6,
        typical_u=1.0,
    ):
        self.mesh = fvm_mesh
        self.xi = 1.0 / (1.0 - porosity)
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

    def compute_loss(
        self,
        bed_model,
        flow_model,
        T_norm,
        device,
        w_ic=10.0,
        sediment_model=None,
        gradation_model=None,
        sediment_transport_loss_fn=None,
    ):
        n_gauss = self.mesh.n_gauss_total
        gauss_coords = torch.tensor(self.mesh.gauss_coords, dtype=torch.float32, device=device)
        gauss_normals = torch.tensor(self.mesh.gauss_normals, dtype=torch.float32, device=device)
        gauss_weights = torch.tensor(self.mesh.gauss_weights, dtype=torch.float32, device=device).unsqueeze(1)

        if self.bounds is not None:
            x_norm = (gauss_coords[:, 0:1] - self.bounds['x_min']) / (self.bounds['x_max'] - self.bounds['x_min'])
            y_norm = (gauss_coords[:, 1:2] - self.bounds['y_min']) / (self.bounds['y_max'] - self.bounds['y_min'])
        else:
            x_norm = gauss_coords[:, 0:1]
            y_norm = gauss_coords[:, 1:2]

        T_tensor = torch.full((n_gauss, 1), T_norm, device=device, dtype=torch.float32, requires_grad=True)
        xyT = torch.cat([x_norm, y_norm, T_tensor], dim=1)
        zb_pred = bed_model(xyT)
        zb_t_norm = torch.autograd.grad(
            zb_pred, T_tensor, torch.ones_like(zb_pred),
            create_graph=True, retain_graph=True,
        )[0]
        zb_t = zb_t_norm / self.T_physical

        Nc = self.mesh.n_cells  # 单元数量
        npp = self.mesh.n_points_per_cell   # 每单元高斯点数
        zb_t_cell = torch.mean(zb_t.view(Nc, npp), dim=1, keepdim=True) 

        if sediment_model is not None and sediment_transport_loss_fn is not None:
            closure = sediment_transport_loss_fn.compute_closure(
                sediment_model, flow_model, T_norm, device,
                gradation_model=gradation_model, freeze_flow_params=True,
            )
            # 计算净沉积量
            net_deposition = torch.sum(closure['D_tk'] - closure['E_tk'], dim=1, keepdim=True)
            # 计算床面变化速率的残差
            bed_source_gauss = self.xi * net_deposition / (sediment_transport_loss_fn.rho_s + 1e-8)
            # 计算每个单元的床面变化速率
            bed_source_cell = torch.mean(bed_source_gauss.view(Nc, npp), dim=1, keepdim=True)
            exner_residual = (zb_t_cell - bed_source_cell) / max(self.typical_zb, 1e-8)
            closure_mode = 'exchange'
        else:
            T_flow = torch.full((n_gauss, 1), T_norm, device=device, dtype=torch.float32)
            xyt_flow = torch.cat([x_norm, y_norm, T_flow], dim=1)
            with torch.no_grad():
                flow_out = flow_model(xyt_flow)
                u_pred = (flow_out[:, 1:2] - 0.5) * 2.0 * self.typical_u
                v_pred = (flow_out[:, 2:3] - 0.5) * 2.0 * self.typical_u
            qx, qy, _ = SedimentPINN.grass_formula(u_pred, v_pred, self.Ag, self.m)
            nx = gauss_normals[:, 0:1]
            ny = gauss_normals[:, 1:2]
            flux_sediment = self.xi * (qx * nx + qy * ny)
            flux_reshaped = flux_sediment.view(Nc, npp)
            weights_reshaped = gauss_weights.view(Nc, npp)
            boundary_integral = torch.sum(flux_reshaped * weights_reshaped, dim=1, keepdim=True)
            volume_term = zb_t_cell * self.mesh.cell_area
            exner_residual = (volume_term + boundary_integral) * self.residual_scale
            closure_mode = 'grass_flux'

        loss_exner = torch.mean(exner_residual ** 2)

        cx = torch.tensor(self.mesh.cell_centers_x, dtype=torch.float32, device=device)
        cy = torch.tensor(self.mesh.cell_centers_y, dtype=torch.float32, device=device)
        if self.bounds is not None:
            cx = (cx - self.bounds['x_min']) / (self.bounds['x_max'] - self.bounds['x_min'])
            cy = (cy - self.bounds['y_min']) / (self.bounds['y_max'] - self.bounds['y_min'])
        T0 = torch.zeros(self.mesh.n_cells, dtype=torch.float32, device=device)
        xyT0 = torch.stack([cx, cy, T0], dim=1)
        zb_ic_pred = bed_model(xyT0)
        zb_ic_true = torch.tensor(self.mesh.zb_initial, dtype=torch.float32, device=device).unsqueeze(1)
        loss_ic = torch.mean((zb_ic_pred - zb_ic_true) ** 2)

        total_loss = loss_exner + w_ic * loss_ic
        return total_loss, {
            'exner': loss_exner.item(), 'ic': loss_ic.item(),
            'total': total_loss.item(), 't_physical': T_norm * self.T_physical,
            'closure': closure_mode,
        }
