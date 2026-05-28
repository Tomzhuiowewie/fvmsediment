# train.py – 解耦训练流程
# 包含：
#   DecoupledTrainer → 水动力 / 输沙 / 河床 / 级配 四模型交替训练器
#   build_boundary_conditions → 边界条件构建
#   run_hump_evolution_test   → hump 算例一键运行入口
#   compute_ic_weight         → 初始条件损失权重衰减函数

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange

from .config import (
    AG_VALUES,
    BBOX,
    BC_DEFAULT,
    BOUNDS,
    GRAIN_DIAMETERS,
    INCLUDE_FLOW_TIME_TERMS,
    N_GAUSS_POINTS,
    NUM_GRAIN_CLASSES,
    RESOLUTION,
    TPHYSICAL,
    TRAINING_SETTINGS,
    TYPICAL_DEPTH,
    TYPICAL_VELOCITY,
)
from .data import FVMeshPreprocessor
from .evaluate import visualize_results
from .model import FlowPINN, BedPINN, SedimentPINN
from .physics import SVEsPhysicsLoss, SedimentTransportLoss, ExnerPhysicsLoss


def compute_ic_weight(T_norm: float, w_max: float = 30.0, w_min: float = 5.0) -> float:
    return max(w_min, w_max * (1 - T_norm))


def hump_initial_bed(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    in_hump = ((x >= 500) & (x <= 700) & (y >= 400) & (y <= 600))
    zb = np.zeros_like(x)
    zb[in_hump] = (
        np.sin(np.pi * (x[in_hump] - 500) / 200) ** 2
        * np.sin(np.pi * (y[in_hump] - 400) / 200) ** 2
    )
    return zb


class DecoupledTrainer:
    def __init__(
        self,
        flow_model,
        sediment_model,
        bed_model,
        fvm_mesh,
        device,
        flow_loss_fn,
        sediment_transport_loss_fn,
        exner_loss_fn,
        flow_lr=1e-4,
        transport_lr=1e-4,
        bed_lr=1e-4,
    ):
        self.flow_model = flow_model
        self.sediment_model = sediment_model
        self.bed_model = bed_model
        self.mesh = fvm_mesh
        self.device = device
        self.flow_loss_fn = flow_loss_fn
        self.sediment_transport_loss_fn = sediment_transport_loss_fn
        self.exner_loss_fn = exner_loss_fn
        self.last_delta = np.zeros(self.mesh.n_cells, dtype=np.float32)
        self.last_bed = self.mesh.zb.copy()
        self.macro_dt_physical = 1.0

        n_grains = len(self.sediment_transport_loss_fn.grain_diameters or GRAIN_DIAMETERS)
        self.active_layer_frac = np.full((self.mesh.n_cells, n_grains), 1.0 / n_grains, dtype=np.float32)
        self.second_layer_frac = self.active_layer_frac.copy()
        self.delta1 = self._active_layer_thickness_np(self.active_layer_frac)
        self.delta2 = np.full(self.mesh.n_cells, 1.0, dtype=np.float32)

        # 水动力模型优化器（SVEs PDE残差 + BC损失）
        self.flow_optimizer = torch.optim.Adam(self.flow_model.parameters(), lr=flow_lr)
        # 输沙模型优化器（水体浓度 C_tk）
        self.sediment_optimizer = (
            torch.optim.Adam(self.sediment_model.parameters(), lr=transport_lr)
            if self.sediment_model is not None else None
        )

        # 河床模型优化器（Exner方程）
        self.bed_optimizer = torch.optim.Adam(self.bed_model.parameters(), lr=bed_lr)

        # 学习率调度器：根据损失变化自动调整学习率，帮助训练更稳定地收敛
        self.flow_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.flow_optimizer, 'min', factor=0.5, patience=500)
        self.sediment_scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(self.sediment_optimizer, 'min', factor=0.5, patience=500)
            if self.sediment_optimizer is not None else None
        )
        self.bed_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.bed_optimizer, 'min', factor=0.5, patience=500)

        self.history = {
            'flow_loss': [],    # 水动力损失（SVEs PDE残差 + BC损失）
            'sediment_loss': [],    # 河床演化损失（Exner PDE残差 + IC损失）
            'transport_loss': [],   # 输沙损失（浓度对流扩散）
            'gradation_loss': [],   # 显式级配更新占位记录
            'capacity_loss': [],    # 容量损失()
            'continuity': [],   # 连续性损失
            'momentum_x': [],
            'momentum_y': [],
            'exner': [],
            'zb_min': [],
            'zb_max': [],
            'ic': [],
            'C_min': [],
            'C_max': [],
            'p_min': [],
            'p_max': [],
        }

    def train_flow_phase(self, n_epochs, T_norm, data_coords=None, data_values=None, data_mask=None):
        self.flow_model.train()
        for epoch in range(n_epochs):
            self.flow_optimizer.zero_grad()
            physics_loss, loss_dict = self.flow_loss_fn.compute_loss(self.flow_model, T_norm, self.device)
            if data_coords is not None:
                data_loss = self._compute_flow_data_loss(data_coords, data_values, data_mask)
                total_loss = physics_loss + 0.5 * data_loss
            else:
                total_loss = physics_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.flow_model.parameters(), max_norm=1.0)
            self.flow_optimizer.step()
            self.flow_scheduler.step(total_loss)
            self.history['flow_loss'].append(total_loss.item())
            self.history['continuity'].append(loss_dict['continuity'])
            self.history['momentum_x'].append(loss_dict['momentum_x'])
            self.history['momentum_y'].append(loss_dict['momentum_y'])
        return total_loss.item()

    def train_sediment_phase(self, n_epochs, T, w_ic=10.0):
        self.flow_model.eval()
        p_k_gauss = self._gradation_at_gauss_tensor()

        for epoch in range(n_epochs):
            # =====================================================
            # Step 1: 更新输沙模型 sediment_model（水体浓度 C_tk）
            #   固定：flow_model, bed_model, active_layer_frac
            # =====================================================
            sediment_dict = None
            if self.sediment_model is not None and self.sediment_optimizer is not None:
                self.sediment_model.train()
                self.bed_model.eval()

                self.sediment_optimizer.zero_grad()
                sediment_loss, sediment_dict = self.sediment_transport_loss_fn.compute_sediment_loss(
                    self.sediment_model, self.flow_model, T, self.device,
                    p_k_override=p_k_gauss,
                )
                sediment_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.sediment_model.parameters(), max_norm=1.0)
                self.sediment_optimizer.step()
                if self.sediment_scheduler is not None:
                    self.sediment_scheduler.step(sediment_loss.detach())

            # =====================================================
            # Step 2: 更新河床模型 bed_model（Exner 方程）
            #   固定：flow_model, sediment_model, active_layer_frac
            # =====================================================
            if self.sediment_model is not None:
                self.sediment_model.eval()
            self.bed_model.train()

            self.bed_optimizer.zero_grad()
            bed_loss, loss_dict = self.exner_loss_fn.compute_loss(
                self.bed_model, self.flow_model, T, self.device, w_ic=w_ic,
                sediment_model=self.sediment_model,
                p_k_override=p_k_gauss,
                sediment_transport_loss_fn=self.sediment_transport_loss_fn,
            )
            bed_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.bed_model.parameters(), max_norm=1.0)
            self.bed_optimizer.step()
            self.bed_scheduler.step(bed_loss.detach())

            # =====================================================
            # 记录历史
            # =====================================================
            total_loss = bed_loss.detach()
            if sediment_dict is not None:
                total_loss = total_loss + sediment_dict['transport'] + sediment_dict['capacity']

            self.history['sediment_loss'].append(total_loss.item())
            self.history['exner'].append(loss_dict['exner'])
            self.history['ic'].append(loss_dict['ic'])
            if sediment_dict is not None:
                self.history['transport_loss'].append(sediment_dict['transport'])
                self.history['capacity_loss'].append(sediment_dict['capacity'])
                self.history['C_min'].append(sediment_dict['C_min'])
                self.history['C_max'].append(sediment_dict['C_max'])
        return total_loss.item()

    def _gradation_at_gauss_tensor(self):
        p = self.active_layer_frac[self.mesh.gauss_cell_id]
        return torch.tensor(p, dtype=torch.float32, device=self.device)

    def _active_layer_thickness_np(self, fractions):
        d = np.asarray(self.sediment_transport_loss_fn.grain_diameters or GRAIN_DIAMETERS, dtype=np.float32)
        order = np.argsort(d)
        d_sorted = d[order]
        f_sorted = fractions[:, order]
        cdf = np.cumsum(f_sorted, axis=1)
        idx = np.argmax(cdf >= 0.9, axis=1)
        d90 = d_sorted[idx]
        return np.maximum(self.sediment_transport_loss_fn.alpha_active_layer * d90, 1.0e-4).astype(np.float32)

    def update_gradation_state(self, T, new_bed):
        if self.sediment_model is None:
            return None

        self.flow_model.eval()
        self.sediment_model.eval()
        p_k_gauss = self._gradation_at_gauss_tensor()
        with torch.no_grad():
            closure = self.sediment_transport_loss_fn.compute_closure(
                self.sediment_model,
                self.flow_model,
                T,
                self.device,
                p_k_override=p_k_gauss,
                freeze_flow_params=True,
            )

        npp = self.mesh.n_points_per_cell
        nc = self.mesh.n_cells
        n_grains = self.active_layer_frac.shape[1]
        net_deposition_rate = (closure['D_tk'] - closure['E_tk']).detach().view(nc, npp, n_grains).mean(dim=1)
        delta_m_bed = net_deposition_rate.cpu().numpy() * self.macro_dt_physical

        rho_bulk = self.sediment_transport_loss_fn.rho_s / max(self.exner_loss_fn.xi, 1.0e-6)
        m1_old = self.active_layer_frac * rho_bulk
        m2_old = self.second_layer_frac * rho_bulk

        delta_zb = np.asarray(new_bed, dtype=np.float32) - self.last_bed.astype(np.float32)
        delta1_old = self.delta1.copy()
        delta2_old = self.delta2.copy()
        delta1_new = self._active_layer_thickness_np(self.active_layer_frac)
        delta2_change = delta_zb - (delta1_new - delta1_old)
        delta2_new = np.maximum(delta2_old + delta2_change, 1.0e-4)

        m_star = np.where(delta2_change[:, None] >= 0.0, m1_old, m2_old)
        m1_new = (delta_m_bed + m1_old * delta1_old[:, None] - m_star * delta2_change[:, None]) / delta1_new[:, None]
        m2_new = (m2_old * delta2_old[:, None] + m_star * delta2_change[:, None]) / delta2_new[:, None]

        m1_new = np.clip(m1_new, 1.0e-12, None)
        m2_new = np.clip(m2_new, 1.0e-12, None)
        self.active_layer_frac = (m1_new / np.sum(m1_new, axis=1, keepdims=True)).astype(np.float32)
        self.second_layer_frac = (m2_new / np.sum(m2_new, axis=1, keepdims=True)).astype(np.float32)
        self.delta1 = delta1_new.astype(np.float32)
        self.delta2 = delta2_new.astype(np.float32)
        self.last_bed = np.asarray(new_bed, dtype=np.float32).copy()
        self.history['gradation_loss'].append(0.0)
        self.history['p_min'].append(float(np.min(self.active_layer_frac)))
        self.history['p_max'].append(float(np.max(self.active_layer_frac)))
        return self.active_layer_frac

    def update_bed_from_model(self, T):
        self.bed_model.eval()
        with torch.no_grad():
            x_centers = torch.tensor(self.mesh.cell_centers_x, dtype=torch.float32, device=self.device)
            y_centers = torch.tensor(self.mesh.cell_centers_y, dtype=torch.float32, device=self.device)
            T_tensor = torch.full_like(x_centers, T)
            if self.flow_loss_fn.bounds is not None:
                bounds = self.flow_loss_fn.bounds
                x_norm = (x_centers - bounds['x_min']) / (bounds['x_max'] - bounds['x_min'])
                y_norm = (y_centers - bounds['y_min']) / (bounds['y_max'] - bounds['y_min'])
            else:
                x_norm = x_centers
                y_norm = y_centers
            xyT = torch.stack([x_norm, y_norm, T_tensor], dim=1)
            zb_new = self.bed_model(xyT).squeeze().cpu().numpy()
        self.mesh.update_bed(zb_new)
        return zb_new

    def pretrain_ic(self, n_epochs):
        """预训练：在正式的耦合训练之前，先单独训练河床模型，使其能够拟合初始的河床形态。"""
        self.bed_model.train()
        cx = torch.tensor(self.mesh.cell_centers_x, dtype=torch.float32, device=self.device)
        cy = torch.tensor(self.mesh.cell_centers_y, dtype=torch.float32, device=self.device)
        b = self.flow_loss_fn.bounds
        if b:
            cx = (cx - b['x_min']) / (b['x_max'] - b['x_min'])
            cy = (cy - b['y_min']) / (b['y_max'] - b['y_min'])
        T0 = torch.zeros(self.mesh.n_cells, dtype=torch.float32, device=self.device)
        xyT0 = torch.stack([cx, cy, T0], dim=1)
        zb_t = torch.tensor(self.mesh.zb_initial, dtype=torch.float32, device=self.device).unsqueeze(1)
        for ep in range(n_epochs):
            self.bed_optimizer.zero_grad()
            loss = F.mse_loss(self.bed_model(xyT0), zb_t)
            loss.backward()
            self.bed_optimizer.step()
        self.bed_optimizer.state.clear()
        return loss.item()

    def _compute_flow_data_loss(self, coords, values, mask=None):
        coords_tensor = torch.tensor(coords, dtype=torch.float32, device=self.device)
        targets = torch.tensor(values, dtype=torch.float32, device=self.device)
        predictions = self.flow_model(coords_tensor)
        if mask is None:
            return F.mse_loss(predictions, targets)
        mask_tensor = torch.tensor(mask, dtype=torch.float32, device=self.device)
        diff2 = (predictions - targets) ** 2 * mask_tensor
        return diff2.sum() / torch.clamp(mask_tensor.sum(), min=1.0)

    def run_decoupled_training(self, n_macro_steps, flow_epochs_per_step, sediment_epochs_per_step, bc_coords=None, bc_values=None, bc_mask=None, warmup_ic_epochs=600, verbose=True):
        self.pretrain_ic(warmup_ic_epochs)
        self.macro_dt_physical = self.exner_loss_fn.T_physical / max(n_macro_steps - 1, 1)
        bed_history = [self.mesh.zb.copy()]
        
        for T_step in trange(n_macro_steps, desc='Macro Steps'):
            T_norm = T_step / (n_macro_steps -1)
            
            if verbose and T_step % 10 == 0:
                print(f'\n  T={T_step}: 训练水动力模型...')
            
            flow_loss = self.train_flow_phase(flow_epochs_per_step, T_norm, data_coords=bc_coords, data_values=bc_values, data_mask=bc_mask)
            
            if verbose and T_step % 10 == 0:
                print(f'  T={T_step}: 训练床面/总输沙/级配模型...')
            
            sediment_loss = self.train_sediment_phase(sediment_epochs_per_step, T_norm, w_ic=compute_ic_weight(T_norm))
            
            current_bed = self.update_bed_from_model(T_norm)
            self.update_gradation_state(T_norm, current_bed)
            bed_history.append(current_bed.copy())
            
            self.history['zb_min'].append(np.min(current_bed))
            self.history['zb_max'].append(np.max(current_bed))
            
            if verbose and T_step % 10 == 0:
                print(f'\n  T={T_step}: Flow Loss={flow_loss:.2e}, Bed+Sed Loss={sediment_loss:.2e}')
                print(f'    河床: [{np.min(current_bed):.4f}, {np.max(current_bed):.4f}]')
        
        return bed_history


def build_boundary_conditions(n_bc=None, t_norm=None, bbox=None, bounds=None):
    if n_bc is None:
        n_bc = int(BC_DEFAULT['n_bc'])
    if t_norm is None:
        t_norm = BC_DEFAULT['t_normalized']
    if bbox is None:
        bbox = BBOX
    if bounds is None:
        bounds = BOUNDS

    h_bc = BC_DEFAULT['h']
    u_bc = BC_DEFAULT['u']
    v_bc = BC_DEFAULT['v']
    bc_value = FlowPINN.encode_target(
        torch.tensor([[h_bc]], dtype=torch.float32),
        torch.tensor([[u_bc]], dtype=torch.float32),
        torch.tensor([[v_bc]], dtype=torch.float32),
        TYPICAL_DEPTH,
        TYPICAL_VELOCITY,
    ).numpy().astype(np.float32)

    x_min = bbox['xmin']
    x_max = bbox['xmax']
    y_min = bbox['ymin']
    y_max = bbox['ymax']
    x_scale = bounds['x_max'] - bounds['x_min']
    y_scale = bounds['y_max'] - bounds['y_min']
    if x_scale == 0 or y_scale == 0:
        raise ValueError("bounds 的 x/y 范围不能为 0。")

    x_left_norm = (x_min - bounds['x_min']) / x_scale
    y_bottom_norm = (y_min - bounds['y_min']) / y_scale
    y_top_norm = (y_max - bounds['y_min']) / y_scale

    # 入口边界条件：在 x=x_min 处，h, u, v 都有约束
    y_inlet = np.linspace(y_min, y_max, n_bc)
    y_inlet_norm = (y_inlet - bounds['y_min']) / y_scale
    coords_inlet = np.stack([np.full(n_bc, x_left_norm), y_inlet_norm, np.ones(n_bc) * t_norm], axis=1).astype(np.float32)
    values_inlet = np.repeat(bc_value, n_bc, axis=0)
    mask_inlet = np.array([[1.0, 1.0, 1.0]] * n_bc, dtype=np.float32)   # 入口边界条件：h, u, v 都有约束

    # 壁面边界条件：在 y=y_min 和 y=y_max 处，只有 v 有约束（无穿透），h 和 u 沿壁面自由变化
    x_wall = np.linspace(x_min, x_max, n_bc)
    x_wall_norm = (x_wall - bounds['x_min']) / x_scale
    coords_wall_bot = np.stack([x_wall_norm, np.full(n_bc, y_bottom_norm), np.ones(n_bc) * t_norm], axis=1).astype(np.float32)
    coords_wall_top = np.stack([x_wall_norm, np.full(n_bc, y_top_norm), np.ones(n_bc) * t_norm], axis=1).astype(np.float32)
    values_wall = np.repeat(bc_value, n_bc, axis=0)
    mask_wall = np.array([[0.0, 0.0, 1.0]] * n_bc, dtype=np.float32)

    bc_coords = np.concatenate([coords_inlet, coords_wall_bot, coords_wall_top], axis=0)
    bc_values = np.concatenate([values_inlet, values_wall, values_wall], axis=0)
    bc_mask = np.concatenate([mask_inlet, mask_wall, mask_wall], axis=0)
    return bc_coords, bc_values, bc_mask


def run_hump_evolution_test(regime='fast'):
    Ag = AG_VALUES.get(regime, 1.0)
    T_physical = TPHYSICAL.get(regime, 600.0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 网格预处理
    fvm_mesh = FVMeshPreprocessor(bbox=BBOX, resolution=RESOLUTION, initial_bed=hump_initial_bed, n_gauss_points=N_GAUSS_POINTS)
    # 1. 水动力方程
    flow_model = FlowPINN(input_dim=3, hidden_dim=64, num_block=4, output_dim=3).to(device)
    # 2. 沉积物输运方程：预测水体中各粒径浓度 C_tk
    sediment_model = SedimentPINN(input_dim=3, hidden_dim=64, num_block=4, output_dim=NUM_GRAIN_CLASSES, positive_output=True).to(device)
    # 3. 河床演化方程：预测河床高程 z_b（Exner方程）
    bed_model = BedPINN(input_dim=3, hidden_dim=64, num_block=4, output_dim=1, zb_scale=1.5).to(device)
    # 水动力物理损失
    flow_loss_fn = SVEsPhysicsLoss(
        fvm_mesh=fvm_mesh, 
        g=9.81, 
        n_manning=0.01, 
        bounds=BOUNDS, 
        typical_depth=TYPICAL_DEPTH,
        typical_velocity=TYPICAL_VELOCITY,
        T_physical=T_physical,
        include_time_terms=INCLUDE_FLOW_TIME_TERMS,
    )

    # 沉积物输运物理损失（浓度对流扩散 + 级配）
    sediment_transport_loss_fn = SedimentTransportLoss(
        fvm_mesh=fvm_mesh,
        bounds=BOUNDS,
        typical_depth=TYPICAL_DEPTH,
        typical_velocity=TYPICAL_VELOCITY,
        beta_default=1.0,
        epsilon_default=0.1,
        residual_scale=1.0,
        grain_diameters=GRAIN_DIAMETERS,    # 粒径数据，单位米
        Ag=Ag,  # 级配对输沙的影响强度，Ag 越大，级配变化对输沙的影响越显著
        m=3,    # Exner 方程中沉积物输运率与水动力条件的非线性关系指数
        adaptation_length=50.0, # 适应长度，用于计算局部的适应性权重，单位米
        T_physical=T_physical,
        alpha_active_layer=10.0,    # 活动层厚度与特征粒径的比例系数，用于定义活动层厚度
        w_capacity=0.05,    # 容量损失权重
    )

    # 河床演化物理损失（Exner 方程）
    exner_loss_fn = ExnerPhysicsLoss(
        fvm_mesh=fvm_mesh,
        porosity=0.4,
        Ag=Ag,
        m=3,
        Q=10.0,
        h0=10.0,
        bounds=BOUNDS,
        T_physical=T_physical,
        typical_u=TYPICAL_VELOCITY,
    )

    trainer = DecoupledTrainer(
        flow_model=flow_model,
        sediment_model=sediment_model,
        bed_model=bed_model,
        fvm_mesh=fvm_mesh,
        device=device,
        flow_loss_fn=flow_loss_fn,
        sediment_transport_loss_fn=sediment_transport_loss_fn,
        exner_loss_fn=exner_loss_fn,
        flow_lr=TRAINING_SETTINGS['flow_lr'],
        transport_lr=TRAINING_SETTINGS['transport_lr'],
        bed_lr=TRAINING_SETTINGS['sediment_lr'],
    )

    bc_coords, bc_values, bc_mask = build_boundary_conditions(bbox=BBOX, bounds=BOUNDS)
    
    bed_history = trainer.run_decoupled_training(
        n_macro_steps=TRAINING_SETTINGS['n_macro_steps'],   # 总的宏观时间步数，即河床演化的时间分辨率
        flow_epochs_per_step=TRAINING_SETTINGS['flow_epochs_per_step'], # 每个宏观时间步内水动力模型的训练轮数
        sediment_epochs_per_step=TRAINING_SETTINGS['sediment_epochs_per_step'], # 每个宏观时间步内沉积物输运模型的训练轮数
        bc_coords=bc_coords,
        bc_values=bc_values,
        bc_mask=bc_mask,
        warmup_ic_epochs=TRAINING_SETTINGS['warmup_ic_epochs'],
        verbose=True,
    )

    visualize_results(fvm_mesh, bed_history, BBOX, RESOLUTION, trainer.history, T_physical, Ag, regime)
    
    return trainer, bed_history
