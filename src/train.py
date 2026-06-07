# train.py - 三阶段 PINN 训练流程
# 阶段 1：只训练水动力 PINN；阶段 2：冻结水动力训练泥沙 PINN；
# 阶段 3：两个网络联合优化。床面历史由泥沙网络输出的 Δzb 生成。

import numpy as np
import torch
import torch.nn.functional as F

from .model import FlowPINN
from .utils import build_xyt


class DecoupledTrainer:
    """三阶段训练控制器。

    这里不直接求解传统数值格式，而是把 PDE、边界条件和床变方程写成 loss：
    1. 先训练 flow_model，使 h/u/v 满足浅水方程和真实边界；
    2. 冻结 flow_model，训练 sediment_model 的 C_k 和 Δzb；
    3. 最后两个网络联合优化，减少分阶段训练带来的不一致。
    """
    def __init__(
        self,
        flow_model,
        sediment_model,
        fvm_mesh,
        device,
        flow_loss_fn,
        sediment_transport_loss_fn,
        simulation_time,
        initial_gradation=None,
        active_layer_thickness=0.5,
        flow_lr=1e-4,
        transport_lr=1e-4,
    ):
        self.flow_model = flow_model
        self.sediment_model = sediment_model
        self.mesh = fvm_mesh
        self.device = device
        self.flow_loss_fn = flow_loss_fn
        self.sediment_transport_loss_fn = sediment_transport_loss_fn
        self.simulation_time = simulation_time
        self.active_layer_thickness = float(active_layer_thickness)
        # active_layer_frac 是每个网格单元的活动层分级比例 p_k。
        # 初始值来自 Excel 主河道级配，后续按 morph_dt 用 Exner 床变速率显式更新。
        self.active_layer_frac = self._init_gradation(initial_gradation)
        # 固壁边界：active 河道单元与 inactive/外部相邻的边。
        # top 入口和 bottom 出口会被排除，不施加 no-flux。
        self.wall_gauss_mask = self._build_wall_gauss_mask()

        # 三个优化器分别对应三阶段训练，避免手动开关参数组。
        self.flow_optimizer = torch.optim.Adam(self.flow_model.parameters(), lr=flow_lr)
        self.sediment_optimizer = (
            torch.optim.Adam(self.sediment_model.parameters(), lr=transport_lr)
            if self.sediment_model is not None else None
        )

        # 学习率调度器：如果 loss 长时间不下降，则降低学习率。
        self.flow_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.flow_optimizer, 'min', factor=0.5, patience=500)
        self.sediment_scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(self.sediment_optimizer, 'min', factor=0.5, patience=500)
            if self.sediment_optimizer is not None else None
        )
        self.joint_optimizer = (
            torch.optim.Adam(
                list(self.flow_model.parameters()) + list(self.sediment_model.parameters()),
                lr=min(flow_lr, transport_lr),
            )
            if self.sediment_model is not None else None
        )
        self.joint_scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(self.joint_optimizer, 'min', factor=0.5, patience=500)
            if self.joint_optimizer is not None else None
        )

        self.history = {
            'flow_loss': [],
            'sediment_loss': [],
            'joint_loss': [],
            'transport_loss': [],
            'capacity_loss': [],
            'initial_sediment_loss': [],
            'inlet_sediment_loss': [],
            'bed_change_loss': [],
            'bed_initial_loss': [],
            'continuity': [],
            'momentum_x': [],
            'momentum_y': [],
            'zb_min': [],
            'zb_max': [],
            'C_min': [],
            'C_max': [],
            'dzb_min': [],
            'dzb_max': [],
            'Ceq_mean': [],
            'p_min': [],
            'p_max': [],
            'diagnostic_time': [],
            'q_in_target': [],
            'q_in_model': [],
            'stage_out_target': [],
            'stage_out_model': [],
            'h_min': [],
            'h_max': [],
            'u_mean': [],
            'v_mean': [],
            'C_mean_diag': [],
            'Ceq_mean_diag': [],
            'dzb_min_diag': [],
            'dzb_max_diag': [],
            'p_min_diag': [],
            'p_max_diag': [],
            'wall_un_abs_mean': [],
        }

    def _init_gradation(self, initial_gradation):
        """用 Excel 级配初始化活动层；如果未提供，则退化为均匀级配。"""
        n_grains = len(self.sediment_transport_loss_fn.grain_diameters)
        if initial_gradation is None:
            base = np.full(n_grains, 1.0 / max(n_grains, 1), dtype=np.float32)
        else:
            base = np.asarray(initial_gradation, dtype=np.float32)
            if base.size != n_grains:
                raise ValueError("初始级配数量必须与 grain_diameters 一致。")
            base = np.clip(base, 1.0e-8, None)
            base = base / np.sum(base)
        return np.tile(base.reshape(1, -1), (self.mesh.n_cells, 1)).astype(np.float32)

    def _gradation_at_gauss_tensor(self):
        """把单元活动层级配映射到高斯点，供输沙能力闭合使用。"""
        p_gauss = self.active_layer_frac[self.mesh.gauss_cell_id]
        return torch.tensor(p_gauss, dtype=torch.float32, device=self.device)

    def _build_wall_gauss_mask(self):
        """识别固壁边界高斯点。

        判据：
        - 当前高斯点所属单元是 active 河道单元；
        - 该边的相邻单元不存在，或相邻单元是 inactive；
        - 排除 top 入口边和 bottom 出口边。

        返回布尔数组，长度等于 mesh.n_gauss_total。
        """
        active = getattr(self.mesh, 'active_cell_mask', np.ones(self.mesh.n_cells, dtype=bool))
        cell_id = self.mesh.gauss_cell_id
        edge_id = self.mesh.gauss_edge_id
        neighbor_id = self.mesh.gauss_neighbor_id

        current_active = active[cell_id]
        neighbor_active = np.zeros_like(current_active, dtype=bool)
        inside = neighbor_id >= 0
        neighbor_active[inside] = active[neighbor_id[inside]]
        wall = current_active & (~neighbor_active)

        active_2d = getattr(self.mesh, 'active_cell_mask_2d', active.reshape(self.mesh.ny, self.mesh.nx))
        active_rows = np.where(active_2d.any(axis=1))[0]
        if active_rows.size == 0:
            return np.zeros_like(wall, dtype=bool)

        top_row = int(active_rows[-1])
        bottom_row = int(active_rows[0])
        top_cols = np.where(active_2d[top_row, :])[0]
        bottom_cols = np.where(active_2d[bottom_row, :])[0]
        top_cell_ids = self.mesh.cell_index[top_row, top_cols]
        bottom_cell_ids = self.mesh.cell_index[bottom_row, bottom_cols]

        inlet_mask = (edge_id == 2) & np.isin(cell_id, top_cell_ids)
        outlet_mask = (edge_id == 0) & np.isin(cell_id, bottom_cell_ids)
        return wall & (~inlet_mask) & (~outlet_mask)

    def train_flow_phase(self, n_epochs, T_norm, bc_builder):
        """阶段 1：训练水动力 PINN。

        T_norm 可以是多个归一化时间点；每个 epoch 会在这些时间点上分别计算
        浅水方程残差和真实边界 loss，再取平均。
        """
        time_list = [float(t) for t in np.asarray(T_norm, dtype=np.float32).ravel()]
        self.flow_model.train()
        total_loss_value = 0.0
        for epoch in range(n_epochs):
            self.flow_optimizer.zero_grad()
            total_loss_value = 0.0
            loss_acc = {'continuity': 0.0, 'momentum_x': 0.0, 'momentum_y': 0.0}
            for t_i in time_list:
                # 水动力阶段：浅水方程残差 + 真实边界（入口流量、出口水位）约束。
                physics_loss, loss_dict = self.flow_loss_fn.compute_loss(self.flow_model, t_i, self.device)
                boundary_loss = self._compute_real_flow_boundary_loss(bc_builder(t_i))
                # 边界 loss 权重先固定为 0.5；后续如果 Q 或 stage 偏差大，可单独配置。
                step_loss = physics_loss + 0.5 * boundary_loss
                (step_loss / len(time_list)).backward()
                total_loss_value += float(step_loss.detach().cpu()) / len(time_list)
                for key in loss_acc:
                    loss_acc[key] += loss_dict[key] / len(time_list)
            torch.nn.utils.clip_grad_norm_(self.flow_model.parameters(), max_norm=1.0)
            self.flow_optimizer.step()
            self.flow_scheduler.step(total_loss_value)
            self.history['flow_loss'].append(total_loss_value)
            self.history['continuity'].append(loss_acc['continuity'])
            self.history['momentum_x'].append(loss_acc['momentum_x'])
            self.history['momentum_y'].append(loss_acc['momentum_y'])
        return total_loss_value

    def train_sediment_phase(self, n_epochs, T_norm, freeze_flow_params=True):
        """阶段 2：训练泥沙 PINN。

        freeze_flow_params=True 时，水动力场只作为已训练好的背景场提供 h/u/v，
        梯度不会更新 flow_model；泥沙网络学习分粒径浓度 C_k 和累计床变 Δzb。
        """
        time_list = [float(t) for t in np.asarray(T_norm, dtype=np.float32).ravel()]
        self.flow_model.eval()
        # 当前活动层级配会影响输沙能力闭合，因此每个训练阶段开始时映射到高斯点。
        p_k_gauss = self._gradation_at_gauss_tensor()
        total_loss_value = 0.0

        for epoch in range(n_epochs):
            total_loss_value = 0.0
            loss_acc = {
                'transport': 0.0,
                'capacity': 0.0,
                'initial': 0.0,
                'inlet': 0.0,
                'bed_change': 0.0,
                'bed_initial': 0.0,
                'C_min': None,
                'C_max': None,
                'dzb_min': None,
                'dzb_max': None,
                'Ceq_mean': 0.0,
            }
            if self.sediment_model is not None and self.sediment_optimizer is not None:
                self.sediment_model.train()

                self.sediment_optimizer.zero_grad()
                for t_i in time_list:
                    # compute_sediment_loss 内部包含：
                    # 输沙 PDE 残差、C≈C_capacity、入口平衡浓度、Δzb 的 Exner 约束。
                    sediment_loss, sediment_dict = self.sediment_transport_loss_fn.compute_sediment_loss(
                        self.sediment_model, self.flow_model, t_i, self.device,
                        p_k_override=p_k_gauss,
                        freeze_flow_params=freeze_flow_params,
                    )
                    (sediment_loss / len(time_list)).backward()
                    total_loss_value += float(sediment_loss.detach().cpu()) / len(time_list)
                    loss_acc['transport'] += sediment_dict['transport'] / len(time_list)
                    loss_acc['capacity'] += sediment_dict['capacity'] / len(time_list)
                    loss_acc['initial'] += sediment_dict['initial'] / len(time_list)
                    loss_acc['inlet'] += sediment_dict['inlet'] / len(time_list)
                    loss_acc['bed_change'] += sediment_dict['bed_change'] / len(time_list)
                    loss_acc['bed_initial'] += sediment_dict['bed_initial'] / len(time_list)
                    loss_acc['Ceq_mean'] += sediment_dict['Ceq_mean'] / len(time_list)
                    loss_acc['C_min'] = (
                        sediment_dict['C_min']
                        if loss_acc['C_min'] is None
                        else min(loss_acc['C_min'], sediment_dict['C_min'])
                    )
                    loss_acc['C_max'] = (
                        sediment_dict['C_max']
                        if loss_acc['C_max'] is None
                        else max(loss_acc['C_max'], sediment_dict['C_max'])
                    )
                    loss_acc['dzb_min'] = (
                        sediment_dict['dzb_min']
                        if loss_acc['dzb_min'] is None
                        else min(loss_acc['dzb_min'], sediment_dict['dzb_min'])
                    )
                    loss_acc['dzb_max'] = (
                        sediment_dict['dzb_max']
                        if loss_acc['dzb_max'] is None
                        else max(loss_acc['dzb_max'], sediment_dict['dzb_max'])
                    )

                torch.nn.utils.clip_grad_norm_(self.sediment_model.parameters(), max_norm=1.0)
                self.sediment_optimizer.step()
                if self.sediment_scheduler is not None:
                    self.sediment_scheduler.step(total_loss_value)

            self.history['sediment_loss'].append(total_loss_value)
            if loss_acc['C_min'] is not None:
                self.history['transport_loss'].append(loss_acc['transport'])
                self.history['capacity_loss'].append(loss_acc['capacity'])
                self.history['initial_sediment_loss'].append(loss_acc['initial'])
                self.history['inlet_sediment_loss'].append(loss_acc['inlet'])
                self.history['bed_change_loss'].append(loss_acc['bed_change'])
                self.history['bed_initial_loss'].append(loss_acc['bed_initial'])
                self.history['C_min'].append(loss_acc['C_min'])
                self.history['C_max'].append(loss_acc['C_max'])
                self.history['dzb_min'].append(loss_acc['dzb_min'])
                self.history['dzb_max'].append(loss_acc['dzb_max'])
                self.history['Ceq_mean'].append(loss_acc['Ceq_mean'])
        return total_loss_value

    def train_joint_phase(self, n_epochs, T_norm, data_coords=None):
        """阶段 3：联合优化水动力和泥沙网络。

        联合阶段不再冻结 flow_model，因此泥沙 loss 中的梯度也可以反馈到 h/u/v。
        这一步用于修正“先水后沙”分阶段训练造成的弱耦合误差。
        """
        if self.sediment_model is None or self.joint_optimizer is None:
            return 0.0

        time_list = [float(t) for t in np.asarray(T_norm, dtype=np.float32).ravel()]
        p_k_gauss = self._gradation_at_gauss_tensor()
        total_loss_value = 0.0

        for epoch in range(n_epochs):
            self.flow_model.train()
            self.sediment_model.train()
            self.joint_optimizer.zero_grad()
            total_loss_value = 0.0

            for t_i in time_list:
                # 联合 loss = 水动力方程/边界 + 泥沙方程/床变约束。
                flow_loss, _ = self.flow_loss_fn.compute_loss(self.flow_model, t_i, self.device)
                flow_loss = flow_loss + 0.5 * self._compute_real_flow_boundary_loss(data_coords(t_i))

                sediment_loss, _ = self.sediment_transport_loss_fn.compute_sediment_loss(
                    self.sediment_model,
                    self.flow_model,
                    t_i,
                    self.device,
                    p_k_override=p_k_gauss,
                    freeze_flow_params=False,
                )
                step_loss = flow_loss + sediment_loss
                (step_loss / len(time_list)).backward()
                total_loss_value += float(step_loss.detach().cpu()) / len(time_list)

            torch.nn.utils.clip_grad_norm_(
                list(self.flow_model.parameters()) + list(self.sediment_model.parameters()),
                max_norm=1.0,
            )
            self.joint_optimizer.step()
            if self.joint_scheduler is not None:
                self.joint_scheduler.step(total_loss_value)
            self.history['joint_loss'].append(total_loss_value)

        return total_loss_value

    def _compute_real_flow_boundary_loss(self, payload):
        """真实边界损失：入口总流量、出口水位、侧壁 no-flux。"""
        inlet_coords = torch.tensor(payload['inlet_coords'], dtype=torch.float32, device=self.device)
        inlet_weights = torch.tensor(payload['inlet_weights'], dtype=torch.float32, device=self.device).view(-1, 1)
        inlet_out = self.flow_model(inlet_coords)
        h_in, _, v_in = FlowPINN.decode_output(
            inlet_out,
            payload['typical_depth'],
            payload['typical_velocity'],
        )
        if payload.get('inlet_edge') == 'top':
            # top 入口的外法向指向 +y，流入河道为负 v，因此流量用 h*(-v)。
            q_pred = torch.sum(h_in * (-v_in) * inlet_weights)
        else:
            q_pred = torch.sum(h_in * v_in * inlet_weights)
        q_target = torch.tensor(float(payload['target_flow']), dtype=torch.float32, device=self.device)
        flow_loss = ((q_pred - q_target) / torch.clamp(torch.abs(q_target), min=1.0)) ** 2

        outlet_coords = torch.tensor(payload['outlet_coords'], dtype=torch.float32, device=self.device)
        outlet_bed = torch.tensor(payload['outlet_bed'], dtype=torch.float32, device=self.device).view(-1, 1)
        outlet_out = self.flow_model(outlet_coords)
        h_out, _, _ = FlowPINN.decode_output(
            outlet_out,
            payload['typical_depth'],
            payload['typical_velocity'],
        )
        stage_target = torch.tensor(float(payload['target_stage']), dtype=torch.float32, device=self.device)
        # Excel 给的是绝对水位 stage，模型输出的是水深 h，所以需要 h=stage-zb。
        h_target = torch.clamp(stage_target - outlet_bed, min=float(payload.get('h_min', 0.05)))
        stage_loss = F.mse_loss(h_out, h_target) / max(float(payload['typical_depth']) ** 2, 1.0)

        # 固壁边界不允许水流穿过河道侧壁：u_n = u*nx + v*ny = 0。
        wall_loss = self._compute_wall_no_flux_loss(
            payload['t_norm'],
            payload['typical_velocity'],
        )
        return flow_loss + stage_loss + wall_loss

    def _compute_wall_no_flux_loss(self, t_norm, typical_velocity):
        """侧壁 no-flux 损失。

        在 active/inactive 交界边和外部侧壁边上，约束法向速度为 0。
        这里只约束速度方向，不约束水深，因此不会和入口流量、出口水位冲突。
        """
        if not np.any(self.wall_gauss_mask):
            return torch.zeros((), dtype=torch.float32, device=self.device)

        coords = torch.tensor(
            self.mesh.gauss_coords[self.wall_gauss_mask],
            dtype=torch.float32,
            device=self.device,
        )
        normals = torch.tensor(
            self.mesh.gauss_normals[self.wall_gauss_mask],
            dtype=torch.float32,
            device=self.device,
        )
        xyt = build_xyt(
            coords,
            float(t_norm),
            self.flow_loss_fn.bounds,
            self.device,
            requires_grad=False,
        )
        wall_out = self.flow_model(xyt)
        _, u_wall, v_wall = FlowPINN.decode_output(
            wall_out,
            self.flow_loss_fn.typical_h,
            self.flow_loss_fn.typical_u,
        )
        un = u_wall * normals[:, 0:1] + v_wall * normals[:, 1:2]
        return torch.mean((un / max(float(typical_velocity), 1.0e-6)) ** 2)

    def _time_slices(self, start_time, end_time, sample_dt):
        """生成训练采样时间点，并转为网络使用的归一化时间 t/T。"""
        times = [start_time]
        t = start_time + sample_dt
        eps = max(sample_dt, self.simulation_time) * 1.0e-9
        while t < end_time - eps:
            times.append(t)
            t += sample_dt
        if end_time > start_time + eps:
            times.append(end_time)
        return np.asarray(times, dtype=np.float32) / max(self.simulation_time, 1.0e-12)

    def _output_times(self, simulation_time, output_dt):
        """生成物理时间序列，可用于输出时间或级配更新时间。"""
        times = [0.0]
        t = float(output_dt)
        eps = max(simulation_time, output_dt) * 1.0e-9
        while t < simulation_time - eps:
            times.append(float(t))
            t += output_dt
        if simulation_time > eps:
            times.append(float(simulation_time))
        return times

    def predict_bed_history(self, output_times):
        """在输出时刻预测床面历史。

        sediment_model 直接输出累计床变 Δzb(x,y,t)，因此床面为：
        zb(t) = zb_initial + Δzb(t)。
        """
        coords = torch.tensor(
            np.stack([self.mesh.cell_centers_x, self.mesh.cell_centers_y], axis=1),
            dtype=torch.float32,
            device=self.device,
        )
        n_grains = len(self.sediment_transport_loss_fn.grain_diameters)
        bed_history = []
        self.sediment_model.eval()
        for t in output_times:
            t_norm = float(t) / max(self.simulation_time, 1.0e-12)
            xyt = build_xyt(coords, t_norm, self.sediment_transport_loss_fn.bounds, self.device, requires_grad=False)
            with torch.no_grad():
                sediment_out = self.sediment_model(xyt)
                # 前 K 列是 C_k，最后一列才是累计床变 Δzb。
                dzb = sediment_out[:, n_grains:n_grains + 1].squeeze(1).detach().cpu().numpy()
            bed_history.append((self.mesh.zb_initial + dzb).astype(np.float32))
        return bed_history

    def update_gradation_history(self, morph_times):
        """按 morph_dt 显式更新活动层级配。

        这里采用轻量活动层守恒更新：Exner 给出各粒径床变速率 dzb_dt_k，
        乘以 morph_dt 后改变活动层内各粒径体积分数。该更新用于后续时间段
        的输沙闭合，不直接反向传播；它和绘图输出 output_dt 解耦。
        """
        if len(morph_times) < 2:
            return
        self.flow_model.eval()
        self.sediment_model.eval()
        for t0, t1 in zip(morph_times[:-1], morph_times[1:]):
            t_norm = float(t1) / max(self.simulation_time, 1.0e-12)
            dt = float(t1 - t0)
            p_k_gauss = self._gradation_at_gauss_tensor()
            closure = self.sediment_transport_loss_fn.compute_closure(
                self.sediment_model,
                self.flow_model,
                t_norm,
                self.device,
                p_k_override=p_k_gauss,
                freeze_flow_params=True,
            )
            dzb_dt_k = self.sediment_transport_loss_fn.exner_dzb_dt_k_cell(closure)
            # dzb_dt_k 是各粒径组对床变的贡献；除以活动层厚度后近似转成比例变化。
            delta_frac = dzb_dt_k.detach().cpu().numpy() * dt / max(self.active_layer_thickness, 1.0e-6)
            active = getattr(self.mesh, 'active_cell_mask', np.ones(self.mesh.n_cells, dtype=bool))
            updated = self.active_layer_frac.copy()
            # 只更新河道活动单元；非河道单元的级配不参与物理计算。
            updated[active] = np.clip(updated[active] + delta_frac[active], 1.0e-8, None)
            updated[active] = updated[active] / np.sum(updated[active], axis=1, keepdims=True)
            self.active_layer_frac = updated.astype(np.float32)
            self.history['p_min'].append(float(np.min(self.active_layer_frac[active])))
            self.history['p_max'].append(float(np.max(self.active_layer_frac[active])))

    def record_diagnostics(self, output_times, bc_builder):
        """在输出时刻记录物理诊断量，便于判断结果是否可信。"""
        active_cell = getattr(self.mesh, 'active_cell_mask', np.ones(self.mesh.n_cells, dtype=bool))
        active_gauss = active_cell[self.mesh.gauss_cell_id]
        active_gauss_t = torch.tensor(active_gauss, dtype=torch.bool, device=self.device)
        coords = torch.tensor(self.mesh.gauss_coords, dtype=torch.float32, device=self.device)

        self.flow_model.eval()
        if self.sediment_model is not None:
            self.sediment_model.eval()

        for t in output_times:
            t_norm = float(t) / max(self.simulation_time, 1.0e-12)
            payload = bc_builder(t_norm)
            # 边界诊断检查模型是否真的满足真实 Q(t) 和 stage(t)。
            boundary_diag = self._flow_boundary_diagnostics(payload)
            # 场变量诊断检查水深、速度、浓度、床变和级配是否出现非物理量级。
            field_diag = self._field_diagnostics(t_norm, coords, active_gauss_t)

            self.history['diagnostic_time'].append(float(t))
            self.history['q_in_target'].append(boundary_diag['q_target'])
            self.history['q_in_model'].append(boundary_diag['q_model'])
            self.history['stage_out_target'].append(boundary_diag['stage_target'])
            self.history['stage_out_model'].append(boundary_diag['stage_model'])
            for key, value in field_diag.items():
                self.history[key].append(value)

    def _flow_boundary_diagnostics(self, payload):
        inlet_coords = torch.tensor(payload['inlet_coords'], dtype=torch.float32, device=self.device)
        inlet_weights = torch.tensor(payload['inlet_weights'], dtype=torch.float32, device=self.device).view(-1, 1)
        outlet_coords = torch.tensor(payload['outlet_coords'], dtype=torch.float32, device=self.device)
        outlet_bed = torch.tensor(payload['outlet_bed'], dtype=torch.float32, device=self.device).view(-1, 1)

        with torch.no_grad():
            h_in, _, v_in = FlowPINN.decode_output(
                self.flow_model(inlet_coords),
                payload['typical_depth'],
                payload['typical_velocity'],
            )
            if payload.get('inlet_edge') == 'top':
                q_model = torch.sum(h_in * (-v_in) * inlet_weights)
            else:
                q_model = torch.sum(h_in * v_in * inlet_weights)

            h_out, _, _ = FlowPINN.decode_output(
                self.flow_model(outlet_coords),
                payload['typical_depth'],
                payload['typical_velocity'],
            )
            stage_model = torch.mean(h_out + outlet_bed)

        return {
            'q_target': float(payload['target_flow']),
            'q_model': float(q_model.detach().cpu()),
            'stage_target': float(payload['target_stage']),
            'stage_model': float(stage_model.detach().cpu()),
        }

    def _field_diagnostics(self, t_norm, coords, active_gauss_t):
        """统计河道 active_mask 内的场变量范围。"""
        xyt = build_xyt(
            coords,
            t_norm,
            self.flow_loss_fn.bounds,
            self.device,
            requires_grad=False,
        )
        with torch.no_grad():
            h, u, v = FlowPINN.decode_output(
                self.flow_model(xyt),
                self.flow_loss_fn.typical_h,
                self.flow_loss_fn.typical_u,
            )
            h_a = h[active_gauss_t]
            u_a = u[active_gauss_t]
            v_a = v[active_gauss_t]

        diag = {
            'h_min': float(torch.min(h_a).detach().cpu()),
            'h_max': float(torch.max(h_a).detach().cpu()),
            'u_mean': float(torch.mean(u_a).detach().cpu()),
            'v_mean': float(torch.mean(v_a).detach().cpu()),
            'C_mean_diag': 0.0,
            'Ceq_mean_diag': 0.0,
            'dzb_min_diag': 0.0,
            'dzb_max_diag': 0.0,
            'p_min_diag': float(np.min(self.active_layer_frac[getattr(self.mesh, 'active_cell_mask', np.ones(self.mesh.n_cells, dtype=bool))])),
            'p_max_diag': float(np.max(self.active_layer_frac[getattr(self.mesh, 'active_cell_mask', np.ones(self.mesh.n_cells, dtype=bool))])),
            'wall_un_abs_mean': self._wall_un_abs_mean(t_norm),
        }

        if self.sediment_model is None:
            return diag

        p_k_gauss = self._gradation_at_gauss_tensor()
        # 这里使用 compute_closure，是为了同时拿到 C、C_capacity 和 Δzb，
        # 避免诊断逻辑重复实现一遍输沙闭合。
        closure = self.sediment_transport_loss_fn.compute_closure(
            self.sediment_model,
            self.flow_model,
            t_norm,
            self.device,
            p_k_override=p_k_gauss,
            freeze_flow_params=True,
        )
        C_a = closure['C_tk'][active_gauss_t]
        Ceq_a = closure['C_capacity'][active_gauss_t]
        dzb_a = closure['dzb_pred'][active_gauss_t]
        diag.update({
            'C_mean_diag': float(torch.mean(C_a).detach().cpu()),
            'Ceq_mean_diag': float(torch.mean(Ceq_a).detach().cpu()),
            'dzb_min_diag': float(torch.min(dzb_a).detach().cpu()),
            'dzb_max_diag': float(torch.max(dzb_a).detach().cpu()),
        })
        return diag

    def _wall_un_abs_mean(self, t_norm):
        """诊断固壁边界平均法向速度绝对值。"""
        if not np.any(self.wall_gauss_mask):
            return 0.0
        coords = torch.tensor(
            self.mesh.gauss_coords[self.wall_gauss_mask],
            dtype=torch.float32,
            device=self.device,
        )
        normals = torch.tensor(
            self.mesh.gauss_normals[self.wall_gauss_mask],
            dtype=torch.float32,
            device=self.device,
        )
        xyt = build_xyt(
            coords,
            float(t_norm),
            self.flow_loss_fn.bounds,
            self.device,
            requires_grad=False,
        )
        with torch.no_grad():
            wall_out = self.flow_model(xyt)
            _, u_wall, v_wall = FlowPINN.decode_output(
                wall_out,
                self.flow_loss_fn.typical_h,
                self.flow_loss_fn.typical_u,
            )
            un = u_wall * normals[:, 0:1] + v_wall * normals[:, 1:2]
        return float(torch.mean(torch.abs(un)).detach().cpu())

    def run_training(
        self,
        simulation_time,
        sample_dt,
        morph_dt,
        output_dt,
        flow_epochs,
        sediment_epochs,
        bc_builder,
        joint_epochs=0,
        verbose=True,
    ):
        """执行完整训练流程。

        sample_dt 控制训练约束时间点；
        morph_dt 控制级配显式更新时间；
        output_dt 只控制床面历史和图像输出时间。
        """

        self.simulation_time = float(simulation_time)
        self.joint_epochs = int(joint_epochs or 0)
        self.history['time'] = [float(simulation_time)]

        T_norm_list = self._time_slices(0.0, simulation_time, sample_dt)
        if verbose:
            print(f'\n阶段 1/3: 训练水动力 PINN，时间点数={len(T_norm_list)}')

        flow_loss = self.train_flow_phase(flow_epochs, T_norm_list, bc_builder)

        if verbose:
            print(f'阶段 1 完成: Flow Loss={flow_loss:.2e}')
            print('\n阶段 2/3: 冻结水动力，训练泥沙 PINN 和累计床变 Δzb')

        sediment_loss = self.train_sediment_phase(
            sediment_epochs,
            T_norm_list,
            freeze_flow_params=True,
        )

        if verbose:
            print(f'阶段 2 完成: Sediment Loss={sediment_loss:.2e}')
            print('\n阶段 3/3: 联合优化水动力 PINN 和泥沙 PINN')

        joint_epochs = int(getattr(self, 'joint_epochs', 0))
        joint_loss = self.train_joint_phase(joint_epochs, T_norm_list, data_coords=bc_builder)

        # 训练完成后再统一生成床面历史、更新级配历史和记录诊断。
        # 目前床面历史来自网络预测 Δzb；级配更新是后处理式显式更新。
        morph_times = self._output_times(simulation_time, morph_dt)
        output_times = self._output_times(simulation_time, output_dt)
        bed_history = self.predict_bed_history(output_times)
        self.update_gradation_history(morph_times)
        self.record_diagnostics(output_times, bc_builder)
        self.history['morph_times'] = morph_times
        self.history['output_times'] = output_times
        self.history['zb_min'] = [float(np.min(zb)) for zb in bed_history]
        self.history['zb_max'] = [float(np.max(zb)) for zb in bed_history]

        if verbose:
            print(f'阶段 3 完成: Joint Loss={joint_loss:.2e}')
            print(f'级配更新时间点: {morph_times}')
            print(f'输出时间点: {output_times}')
            self.print_diagnostic_summary()

        return bed_history

    def print_diagnostic_summary(self):
        """打印最后一个输出时刻的关键诊断量。"""
        if not self.history.get('diagnostic_time'):
            return
        i = -1
        print('\n关键物理诊断（最后输出时刻）:')
        print(
            f"  Q_in: model={self.history['q_in_model'][i]:.4f}, "
            f"target={self.history['q_in_target'][i]:.4f} m3/s"
        )
        print(
            f"  stage_out: model={self.history['stage_out_model'][i]:.4f}, "
            f"target={self.history['stage_out_target'][i]:.4f} m"
        )
        print(
            f"  h=[{self.history['h_min'][i]:.4f}, {self.history['h_max'][i]:.4f}] m, "
            f"u_mean={self.history['u_mean'][i]:.4f} m/s, "
            f"v_mean={self.history['v_mean'][i]:.4f} m/s"
        )
        print(f"  wall |u_n| mean={self.history['wall_un_abs_mean'][i]:.4e} m/s")
        print(
            f"  C_mean={self.history['C_mean_diag'][i]:.4e}, "
            f"Ceq_mean={self.history['Ceq_mean_diag'][i]:.4e}, "
            f"dzb=[{self.history['dzb_min_diag'][i]:.4f}, {self.history['dzb_max_diag'][i]:.4f}] m"
        )
        print(
            f"  p_k=[{self.history['p_min_diag'][i]:.4e}, "
            f"{self.history['p_max_diag'][i]:.4e}]"
        )
