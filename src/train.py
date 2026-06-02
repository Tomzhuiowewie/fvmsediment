# train.py – 解耦训练流程
# 包含：
#   DecoupledTrainer → 水动力 / 输沙训练 + Exner 显式河床更新
#   build_boundary_conditions → 边界条件构建

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .physics import MorphodynamicsUpdater


class DecoupledTrainer:
    def __init__(
        self,
        flow_model,
        sediment_model,
        fvm_mesh,
        device,
        flow_loss_fn,
        sediment_transport_loss_fn,
        simulation_time,
        porosity=0.4,
        bed_slope_coefficient=0.2,
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
        self.window_dt = 1.0

        # 水动力、输沙模型优化器
        self.flow_optimizer = torch.optim.Adam(self.flow_model.parameters(), lr=flow_lr)
        self.sediment_optimizer = (
            torch.optim.Adam(self.sediment_model.parameters(), lr=transport_lr)
            if self.sediment_model is not None else None
        )

        # 学习率调度器：根据损失变化自动调整学习率
        self.flow_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.flow_optimizer, 'min', factor=0.5, patience=500)
        self.sediment_scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(self.sediment_optimizer, 'min', factor=0.5, patience=500)
            if self.sediment_optimizer is not None else None
        )

        self.history = {
            'flow_loss': [],    # 水动力损失（SVEs PDE残差 + BC损失）
            'sediment_loss': [],    # 输沙总损失（输沙 PDE 残差 + 容量损失）
            'transport_loss': [],   # 输沙损失（浓度对流扩散）
            'capacity_loss': [],    # 容量损失()
            'initial_sediment_loss': [],
            'inlet_sediment_loss': [],
            'continuity': [],   # 连续性损失
            'momentum_x': [],
            'momentum_y': [],
            'exner_dzb_dt_min': [],
            'exner_dzb_dt_max': [],
            'exchange_dzb_dt_min': [],
            'exchange_dzb_dt_max': [],
            'bed_dt_scale': [],
            'bed_dt_effective': [],
            'bed_delta_max': [],
            'flow_extra_epochs': [],
            'sediment_extra_epochs': [],
            'zb_min': [],
            'zb_max': [],
            'C_min': [],
            'C_max': [],
            'Ceq_mean': [],
            'p_min': [],
            'p_max': [],
        }
        self.morphodynamics = MorphodynamicsUpdater(
            fvm_mesh=self.mesh,
            device=self.device,
            sediment_transport_loss_fn=self.sediment_transport_loss_fn,
            porosity=porosity,
            bed_slope_coefficient=bed_slope_coefficient,
            history=self.history,
        )

    def train_flow_phase(self, n_epochs, T_norm, data_coords=None, data_values=None, data_mask=None):
        time_list = [float(t) for t in np.asarray(T_norm, dtype=np.float32).ravel()]
        self.flow_model.train()
        total_loss = torch.tensor(0.0, device=self.device)
        for epoch in range(n_epochs):
            self.flow_optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=self.device)
            loss_acc = {'continuity': 0.0, 'momentum_x': 0.0, 'momentum_y': 0.0}
            for t_i in time_list:
                # PDE物理损失计算
                physics_loss, loss_dict = self.flow_loss_fn.compute_loss(self.flow_model, t_i, self.device)
                # 物理损失（SVEs PDE残差 + BC损失）
                if callable(data_coords):
                    coords_i, values_i, mask_i = data_coords(t_i)
                else:
                    print("警告: data_coords 不是可调用的边界条件构建器，无法提供训练数据。")
                if coords_i is not None:
                    data_loss = self._compute_flow_data_loss(coords_i, values_i, mask_i)
                    step_loss = physics_loss + 0.5 * data_loss
                else:
                    step_loss = physics_loss
                    print("当前训练步骤仅使用PDE物理损失。")
                # 累积损失并反向传播
                total_loss = total_loss + step_loss / len(time_list)
                for key in loss_acc:
                    loss_acc[key] += loss_dict[key] / len(time_list)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.flow_model.parameters(), max_norm=1.0)
            self.flow_optimizer.step()
            self.flow_scheduler.step(total_loss.detach())
            self.history['flow_loss'].append(total_loss.item())
            self.history['continuity'].append(loss_acc['continuity'])
            self.history['momentum_x'].append(loss_acc['momentum_x'])
            self.history['momentum_y'].append(loss_acc['momentum_y'])
        return total_loss.item()

    def train_sediment_phase(self, n_epochs, T_norm):
        time_list = [float(t) for t in np.asarray(T_norm, dtype=np.float32).ravel()]
        self.flow_model.eval()
        # 将河床活动层级配 p_k 映射到高斯点
        p_k_gauss = self.morphodynamics.gradation_at_gauss_tensor()
        total_loss = torch.tensor(0.0, device=self.device)

        for epoch in range(n_epochs):
            loss_acc = {
                'transport': 0.0,   # 输沙损失（浓度对流扩散）
                'capacity': 0.0,    # 容量损失
                'initial': 0.0, # 初始条件损失
                'inlet': 0.0,   #  入口条件损失
                'C_min': None,  # 预测浓度最小值
                'C_max': None,  # 预测浓度最大值
                'Ceq_mean': 0.0,    # 平衡浓度平均值（衡量整体预测合理性）
            }
            if self.sediment_model is not None and self.sediment_optimizer is not None:
                self.sediment_model.train()

                self.sediment_optimizer.zero_grad()
                total_loss = torch.tensor(0.0, device=self.device)
                for t_i in time_list:
                    sediment_loss, sediment_dict = self.sediment_transport_loss_fn.compute_sediment_loss(
                        self.sediment_model, self.flow_model, t_i, self.device,
                        p_k_override=p_k_gauss,
                    )
                    total_loss = total_loss + sediment_loss / len(time_list)
                    loss_acc['transport'] += sediment_dict['transport'] / len(time_list)
                    loss_acc['capacity'] += sediment_dict['capacity'] / len(time_list)
                    loss_acc['initial'] += sediment_dict['initial'] / len(time_list)
                    loss_acc['inlet'] += sediment_dict['inlet'] / len(time_list)
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

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.sediment_model.parameters(), max_norm=1.0)
                self.sediment_optimizer.step()
                if self.sediment_scheduler is not None:
                    self.sediment_scheduler.step(total_loss.detach())

            self.history['sediment_loss'].append(total_loss.item())
            if loss_acc['C_min'] is not None:
                self.history['transport_loss'].append(loss_acc['transport'])
                self.history['capacity_loss'].append(loss_acc['capacity'])
                self.history['initial_sediment_loss'].append(loss_acc['initial'])
                self.history['inlet_sediment_loss'].append(loss_acc['inlet'])
                self.history['C_min'].append(loss_acc['C_min'])
                self.history['C_max'].append(loss_acc['C_max'])
                self.history['Ceq_mean'].append(loss_acc['Ceq_mean'])
        return total_loss.item()

    def _compute_flow_data_loss(self, coords, values, mask=None):
        coords_tensor = torch.tensor(coords, dtype=torch.float32, device=self.device)
        targets = torch.tensor(values, dtype=torch.float32, device=self.device)
        predictions = self.flow_model(coords_tensor)
        if mask is None:
            return F.mse_loss(predictions, targets)
        mask_tensor = torch.tensor(mask, dtype=torch.float32, device=self.device)
        diff2 = (predictions - targets) ** 2 * mask_tensor
        return diff2.sum() / torch.clamp(mask_tensor.sum(), min=1.0)

    def _time_slices(self, start_time, end_time, sample_dt):
        times = [start_time]
        t = start_time + sample_dt
        eps = max(sample_dt, self.simulation_time) * 1.0e-9
        while t < end_time - eps:
            times.append(t)
            t += sample_dt
        if end_time > start_time + eps:
            times.append(end_time)
        return np.asarray(times, dtype=np.float32) / max(self.simulation_time, 1.0e-12)

    def run_training(
        self,
        simulation_time,
        sample_dt,
        window_dt,
        output_dt,
        flow_epochs_per_window,
        sediment_epochs_per_window,
        bc_builder,
        flow_loss_tol=1e-4,
        sediment_loss_tol=1e-4,
        extra_train_chunk=100,
        max_extra_flow_epochs=0,
        max_extra_sediment_epochs=0,
        max_bed_change_per_step=None,
        verbose=True,
    ):

        self.simulation_time = float(simulation_time)
        self.window_dt = float(window_dt)
        self.history['time'] = []
        self.history['output_times'] = [0.0]

        bed_history = [self.mesh.zb.copy()] # 初始河床状态
        current_time = 0.0
        next_output_time = float(output_dt)
        step_index = 0
        eps = max(simulation_time, self.window_dt, output_dt) * 1.0e-9
        n_windows = int(np.ceil(simulation_time / self.window_dt))   # 窗口数

        with tqdm(total=n_windows) as pbar:
            while current_time < simulation_time - eps:
                # 适应最后一个窗口可能不足 window_dt 的情况
                end_time = min(current_time + self.window_dt, simulation_time)
                actual_window_dt = end_time - current_time
                # 计算当前窗口的训练时间点列表（归一化）
                T_norm_list = self._time_slices(current_time, end_time, sample_dt)
                T_end_norm = float(end_time / simulation_time)

                if verbose and step_index % 10 == 0:
                    print(f'\n  t={current_time:.3f}s→{end_time:.3f}s: '
                        f'窗口时间点数={len(T_norm_list)}')

                # 水动力训练阶段
                flow_loss = self.train_flow_phase(
                    flow_epochs_per_window,
                    T_norm_list,
                    data_coords=bc_builder,
                )
                flow_extra_epochs = 0
                while (
                    flow_loss > flow_loss_tol
                    and flow_extra_epochs < max_extra_flow_epochs
                    and extra_train_chunk > 0
                ):
                    n_extra = min(extra_train_chunk, max_extra_flow_epochs - flow_extra_epochs)
                    flow_loss = self.train_flow_phase(n_extra, T_norm_list, data_coords=bc_builder)
                    flow_extra_epochs += n_extra

                if verbose and step_index % 10 == 0:
                    print(f'  t={end_time:.3f}s: 训练输沙并更新河床...')

                # 泥沙训练阶段
                sediment_loss = self.train_sediment_phase(sediment_epochs_per_window, T_norm_list)
                sediment_extra_epochs = 0
                while (
                    sediment_loss > sediment_loss_tol
                    and sediment_extra_epochs < max_extra_sediment_epochs
                    and extra_train_chunk > 0
                ):
                    n_extra = min(extra_train_chunk, max_extra_sediment_epochs - sediment_extra_epochs)
                    sediment_loss = self.train_sediment_phase(n_extra, T_norm_list)
                    sediment_extra_epochs += n_extra

                self.history['flow_extra_epochs'].append(flow_extra_epochs)
                self.history['sediment_extra_epochs'].append(sediment_extra_epochs)

                # 河床更新
                current_bed, closure, dzb_dt_k = self.morphodynamics.update_bed_explicit(
                    self.sediment_model,
                    self.flow_model,
                    T_end_norm,
                    window_dt=actual_window_dt,
                    max_bed_change_per_step=max_bed_change_per_step,
                )

                # 级配更新
                self.morphodynamics.update_gradation_state(
                    self.sediment_model,
                    self.flow_model,
                    T_end_norm,
                    current_bed,
                    closure=closure,
                    bed_change_rate_k=dzb_dt_k,
                )

                current_time = end_time
                self.history['time'].append(float(current_time))
                self.history['zb_min'].append(float(np.min(current_bed)))
                self.history['zb_max'].append(float(np.max(current_bed)))

                should_output = current_time >= next_output_time - eps or current_time >= simulation_time - eps
                if should_output:
                    bed_history.append(current_bed.copy())
                    self.history['output_times'].append(float(current_time))
                    while next_output_time <= current_time + eps:
                        next_output_time += output_dt

                if verbose and step_index % 10 == 0:
                    print(f'\n  t={current_time:.3f}s: Flow Loss={flow_loss:.2e}, Sed Loss={sediment_loss:.2e}')
                    if flow_extra_epochs or sediment_extra_epochs:
                        print(f'    追加训练: flow={flow_extra_epochs}, sediment={sediment_extra_epochs} epochs')
                    if self.history['bed_dt_scale'][-1] < 1.0:
                        print(
                            f'    Exner限幅: dt_scale={self.history["bed_dt_scale"][-1]:.3f}, '
                            f'max|dzb|={self.history["bed_delta_max"][-1]:.4f}m'
                        )
                    print(f'    河床: [{np.min(current_bed):.4f}, {np.max(current_bed):.4f}]')

                step_index += 1
                pbar.update(1)

        return bed_history
