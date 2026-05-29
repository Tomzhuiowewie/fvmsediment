# train.py – 解耦训练流程
# 包含：
#   DecoupledTrainer → 水动力 / 输沙训练 + Exner 显式河床更新
#   build_boundary_conditions → 边界条件构建

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


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
        self.porosity = porosity
        self.xi = 1.0 / (1.0 - porosity)
        self.rho_bulk = self.sediment_transport_loss_fn.rho_s * (1.0 - porosity)
        self.last_bed = self.mesh.zb.copy()
        self.window_dt = 1.0
        self.window_dt_current = 1.0
        n_grains = len(self.sediment_transport_loss_fn.grain_diameters)
        self.active_layer_frac = np.full((self.mesh.n_cells, n_grains), 1.0 / n_grains, dtype=np.float32)
        self.second_layer_frac = self.active_layer_frac.copy()
        self.delta1 = self._active_layer_thickness_np(self.active_layer_frac)
        self.delta2 = np.full(self.mesh.n_cells, 1.0, dtype=np.float32)

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
            'gradation_loss': [],   # 显式级配更新占位记录
            'capacity_loss': [],    # 容量损失()
            'continuity': [],   # 连续性损失
            'momentum_x': [],
            'momentum_y': [],
            'exner_dzb_dt_min': [],
            'exner_dzb_dt_max': [],
            'bed_dt_scale': [],
            'bed_dt_effective': [],
            'bed_delta_max': [],
            'flow_extra_epochs': [],
            'sediment_extra_epochs': [],
            'zb_min': [],
            'zb_max': [],
            'C_min': [],
            'C_max': [],
            'p_min': [],
            'p_max': [],
        }

    def _resolve_flow_data(self, data_coords, data_values, data_mask, T_norm):
        if callable(data_coords):
            return data_coords(T_norm)
        return data_coords, data_values, data_mask

    def train_flow_phase(self, n_epochs, T_norm, data_coords=None, data_values=None, data_mask=None):
        time_list = [float(t) for t in np.asarray(T_norm, dtype=np.float32).ravel()]
        self.flow_model.train()
        total_loss = torch.tensor(0.0, device=self.device)
        for epoch in range(n_epochs):
            self.flow_optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=self.device)
            loss_acc = {'continuity': 0.0, 'momentum_x': 0.0, 'momentum_y': 0.0}
            for t_i in time_list:
                physics_loss, loss_dict = self.flow_loss_fn.compute_loss(self.flow_model, t_i, self.device)
                coords_i, values_i, mask_i = self._resolve_flow_data(data_coords, data_values, data_mask, t_i)
                if coords_i is not None:
                    data_loss = self._compute_flow_data_loss(coords_i, values_i, mask_i)
                    step_loss = physics_loss + 0.5 * data_loss
                else:
                    step_loss = physics_loss
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
        p_k_gauss = self._gradation_at_gauss_tensor()
        total_loss = torch.tensor(0.0, device=self.device)

        for epoch in range(n_epochs):
            loss_acc = {
                'transport': 0.0,
                'capacity': 0.0,
                'C_min': None,
                'C_max': None,
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
                self.history['C_min'].append(loss_acc['C_min'])
                self.history['C_max'].append(loss_acc['C_max'])
        return total_loss.item()

    def _gradation_at_gauss_tensor(self):
        p = self.active_layer_frac[self.mesh.gauss_cell_id]
        return torch.tensor(p, dtype=torch.float32, device=self.device)

    def _active_layer_thickness_np(self, fractions):
        d = np.asarray(self.sediment_transport_loss_fn.grain_diameters, dtype=np.float32)
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
        delta_m_bed = net_deposition_rate.cpu().numpy() * self.window_dt_current

        rho_bulk = self.rho_bulk
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

    def _exner_dzb_dt_cell(self, T):
        """用总输沙通量边界积分显式计算每个单元的床面变化率。"""
        if self.sediment_model is None:
            return np.zeros(self.mesh.n_cells, dtype=np.float32)

        self.flow_model.eval()
        self.sediment_model.eval()
        p_k_gauss = self._gradation_at_gauss_tensor()

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
        loss_fn = self.sediment_transport_loss_fn
        weights = loss_fn._gauss_weights_t.view(nc, npp, 1)
        nx = loss_fn._gauss_normals_t[:, 0:1]
        ny = loss_fn._gauss_normals_t[:, 1:2]

        h = closure['h']
        u = closure['u']
        v = closure['v']
        C_tk = closure['C_tk']
        epsilon_thk = closure['epsilon_thk']

        adv_flux = h * C_tk * (u * nx + v * ny)
        Cx = loss_fn._physical_grad(C_tk, closure['xyt'], 0)
        Cy = loss_fn._physical_grad(C_tk, closure['xyt'], 1)
        diff_flux = epsilon_thk * h * (Cx * nx + Cy * ny)
        total_sediment_flux = torch.sum(adv_flux - diff_flux, dim=1, keepdim=True)

        boundary_integral = torch.sum(
            total_sediment_flux.view(nc, npp, 1) * weights,
            dim=1,
        ).squeeze(1)
        dzb_dt = -self.xi * boundary_integral / (
            (loss_fn.rho_s + 1.0e-8) * self.mesh.cell_area
        )
        return dzb_dt.detach().cpu().numpy().astype(np.float32)

    def update_bed_explicit(self, T, window_dt=None, max_bed_change_per_step=None):
        if window_dt is None:
            window_dt = self.window_dt
        dzb_dt = self._exner_dzb_dt_cell(T)
        raw_delta_zb = dzb_dt * window_dt
        max_delta = float(np.max(np.abs(raw_delta_zb))) if raw_delta_zb.size else 0.0
        dt_scale = 1.0
        if max_bed_change_per_step is not None and max_bed_change_per_step > 0.0 and max_delta > max_bed_change_per_step:
            dt_scale = max_bed_change_per_step / max(max_delta, 1.0e-12)

        self.window_dt_current = window_dt * dt_scale
        new_bed = self.mesh.zb.astype(np.float32) + dzb_dt * self.window_dt_current
        self.mesh.update_bed(new_bed)
        self.history['exner_dzb_dt_min'].append(float(np.min(dzb_dt)))
        self.history['exner_dzb_dt_max'].append(float(np.max(dzb_dt)))
        self.history['bed_dt_scale'].append(float(dt_scale))
        self.history['bed_dt_effective'].append(float(self.window_dt_current))
        self.history['bed_delta_max'].append(float(max_delta * dt_scale))
        return new_bed

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
        bc_builder=None,
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

        bed_history = [self.mesh.zb.copy()]
        current_time = 0.0
        next_output_time = float(output_dt)
        step_index = 0
        eps = max(simulation_time, window_dt, output_dt) * 1.0e-9
        n_windows = int(np.ceil(simulation_time / window_dt))

        with tqdm(total=n_windows, desc='RAS-like Morph Steps') as pbar:
            while current_time < simulation_time - eps:
                end_time = min(current_time + window_dt, simulation_time)
                actual_window_dt = end_time - current_time
                T_norm_list = self._time_slices(current_time, end_time, sample_dt)
                T_end_norm = float(end_time / simulation_time)

                if verbose and step_index % 10 == 0:
                    print(
                        f'\n  t={current_time:.3f}s→{end_time:.3f}s: '
                        f'窗口时间点数={len(T_norm_list)}'
                    )

                flow_data = bc_builder if bc_builder is not None else None
                flow_loss = self.train_flow_phase(
                    flow_epochs_per_window,
                    T_norm_list,
                    data_coords=flow_data,
                )
                flow_extra_epochs = 0
                # while (
                #     flow_loss > flow_loss_tol
                #     and flow_extra_epochs < max_extra_flow_epochs
                #     and extra_train_chunk > 0
                # ):
                #     n_extra = min(extra_train_chunk, max_extra_flow_epochs - flow_extra_epochs)
                #     flow_loss = self.train_flow_phase(n_extra, T_norm_list, data_coords=flow_data)
                #     flow_extra_epochs += n_extra

                # if verbose and step_index % 10 == 0:
                #     print(f'  t={end_time:.3f}s: 训练输沙并更新河床...')

                sediment_loss = self.train_sediment_phase(sediment_epochs_per_window, T_norm_list)
                sediment_extra_epochs = 0
                # while (
                #     sediment_loss > sediment_loss_tol
                #     and sediment_extra_epochs < max_extra_sediment_epochs
                #     and extra_train_chunk > 0
                # ):
                #     n_extra = min(extra_train_chunk, max_extra_sediment_epochs - sediment_extra_epochs)
                #     sediment_loss = self.train_sediment_phase(n_extra, T_norm_list)
                #     sediment_extra_epochs += n_extra

                self.history['flow_extra_epochs'].append(flow_extra_epochs)
                self.history['sediment_extra_epochs'].append(sediment_extra_epochs)

                current_bed = self.update_bed_explicit(
                    T_end_norm,
                    window_dt=actual_window_dt,
                    max_bed_change_per_step=max_bed_change_per_step,
                )
                self.update_gradation_state(T_end_norm, current_bed)

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

    def run_decoupled_training(
        self,
        n_windows,
        flow_epochs_per_step,
        sediment_epochs_per_step,
        bc_coords=None,
        bc_values=None,
        bc_mask=None,
        flow_loss_tol=1e-4,
        sediment_loss_tol=1e-4,
        extra_train_chunk=100,
        max_extra_flow_epochs=0,
        max_extra_sediment_epochs=0,
        max_bed_change_per_step=None,
        verbose=True,
    ):
        window_dt = self.simulation_time / max(n_windows - 1, 1)

        if bc_coords is None:
            bc_builder = None
        else:
            def bc_builder(_t_norm):
                return bc_coords, bc_values, bc_mask

        return self.run_training(
            simulation_time=self.simulation_time,
            sample_dt=window_dt,
            window_dt=window_dt,
            output_dt=window_dt,
            flow_epochs_per_window=flow_epochs_per_step,
            sediment_epochs_per_window=sediment_epochs_per_step,
            bc_builder=bc_builder,
            flow_loss_tol=flow_loss_tol,
            sediment_loss_tol=sediment_loss_tol,
            extra_train_chunk=extra_train_chunk,
            max_extra_flow_epochs=max_extra_flow_epochs,
            max_extra_sediment_epochs=max_extra_sediment_epochs,
            max_bed_change_per_step=max_bed_change_per_step,
            verbose=verbose,
        )
