# train.py - 三阶段 PINN 训练流程
# 阶段 1：只训练水动力 PINN；阶段 2：冻结水动力训练泥沙 PINN；
# 阶段 3：全时域 DEM-水动力-泥沙固定点耦合与联合优化。

import numpy as np
import os
import sys
import time
import torch
import torch.nn.functional as F

from .data import select_boundary_component
from .model import FlowPINN
from .utils import build_xyt


class _ProgressBar:
    """轻量训练进度条，避免服务器环境额外依赖 tqdm。"""

    def __init__(self, title, total, width=28):
        self.title = title
        self.total = max(int(total), 1)
        self.width = width
        self.current = 0
        self.start = time.time()
        self.last_message_len = 0
        self._render()

    def update(self, step=1, loss=None):
        self.current = min(self.total, self.current + int(step))
        self._render(loss=loss)

    def close(self, loss=None):
        self.current = self.total
        self._render(loss=loss)
        sys.stdout.write('\n')
        sys.stdout.flush()

    def finish(self, loss=None):
        self._render(loss=loss)
        sys.stdout.write('\n')
        sys.stdout.flush()

    def _render(self, loss=None):
        ratio = self.current / self.total
        filled = int(self.width * ratio)
        bar = '#' * filled + '-' * (self.width - filled)
        elapsed = time.time() - self.start
        if self.current == 0:
            remain_text = '--'
        else:
            rate = self.current / max(elapsed, 1.0e-8)
            remain = (self.total - self.current) / max(rate, 1.0e-8)
            remain_text = f'{remain:6.1f}s'
        msg = (
            f"\r{self.title} [{bar}] {self.current}/{self.total} "
            f"{ratio * 100:5.1f}% elapsed={elapsed:6.1f}s eta={remain_text}"
        )
        if loss is not None:
            msg += f" loss={loss:.3e}"
        pad = max(self.last_message_len - len(msg), 0)
        sys.stdout.write(msg + ' ' * pad)
        sys.stdout.flush()
        self.last_message_len = len(msg)


class DecoupledTrainer:
    """三阶段训练控制器。

    这里不直接求解传统数值格式，而是把 PDE、边界条件和床变方程写成 loss：
    1. 先训练 flow_model，使 h/u/v 满足浅水方程和真实边界；
    2. 冻结 flow_model，训练 sediment_model 的 C_k 和 Δzb_k；
    3. 用 sediment_model 预测的分粒径床变更新 DEM/级配，并联合优化两个网络直到耦合收敛。
    """

    # -------------------------------------------------------------------------
    # 初始化、优化器和 history
    # -------------------------------------------------------------------------

    def __init__(
        self,
        flow_model,
        sediment_model,
        fvm_mesh,
        device,
        flow_loss_fn,
        sediment_loss_fn,
        simulation_time,
        initial_gradation=None,
        active_layer_thickness=0.5,
        flow_lr=1e-4,
        transport_lr=1e-4,
        sediment_cell_batch_size=1024,
        flow_loss_tol=0.0,
        sediment_loss_tol=0.0,
        joint_loss_tol=0.0,
        joint_flow_weight=1.0,
        joint_sediment_weight=1.0,
        early_stop_patience=0,
        early_stop_min_delta=0.0,
        coupling_iterations=5,
        coupling_relaxation=0.3,
        coupling_bed_tol=1.0e-5,
        flow_boundary_weight=0.5,
        adaptive_boundary_weighting=True,
        boundary_weight_ema_decay=0.95,
        boundary_weight_min=1.0e-4,
        boundary_weight_max=1.0,
        run_timestamp=None,
        checkpoint_dir=None,
    ):
        self.flow_model = flow_model
        self.sediment_model = sediment_model
        self.mesh = fvm_mesh
        self.device = device
        self.flow_loss_fn = flow_loss_fn
        self.sediment_loss_fn = sediment_loss_fn
        self.simulation_time = simulation_time
        self.active_layer_thickness = float(active_layer_thickness)
        self.sediment_cell_batch_size = int(sediment_cell_batch_size)
        self.flow_loss_tol = float(flow_loss_tol or 0.0)
        self.sediment_loss_tol = float(sediment_loss_tol or 0.0)
        self.joint_loss_tol = float(joint_loss_tol or 0.0)
        self.joint_flow_weight = max(float(joint_flow_weight), 0.0)
        self.joint_sediment_weight = max(float(joint_sediment_weight), 0.0)
        self.early_stop_patience = int(early_stop_patience or 0)
        self.early_stop_min_delta = float(early_stop_min_delta or 0.0)
        self.coupling_iterations = max(int(coupling_iterations), 1)
        self.coupling_relaxation = float(np.clip(coupling_relaxation, 1.0e-6, 1.0))
        self.coupling_bed_tol = max(float(coupling_bed_tol), 0.0)
        self.flow_boundary_weight = max(float(flow_boundary_weight), 0.0)
        self.adaptive_boundary_weighting = bool(adaptive_boundary_weighting)
        self.boundary_weight_ema_decay = float(
            np.clip(boundary_weight_ema_decay, 0.0, 0.9999)
        )
        self.boundary_weight_min = max(float(boundary_weight_min), 0.0)
        self.boundary_weight_max = max(
            float(boundary_weight_max),
            self.boundary_weight_min,
        )
        self._physics_loss_ema = None
        self._boundary_loss_ema = None
        self._current_boundary_weight = self.flow_boundary_weight
        self._best_checkpoint_losses = {}
        self.run_timestamp = str(run_timestamp) if run_timestamp else None
        self.checkpoint_dir = checkpoint_dir
        # active_layer_frac 是每个网格单元的活动层分级比例 p_k。
        # 初始值来自 Excel 主河道级配，阶段 3 按全时域累计分粒径床变更新。
        self.active_layer_frac = self._init_gradation(initial_gradation)
        self.initial_active_layer_frac = self.active_layer_frac.copy()
        # 固壁边界：active 河道单元与 inactive/外部相邻的边。
        # top 入口和 bottom 出口会被排除，不施加 no-flux。
        self.wall_gauss_mask = self._build_wall_gauss_mask()
        self.wall_coords_t = torch.as_tensor(
            self.mesh.gauss_coords[self.wall_gauss_mask],
            dtype=torch.float32,
            device=self.device,
        )
        self.wall_normals_t = torch.as_tensor(
            self.mesh.gauss_normals[self.wall_gauss_mask],
            dtype=torch.float32,
            device=self.device,
        )

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
            'run_timestamp': self.run_timestamp,
            'flow_loss': [],
            'sediment_loss': [],
            'joint_loss': [],
            'transport_loss': [],
            'capacity_loss': [],
            'initial_sediment_loss': [],
            'inlet_sediment_loss': [],
            'bed_change_loss': [],
            'bed_initial_loss': [],
            'weighted_transport_loss': [],
            'weighted_inlet_sediment_loss': [],
            'weighted_bed_change_loss': [],
            'continuity': [],
            'momentum_x': [],
            'momentum_y': [],
            'weighted_continuity': [],
            'weighted_momentum_x': [],
            'weighted_momentum_y': [],
            'weight_continuity': [],
            'weight_momentum_x': [],
            'weight_momentum_y': [],
            'flow_boundary_loss': [],
            'flow_boundary_weight': [],
            'weighted_flow_boundary': [],
            'zb_min': [],
            'zb_max': [],
            'C_min': [],
            'C_max': [],
            'dzb_min': [],
            'dzb_max': [],
            'Ceq_mean': [],
            'p_min': [],
            'p_max': [],
            'coupling_bed_error': [],
            'coupling_dzb_min': [],
            'coupling_dzb_max': [],
            'coupling_joint_loss': [],
            'joint_flow_loss': [],
            'weighted_joint_flow_loss': [],
            'joint_flow_boundary_loss': [],
            'joint_sediment_loss': [],
            'weighted_joint_sediment_loss': [],
            'joint_transport_loss': [],
            'joint_inlet_sediment_loss': [],
            'joint_bed_change_loss': [],
            'joint_bed_initial_loss': [],
            'final_projection_error': [],
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
            'stop_reason': {},
        }

    # -------------------------------------------------------------------------
    # Checkpoint 与通用训练辅助
    # -------------------------------------------------------------------------

    def save_checkpoint(self, name, loss=None, epoch=None, phase=None):
        """保存阶段 checkpoint，长训练中断后至少保留已完成阶段结果。"""
        if not self.checkpoint_dir:
            return
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        suffix = f'_{self.run_timestamp}' if self.run_timestamp else ''
        path = os.path.join(self.checkpoint_dir, f'{name}{suffix}.pt')
        payload = {
            'flow_model': self.flow_model.state_dict(),
            'sediment_model': self.sediment_model.state_dict() if self.sediment_model is not None else None,
            'flow_optimizer': self.flow_optimizer.state_dict(),
            'sediment_optimizer': (
                self.sediment_optimizer.state_dict()
                if self.sediment_optimizer is not None else None
            ),
            'joint_optimizer': (
                self.joint_optimizer.state_dict()
                if self.joint_optimizer is not None else None
            ),
            'active_layer_frac': self.active_layer_frac,
            'mesh_zb': self.mesh.zb,
            'history': self.history,
            'simulation_time': self.simulation_time,
            'run_timestamp': self.run_timestamp,
            'checkpoint_loss': None if loss is None else float(loss),
            'checkpoint_epoch': None if epoch is None else int(epoch),
            'checkpoint_phase': phase,
        }
        torch.save(payload, path)
        print(f"  checkpoint saved: {path}")

    def _save_best_checkpoint(self, name, phase, epoch, loss_value, best_loss):
        """如果当前 loss 刷新阶段最优，则保存 best checkpoint 并返回新的 best_loss。"""
        previous_best = self._best_checkpoint_losses.get(name, best_loss)
        if previous_best is None or float(loss_value) < float(previous_best):
            self._best_checkpoint_losses[name] = float(loss_value)
            self.save_checkpoint(
                name,
                loss=float(loss_value),
                epoch=int(epoch),
                phase=phase,
            )
            return float(loss_value)
        return previous_best

    def _early_stop_check(self, phase, epoch, loss_value, best_loss, stale_epochs, loss_tol):
        """检查单阶段训练是否满足提前停止条件。"""
        if loss_tol > 0.0 and loss_value <= loss_tol:
            return best_loss, stale_epochs, f"{phase}: loss {loss_value:.3e} <= tol {loss_tol:.3e}"

        if best_loss is None or (best_loss - loss_value) > self.early_stop_min_delta:
            return loss_value, 0, None

        stale_epochs += 1
        if self.early_stop_patience > 0 and stale_epochs >= self.early_stop_patience:
            return best_loss, stale_epochs, (
                f"{phase}: no improvement > {self.early_stop_min_delta:.3e} "
                f"for {self.early_stop_patience} epoch(s)"
            )
        return best_loss, stale_epochs, None

    def _required_epoch_count(self, name, value):
        """把配置里的 epoch 数转成非负整数；缺失时给出明确错误。"""
        if value is None:
            raise ValueError(f"training.{name} 不能为空，请在 config.yaml 中设置整数。")
        try:
            return max(int(value), 0)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"training.{name} 必须是整数，当前值为 {value!r}。") from exc

    def _init_gradation(self, initial_gradation):
        """用 Excel 级配初始化活动层；如果未提供，则退化为均匀级配。"""
        n_grains = len(self.sediment_loss_fn.grain_diameters)
        if initial_gradation is None:
            base = np.full(n_grains, 1.0 / max(n_grains, 1), dtype=np.float32)
        else:
            base = np.asarray(initial_gradation, dtype=np.float32)
            if base.size != n_grains:
                raise ValueError("初始级配数量必须与 grain_diameters 一致。")
            base = np.clip(base, 1.0e-8, None)
            base = base / np.sum(base)
        return np.tile(base.reshape(1, -1), (self.mesh.n_cells, 1)).astype(np.float32)

    def _active_cell_batches(self, batch_size):
        """把河道 active 单元切成小批，降低泥沙自动微分显存占用。"""
        active = self.mesh.active_cell_ids
        if batch_size is None or batch_size <= 0 or batch_size >= active.size:
            return [active]
        return [active[i:i + batch_size] for i in range(0, active.size, batch_size)]

    def _gradation_at_gauss_tensor(self):
        """把单元活动层级配 p_k 映射到每个高斯点。"""
        p_gauss = self.active_layer_frac[self.mesh.gauss_cell_id]
        return torch.as_tensor(p_gauss, dtype=torch.float32, device=self.device)

    # -------------------------------------------------------------------------
    # 边界几何：固壁高斯点
    # -------------------------------------------------------------------------

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
        if not np.any(active_2d):
            return np.zeros_like(wall, dtype=bool)

        top_selected = select_boundary_component(active_2d, 'top')
        bottom_selected = select_boundary_component(active_2d, 'bottom')
        top_cell_ids = self.mesh.cell_index[top_selected]
        bottom_cell_ids = self.mesh.cell_index[bottom_selected]

        inlet_mask = (edge_id == 2) & np.isin(cell_id, top_cell_ids)
        outlet_mask = (edge_id == 0) & np.isin(cell_id, bottom_cell_ids)
        return wall & (~inlet_mask) & (~outlet_mask)

    # -------------------------------------------------------------------------
    # 三阶段训练：flow / sediment / joint
    # -------------------------------------------------------------------------

    def train_flow_phase(self, n_epochs, T_norm, bc_builder):
        """阶段 1：训练水动力 PINN。

        T_norm 可以是多个归一化时间点；每个 epoch 会在这些时间点上分别计算
        浅水方程残差和真实边界 loss，再取平均。
        """
        n_epochs = self._required_epoch_count('flow_epochs', n_epochs)
        self.flow_model.train()
        time_list = [float(t) for t in np.asarray(T_norm, dtype=np.float32).ravel()]
        progress = _ProgressBar('Flow PINN', n_epochs * len(time_list))
        
        total_loss_value = 0.0
        best_loss = None
        best_checkpoint_loss = None
        stale_epochs = 0    # 连续若干 epoch loss 没有改善的计数器
        stop_reason = None

        for epoch in range(n_epochs):
            self.flow_optimizer.zero_grad()
            total_loss_value = 0.0
            loss_keys = (
                'continuity',
                'momentum_x',
                'momentum_y',
                'weighted_continuity',
                'weighted_momentum_x',
                'weighted_momentum_y',
                'weight_continuity',
                'weight_momentum_x',
                'weight_momentum_y',
            )
            loss_acc = {key: 0.0 for key in loss_keys}
            boundary_loss_acc = 0.0
            boundary_weight_acc = 0.0
            weighted_boundary_acc = 0.0
            for t_i in time_list:
                # 水动力阶段：浅水方程残差 + 真实边界（入口流量、出口水位）约束。
                physics_loss, loss_dict = self.flow_loss_fn.compute_loss(self.flow_model, t_i, self.device)
                boundary_loss = self._compute_real_flow_boundary_loss(bc_builder(t_i))
                step_loss, boundary_weight = self._balance_flow_boundary_loss(
                    physics_loss,
                    boundary_loss,
                )
                (step_loss / len(time_list)).backward()
                total_loss_value += float(step_loss.detach().cpu()) / len(time_list)
                for key in loss_acc:
                    loss_acc[key] += loss_dict[key] / len(time_list)
                boundary_value = float(boundary_loss.detach().cpu())
                boundary_loss_acc += boundary_value / len(time_list)
                boundary_weight_acc += boundary_weight / len(time_list)
                weighted_boundary_acc += boundary_weight * boundary_value / len(time_list)
                progress.update(loss=total_loss_value)
            torch.nn.utils.clip_grad_norm_(self.flow_model.parameters(), max_norm=1.0)
            self.flow_optimizer.step()
            self.flow_scheduler.step(total_loss_value)
            self.history['flow_loss'].append(total_loss_value)
            for key, value in loss_acc.items():
                self.history[key].append(value)
            self.history['flow_boundary_loss'].append(boundary_loss_acc)
            self.history['flow_boundary_weight'].append(boundary_weight_acc)
            self.history['weighted_flow_boundary'].append(weighted_boundary_acc)
            best_checkpoint_loss = self._save_best_checkpoint(
                'best_phase1_flow',
                'flow',
                epoch,
                total_loss_value,
                best_checkpoint_loss,
            )
            best_loss, stale_epochs, stop_reason = self._early_stop_check(
                'flow', epoch, total_loss_value, best_loss, stale_epochs, self.flow_loss_tol
            )
            if stop_reason is not None:
                self.history['stop_reason']['flow'] = stop_reason
                break
        if stop_reason is None:
            progress.close(loss=total_loss_value)
        else:
            progress.finish(loss=total_loss_value)
            print(f"  Early stop: {stop_reason}")
        return total_loss_value

    def train_sediment_phase(self, n_epochs, T_norm, freeze_flow_params=True):
        """阶段 2：训练泥沙 PINN。

        freeze_flow_params=True 时，水动力场只作为已训练好的背景场提供 h/u/v，
        梯度不会更新 flow_model；泥沙网络学习分粒径浓度 C_k 和分粒径累计床变 Δzb_k。
        """
        n_epochs = self._required_epoch_count('sediment_epochs', n_epochs)
        time_list = [float(t) for t in np.asarray(T_norm, dtype=np.float32).ravel()]
        self.flow_model.eval()
        # 当前活动层级配会影响输沙能力闭合，因此每个训练阶段开始时映射到高斯点。
        p_k_gauss = self._gradation_at_gauss_tensor()
        # 泥沙损失按 cell batch 分摊计算，降低自动微分显存占用；每个 epoch 遍历所有时间点和 cell batch。
        cell_batches = self._active_cell_batches(self.sediment_cell_batch_size)
        
        total_loss_value = 0.0
        progress = _ProgressBar('Sediment PINN', n_epochs * len(time_list) * len(cell_batches))
        best_loss = None
        best_checkpoint_loss = None
        stale_epochs = 0 # 连续若干 epoch loss 没有改善的计数器
        stop_reason = None

        for epoch in range(n_epochs):
            total_loss_value = 0.0
            loss_acc = {
                'transport': 0.0,   # 输沙 PDE 残差
                'capacity': 0.0,    # C≈C_capacity 约束
                'initial': 0.0,   # 初始条件约束
                'inlet': 0.0,   # 入口平衡浓度约束
                'bed_change': 0.0,  # Δzb_k 的 Exner 约束
                'bed_initial': 0.0, # 初始床变约束
                'weight_transport': 0.0,
                'weight_capacity': 0.0,
                'weight_inlet': 0.0,
                'weight_bed_change': 0.0,
                'weighted_transport': 0.0,
                'weighted_inlet': 0.0,
                'weighted_bed_change': 0.0,
                'C_min': None,  # 记录所有 cell batch 的 C_k 最小值，检查是否有负值或数值不稳定。
                'C_max': None,  # 记录所有 cell batch 的 C_k 最大值，检查是否有数值不稳定。
                'dzb_min': None,    # 记录所有 cell batch 的 Δzb_k 最小值，检查是否有过度侵蚀或数值不稳定。
                'dzb_max': None,    # 记录所有 cell batch 的 Δzb_k 最大值，检查是否有过度淤积或数值不稳定。
                'Ceq_mean': 0.0,    # 记录所有 cell batch 的 C_k/C_capacity 平均值，检查是否有普遍过度饱和或过稀释。
            }
            if self.sediment_model is not None and self.sediment_optimizer is not None:
                self.sediment_model.train()

                self.sediment_optimizer.zero_grad()
                for t_i in time_list:
                    for cell_batch in cell_batches:

                        # 输沙 PDE 残差、C≈C_capacity、入口平衡浓度、Δzb_k 的 Exner 约束。
                        sediment_loss, sediment_dict = self.sediment_loss_fn.compute_sediment_loss(
                            self.sediment_model, self.flow_model, t_i, self.device,
                            p_k_override=p_k_gauss,
                            freeze_flow_params=freeze_flow_params,
                            cell_indices=cell_batch,
                        )
                        weight = 1.0 / (len(time_list) * len(cell_batches))
                        
                        (sediment_loss * weight).backward()
                        
                        total_loss_value += float(sediment_loss.detach().cpu()) * weight
                        loss_acc['transport'] += sediment_dict['transport'] * weight
                        # loss_acc['capacity'] += sediment_dict['capacity'] * weight
                        loss_acc['initial'] += sediment_dict['initial'] * weight
                        loss_acc['inlet'] += sediment_dict['inlet'] * weight
                        loss_acc['bed_change'] += sediment_dict['bed_change'] * weight
                        loss_acc['bed_initial'] += sediment_dict['bed_initial'] * weight
                        loss_acc['weight_transport'] += sediment_dict['weight_transport'] * weight
                        loss_acc['weight_capacity'] += sediment_dict['weight_capacity'] * weight
                        loss_acc['weight_inlet'] += sediment_dict['weight_inlet'] * weight
                        loss_acc['weight_bed_change'] += sediment_dict['weight_bed_change'] * weight
                        loss_acc['weighted_transport'] += sediment_dict['weighted_transport'] * weight
                        loss_acc['weighted_inlet'] += sediment_dict['weighted_inlet'] * weight
                        loss_acc['weighted_bed_change'] += sediment_dict['weighted_bed_change'] * weight
                        loss_acc['Ceq_mean'] += sediment_dict['Ceq_mean'] * weight
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
                        progress.update(loss=total_loss_value)

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
                self.history['weighted_transport_loss'].append(loss_acc['weighted_transport'])
                self.history['weighted_inlet_sediment_loss'].append(loss_acc['weighted_inlet'])
                self.history['weighted_bed_change_loss'].append(loss_acc['weighted_bed_change'])
                self.history.setdefault('weight_transport', []).append(loss_acc['weight_transport'])
                self.history.setdefault('weight_capacity', []).append(loss_acc['weight_capacity'])
                self.history.setdefault('weight_inlet', []).append(loss_acc['weight_inlet'])
                self.history.setdefault('weight_bed_change', []).append(loss_acc['weight_bed_change'])
                self.history['C_min'].append(loss_acc['C_min'])
                self.history['C_max'].append(loss_acc['C_max'])
                self.history['dzb_min'].append(loss_acc['dzb_min'])
                self.history['dzb_max'].append(loss_acc['dzb_max'])
                self.history['Ceq_mean'].append(loss_acc['Ceq_mean'])
            best_checkpoint_loss = self._save_best_checkpoint(
                'best_phase2_sediment',
                'sediment',
                epoch,
                total_loss_value,
                best_checkpoint_loss,
            )
            best_loss, stale_epochs, stop_reason = self._early_stop_check(
                'sediment', epoch, total_loss_value, best_loss, stale_epochs, self.sediment_loss_tol
            )
            if stop_reason is not None:
                self.history['stop_reason']['sediment'] = stop_reason
                break
        if stop_reason is None:
            progress.close(loss=total_loss_value)
        else:
            progress.finish(loss=total_loss_value)
            print(f"  Early stop: {stop_reason}")
        return total_loss_value

    def train_joint_phase(self, n_epochs, T_norm, data_coords=None):
        """阶段 3：联合优化水动力和泥沙网络。

        联合阶段不再冻结 flow_model，因此泥沙 loss 中的梯度也可以反馈到 h/u/v。
        这一步用于修正“先水后沙”分阶段训练造成的弱耦合误差。
        """
        n_epochs = self._required_epoch_count('joint_epochs', n_epochs)
        if n_epochs <= 0 or self.sediment_model is None or self.joint_optimizer is None:
            return 0.0

        time_list = [float(t) for t in np.asarray(T_norm, dtype=np.float32).ravel()]
        p_k_gauss = self._gradation_at_gauss_tensor()
        cell_batches = self._active_cell_batches(self.sediment_cell_batch_size)
        total_loss_value = 0.0
        progress = _ProgressBar('Joint PINN', max(n_epochs, 0) * len(time_list) * (1 + len(cell_batches)))
        best_loss = None
        best_checkpoint_loss = None
        stale_epochs = 0
        stop_reason = None

        for epoch in range(n_epochs):
            self.flow_model.train()
            self.sediment_model.train()
            self.joint_optimizer.zero_grad()
            total_loss_value = 0.0
            joint_acc = {
                'flow': 0.0,
                'flow_boundary': 0.0,
                'sediment': 0.0,
                'transport': 0.0,
                'inlet': 0.0,
                'bed_change': 0.0,
                'bed_initial': 0.0,
            }

            for t_i in time_list:
                # 联合 loss = 水动力方程/边界 + 泥沙方程/床变约束。
                flow_physics_loss, _ = self.flow_loss_fn.compute_loss(self.flow_model, t_i, self.device)
                boundary_loss = self._compute_real_flow_boundary_loss(data_coords(t_i))
                flow_loss, _ = self._balance_flow_boundary_loss(flow_physics_loss, boundary_loss)

                # 水动力 loss 已经覆盖全 active 河道；泥沙 loss 按 cell batch 分摊。
                weighted_flow_loss = self.joint_flow_weight * flow_loss
                (weighted_flow_loss / len(time_list)).backward(retain_graph=False)
                total_loss_value += float(weighted_flow_loss.detach().cpu()) / len(time_list)
                joint_acc['flow'] += float(flow_loss.detach().cpu()) / len(time_list)
                joint_acc['flow_boundary'] += float(boundary_loss.detach().cpu()) / len(time_list)
                progress.update(loss=total_loss_value)
                for cell_batch in cell_batches:
                    sediment_loss, sediment_dict = self.sediment_loss_fn.compute_sediment_loss(
                        self.sediment_model,
                        self.flow_model,
                        t_i,
                        self.device,
                        p_k_override=p_k_gauss,
                        freeze_flow_params=False,
                        cell_indices=cell_batch,
                    )
                    weight = 1.0 / (len(time_list) * len(cell_batches))
                    weighted_sediment_loss = self.joint_sediment_weight * sediment_loss
                    (weighted_sediment_loss * weight).backward()
                    total_loss_value += float(weighted_sediment_loss.detach().cpu()) * weight
                    joint_acc['sediment'] += float(sediment_loss.detach().cpu()) * weight
                    joint_acc['transport'] += sediment_dict['transport'] * weight
                    joint_acc['inlet'] += sediment_dict['inlet'] * weight
                    joint_acc['bed_change'] += sediment_dict['bed_change'] * weight
                    joint_acc['bed_initial'] += sediment_dict['bed_initial'] * weight
                    progress.update(loss=total_loss_value)

            torch.nn.utils.clip_grad_norm_(
                list(self.flow_model.parameters()) + list(self.sediment_model.parameters()),
                max_norm=1.0,
            )
            self.joint_optimizer.step()
            if self.joint_scheduler is not None:
                self.joint_scheduler.step(total_loss_value)
            self.history['joint_loss'].append(total_loss_value)
            self.history['joint_flow_loss'].append(joint_acc['flow'])
            self.history['weighted_joint_flow_loss'].append(
                self.joint_flow_weight * joint_acc['flow']
            )
            self.history['joint_flow_boundary_loss'].append(joint_acc['flow_boundary'])
            self.history['joint_sediment_loss'].append(joint_acc['sediment'])
            self.history['weighted_joint_sediment_loss'].append(
                self.joint_sediment_weight * joint_acc['sediment']
            )
            self.history['joint_transport_loss'].append(joint_acc['transport'])
            self.history['joint_inlet_sediment_loss'].append(joint_acc['inlet'])
            self.history['joint_bed_change_loss'].append(joint_acc['bed_change'])
            self.history['joint_bed_initial_loss'].append(joint_acc['bed_initial'])
            best_checkpoint_loss = self._save_best_checkpoint(
                'best_phase3_joint',
                'joint',
                epoch,
                total_loss_value,
                best_checkpoint_loss,
            )
            best_loss, stale_epochs, stop_reason = self._early_stop_check(
                'joint', epoch, total_loss_value, best_loss, stale_epochs, self.joint_loss_tol
            )
            if stop_reason is not None:
                self.history['stop_reason']['joint'] = stop_reason
                break

        if stop_reason is None:
            progress.close(loss=total_loss_value)
        else:
            progress.finish(loss=total_loss_value)
            print(f"  Early stop: {stop_reason}")
        return total_loss_value

    # -------------------------------------------------------------------------
    # 水动力真实边界 loss：入口流量、出口水位
    # -------------------------------------------------------------------------

    def _balance_flow_boundary_loss(self, physics_loss, boundary_loss):
        """使用固定边界权重叠加水动力 PDE 和真实边界损失。"""
        physics_value = max(float(physics_loss.detach().cpu()), 1.0e-12)
        boundary_value = max(float(boundary_loss.detach().cpu()), 1.0e-12)
        self._physics_loss_ema = physics_value
        self._boundary_loss_ema = boundary_value
        weight = self.flow_boundary_weight
        self._current_boundary_weight = weight
        return physics_loss + weight * boundary_loss, weight

    def _compute_real_flow_boundary_loss(self, payload):
        """真实边界损失：入口总流量、出口水位。"""
        inlet_coords = torch.as_tensor(payload['inlet_coords'], dtype=torch.float32, device=self.device)
        inlet_weights = torch.as_tensor(payload['inlet_weights'], dtype=torch.float32, device=self.device).view(-1, 1)
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
        q_target = torch.as_tensor(float(payload['target_flow']), dtype=torch.float32, device=self.device)
        flow_loss = ((q_pred - q_target) / torch.clamp(torch.abs(q_target), min=1.0)) ** 2

        outlet_coords = torch.as_tensor(payload['outlet_coords'], dtype=torch.float32, device=self.device)
        outlet_bed = torch.as_tensor(payload['outlet_bed'], dtype=torch.float32, device=self.device).view(-1, 1)
        outlet_out = self.flow_model(outlet_coords)
        h_out, _, _ = FlowPINN.decode_output(
            outlet_out,
            payload['typical_depth'],
            payload['typical_velocity'],
        )
        stage_target = torch.as_tensor(float(payload['target_stage']), dtype=torch.float32, device=self.device)
        # Excel 给的是绝对水位 stage，模型输出的是水深 h，所以需要 h=stage-zb。
        h_target = torch.clamp(stage_target - outlet_bed, min=float(payload.get('h_min', 0.05)))
        stage_loss = F.mse_loss(h_out, h_target) / max(float(payload['typical_depth']) ** 2, 1.0)
        return flow_loss + stage_loss

    # -------------------------------------------------------------------------
    # 时间序列工具
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # Exner 积分、DEM 更新和活动层级配更新
    # -------------------------------------------------------------------------

    def _exner_rate_field(self, t_norm):
        """计算一个全局时刻所有活动单元的分粒径 Exner 床变率。"""
        self.flow_model.eval()
        self.sediment_model.eval()
        n_grains = len(self.sediment_loss_fn.grain_diameters)
        rate = np.zeros((self.mesh.n_cells, n_grains), dtype=np.float64)

        for cell_batch in self._active_cell_batches(self.sediment_cell_batch_size):
            p_k_gauss = self._gradation_at_gauss_tensor()
            closure = self.sediment_loss_fn.compute_closure(
                self.sediment_model,
                self.flow_model,
                t_norm,
                self.device,
                p_k_override=p_k_gauss,
                freeze_flow_params=True,
                cell_indices=cell_batch,
            )
            rate[cell_batch] = (
                self.sediment_loss_fn.exner_dzb_dt_k_cell(closure)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
        return rate

    def integrate_exner_history(self, times_norm, return_history=False):
        """在全时域上用梯形法积分分粒径 Exner 床变率。

        该函数保留作 Exner 积分诊断。主耦合更新使用 sediment_model
        直接预测的分粒径累计床变 Δzb_k。
        """
        times_norm = np.asarray(times_norm, dtype=np.float64).ravel()
        if times_norm.size == 0:
            raise ValueError("Exner 积分至少需要一个时间点。")
        if np.any(np.diff(times_norm) < 0.0):
            raise ValueError("Exner 积分时间点必须单调递增。")

        n_grains = len(self.sediment_loss_fn.grain_diameters)
        cumulative_k = np.zeros((self.mesh.n_cells, n_grains), dtype=np.float64)
        bed_history = []
        gradation_history = []
        previous_rate = None
        previous_time = None

        for t_norm in times_norm:
            current_rate = self._exner_rate_field(float(t_norm))
            physical_time = float(t_norm) * self.simulation_time
            if previous_rate is not None:
                dt = physical_time - previous_time
                cumulative_k += 0.5 * (previous_rate + current_rate) * dt

            if return_history:
                bed, fractions = self._state_from_cumulative_bed_change(cumulative_k)
                bed_history.append(bed)
                gradation_history.append(fractions)

            previous_rate = current_rate
            previous_time = physical_time

        return cumulative_k, bed_history, gradation_history

    def predict_cumulative_bed_change(self, t_norm):
        """用 SedimentPINN 在单元中心预测分粒径累计床变 Δzb_k。"""
        self.sediment_model.eval()
        n_grains = len(self.sediment_loss_fn.grain_diameters)
        cumulative_k = np.zeros((self.mesh.n_cells, n_grains), dtype=np.float64)
        coords_all = np.stack(
            [self.mesh.cell_centers_x, self.mesh.cell_centers_y],
            axis=1,
        )

        with torch.no_grad():
            for cell_batch in self._active_cell_batches(self.sediment_cell_batch_size):
                coords = torch.tensor(
                    coords_all[cell_batch],
                    dtype=torch.float32,
                    device=self.device,
                )
                xyt = build_xyt(
                    coords,
                    float(t_norm),
                    self.sediment_loss_fn.bounds,
                    self.device,
                    requires_grad=False,
                )
                sediment_out = self.sediment_model(xyt)
                dzb_raw = sediment_out[:, n_grains:]
                if dzb_raw.shape[1] >= n_grains:
                    dzb_k = dzb_raw[:, :n_grains]
                elif dzb_raw.shape[1] == 1:
                    dzb_k = dzb_raw.expand(-1, n_grains) / float(n_grains)
                else:
                    dzb_k = torch.zeros_like(sediment_out[:, :n_grains])
                cumulative_k[cell_batch] = dzb_k.detach().cpu().numpy().astype(np.float64)
        return cumulative_k

    def _state_from_cumulative_bed_change(self, cumulative_k):
        """由全时域累计分粒径床变构造绝对床面和活动层级配。"""
        active = getattr(self.mesh, 'active_cell_mask', np.ones(self.mesh.n_cells, dtype=bool))
        bed = self.mesh.zb_initial.astype(np.float64).copy()
        bed[active] += np.sum(cumulative_k[active], axis=1)

        fractions = self.initial_active_layer_frac.astype(np.float64).copy()
        fractions[active] = np.clip(
            fractions[active]
            + cumulative_k[active] / max(self.active_layer_thickness, 1.0e-6),
            1.0e-8,
            None,
        )
        fractions[active] /= np.sum(fractions[active], axis=1, keepdims=True)
        return bed, fractions.astype(np.float32)

    def apply_coupled_state(self, cumulative_k, relaxation=None):
        """松弛更新当前 DEM 和级配，返回本轮最大床面改变量。"""
        omega = self.coupling_relaxation if relaxation is None else float(relaxation)
        omega = float(np.clip(omega, 1.0e-6, 1.0))
        active = getattr(self.mesh, 'active_cell_mask', np.ones(self.mesh.n_cells, dtype=bool))
        candidate_bed, candidate_frac = self._state_from_cumulative_bed_change(cumulative_k)

        old_bed = self.mesh.zb.copy()
        new_bed = old_bed.copy()
        new_bed[active] = (
            (1.0 - omega) * old_bed[active]
            + omega * candidate_bed[active]
        )

        new_frac = self.active_layer_frac.astype(np.float64).copy()
        new_frac[active] = (
            (1.0 - omega) * new_frac[active]
            + omega * candidate_frac[active]
        )
        new_frac[active] = np.clip(new_frac[active], 1.0e-8, None)
        new_frac[active] /= np.sum(new_frac[active], axis=1, keepdims=True)

        bed_error = float(np.max(np.abs(new_bed[active] - old_bed[active])))
        self.mesh.update_bed(new_bed)
        self.active_layer_frac = new_frac.astype(np.float32)
        total_dzb = np.sum(cumulative_k[active], axis=1)
        self.history['coupling_bed_error'].append(bed_error)
        self.history['coupling_dzb_min'].append(float(np.min(total_dzb)))
        self.history['coupling_dzb_max'].append(float(np.max(total_dzb)))
        self.history['p_min'].append(float(np.min(self.active_layer_frac[active])))
        self.history['p_max'].append(float(np.max(self.active_layer_frac[active])))
        return bed_error

    @staticmethod
    def _joint_epoch_schedule(total_epochs, n_iterations):
        """把联合训练总 epoch 数尽量均匀地分配到各耦合迭代。"""
        if n_iterations <= 0:
            return []
        total_epochs = max(int(total_epochs), 0)
        base, remainder = divmod(total_epochs, n_iterations)
        return [base + (1 if i < remainder else 0) for i in range(n_iterations)]

    # -------------------------------------------------------------------------
    # Diagnostics：边界、场变量、固壁法向速度
    # -------------------------------------------------------------------------

    def record_diagnostics(
        self,
        output_times,
        bc_builder,
        bed_history=None,
        gradation_history=None,
    ):
        """在输出时刻记录物理诊断量，便于判断结果是否可信。"""
        active_cell = self.mesh.active_cell_mask
        active_gauss = active_cell[self.mesh.gauss_cell_id]
        active_gauss_t = torch.as_tensor(active_gauss, dtype=torch.bool, device=self.device)
        coords = torch.as_tensor(self.mesh.gauss_coords, dtype=torch.float32, device=self.device)

        self.flow_model.eval()
        if self.sediment_model is not None:
            self.sediment_model.eval()

        final_bed = self.mesh.zb.copy()
        final_gradation = self.active_layer_frac.copy()
        for i, t in enumerate(output_times):
            if bed_history is not None:
                self.mesh.update_bed(bed_history[i])
            if gradation_history is not None:
                self.active_layer_frac = gradation_history[i].copy()
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
        self.mesh.update_bed(final_bed)
        self.active_layer_frac = final_gradation

    def _flow_boundary_diagnostics(self, payload):
        inlet_coords = torch.as_tensor(payload['inlet_coords'], dtype=torch.float32, device=self.device)
        inlet_weights = torch.as_tensor(payload['inlet_weights'], dtype=torch.float32, device=self.device).view(-1, 1)
        outlet_coords = torch.as_tensor(payload['outlet_coords'], dtype=torch.float32, device=self.device)
        outlet_bed = torch.as_tensor(payload['outlet_bed'], dtype=torch.float32, device=self.device).view(-1, 1)

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
            'p_min_diag': float(np.min(self.active_layer_frac[self.mesh.active_cell_mask])),
            'p_max_diag': float(np.max(self.active_layer_frac[self.mesh.active_cell_mask])),
            'wall_un_abs_mean': self._wall_un_abs_mean(t_norm),
        }

        if self.sediment_model is None:
            return diag

        c_sum = 0.0
        ceq_sum = 0.0
        n_val = 0
        dzb_min = None
        dzb_max = None
        # 这里使用 compute_closure，是为了同时拿到 C、C_capacity 和 Δzb；
        # 按 cell batch 统计，避免诊断阶段再次占满显存。
        p_k_gauss = self._gradation_at_gauss_tensor()
        for cell_batch in self._active_cell_batches(self.sediment_cell_batch_size):
            closure = self.sediment_loss_fn.compute_closure(
                self.sediment_model,
                self.flow_model,
                t_norm,
                self.device,
                p_k_override=p_k_gauss,
                freeze_flow_params=True,
                cell_indices=cell_batch,
            )
            C_batch = closure['C_tk']
            Ceq_batch = closure['C_capacity']
            dzb_batch = closure['dzb_pred']
            n_batch = int(C_batch.numel())
            c_sum += float(torch.sum(C_batch).detach().cpu())
            ceq_sum += float(torch.sum(Ceq_batch).detach().cpu())
            n_val += n_batch
            dzb_min_batch = float(torch.min(dzb_batch).detach().cpu())
            dzb_max_batch = float(torch.max(dzb_batch).detach().cpu())
            dzb_min = dzb_min_batch if dzb_min is None else min(dzb_min, dzb_min_batch)
            dzb_max = dzb_max_batch if dzb_max is None else max(dzb_max, dzb_max_batch)
        diag.update({
            'C_mean_diag': c_sum / max(n_val, 1),
            'Ceq_mean_diag': ceq_sum / max(n_val, 1),
            'dzb_min_diag': float(dzb_min or 0.0),
            'dzb_max_diag': float(dzb_max or 0.0),
        })
        return diag

    def _wall_un_abs_mean(self, t_norm):
        """诊断固壁边界平均法向速度绝对值。"""
        if self.wall_coords_t.numel() == 0:
            return 0.0
        xyt = build_xyt(
            self.wall_coords_t,
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
            un = u_wall * self.wall_normals_t[:, 0:1] + v_wall * self.wall_normals_t[:, 1:2]
        return float(torch.mean(torch.abs(un)).detach().cpu())

    # -------------------------------------------------------------------------
    # 完整训练编排
    # -------------------------------------------------------------------------

    def run_training(
        self,
        simulation_time,
        sample_dt,
        output_dt,
        flow_epochs,
        sediment_epochs,
        bc_builder,
        joint_epochs=0,
        verbose=True,
    ):
        """执行完整训练流程。

        sample_dt 控制全时域训练时间点；
        output_dt 控制直接预测床面历史和图像输出时间。
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
            print('\n阶段 2/3: 冻结水动力，训练泥沙 PINN 和分粒径累计床变 Δzb_k')
        self.save_checkpoint('phase1_flow')

        sediment_loss = self.train_sediment_phase(
            sediment_epochs,
            T_norm_list,
            freeze_flow_params=True,
        )

        if verbose:
            print(f'阶段 2 完成: Sediment Loss={sediment_loss:.2e}')
            print('\n阶段 3/3: 联合优化水动力 PINN 和泥沙 PINN')
        self.save_checkpoint('phase2_sediment')

        # 阶段 3：不划分形态窗口。每轮先由当前泥沙 PINN 在最终时刻预测
        # 分粒径累计床变并更新 DEM/级配，再在更新后的 DEM 上联合优化两个网络。
        joint_epochs = int(getattr(self, 'joint_epochs', 0))
        epoch_schedule = self._joint_epoch_schedule(
            joint_epochs,
            self.coupling_iterations,
        )
        joint_loss = 0.0
        completed_iterations = 0
        for iteration, iteration_epochs in enumerate(epoch_schedule, start=1):
            cumulative_k = self.predict_cumulative_bed_change(1.0)
            bed_error = self.apply_coupled_state(cumulative_k)
            if verbose:
                print(
                    f'\n耦合迭代 {iteration}/{len(epoch_schedule)}: '
                    f'bed_error={bed_error:.3e}m, joint_epochs={iteration_epochs}'
                )
            if iteration_epochs > 0:
                joint_loss = self.train_joint_phase(
                    iteration_epochs,
                    T_norm_list,
                    data_coords=bc_builder,
                )
            self.history['coupling_joint_loss'].append(float(joint_loss))
            completed_iterations = iteration
            if self.coupling_bed_tol > 0.0 and bed_error <= self.coupling_bed_tol:
                self.history['stop_reason']['coupling'] = (
                    f"coupling: bed error {bed_error:.3e} <= "
                    f"tol {self.coupling_bed_tol:.3e}"
                )
                if verbose:
                    print(f"  耦合收敛: {self.history['stop_reason']['coupling']}")
                break
        self.save_checkpoint('phase3_joint')

        # 用最终联合模型直接预测各输出时刻的累计床变，生成正式床面和级配历史。
        output_times = self._output_times(simulation_time, output_dt)
        bed_history = []
        gradation_history = []
        for output_time in output_times:
            t_norm = float(output_time) / max(self.simulation_time, 1.0e-12)
            cumulative_k = self.predict_cumulative_bed_change(t_norm)
            bed, fractions = self._state_from_cumulative_bed_change(cumulative_k)
            bed_history.append(bed)
            gradation_history.append(fractions)
        active = getattr(self.mesh, 'active_cell_mask', np.ones(self.mesh.n_cells, dtype=bool))
        projection_error = float(np.max(np.abs(
            bed_history[-1][active] - self.mesh.zb[active]
        )))
        self.history['final_projection_error'].append(projection_error)
        self.mesh.update_bed(bed_history[-1])
        self.active_layer_frac = gradation_history[-1].copy()
        self.record_diagnostics(
            output_times,
            bc_builder,
            bed_history=bed_history,
            gradation_history=gradation_history,
        )
        self.history['integration_times'] = output_times
        self.history['coupling_iterations_completed'] = completed_iterations
        self.history['output_times'] = output_times
        self.history['zb_min'] = [float(np.min(zb)) for zb in bed_history]
        self.history['zb_max'] = [float(np.max(zb)) for zb in bed_history]

        if verbose:
            print(f'阶段 3 完成: Joint Loss={joint_loss:.2e}')
            print(f'完成耦合迭代: {completed_iterations}')
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
