
import json
import os
import shutil

import numpy as np
import torch

from src.config import load_config
from src.data import (
    FVMeshPreprocessor,
    RealBoundaryConditionBuilder,
    load_real_case_data,
)
from src.evaluate import visualize_results
from src.model import FlowPINN, SedimentPINN
from src.physics import SVEsPhysicsLoss, SedimentTransportLoss
from src.train import DecoupledTrainer


def run_real_case(config_path="config.yaml"):
    """
    流程：加载真实配置 → 构建 DEM/FVM 网格 → 创建模型 → 三阶段训练 → 可视化结果。

    这里的真实数据只作为物理输入条件：
    - DEM 给初始床面 zb 和河道有效单元 mask；
    - Excel 给上游流量、下游水位和床沙级配；

    """
    # 1. 加载配置 
    cfg = load_config(config_path)
    output_dir = cfg.training.get('output_dir', 'outputs')
    checkpoint_dir = cfg.training.get('checkpoint_dir', os.path.join(output_dir, 'checkpoints'))
    os.makedirs(output_dir, exist_ok=True)

    # 2. 读取真实 DEM、边界过程线和床沙级配；网格范围和分辨率由 DEM 决定。
    real_case = load_real_case_data(cfg.data)

    # 粒径组可以由 config.yaml 显式指定；若留空，则直接使用 Excel 中 MainChannel 级配。
    grain_diameters = cfg.grain_diameters
    if not grain_diameters:
        grain_diameters = real_case.grain_diameters
    print(
        f"DEM: {real_case.bed_grid.shape[1]}x{real_case.bed_grid.shape[0]}, "
        f"resolution={real_case.resolution:.3f}m, active_cells={int(real_case.active_mask.sum())}"
    )
    print(
        f"Hydrograph: Q={len(real_case.flow_times)} points, "
        f"stage={len(real_case.stage_times)} points, "
        f"gradation={len(real_case.grain_diameters)} classes"
    )
    print(
        "Training settings: "
        f"flow_epochs={cfg.training['flow_epochs']}, "
        f"sediment_epochs={cfg.training['sediment_epochs']}, "
        f"joint_epochs={cfg.training['joint_epochs']}, "
        f"sediment_batch={cfg.training.get('sediment_cell_batch_size', 1024)}, "
        f"output_dir={output_dir}"
    )

    mesh = FVMeshPreprocessor(
        real_case.bbox,
        real_case.resolution,
        initial_bed=real_case.bed_grid,
        n_gauss_points=cfg.n_gauss_points,
        # active_mask=True 的单元参与 PDE、诊断和绘图；非河道单元只保留 DEM 背景。
        active_mask=real_case.active_mask,
    )

    # 3. 设备选择 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 4. 创建水动力与输沙 PINN 模型。
    #    flow_model 输出归一化的 h、u、v；
    #    sediment_model 前 K 个输出为分粒径浓度 C_k，最后一个输出为累计床变 Δzb。
    flow_model = FlowPINN(input_dim=3, hidden_dim=64, num_block=4, output_dim=3).to(device)

    n_grains = len(grain_diameters)
    initial_concentration = _match_concentration(cfg.initial_sediment_concentration, n_grains)
    sediment_model = SedimentPINN(
        input_dim=3,
        hidden_dim=64,
        num_block=4,
        output_dim=n_grains + 1,
        n_concentration_outputs=n_grains,
        initial_concentration=initial_concentration,
        bed_change_scale=cfg.bed_change_scale,
    ).to(device)

    # 5. 创建物理损失函数。
    #    水动力损失：非恒定浅水方程 FVM 残差；
    #    泥沙损失：分粒径输沙方程、输沙能力闭合、入口平衡浓度和 Exner 床变约束。
    flow_loss_fn = SVEsPhysicsLoss(
        fvm_mesh=mesh,
        g=cfg.g,
        n_manning=cfg.n_manning,
        bounds=real_case.bounds,
        typical_depth=cfg.typical_depth,
        typical_velocity=cfg.typical_velocity,
        simulation_time=cfg.simulation_time,
        include_time_terms=cfg.include_time_terms,
    )

    sediment_transport_loss_fn = SedimentTransportLoss(
        fvm_mesh=mesh,
        bounds=real_case.bounds,
        include_time_terms=cfg.include_time_terms,
        grain_diameters=grain_diameters,
        beta_default=cfg.beta_default,
        epsilon_default=cfg.epsilon_default,
        residual_scale=cfg.sediment_residual_scale,
        adaptation_length=cfg.adaptation_length,
        rho_s=cfg.rho_s,
        rho_w=cfg.rho_w,
        g=cfg.g,
        n_manning=cfg.n_manning,
        kinematic_viscosity=cfg.kinematic_viscosity,
        wu_theta_cr=cfg.wu_theta_cr,
        skin_shear_factor=cfg.skin_shear_factor,
        alpha_active_layer=cfg.alpha_active_layer,
        w_capacity=cfg.w_capacity,
        w_initial_sediment=cfg.w_initial_sediment,
        initial_sediment_concentration=initial_concentration,
        w_inlet_sediment=cfg.w_inlet_sediment,
        w_bed_change=cfg.training.get('w_bed_change', 1.0),
        porosity=cfg.porosity,
        bed_slope_coefficient=cfg.bed_slope_coefficient,
        bed_slope_diffusion_weight=cfg.bed_slope_diffusion_weight,
        exchange_weight=cfg.exchange_weight,
        source_sharpness=cfg.source_sharpness,
        simulation_time=cfg.simulation_time,
        typical_depth=cfg.typical_depth,
        typical_velocity=cfg.typical_velocity,
    )

    # 6. 创建三阶段训练器。
    #    initial_gradation 是活动层初始级配，来自 Excel 的 MainChannel 累计级配换算。
    trainer = DecoupledTrainer(
        fvm_mesh=mesh,
        device=device,
        flow_model=flow_model,
        sediment_model=sediment_model,
        flow_loss_fn=flow_loss_fn,
        sediment_transport_loss_fn=sediment_transport_loss_fn,
        simulation_time=cfg.simulation_time,
        initial_gradation=real_case.grain_fractions,
        active_layer_thickness=cfg.active_layer_thickness,
        flow_lr=cfg.training.get('flow_lr', 1e-4),
        transport_lr=cfg.training.get('transport_lr', 1e-4),
        sediment_cell_batch_size=cfg.training.get('sediment_cell_batch_size', 1024),
        flow_loss_tol=cfg.training.get('flow_loss_tol', 0.0),
        sediment_loss_tol=cfg.training.get('sediment_loss_tol', 0.0),
        joint_loss_tol=cfg.training.get('joint_loss_tol', 0.0),
        early_stop_patience=cfg.training.get('early_stop_patience', 0),
        early_stop_min_delta=cfg.training.get('early_stop_min_delta', 0.0),
        checkpoint_dir=checkpoint_dir,
    )

    # 7. 真实边界条件：
    #    上边界不是固定速度，而是约束断面流量积分 ∫h v_n dS = Q(t)；
    #    下边界不是固定水深，而是用 stage(t)-zb 转换为目标水深。
    bc_builder = RealBoundaryConditionBuilder(
        mesh=mesh,
        real_case=real_case,
        typical_depth=cfg.typical_depth,
        typical_velocity=cfg.typical_velocity,
        simulation_time=cfg.simulation_time,
    )
    # 8. 运行三阶段训练：
    #    阶段 1 只训练水动力 PINN；
    #    阶段 2 冻结水动力，训练泥沙浓度和床变；
    #    阶段 3 水动力和泥沙网络一起微调。
    print("开始真实 DEM 泥沙演变训练...")

    bed_history = trainer.run_training(
        simulation_time=cfg.simulation_time,
        sample_dt=cfg.sample_dt,
        morph_dt=cfg.morph_dt,
        output_dt=cfg.output_dt,
        flow_epochs=cfg.training['flow_epochs'],
        sediment_epochs=cfg.training['sediment_epochs'],
        bc_builder=bc_builder,
        joint_epochs=cfg.training['joint_epochs'],
    )

    # 9. 保存正式训练产物：模型、床面历史、级配、history 和配置快照。
    save_training_outputs(
        output_dir=output_dir,
        config_path=config_path,
        flow_model=flow_model,
        sediment_model=sediment_model,
        trainer=trainer,
        bed_history=bed_history,
    )

    # 10. 可视化结果 
    visualize_results(
        mesh=mesh,
        bed_history=bed_history,
        bbox=real_case.bbox,
        resolution=real_case.resolution,
        history=trainer.history,
        simulation_time=cfg.simulation_time,
        case_name='real',
        output_dir=output_dir,
    )

    return bed_history, trainer.history


def save_training_outputs(output_dir, config_path, flow_model, sediment_model, trainer, bed_history):
    """保存正式训练结果，便于后处理、复现实验和中断恢复。"""
    os.makedirs(output_dir, exist_ok=True)
    torch.save(flow_model.state_dict(), os.path.join(output_dir, 'flow_model.pt'))
    torch.save(sediment_model.state_dict(), os.path.join(output_dir, 'sediment_model.pt'))
    torch.save(
        {
            'flow_model': flow_model.state_dict(),
            'sediment_model': sediment_model.state_dict(),
            'active_layer_frac': trainer.active_layer_frac,
            'history': trainer.history,
            'simulation_time': trainer.simulation_time,
        },
        os.path.join(output_dir, 'final_checkpoint.pt'),
    )
    np.savez_compressed(
        os.path.join(output_dir, 'training_results.npz'),
        bed_history=np.asarray(bed_history, dtype=np.float32),
        active_layer_frac=np.asarray(trainer.active_layer_frac, dtype=np.float32),
        output_times=np.asarray(trainer.history.get('output_times', []), dtype=np.float32),
        morph_times=np.asarray(trainer.history.get('morph_times', []), dtype=np.float32),
    )
    with open(os.path.join(output_dir, 'history.json'), 'w', encoding='utf-8') as f:
        json.dump(_json_safe(trainer.history), f, ensure_ascii=False, indent=2)
    if config_path and os.path.exists(config_path):
        shutil.copyfile(config_path, os.path.join(output_dir, 'config_used.yaml'))
    print(f"训练结果已保存到: {output_dir}")


def _json_safe(value):
    """把 NumPy/PyTorch 标量递归转成 JSON 可写对象。"""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def _match_concentration(values, n_grains):
    if len(values) == n_grains:
        return values
    if len(values) == 1:
        return values * n_grains
    return [0.0] * n_grains


if __name__ == '__main__':

    config_path = "config.yaml"
    run_real_case(config_path=config_path)
