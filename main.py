import os
from datetime import datetime

import torch

from src.config import load_config
from src.data import (
    FVMeshPreprocessor,
    RealBoundaryConditionBuilder,
    load_real_case_data,
)
from src.model import FlowPINN, SedimentPINN
from src.physics import SVEsPhysicsLoss, SedimentPhysicsLoss
from src.plot import plot_dem_overview, visualize_results
from src.train import DecoupledTrainer
from src.utils import match_concentration, save_training_outputs


def run_real_case(config_path="config.yaml"):
    """
    流程：加载真实配置 → 构建 DEM/FVM 网格 → 创建模型 → 三阶段训练 → 可视化结果。

    这里的真实数据只作为物理输入条件：
    - DEM 给初始床面 zb 和河道有效单元 mask；
    - Excel 给上游流量、下游水位和床沙级配；

    """
    # 1. 加载配置 
    cfg = load_config(config_path)
    time_cfg = cfg.time
    flow_cfg = cfg.flow
    sediment_cfg = cfg.sediment
    morph_cfg = cfg.morphodynamics
    training_cfg = cfg.training
    run_timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    output_dir = training_cfg.get('output_dir', 'outputs')
    checkpoint_dir = training_cfg.get('checkpoint_dir', os.path.join(output_dir, 'checkpoints'))
    os.makedirs(output_dir, exist_ok=True)

    # 2. 读取真实 DEM、边界过程线和床沙级配；网格范围和分辨率由 DEM 决定。
    real_case = load_real_case_data(cfg.data)

    # 粒径组可以由 config.yaml 显式指定；若留空，则直接使用 Excel 中 MainChannel 级配。
    grain_diameters = sediment_cfg.grain_diameters
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
        f"flow_epochs={training_cfg['flow_epochs']}, "
        f"sediment_epochs={training_cfg['sediment_epochs']}, "
        f"joint_epochs={training_cfg['joint_epochs']}, "
        f"coupling_iterations={training_cfg.get('coupling_iterations', 5)}, "
        f"coupling_relaxation={training_cfg.get('coupling_relaxation', 0.3)}, "
        f"coupling_bed_tol={training_cfg.get('coupling_bed_tol', 1.0e-5)}, "
        f"sediment_batch={training_cfg.get('sediment_cell_batch_size', 1024)}, "
        f"output_dir={output_dir}"
    )
    plot_dem_overview(
        bed_grid=real_case.bed_grid,
        active_mask=real_case.active_mask,
        bbox=real_case.bbox,
        save_path=os.path.join(output_dir, f'real_dem_mask_{run_timestamp}.png'),
        title='Initial DEM and Active Mask',
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
    #    sediment_model 前 K 个输出为分粒径浓度 C_k，后 K 个输出为分粒径累计床变 Δzb_k。
    flow_model = FlowPINN(input_dim=3, hidden_dim=64, num_block=4, output_dim=3).to(device)

    n_grains = len(grain_diameters)
    initial_concentration = match_concentration(sediment_cfg.initial_concentration, n_grains)
    sediment_model = SedimentPINN(
        input_dim=3,
        hidden_dim=64,
        num_block=4,
        output_dim=2 * n_grains,
        n_concentration_outputs=n_grains,
        initial_concentration=initial_concentration,
        bed_change_scale=morph_cfg.bed_change_scale,
    ).to(device)

    # 5. 创建物理损失函数。
    #    水动力损失：非恒定浅水方程 FVM 残差；
    #    泥沙损失：分粒径输沙方程、输沙能力闭合、入口平衡浓度和 Exner 床变约束。
    flow_loss_fn = SVEsPhysicsLoss(
        fvm_mesh=mesh,
        g=flow_cfg.g,
        n_manning=flow_cfg.n_manning,
        bounds=real_case.bounds,
        typical_depth=flow_cfg.typical_depth,
        typical_velocity=flow_cfg.typical_velocity,
        simulation_time=time_cfg.simulation_time,
        include_time_terms=time_cfg.include_time_terms,
        adaptive_weighting=flow_cfg.adaptive_loss_weighting,
        adaptive_weight_ema_decay=flow_cfg.adaptive_weight_ema_decay,
        adaptive_weight_min=flow_cfg.adaptive_weight_min,
        adaptive_weight_max=flow_cfg.adaptive_weight_max,
    )

    sediment_loss_fn = SedimentPhysicsLoss(
        fvm_mesh=mesh,
        bounds=real_case.bounds,
        include_time_terms=time_cfg.include_time_terms,
        grain_diameters=grain_diameters,
        beta_default=sediment_cfg.beta_default,
        epsilon_default=sediment_cfg.epsilon_default,
        residual_scale=sediment_cfg.residual_scale,
        adaptation_length=sediment_cfg.adaptation_length,
        rho_s=sediment_cfg.rho_s,
        rho_w=sediment_cfg.rho_w,
        g=flow_cfg.g,
        n_manning=flow_cfg.n_manning,
        kinematic_viscosity=sediment_cfg.kinematic_viscosity,
        wu_theta_cr=sediment_cfg.wu_theta_cr,
        skin_shear_factor=sediment_cfg.skin_shear_factor,
        alpha_active_layer=sediment_cfg.alpha_active_layer,
        w_capacity=sediment_cfg.w_capacity,
        w_initial_sediment=sediment_cfg.w_initial_sediment,
        initial_sediment_concentration=initial_concentration,
        w_inlet_sediment=sediment_cfg.w_inlet_sediment,
        w_bed_change=training_cfg.get('w_bed_change', 1.0),
        porosity=morph_cfg.porosity,
        bed_slope_coefficient=morph_cfg.bed_slope_coefficient,
        bed_slope_diffusion_weight=morph_cfg.bed_slope_diffusion_weight,
        exchange_weight=morph_cfg.exchange_weight,
        source_sharpness=sediment_cfg.source_sharpness,
        simulation_time=time_cfg.simulation_time,
        typical_depth=flow_cfg.typical_depth,
        typical_velocity=flow_cfg.typical_velocity,
    )

    # 6. 创建三阶段训练器。
    #    initial_gradation 是活动层初始级配，来自 Excel 的 MainChannel 累计级配换算。
    trainer = DecoupledTrainer(
        fvm_mesh=mesh,
        device=device,
        flow_model=flow_model,
        sediment_model=sediment_model,
        flow_loss_fn=flow_loss_fn,
        sediment_loss_fn=sediment_loss_fn,
        simulation_time=time_cfg.simulation_time,
        initial_gradation=real_case.grain_fractions,
        active_layer_thickness=morph_cfg.active_layer_thickness,
        flow_lr=training_cfg.get('flow_lr', 1e-4),
        transport_lr=training_cfg.get('transport_lr', 1e-4),
        sediment_cell_batch_size=training_cfg.get('sediment_cell_batch_size', 1024),
        flow_loss_tol=training_cfg.get('flow_loss_tol', 0.0),
        sediment_loss_tol=training_cfg.get('sediment_loss_tol', 0.0),
        joint_loss_tol=training_cfg.get('joint_loss_tol', 0.0),
        early_stop_patience=training_cfg.get('early_stop_patience', 0),
        early_stop_min_delta=training_cfg.get('early_stop_min_delta', 0.0),
        coupling_iterations=training_cfg.get('coupling_iterations', 5),
        coupling_relaxation=training_cfg.get('coupling_relaxation', 0.3),
        coupling_bed_tol=training_cfg.get('coupling_bed_tol', 1.0e-5),
        flow_boundary_weight=training_cfg.get('flow_boundary_weight', 0.5),
        adaptive_boundary_weighting=training_cfg.get('adaptive_boundary_weighting', True),
        boundary_weight_ema_decay=training_cfg.get('boundary_weight_ema_decay', 0.95),
        boundary_weight_min=training_cfg.get('boundary_weight_min', 1.0e-4),
        boundary_weight_max=training_cfg.get('boundary_weight_max', 1.0),
        run_timestamp=run_timestamp,
        checkpoint_dir=checkpoint_dir,
    )

    # 7. 真实边界条件：
    #    上边界不是固定速度，而是约束断面流量积分 ∫h v_n dS = Q(t)；
    #    下边界不是固定水深，而是用 stage(t)-zb 转换为目标水深。
    bc_builder = RealBoundaryConditionBuilder(
        mesh=mesh,
        real_case=real_case,
        typical_depth=flow_cfg.typical_depth,
        typical_velocity=flow_cfg.typical_velocity,
        simulation_time=time_cfg.simulation_time,
    )
    # 8. 运行三阶段训练：
    #    阶段 1 只训练水动力 PINN；
    #    阶段 2 冻结水动力，训练泥沙浓度和床变；
    #    阶段 3 做全时域固定点耦合：泥沙模型先预测累计床变并更新 DEM，
    #    再在新 DEM 上联合优化水动力和泥沙，重复直到床面收敛或达到迭代上限。
    print("开始真实 DEM 泥沙演变训练...")

    bed_history = trainer.run_training(
        simulation_time=time_cfg.simulation_time,
        sample_dt=time_cfg.sample_dt,
        output_dt=time_cfg.output_dt,
        flow_epochs=training_cfg['flow_epochs'],
        sediment_epochs=training_cfg['sediment_epochs'],
        bc_builder=bc_builder,
        joint_epochs=training_cfg['joint_epochs'],
    )

    # 9. 保存正式训练产物：模型、床面历史、级配、history 和配置快照。
    save_training_outputs(
        output_dir=output_dir,
        config_path=config_path,
        flow_model=flow_model,
        sediment_model=sediment_model,
        trainer=trainer,
        bed_history=bed_history,
        run_timestamp=run_timestamp,
    )

    # 10. 可视化结果 
    visualize_results(
        mesh=mesh,
        bed_history=bed_history,
        bbox=real_case.bbox,
        resolution=real_case.resolution,
        history=trainer.history,
        simulation_time=time_cfg.simulation_time,
        case_name='real',
        output_dir=output_dir,
        run_timestamp=run_timestamp,
    )

    return bed_history, trainer.history

if __name__ == '__main__':

    config_path = "config.yaml"
    run_real_case(config_path=config_path)
