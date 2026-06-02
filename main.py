
import torch

from src.config import load_config
from src.data import FVMeshPreprocessor, hump_initial_bed, build_boundary_conditions
from src.evaluate import visualize_results
from src.model import FlowPINN, SedimentPINN
from src.physics import SVEsPhysicsLoss, SedimentTransportLoss
from src.train import DecoupledTrainer


def run_hump_evolution_test(config_path="config.yaml"):
    """
    流程：加载配置 → 构建网格 → 创建模型 → 解耦训练 → 可视化结果。
    """
    # 1. 加载配置 
    cfg = load_config(config_path)

    # 2. 初始床面的 FVM 网格
    mesh = FVMeshPreprocessor(
        cfg.bbox, 
        cfg.resolution,
        initial_bed=hump_initial_bed,
        n_gauss_points=cfg.n_gauss_points,
    )

    # 3. 设备选择 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 4. 创建水动力与输沙 PINN 模型
    flow_model = FlowPINN(input_dim=3, hidden_dim=64, num_block=4, output_dim=3).to(device)

    n_grains = cfg.num_grain_classes
    sediment_model = SedimentPINN(
        input_dim=3,
        hidden_dim=64,
        num_block=4,
        output_dim=n_grains,
        initial_concentration=cfg.initial_sediment_concentration,
    ).to(device)

    # 5. 创建物理损失函数
    flow_loss_fn = SVEsPhysicsLoss(
        fvm_mesh=mesh,
        g=cfg.g,
        n_manning=cfg.n_manning,
        bounds=cfg.bounds,
        typical_depth=cfg.typical_depth,
        typical_velocity=cfg.typical_velocity,
        simulation_time=cfg.simulation_time,
        include_time_terms=cfg.include_time_terms,
    )

    sediment_transport_loss_fn = SedimentTransportLoss(
        fvm_mesh=mesh,
        bounds=cfg.bounds,
        include_time_terms=cfg.include_time_terms,
        grain_diameters=cfg.grain_diameters,
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
        initial_sediment_concentration=cfg.initial_sediment_concentration,
        w_inlet_sediment=cfg.w_inlet_sediment,
        inlet_sediment_concentration=cfg.inlet_sediment_concentration,
        source_sharpness=cfg.source_sharpness,
        simulation_time=cfg.simulation_time,
        typical_depth=cfg.typical_depth,
        typical_velocity=cfg.typical_velocity,
    )

    # 6. 创建解耦训练器 
    trainer = DecoupledTrainer(
        fvm_mesh=mesh,
        device=device,
        flow_model=flow_model,
        sediment_model=sediment_model,
        flow_loss_fn=flow_loss_fn,
        sediment_transport_loss_fn=sediment_transport_loss_fn,
        simulation_time=cfg.simulation_time,
        porosity=cfg.porosity,
        bed_slope_coefficient=cfg.bed_slope_coefficient,
        flow_lr=cfg.training.get('flow_lr', 1e-4),
        transport_lr=cfg.training.get('transport_lr', 1e-4),
    )

    # 7. 边界条件构建器 
    def bc_builder(t_norm):
        return build_boundary_conditions(
            t_norm=t_norm,
            bbox=cfg.bbox,
            bounds=cfg.bounds,
            bc_default=cfg.bc_default,
            typical_depth=cfg.typical_depth,
            typical_velocity=cfg.typical_velocity,
        )

    # 8. 运行解耦训练 
    print("开始 hump 演变测试...")

    bed_history = trainer.run_training(
        simulation_time=cfg.simulation_time,
        sample_dt=cfg.sample_dt,
        window_dt=cfg.window_dt,
        output_dt=cfg.output_dt,
        flow_epochs_per_window=cfg.training.get('flow_epochs_per_step'),
        sediment_epochs_per_window=cfg.training.get('sediment_epochs_per_step'),
        bc_builder=bc_builder,
        flow_loss_tol=cfg.training.get('flow_loss_tol'),
        sediment_loss_tol=cfg.training.get('sediment_loss_tol'),
        extra_train_chunk=cfg.training.get('extra_train_chunk'),
        max_extra_flow_epochs=cfg.training.get('max_extra_flow_epochs'),
        max_extra_sediment_epochs=cfg.training.get('max_extra_sediment_epochs'),
        max_bed_change_per_step=cfg.training.get('max_bed_change_per_step'),
    )

    # 9. 可视化结果 
    visualize_results(
        mesh=mesh,
        bed_history=bed_history,
        bbox=cfg.bbox,
        resolution=cfg.resolution,
        history=trainer.history,
        simulation_time=cfg.simulation_time,
        case_name='hump',
        output_dir='outputs',
    )

    return bed_history, trainer.history


if __name__ == '__main__':
    
    config_path = "config.yaml"
    run_hump_evolution_test(config_path=config_path)
