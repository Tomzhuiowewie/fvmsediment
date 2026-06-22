# utils.py - 通用工具
# 提供坐标归一化、张量闭包匹配、结果保存、JSON 序列化等辅助函数。

import json
import os
import shutil
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .config import EPS_DIVISION


# =============================================================================
# 数值与张量工具
# =============================================================================

def normalize_coordinates(coords: torch.Tensor, bounds: dict) -> torch.Tensor:
    """将 (x, y) 坐标按给定上下界映射到 [0, 1] 区间。"""
    x = (coords[:, 0:1] - bounds['x_min']) / (bounds['x_max'] - bounds['x_min'])
    y = (coords[:, 1:2] - bounds['y_min']) / (bounds['y_max'] - bounds['y_min'])
    return torch.cat([x, y], dim=1)


def build_xyt(coords: torch.Tensor, t_norm: float, bounds: Optional[dict],
              device: torch.device, requires_grad: bool = True) -> torch.Tensor:
    """构造归一化的 (x, y, t) 输入张量，供网络前向与自动微分使用。"""
    coords = coords.to(device=device, dtype=torch.float32)
    if bounds is not None:
        xyt_xy = normalize_coordinates(coords, bounds)
    else:
        xyt_xy = coords[:, 0:2]
    t_tensor = torch.full((coords.shape[0], 1), t_norm, dtype=torch.float32, device=device)
    xyt = torch.cat([xyt_xy, t_tensor], dim=1)
    return xyt.requires_grad_(requires_grad)


# def match_closure(value, reference: torch.Tensor, default: float) -> torch.Tensor:
#     """将标量/数组闭合量整理成与 reference 形状一致的张量。"""
#     if value is None:
#         return torch.ones_like(reference) * default
#     if torch.is_tensor(value):
#         value_tensor = value.to(dtype=reference.dtype, device=reference.device)
#     else:
#         value_tensor = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
#     return value_tensor.expand_as(reference) if value_tensor.numel() == 1 else value_tensor


def time_derivative(
    q: torch.Tensor,
    xyt: torch.Tensor,
    simulation_time: float,
    include_time_terms: bool = True,
) -> torch.Tensor:
    """计算归一化网络时间对应的物理时间导数 dq/dt。"""
    if not include_time_terms:
        return torch.zeros_like(q if q.dim() > 1 else q.unsqueeze(1))

    def derivative_one(q_one: torch.Tensor) -> torch.Tensor:
        grad = torch.autograd.grad(
            q_one,
            xyt,
            grad_outputs=torch.ones_like(q_one),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]
        if grad is None:
            return torch.zeros_like(q_one)
        return grad[:, 2:3] / max(simulation_time, EPS_DIVISION)

    if q.dim() == 1 or q.shape[1] == 1:
        return derivative_one(q if q.dim() > 1 else q.unsqueeze(1))
    return torch.cat([derivative_one(q[:, k:k + 1]) for k in range(q.shape[1])], dim=1)


def smooth_positive(x, beta=10.0):
    """平滑正部函数，用 softplus 近似 max(x, 0)，在 x=0 处可导。"""
    # return F.softplus(x / sharpness) * sharpness
    return F.softplus(beta * x) / beta

def match_concentration(values, n_grains):
    """将初始泥沙浓度配置整理成与粒径组数量一致的列表"""
    if len(values) == n_grains:
        return values
    if len(values) == 1:
        return values * n_grains
    return [0.0] * n_grains


# =============================================================================
# 输出保存
# =============================================================================

def save_training_outputs(
    output_dir,
    config_path,
    flow_model,
    sediment_model,
    trainer,
    bed_history,
    run_timestamp,
):
    """保存正式训练结果，便于后处理、复现实验和中断恢复。"""
    os.makedirs(output_dir, exist_ok=True)
    torch.save(
        flow_model.state_dict(),
        os.path.join(output_dir, f'flow_model_{run_timestamp}.pt'),
    )
    torch.save(
        sediment_model.state_dict(),
        os.path.join(output_dir, f'sediment_model_{run_timestamp}.pt'),
    )
    torch.save(
        {
            'flow_model': flow_model.state_dict(),
            'sediment_model': sediment_model.state_dict(),
            'active_layer_frac': trainer.active_layer_frac,
            'mesh_zb': trainer.mesh.zb,
            'history': trainer.history,
            'simulation_time': trainer.simulation_time,
            'run_timestamp': run_timestamp,
        },
        os.path.join(output_dir, f'final_checkpoint_{run_timestamp}.pt'),
    )
    np.savez_compressed(
        os.path.join(output_dir, f'training_results_{run_timestamp}.npz'),
        bed_history=np.asarray(bed_history, dtype=np.float64),
        active_layer_frac=np.asarray(trainer.active_layer_frac, dtype=np.float32),
        output_times=np.asarray(trainer.history.get('output_times', []), dtype=np.float32),
        integration_times=np.asarray(trainer.history.get('integration_times', []), dtype=np.float32),
    )
    save_time_point_outputs(
        output_dir=output_dir,
        bed_history=bed_history,
        output_times=trainer.history.get('output_times', []),
        run_timestamp=run_timestamp,
    )
    with open(
        os.path.join(output_dir, f'history_{run_timestamp}.json'),
        'w',
        encoding='utf-8',
    ) as f:
        json.dump(json_safe(trainer.history), f, ensure_ascii=False, indent=2)
    if config_path and os.path.exists(config_path):
        shutil.copyfile(
            config_path,
            os.path.join(output_dir, f'config_used_{run_timestamp}.yaml'),
        )
    print(f"训练结果已保存到: {output_dir}")


def save_time_point_outputs(output_dir, bed_history, output_times, run_timestamp):
    """按模拟时间点分别保存床面高程和累计床变。"""
    bed_array = np.asarray(bed_history, dtype=np.float64)
    times = np.asarray(output_times, dtype=np.float64).reshape(-1)
    if bed_array.ndim != 2:
        raise ValueError("bed_history 必须为 [时间点, 网格单元] 二维数组。")
    if times.size != bed_array.shape[0]:
        raise ValueError("output_times 数量必须与 bed_history 时间维一致。")

    time_dir = os.path.join(output_dir, f'time_points_{run_timestamp}')
    os.makedirs(time_dir, exist_ok=True)
    initial_bed = bed_array[0]
    index = []
    for time_seconds, bed in zip(times, bed_array):
        time_label = format_time_label(time_seconds)
        filename = f'bed_{time_label}.npz'
        np.savez_compressed(
            os.path.join(time_dir, filename),
            time_seconds=np.float64(time_seconds),
            time_days=np.float64(time_seconds / 86400.0),
            bed_elevation=bed,
            bed_change=bed - initial_bed,
        )
        index.append({
            'time_seconds': float(time_seconds),
            'time_days': float(time_seconds / 86400.0),
            'file': filename,
        })

    with open(os.path.join(time_dir, 'index.json'), 'w', encoding='utf-8') as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    print(f"分时刻床面结果已保存到: {time_dir}")


def format_time_label(time_seconds):
    """生成可排序的模拟时间标签，例如 t_0034d_00h_2937600s。"""
    total_seconds = int(round(float(time_seconds)))
    days, remainder = divmod(total_seconds, 86400)
    hours = remainder // 3600
    return f't_{days:04d}d_{hours:02d}h_{total_seconds:010d}s'


def json_safe(value):
    """把 NumPy/PyTorch 标量递归转成 JSON 可写对象。"""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


# =============================================================================
# 配置展示
# =============================================================================

def summarize_config(cfg):
    """按功能分组返回配置摘要，便于 notebook 调试查看。"""
    time_cfg = cfg.time
    flow = cfg.flow
    sediment = cfg.sediment
    morph = cfg.morphodynamics
    training = cfg.training
    return {
        '数据输入': {
            'DEM 路径': cfg.data.get('dem_path'),
            '水沙过程线': cfg.data.get('hydro_sediment_path'),
            '河道高程阈值(ft)': cfg.data.get('channel_elevation_threshold_ft'),
            '入口方向': cfg.data.get('inlet_edge', 'top'),
            '出口方向': cfg.data.get('outlet_edge', 'bottom'),
        },
        '时间控制': {
            '总模拟时间(s)': time_cfg.simulation_time,
            '训练采样间隔(s)': time_cfg.sample_dt,
            '输出间隔(s)': time_cfg.output_dt,
            '是否包含非恒定项': time_cfg.include_time_terms,
        },
        '网格与尺度': {
            '边界高斯点数': cfg.n_gauss_points,
            '典型水深(m)': flow.typical_depth,
            '典型流速(m/s)': flow.typical_velocity,
        },
        '水动力': {
            '重力加速度(m/s2)': flow.g,
            '曼宁糙率': flow.n_manning,
            '水动力自适应权重': flow.adaptive_loss_weighting,
            '权重 EMA': flow.adaptive_weight_ema_decay,
            '权重下限': flow.adaptive_weight_min,
            '权重上限': flow.adaptive_weight_max,
        },
        '泥沙输运': {
            '粒径组数': len(sediment.grain_diameters),
            '粒径组(mm)': [round(d * 1000.0, 6) for d in sediment.grain_diameters],
            'beta_default': sediment.beta_default,
            'epsilon_default': sediment.epsilon_default,
            '残差缩放': sediment.residual_scale,
            '适应长度(m)': sediment.adaptation_length,
            '颗粒密度(kg/m3)': sediment.rho_s,
            '水密度(kg/m3)': sediment.rho_w,
            '运动黏度(m2/s)': sediment.kinematic_viscosity,
            'Wu 临界 Shields': sediment.wu_theta_cr,
            'skin shear factor': sediment.skin_shear_factor,
            '输沙能力权重': sediment.w_capacity,
            '初始浓度权重': sediment.w_initial_sediment,
            '入口浓度权重': sediment.w_inlet_sediment,
            '初始浓度': sediment.initial_concentration,
        },
        '床面形态': {
            '孔隙率': morph.porosity,
            '活动层厚度(m)': morph.active_layer_thickness,
            '床面变化输出尺度(m)': morph.bed_change_scale,
            '床坡系数': morph.bed_slope_coefficient,
            '床坡扩散权重': morph.bed_slope_diffusion_weight,
            '交换项权重': morph.exchange_weight,
            '活动层系数': sediment.alpha_active_layer,
        },
        '训练轮次与学习率': {
            '水动力 epoch': training.get('flow_epochs'),
            '泥沙 epoch': training.get('sediment_epochs'),
            '联合 epoch': training.get('joint_epochs'),
            '水动力学习率': training.get('flow_lr'),
            '泥沙学习率': training.get('transport_lr'),
            '泥沙 batch 单元数': training.get('sediment_cell_batch_size'),
        },
        '耦合与停止': {
            '耦合迭代次数': training.get('coupling_iterations'),
            '床面松弛系数': training.get('coupling_relaxation'),
            '床面收敛阈值(m)': training.get('coupling_bed_tol'),
            '水动力 loss 停止阈值': training.get('flow_loss_tol'),
            '泥沙 loss 停止阈值': training.get('sediment_loss_tol'),
            '联合 loss 停止阈值': training.get('joint_loss_tol'),
            '早停 patience': training.get('early_stop_patience'),
            '早停最小改善': training.get('early_stop_min_delta'),
        },
        '边界损失与输出': {
            '流量边界固定权重': training.get('flow_boundary_weight'),
            '边界自适应权重': training.get('adaptive_boundary_weighting'),
            '边界权重 EMA': training.get('boundary_weight_ema_decay'),
            '边界权重下限': training.get('boundary_weight_min'),
            '边界权重上限': training.get('boundary_weight_max'),
            '床变损失权重': training.get('w_bed_change'),
            '输出目录': training.get('output_dir'),
            'checkpoint 目录': training.get('checkpoint_dir'),
        },
    }


def print_config_summary(cfg):
    """以纯文本打印配置摘要。"""
    for group, values in summarize_config(cfg).items():
        print(f'\n[{group}]')
        for name, value in values.items():
            print(f'  {name}: {value}')
