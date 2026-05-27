# config.py – 模拟配置
# 提供 SimulationConfig dataclass 和统一的数值常量。
# 支持通过实例化不同 SimulationConfig 切换实验参数，
# 同时保留模块级常量作为默认配置的快捷访问。

from dataclasses import dataclass, field
from typing import Dict, List

import torch


# ── 统一数值 epsilon 常量 ──
EPS_SAFE = 1e-6           # 通用安全下限（平方根、除法分母等）
EPS_DIVISION = 1e-8       # 严格防零除（rho_s、beta_tk 等）
EPS_VELOCITY_CLAMP = 1e-3 # 流速下限截断（适应长度计算）


@dataclass
class SimulationConfig:
    device: torch.device = field(default_factory=lambda: torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # 域边界和分辨率
    bbox: Dict[str, float] = field(default_factory=lambda: {'xmin': 0.0, 'xmax': 1000.0, 'ymin': 0.0, 'ymax': 1000.0})
    resolution: float = 25.0

    # 每条边高斯积分点数量
    n_gauss_points: int = 2

    # 坐标归一化边界
    bounds: Dict[str, float] = field(default_factory=lambda: {'x_min': 0.0, 'x_max': 1000.0, 'y_min': 0.0, 'y_max': 1000.0})

    # 典型深度和速度，用于损失函数的物理量级调整
    typical_depth: float = 10.0
    typical_velocity: float = 1.0

    # 颗粒直径和类别数量
    grain_diameters: List[float] = field(default_factory=lambda: [2e-4, 5e-4])

    # 不同演化速率对应的 A_g 值和物理时间长度
    ag_values: Dict[str, float] = field(default_factory=lambda: {'slow': 0.001, 'fast': 1.0})
    t_physical: Dict[str, float] = field(default_factory=lambda: {'slow': 360000.0, 'fast': 600.0})

    # 训练设置
    training: Dict[str, float] = field(default_factory=lambda: {
        'flow_lr': 1e-3,
        'sediment_lr': 1e-3,
        'transport_lr': 1e-3,
        'gradation_lr': 1e-3,
        'warmup_ic_epochs': 800,
        'flow_epochs_per_step': 300,
        'sediment_epochs_per_step': 400,
        'n_macro_steps': 200,
    })

    # 边界条件默认设置
    bc_default: Dict[str, float] = field(default_factory=lambda: {
        'n_bc': 50,
        't_normalized': 0.5,
        'h_norm': 1.0,
        'u_norm': 1.0,
        'v_norm': 0.5,
    })

    @property
    def num_grain_classes(self) -> int:
        return len(self.grain_diameters)


# ── 向后兼容的模块级别名 ──
_default = SimulationConfig()

DEVICE = _default.device
BBOX = _default.bbox
RESOLUTION = _default.resolution
N_GAUSS_POINTS = _default.n_gauss_points
BOUNDS = _default.bounds
TYPICAL_DEPTH = _default.typical_depth
TYPICAL_VELOCITY = _default.typical_velocity
GRAIN_DIAMETERS = _default.grain_diameters
NUM_GRAIN_CLASSES = _default.num_grain_classes
AG_VALUES = _default.ag_values
TPHYSICAL = _default.t_physical
TRAINING_SETTINGS = _default.training
BC_DEFAULT = _default.bc_default
