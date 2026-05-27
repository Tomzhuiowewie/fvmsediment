import torch

# 设备配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 域边界和分辨率
BBOX = {'xmin': 0.0, 'xmax': 1000.0, 'ymin': 0.0, 'ymax': 1000.0}
RESOLUTION = 25.0

# 每条边高斯积分点数量
N_GAUSS_POINTS = 2

# 坐标归一化边界
BOUNDS = {'x_min': 0.0, 'x_max': 1000.0, 'y_min': 0.0, 'y_max': 1000.0}

# 典型深度和速度，用于损失函数的物理量级调整
TYPICAL_DEPTH = 10.0
TYPICAL_VELOCITY = 1.0

# 颗粒直径和类别数量
GRAIN_DIAMETERS = [2e-4, 5e-4]
NUM_GRAIN_CLASSES = len(GRAIN_DIAMETERS)

# 不同演化速率对应的A_g值和物理时间长度
AG_VALUES = {'slow': 0.001, 'fast': 1.0}
TPHYSICAL = {'slow': 360000.0, 'fast': 600.0}

# 训练设置，包括学习率、预热轮数、每步训练轮数和宏观步骤数
TRAINING_SETTINGS = {
    'flow_lr': 1e-3,
    'sediment_lr': 1e-3,
    'transport_lr': 1e-3,
    'gradation_lr': 1e-3,
    'warmup_ic_epochs': 800,
    'flow_epochs_per_step': 300,
    'sediment_epochs_per_step': 400,
    'n_macro_steps': 200,
}

# 边界条件默认设置，包括边界点数量、时间归一化值和水深、速度的归一化值
BC_DEFAULT = {
    'n_bc': 50,
    't_normalized': 0.5,
    'h_norm': 1.0,
    'u_norm': 1.0,
    'v_norm': 0.5,
}
