
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm, trange
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.animation import FuncAnimation
    HAS_MATPLOTLIB = True
except ImportError:
    # 训练本身不依赖 matplotlib；缺少绘图库时只跳过结果绘图。
    plt = None
    GridSpec = None
    FuncAnimation = None
    HAS_MATPLOTLIB = False
import warnings
warnings.filterwarnings("ignore")


# 生成规则二维有限体积网络，并预计算每个单元边界上的高斯积分点（数值积分）
class FVMeshPreprocessor:
    def __init__(self, bbox, resolution, initial_bed=None, n_gauss_points=2):
       """
       初始化FVM网格预处理器
       参数:
       bbox: 计算域边界
       resolution: 网格分辨率
       initial_bed: 初始床面
       n_gauss_points: 高斯点数
       """
       self.bbox = bbox
       self.resolution = resolution
       self.n_gauss_points = n_gauss_points

       print(f"\n初始化FVM网格预处理器")
       print(f"  计算域: X=[{bbox['xmin']}, {bbox['xmax']}], Y=[{bbox['ymin']}, {bbox['ymax']}]")
       print(f"  分辨率: {resolution}m, 高斯点数: {n_gauss_points}")

       self._generate_mesh()               # 生成FVM网格
       self._initialize_bed(initial_bed)   # 初始化床面
       self._setup_gauss_quadrature()      # 设置高斯积分点
       self._precompute_edge_data()        # 预计算每个单元边界上的高斯积分点

       print(f" FVM预处理完成: {self.n_cells}个单元, {self.n_gauss_total}个高斯点")

    def _generate_mesh(self):

        xmin, xmax = self.bbox['xmin'], self.bbox['xmax']
        ymin, ymax = self.bbox['ymin'], self.bbox['ymax']

        # 单元数量
        self.nx = int((xmax - xmin) / self.resolution)
        self.ny = int((ymax - ymin) / self.resolution)

        # 单元中心坐标
        x_centers = np.linspace(xmin + self.resolution / 2, xmax - self.resolution / 2, self.nx)
        y_centers = np.linspace(ymin + self.resolution / 2, ymax - self.resolution / 2, self.ny)

        self.cell_centers_x, self.cell_centers_y = np.meshgrid(x_centers, y_centers)
        self.cell_centers_x = self.cell_centers_x.flatten()
        self.cell_centers_y = self.cell_centers_y.flatten()

        self.n_cells = len(self.cell_centers_x)
        self.cell_area = self.resolution ** 2

        # 建立单元索引映射 (i,j) -> cell_id
        self.cell_index = np.arange(self.n_cells).reshape(self.ny, self.nx)

        print(f"  网格: {self.nx} x {self.ny} = {self.n_cells}个单元")
        print(f"  单元面积: {self.cell_area} m²")

    def _initialize_bed(self, initial_bed):
        if initial_bed is None:
            # 默认平床
            self.zb = np.zeros(self.n_cells)
        elif callable(initial_bed):
            # 函数形式
            self.zb = initial_bed(self.cell_centers_x, self.cell_centers_y)
        else:
            # 固定值
            self.zb = np.full(self.n_cells, initial_bed)

        self.zb_initial = self.zb.copy()  # 保存初始床面供 IC LOSS 使用
        print(f"  初始河床高程: {np.min(self.zb):.3f} ~ {np.max(self.zb):.3f} m")

    def _setup_gauss_quadrature(self):
        if self.n_gauss_points == 2:
            sqrt_1_3 = np.sqrt(1.0 / 3.0)
            self.gauss_xi = np.array([-sqrt_1_3, sqrt_1_3])
            self.gauss_weights_1d = np.array([1.0, 1.0])
        else:
            self.gauss_xi = np.array([0.0])
            self.gauss_weights_1d = np.array([2.0])

    def _precompute_edge_data(self):
        half_res = self.resolution / 2
        n_edges_per_cell = 4
        n_points_per_cell = n_edges_per_cell * self.n_gauss_points
        self.n_points_per_cell = n_points_per_cell
        self.n_gauss_total = self.n_cells * n_points_per_cell

        # 预分配数组
        self.gauss_coords = np.zeros((self.n_gauss_total, 2))   # 高斯点坐标
        self.gauss_normals = np.zeros((self.n_gauss_total, 2))   # 高斯点法向量
        self.gauss_weights = np.zeros(self.n_gauss_total)   # 高斯点权重    
        self.gauss_cell_id = np.zeros(self.n_gauss_total, dtype=int)   # 高斯点所属单元索引
        self.gauss_edge_id = np.zeros(self.n_gauss_total, dtype=int)  # 高斯点所属边索引(0=S, 1=E, 2=N, 3=W)

        # 邻居单元索引 (-1表示边界)
        self.gauss_neighbor_id = np.full(self.n_gauss_total, -1, dtype=int)

        idx = 0
        for cell_i in range(self.n_cells):
            cx = self.cell_centers_x[cell_i]    # 单元中心x坐标
            cy = self.cell_centers_y[cell_i]    # 单元中心y坐标

            # 计算单元在网格中的位置(一维索引转二维坐标)
            grid_i = cell_i // self.nx  # 行
            grid_j = cell_i % self.nx  # 列

            # 四条边(S, E, N, W)定义: [法向量, 边中心偏移, 切向偏移方向, 邻居偏移]
            edges = [
                ([0, -1], (0, -half_res), (1, 0), (-1, 0)),  # South 
                ([1, 0], (half_res, 0), (0, 1), (0, 1)),  # East
                ([0, 1], (0, half_res), (1, 0), (1, 0)),  # North
                ([-1, 0], (-half_res, 0), (0, 1), (0, -1)),  # West
            ]

            for edge_id, (normal, center_offset, tangent, neighbor_offset) in enumerate(edges):
                # 计算邻居单元
                ni = grid_i + neighbor_offset[0]
                nj = grid_j + neighbor_offset[1]

                if 0 <= ni < self.ny and 0 <= nj < self.nx:
                    neighbor_cell = ni * self.nx + nj
                else:
                    neighbor_cell = -1  # 边界

                for j, xi in enumerate(self.gauss_xi):
                    x_g = cx + center_offset[0] + xi * half_res * tangent[0]    # 高斯点x坐标(中心点 + 边中心偏移 + 沿边方向的高斯点偏移)
                    y_g = cy + center_offset[1] + xi * half_res * tangent[1]    # 高斯点y坐标(中心点 + 边中心偏移 + 沿边方向的高斯点偏移)

                    self.gauss_coords[idx] = [x_g, y_g]   # 高斯点坐标
                    self.gauss_normals[idx] = normal   # 高斯点法向量
                    self.gauss_weights[idx] = self.gauss_weights_1d[j] * half_res   # 高斯点权重
                    self.gauss_cell_id[idx] = cell_i   # 高斯点所属单元索引
                    self.gauss_edge_id[idx] = edge_id   # 高斯点所属边索引
                    self.gauss_neighbor_id[idx] = neighbor_cell   # 高斯点所属邻居单元索引
                    idx += 1   # 高斯点索引递增

        print(f"  高斯点总数: {self.n_gauss_total}")

    def update_bed(self, new_zb):
        self.zb = np.clip(np.array(new_zb), -5.0, 5.0)

    def get_bed_at_gauss_points(self):
        return self.zb[self.gauss_cell_id]

    def get_bed_gradient(self):
        zb_2d = self.zb.reshape(self.ny, self.nx)
        dzb_dx = np.zeros_like(zb_2d)
        dzb_dy = np.zeros_like(zb_2d)

        dzb_dx[:, 1:-1] = (zb_2d[:, 2:] - zb_2d[:, :-2]) / (2 * self.resolution)
        dzb_dx[:, 0] = (zb_2d[:, 1] - zb_2d[:, 0]) / self.resolution
        dzb_dx[:, -1] = (zb_2d[:, -1] - zb_2d[:, -2]) / self.resolution

        dzb_dy[1:-1, :] = (zb_2d[2:, :] - zb_2d[:-2, :]) / (2 * self.resolution)
        dzb_dy[0, :] = (zb_2d[1, :] - zb_2d[0, :]) / self.resolution
        dzb_dy[-1, :] = (zb_2d[-1, :] - zb_2d[-2, :]) / self.resolution

        return dzb_dx.flatten(), dzb_dy.flatten()


# 残差块，包含两个全连接层和一个跳跃连接
class ResBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.Tanh()

    def forward(self, x):
        residual = x
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return self.activation(x + residual)


# 1.水动力-PINN模型：输入x y t， 输出h u v
class FlowPINN(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, num_block=4, output_dim=3):
        super().__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh()
        )

        self.res_blocks = nn.ModuleList([
            ResBlock(hidden_dim) for _ in range(num_block)])

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid()    # 这个是否合适？
        )
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, xyt):
        x = self.input_layer(xyt)
        for block in self.res_blocks:
            x = block(x)
        return self.output_layer(x)


# 2.总输沙输移方程PINN模型：输入x y t，输出各粒径级总输沙浓度 C_tk
class SedimentPINN(nn.Module):
    """
    总输沙输移方程网络。

    对应 HEC-RAS 2D Sediment Technical Reference Manual 的原文总输沙方程：
        ∂/∂t(h C_tk / β_tk) + ∇·(h U C_tk)
        = ∇·(ε_thk h ∇C_tk) + E_tk - D_tk

    这里 forward 只预测各粒径级总输沙浓度 C_tk；
    total_load_residual() 负责把原文 PDE 写成 PINN 残差。
    """

    def __init__(self, input_dim=3, hidden_dim=64, num_block=4, output_dim=1, positive_output=True, **kwargs):
        """
        参数:
            input_dim: 输入维度，默认 (x, y, t) 共 3 维
            hidden_dim: 隐藏层宽度
            num_block: 残差块数量
            output_dim: 粒径级数量 K，输出 C_1...C_K
            positive_output: 是否用 softplus 保证 C_tk >= 0
            **kwargs: 兼容旧代码传入的 zb_scale 等无关参数
        """
        super().__init__()
        self.positive_output = positive_output

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh()
        )

        self.res_blocks = nn.ModuleList([
            ResBlock(hidden_dim) for _ in range(num_block)
        ])

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, xyt):
        """预测总输沙浓度 C_tk，形状为 [N, K]。"""
        x = self.input_layer(xyt)
        for block in self.res_blocks:
            x = block(x)
        out = self.output_layer(x)

        # 浓度不能为负；softplus 比 ReLU 更平滑，适合 PINN 自动微分。
        return F.softplus(out) if self.positive_output else out

    @staticmethod
    def _grad(q, xyt, dim):
        """对 xyt 的某一列求偏导；dim=0,1,2 分别代表 x,y,t。

        注意：torch.autograd.grad 默认会把多输出张量求和后再求导。
        因此当 q 形状为 [N, K] 时，需要逐粒径级单独求导，避免把
        C_1...C_K 或 p_1...p_K 的梯度混在一起。
        """
        if not xyt.requires_grad:
            raise ValueError("xyt 必须设置 requires_grad=True，才能计算 PDE 残差。")

        def grad_one(q_one):
            g = torch.autograd.grad(
                q_one, xyt,
                grad_outputs=torch.ones_like(q_one),
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )[0]
            if g is None:
                return torch.zeros_like(q_one)
            return g[:, dim:dim + 1]

        if q.dim() == 1 or q.shape[1] == 1:
            return grad_one(q if q.dim() > 1 else q.unsqueeze(1))

        return torch.cat([grad_one(q[:, k:k + 1]) for k in range(q.shape[1])], dim=1)

    @staticmethod
    def _dx(q, xyt):
        return SedimentPINN._grad(q, xyt, 0)

    @staticmethod
    def _dy(q, xyt):
        return SedimentPINN._grad(q, xyt, 1)

    @staticmethod
    def _dt(q, xyt):
        return SedimentPINN._grad(q, xyt, 2)

    @staticmethod
    def total_load_residual(xyt, h, u, v, C_tk, beta_tk, epsilon_thk, E_tk, D_tk):
        """
        HEC-RAS 原文总输沙输移方程残差。

        原文 PDE：
            ∂/∂t(h C_tk / β_tk) + ∇·(h U C_tk)
            = ∇·(ε_thk h ∇C_tk) + E_tk - D_tk

        残差写成：
            R_Ck = ∂/∂t(h C_tk / β_tk)
                   + ∂(h u C_tk)/∂x + ∂(h v C_tk)/∂y
                   - ∂(ε_thk h ∂C_tk/∂x)/∂x
                   - ∂(ε_thk h ∂C_tk/∂y)/∂y
                   - E_tk + D_tk

        理想情况下 R_Ck = 0。
        所有输入建议形状为 [N, K]；h,u,v 可为 [N,1]，会自动广播到 K 个粒径级。
        """
        eps = 1e-8
        beta_safe = torch.clamp(beta_tk, min=eps)

        # 储量项：严格按 HEC-RAS 原文，β_tk 在时间项分母。
        storage = h * C_tk / beta_safe
        storage_t = SedimentPINN._dt(storage, xyt)

        # 平流项：∇·(h U C_tk)。
        adv_x = h * u * C_tk
        adv_y = h * v * C_tk
        advection = SedimentPINN._dx(adv_x, xyt) + SedimentPINN._dy(adv_y, xyt)

        # 扩散项：∇·(ε_thk h ∇C_tk)。
        diff_x = epsilon_thk * h * SedimentPINN._dx(C_tk, xyt)
        diff_y = epsilon_thk * h * SedimentPINN._dy(C_tk, xyt)
        diffusion = SedimentPINN._dx(diff_x, xyt) + SedimentPINN._dy(diff_y, xyt)

        # 源汇项：侵蚀 E_tk 为源，沉积 D_tk 为汇。
        return storage_t + advection - diffusion - E_tk + D_tk

    @staticmethod
    def total_load_loss(xyt, h, u, v, C_tk, beta_tk, epsilon_thk, E_tk, D_tk):
        """总输沙 PDE 损失：MSE(R_Ck)。"""
        residual = SedimentPINN.total_load_residual(
            xyt, h, u, v, C_tk, beta_tk, epsilon_thk, E_tk, D_tk
        )
        return torch.mean(residual ** 2), residual

    @staticmethod
    def grass_formula(u, v, Ag=0.001, m=3, eps=1e-6):
        """
        Grass 简化输沙通量公式，保留给旧的 ExnerPhysicsLoss 使用。
        qx = Ag * u * |U|^(m-1), qy = Ag * v * |U|^(m-1)
        """
        vel_mag = torch.sqrt(u ** 2 + v ** 2 + eps)
        factor = Ag * torch.pow(vel_mag, m - 1)
        qx = factor * u
        qy = factor * v
        return qx, qy, vel_mag
    

# 3.Exner床面演变-PINN模型：输入x y t， 输出zb 河床高程变化
class BedPINN(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, num_block=4, output_dim=1,  zb_scale=1.5):    # zb 物理范围上界 (m) zb_scale
        super().__init__()
        self.zb_scale = zb_scale
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh()
        )

        self.res_blocks = nn.ModuleList([
            ResBlock(hidden_dim) for _ in range(num_block)
        ])

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, xyT):
        T = xyT[:, 2:3]  # 提取T分量

        x = self.input_layer(xyT)
        for block in self.res_blocks:
            x = block(x)
        out = self.output_layer(x)

#         return out * T
        return torch.tanh(out) * self.zb_scale   # 绝对zb，范围(-1.5, +1.5)m


# 4.床沙级配 / 活动层质量守恒方程PINN模型：输入x y t，输出各粒径级床沙比例 p_k
class GradationPINN(nn.Module):
    """
    床沙级配 / 活动层质量守恒网络。

    对应关系：
        GradationNet(x, y, t) -> p_1, ..., p_K

    其中 p_k 表示活动层内第 k 个粒径级的体积分数。forward 使用 softmax，
    因此天然满足：
        p_k >= 0,  sum_k p_k = 1

    active_layer_residual() 给出一个可微 PINN 残差，用于约束活动层级配随
    侵蚀/沉积交换项 E_k、D_k 的变化。
    """

    def __init__(self, input_dim=3, hidden_dim=64, num_block=4, output_dim=1):
        """
        参数:
            input_dim: 输入维度，默认 (x, y, t)
            hidden_dim: 隐藏层宽度
            num_block: 残差块数量
            output_dim: 粒径级数量 K，输出 p_1...p_K
        """
        super().__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh()
        )

        self.res_blocks = nn.ModuleList([
            ResBlock(hidden_dim) for _ in range(num_block)
        ])

        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, xyt):
        """预测床沙级配 p_k，形状为 [N, K]。"""
        x = self.input_layer(xyt)
        for block in self.res_blocks:
            x = block(x)
        raw_p = self.output_layer(x)

        # softmax 保证每个点的粒径比例非负且和为 1。
        return torch.softmax(raw_p, dim=-1)

    @staticmethod
    def gradation_constraint(p_k):
        """级配归一约束残差：sum_k p_k - 1。"""
        return torch.sum(p_k, dim=1, keepdim=True) - 1.0

    @staticmethod
    def active_layer_thickness(d_k, p_k=None, alpha_a=1.0):
        """
        活动层厚度 L_a 的简化闭合。

        若给定 p_k，则用级配加权平均粒径代表活动层尺度；否则用最大粒径代表。
        工程上也可替换为 L_a = alpha_a * D90。
        """
        if not torch.is_tensor(d_k):
            like = p_k if p_k is not None else None
            d_k = torch.as_tensor(d_k, dtype=torch.float32, device=None if like is None else like.device)
        if p_k is not None:
            d_k = d_k.to(dtype=p_k.dtype, device=p_k.device).view(1, -1)
            d_ref = torch.sum(p_k * d_k, dim=1, keepdim=True)
        else:
            d_ref = torch.max(d_k).view(1, 1)
        return alpha_a * d_ref

    @staticmethod
    def active_layer_residual(xyt, p_k, L_a, E_k, D_k, rho_s=2650.0):
        """
        活动层级配守恒残差。

        简化形式：
            R_pk = ∂(L_a p_k)/∂t - (D_k - E_k)/rho_s
                   + p_k * sum_j((D_j - E_j)/rho_s)

        含义：某粒径级的活动层储量变化，等于该粒径级沉积/侵蚀交换量，
        并扣除总床面升降对各粒径比例的稀释/富集影响。理想情况下 R_pk = 0。
        """
        if not torch.is_tensor(rho_s):
            rho_s = torch.as_tensor(rho_s, dtype=p_k.dtype, device=p_k.device)

        exchange = (D_k - E_k) / (rho_s + 1e-8)
        total_exchange = torch.sum(exchange, dim=1, keepdim=True)

        storage = L_a * p_k
        storage_t = SedimentPINN._dt(storage, xyt)

        return storage_t - exchange + p_k * total_exchange

    @staticmethod
    def active_layer_loss(xyt, p_k, L_a, E_k, D_k, rho_s=2650.0, w_sum=1.0):
        """活动层级配 PDE 损失：MSE(R_pk) + 归一约束。"""
        residual = GradationPINN.active_layer_residual(xyt, p_k, L_a, E_k, D_k, rho_s)
        sum_residual = GradationPINN.gradation_constraint(p_k)
        loss = torch.mean(residual ** 2) + w_sum * torch.mean(sum_residual ** 2)
        return loss, residual


class CoupledFlowSedimentPINN(nn.Module):
    """
    四网络耦合容器，明确每个子网络的物理职责：
        FlowPINN      -> h, u, v
        SedimentPINN  -> C_tk
        BedPINN       -> z_b
        GradationPINN -> p_k

    该类不改变现有解耦训练流程，只提供一个统一入口，方便后续做全耦合训练。
    """

    def __init__(self, num_grain_classes=1, input_dim=3, hidden_dim=64,
                 num_block=4, zb_scale=1.5):
        super().__init__()
        self.flow_net = FlowPINN(
            input_dim=input_dim, hidden_dim=hidden_dim,
            num_block=num_block, output_dim=3
        )
        self.sediment_net = SedimentPINN(
            input_dim=input_dim, hidden_dim=hidden_dim,
            num_block=num_block, output_dim=num_grain_classes
        )
        self.bed_net = BedPINN(
            input_dim=input_dim, hidden_dim=hidden_dim,
            num_block=num_block, output_dim=1, zb_scale=zb_scale
        )
        self.gradation_net = GradationPINN(
            input_dim=input_dim, hidden_dim=hidden_dim,
            num_block=num_block, output_dim=num_grain_classes
        )

    def forward(self, xyt):
        flow = self.flow_net(xyt)
        return {
            'flow': flow,
            'h': flow[:, 0:1],
            'u': flow[:, 1:2],
            'v': flow[:, 2:3],
            'C_tk': self.sediment_net(xyt),
            'zb': self.bed_net(xyt),
            'p_k': self.gradation_net(xyt),
        }


# 浅水方程物理损失计算器：连续性方程 + 动量方程
# 连续性方程： ∂h/∂t + ∇·(hu, hv) = 0
# 动量方程： ∂(hu)/∂t + ∇·(hu² + 0.5gh², huv) = -gh∂zb/∂x + 摩擦项
class SVEsPhysicsLoss:
    def __init__(self, fvm_mesh, g=9.81, n_manning=0.01, bounds=None,
                 typical_depth=10.0, typical_velocity=1.0, eps=1e-6):
        self.mesh = fvm_mesh
        self.g = g
        self.n = n_manning
        self.bounds = bounds
        self.typical_h = typical_depth  # 典型水深 (m)
        self.typical_u = typical_velocity   
        self.eps = eps  # 数值稳定性小常数

        self.residual_scale = 1.0 / (fvm_mesh.cell_area * typical_depth * typical_velocity)
        print(f"\n初始化SWEs物理损失计算器")
        print(f"  g={g}, n={n_manning}")
        print(f"  典型水深: {typical_depth}m, 典型速度: {typical_velocity}m/s")
        print(f"  残差归一化因子: {self.residual_scale:.2e}")

    def compute_loss(self, model, t, device, zb_tensor=None):
        n_gauss = self.mesh.n_gauss_total
        # 获取河床高程
        if zb_tensor is None:
            zb_at_gauss = torch.tensor(
                self.mesh.get_bed_at_gauss_points(),
                dtype=torch.float32, device=device
            )
        else:
            zb_at_gauss = zb_tensor[self.mesh.gauss_cell_id]

        # 高斯点坐标、法向量、权重，并进行归一化处理
        gauss_coords = torch.tensor(
            self.mesh.gauss_coords, dtype=torch.float32, device=device
        )
        gauss_normals = torch.tensor(
            self.mesh.gauss_normals, dtype=torch.float32, device=device
        )
        gauss_weights = torch.tensor(
            self.mesh.gauss_weights, dtype=torch.float32, device=device
        ).unsqueeze(1)

        if self.bounds is not None:
            x_norm = (gauss_coords[:, 0:1] - self.bounds['x_min']) / (self.bounds['x_max'] - self.bounds['x_min'])
            y_norm = (gauss_coords[:, 1:2] - self.bounds['y_min']) / (self.bounds['y_max'] - self.bounds['y_min'])
        else:
            x_norm = gauss_coords[:, 0:1]
            y_norm = gauss_coords[:, 1:2]

        # t_tensor = torch.full((n_gauss, 1), t, device=device, dtype=torch.float32, requires_grad=True)
        t_tensor = torch.full((n_gauss, 1), t, device=device, dtype=torch.float32)
        xyt = torch.cat([x_norm, y_norm, t_tensor], dim=1)

        # 前向传播
        outputs = model(xyt)
        h_norm = outputs[:, 0:1]
        u_norm = outputs[:, 1:2]
        v_norm = outputs[:, 2:3]

        # 反归一化到物理单位
        h = h_norm * self.typical_h
        u = (u_norm - 0.5) * 2 * self.typical_u
        v = (v_norm - 0.5) * 2 * self.typical_u

        # 计算时间导数
        # h_t = torch.autograd.grad(h, t_tensor, torch.ones_like(h), create_graph=True, retain_graph=True)[0]
        # hu_t = torch.autograd.grad(h * u, t_tensor, torch.ones_like(h), create_graph=True, retain_graph=True)[0]
        # hv_t = torch.autograd.grad(h * v, t_tensor, torch.ones_like(h), create_graph=True, retain_graph=True)[0]

        # 法向量
        nx = gauss_normals[:, 0:1]
        ny = gauss_normals[:, 1:2]
        # 新加的
        Nc = self.mesh.n_cells  # 单元数量
        npp = self.mesh.n_points_per_cell   # 每个单元的高斯点数量
        weights_r = gauss_weights.view(Nc, npp) # 重整形为 [单元数量, 每个单元的高斯点数量]

        h_r = h.view(Nc, npp)   # 重整形为 [单元数量, 每个单元的高斯点数量]
        u_r = u.view(Nc, npp)
        v_r = v.view(Nc, npp)
        h_cell = torch.mean(h_r, dim=1, keepdim=True) # 每个单元内水深的平均值

        flux_h = (h * u) * nx + (h * v) * ny # 每个高斯点：连续方程中的通量项 hu·n = hu*nx + hv*ny
        bnd_h = torch.sum(flux_h.view(Nc, npp) * weights_r, dim=1, keepdim=True)    # 每个单元：边界积分 ∫(hu·n) dS ≈ Σ(hu·n * w) 其中 hu·n 已经重整形为 [Nc, npp]，权重也重整形为 [Nc, npp]，最后对每个单元的高斯点求和得到边界积分结果 [Nc, 1]
        loss_continuity = torch.mean((bnd_h * self.residual_scale) ** 2)

        dzb_dx, dzb_dy = self.mesh.get_bed_gradient()
        dzb_dx_t = torch.tensor(dzb_dx, dtype=torch.float32, device=device)
        dzb_dy_t = torch.tensor(dzb_dy, dtype=torch.float32, device=device)
        slope_x = -self.g * h_cell * dzb_dx_t.unsqueeze(1) * self.mesh.cell_area
        slope_y = -self.g * h_cell * dzb_dy_t.unsqueeze(1) * self.mesh.cell_area

        h_safe = torch.clamp(h_r, min=0.05)
        vel_mag = torch.sqrt(u_r ** 2 + v_r ** 2 + self.eps)
        fric_tx = torch.mean(self.g * self.n ** 2 * vel_mag * u_r / torch.pow(h_safe, 1. / 3.),
                             dim=1, keepdim=True) * self.mesh.cell_area
        fric_ty = torch.mean(self.g * self.n ** 2 * vel_mag * v_r / torch.pow(h_safe, 1. / 3.),
                             dim=1, keepdim=True) * self.mesh.cell_area

        flux_mx = (h * u * u + 0.5 * self.g * h * h) * nx + (h * u * v) * ny
        bnd_mx = torch.sum(flux_mx.view(Nc, npp) * weights_r, dim=1, keepdim=True)

        mom_x_res = (bnd_mx - slope_x + fric_tx) * self.residual_scale
        loss_momentum_x = torch.mean(mom_x_res ** 2)

        flux_my = (h * v * u) * nx + (h * v * v + 0.5 * self.g * h * h) * ny
        bnd_my = torch.sum(flux_my.view(Nc, npp) * weights_r, dim=1, keepdim=True)
        mom_y_res = (bnd_my - slope_y + fric_ty) * self.residual_scale
        loss_momentum_y = torch.mean(mom_y_res ** 2)

        total_loss = loss_continuity + loss_momentum_x + loss_momentum_y

        loss_dict = {
            'continuity': loss_continuity.item(),
            'momentum_x': loss_momentum_x.item(),
            'momentum_y': loss_momentum_y.item(),
            'total': total_loss.item()
        }

        return total_loss, loss_dict


class SedimentTransportLoss:
    """总输沙输移方程损失计算器。

    该类对应：
        SedimentPINN(x, y, t) -> C_1...C_K

    并调用 SedimentPINN.total_load_residual() 计算 HEC-RAS 原文总输沙
    输移方程残差。默认 beta 为 1，epsilon 为常数；E、D 由输沙能力浓度
    C*_tk 和适应时间 T_a 闭合，也可从外部传入更精细的 HEC-RAS 源汇项。
    """

    def __init__(self, fvm_mesh, bounds=None, typical_depth=10.0,
                 typical_velocity=1.0, beta_default=1.0,
                 epsilon_default=0.1, residual_scale=1.0,
                 grain_diameters=None, Ag=0.001, m=3,
                 adaptation_length=50.0, rho_s=2650.0,
                 alpha_active_layer=10.0, w_gradation=1.0,
                 w_capacity=0.05, source_sharpness=1e-3):
        self.mesh = fvm_mesh
        self.bounds = bounds
        self.typical_h = typical_depth
        self.typical_u = typical_velocity
        self.beta_default = beta_default
        self.epsilon_default = epsilon_default
        self.residual_scale = residual_scale
        self.grain_diameters = grain_diameters
        self.Ag = Ag
        self.m = m
        self.adaptation_length = adaptation_length
        self.rho_s = rho_s
        self.alpha_active_layer = alpha_active_layer
        self.w_gradation = w_gradation
        self.w_capacity = w_capacity
        self.source_sharpness = source_sharpness

    def _build_xyt_at_gauss(self, T_norm, device, requires_grad=True):
        """构造高斯点上的归一化 (x, y, t) 输入。"""
        gauss_coords = torch.tensor(self.mesh.gauss_coords, dtype=torch.float32, device=device)
        if self.bounds is not None:
            x_norm = (gauss_coords[:, 0:1] - self.bounds['x_min']) / (self.bounds['x_max'] - self.bounds['x_min'])
            y_norm = (gauss_coords[:, 1:2] - self.bounds['y_min']) / (self.bounds['y_max'] - self.bounds['y_min'])
        else:
            x_norm = gauss_coords[:, 0:1]
            y_norm = gauss_coords[:, 1:2]
        t_tensor = torch.full((self.mesh.n_gauss_total, 1), T_norm, device=device, dtype=torch.float32)
        xyt = torch.cat([x_norm, y_norm, t_tensor], dim=1)
        return xyt.requires_grad_(requires_grad)

    def _match_closure(self, value, like, default):
        """把常数/数组闭合量整理成和 C_tk 一致的 [N, K] 张量。"""
        if value is None:
            return torch.ones_like(like) * default
        if torch.is_tensor(value):
            value = value.to(dtype=like.dtype, device=like.device)
        else:
            value = torch.as_tensor(value, dtype=like.dtype, device=like.device)
        return value.expand_as(like) if value.numel() == 1 else value

    def _grain_diameters_like(self, like):
        """返回粒径向量 d_k，默认给所有粒径级同一个代表粒径。"""
        k = like.shape[1]
        if self.grain_diameters is None:
            d_k = torch.ones(k, dtype=like.dtype, device=like.device) * 2e-4
        else:
            d_k = torch.as_tensor(self.grain_diameters, dtype=like.dtype, device=like.device)
            if d_k.numel() != k:
                raise ValueError(f"grain_diameters 长度应为 K={k}，当前为 {d_k.numel()}。")
        return d_k

    def _smooth_positive(self, x):
        """平滑正部函数，避免 max(x,0) 在 0 点不可导。"""
        sharp = self.source_sharpness
        return F.softplus(x / sharp) * sharp

    def _capacity_closure(self, h, u, v, C_tk, p_k):
        """
        简化 HEC-RAS 式源汇闭合：
            |U| = sqrt(u^2+v^2)
            q*_tk = p_k A_g |U|^m
            C*_tk = q*_tk / (h |U|)
            S_k = E_k - D_k = h(C*_tk - C_tk) / T_a
            T_a = L_a / |U|

        其中 C*_tk 是输沙能力浓度，T_a 是适应时间；S_k>0 表示侵蚀，
        S_k<0 表示沉积。这里用 smooth_positive 分解为 E_k、D_k。
        """
        vel_mag = torch.sqrt(u ** 2 + v ** 2 + 1e-8)
        q_capacity = p_k * self.Ag * torch.pow(vel_mag, self.m)
        C_capacity = q_capacity / torch.clamp(h * vel_mag, min=1e-6)
        adapt_time = self.adaptation_length / torch.clamp(vel_mag, min=1e-3)
        net_source = h * (C_capacity - C_tk) / adapt_time
        E_tk = self._smooth_positive(net_source)
        D_tk = self._smooth_positive(-net_source)
        return C_capacity, E_tk, D_tk, vel_mag

    def compute_loss(self, sediment_model, flow_model, T_norm, device,
                     gradation_model=None, beta_tk=None, epsilon_thk=None,
                     E_tk=None, D_tk=None, freeze_flow_params=True):
        """计算总输沙 PDE 损失。

        参数:
            sediment_model: SedimentPINN，输出 C_tk
            flow_model: FlowPINN，输出归一化 h,u,v
            gradation_model: GradationPINN，输出 p_k；None 时各粒径等比例
            T_norm: 当前归一化时间
            beta_tk: 推移/悬移修正系数，None 时取常数 1
            epsilon_thk: 水平扩散系数，None 时取常数 epsilon_default
            E_tk, D_tk: 侵蚀、沉积源汇项，None 时由 C*_tk 适应公式闭合
            freeze_flow_params: 只冻结 FlowPINN 参数梯度，但保留 h,u,v 对 x,y,t
                的导数，使 ∇·(hUC) 仍按 PDE 正确求导。
        """
        xyt = self._build_xyt_at_gauss(T_norm, device, requires_grad=True)

        old_requires_grad = None
        if freeze_flow_params:
            old_requires_grad = [p.requires_grad for p in flow_model.parameters()]
            for p in flow_model.parameters():
                p.requires_grad_(False)
        try:
            flow_out = flow_model(xyt)
            h = flow_out[:, 0:1] * self.typical_h
            u = (flow_out[:, 1:2] - 0.5) * 2.0 * self.typical_u
            v = (flow_out[:, 2:3] - 0.5) * 2.0 * self.typical_u
        finally:
            if old_requires_grad is not None:
                for p, old_flag in zip(flow_model.parameters(), old_requires_grad):
                    p.requires_grad_(old_flag)

        C_tk = sediment_model(xyt)
        if gradation_model is None:
            p_k = torch.ones_like(C_tk) / C_tk.shape[1]
        else:
            p_k = gradation_model(xyt)

        beta_tk = self._match_closure(beta_tk, C_tk, self.beta_default)
        epsilon_thk = self._match_closure(epsilon_thk, C_tk, self.epsilon_default)
        if E_tk is None or D_tk is None:
            C_capacity, E_closure, D_closure, vel_mag = self._capacity_closure(h, u, v, C_tk, p_k)
            E_tk = E_closure if E_tk is None else self._match_closure(E_tk, C_tk, 0.0)
            D_tk = D_closure if D_tk is None else self._match_closure(D_tk, C_tk, 0.0)
        else:
            C_capacity = C_tk.detach()
            vel_mag = torch.sqrt(u ** 2 + v ** 2 + 1e-8)
            E_tk = self._match_closure(E_tk, C_tk, 0.0)
            D_tk = self._match_closure(D_tk, C_tk, 0.0)

        _, residual = SedimentPINN.total_load_loss(
            xyt, h, u, v, C_tk, beta_tk, epsilon_thk, E_tk, D_tk
        )
        transport_loss = torch.mean((residual * self.residual_scale) ** 2)
        capacity_loss = torch.mean((C_tk - C_capacity.detach()) ** 2)

        gradation_loss = torch.zeros((), dtype=C_tk.dtype, device=C_tk.device)
        if gradation_model is not None:
            d_k = self._grain_diameters_like(C_tk)
            L_a = GradationPINN.active_layer_thickness(
                d_k, p_k=p_k, alpha_a=self.alpha_active_layer
            )
            gradation_loss, _ = GradationPINN.active_layer_loss(
                xyt, p_k, L_a, E_tk, D_tk, rho_s=self.rho_s
            )

        loss = transport_loss + self.w_capacity * capacity_loss + self.w_gradation * gradation_loss
        return loss, {
            'transport': transport_loss.item(),
            'gradation': gradation_loss.item(),
            'capacity': capacity_loss.item(),
            'total': loss.item(),
            'residual_mean': torch.mean(torch.abs(residual)).item(),
            'C_min': torch.min(C_tk).item(),
            'C_max': torch.max(C_tk).item(),
            'Ceq_mean': torch.mean(C_capacity).item(),
            'p_min': torch.min(p_k).item(),
            'p_max': torch.max(p_k).item(),
            'U_mean': torch.mean(vel_mag).item(),
        }


# ExnerPhysicsLoss  自洽耦合 + IC loss  这里有修改 typical_u
class ExnerPhysicsLoss:

    def __init__(self, fvm_mesh, porosity=0.4, Ag=0.001, m=3, Q=10.0, h0=10.0,
                 bounds=None, typical_zb=1.0, T_physical=360000.0, eps=1e-6,
                 typical_u=1.0):
        """ 初始化Exner方程物理损失计算器
        参数: fvm_mesh - FVM网格
              porosity - 孔隙率
              Ag - Grass系数
              m - Grass指数
              Q - 单宽流量 (m²/s)
              h0 - 基准水深 (m)
              bounds - 边界条件
              typical_zb - 典型床面高程 (m)
              T_physical - 真实物理时间 (s)
              eps - 数值稳定性小常数
              typical_u - 典型流速 (m/s) 新增的参数，用于流速反归一化
        """
        self.mesh = fvm_mesh
        self.xi = 1.0 / (1.0 - porosity)
        self.Ag = Ag
        self.m = m
        self.Q = Q
        self.h0 = h0
        self.bounds = bounds
        self.typical_zb = typical_zb
        self.T_physical = T_physical
        self.eps = eps
        self.typical_u = typical_u 
        self.residual_scale = 1.0 / (fvm_mesh.cell_area * typical_zb)

        print(f"\n初始化Exner物理损失 (论文4.3参数)")
        print(f"  孔隙率ε: {porosity}, ξ={self.xi:.3f}")
        print(f"  Grass系数Ag: {Ag}, 指数m: {m}")
        print(f"  单宽流量Q: {Q} m²/s, 基准水深h0: {h0} m")
        print(f"  物理时间尺度: {T_physical}s = {T_physical / 3600:.1f}h")

    def compute_loss(self, bed_model, flow_model, T_norm, device, w_ic=10.0):
        n_gauss = self.mesh.n_gauss_total

        # 高斯点坐标
        gauss_coords = torch.tensor(self.mesh.gauss_coords, dtype=torch.float32, device=device)
        gauss_normals = torch.tensor(self.mesh.gauss_normals, dtype=torch.float32, device=device)
        gauss_weights = torch.tensor(self.mesh.gauss_weights, dtype=torch.float32, device=device).unsqueeze(1)

        # 归一化坐标
        if self.bounds is not None:
            x_norm = (gauss_coords[:, 0:1] - self.bounds['x_min']) / (self.bounds['x_max'] - self.bounds['x_min'])
            y_norm = (gauss_coords[:, 1:2] - self.bounds['y_min']) / (self.bounds['y_max'] - self.bounds['y_min'])
        else:
            x_norm = gauss_coords[:, 0:1]
            y_norm = gauss_coords[:, 1:2]

        # ① 用 FlowNet 预测当前床面下的稳态2D流场（解耦：no_grad）
        T_flow = torch.full((n_gauss, 1), T_norm, device=device, dtype=torch.float32)
        xyt_flow = torch.cat([x_norm, y_norm, T_flow], dim=1)

        with torch.no_grad():
            flow_out = flow_model(xyt_flow)  # Sigmoid输出 [0,1]
            h_pred = flow_out[:, 0:1] * self.h0  # 反归一化
            u_pred = (flow_out[:, 1:2] - 0.5) * 2.0 * self.typical_u
            v_pred = (flow_out[:, 2:3] - 0.5) * 2.0 * self.typical_u

        # ② BedNet 预测 zb(x,y,T)，梯度只走 BedNet
        T_tensor = torch.full((n_gauss, 1), T_norm, device=device,
                              dtype=torch.float32, requires_grad=True)
        xyT = torch.cat([x_norm, y_norm, T_tensor], dim=1)
        zb_pred = bed_model(xyT)  # (N_gauss, 1)

        # ③ ∂zb/∂T
        zb_t_norm = torch.autograd.grad(
            zb_pred, T_tensor, torch.ones_like(zb_pred),
            create_graph=True, retain_graph=True)[0]
        zb_t = zb_t_norm / self.T_physical

        # ④ 用 FlowNet 的流速驱动 Grass 输沙（自洽：h来自FlowNet）
        qx, qy, _ = SedimentPINN.grass_formula(
            u_pred, v_pred, self.Ag, self.m)


        # 转换为对真实物理时间的导数: ∂zb/∂t = (∂zb/∂T_norm) / T_physical
        zb_t = zb_t_norm / self.T_physical

        # 法向量
        nx = gauss_normals[:, 0:1]
        ny = gauss_normals[:, 1:2]

        # 泥沙通量
        flux_sediment = self.xi * (qx * nx + qy * ny)

        # 重整形
        flux_reshaped = flux_sediment.view(self.mesh.n_cells, self.mesh.n_points_per_cell)
        weights_reshaped = gauss_weights.view(self.mesh.n_cells, self.mesh.n_points_per_cell)
        zb_t_reshaped = zb_t.view(self.mesh.n_cells, self.mesh.n_points_per_cell)

        # FVM离散
        boundary_integral = torch.sum(flux_reshaped * weights_reshaped, dim=1, keepdim=True)
        volume_term = torch.mean(zb_t_reshaped, dim=1, keepdim=True) * self.mesh.cell_area

        # Exner残差: ∂zb/∂t + ξ∇·q = 0
        exner_residual = (volume_term + boundary_integral) * self.residual_scale
        loss_exner = torch.mean(exner_residual ** 2)

        # 这里新加 IC LOSS
        # IC loss: 强制 zb(x,y,T=0) = 初始沙丘
        cx = torch.tensor(self.mesh.cell_centers_x, dtype=torch.float32, device=device)
        cy = torch.tensor(self.mesh.cell_centers_y, dtype=torch.float32, device=device)
        if self.bounds is not None:
            cx = (cx - self.bounds['x_min']) / (self.bounds['x_max'] - self.bounds['x_min'])
            cy = (cy - self.bounds['y_min']) / (self.bounds['y_max'] - self.bounds['y_min'])
        T0 = torch.zeros(self.mesh.n_cells, dtype=torch.float32, device=device)
        xyT0 = torch.stack([cx, cy, T0], dim=1)
        zb_ic_pred = bed_model(xyT0)
        zb_ic_true = torch.tensor(self.mesh.zb_initial, dtype=torch.float32, device=device).unsqueeze(1)
        loss_ic = torch.mean((zb_ic_pred - zb_ic_true) ** 2)

        # 原来只返回 loss_exner，现在加权求he
        total_loss = loss_exner + w_ic * loss_ic

        # return loss_exner, loss_dict
        return total_loss, {
            'exner': loss_exner.item(), 'ic': loss_ic.item(),
            'total': total_loss.item(), 't_physical': T_norm * self.T_physical
        }


# 解耦训练循环
class DecoupledTrainer:
    def __init__(self, flow_model, bed_model, fvm_mesh, device,
                 flow_loss_fn, sediment_loss_fn,
                 sediment_model=None, gradation_model=None,
                 sediment_transport_loss_fn=None,
                 flow_lr=1e-4, sediment_lr=1e-4,
                 transport_lr=1e-4):
        self.flow_model = flow_model
        self.bed_model = bed_model
        self.sediment_model = sediment_model
        self.gradation_model = gradation_model
        self.mesh = fvm_mesh
        self.device = device
        self.flow_loss_fn = flow_loss_fn
        self.sediment_loss_fn = sediment_loss_fn
        self.sediment_transport_loss_fn = sediment_transport_loss_fn
        self.last_delta = np.zeros(self.mesh.n_cells, dtype=np.float32)  # 新加

        self.flow_optimizer = torch.optim.Adam(flow_model.parameters(), lr=flow_lr)
        self.sediment_optimizer = torch.optim.Adam(bed_model.parameters(), lr=sediment_lr)
        transport_params = []
        if sediment_model is not None:
            transport_params += list(sediment_model.parameters())
        if gradation_model is not None:
            transport_params += list(gradation_model.parameters())
        self.transport_optimizer = (
            torch.optim.Adam(transport_params, lr=transport_lr)
            if transport_params else None
        )

        # 学习率调度器
        self.flow_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.flow_optimizer, 'min', factor=0.5, patience=500
        )
        self.sediment_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.sediment_optimizer, 'min', factor=0.5, patience=500
        )
        self.transport_scheduler = (
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.transport_optimizer, 'min', factor=0.5, patience=500
            )
            if self.transport_optimizer is not None else None
        )

        # 训练历史
        self.history = {
            'flow_loss': [],
            'sediment_loss': [],
            'transport_loss': [],
            'gradation_loss': [],
            'capacity_loss': [],
            'continuity': [],
            'momentum_x': [],
            'momentum_y': [],
            'exner': [],
            'zb_min': [], 'zb_max': [], 'ic': [],
            'C_min': [], 'C_max': [], 'p_min': [], 'p_max': []
        }   # 新加的  ic

    def train_flow_phase(self, n_epochs, T_norm, data_coords=None, data_values=None):
        self.flow_model.train()
        for epoch in range(n_epochs):
            self.flow_optimizer.zero_grad()

            # 随机时间点
            # t = np.random.rand()

            # 物理损失
            physics_loss, loss_dict = self.flow_loss_fn.compute_loss(
                self.flow_model, T_norm, self.device
            )
            # 数据损失 (如果有)
            if data_coords is not None:
                data_loss = self._compute_flow_data_loss(data_coords, data_values)
                total_loss = physics_loss + 0.5 * data_loss
            else:
                total_loss = physics_loss
            # 反向传播
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.flow_model.parameters(), max_norm=1.0)
            self.flow_optimizer.step()
            self.flow_scheduler.step(total_loss)

            # 记录历史
            self.history['flow_loss'].append(total_loss.item())
            self.history['continuity'].append(loss_dict['continuity'])
            self.history['momentum_x'].append(loss_dict['momentum_x'])
            self.history['momentum_y'].append(loss_dict['momentum_y'])

        return total_loss.item()

    def train_sediment_phase(self, n_epochs, T, w_ic=10.0):
        self.bed_model.train()
        if self.sediment_model is not None:
            self.sediment_model.train()
        if self.gradation_model is not None:
            self.gradation_model.train()
        self.flow_model.eval()  # 流场模型设为评估模式

        for epoch in range(n_epochs):
            self.sediment_optimizer.zero_grad()
            if self.transport_optimizer is not None:
                self.transport_optimizer.zero_grad()

            # 床面 Exner 损失：BedPINN -> z_b
            bed_loss, loss_dict = self.sediment_loss_fn.compute_loss(
                self.bed_model, self.flow_model, T, self.device, w_ic=w_ic
            )
            total_loss = bed_loss
            transport_dict = None

            # 总输沙 + 活动层级配损失：SedimentPINN -> C_tk, GradationPINN -> p_k
            if (self.sediment_transport_loss_fn is not None
                    and self.sediment_model is not None):
                transport_loss, transport_dict = self.sediment_transport_loss_fn.compute_loss(
                    self.sediment_model, self.flow_model, T, self.device,
                    gradation_model=self.gradation_model
                )
                total_loss = total_loss + transport_loss

            # 反向传播
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.bed_model.parameters(), max_norm=1.0)
            if self.sediment_model is not None:
                torch.nn.utils.clip_grad_norm_(self.sediment_model.parameters(), max_norm=1.0)
            if self.gradation_model is not None:
                torch.nn.utils.clip_grad_norm_(self.gradation_model.parameters(), max_norm=1.0)
            self.sediment_optimizer.step()
            if self.transport_optimizer is not None:
                self.transport_optimizer.step()
            self.sediment_scheduler.step(bed_loss.detach())
            if self.transport_scheduler is not None and transport_dict is not None:
                self.transport_scheduler.step(transport_loss.detach())

            # 记录历史
            self.history['sediment_loss'].append(total_loss.item())
            self.history['exner'].append(loss_dict['exner'])
            self.history['ic'].append(loss_dict['ic'])  # 加这行
            if transport_dict is not None:
                self.history['transport_loss'].append(transport_dict['transport'])
                self.history['gradation_loss'].append(transport_dict['gradation'])
                self.history['capacity_loss'].append(transport_dict['capacity'])
                self.history['C_min'].append(transport_dict['C_min'])
                self.history['C_max'].append(transport_dict['C_max'])
                self.history['p_min'].append(transport_dict['p_min'])
                self.history['p_max'].append(transport_dict['p_max'])

        return total_loss.item()

    def update_bed_from_model(self, T):
        self.bed_model.eval()

        with torch.no_grad():
            # 在单元中心评估河床
            x_centers = torch.tensor(self.mesh.cell_centers_x, dtype=torch.float32, device=self.device)
            y_centers = torch.tensor(self.mesh.cell_centers_y, dtype=torch.float32, device=self.device)
            T_tensor = torch.full_like(x_centers, T)

            # 归一化
            if self.flow_loss_fn.bounds is not None:
                bounds = self.flow_loss_fn.bounds
                x_norm = (x_centers - bounds['x_min']) / (bounds['x_max'] - bounds['x_min'])
                y_norm = (y_centers - bounds['y_min']) / (bounds['y_max'] - bounds['y_min'])
            else:
                x_norm = x_centers
                y_norm = y_centers

            xyT = torch.stack([x_norm, y_norm, T_tensor], dim=1)
            zb_new = self.bed_model(xyT).squeeze().cpu().numpy()
        # delta_increment = delta_zb - self.last_delta
        # # 基于当前网格累加增量（而不是基于 zb_initial）
        # zb_new = self.mesh.zb + delta_increment
        # # 更新记录
        # self.last_delta = delta_zb.copy()
        # 改成（SedNet 直接输出绝对 zb）
        self.mesh.update_bed(zb_new)
        return zb_new

    def pretrain_ic(self, n_epochs):
        """[新增] Warm-up: 先让 BedNet 认识初始沙丘形状"""
        print(f"\n[Warm-up] 预训练IC ({n_epochs} epochs)...")
        self.bed_model.train()
        cx = torch.tensor(self.mesh.cell_centers_x, dtype=torch.float32, device=self.device)
        cy = torch.tensor(self.mesh.cell_centers_y, dtype=torch.float32, device=self.device)
        b = self.flow_loss_fn.bounds
        if b:
            cx = (cx - b['x_min']) / (b['x_max'] - b['x_min'])
            cy = (cy - b['y_min']) / (b['y_max'] - b['y_min'])
        T0 = torch.zeros(self.mesh.n_cells, dtype=torch.float32, device=self.device)
        xyT0 = torch.stack([cx, cy, T0], dim=1)
        zb_t = torch.tensor(self.mesh.zb_initial, dtype=torch.float32,
                            device=self.device).unsqueeze(1)    # 真实初始河床高程
        opt = torch.optim.Adam(self.bed_model.parameters(), lr=5e-4)
        for ep in range(n_epochs):
            opt.zero_grad()
            loss = F.mse_loss(self.bed_model(xyT0), zb_t)
            loss.backward()
            opt.step()
            if ep % 200 == 0:
                print(f"  ep={ep:4d}  IC_loss={loss.item():.4e}")
        print(f"  Warm-up完成  IC_loss={loss.item():.4e}\n")

    def _compute_flow_data_loss(self, coords, values):

        coords_tensor = torch.tensor(coords, dtype=torch.float32, device=self.device)
        targets = torch.tensor(values, dtype=torch.float32, device=self.device)

        predictions = self.flow_model(coords_tensor)

        return F.mse_loss(predictions, targets)

    def run_decoupled_training(self, n_macro_steps, flow_epochs_per_step,
                               sediment_epochs_per_step,  bc_coords=None, bc_values=None,   warmup_ic_epochs=600, verbose=True):
        """
        参数:
            n_macro_steps: 大时间步数
            flow_epochs_per_step: 每个大时间步的水流训练轮数
            sediment_epochs_per_step: 每个大时间步的泥沙训练轮数
        """
        print("\n" + "=" * 60)
        print(" 开始解耦训练")
        print("=" * 60)
        print(f"  大时间步数: {n_macro_steps}")
        print(f"  每步水流训练: {flow_epochs_per_step} epochs")
        print(f"  每步床面/泥沙/级配训练: {sediment_epochs_per_step} epochs")
        if self.sediment_transport_loss_fn is None or self.sediment_model is None:
            print("  注意: 未提供 SedimentTransportLoss 或 SedimentPINN，仅训练 FlowPINN + BedPINN。")

        self.pretrain_ic(warmup_ic_epochs)  # 对床面模型进行IC预训练，在正式的解耦训练前先让BedNet认识初始沙丘形状
        bed_history = [self.mesh.zb.copy()]

        for T_step in trange(n_macro_steps, desc="Macro Steps"):
            T_norm = T_step / n_macro_steps

            # Phase 1: 训练水流
            if verbose and T_step % 10 == 0:
                print(f"\n  T={T_step}: 训练水流模型...")
            # flow_loss = self.train_flow_phase(flow_epochs_per_step, data_coords=bc_coords,
            #                               data_values=bc_values)
            flow_loss = self.train_flow_phase(flow_epochs_per_step, T_norm, data_coords=bc_coords, data_values=bc_values)  # 传T_norm

            # Phase 2: 训练泥沙
            if verbose and T_step % 10 == 0:
                print(f"  T={T_step}: 训练床面/总输沙/级配模型...")
            # sediment_loss = self.train_sediment_phase(sediment_epochs_per_step, T_norm)
            # sediment_loss = self.train_sediment_phase(sediment_epochs_per_step, T_norm,
            #                                           w_ic=max(3.0, 20.0 * (1 - T_norm)))
            sediment_loss = self.train_sediment_phase(sediment_epochs_per_step, T_norm,
                                                      w_ic=max(5.0, 30.0 * (1 - T_norm)))


            # Phase 3: 更新河床
            current_bed = self.update_bed_from_model(T_norm)
            bed_history.append(current_bed.copy())

            # 记录历史
            self.history['zb_min'].append(np.min(current_bed))
            self.history['zb_max'].append(np.max(current_bed))

            if verbose and T_step % 10 == 0:
                print(f"\n  T={T_step}: Flow Loss={flow_loss:.2e}, Bed+Sed Loss={sediment_loss:.2e}")
                print(f"    河床: [{np.min(current_bed):.4f}, {np.max(current_bed):.4f}]")
        print("\n✓ 解耦训练完成")

        return bed_history


# 测试案例  初始床面 可视化 主函数
def hump_initial_bed(x, y):
    """
       Z(0,x,y) = sin²((x-500)π/200) * sin²((y-400)π/200)  if (x,y) ∈ [500,700]×[400,600]
                = 0                                          otherwise
       """
    # 沙丘区域 Ω = [500, 700] × [400, 600]
    in_hump = ((x >= 500) & (x <= 700) & (y >= 400) & (y <= 600))

    zb = np.zeros_like(x)
    zb[in_hump] = (
            np.sin(np.pi * (x[in_hump] - 500) / 200) ** 2 *
            np.sin(np.pi * (y[in_hump] - 400) / 200) ** 2
    )

    return zb


def visualize_results(mesh, bed_history, bbox, resolution, history,
                      T_physical, Ag, regime='fast'):
    if not HAS_MATPLOTLIB:
        print("\n未安装 matplotlib，跳过结果绘图；训练历史和 bed_history 仍会返回。")
        return

    nx = int((bbox['xmax']-bbox['xmin'])/resolution)
    ny = int((bbox['ymax']-bbox['ymin'])/resolution)
    xc = np.linspace(bbox['xmin']+resolution/2, bbox['xmax']-resolution/2, nx)
    yc = np.linspace(bbox['ymin']+resolution/2, bbox['ymax']-resolution/2, ny)
    X, Y = np.meshgrid(xc, yc)
    n_t  = len(bed_history)
    t_u  = 'h' if T_physical>3600 else 's'
    t_sc = 3600.0 if T_physical>3600 else 1.0
    tids = np.linspace(0, n_t-1, 6, dtype=int)

    # 床面演化
    fig, axes = plt.subplots(2, 3, figsize=(18,11))
    for ax, tid in zip(axes.flatten(), tids):
        zb  = bed_history[tid].reshape(ny,nx)
        t_v = (tid/max(n_t-1,1))*T_physical/t_sc
        lv  = np.linspace(min(zb.min()-0.01,-0.05), max(zb.max()+0.01,0.1), 25)
        im  = ax.contourf(X, Y, zb, levels=lv, cmap='terrain')
        ax.contour(X, Y, zb, levels=5, colors='k', linewidths=0.4)
        ax.set_title(f't={t_v:.1f}{t_u} max={zb.max():.3f}m'); ax.set_aspect('equal')
        ax.set_xlabel('x(m)'); ax.set_ylabel('y(m)')
        plt.colorbar(im, ax=ax)
        ax.plot([500,700,700,500,500],[400,400,600,600,400],'r--',lw=1,alpha=0.5)
    plt.suptitle(f'床面演化 A={Ag}', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'{regime}_bed_slow2.png', dpi=150, bbox_inches='tight'); plt.close()
    print(f"\n✓ {regime}_bed.png")

    # 剖面对比
    fig, axes = plt.subplots(1, 2, figsize=(14,5))
    j500=np.argmin(np.abs(yc-500)); i600=np.argmin(np.abs(xc-600))
    colors = plt.cm.plasma(np.linspace(0,1,len(tids)))
    for c,tid in zip(colors,tids):
        zb=bed_history[tid].reshape(ny,nx)
        t_v=(tid/max(n_t-1,1))*T_physical/t_sc
        axes[0].plot(xc,zb[j500,:],color=c,lw=2,label=f't={t_v:.1f}{t_u}')
        axes[1].plot(yc,zb[:,i600],color=c,lw=2,label=f't={t_v:.1f}{t_u}')
    for ax,xl,lb in zip(axes,[(300,800),(300,700)],
                        ['y=500m 中心线 (对照论文Fig.7/12)','x=600m 横断面']):
        ax.set_xlim(xl); ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_ylabel('zb(m)')
        ax.set_title(f'{lb}  A={Ag}')
    axes[0].set_xlabel('x(m)'); axes[1].set_xlabel('y(m)')
    plt.tight_layout()
    plt.savefig(f'{regime}_profiles_slow2.png', dpi=150, bbox_inches='tight'); plt.close()
    print(f"✓ {regime}_profiles.png")

    # Loss曲线
    fig, axes = plt.subplots(2, 2, figsize=(13,8))
    def pl(ax,d,lb,c,ls='-'):
        if d: ax.semilogy(d,color=c,lw=1.5,ls=ls,label=lb)
    pl(axes[0,0],history['flow_loss'],'Flow','b')
    pl(axes[0,0],history['continuity'],'Cont','c','--')
    axes[0,0].set_title('Flow Loss (准稳态)'); axes[0,0].legend(); axes[0,0].grid(alpha=0.3)
    pl(axes[0,1],history['momentum_x'],'Mom-x','r')
    pl(axes[0,1],history['momentum_y'],'Mom-y','m')
    axes[0,1].set_title('Momentum'); axes[0,1].legend(); axes[0,1].grid(alpha=0.3)
    pl(axes[1,0],history['sediment_loss'],'Bed+Sed','g')
    pl(axes[1,0],history['exner'],'Exner','olive','--')
    pl(axes[1,0],history['ic'], 'IC', 'orange', '--')
    pl(axes[1,0],history.get('transport_loss', []),'C PDE','teal',':')
    pl(axes[1,0],history.get('gradation_loss', []),'p PDE','brown',':')
    axes[1,0].set_title('Sediment Loss (Exner+C+p)'); axes[1,0].legend(); axes[1,0].grid(alpha=0.3)
    if history['zb_max']:
        ta=np.linspace(0,T_physical/t_sc,len(history['zb_max']))
        axes[1,1].plot(ta,history['zb_max'],'g-',lw=2,label='max')
        axes[1,1].plot(ta,history['zb_min'],'r-',lw=2,label='min')
        axes[1,1].set_xlabel(f'Time({t_u})'); axes[1,1].set_ylabel('zb(m)')
        axes[1,1].legend(); axes[1,1].grid(alpha=0.3)
    plt.suptitle(f'训练历史 A={Ag}')
    plt.tight_layout()
    plt.savefig(f'{regime}_losses_slow2.png', dpi=150, bbox_inches='tight'); plt.close()
    print(f"✓ {regime}_losses.png")

    # 扩散角 (slow case)
    if Ag < 0.01:
        zb_f = bed_history[-1].reshape(ny,nx)
        fig, ax = plt.subplots(figsize=(7,7))
        ax.contourf(xc,yc,zb_f,levels=20,cmap='terrain')
        ax.contour(xc,yc,zb_f,levels=[zb_f.max()*0.05],colors='red',linewidths=2)
        theta = np.degrees(np.arctan(3*np.sqrt(3)*2/26))
        for s in [1,-1]:
            dx=np.linspace(0,400,200)
            ax.plot(600+dx,500+s*dx*np.tan(np.radians(theta)),'w--',lw=2,
                    label=f'Theory {theta:.1f}°')
        ax.set_title(f'扩散角 理论={theta:.2f}° A={Ag}')
        ax.set_aspect('equal'); ax.legend()
        plt.tight_layout()
        plt.savefig(f'{regime}_angle.png',dpi=150,bbox_inches='tight'); plt.close()
        print(f"✓ {regime}_angle.png (理论扩散角={theta:.2f}°)")

    dz = bed_history[-1].max()-bed_history[0].max()
    print(f"\n  初始峰值: {bed_history[0].max():.4f}m  最终峰值: {bed_history[-1].max():.4f}m")
    print(f"  峰值变化: {dz:+.4f}m ({dz/max(bed_history[0].max(),1e-6)*100:.1f}%)")


def run_hump_evolution_test(regime='fast'):
    print("\n"+"="*70)
    print(f" 沙丘演变测试 — {regime.upper()} (修正版)")
    print("="*70)

    Ag         = 0.001    if regime=='slow' else 1.0    # 泥沙扩散系数
    T_physical = 360000.0 if regime=='slow' else 600.0  # 物理时间尺度

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Ag={Ag}, T={T_physical}s, device={device}")

    bbox = {'xmin':0,'xmax':1000,'ymin':0,'ymax':1000}  # 计算域边界
    fvm_mesh = FVMeshPreprocessor(bbox=bbox, 
                                  resolution=25.0,  # 网格分辨率
                                  initial_bed=hump_initial_bed, # 初始床面
                                  n_gauss_points=2,  # 高斯点数
                                  )

    flow_model = FlowPINN(input_dim=3,  # 输入 (x,y,t)
                          hidden_dim=64,    # 隐藏层维度
                          num_block=4,  # 隐藏层数
                          output_dim=3,  # 输出 (h,u,v)
                          ).to(device)
    
    bed_model = BedPINN(input_dim=3,  # 输入 (x,y,t)
                        hidden_dim=64,    # 隐藏层维度
                        num_block=4,  # 隐藏层数
                        output_dim=1,  # 输出 (zb)
                        zb_scale=1.5).to(device)  # 输出缩放，帮助训练稳定

    num_grain_classes = 2
    grain_diameters = [2e-4, 5e-4]  # 两个代表粒径级 d_k (m)，可按实测级配替换
    sediment_model = SedimentPINN(input_dim=3,  # 输入 (x,y,t)
                                  hidden_dim=64,
                                  num_block=4,
                                  output_dim=num_grain_classes,  # 输出 C_tk
                                  positive_output=True).to(device)
    gradation_model = GradationPINN(input_dim=3,  # 输入 (x,y,t)
                                    hidden_dim=64,
                                    num_block=4,
                                    output_dim=num_grain_classes).to(device)

    bounds = {'x_min':0,'x_max':1000,'y_min':0,'y_max':1000}    # 归一化边界

    flow_loss_fn = SVEsPhysicsLoss(fvm_mesh=fvm_mesh,   # 网格
                                   g=9.81,  # 重力加速度
                                   n_manning=0.01,  # Manning糙率
                                   bounds=bounds,   # 归一化边界
                                   typical_depth=10.0)  # 水深尺度，帮助损失权重平衡
    
    sediment_loss_fn = ExnerPhysicsLoss(fvm_mesh=fvm_mesh, # 网格
                                        porosity=0.4,   # 孔隙率 
                                        Ag=Ag,  # Grass系数
                                        m=3,    # Grass指数
                                        Q=10.0, # 单宽流量 (m²/s)
                                        h0=10.0,# 基准水深 (m)
                                        T_physical=T_physical,  # 物理时间尺度 (s)
                                        bounds=bounds,  # 归一化边界
                                        typical_u=1.0   # 速度尺度，帮助损失权重平衡
                                        )  

    sediment_transport_loss_fn = SedimentTransportLoss(
        fvm_mesh=fvm_mesh,
        bounds=bounds,
        typical_depth=10.0,
        typical_velocity=1.0,
        beta_default=1.0,
        epsilon_default=0.1,
        residual_scale=1.0,
        grain_diameters=grain_diameters,
        Ag=Ag,
        m=3,
        adaptation_length=50.0,
        alpha_active_layer=10.0,
        w_gradation=1.0,
        w_capacity=0.05
    )

    trainer = DecoupledTrainer(flow_model=flow_model,   # 水流模型
                               bed_model=bed_model,   # 床面模型
                               fvm_mesh=fvm_mesh,   # 网格预处理器
                               device=device,   # 计算设备
                               flow_loss_fn=flow_loss_fn,   # 水流损失函数
                               sediment_loss_fn=sediment_loss_fn,   # 泥沙损失函数
                               sediment_model=sediment_model,  # 总输沙浓度模型 C_tk
                               gradation_model=gradation_model,  # 活动层级配模型 p_k
                               sediment_transport_loss_fn=sediment_transport_loss_fn,
                               flow_lr=1e-3,    # 水流学习率
                               sediment_lr=1e-3, # 床面学习率
                               transport_lr=1e-3 # C_tk 与 p_k 学习率
                               )

    # 这里修改
    # n_bc=50
    # y_bc=np.linspace(0,1000,n_bc)
    # bc_coords=np.stack([np.zeros(n_bc)/1000, y_bc/1000, np.ones(n_bc)*0.5],axis=1).astype(np.float32)
    # bc_values=np.array([[1.0, 1.0, 0.5]]*n_bc, dtype=np.float32)

    n_bc = 50
    # 入口 x=0：h=10m(归一化=1.0), u=Q/h=1m/s(归一化=1.0), v=0(归一化=0.5)
    y_inlet = np.linspace(0, 1000, n_bc)
    # 构建坐标矩阵（归一化）: x=0 → 0, y/1000, t=0.5 (代表时刻)
    coords_inlet = np.stack([
        np.zeros(n_bc),
        y_inlet / 1000.0,   
        np.ones(n_bc) * 0.5
    ], axis=1).astype(np.float32)
    # 构建对应的目标值矩阵: h=1.0, u=1.0, v=0.5 (归一化值)
    values_inlet = np.array([[1.0, 1.0, 0.5]] * n_bc, dtype=np.float32)
    # h_norm=1.0 → h=10m; u_norm=1.0 → u=(1-0.5)*2*1=1m/s; v_norm=0.5 → v=0

    # 下壁 y=0：v=0
    x_wall = np.linspace(0, 1000, n_bc)
    # 构建坐标矩阵（归一化）: x/1000, y=0, t=0.5 (代表时刻)
    coords_wall_bot = np.stack([
        x_wall / 1000.0,
        np.zeros(n_bc),
        np.ones(n_bc) * 0.5
    ], axis=1).astype(np.float32)
    # 构建对应的目标值矩阵: h=1.0, u=1.0, v=0.5 (归一化值)
    values_wall_bot = np.array([[1.0, 1.0, 0.5]] * n_bc, dtype=np.float32)
    # v_norm=0.5 → v=0，h和u用网络自己学，这里给个合理初始值

    # 上壁 y=1000：v=0
    coords_wall_top = np.stack([
        x_wall / 1000.0,
        np.ones(n_bc),
        np.ones(n_bc) * 0.5
    ], axis=1).astype(np.float32)
    values_wall_top = values_wall_bot.copy()

    # 合并所有BC
    bc_coords = np.concatenate([coords_inlet, coords_wall_bot, coords_wall_top], axis=0)
    bc_values = np.concatenate([values_inlet, values_wall_bot, values_wall_top], axis=0)


    # 轮次: GPU ~25-30min / CPU ~20min
    USE_GPU = torch.cuda.is_available()
    N_STEPS = 200 if USE_GPU else 100
    FLOW_EP = 300 if USE_GPU else 150    # 原来200→300 (准稳态需更多epoch)
    SED_EP  = 400 if USE_GPU else 200    # 原来300→400 (真实物理损失)
    W_EP    = 800 if USE_GPU else 400    # 新增 warm-up

    bed_history = trainer.run_decoupled_training(
        n_macro_steps=N_STEPS,  # 大时间步数
        flow_epochs_per_step=FLOW_EP,   # 每个大时间步的水流训练轮数
        sediment_epochs_per_step=SED_EP,    # 每个大时间步的泥沙训练轮数
        bc_coords=bc_coords, bc_values=bc_values,   # 边界条件数据
        warmup_ic_epochs=W_EP,  # 新增 warm-up 轮数
        verbose=True)   # 是否打印训练进度

    visualize_results(fvm_mesh, bed_history, bbox, 25.0,
                      trainer.history, T_physical, Ag, regime)
    return trainer, bed_history


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print(" PINN+FVM 二维水沙耦合方程求解器")
    print(" (解耦方法 - Quasi-steady Approach)")
    print("=" * 70)

    # 运行沙丘演变测试
    # trainer, bed_history = run_hump_evolution_test()
    #
    # plot_training_losses(trainer.history)

    # regime = sys.argv[1] if len(sys.argv) > 1 else 'fast'

    regime = sys.argv[1] if len(sys.argv) > 1 else 'slow'
    run_hump_evolution_test(regime=regime)
