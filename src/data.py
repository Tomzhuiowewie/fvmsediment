
import numpy as np
import torch

from .model import FlowPINN


def hump_initial_bed(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """hump 算例初始床面：在 [500,700]×[400,600] 区域生成正弦形凸起"""
    in_hump = ((x >= 500) & (x <= 700) & (y >= 400) & (y <= 600))
    zb = np.zeros_like(x)
    zb[in_hump] = (
        np.sin(np.pi * (x[in_hump] - 500) / 200) ** 2
        * np.sin(np.pi * (y[in_hump] - 400) / 200) ** 2
    )
    return zb


def build_boundary_conditions(
    t_norm,
    bbox,
    bounds,
    bc_default,
    typical_depth,
    typical_velocity,
    n_bc=None,
):
    if n_bc is None:
        n_bc = int(bc_default['n_bc'])

    h_bc = bc_default['h']
    u_bc = bc_default['u']
    v_bc = bc_default['v']
    bc_value = FlowPINN.encode_target(
        torch.tensor([[h_bc]], dtype=torch.float32),
        torch.tensor([[u_bc]], dtype=torch.float32),
        torch.tensor([[v_bc]], dtype=torch.float32),
        typical_depth,
        typical_velocity,
    ).numpy().astype(np.float32)

    x_min = bbox['xmin']
    x_max = bbox['xmax']
    y_min = bbox['ymin']
    y_max = bbox['ymax']
    x_scale = bounds['x_max'] - bounds['x_min']
    y_scale = bounds['y_max'] - bounds['y_min']
    if x_scale == 0 or y_scale == 0:
        raise ValueError("bounds 的 x/y 范围不能为 0。")

    x_left_norm = (x_min - bounds['x_min']) / x_scale
    y_bottom_norm = (y_min - bounds['y_min']) / y_scale
    y_top_norm = (y_max - bounds['y_min']) / y_scale

    # 入口边界条件：在 x=x_min 处，h, u, v 都有约束
    y_inlet = np.linspace(y_min, y_max, n_bc)
    y_inlet_norm = (y_inlet - bounds['y_min']) / y_scale
    coords_inlet = np.stack([np.full(n_bc, x_left_norm), y_inlet_norm, np.ones(n_bc) * t_norm], axis=1).astype(np.float32)
    values_inlet = np.repeat(bc_value, n_bc, axis=0)
    mask_inlet = np.array([[1.0, 1.0, 1.0]] * n_bc, dtype=np.float32)   # 入口边界条件：h, u, v 都有约束

    # 壁面边界条件：在 y=y_min 和 y=y_max 处，只有 v 有约束（无穿透），h 和 u 沿壁面自由变化
    x_wall = np.linspace(x_min, x_max, n_bc)
    x_wall_norm = (x_wall - bounds['x_min']) / x_scale
    coords_wall_bot = np.stack([x_wall_norm, np.full(n_bc, y_bottom_norm), np.ones(n_bc) * t_norm], axis=1).astype(np.float32)
    coords_wall_top = np.stack([x_wall_norm, np.full(n_bc, y_top_norm), np.ones(n_bc) * t_norm], axis=1).astype(np.float32)
    values_wall = np.repeat(bc_value, n_bc, axis=0)
    mask_wall = np.array([[0.0, 0.0, 1.0]] * n_bc, dtype=np.float32)

    bc_coords = np.concatenate([coords_inlet, coords_wall_bot, coords_wall_top], axis=0)
    bc_values = np.concatenate([values_inlet, values_wall, values_wall], axis=0)
    bc_mask = np.concatenate([mask_inlet, mask_wall, mask_wall], axis=0)
    return bc_coords, bc_values, bc_mask


class FVMeshPreprocessor:
    def __init__(self, bbox, resolution, initial_bed=None, n_gauss_points=2):
        self.bbox = bbox
        self.resolution = resolution
        self.n_gauss_points = n_gauss_points
        self._generate_mesh()
        self._initialize_bed(initial_bed)   # 初始化床面高程数据
        self._setup_gauss_quadrature()  # 设置高斯积分点位置和权重
        self._precompute_edge_data()    # 预计算边界积分相关数据

    def _generate_mesh(self):
        xmin, xmax = self.bbox['xmin'], self.bbox['xmax']
        ymin, ymax = self.bbox['ymin'], self.bbox['ymax']
        x_length = xmax - xmin
        y_length = ymax - ymin
        
        # 网格数量
        self.nx = int(round(x_length / self.resolution))
        self.ny = int(round(y_length / self.resolution))
        
        # 格心坐标
        x_centers = xmin + (np.arange(self.nx) + 0.5) * self.resolution
        y_centers = ymin + (np.arange(self.ny) + 0.5) * self.resolution
        
        # 生成网格格心坐标，并展平为一维数组
        self.cell_centers_x, self.cell_centers_y = np.meshgrid(x_centers, y_centers)
        self.cell_centers_x = self.cell_centers_x.flatten()
        self.cell_centers_y = self.cell_centers_y.flatten()
        
        # 网格总单元数和单元面积
        self.n_cells = len(self.cell_centers_x)
        self.cell_area = self.resolution ** 2
        self.cell_index = np.arange(self.n_cells).reshape(self.ny, self.nx)

    def _initialize_bed(self, initial_bed):
        if initial_bed is None:
            self.zb = np.zeros(self.n_cells)
        elif callable(initial_bed):
            self.zb = initial_bed(self.cell_centers_x, self.cell_centers_y)
        else:
            self.zb = np.full(self.n_cells, initial_bed)
        self.zb_initial = self.zb.copy()

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
        self.n_points_per_cell = n_edges_per_cell * self.n_gauss_points
        self.n_gauss_total = self.n_cells * self.n_points_per_cell
        self.gauss_coords = np.zeros((self.n_gauss_total, 2))
        self.gauss_normals = np.zeros((self.n_gauss_total, 2))
        self.gauss_weights = np.zeros(self.n_gauss_total)
        self.gauss_cell_id = np.zeros(self.n_gauss_total, dtype=int)
        self.gauss_edge_id = np.zeros(self.n_gauss_total, dtype=int)
        self.gauss_neighbor_id = np.full(self.n_gauss_total, -1, dtype=int)
        idx = 0
        for cell_i in range(self.n_cells):
            cx = self.cell_centers_x[cell_i]
            cy = self.cell_centers_y[cell_i]
            grid_i = cell_i // self.nx
            grid_j = cell_i % self.nx
            edges = [
                ([0, -1], (0, -half_res), (1, 0), (-1, 0)),
                ([1, 0], (half_res, 0), (0, 1), (0, 1)),
                ([0, 1], (0, half_res), (1, 0), (1, 0)),
                ([-1, 0], (-half_res, 0), (0, 1), (0, -1)),
            ]
            for edge_id, (normal, center_offset, tangent, neighbor_offset) in enumerate(edges):
                ni = grid_i + neighbor_offset[0]
                nj = grid_j + neighbor_offset[1]
                neighbor_cell = ni * self.nx + nj if 0 <= ni < self.ny and 0 <= nj < self.nx else -1
                for j, xi in enumerate(self.gauss_xi):
                    x_g = cx + center_offset[0] + xi * half_res * tangent[0]
                    y_g = cy + center_offset[1] + xi * half_res * tangent[1]
                    self.gauss_coords[idx] = [x_g, y_g]
                    self.gauss_normals[idx] = normal
                    self.gauss_weights[idx] = self.gauss_weights_1d[j] * half_res
                    self.gauss_cell_id[idx] = cell_i
                    self.gauss_edge_id[idx] = edge_id
                    self.gauss_neighbor_id[idx] = neighbor_cell
                    idx += 1

    def update_bed(self, new_zb):
        self.zb = np.clip(np.array(new_zb), -5.0, 5.0)

    def get_bed_gradient(self,device):
        zb_2d = self.zb.reshape(self.ny, self.nx)
        dzb_dx = np.zeros_like(zb_2d)
        dzb_dy = np.zeros_like(zb_2d)

        if self.nx > 1:
            if self.nx > 2:
                dzb_dx[:, 1:-1] = (zb_2d[:, 2:] - zb_2d[:, :-2]) / (2 * self.resolution)
            dzb_dx[:, 0] = (zb_2d[:, 1] - zb_2d[:, 0]) / self.resolution
            dzb_dx[:, -1] = (zb_2d[:, -1] - zb_2d[:, -2]) / self.resolution

        if self.ny > 1:
            if self.ny > 2:
                dzb_dy[1:-1, :] = (zb_2d[2:, :] - zb_2d[:-2, :]) / (2 * self.resolution)
            dzb_dy[0, :] = (zb_2d[1, :] - zb_2d[0, :]) / self.resolution
            dzb_dy[-1, :] = (zb_2d[-1, :] - zb_2d[-2, :]) / self.resolution

        return (
            torch.tensor(dzb_dx.flatten(), dtype=torch.float32, device=device),
            torch.tensor(dzb_dy.flatten(), dtype=torch.float32, device=device),
        )


