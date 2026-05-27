# data.py – 地形与网格数据预处理
# 包含：
#   FVMeshPreprocessor    → FVM 结构化网格生成、高斯积分点预计算、床面梯度
#   DemPreprocessor       → DEM GeoTIFF 读取与局部坐标转换
#   SedimentDataProcessor → 河床粒径级配数据解析

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import ndimage


class FVMeshPreprocessor:
    def __init__(self, bbox, resolution, initial_bed=None, n_gauss_points=2):
        self.bbox = bbox
        self.resolution = resolution
        self.n_gauss_points = n_gauss_points
        self._generate_mesh()
        self._initialize_bed(initial_bed)
        self._setup_gauss_quadrature()
        self._precompute_edge_data()

    def _generate_mesh(self):
        xmin, xmax = self.bbox['xmin'], self.bbox['xmax']
        ymin, ymax = self.bbox['ymin'], self.bbox['ymax']
        self.nx = int((xmax - xmin) / self.resolution)
        self.ny = int((ymax - ymin) / self.resolution)
        x_centers = np.linspace(xmin + self.resolution / 2, xmax - self.resolution / 2, self.nx)
        y_centers = np.linspace(ymin + self.resolution / 2, ymax - self.resolution / 2, self.ny)
        self.cell_centers_x, self.cell_centers_y = np.meshgrid(x_centers, y_centers)
        self.cell_centers_x = self.cell_centers_x.flatten()
        self.cell_centers_y = self.cell_centers_y.flatten()
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

    def get_bed_at_gauss_points(self):
        return self.zb[self.gauss_cell_id]

    def get_bed_gradient(self):
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

        return dzb_dx.flatten(), dzb_dy.flatten()

    def get_bed_gradient_tensor(self, zb_tensor, device):
        """统一的床面梯度计算，支持 NumPy (zb_tensor=None) 和 PyTorch 张量两种输入。"""
        if zb_tensor is None:
            dzb_dx, dzb_dy = self.get_bed_gradient()
            return (torch.tensor(dzb_dx, dtype=torch.float32, device=device),
                    torch.tensor(dzb_dy, dtype=torch.float32, device=device))

        zb = zb_tensor.to(dtype=torch.float32, device=device).reshape(self.ny, self.nx)
        dzb_dx = torch.zeros_like(zb)
        dzb_dy = torch.zeros_like(zb)
        if self.nx > 1:
            if self.nx > 2:
                dzb_dx[:, 1:-1] = (zb[:, 2:] - zb[:, :-2]) / (2 * self.resolution)
            dzb_dx[:, 0] = (zb[:, 1] - zb[:, 0]) / self.resolution
            dzb_dx[:, -1] = (zb[:, -1] - zb[:, -2]) / self.resolution
        if self.ny > 1:
            if self.ny > 2:
                dzb_dy[1:-1, :] = (zb[2:, :] - zb[:-2, :]) / (2 * self.resolution)
            dzb_dy[0, :] = (zb[1, :] - zb[0, :]) / self.resolution
            dzb_dy[-1, :] = (zb[-1, :] - zb[-2, :]) / self.resolution
        return dzb_dx.flatten(), dzb_dy.flatten()

# 英尺到米的转换系数
US_SURVEY_FOOT_TO_METER = 0.3048006096012192

class DemPreprocessor:
    """读取 DEM GeoTIFF，并转换为局部米制坐标和高程。"""

    def __init__(
        self,
        tif_path,
        elevation_to_meter=US_SURVEY_FOOT_TO_METER,
        mark_river_region=True,
        river_elevation_max_m=None,
        river_elevation_min_m=None,
        river_seed_index=None,
        river_seed_point_m=None,
    ):
        """初始化 DEM 数据，并可按高程阈值和种子点标记河道区域。"""
        rasterio = importlib.import_module("rasterio")
        tif_path = Path(tif_path)

        with rasterio.open(tif_path) as src:
            dem = src.read(1).astype("float32")
            transform = src.transform
            nodata = src.nodata
            dem = np.where(dem == nodata, np.nan, dem)
            coordinate_to_meter = float(src.crs.linear_units_factor[-1]) if src.crs else elevation_to_meter
            
        rows, cols = np.indices(dem.shape, dtype="float64")  # 获取 DEM 网格索引
        x_raw, y_raw = transform * (cols + 0.5, rows + 0.5)  # 获取 DEM 网格格心坐标

        # 转换为米制坐标和高程，并以最小坐标作为局部原点，方便后续计算和可视化
        x_grid_m = np.asarray(x_raw, dtype="float64") * coordinate_to_meter
        y_grid_m = np.asarray(y_raw, dtype="float64") * coordinate_to_meter
        elevation_m = dem.astype("float64") * elevation_to_meter

        x_origin_m = np.nanmin(x_grid_m)  # 获取 x 坐标最小值作为局部坐标原点
        y_origin_m = np.nanmin(y_grid_m)  # 获取 y 坐标最小值作为局部坐标原点

        # 存储为实例属性
        self.x_grid_m = x_grid_m - x_origin_m  # 相对坐标网格，单位 m
        self.y_grid_m = y_grid_m - y_origin_m  # 相对坐标网格，单位 m
        self.elevation_m = elevation_m         # 高程网格，单位 m
        self.x_origin_m = x_origin_m           # 原始 x 坐标原点，单位 m
        self.y_origin_m = y_origin_m           # 原始 y 坐标原点，单位 m
        self.river_region_mask = self._build_river_region_mask(
            mark_river_region,
            river_elevation_min_m,
            river_elevation_max_m,
            river_seed_index,
            river_seed_point_m,
        )

    def _build_river_region_mask(
        self,
        mark_river_region,
        river_elevation_min_m,
        river_elevation_max_m,
        river_seed_index,
        river_seed_point_m,
    ):
        """按高程范围生成掩膜，并只保留种子点所在连通域。"""
        if not mark_river_region:
            return None

        mask = np.isfinite(self.elevation_m)
        if river_elevation_min_m is not None:
            mask &= self.elevation_m >= river_elevation_min_m
        if river_elevation_max_m is not None:
            mask &= self.elevation_m <= river_elevation_max_m

        # 连通域构建
        labels, num_features = ndimage.label(mask)
        if num_features == 0:
            return mask

        seed_index = self._resolve_river_seed_index(river_seed_index, river_seed_point_m)
        if seed_index is None:
            # 未指定种子点时，退回到最大连通域，避免返回多个碎片区域。
            component_sizes = np.bincount(labels.ravel())   # 二维数据→一维标签计数，统计每个编号出现次数
            component_sizes[0] = 0  # 标签0是背景，计数置零
            return labels == component_sizes.argmax()

        seed_row, seed_col = seed_index
        seed_label = labels[seed_row, seed_col]
        if seed_label == 0:
            raise ValueError("河道种子点不在高程阈值筛选出的区域内，请调整种子点或高程阈值。")

        return labels == seed_label

    def _resolve_river_seed_index(self, river_seed_index, river_seed_point_m):
        """把用户给定的种子点转换为 DEM 行列索引。"""
        if river_seed_index is not None and river_seed_point_m is not None:
            raise ValueError("river_seed_index 和 river_seed_point_m 只能指定一个。")

        if river_seed_index is not None:
            row, col = map(int, river_seed_index)
            if not (0 <= row < self.elevation_m.shape[0] and 0 <= col < self.elevation_m.shape[1]):
                raise ValueError("river_seed_index 超出 DEM 网格范围。")
            return row, col

        if river_seed_point_m is None:
            return None

        seed_x_m, seed_y_m = river_seed_point_m
        distance2 = (self.x_grid_m - seed_x_m) ** 2 + (self.y_grid_m - seed_y_m) ** 2
        return tuple(np.unravel_index(np.nanargmin(distance2), distance2.shape))

    def to_dict(self):
        """以字典形式返回 DEM 数据"""
        return {
            "x_grid_m": self.x_grid_m,
            "y_grid_m": self.y_grid_m,
            "elevation_m": self.elevation_m,
            "x_origin_m": self.x_origin_m,
            "y_origin_m": self.y_origin_m,
            "river_region_mask": self.river_region_mask,
        }


class SedimentDataProcessor:
    """河床粒径数据处理类"""

    def __init__(self, bed_template, bed_gradation, gradation_layers):
        self.bed_template = bed_template
        self.gradation_layers = gradation_layers
        self.bed_gradation = bed_gradation

    @classmethod
    def from_excel(cls, path, sheet_name="Sediment Data", skiprows=2):
        """从 Excel 文件解析河床粒径数据。"""
        df = pd.read_excel(path, sheet_name=sheet_name, skiprows=skiprows, header=None).ffill()

        bed_template = dict(zip(df.iloc[:3, 2:4].iloc[:, 0], df.iloc[:3, 2:4].iloc[:, 1]))

        gradation_layers = df.iloc[27:, 1:4].ffill(axis=0)
        gradation_layers.columns = gradation_layers.iloc[0]
        gradation_layers = gradation_layers.iloc[1:].reset_index(drop=True)

        bed_gradation = df.iloc[6:27, 1:6].replace({np.nan: 0})
        bed_gradation.columns = bed_gradation.iloc[0]
        bed_gradation = bed_gradation.iloc[1:].reset_index(drop=True)

        return cls(bed_template, bed_gradation, gradation_layers)


if __name__ == "__main__":
    bed_path = "data/FlowSedimentData.xlsx"
    processor = SedimentDataProcessor.from_excel(bed_path)
    print(processor.bed_template)
    print(processor.bed_gradation)
    print(processor.gradation_layers)

