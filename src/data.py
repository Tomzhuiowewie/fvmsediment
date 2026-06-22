
from dataclasses import dataclass
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import torch
from PIL import Image

FT_TO_M = 0.3048
CFS_TO_CMS = 0.028316846592
MM_TO_M = 1.0e-3


# =============================================================================
# 数据容器
# =============================================================================

@dataclass
class RealCaseData:
    """真实算例输入数据容器。

    所有字段在进入模型前已经统一到 SI 单位：
    坐标/床面/水位为 m，流量为 m3/s，时间为 s，粒径为 m。
    """
    bbox: dict
    bounds: dict
    resolution: float
    bed_grid: np.ndarray
    active_mask: np.ndarray
    flow_times: np.ndarray
    flow_values: np.ndarray
    stage_times: np.ndarray
    stage_values: np.ndarray
    grain_diameters: list[float]
    grain_fractions: list[float]


# =============================================================================
# FVM 规则网格与 DEM 床面
# =============================================================================

class FVMeshPreprocessor:
    def __init__(self, bbox, resolution, initial_bed=None, n_gauss_points=2, active_mask=None):
        """根据 DEM 规则网格生成 FVM 单元和边界高斯积分点。

        当前仍采用矩形规则网格；河道范围通过 active_mask 过滤。
        这样第一版可以保留 DEM 的原始栅格结构，同时避免非河道区域进入 PDE loss。
        """
        self.bbox = bbox
        self.resolution = resolution
        self.n_gauss_points = n_gauss_points
        self._active_mask_input = active_mask
        self._generate_mesh()
        self._initialize_bed(initial_bed)   # 初始化床面高程数据
        self._initialize_active_mask(active_mask)
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
        """初始化床面高程 zb。

        真实算例中 initial_bed 是 DEM 栅格；如果后续做理想算例，也可以传函数或常数。
        """
        if initial_bed is None:
            self.zb = np.zeros(self.n_cells)
        elif callable(initial_bed):
            self.zb = initial_bed(self.cell_centers_x, self.cell_centers_y)
        else:
            bed_array = np.asarray(initial_bed, dtype=np.float64)
            if bed_array.shape == (self.ny, self.nx):
                self.zb = bed_array.reshape(-1)
            elif bed_array.size == self.n_cells:
                self.zb = bed_array.reshape(-1)
            else:
                self.zb = np.full(self.n_cells, float(initial_bed))
        self.zb_initial = self.zb.copy()

    def _initialize_active_mask(self, active_mask):
        """初始化河道有效单元 mask。

        active=True 的单元参与物理残差、级配更新和结果统计；
        inactive 单元保留在规则网格中，但不作为河道计算域。
        """
        if active_mask is None:
            self.active_cell_mask = np.ones(self.n_cells, dtype=bool)
            self.active_cell_mask_2d = self.active_cell_mask.reshape(self.ny, self.nx)
            return
        mask_array = np.asarray(active_mask, dtype=bool)
        if mask_array.shape == (self.ny, self.nx):
            self.active_cell_mask_2d = mask_array
            self.active_cell_mask = mask_array.reshape(-1)
        elif mask_array.size == self.n_cells:
            self.active_cell_mask = mask_array.reshape(-1)
            self.active_cell_mask_2d = self.active_cell_mask.reshape(self.ny, self.nx)
        else:
            raise ValueError("active_mask 形状必须与 FVM 网格一致。")
        self.active_cell_ids = np.where(self.active_cell_mask)[0].astype(np.int64)

    def _setup_gauss_quadrature(self):
        if self.n_gauss_points == 2:
            sqrt_1_3 = np.sqrt(1.0 / 3.0)
            self.gauss_xi = np.array([-sqrt_1_3, sqrt_1_3])
            self.gauss_weights_1d = np.array([1.0, 1.0])
        else:
            self.gauss_xi = np.array([0.0])
            self.gauss_weights_1d = np.array([2.0])

    def _precompute_edge_data(self):
        """预计算每个单元四条边上的高斯点。

        gauss_cell_id 记录高斯点属于哪个单元；
        gauss_edge_id 记录属于下/右/上/左哪条边；
        gauss_neighbor_id 用于识别外边界和后续扩展固壁边界。
        """
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
        self.active_gauss_ids = (
            self.active_cell_ids[:, None] * self.n_points_per_cell
            + np.arange(self.n_points_per_cell, dtype=np.int64)[None, :]
        ).reshape(-1)

    def update_bed(self, new_zb):
        """更新当前绝对床面高程，保留初始床面用于累计变化诊断。"""
        bed = np.asarray(new_zb, dtype=np.float64).reshape(-1)
        if bed.size != self.n_cells:
            raise ValueError("new_zb 的单元数必须与 FVM 网格一致。")
        if not np.all(np.isfinite(bed)):
            raise ValueError("new_zb 包含 NaN 或 Inf。")
        self.zb = bed
        self._bed_gradient_cache = {}

    def get_bed_gradient(self, device):
        """用中心差分计算床面坡度 ∂zb/∂x 和 ∂zb/∂y。"""
        cache = getattr(self, '_bed_gradient_cache', {})
        if device in cache:
            return cache[device]

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

        tensors = (
            torch.tensor(dzb_dx.flatten(), dtype=torch.float32, device=device),
            torch.tensor(dzb_dy.flatten(), dtype=torch.float32, device=device),
        )
        self._bed_gradient_cache = cache
        self._bed_gradient_cache[device] = tensors
        return tensors


# =============================================================================
# 河道边界连通段筛选
# =============================================================================

def boundary_exposed_mask(active_mask_2d, edge):
    """筛选 active 区域在指定方向上邻接 inactive/外部的暴露边单元。"""
    mask = np.asarray(active_mask_2d, dtype=bool)
    if edge == 'top':
        neighbor_active = np.pad(mask[1:, :], ((0, 1), (0, 0)), constant_values=False)
    elif edge == 'bottom':
        neighbor_active = np.pad(mask[:-1, :], ((1, 0), (0, 0)), constant_values=False)
    else:
        raise ValueError("当前只支持 top/bottom 边界。")
    return mask & (~neighbor_active)


def select_boundary_component(active_mask_2d, edge):
    """从暴露边中选出主入口/出口连通段。

    DEM 河道可能斜向穿过边界，使用 8 邻域把斜向相邻的暴露边连起来。
    top 选平均行号最大的连通段；bottom 选平均行号最小的连通段。
    """
    exposed = boundary_exposed_mask(active_mask_2d, edge)
    selected = np.zeros_like(exposed, dtype=bool)
    if not np.any(exposed):
        return selected

    visited = np.zeros_like(exposed, dtype=bool)
    neighbors = [
        (dr, dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if not (dr == 0 and dc == 0)
    ]
    best_rows = None
    best_cols = None
    best_key = None

    for row, col in zip(*np.where(exposed)):
        if visited[row, col]:
            continue
        stack = [(int(row), int(col))]
        visited[row, col] = True
        rows = []
        cols = []
        while stack:
            r, c = stack.pop()
            rows.append(r)
            cols.append(c)
            for dr, dc in neighbors:
                nr = r + dr
                nc = c + dc
                if (
                    0 <= nr < exposed.shape[0]
                    and 0 <= nc < exposed.shape[1]
                    and exposed[nr, nc]
                    and not visited[nr, nc]
                ):
                    visited[nr, nc] = True
                    stack.append((nr, nc))

        rows_arr = np.asarray(rows, dtype=np.int64)
        cols_arr = np.asarray(cols, dtype=np.int64)
        mean_row = float(np.mean(rows_arr))
        size = int(rows_arr.size)
        key = (mean_row, size) if edge == 'top' else (-mean_row, size)
        if best_key is None or key > best_key:
            best_key = key
            best_rows = rows_arr
            best_cols = cols_arr

    selected[best_rows, best_cols] = True
    return selected


# =============================================================================
# 真实数据读取：DEM / Excel
# =============================================================================

def load_real_case_data(data_cfg) -> RealCaseData:
    """
    读取真实算例输入，并统一转换到 SI 单位。
    DEM 提供初始床面和河道掩膜；
    Excel 提供非恒定边界过程线和床沙级配。
    """
    dem_path = Path(data_cfg['dem_path'])
    xlsx_path = Path(data_cfg['hydro_sediment_path'])
    threshold_ft = float(data_cfg['channel_elevation_threshold_ft'])

    bed_grid, active_mask, resolution = _read_dem_grid(dem_path, threshold_ft)
    ny, nx = bed_grid.shape
    bbox = {
        'xmin': 0.0,
        'xmax': float(nx * resolution),
        'ymin': 0.0,
        'ymax': float(ny * resolution),
    }
    bounds = {
        'x_min': bbox['xmin'],
        'x_max': bbox['xmax'],
        'y_min': bbox['ymin'],
        'y_max': bbox['ymax'],
    }
    flow_times, flow_values, stage_times, stage_values = _read_hydrographs(xlsx_path)
    grain_diameters, grain_fractions = _read_main_channel_gradation(xlsx_path)
    return RealCaseData(
        bbox=bbox,
        bounds=bounds,
        resolution=resolution,
        bed_grid=bed_grid,
        active_mask=active_mask,
        flow_times=flow_times,
        flow_values=flow_values,
        stage_times=stage_times,
        stage_values=stage_values,
        grain_diameters=grain_diameters,
        grain_fractions=grain_fractions,
    )


def _read_dem_grid(path: Path, threshold_ft: float):
    """读取 GeoTIFF DEM，并用绝对高程阈值划定河道活动单元。"""
    with Image.open(path) as im:
        arr_ft = np.asarray(im, dtype=np.float32)
        tags = im.tag_v2
        nodata = float(tags.get(42113, -9999))
        pixel_scale = tags.get(33550, (25.0, 25.0, 0.0))
        resolution = float(pixel_scale[0]) * FT_TO_M

    # NoData 不参与河道判断；低于阈值的有效 DEM 单元视为河道活动单元。
    valid = np.isfinite(arr_ft) & (arr_ft != nodata)
    active = valid & (arr_ft <= threshold_ft)

    # 非有效 DEM 单元用有效高程最大值填充，避免模型/绘图出现 NaN。
    # 这些单元通常被 active_mask 排除，不参与物理训练。
    fill_ft = float(np.nanmax(np.where(valid, arr_ft, np.nan)))
    bed_ft = np.where(valid, arr_ft, fill_ft)
    bed_m = bed_ft * FT_TO_M
    return bed_m[::-1, :].astype(np.float32), active[::-1, :], resolution


def _xlsx_tables(path: Path):
    """直接解析 xlsx 内部 XML，避免为了科研脚本额外依赖 openpyxl。"""
    zf = zipfile.ZipFile(path)
    ns = {'a': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    shared = []
    if 'xl/sharedStrings.xml' in zf.namelist():
        root = ET.fromstring(zf.read('xl/sharedStrings.xml'))
        for si in root.findall('a:si', ns):
            shared.append(''.join(t.text or '' for t in si.findall('.//a:t', ns)))

    rels = ET.fromstring(zf.read('xl/_rels/workbook.xml.rels'))
    rel_map = {r.attrib['Id']: r.attrib['Target'] for r in rels}
    wb = ET.fromstring(zf.read('xl/workbook.xml'))
    tables = {}
    for sheet in wb.findall('.//a:sheet', ns):
        name = sheet.attrib['name']
        rid = sheet.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']
        ws = ET.fromstring(zf.read('xl/' + rel_map[rid]))
        rows = []
        for row in ws.findall('.//a:sheetData/a:row', ns):
            values = []
            for cell in row.findall('a:c', ns):
                v = cell.find('a:v', ns)
                value = '' if v is None else v.text
                if cell.attrib.get('t') == 's' and value != '':
                    value = shared[int(value)]
                values.append(value)
            rows.append(values)
        tables[name] = rows
    return tables


def _to_float(value):
    if value in (None, ''):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_hydrographs(path: Path):
    """读取上游流量和下游水位过程线，并转成秒、m3/s、m。"""
    rows = _xlsx_tables(path)['Unsteady Flow Data'][1:]
    flow_t, flow_q, stage_t, stage_h = [], [], [], []
    for row in rows:
        if len(row) >= 3:
            t = _to_float(row[1])
            q = _to_float(row[2])
            if t is not None and q is not None:
                # Excel 中时间单位为小时，流量单位为 cfs。
                flow_t.append(t * 3600.0)
                flow_q.append(q * CFS_TO_CMS)
        if len(row) >= 5:
            t = _to_float(row[3])
            h = _to_float(row[4])
            if t is not None and h is not None:
                # 下游 stage 用绝对水位表示，单位 ft。
                stage_t.append(t * 3600.0)
                stage_h.append(h * FT_TO_M)
    return (
        np.asarray(flow_t, dtype=np.float32),
        np.asarray(flow_q, dtype=np.float32),
        np.asarray(stage_t, dtype=np.float32),
        np.asarray(stage_h, dtype=np.float32),
    )


def _read_main_channel_gradation(path: Path):
    """读取 MainChannel 的累计级配曲线，并转成分组比例。"""
    rows = _xlsx_tables(path)['Sediment Data']
    diameters_mm, finer_pct = [], []
    for row in rows:
        if len(row) < 5:
            continue
        d = _to_float(row[2])
        f = _to_float(row[4])
        if d is not None and f is not None:
            diameters_mm.append(d)
            finer_pct.append(f)

    diameters = np.asarray(diameters_mm, dtype=np.float32) * MM_TO_M
    finer = np.asarray(finer_pct, dtype=np.float32) / 100.0
    # Excel 给的是“累计通过百分比 % finer”，模型需要每个粒径组的体积分数。
    fractions = np.diff(np.concatenate([[0.0], finer]))
    keep = fractions > 1.0e-6
    return diameters[keep].tolist(), fractions[keep].tolist()


# =============================================================================
# 真实流量/水位边界构造
# =============================================================================

class RealBoundaryConditionBuilder:
    def __init__(self, mesh, real_case: RealCaseData, typical_depth, typical_velocity,
                 simulation_time, h_min=0.05):
        """构造真实边界条件。

        当前约定：
        - top 为上游入口，使用流量过程线 Q(t)；
        - bottom 为下游出口，使用水位过程线 stage(t)。
        """
        self.mesh = mesh
        self.real_case = real_case
        self.typical_depth = typical_depth
        self.typical_velocity = typical_velocity
        self.simulation_time = float(simulation_time)
        self.h_min = h_min
        self.inlet = self._edge_payload('top')
        self.outlet = self._edge_payload('bottom')

    def __call__(self, t_norm):
        # 训练器传入归一化时间；这里还原成物理时间并插值真实过程线。
        t_physical = float(t_norm) * max(self.simulation_time, 1.0)
        q = float(np.interp(t_physical, self.real_case.flow_times, self.real_case.flow_values))
        stage = float(np.interp(t_physical, self.real_case.stage_times, self.real_case.stage_values))
        return {
            'kind': 'real_flow_stage',
            't_norm': float(t_norm),
            'inlet_coords': self._coords_with_time(self.inlet['coords_norm'], t_norm),
            'inlet_weights': self.inlet['weights'],
            'target_flow': q,
            'outlet_coords': self._coords_with_time(self.outlet['coords_norm'], t_norm),
            # 床面在联合形态步中会更新，下游目标水深必须使用当前出口床高。
            'outlet_bed': self.mesh.zb[self.outlet['cell_ids']].astype(np.float32),
            'target_stage': stage,
            'typical_depth': self.typical_depth,
            'typical_velocity': self.typical_velocity,
            'h_min': self.h_min,
            'inlet_edge': 'top',
            'outlet_edge': 'bottom',
        }

    def _edge_payload(self, edge):
        """从 active_mask 自动抽取入口/出口断面离散点。

        top 入口先取 active 单元中上邻为空/非 active 的上边，再保留主连通段；
        bottom 出口先取 active 单元中下邻为空/非 active 的下边，再保留主连通段。
        每条暴露边贡献一个断面点，权重近似为 DEM 像元宽度。
        """
        mask = self.mesh.active_cell_mask_2d
        if edge == 'top':
            selected = select_boundary_component(mask, edge)
            rows, cols = np.where(selected)
            y = self.mesh.bbox['ymin'] + (rows + 1.0) * self.mesh.resolution
        elif edge == 'bottom':
            selected = select_boundary_component(mask, edge)
            rows, cols = np.where(selected)
            y = self.mesh.bbox['ymin'] + rows * self.mesh.resolution
        else:
            raise ValueError("当前真实数据边界只实现 top/bottom。")
        if cols.size == 0:
            raise ValueError(f"{edge} 边界没有河道活动单元。")
        x = self.mesh.bbox['xmin'] + (cols + 0.5) * self.mesh.resolution
        coords_phys = np.stack([x, y.astype(np.float32)], axis=1)
        coords_norm = np.stack([
            (coords_phys[:, 0] - self.real_case.bounds['x_min'])
            / (self.real_case.bounds['x_max'] - self.real_case.bounds['x_min']),
            (coords_phys[:, 1] - self.real_case.bounds['y_min'])
            / (self.real_case.bounds['y_max'] - self.real_case.bounds['y_min']),
        ], axis=1).astype(np.float32)
        cell_ids = rows * self.mesh.nx + cols
        return {
            'coords_norm': coords_norm,
            'weights': np.full(cols.size, self.mesh.resolution, dtype=np.float32),
            'cell_ids': np.asarray(cell_ids, dtype=np.int64),
        }

    @staticmethod
    def _coords_with_time(coords_norm, t_norm):
        return np.concatenate(
            [coords_norm, np.full((coords_norm.shape[0], 1), float(t_norm), dtype=np.float32)],
            axis=1,
        )
