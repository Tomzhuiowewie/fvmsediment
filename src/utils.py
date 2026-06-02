# utils.py - 通用数值工具
# 提供坐标归一化、张量闭包匹配、平滑正部等辅助函数。

from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .config import EPS_DIVISION


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


def match_closure(value, reference: torch.Tensor, default: float) -> torch.Tensor:
    """将标量/数组闭合量整理成与 reference 形状一致的张量。"""
    if value is None:
        return torch.ones_like(reference) * default
    if torch.is_tensor(value):
        value_tensor = value.to(dtype=reference.dtype, device=reference.device)
    else:
        value_tensor = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    return value_tensor.expand_as(reference) if value_tensor.numel() == 1 else value_tensor


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


def smooth_positive(x: torch.Tensor, sharpness: float = 1e-3) -> torch.Tensor:
    """平滑正部函数，用 softplus 近似 max(x, 0)，在 x=0 处可导。"""
    return F.softplus(x / sharpness) * sharpness


def visualize_hump_initial_bed(
    bed_fn: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
    bbox: Optional[dict] = None,
    resolution: float = 20.0,
    save_path: Optional[str] = 'hump_initial_bed.png',
    show: bool = False,
    elev: float = 24.0,
    azim: float = -72.0,
):
    """可视化 hump 初始床面，生成带三角网格线的 3D 曲面图。

    Args:
        bed_fn: 初始床面函数，签名为 ``bed_fn(x, y) -> zb``。不传时使用
            ``src.data.hump_initial_bed``。
        bbox: 物理区域，默认 ``{'xmin': 0, 'xmax': 1000, 'ymin': 0, 'ymax': 1000}``。
        resolution: 规则采样间距，越小曲面越细。
        save_path: 图片保存路径；传 ``None`` 时不保存。
        show: 是否弹出显示窗口。
        elev: 3D 视角仰角。
        azim: 3D 视角方位角。

    Returns:
        matplotlib 的 ``(fig, ax)`` 对象，便于调用端继续调整。
    """
    if bed_fn is None:
        from .data import hump_initial_bed

        bed_fn = hump_initial_bed

    if bbox is None:
        bbox = {'xmin': 0.0, 'xmax': 1000.0, 'ymin': 0.0, 'ymax': 1000.0}

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("visualize_hump_initial_bed 需要安装 matplotlib。") from exc

    x = np.arange(bbox['xmin'], bbox['xmax'] + resolution, resolution)
    y = np.arange(bbox['ymin'], bbox['ymax'] + resolution, resolution)
    X, Y = np.meshgrid(x, y)
    Z = bed_fn(X, Y)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_trisurf(
        X.ravel(),
        Y.ravel(),
        Z.ravel(),
        cmap='terrain',
        edgecolor='k',
        linewidth=0.35,
        antialiased=True,
    )
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim(bbox['xmin'], bbox['xmax'])
    ax.set_ylim(bbox['ymin'], bbox['ymax'])
    ax.set_zlim(min(0.0, float(np.min(Z))), max(float(np.max(Z)), 1.0))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('zb')
    ax.set_title('Hump Initial Bed')
    ax.grid(False)
    fig.colorbar(surf, ax=ax, shrink=0.55, aspect=16, pad=0.08, label='zb')
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=160, bbox_inches='tight')
    if show:
        plt.show()
    return fig, ax
