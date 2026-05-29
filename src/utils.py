# utils.py - 通用数值工具
# 提供坐标归一化、张量闭包匹配、平滑正部等辅助函数。

from typing import Optional

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
