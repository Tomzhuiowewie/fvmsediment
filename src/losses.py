# losses.py – 损失函数通用工具
# 提供坐标归一化、张量闭包匹配、平滑正部等辅助函数，
# 供 physics 模块的各个损失计算器复用。

import torch
import torch.nn.functional as F
from typing import Optional


def mse_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """均方误差损失，用于数据约束（如边界条件点）。"""
    return F.mse_loss(prediction, target)


def normalize_coordinates(coords: torch.Tensor, bounds: dict) -> torch.Tensor:
    """将 (x, y) 坐标按给定上下界映射到 [0, 1] 区间。

    Args:
        coords: 形状 [N, 2] 或 [N, 3] 的坐标张量。
        bounds: 包含 'x_min','x_max','y_min','y_max' 的字典。

    Returns:
        归一化后的 (x, y) 张量，形状 [N, 2]。
    """
    x = (coords[:, 0:1] - bounds['x_min']) / (bounds['x_max'] - bounds['x_min'])
    y = (coords[:, 1:2] - bounds['y_min']) / (bounds['y_max'] - bounds['y_min'])
    return torch.cat([x, y], dim=1)


def build_xyt(coords: torch.Tensor, t_norm: float, bounds: Optional[dict],
              device: torch.device, requires_grad: bool = True) -> torch.Tensor:
    """构造归一化的 (x, y, t) 输入张量，供网络前向与自动微分使用。

    Args:
        coords: 形状 [N, 2] 或 [N, 3] 的物理坐标。
        t_norm: 归一化时间标量（0~1）。
        bounds: 归一化边界字典或 None。
        device: 目标设备。
        requires_grad: 是否开启自动微分（PDE 残差计算需要 True）。

    Returns:
        形状 [N, 3] 的张量，列依次为 x_norm, y_norm, t_norm。
    """
    coords = coords.to(device=device, dtype=torch.float32)
    if bounds is not None:
        xyt_xy = normalize_coordinates(coords, bounds)
    else:
        xyt_xy = coords[:, 0:2]
    t_tensor = torch.full((coords.shape[0], 1), t_norm, dtype=torch.float32, device=device)
    xyt = torch.cat([xyt_xy, t_tensor], dim=1)
    return xyt.requires_grad_(requires_grad)


def match_closure(value, reference: torch.Tensor, default: float) -> torch.Tensor:
    """将标量/数组闭合量整理成与 reference 形状一致的张量。

    若 value 为 None，用 default 填充；若为标量，广播到 reference 形状。
    用于处理 β_tk、ε_thk、E_tk、D_tk 等可能由外部传入或自动闭合的参数。
    """
    if value is None:
        return torch.ones_like(reference) * default
    if torch.is_tensor(value):
        value_tensor = value.to(dtype=reference.dtype, device=reference.device)
    else:
        value_tensor = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    return value_tensor.expand_as(reference) if value_tensor.numel() == 1 else value_tensor


def grain_diameters_like(grain_diameters, reference: torch.Tensor) -> torch.Tensor:
    """返回粒径向量 d_k，形状与 reference 的第 1 维（粒径级数 K）对齐。

    Args:
        grain_diameters: None 或用列表/数组给出的各粒径级代表粒径。
        reference: 用于推断 dtype/device/K 的参考张量。

    Returns:
        形状 [K] 的粒径张量。
    """
    k = reference.shape[1]
    if grain_diameters is None:
        return torch.ones(k, dtype=reference.dtype, device=reference.device) * 2e-4
    d_k = torch.as_tensor(grain_diameters, dtype=reference.dtype, device=reference.device)
    if d_k.numel() != k:
        raise ValueError(f'grain_diameters length must be K={k}, got {d_k.numel()}.')
    return d_k


def smooth_positive(x: torch.Tensor, sharpness: float = 1e-3) -> torch.Tensor:
    """平滑正部函数，用 softplus 近似 max(x, 0)，在 x=0 处可导。

    用于侵蚀/沉积分解时将净源汇项拆分为 E_tk 和 D_tk，保证自动微分连续。
    """
    return F.softplus(x / sharpness) * sharpness
