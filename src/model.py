# model.py – PINN 网络模型定义
# 包含：残差块、基类 PINN、以及两个物理子网络
#   FlowPINN      → 水深 h、流速 u, v
#   SedimentPINN  → 总输沙浓度 C_tk

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """残差块：两个全连接层 + Tanh 激活 + 跳跃连接。

    跳跃连接有助于缓解深层 PINN 中的梯度消失问题，
    让网络更容易学习 PDE 的复杂解。
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x                           # 保存跳跃输入
        x = self.activation(self.fc1(x))       # FC1 + Tanh
        x = self.fc2(x)                        # FC2
        return self.activation(x + residual)   # 残差连接 + Tanh


class BasePINN(nn.Module):
    """PINN 网络基类，封装公共的前向结构与 Xavier 初始化。

    结构：输入层 → N 个 ResBlock → 输出层（含可选的最终激活函数）。
    Flow/Sediment 子网络均继承此类，
    只需定制输出维度和最终激活即可。
    """

    def __init__(
        self,
        input_dim: int = 3,          # 输入维度，默认 (x, y, t)
        hidden_dim: int = 64,        # 隐藏层宽度
        num_block: int = 4,          # 残差块数量
        output_dim: int = 1,         # 输出维度
        final_activation: nn.Module = None,  # 可选的最终激活函数
    ):
        super().__init__()
        self.input_layer = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())
        self.res_blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(num_block)])
        layers = [
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, output_dim),
        ]
        if final_activation is not None:
            layers.append(final_activation)
        self.output_layer = nn.Sequential(*layers)
        self.init_weights()

    def init_weights(self):
        """Xavier 均匀初始化所有权重，偏置置零。"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, xyt: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(xyt)
        for block in self.res_blocks:
            x = block(x)
        return self.output_layer(x)


class FlowPINN(BasePINN):
    """水动力 PINN：输入 (x, y, t)，输出归一化的 (h, u, v)。

    输出经过 Sigmoid 约束到 [0, 1]，再由 decode_output 反归一化到物理量：
        h = h_norm * typical_h
        u = (u_norm - 0.5) * 2 * typical_u
        v = (v_norm - 0.5) * 2 * typical_u
    """
    def __init__(self, input_dim=3, hidden_dim=64, num_block=4, output_dim=3):
        super().__init__(input_dim, hidden_dim, num_block, output_dim,
                         final_activation=nn.Sigmoid())

    @staticmethod
    def encode_target(
        h: torch.Tensor, u: torch.Tensor, v: torch.Tensor,
        typical_h: float, typical_u: float,
    ) -> torch.Tensor:
        h_norm = h / typical_h
        u_norm = u / (2.0 * typical_u) + 0.5
        v_norm = v / (2.0 * typical_u) + 0.5
        return torch.cat([h_norm, u_norm, v_norm], dim=1)

    @staticmethod
    def decode_output(
        raw: torch.Tensor, typical_h: float, typical_u: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = raw[:, 0:1] * typical_h
        u = (raw[:, 1:2] - 0.5) * 2.0 * typical_u
        v = (raw[:, 2:3] - 0.5) * 2.0 * typical_u
        return h, u, v


class SedimentPINN(BasePINN):
    """总输沙输移 PINN：输入 (x, y, t)，输出各粒径级总输沙浓度 C_tk。

    正向输出通过 softplus 保证 C_tk ≥ 0。
    """

    def __init__(self, input_dim=3, hidden_dim=64, num_block=4,
                 output_dim=1, positive_output=True, **kwargs):
        self.positive_output = positive_output
        super().__init__(input_dim, hidden_dim, num_block, output_dim)

    def forward(self, xyt: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(xyt)
        for block in self.res_blocks:
            x = block(x)
        out = self.output_layer(x)
        # softplus 确保浓度非负，且比 ReLU 更平滑，适合自动微分
        return F.softplus(out) if self.positive_output else out

    # ---------- 自动微分工具 ----------

    @staticmethod
    def _grad(q: torch.Tensor, xyt: torch.Tensor, dim: int) -> torch.Tensor:
        """对 xyt 的某一列求偏导；dim=0,1,2 分别代表 ∂/∂x, ∂/∂y, ∂/∂t。

        当 q 为多输出（形状 [N, K]）时，逐列求导避免把 C_1...C_K
        的梯度混在一起。
        """
        if not xyt.requires_grad:
            raise ValueError('计算 PDE 残差时 xyt 必须 requires_grad=True。')

        def grad_one(q_one: torch.Tensor) -> torch.Tensor:
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
