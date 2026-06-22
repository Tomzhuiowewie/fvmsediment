# model.py – PINN 网络模型定义
# 包含：残差块、基类 PINN、以及两个物理子网络
#   FlowPINN      → 水深 h、流速 u, v
#   SedimentPINN  → 总输沙浓度 C_tk 和分粒径累计床变 Δzb_k

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """
    残差块：两个全连接层 + Tanh 激活 + 跳跃连接。
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
    """
    PINN 网络基类，封装公共的前向结构与 Xavier 初始化。
    结构：输入层 → N 个 ResBlock → 输出层。
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
    """
    水动力 PINN：输入 (x, y, t)，输出归一化的 (h, u, v)。
    输出通过 Sigmoid 激活限制在 (0, 1)，并在 encode/decode 中进行物理量的归一化和反归一化。
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
    """
    总输沙输移 PINN：输入 (x, y, t)，输出各粒径级总输沙浓度 C_tk 和分粒径累计床变 Δzb_k。
    """

    def __init__(self, input_dim=3, hidden_dim=64, num_block=4,
                 output_dim=1, positive_output=True, initial_concentration=None,
                 n_concentration_outputs=None, bed_change_scale=0.1):
        self.positive_output = positive_output
        self.n_concentration_outputs = (
            output_dim if n_concentration_outputs is None else int(n_concentration_outputs)
        )
        self.bed_change_scale = float(bed_change_scale)
        if output_dim < self.n_concentration_outputs:
            raise ValueError("SedimentPINN output_dim 不能小于浓度输出维度。")
        super().__init__(input_dim, hidden_dim, num_block, output_dim)
        if self.positive_output:
            self._init_positive_output(output_dim, initial_concentration)

    def _init_positive_output(self, output_dim, initial_concentration):
        """Start near the prescribed sediment background instead of softplus(0)."""
        final_linear = self.output_layer[-1]
        if not isinstance(final_linear, nn.Linear):
            return

        n_c = self.n_concentration_outputs
        if initial_concentration is None:
            target = torch.full((n_c,), 1.0e-6, dtype=final_linear.bias.dtype)
        else:
            target = torch.as_tensor(initial_concentration, dtype=final_linear.bias.dtype)
            if target.numel() == 1:
                target = target.repeat(n_c)
            if target.numel() != n_c:
                raise ValueError("初始泥沙浓度维度必须与 SedimentPINN 输出维度一致。")
            target = torch.clamp(target, min=1.0e-6)

        bias = torch.zeros(output_dim, dtype=final_linear.bias.dtype)
        bias[:n_c] = torch.log(torch.expm1(target))
        with torch.no_grad():
            nn.init.normal_(final_linear.weight, mean=0.0, std=1.0e-4)
            final_linear.bias.copy_(bias)

    def forward(self, xyt: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(xyt)
        for block in self.res_blocks:
            x = block(x)
        out = self.output_layer(x)
        if not self.positive_output:
            return out
        c = F.softplus(out[:, :self.n_concentration_outputs])
        if out.shape[1] == self.n_concentration_outputs:
            return c
        dzb = out[:, self.n_concentration_outputs:] * self.bed_change_scale
        return torch.cat([c, dzb], dim=1)

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
