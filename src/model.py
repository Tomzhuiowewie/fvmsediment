# model.py – PINN 网络模型定义
# 包含：残差块、基类 PINN、以及四个物理子网络
#   FlowPINN      → 水深 h、流速 u, v
#   SedimentPINN  → 总输沙浓度 C_tk
#   BedPINN       → 河床高程 z_b
#   GradationPINN → 床沙级配 p_k
# 同时提供 CoupledFlowSedimentPINN 作为统一耦合容器。

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
    所有子网络（Flow/Sediment/Bed/Gradation）均继承此类，
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

    输出经过 Sigmoid 约束到 [0, 1]，再由损失函数反归一化到物理量：
        h = h_norm * typical_h
        u = (u_norm - 0.5) * 2 * typical_u
        v = (v_norm - 0.5) * 2 * typical_u
    """
    def __init__(self, input_dim=3, hidden_dim=64, num_block=4, output_dim=3):
        super().__init__(input_dim, hidden_dim, num_block, output_dim,
                         final_activation=nn.Sigmoid())


class SedimentPINN(BasePINN):
    """总输沙输移 PINN：输入 (x, y, t)，输出各粒径级总输沙浓度 C_tk。

    正向输出通过 softplus 保证 C_tk ≥ 0。
    类中还包含总输沙 PDE 残差计算、Grass 输沙公式等静态方法。
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

    @staticmethod
    def _dx(q: torch.Tensor, xyt: torch.Tensor) -> torch.Tensor:
        """∂q/∂x"""
        return SedimentPINN._grad(q, xyt, 0)

    @staticmethod
    def _dy(q: torch.Tensor, xyt: torch.Tensor) -> torch.Tensor:
        """∂q/∂y"""
        return SedimentPINN._grad(q, xyt, 1)

    @staticmethod
    def _dt(q: torch.Tensor, xyt: torch.Tensor) -> torch.Tensor:
        """∂q/∂t"""
        return SedimentPINN._grad(q, xyt, 2)

    # ---------- HEC-RAS 总输沙输移方程 ----------

    @staticmethod
    def total_load_residual(
        xyt: torch.Tensor,
        h: torch.Tensor,   # 水深
        u: torch.Tensor,   # x 方向流速
        v: torch.Tensor,   # y 方向流速
        C_tk: torch.Tensor,        # 总输沙浓度
        beta_tk: torch.Tensor,     # 总输沙修正系数
        epsilon_thk: torch.Tensor, # 扩散系数
        E_tk: torch.Tensor,        # 侵蚀率（源）
        D_tk: torch.Tensor,        # 沉积率（汇）
    ) -> torch.Tensor:
        """HEC-RAS 原文总输沙输移方程残差。

        PDE:
            ∂/∂t(h C_tk / β_tk) + ∇·(h U C_tk)
            = ∇·(ε_thk h ∇C_tk) + E_tk - D_tk

        残差 R_Ck = 储量项 + 对流项 - 扩散项 - 源 + 汇。
        理想情况下 R_Ck = 0。
        """
        eps = 1e-8
        beta_safe = torch.clamp(beta_tk, min=eps)          # 防止除以零

        # ① 储量项：∂(h C_tk / β_tk) / ∂t
        storage = h * C_tk / beta_safe
        storage_t = SedimentPINN._dt(storage, xyt)

        # ② 对流项：∂(h u C_tk)/∂x + ∂(h v C_tk)/∂y
        adv_x = h * u * C_tk
        adv_y = h * v * C_tk
        advection = SedimentPINN._dx(adv_x, xyt) + SedimentPINN._dy(adv_y, xyt)

        # ③ 扩散项：∂(ε h ∂C/∂x)/∂x + ∂(ε h ∂C/∂y)/∂y
        diff_x = epsilon_thk * h * SedimentPINN._dx(C_tk, xyt)
        diff_y = epsilon_thk * h * SedimentPINN._dy(C_tk, xyt)
        diffusion = SedimentPINN._dx(diff_x, xyt) + SedimentPINN._dy(diff_y, xyt)

        return storage_t + advection - diffusion - E_tk + D_tk

    @staticmethod
    def total_load_loss(
        xyt, h, u, v, C_tk, beta_tk, epsilon_thk, E_tk, D_tk,
    ) -> tuple:
        """总输沙 PDE 损失：MSE(R_Ck)，返回 (标量损失, 残差张量)。"""
        residual = SedimentPINN.total_load_residual(
            xyt, h, u, v, C_tk, beta_tk, epsilon_thk, E_tk, D_tk
        )
        return torch.mean(residual ** 2), residual

    @staticmethod
    def grass_formula(u: torch.Tensor, v: torch.Tensor,
                      Ag=0.001, m=3, eps=1e-6):
        """Grass 简化输沙通量公式。

        qx = Ag * u * |U|^(m-1)
        qy = Ag * v * |U|^(m-1)

        返回 (qx, qy, |U|)。
        """
        vel_mag = torch.sqrt(u ** 2 + v ** 2 + eps)
        factor = Ag * torch.pow(vel_mag, m - 1)
        qx = factor * u
        qy = factor * v
        return qx, qy, vel_mag


class BedPINN(BasePINN):
    """床面演变 PINN：输入 (x, y, t)，输出河床高程 z_b。

    输出经 tanh 缩放至 [-zb_scale, +zb_scale]，代表绝对河床高程变化。
    """

    def __init__(self, input_dim=3, hidden_dim=64, num_block=4,
                 output_dim=1, zb_scale=1.5):
        super().__init__(input_dim, hidden_dim, num_block, output_dim)
        self.zb_scale = zb_scale   # 河床高程物理范围上界 (m)

    def forward(self, xyT: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(xyT)
        for block in self.res_blocks:
            x = block(x)
        out = self.output_layer(x)
        return torch.tanh(out) * self.zb_scale   # 绝对 zb ∈ [-1.5, +1.5] m


class GradationPINN(BasePINN):
    """床沙级配 PINN：输入 (x, y, t)，输出各粒径级体积分数 p_k。

    forward 使用 softmax，天然满足：
        p_k ≥ 0,  Σ_k p_k = 1

    类中还包含活动层厚度闭合、级配守恒残差等静态方法。
    """

    def __init__(self, input_dim=3, hidden_dim=64, num_block=4, output_dim=1):
        super().__init__(input_dim, hidden_dim, num_block, output_dim)

    def forward(self, xyt: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(xyt)
        for block in self.res_blocks:
            x = block(x)
        raw_p = self.output_layer(x)
        # softmax 保证每个空间点的粒径比例非负且和为 1
        return torch.softmax(raw_p, dim=-1)

    @staticmethod
    def gradation_constraint(p_k: torch.Tensor) -> torch.Tensor:
        """级配归一约束残差：Σ_k p_k - 1。理想值为 0。"""
        return torch.sum(p_k, dim=1, keepdim=True) - 1.0

    @staticmethod
    def active_layer_thickness(d_k, p_k=None, alpha_a=1.0) -> torch.Tensor:
        """活动层厚度 L_a 简化闭合。

        若提供 p_k，则用级配加权平均粒径代表活动层尺度；
        否则用最大粒径。
        工程上也可替换为 L_a = alpha_a * D90。
        """
        if not torch.is_tensor(d_k):
            like = p_k if p_k is not None else None
            d_k = torch.as_tensor(d_k, dtype=like.dtype, device=like.device) \
                if like is not None else torch.as_tensor(d_k)
        if p_k is not None:
            d_k = d_k.to(dtype=p_k.dtype, device=p_k.device).view(1, -1)
            d_ref = torch.sum(p_k * d_k, dim=1, keepdim=True)
        else:
            d_ref = torch.max(d_k).view(1, 1)
        return alpha_a * d_ref

    @staticmethod
    def active_layer_residual(xyt, p_k, L_a, E_k, D_k, rho_s=2650.0):
        """活动层级配守恒残差。

        简化形式：
            R_pk = ∂(L_a p_k)/∂t - (D_k - E_k)/ρ_s
                   + p_k * Σ_j((D_j - E_j)/ρ_s)

        含义：某粒径级的活动层储量变化 = 该粒径级净沉积量，
        并扣除总床面升降对各粒径比例的稀释/富集影响。
        """
        if not torch.is_tensor(rho_s):
            rho_s = torch.as_tensor(rho_s, dtype=p_k.dtype, device=p_k.device)

        exchange = (D_k - E_k) / (rho_s + 1e-8)     # 净沉积体积通量
        total_exchange = torch.sum(exchange, dim=1, keepdim=True)
        storage = L_a * p_k
        storage_t = SedimentPINN._dt(storage, xyt)   # ∂(L_a p_k)/∂t

        return storage_t - exchange + p_k * total_exchange

    @staticmethod
    def active_layer_loss(xyt, p_k, L_a, E_k, D_k,
                          rho_s=2650.0, w_sum=1.0):
        """活动层级配 PDE 损失：MSE(R_pk) + 归一约束。"""
        residual = GradationPINN.active_layer_residual(
            xyt, p_k, L_a, E_k, D_k, rho_s
        )
        sum_residual = GradationPINN.gradation_constraint(p_k)
        loss = torch.mean(residual ** 2) + w_sum * torch.mean(sum_residual ** 2)
        return loss, residual


class CoupledFlowSedimentPINN(nn.Module):
    """四网络耦合容器。

    将 FlowPINN / SedimentPINN / BedPINN / GradationPINN
    封装为一个模块，方便后续全耦合训练或联合推理。

    目前训练流程仍采用解耦方式，此类仅提供统一入口。
    """

    def __init__(self, num_grain_classes=1, input_dim=3, hidden_dim=64,
                 num_block=4, zb_scale=1.5):
        super().__init__()
        self.flow_net = FlowPINN(
            input_dim=input_dim, hidden_dim=hidden_dim,
            num_block=num_block, output_dim=3,
        )
        self.sediment_net = SedimentPINN(
            input_dim=input_dim, hidden_dim=hidden_dim,
            num_block=num_block, output_dim=num_grain_classes,
        )
        self.bed_net = BedPINN(
            input_dim=input_dim, hidden_dim=hidden_dim,
            num_block=num_block, output_dim=1, zb_scale=zb_scale,
        )
        self.gradation_net = GradationPINN(
            input_dim=input_dim, hidden_dim=hidden_dim,
            num_block=num_block, output_dim=num_grain_classes,
        )

    def forward(self, xyt: torch.Tensor) -> dict:
        """一次前向得到全部物理量，以字典形式返回。"""
        flow = self.flow_net(xyt)
        return {
            'flow': flow,
            'h': flow[:, 0:1],       # 水深
            'u': flow[:, 1:2],       # x 流速
            'v': flow[:, 2:3],       # y 流速
            'C_tk': self.sediment_net(xyt),
            'zb': self.bed_net(xyt),
            'p_k': self.gradation_net(xyt),
        }
