"""二维水沙 PINN 的可微 PDE 残差与闭合公式库。

本模块按照 HEC-RAS 类二维水沙计算流程，把公式按类型组织：

1. 水动力：水面高程、SWE/DWE 残差、Manning 床面剪切力。
2. 输沙能力：Shields 参数、临界剪切力、掩蔽暴露和若干输沙能力闭合。
3. 落速与推悬分配：沉降速度、Rouse 数、悬移/推移比例。
4. 总输沙方程：对流-扩散-源汇形式的泥沙输移残差。
5. 侵蚀/沉积：非黏性适应公式、黏性侵蚀与沉积公式。
6. 床面演变：Exner 类床面变化与床坡修正。
7. 床面更新：活动层级配、孔隙率和糙率辅助公式。

所有张量函数均面向 PyTorch autograd 编写。除特别说明外，标量场形状
为 (N, 1)，多粒径场形状为 (N, K)。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

Tensor = torch.Tensor


# -----------------------------------------------------------------------------
# 通用张量与自动微分辅助函数
# -----------------------------------------------------------------------------


def _as_tensor(value, like: Optional[Tensor] = None) -> Tensor:
    """把标量或序列转换为张量，并尽量与 ``like`` 保持相同 dtype/device。"""
    if torch.is_tensor(value):
        return value
    if like is None:
        return torch.as_tensor(value, dtype=torch.float32)
    return torch.as_tensor(value, dtype=like.dtype, device=like.device)


def safe_sqrt(x: Tensor, eps: float = 1.0e-12) -> Tensor:
    """安全开平方：先把输入裁剪到 eps 以上，避免 sqrt(负数/零) 导致 NaN。"""
    return torch.sqrt(torch.clamp(x, min=eps))


def safe_div(num: Tensor, den: Tensor, eps: float = 1.0e-12) -> Tensor:
    """安全除法：分母加 eps，避免水深、速度等接近 0 时除零。"""
    return num / (den + eps)


def velocity_magnitude(u: Tensor, v: Tensor, eps: float = 1.0e-12) -> Tensor:
    """二维流速模长 U = sqrt(u^2 + v^2)。"""
    return safe_sqrt(u * u + v * v, eps)


def grad(outputs: Tensor, inputs: Tensor, dim: int) -> Tensor:
    """用 autograd 计算偏导数 d(outputs)/d(inputs[:, dim])。"""
    if not inputs.requires_grad:
        raise ValueError("inputs must have requires_grad=True for PDE residuals")
    g = torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )[0]
    if g is None:
        return torch.zeros_like(outputs)
    return g[:, dim : dim + 1]


def dx(q: Tensor, xyt: Tensor) -> Tensor:
    """对 x 求偏导；约定 xyt[:, 0] 是 x。"""
    return grad(q, xyt, 0)


def dy(q: Tensor, xyt: Tensor) -> Tensor:
    """对 y 求偏导；约定 xyt[:, 1] 是 y。"""
    return grad(q, xyt, 1)


def dt(q: Tensor, xyt: Tensor) -> Tensor:
    """对 t 求偏导；约定 xyt[:, 2] 是 t。"""
    return grad(q, xyt, 2)


def divergence(qx: Tensor, qy: Tensor, xyt: Tensor) -> Tensor:
    """二维散度 div(q) = dqx/dx + dqy/dy。"""
    return dx(qx, xyt) + dy(qy, xyt)


def mse_zero(*residuals: Tensor) -> Tensor:
    """把一个或多个 PDE 残差与 0 做均方误差，用作物理损失。"""
    losses = [torch.mean(r * r) for r in residuals if r is not None]
    if not losses:
        raise ValueError("at least one residual is required")
    return sum(losses)


# -----------------------------------------------------------------------------
# 参数配置
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class FluidParams:
    """水动力参数：控制浅水方程、摩阻和落速计算。"""

    g: float = 9.81  # 重力加速度 [m/s^2]
    rho_w: float = 1000.0  # 水密度 [kg/m^3]
    nu: float = 1.0e-6  # 水的运动黏度 [m^2/s]
    kappa: float = 0.4  # von Karman 常数，用于 Rouse 数
    manning_n: float = 0.03  # Manning 糙率系数
    eddy_viscosity: float = 0.0  # 湍流涡黏性；0 表示不额外加黏性项


@dataclass(frozen=True)
class SedimentParams:
    """泥沙参数：控制多粒径、输沙能力、扩散和床面演变。"""

    grain_diameters: Sequence[float]  # 各粒径级代表粒径 d_k [m]
    rho_s: float = 2650.0  # 颗粒密度 [kg/m^3]
    porosity: float = 0.40  # 床沙孔隙率 lambda_p [-]
    active_layer_factor: float = 1.0  # 活动层厚度系数，L_a = factor * D90
    adaptation_alpha: float = 1.0  # 非黏性适应系数 alpha
    schmidt: float = 1.0  # Schmidt 数，用于悬移质扩散
    bedload_diffusion_coeff: float = 0.01  # 推移质扩散经验系数 c_b
    horizontal_mixing_coeff: float = 1.0  # 悬移质水平混合经验系数 c_m
    critical_shields: float = 0.047  # 默认临界 Shields 数


@dataclass(frozen=True)
class CohesiveParams:
    """黏性泥沙参数：控制表面侵蚀、块体侵蚀和沉积概率。"""

    erosion_coeff: float = 1.0e-5  # 表面侵蚀系数 M
    mass_erosion_coeff: float = 1.0e-4  # 块体侵蚀系数 M_M
    tau_ce: float = 0.2  # 表面侵蚀临界剪切力 [Pa]
    tau_cm: float = 1.0  # 块体侵蚀临界剪切力 [Pa]
    tau_cd: Optional[float] = None  # 沉积临界剪切力；None 时不启用 Krone 概率
    continuous_deposition: bool = True  # True 表示 P_d=1 的连续沉积


# -----------------------------------------------------------------------------
# 1. 水动力公式
# -----------------------------------------------------------------------------


class Hydrodynamics:
    """水面高程、床面剪切力、二维浅水方程和扩散波方程。"""

    @staticmethod
    def water_surface(h: Tensor, zb: Tensor) -> Tensor:
        """水面高程：eta = h + zb。"""
        return h + zb

    @staticmethod
    def manning_friction_slope(h: Tensor, u: Tensor, v: Tensor, n: float, eps: float = 1.0e-8) -> Tensor:
        """Manning 摩阻坡降：Sf = n^2 U^2 / h^(4/3)。"""
        U = velocity_magnitude(u, v, eps)
        return safe_div((n * n) * U * U, torch.clamp(h, min=eps).pow(4.0 / 3.0), eps)

    @staticmethod
    def manning_bed_shear(
        h: Tensor,
        u: Tensor,
        v: Tensor,
        n: float,
        rho_w: float = 1000.0,
        g: float = 9.81,
        eps: float = 1.0e-8,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Manning 床面剪切力：tau_bx=rho*g*n^2*u*U/h^(1/3)。"""
        U = velocity_magnitude(u, v, eps)
        h13 = torch.clamp(h, min=eps).pow(1.0 / 3.0)
        coeff = rho_w * g * n * n
        tau_bx = safe_div(coeff * u * U, h13, eps)
        tau_by = safe_div(coeff * v * U, h13, eps)
        tau_b = velocity_magnitude(tau_bx, tau_by, eps)
        return tau_bx, tau_by, tau_b

    @staticmethod
    def shear_velocity(tau_b: Tensor, rho_w: float = 1000.0, eps: float = 1.0e-12) -> Tensor:
        """剪切速度：u_* = sqrt(tau_b / rho_w)。"""
        return safe_sqrt(tau_b / rho_w, eps)

    @staticmethod
    def shallow_water_residuals(
        xyt: Tensor,
        h: Tensor,
        u: Tensor,
        v: Tensor,
        zb: Tensor,
        tau_bx: Tensor,
        tau_by: Tensor,
        fluid: FluidParams = FluidParams(),
        q_source: Optional[Tensor] = None,
        nu_t: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """保守型二维浅水方程残差，包含 Manning 摩阻和可选黏性项。

        R_h = h_t + (hu)_x + (hv)_y - q_source
        R_u = (hu)_t + (hu^2+0.5gh^2)_x + (huv)_y + gh zb_x + tau_bx/rho_w - visc_u
        R_v = (hv)_t + (huv)_x + (hv^2+0.5gh^2)_y + gh zb_y + tau_by/rho_w - visc_v
        """
        g = fluid.g
        rho_w = fluid.rho_w
        q_source = torch.zeros_like(h) if q_source is None else q_source
        nu_t = torch.zeros_like(h) + fluid.eddy_viscosity if nu_t is None else nu_t

        hu = h * u
        hv = h * v
        R_h = dt(h, xyt) + dx(hu, xyt) + dy(hv, xyt) - q_source

        visc_u = divergence(h * nu_t * dx(u, xyt), h * nu_t * dy(u, xyt), xyt)
        visc_v = divergence(h * nu_t * dx(v, xyt), h * nu_t * dy(v, xyt), xyt)

        R_u = (
            dt(hu, xyt)
            + dx(h * u * u + 0.5 * g * h * h, xyt)
            + dy(h * u * v, xyt)
            + g * h * dx(zb, xyt)
            + tau_bx / rho_w
            - visc_u
        )
        R_v = (
            dt(hv, xyt)
            + dx(h * u * v, xyt)
            + dy(h * v * v + 0.5 * g * h * h, xyt)
            + g * h * dy(zb, xyt)
            + tau_by / rho_w
            - visc_v
        )
        return R_h, R_u, R_v

    @staticmethod
    def diffusion_wave_residuals(
        xyt: Tensor,
        h: Tensor,
        u: Tensor,
        v: Tensor,
        zb: Tensor,
        n: float,
        q_source: Optional[Tensor] = None,
        eps: float = 1.0e-8,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """扩散波残差：质量守恒 + 摩阻坡降与水面坡降平衡 Sf + grad(eta)=0。"""
        q_source = torch.zeros_like(h) if q_source is None else q_source
        eta = Hydrodynamics.water_surface(h, zb)
        U = velocity_magnitude(u, v, eps)
        sf_coeff = safe_div(n * n * U, torch.clamp(h, min=eps).pow(4.0 / 3.0), eps)
        sf_x = sf_coeff * u
        sf_y = sf_coeff * v
        R_h = dt(h, xyt) + dx(h * u, xyt) + dy(h * v, xyt) - q_source
        R_mx = sf_x + dx(eta, xyt)
        R_my = sf_y + dy(eta, xyt)
        return R_h, R_mx, R_my


# -----------------------------------------------------------------------------
# 2. 输沙能力与掩蔽暴露公式
# -----------------------------------------------------------------------------


class TransportCapacity:
    """输沙能力闭合公式；复杂 HEC-RAS 公式通过 dispatch 接口保留。"""

    @staticmethod
    def submerged_specific_gravity(rho_s: Tensor, rho_w: float = 1000.0) -> Tensor:
        return rho_s / rho_w - 1.0

    @staticmethod
    def shields_parameter(tau_b: Tensor, d: Tensor, rho_s: float = 2650.0, rho_w: float = 1000.0, g: float = 9.81) -> Tensor:
        """Shields 参数：theta = tau_b / ((rho_s-rho_w) g d)。"""
        d = _as_tensor(d, tau_b)
        return safe_div(tau_b, (rho_s - rho_w) * g * d)

    @staticmethod
    def critical_shear_stress(
        d: Tensor,
        theta_cr: float = 0.047,
        rho_s: float = 2650.0,
        rho_w: float = 1000.0,
        g: float = 9.81,
    ) -> Tensor:
        """临界剪切力：tau_cr = theta_cr (rho_s-rho_w) g d。"""
        d = _as_tensor(d)
        return theta_cr * (rho_s - rho_w) * g * d

    @staticmethod
    def hiding_exposure_power(d: Tensor, d_ref: Tensor, exponent: float = 0.6) -> Tensor:
        """通用幂函数掩蔽/暴露因子：xi=(d/d_ref)^(-m)。"""
        d_ref = _as_tensor(d_ref, d)
        return torch.clamp(safe_div(d, d_ref), min=1.0e-8).pow(-exponent)

    @staticmethod
    def apply_hiding_to_critical_shear(tau_cr: Tensor, xi: Tensor) -> Tensor:
        """用掩蔽暴露因子修正临界剪切力：tau_cr' = xi tau_cr。"""
        return xi * tau_cr

    @staticmethod
    def apply_hiding_to_capacity(q_star: Tensor, xi: Tensor) -> Tensor:
        """用掩蔽暴露因子修正输沙能力：q_star' = xi q_star。"""
        return xi * q_star

    @staticmethod
    def meyer_peter_muller(
        h: Tensor,
        u: Tensor,
        v: Tensor,
        tau_b: Tensor,
        d: Tensor,
        rho_s: float = 2650.0,
        rho_w: float = 1000.0,
        g: float = 9.81,
        theta_cr: float = 0.047,
    ) -> Tensor:
        """Meyer-Peter and Muller 推移质单位宽输沙能力 q_b [m^2/s]。

        Phi = 8 (theta-theta_cr)^(3/2), q_b = Phi sqrt((s-1) g d^3)。
        返回体积输沙率；如需质量输沙率，可再乘泥沙密度。
        """
        d = _as_tensor(d, tau_b)
        theta = TransportCapacity.shields_parameter(tau_b, d, rho_s, rho_w, g)
        s_minus_1 = rho_s / rho_w - 1.0
        phi = 8.0 * F.relu(theta - theta_cr).pow(1.5)
        return phi * safe_sqrt(s_minus_1 * g * d.pow(3.0))

    @staticmethod
    def engelund_hansen(
        h: Tensor,
        u: Tensor,
        v: Tensor,
        tau_b: Tensor,
        d: Tensor,
        rho_s: float = 2650.0,
        rho_w: float = 1000.0,
        g: float = 9.81,
    ) -> Tensor:
        """简化 Engelund-Hansen 总输沙单位宽输沙能力。

        Phi = 0.05 theta^(5/2) / Cf, q_t = Phi sqrt((s-1) g d^3)。
        Cf 用 tau/(rho U^2) 估计。该紧凑形式用于 PINN 可微闭合，
        不是对 HEC-RAS 标定公式的完整替代。
        """
        d = _as_tensor(d, tau_b)
        U = velocity_magnitude(u, v)
        theta = TransportCapacity.shields_parameter(tau_b, d, rho_s, rho_w, g)
        cf = torch.clamp(safe_div(tau_b, rho_w * U * U), min=1.0e-6)
        phi = 0.05 * theta.pow(2.5) / cf
        return phi * safe_sqrt((rho_s / rho_w - 1.0) * g * d.pow(3.0))

    @staticmethod
    def soulsby_van_rijn(
        h: Tensor,
        u: Tensor,
        v: Tensor,
        tau_b: Tensor,
        d: Tensor,
        rho_s: float = 2650.0,
        rho_w: float = 1000.0,
        g: float = 9.81,
        theta_cr: float = 0.047,
    ) -> Tensor:
        """基于超额 Shields 应力的紧凑 Soulsby-van Rijn 型输沙能力。"""
        d = _as_tensor(d, tau_b)
        theta = TransportCapacity.shields_parameter(tau_b, d, rho_s, rho_w, g)
        excess = F.relu(theta - theta_cr)
        return 0.012 * excess.pow(1.5) * safe_sqrt((rho_s / rho_w - 1.0) * g * d.pow(3.0))

    @staticmethod
    def wu_et_al(
        h: Tensor,
        u: Tensor,
        v: Tensor,
        tau_b: Tensor,
        d: Tensor,
        p: Optional[Tensor] = None,
        rho_s: float = 2650.0,
        rho_w: float = 1000.0,
        g: float = 9.81,
        theta_cr: float = 0.047,
        hiding_exponent: float = 0.6,
    ) -> Tensor:
        """带幂函数掩蔽暴露修正的 Wu 类多粒径输沙能力。"""
        d = _as_tensor(d, tau_b)
        if p is None:
            d_ref = torch.mean(d)
        else:
            d_ref = torch.sum(p * d, dim=-1, keepdim=True)
        xi = TransportCapacity.hiding_exposure_power(d, d_ref, hiding_exponent)
        tau_cr = TransportCapacity.critical_shear_stress(d, theta_cr, rho_s, rho_w, g)
        tau_cr_corr = TransportCapacity.apply_hiding_to_critical_shear(tau_cr, xi)
        theta_corr = safe_div(tau_cr_corr, (rho_s - rho_w) * g * d)
        return TransportCapacity.meyer_peter_muller(h, u, v, tau_b, d, rho_s, rho_w, g, float(theta_cr)) * F.relu(
            safe_div(TransportCapacity.shields_parameter(tau_b, d, rho_s, rho_w, g), theta_corr) - 1.0
        )

    @staticmethod
    def capacity(
        method: str,
        h: Tensor,
        u: Tensor,
        v: Tensor,
        tau_b: Tensor,
        d: Tensor,
        p: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """按名称分发输沙能力公式。

        已实现紧凑可微闭合：mpm、engelund_hansen、soulsby_van_rijn、wu、
        van_rijn（初版中暂用 soulsby_van_rijn 形式）。为保持 HEC-RAS 选项
        命名兼容，未实现的标定公式会抛出 NotImplementedError。
        """
        name = method.lower().replace("-", "_").replace(" ", "_")
        if name in {"mpm", "meyer_peter_muller", "meyer_peter_and_muller"}:
            return TransportCapacity.meyer_peter_muller(h, u, v, tau_b, d, **kwargs)
        if name in {"engelund_hansen", "england_and_hansen"}:
            return TransportCapacity.engelund_hansen(h, u, v, tau_b, d, **kwargs)
        if name in {"soulsby_van_rijn", "soulsby"}:
            return TransportCapacity.soulsby_van_rijn(h, u, v, tau_b, d, **kwargs)
        if name in {"van_rijn", "vanrijn"}:
            return TransportCapacity.soulsby_van_rijn(h, u, v, tau_b, d, **kwargs)
        if name in {"wu", "wu_et_al", "wu_2000"}:
            return TransportCapacity.wu_et_al(h, u, v, tau_b, d, p=p, **kwargs)
        raise NotImplementedError(
            f"Transport formula '{method}' is listed for HEC-RAS compatibility, "
            "but this module only includes compact differentiable closures for "
            "MPM, Engelund-Hansen, Soulsby-van Rijn/van Rijn, and Wu-like forms."
        )


# -----------------------------------------------------------------------------
# 3. 落速、悬移比例、修正系数与扩散
# -----------------------------------------------------------------------------


class SedimentMode:
    """沉降速度、Rouse 数、推悬分配、beta 修正和扩散系数。"""

    @staticmethod
    def fall_velocity_stokes(d: Tensor, rho_s: float = 2650.0, rho_w: float = 1000.0, nu: float = 1.0e-6, g: float = 9.81) -> Tensor:
        """细颗粒 Stokes 沉降速度。"""
        d = _as_tensor(d)
        return (rho_s / rho_w - 1.0) * g * d * d / (18.0 * nu)

    @staticmethod
    def fall_velocity_soulsby(d: Tensor, rho_s: float = 2650.0, rho_w: float = 1000.0, nu: float = 1.0e-6, g: float = 9.81) -> Tensor:
        """Soulsby 型显式沉降速度。"""
        d = _as_tensor(d)
        s_minus_1 = rho_s / rho_w - 1.0
        scale = torch.as_tensor((s_minus_1 * g) / (nu * nu), dtype=d.dtype, device=d.device)
        d_star = d * scale.pow(1.0 / 3.0)
        return (nu / d) * (safe_sqrt(10.36 * 10.36 + 1.049 * d_star.pow(3.0)) - 10.36)

    @staticmethod
    def fall_velocity_rubey(d: Tensor, rho_s: float = 2650.0, rho_w: float = 1000.0, nu: float = 1.0e-6, g: float = 9.81) -> Tensor:
        """Rubey 型混合沉降速度。"""
        d = _as_tensor(d)
        s_minus_1 = rho_s / rho_w - 1.0
        return safe_sqrt((2.0 / 3.0) * s_minus_1 * g * d + 36.0 * nu * nu / (d * d)) - 6.0 * nu / d

    @staticmethod
    def fall_velocity(method: str, d: Tensor, **kwargs) -> Tensor:
        """按名称分发落速公式。

        已实现：stokes、rubey、soulsby。PINN 原型中，Dietrich、Toffaleti、
        Report 12、van Rijn、Wu-Wang 可在后续替换实现，不影响残差代码。
        """
        name = method.lower().replace("-", "_").replace(" ", "_")
        if name == "stokes":
            return SedimentMode.fall_velocity_stokes(d, **kwargs)
        if name == "rubey":
            return SedimentMode.fall_velocity_rubey(d, **kwargs)
        if name in {"soulsby", "van_rijn", "wu_wang", "dietrich", "toffaleti", "report_12"}:
            return SedimentMode.fall_velocity_soulsby(d, **kwargs)
        raise NotImplementedError(f"Unknown fall velocity method: {method}")

    @staticmethod
    def rouse_number(omega: Tensor, u_star: Tensor, kappa: float = 0.4) -> Tensor:
        """Rouse 数：P = omega / (kappa u_*)。"""
        return safe_div(omega, kappa * u_star)

    @staticmethod
    def suspended_fraction_capacity(q_s_star: Tensor, q_t_star: Tensor) -> Tensor:
        """输沙能力法计算悬移比例：f_s = q_s*/q_t*。"""
        return torch.clamp(safe_div(q_s_star, q_t_star), 0.0, 1.0)

    @staticmethod
    def suspended_fraction_rouse(omega: Tensor, u_star: Tensor, kappa: float = 0.4, sharpness: float = 4.0) -> Tensor:
        """基于 Rouse 数的平滑 Greimann/van-Rijn 型悬移比例近似。"""
        P = SedimentMode.rouse_number(omega, u_star, kappa)
        return torch.sigmoid(sharpness * (1.0 - P))

    @staticmethod
    def bedload_fraction(f_s: Tensor) -> Tensor:
        return 1.0 - torch.clamp(f_s, 0.0, 1.0)

    @staticmethod
    def bedload_correction(u_b: Tensor, U: Tensor) -> Tensor:
        """推移质修正系数：beta_b = u_b/U。"""
        return torch.clamp(safe_div(u_b, U), min=0.0)

    @staticmethod
    def total_load_correction(f_s: Tensor, beta_s: Tensor, beta_b: Tensor) -> Tensor:
        """总输沙修正系数：beta_t = f_s beta_s + (1-f_s) beta_b。"""
        f_s = torch.clamp(f_s, 0.0, 1.0)
        return f_s * beta_s + (1.0 - f_s) * beta_b

    @staticmethod
    def suspended_diffusion(u_star: Tensor, h: Tensor, c_m: float = 1.0, schmidt: float = 1.0) -> Tensor:
        """悬移质水平扩散系数：epsilon_s = c_m u_* h / Sc。"""
        return c_m * u_star * h / schmidt

    @staticmethod
    def bedload_diffusion(u_star: Tensor, d: Tensor, c_b: float = 0.01) -> Tensor:
        """推移质水平扩散系数：epsilon_b = c_b u_* d。"""
        d = _as_tensor(d, u_star)
        return c_b * u_star * d

    @staticmethod
    def total_diffusion(f_s: Tensor, eps_s: Tensor, eps_b: Tensor) -> Tensor:
        """总输沙扩散系数：epsilon_t = f_s epsilon_s + (1-f_s) epsilon_b。"""
        f_s = torch.clamp(f_s, 0.0, 1.0)
        return f_s * eps_s + (1.0 - f_s) * eps_b


# -----------------------------------------------------------------------------
# 4. 泥沙输移 PDE
# -----------------------------------------------------------------------------


class SedimentTransportPDE:
    """推移质、悬移质和总输沙输移方程残差。"""

    @staticmethod
    def transport_residual(
        xyt: Tensor,
        h: Tensor,
        u: Tensor,
        v: Tensor,
        C: Tensor,
        beta: Tensor,
        epsilon: Tensor,
        E: Tensor,
        D: Tensor,
    ) -> Tensor:
        """泥沙输移残差：R_C = (hC)_t + div(h beta U C) - div(h eps grad C) - E + D。"""
        adv_x = h * beta * u * C
        adv_y = h * beta * v * C
        diff_x = h * epsilon * dx(C, xyt)
        diff_y = h * epsilon * dy(C, xyt)
        return dt(h * C, xyt) + divergence(adv_x, adv_y, xyt) - divergence(diff_x, diff_y, xyt) - E + D

    @staticmethod
    def total_load_residual(
        xyt: Tensor,
        h: Tensor,
        u: Tensor,
        v: Tensor,
        Ck: Tensor,
        beta_tk: Tensor,
        epsilon_tk: Tensor,
        Ek: Tensor,
        Dk: Tensor,
    ) -> Tensor:
        """K 个粒径级的向量化总输沙残差。"""
        residuals = []
        for k in range(Ck.shape[1]):
            residuals.append(
                SedimentTransportPDE.transport_residual(
                    xyt,
                    h,
                    u,
                    v,
                    Ck[:, k : k + 1],
                    beta_tk[:, k : k + 1],
                    epsilon_tk[:, k : k + 1],
                    Ek[:, k : k + 1],
                    Dk[:, k : k + 1],
                )
            )
        return torch.cat(residuals, dim=1)

    bedload_residual = transport_residual
    suspended_load_residual = transport_residual


# -----------------------------------------------------------------------------
# 5. 侵蚀与沉积公式
# -----------------------------------------------------------------------------


class ExchangeTerms:
    """非黏性和黏性泥沙的侵蚀/沉积项。"""

    @staticmethod
    def equilibrium_concentration(q_star: Tensor, h: Tensor, u: Tensor, v: Tensor, eps: float = 1.0e-8) -> Tensor:
        """平衡浓度：C* = q*/(h U + eps)。"""
        U = velocity_magnitude(u, v, eps)
        return safe_div(q_star, h * U, eps)

    @staticmethod
    def noncohesive_net_exchange(alpha: Tensor, omega: Tensor, C_star: Tensor, C: Tensor) -> Tensor:
        """净交换项：E-D = alpha omega (C* - C)，正值表示侵蚀占优。"""
        return alpha * omega * (C_star - C)

    @staticmethod
    def split_net_exchange(net: Tensor) -> Tuple[Tensor, Tensor]:
        """把净交换项 E-D 拆分为非负的 E 和 D。"""
        E = F.relu(net)
        D = F.relu(-net)
        return E, D

    @staticmethod
    def noncohesive_adaptation(
        q_star: Tensor,
        h: Tensor,
        u: Tensor,
        v: Tensor,
        C: Tensor,
        omega: Tensor,
        alpha: float | Tensor = 1.0,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """非黏性适应公式闭合，返回 E、D 和 C*。"""
        alpha = _as_tensor(alpha, C)
        C_star = ExchangeTerms.equilibrium_concentration(q_star, h, u, v)
        net = ExchangeTerms.noncohesive_net_exchange(alpha, omega, C_star, C)
        E, D = ExchangeTerms.split_net_exchange(net)
        return E, D, C_star

    @staticmethod
    def cohesive_surface_erosion(tau_b: Tensor, M: float = 1.0e-5, tau_ce: float = 0.2) -> Tensor:
        """Partheniades/Ariathurai 型黏性表面侵蚀：E = M max(tau/tau_ce - 1, 0)。"""
        return M * F.relu(tau_b / tau_ce - 1.0)

    @staticmethod
    def cohesive_piecewise_erosion(
        tau_b: Tensor,
        M: float = 1.0e-5,
        M_mass: float = 1.0e-4,
        tau_ce: float = 0.2,
        tau_cm: float = 1.0,
    ) -> Tensor:
        """tau_cm 以下为表面侵蚀，超过 tau_cm 后叠加块体侵蚀。"""
        surface = M * F.relu(torch.minimum(tau_b, torch.as_tensor(tau_cm, dtype=tau_b.dtype, device=tau_b.device)) / tau_ce - 1.0)
        mass = M_mass * F.relu(tau_b / tau_cm - 1.0)
        return surface + mass

    @staticmethod
    def cohesive_deposition(
        C: Tensor,
        omega_f: Tensor,
        tau_b: Optional[Tensor] = None,
        tau_cd: Optional[float] = None,
        continuous: bool = True,
    ) -> Tensor:
        """Krone 型沉积：D = omega_f C P_d；HEC-RAS 2D 常采用 P_d=1 的连续沉积。"""
        if continuous or tau_b is None or tau_cd is None:
            P_d = torch.ones_like(C)
        else:
            P_d = torch.clamp(1.0 - tau_b / tau_cd, min=0.0, max=1.0)
        return omega_f * C * P_d

    @staticmethod
    def splash_sheet_erosion(E_total: Tensor, grain_weights: Tensor, dry_fraction: Tensor) -> Tensor:
        """将片流/雨滴溅蚀潜力按粒径权重分配到各粒径级。"""
        weights = safe_div(grain_weights, torch.sum(grain_weights, dim=-1, keepdim=True))
        return dry_fraction * E_total * weights


# -----------------------------------------------------------------------------
# 6. 床面演变公式
# -----------------------------------------------------------------------------


class BedEvolution:
    """Exner 类床面变化和床坡修正。"""

    @staticmethod
    def slope_flux(qb_mag: Tensor, dzb_dx: Tensor, dzb_dy: Tensor, kappa_s: float = 1.0) -> Tuple[Tensor, Tensor]:
        """床坡通量：q_slope = -kappa_s |q_b| grad(zb)。"""
        return -kappa_s * qb_mag * dzb_dx, -kappa_s * qb_mag * dzb_dy

    @staticmethod
    def exner_residual(
        xyt: Tensor,
        zb: Tensor,
        E: Tensor,
        D: Tensor,
        rho_s: float | Tensor = 2650.0,
        porosity: float | Tensor = 0.40,
        q_slope_x: Optional[Tensor] = None,
        q_slope_y: Optional[Tensor] = None,
    ) -> Tensor:
        """床面演变残差：R_z=(1-lambda) zb_t - sum((D-E)/rho_s) + div(q_slope)。"""
        rho_s = _as_tensor(rho_s, E)
        porosity = _as_tensor(porosity, E)
        source = torch.sum(safe_div(D - E, rho_s), dim=1, keepdim=True)
        slope_div = torch.zeros_like(zb)
        if q_slope_x is not None and q_slope_y is not None:
            slope_div = divergence(q_slope_x, q_slope_y, xyt)
        return (1.0 - porosity) * dt(zb, xyt) - source + slope_div

    @staticmethod
    def bed_change_rate(E: Tensor, D: Tensor, rho_s: float | Tensor = 2650.0, porosity: float = 0.40) -> Tensor:
        """床面变化率：zb_t = sum((D-E)/rho_s)/(1-porosity)。"""
        rho_s = _as_tensor(rho_s, E)
        return torch.sum(safe_div(D - E, rho_s), dim=1, keepdim=True) / (1.0 - porosity)


# -----------------------------------------------------------------------------
# 7. 床沙组成、孔隙率和糙率更新
# -----------------------------------------------------------------------------


class BedUpdate:
    """活动层、级配、孔隙率和糙率辅助公式。"""

    @staticmethod
    def normalize_gradation(raw_p: Tensor) -> Tensor:
        """用 softmax 保证 p_k>=0 且 sum_k p_k=1。"""
        return torch.softmax(raw_p, dim=-1)

    @staticmethod
    def gradation_constraint(p: Tensor) -> Tensor:
        """级配归一约束残差：sum_k p_k - 1。"""
        return torch.sum(p, dim=-1, keepdim=True) - 1.0

    @staticmethod
    def active_layer_thickness(D90: Tensor, alpha_a: float = 1.0) -> Tensor:
        """活动层厚度：L_a = alpha_a D90。"""
        return alpha_a * D90

    @staticmethod
    def active_layer_residual(
        xyt: Tensor,
        p: Tensor,
        L_a: Tensor,
        E: Tensor,
        D: Tensor,
        rho_s: float | Tensor = 2650.0,
    ) -> Tensor:
        """活动层级配残差：R_pk = (L_a p_k)_t - (D-E)/rho_s + p_k sum_j((D-E)/rho_s)。"""
        rho_s = _as_tensor(rho_s, E)
        exchange = safe_div(D - E, rho_s)
        total_exchange = torch.sum(exchange, dim=1, keepdim=True)
        residuals = []
        for k in range(p.shape[1]):
            residuals.append(dt(L_a * p[:, k : k + 1], xyt) - exchange[:, k : k + 1] + p[:, k : k + 1] * total_exchange)
        return torch.cat(residuals, dim=1)

    @staticmethod
    def porosity_from_dry_density(rho_d: Tensor, rho_s: float = 2650.0) -> Tensor:
        """由干容重计算孔隙率：lambda_p = 1 - rho_d/rho_s。"""
        return torch.clamp(1.0 - rho_d / rho_s, min=0.0, max=0.95)

    @staticmethod
    def dry_density_from_porosity(porosity: Tensor, rho_s: float = 2650.0) -> Tensor:
        """由孔隙率计算干容重：rho_d = (1-lambda_p) rho_s。"""
        return (1.0 - porosity) * rho_s

    @staticmethod
    def limerinos_manning(h: Tensor, d84: Tensor, eps: float = 1.0e-8) -> Tensor:
        """Limerinos 型颗粒糙率 Manning n 近似。"""
        d84 = _as_tensor(d84, h)
        return torch.clamp(h.pow(1.0 / 6.0) / (20.0 * torch.log10(torch.clamp(h / d84, min=1.01)) + eps), min=0.005)

    @staticmethod
    def strickler_manning(d50: Tensor, coeff: float = 0.041) -> Tensor:
        """简单颗粒糙率公式：n = coeff d50^(1/6)。"""
        d50 = _as_tensor(d50)
        return coeff * torch.clamp(d50, min=1.0e-8).pow(1.0 / 6.0)


# -----------------------------------------------------------------------------
# PINN 输出拆包与分组损失辅助函数
# -----------------------------------------------------------------------------


def unpack_pinn_output(output: Tensor, num_grain_classes: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """拆包网络输出 [h,u,v,C_1..C_K,zb,p_1..p_K]，并施加物理范围约束。"""
    # K 是粒径级数量，决定 C_k 和 p_k 的列数。
    k = num_grain_classes

    # 输出顺序固定为：h,u,v,C_1..C_K,zb,p_1..p_K，因此总维度为 4+2K。
    expected = 4 + 2 * k
    if output.shape[1] != expected:
        raise ValueError(f"expected output_dim={expected}, got {output.shape[1]}")

    # h 必须非负，用 softplus 把网络原始输出映射到正值。
    h = F.softplus(output[:, 0:1])

    # u、v 可以为正或负，分别表示 x/y 方向流速。
    u = output[:, 1:2]
    v = output[:, 2:3]

    # C_k 是各粒径级水体泥沙浓度，必须非负。
    C = F.softplus(output[:, 3 : 3 + k])

    # zb 是床面高程，可正可负，不强制非负。
    zb = output[:, 3 + k : 4 + k]

    # p_k 是床沙级配比例，用 softmax 保证每行非负且总和为 1。
    p = BedUpdate.normalize_gradation(output[:, 4 + k : 4 + 2 * k])
    return h, u, v, C, zb, p


def compute_all_residuals(
    xyt: Tensor,
    model: Callable[[Tensor], Tensor],
    sediment: SedimentParams,
    fluid: FluidParams = FluidParams(),
    hydro_equation: str = "swe",
    transport_method: str = "wu",
    fall_velocity_method: str = "soulsby",
) -> dict[str, Tensor]:
    """计算一个完整水沙 PINN batch 的所有分组残差。"""
    # PINN 残差需要对 x、y、t 求导，因此采样点必须开启 autograd。
    xyt = xyt.requires_grad_(True)

    # 根据粒径级数量 K 拆包网络输出。
    K = len(sediment.grain_diameters)
    h, u, v, C, zb, p = unpack_pinn_output(model(xyt), K)

    # 第 1 类派生量：由水深、流速和 Manning n 计算床面剪切力。
    tau_bx, tau_by, tau_b = Hydrodynamics.manning_bed_shear(h, u, v, fluid.manning_n, fluid.rho_w, fluid.g)

    # 剪切速度 u_* 是输沙能力、Rouse 数、扩散系数的重要输入。
    u_star = Hydrodynamics.shear_velocity(tau_b, fluid.rho_w)

    # 粒径数组 d 的形状整理为 (1, K)，方便与 (N, K) 的 C、p 广播。
    d = _as_tensor(sediment.grain_diameters, h).view(1, K)

    # 第 2 类派生量：计算每个粒径级的沉降速度 omega_sk。
    omega = SedimentMode.fall_velocity(fall_velocity_method, d, rho_s=sediment.rho_s, rho_w=fluid.rho_w, nu=fluid.nu, g=fluid.g)
    omega = omega.expand_as(C)

    # 第 3 类派生量：计算各粒径级输沙能力 q*_tk。
    q_star = TransportCapacity.capacity(
        transport_method,
        h,
        u,
        v,
        tau_b,
        d,
        p=p,
        rho_s=sediment.rho_s,
        rho_w=fluid.rho_w,
        g=fluid.g,
        theta_cr=sediment.critical_shields,
    )
    # 如果某个输沙公式只返回一个总能力，则按床沙级配 p_k 分配到各粒径级。
    if q_star.shape[1] == 1 and K > 1:
        q_star = q_star * p

    # 第 4 类派生量：由 Rouse 数近似悬移比例 f_s，并得到推移比例 f_b。
    f_s = SedimentMode.suspended_fraction_rouse(omega, u_star, fluid.kappa)
    f_b = SedimentMode.bedload_fraction(f_s)

    # 当前初版暂不做复杂垂向速度/浓度修正，令 beta_s=beta_b=1。
    # 保留 beta_t 结构，后续可替换为 Rouse/指数剖面和推移质速度公式。
    beta_s = torch.ones_like(C)
    beta_b = torch.ones_like(C)
    beta_t = SedimentMode.total_load_correction(f_s, beta_s, beta_b)

    # 悬移质扩散、推移质扩散按 f_s/f_b 加权成总输沙扩散系数。
    eps_s = SedimentMode.suspended_diffusion(u_star, h, sediment.horizontal_mixing_coeff, sediment.schmidt).expand_as(C)
    eps_b = SedimentMode.bedload_diffusion(u_star, d, sediment.bedload_diffusion_coeff).expand_as(C)
    eps_t = SedimentMode.total_diffusion(f_s, eps_s, eps_b)

    # 第 5 类派生量：用非黏性适应公式计算侵蚀 E、沉积 D 和平衡浓度 C*。
    E, D, C_star = ExchangeTerms.noncohesive_adaptation(q_star, h, u, v, C, omega, sediment.adaptation_alpha)

    # 第 6 类残差：按用户选择计算水动力残差，可选 SWE 或 DWE。
    if hydro_equation.lower() == "dwe":
        R_h, R_u, R_v = Hydrodynamics.diffusion_wave_residuals(xyt, h, u, v, zb, fluid.manning_n)
    else:
        R_h, R_u, R_v = Hydrodynamics.shallow_water_residuals(xyt, h, u, v, zb, tau_bx, tau_by, fluid)

    # 第 7 类残差：总输沙对流-扩散-源汇方程残差 R_Ck。
    R_C = SedimentTransportPDE.total_load_residual(xyt, h, u, v, C, beta_t, eps_t, E, D)

    # 第 8 类残差：Exner 类床面演变残差 R_z。
    R_z = BedEvolution.exner_residual(xyt, zb, E, D, sediment.rho_s, sediment.porosity)

    # 这里用加权平均粒径近似 D90；严格 D90 后续可由累计级配插值得到。
    D90 = torch.sum(p * d, dim=1, keepdim=True)

    # 活动层厚度 L_a = factor * D90，用于床沙级配守恒残差。
    L_a = BedUpdate.active_layer_thickness(D90, sediment.active_layer_factor)

    # 第 9 类残差：活动层内各粒径级 p_k 的质量守恒残差。
    R_p = BedUpdate.active_layer_residual(xyt, p, L_a, E, D, sediment.rho_s)

    # 返回预测变量、派生物理量和所有残差，训练与诊断都从这里取。
    return {
        "h": h,
        "u": u,
        "v": v,
        "C": C,
        "zb": zb,
        "p": p,
        "tau_b": tau_b,
        "u_star": u_star,
        "omega": omega,
        "q_star": q_star,
        "C_star": C_star,
        "E": E,
        "D": D,
        "R_h": R_h,
        "R_u": R_u,
        "R_v": R_v,
        "R_C": R_C,
        "R_z": R_z,
        "R_p": R_p,
        "R_p_sum": BedUpdate.gradation_constraint(p),
    }


def physics_loss(residuals: dict[str, Tensor], weights: Optional[dict[str, float]] = None) -> Tensor:
    """水动力、泥沙、床面和级配守恒的分组物理损失。"""
    # weights 允许训练时调节不同物理方程的相对重要性。
    weights = weights or {}

    # 水动力损失：连续方程 + 两个方向动量方程/扩散波动量平衡。
    loss_hydro = mse_zero(residuals["R_h"], residuals["R_u"], residuals["R_v"])

    # 泥沙输移损失：所有粒径级总输沙方程残差。
    loss_sed = mse_zero(residuals["R_C"])

    # 床面损失：床面高程变化 + 活动层级配守恒。
    loss_bed = mse_zero(residuals["R_z"], residuals["R_p"])

    # 级配归一损失：理论上 softmax 已保证 sum(p_k)=1，这里作为保险项。
    loss_cons = mse_zero(residuals["R_p_sum"])

    # 总物理损失：训练循环中还会叠加初始条件、边界条件和观测数据损失。
    return (
        weights.get("hydro", 1.0) * loss_hydro
        + weights.get("sediment", 1.0) * loss_sed
        + weights.get("bed", 1.0) * loss_bed
        + weights.get("cons", 1.0) * loss_cons
    )
