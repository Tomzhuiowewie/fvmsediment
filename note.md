# Flow-Sediment Simulation 计算说明

本文档按当前代码整理模型方程、闭合关系、时间推进流程和主要参数。当前实现以 HEC-RAS 2D sediment 的算法思想为参照，但仍是简化版本。

## 1. 当前代码结构

- `src/model.py`
  - `FlowPINN(x,y,t) -> h,u,v`：输出水深和二维流速。
  - `SedimentPINN(x,y,t) -> C_tk`：输出每个粒径组的总输沙浓度。
- `src/physics.py`
  - `SVEsPhysicsLoss`：二维浅水方程有限体积残差。
  - `SedimentTransportLoss`：总输沙方程有限体积残差。
  - `ClosureFormulation`：沉速公式和 Wu 总输沙潜力公式。
- `src/train.py`
  - `DecoupledTrainer`：按时间窗口训练水动力、训练输沙、显式更新床面和级配。
- `config.yaml`
  - 控制模拟时长、窗口大小、时间项开关、输沙参数、床变参数和训练参数。

## 2. 时间推进逻辑

时间由 `config.yaml` 控制：

- `simulation_time`：总物理模拟时长。
- `window_dt`：床面/级配更新的宏时间窗口。
- `sample_dt`：每个时间窗口内 PINN 残差采样间隔。
- `output_dt`：结果输出间隔。
- `include_time_terms`：水动力和输沙 PDE 是否包含时间导数。`true` 表示非恒定时间窗口训练；`false` 表示准稳态残差。

每个窗口 `[t_n, t_{n+1}]` 的流程：

1. 在窗口内多个时间点训练水动力 `FlowPINN`。
2. 在同一窗口内多个时间点训练输沙 `SedimentPINN`。
3. 在窗口末端 `t_{n+1}` 计算闭合量 `h,u,v,C_tk,p_k,E_tk,D_tk,q_bk`。
4. 用 HEC-RAS 风格床变方程显式更新 `z_b`。
5. 用同一组分粒径床变贡献 `dzb_dt_k` 更新活动层和第二层级配。
6. 进入下一个时间窗口。

## 3. 水动力 PDE：二维浅水方程

变量：

- `h`：水深。
- `u, v`：x/y 方向深度平均流速。
- `z_b`：床面高程。
- `g`：重力加速度。
- `n`：Manning 糙率。

连续方程：

$$
\frac{\partial h}{\partial t}
+ \frac{\partial (hu)}{\partial x}
+ \frac{\partial (hv)}{\partial y}
= 0
$$

x 方向动量方程：

$$
\frac{\partial (hu)}{\partial t}
+ \frac{\partial}{\partial x}
\left(hu^2+\frac{1}{2}gh^2\right)
+ \frac{\partial (huv)}{\partial y}
= -gh\frac{\partial z_b}{\partial x} - \tau_{fx}
$$

y 方向动量方程：

$$
\frac{\partial (hv)}{\partial t}
+ \frac{\partial (huv)}{\partial x}
+ \frac{\partial}{\partial y}
\left(hv^2+\frac{1}{2}gh^2\right)
= -gh\frac{\partial z_b}{\partial y} - \tau_{fy}
$$

Manning 摩擦项：

$$
\tau_{fx}=g n^2 \frac{|\mathbf U|u}{h^{1/3}},
\quad
\tau_{fy}=g n^2 \frac{|\mathbf U|v}{h^{1/3}},
\quad
|\mathbf U|=\sqrt{u^2+v^2+\epsilon}
$$

代码位置：

- `src/physics.py`：`SVEsPhysicsLoss.compute_loss()`
- 使用单元边界高斯点做有限体积通量积分。
- 时间导数通过 `time_derivative()` 对 PINN 的归一化时间输入求导，并除以 `simulation_time` 转成物理时间导数。

## 4. 总输沙 PDE

变量：

- `C_tk`：第 `k` 粒径组总输沙浓度。
- `beta_tk`：总输沙修正系数，当前代码默认常数 `beta_default`。
- `epsilon_thk`：总输沙扩散/混合系数，当前代码默认常数 `epsilon_default`。
- `E_tk`：第 `k` 粒径组侵蚀率。
- `D_tk`：第 `k` 粒径组沉积率。

总输沙方程：

$$
\frac{\partial}{\partial t}
\left(\frac{h C_{tk}}{\beta_{tk}}\right)
+ \nabla\cdot(h C_{tk}\mathbf U)
- \nabla\cdot(\epsilon_{thk}h\nabla C_{tk})
- (E_{tk}-D_{tk})
=0
$$

有限体积残差：

$$
R_{Ck}
=
A\overline{
\frac{\partial}{\partial t}
\left(\frac{h C_{tk}}{\beta_{tk}}\right)}
+ \oint_{\partial\Omega} h C_{tk}(\mathbf U\cdot \mathbf n)dS
- \oint_{\partial\Omega}\epsilon_{thk}h(\nabla C_{tk}\cdot \mathbf n)dS
- A\overline{(E_{tk}-D_{tk})}
$$

代码位置：

- `src/physics.py`：`SedimentTransportLoss.total_load_fvm_loss()`

## 5. Wu 总输沙潜力闭合

当前容量浓度由 Wu 公式计算。

速度模长：

$$
U=\sqrt{u^2+v^2+\epsilon}
$$

床面剪应力：

$$
\tau_b
=
\rho_w g n^2 \frac{U^2}{h^{1/3}}
$$

skin shear：

$$
\tau'_b = f_{skin}\tau_b
$$

临界剪应力：

$$
\tau_{crk} = \theta_{cr}(\rho_s-\rho_w)g d_k
$$

相对密度：

$$
R=\frac{\rho_s}{\rho_w}-1
$$

Wu 推移质潜力：

$$
q^*_{bk}
=
0.0053\sqrt{Rg d_k^3}
\left(\frac{\tau'_b}{\tau_{crk}}-1\right)^{2.2}
$$

当 `tau_skin <= tau_cr` 时置零。

Wu 悬移质潜力：

$$
q^*_{sk}
=
2.62\times10^{-5}\sqrt{Rg d_k^3}
\left[
\left(\frac{\tau_b}{\tau_{crk}}-1\right)
\frac{U}{\omega_{sk}}
\right]^{1.74}
$$

当 `tau_b <= tau_cr` 或 `tau_skin <= tau_cr` 时置零。

分粒径总输沙潜力和容量浓度：

$$
q^*_{tk}=p_k(q^*_{bk}+q^*_{sk})
$$

$$
C^*_{tk}=\frac{q^*_{tk}}{hU}
$$

代码位置：

- `src/physics.py`：`ClosureFormulation.TransportPotential_Wu()`
- `p_k` 来自当前活动层级配，映射到高斯点。

## 6. 沉速公式

当前使用 Soulsby 1997：

$$
\omega_s
=
\frac{\nu}{d}
\left[
\sqrt{10.36^2+1.049d_*^3}
-10.36
\right]
$$

其中：

$$
d_* = d\left(\frac{Rg}{\nu^2}\right)^{1/3}
$$

代码位置：

- `src/physics.py`：`ClosureFormulation.fall_velocity()`

## 7. 适应长度和 E/D 源汇

当前使用 HEC-RAS 常数总输沙适应长度思想。适应时间：

$$
T_{adapt}=\frac{L_t}{\max(U,\epsilon_U)}
$$

净交换：

$$
S_k
=
\frac{h(C^*_{tk}-C_{tk})}{T_{adapt}}
=
\frac{hU}{L_t}(C^*_{tk}-C_{tk})
$$

代码当前用平滑正部函数拆分侵蚀和沉积：

$$
E_{tk}=\operatorname{smooth\_positive}(S_k)
$$

$$
D_{tk}=\operatorname{smooth\_positive}(-S_k)
$$

说明：

- 这保证 `E_tk >= 0`、`D_tk >= 0`。
- 该拆分是“净源项正负分离”实现。
- HEC-RAS 文档中更标准的写法是分别计算 `E_tk = alpha_t omega_s C^*`、`D_tk = alpha_t omega_s C`，二者可同时为正；当前代码在净源项上等价，但 E/D 拆分形式更简化。

代码位置：

- `src/physics.py`：`SedimentTransportLoss.compute_closure()`

## 8. Exner 床变方程

当前床变按 HEC-RAS 风格分粒径计算：

$$
\rho_s(1-\phi_b)
\left(\frac{\partial z_b}{\partial t}\right)_k
=
D_{tk}-E_{tk}
+ \nabla\cdot
\left(\kappa_{bk}|q_{bk}|\nabla z_b\right)
$$

总床变：

$$
\frac{\partial z_b}{\partial t}
=
\sum_k
\left(\frac{\partial z_b}{\partial t}\right)_k
$$

床坡系数：

$$
\kappa_{bk}
=
\kappa_{b0}
\sqrt{
\frac{\tau_{crk}}{\max(\tau'_b,\tau_{crk})}
}
$$

当前代码中：

$$
q_{bk}=p_k q^*_{bk}
$$

床坡项有限体积离散：

$$
\nabla\cdot(\kappa_{bk}|q_{bk}|\nabla z_b)
\approx
\frac{1}{A}
\oint_{\partial\Omega}
\kappa_{bk}|q_{bk}|(\nabla z_b\cdot \mathbf n)dS
$$

显式床面更新：

$$
z_b^{n+1}
=
z_b^n
+
\left(\frac{\partial z_b}{\partial t}\right)
\Delta t_{eff}
$$

时间步限幅：

- 先用 `window_dt` 计算原始 `max(|Delta z_b|)`。
- 若超过 `max_bed_change_per_step`，则缩小 `dt_scale`。
- 实际床变时间步为 `window_dt_current = window_dt * dt_scale`。

代码位置：

- `src/train.py`：`_exner_dzb_dt_cell()`
- `src/train.py`：`update_bed_explicit()`

## 9. 活动层和级配更新

当前使用两层结构：

- 活动层：`active_layer_frac = f_{1k}`。
- 第二层：`second_layer_frac = f_{2k}`。

活动层厚度：

$$
\delta_1=\max(\alpha_{active}d_{90},10^{-4})
$$

第二层厚度变化：

$$
\Delta\delta_2
=
\Delta z_b
-
(\delta_1^{new}-\delta_1^{old})
$$

分粒径床变质量进入级配更新：

$$
\Delta M_{bed,k}
=
\rho_s(1-\phi_b)
\left(\frac{\partial z_b}{\partial t}\right)_k
\Delta t_{eff}
$$

其中 `(partial z_b / partial t)_k` 已包含：

$$
D_{tk}-E_{tk}
+ \nabla\cdot(\kappa_{bk}|q_{bk}|\nabla z_b)
$$

界面交换材料：

$$
m_k^*
=
\begin{cases}
m_{1k}^{old}, & \Delta\delta_2 \ge 0 \\
m_{2k}^{old}, & \Delta\delta_2 < 0
\end{cases}
$$

活动层更新：

$$
m_{1k}^{new}
=
\frac{
\Delta M_{bed,k}
+ m_{1k}^{old}\delta_1^{old}
- m_k^*\Delta\delta_2
}
{\delta_1^{new}}
$$

第二层更新：

$$
m_{2k}^{new}
=
\frac{
m_{2k}^{old}\delta_2^{old}
+ m_k^*\Delta\delta_2
}
{\delta_2^{new}}
$$

归一化得到级配：

$$
f_{1k}^{new}
=
\frac{m_{1k}^{new}}{\sum_k m_{1k}^{new}}
$$

$$
f_{2k}^{new}
=
\frac{m_{2k}^{new}}{\sum_k m_{2k}^{new}}
$$

代码位置：

- `src/train.py`：`update_gradation_state()`

## 10. 与 HEC-RAS 的一致性和简化

已经对齐的部分：

- 水动力使用二维浅水方程的有限体积残差。
- 输沙使用总输沙浓度方程。
- 输沙能力使用 Wu 床载 + 悬移分量。
- 床变使用分粒径 `D-E + bed-slope` 形式。
- 级配更新使用分粒径床变贡献 `rho_s(1-phi)(dzb/dt)_k`，床坡项进入级配质量账。
- 使用活动层/第二层的两层质量守恒更新。

仍是简化的部分：

- E/D 当前是净源项正负拆分，不是 HEC-RAS 分别计算 `E=alpha omega C*`、`D=alpha omega C` 的完整形式。
- 活动层厚度没有加入 HEC-RAS 工程实现里的床形高度下限。
- 只有活动层和第二层，没有多层床材料、非冲刷层和复杂工程边界。
- `rho_s` 和孔隙率是常数。
- 床坡项使用现有结构的高斯边界积分，`kappa_b0` 是常数配置参数。

## 11. 主要配置参数

时间：

- `physics.simulation_time`
- `physics.window_dt`
- `physics.sample_dt`
- `physics.output_dt`
- `physics.include_time_terms`

水动力：

- `physics.flow.typical_depth`
- `physics.flow.typical_velocity`
- `physics.flow.g`
- `physics.flow.n_manning`

输沙：

- `physics.sediment.grain_diameters`
- `physics.sediment.adaptation_length`
- `physics.sediment.rho_s`
- `physics.sediment.rho_w`
- `physics.sediment.kinematic_viscosity`
- `physics.sediment.wu_theta_cr`
- `physics.sediment.skin_shear_factor`
- `physics.sediment.beta_default`
- `physics.sediment.epsilon_default`
- `physics.sediment.alpha_active_layer`

床变：

- `physics.morphodynamics.porosity`
- `physics.morphodynamics.bed_slope_coefficient`

训练：

- `training.flow_epochs_per_step`
- `training.sediment_epochs_per_step`
- `training.max_bed_change_per_step`
