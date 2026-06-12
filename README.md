# FVM-PINN 二维水沙与河床演变模拟

本项目使用真实 DEM、水文过程线和床沙级配，训练二维水动力 PINN 与分粒径泥沙 PINN，并通过有限体积积分残差、Wu 输沙能力和 Exner 床变关系约束网络。

当前实现不使用观测结果场作为监督标签。DEM、流量、水位和床沙级配只作为初始条件、边界条件与物理闭合输入。

## 1. 当前方法概览

```text
GeoTIFF DEM + Excel 流量/水位过程线 + Excel 床沙级配
                         |
                         v
              规则 FVM 网格与河道 mask
                         |
             +-----------+-----------+
             |                       |
             v                       v
  FlowPINN(x,y,t)             SedimentPINN(x,y,t)
       -> h,u,v                   -> C_1...C_K, Δzb
             |                       |
             +-----------+-----------+
                         |
       SWE + 总输沙方程 + Wu 闭合 + Exner 约束
                         |
                         v
       全时域 DEM-水动力-泥沙固定点耦合
                         |
                         v
         床面历史、最终级配、模型和诊断图
```

核心特点：

- 在规则 DEM 栅格上构造有限体积单元。
- 只对 `active_mask=True` 的河道单元计算物理损失。
- 水动力采用二维非恒定浅水方程。
- 泥沙采用分粒径总输沙对流-扩散-源汇方程。
- 沉速采用 Soulsby (1997) 公式。
- 输沙能力采用 Wu 类型推移质与悬移质潜力。
- 床变采用分粒径 Exner 关系。
- 阶段 3 不划分形态窗口，而是在完整时间域上迭代更新 DEM 和级配。

## 2. 项目结构

```text
.
├── main.py              # 程序入口、对象组装、结果保存
├── config.yaml          # 数据、物理和训练参数
├── data/
│   ├── 100ft.test1d.tif
│   └── FlowSedimentData.xlsx
├── src/
│   ├── config.py        # YAML 配置解析
│   ├── data.py          # DEM/Excel 读取、FVM 网格、边界条件
│   ├── model.py         # FlowPINN 和 SedimentPINN
│   ├── physics.py       # SWE、输沙、Wu 和 Exner 物理关系
│   ├── train.py         # 三阶段训练与全时域耦合
│   ├── evaluate.py      # 床面和损失可视化
│   └── utils.py         # 坐标归一化与自动微分工具
└── outputs/             # checkpoint、模型、结果和 PNG
```

## 3. 输入数据

### 3.1 DEM

`data.dem_path` 指向 GeoTIFF DEM。

读取时执行：

1. 高程由 ft 转为 m。
2. 像元分辨率由 GeoTIFF `ModelPixelScaleTag` 读取并转为 m。
3. 有效且不高于 `channel_elevation_threshold_ft` 的像元作为活动河道。
4. NoData 单元用有效高程最大值填充，但不进入河道物理损失。
5. DEM 行方向翻转，使模型坐标的 y 方向与网格定义一致。

网格坐标从 `(0,0)` 开始，不保留 GeoTIFF 的地理坐标原点。

### 3.2 水文过程线

`data.hydro_sediment_path` 指向 Excel 文件。

代码直接解析 XLSX 内部 XML，不依赖 `openpyxl`。当前要求存在：

- `Unsteady Flow Data`：上游流量和下游水位过程线。
- `Sediment Data`：主河道累计床沙级配。

单位转换：

- 时间：h -> s
- 流量：cfs -> m3/s
- 水位：ft -> m
- 粒径：mm -> m

训练时通过线性插值得到任意时刻的 `Q(t)` 和 `stage(t)`。

### 3.3 初始床沙级配

Excel 中的累计通过率 `% finer` 转换为分粒径比例：

```math
p_k = F_k - F_{k-1}
```

初始时所有网格单元使用同一组主河道级配。只有活动河道单元参与后续闭合与更新。

如果 `config.yaml` 中显式给出 `grain_diameters`，模型粒径组使用配置值；否则使用 Excel 粒径。

当前活动层初始级配仍来自 Excel，因此配置粒径组数量必须与 Excel 转换后的有效级配组数量一致，否则训练器初始化会报错。

## 4. FVM 网格

`FVMeshPreprocessor` 保留 DEM 的规则矩形栅格。每个单元记录：

- 格心坐标；
- 单元面积；
- 四条边；
- 边界高斯点；
- 高斯权重和外法向；
- 相邻单元编号；
- 初始床面 `zb_initial`；
- 当前耦合床面 `zb`；
- 活动河道 mask。

`n_gauss_points=1` 时，每条边使用中点积分；设置为 `2` 时使用两点 Gauss-Legendre 积分。

床面使用 `float64` 保存，物理网络和自动微分使用 `float32`。

## 5. PINN 网络

两个网络均采用：

```text
输入层 -> Tanh -> 多个全连接残差块 -> 输出层
```

默认结构：

- 输入：归一化 `(x,y,t)`
- 隐藏宽度：64
- 残差块：4
- Xavier 权重初始化

### 5.1 FlowPINN

```text
FlowPINN(x,y,t) -> h,u,v
```

输出经 Sigmoid 后反归一化：

```math
h = r_h H_{typ}
```

```math
u = r_u U_{typ}
```

```math
v = (r_v-0.5)2U_{typ}
```

因此当前输出范围为：

```text
0 < h < typical_depth
0 < u < typical_velocity
-typical_velocity < v < typical_velocity
```

注意：当前 `u` 不能为负值，使用时应保证坐标方向与主流方向设置合理。

### 5.2 SedimentPINN

```text
SedimentPINN(x,y,t) -> C_1,...,C_K,Δzb
```

- 前 `K` 个输出为分粒径总输沙浓度。
- 浓度经过 Softplus，保证非负。
- 最后一个输出为网络累计床变 `Δzb`。
- 原始床变输出乘 `bed_change_scale`。

网络 `Δzb` 用于 Exner 时间导数约束和诊断。正式床面历史由 Exner 床变率的全时域积分生成。

## 6. 水动力方程

水动力损失使用二维非恒定浅水方程的有限体积积分残差。

连续方程：

```math
\frac{\partial h}{\partial t}
+\frac{\partial hu}{\partial x}
+\frac{\partial hv}{\partial y}=0
```

x 动量方程：

```math
\frac{\partial hu}{\partial t}
+\frac{\partial}{\partial x}\left(hu^2+\frac12gh^2\right)
+\frac{\partial huv}{\partial y}
=-gh\frac{\partial z_b}{\partial x}-S_{fx}
```

y 动量方程：

```math
\frac{\partial hv}{\partial t}
+\frac{\partial huv}{\partial x}
+\frac{\partial}{\partial y}\left(hv^2+\frac12gh^2\right)
=-gh\frac{\partial z_b}{\partial y}-S_{fy}
```

Manning 摩阻：

```math
S_{fx}=gn^2\frac{|\mathbf U|u}{h^{1/3}},
\qquad
S_{fy}=gn^2\frac{|\mathbf U|v}{h^{1/3}}
```

时间导数由网络对归一化时间的自动微分获得，再除以 `simulation_time` 转换为物理时间导数。

床坡通过当前 `mesh.zb` 的有限差分计算。因此阶段 3 更新 DEM 后，后续联合优化使用新的床坡。

## 7. 水动力边界条件

当前边界方向固定为：

- top：上游入口；
- bottom：下游出口；
- active/inactive 交界与其余外边界：固壁。

虽然配置文件包含 `inlet_edge` 和 `outlet_edge`，当前实现尚未读取这两个字段来改变边界方向。

### 7.1 上游流量

入口不指定固定速度，而是约束断面流量：

```math
Q_{model}=\sum_i h_i(-v_i)\Delta s_i
```

```math
L_Q=
\left[
\frac{Q_{model}-Q(t)}
{\max(|Q(t)|,1)}
\right]^2
```

### 7.2 下游水位

Excel 提供绝对水位，模型输出水深：

```math
h_{target}=\max(stage(t)-z_b,\ h_{min})
```

出口床高从当前 `mesh.zb` 读取，因此 DEM 更新后边界目标水深同步变化。

### 7.3 固壁

固壁约束法向速度：

```math
u_n=un_x+vn_y=0
```

## 8. 分粒径总输沙方程

每个粒径组满足：

```math
\frac{\partial}{\partial t}
\left(\frac{hC_k}{\beta_k}\right)
+\nabla\cdot(hC_k\mathbf U)
-\nabla\cdot(\epsilon_kh\nabla C_k)
-(E_k-D_k)=0
```

当前：

```text
beta_k = beta_default
epsilon_k = epsilon_default
```

二者是全域常数，不随粒径、空间和时间变化。

FVM 损失包含：

- 单元存储项；
- 边界对流通量；
- 边界扩散通量；
- 单元侵蚀沉积源汇。

## 9. 输沙闭合

### 9.1 沉速

采用 Soulsby (1997) 非黏性颗粒沉速：

```math
d_*=
d\left(\frac{Rg}{\nu^2}\right)^{1/3}
```

```math
\omega_s=
\frac{\nu}{d}
\left[
\sqrt{10.36^2+1.049d_*^3}-10.36
\right]
```

其中：

```math
R=\frac{\rho_s}{\rho_w}-1
```

### 9.2 床面剪应力

```math
\tau_b=
\rho_wgn^2\frac{|\mathbf U|^2}{h^{1/3}}
```

```math
\tau_{skin}=f_{skin}\tau_b
```

```math
\tau_{cr,k}
=\theta_{cr}(\rho_s-\rho_w)gd_k
```

### 9.3 Wu 输沙潜力

推移质潜力：

```math
q_{b,k}=
0.0053\sqrt{Rgd_k^3}
\left(\frac{\tau_{skin}}{\tau_{cr,k}}-1\right)^{2.2}
```

悬移质潜力：

```math
q_{s,k}=
2.62\times10^{-5}\sqrt{Rgd_k^3}
\left[
\left(\frac{\tau_b}{\tau_{cr,k}}-1\right)
\frac{|\mathbf U|}{\omega_{s,k}}
\right]^{1.74}
```

未超过临界剪应力时，相应输沙潜力置零。

考虑活动层级配后的能力输沙量和浓度：

```math
q_{capacity,k}=p_k(q_{b,k}+q_{s,k})
```

```math
C_{capacity,k}=
\frac{q_{capacity,k}}{h|\mathbf U|}
```

## 10. 侵蚀与沉积

适应时间：

```math
T_{adapt}=
\frac{L_{adapt}}{\max(|\mathbf U|,U_{min})}
```

净交换量：

```math
S_k=
\frac{h(C_{capacity,k}-C_k)}{T_{adapt}}
```

代码使用平滑正部函数拆分：

```math
E_k=smoothPositive(S_k)
```

```math
D_k=smoothPositive(-S_k)
```

因此：

- `C_k < C_capacity,k` 时倾向侵蚀；
- `C_k > C_capacity,k` 时倾向沉积。

## 11. Exner 床变与活动层

分粒径净沉积率：

```math
R_{dep,k}=D_k-E_k
```

代码还根据当前床坡、推移质潜力和剪应力计算床坡扩散项。分粒径床变率为：

```math
\dot z_{b,k}=
\frac{
w_{exchange}R_{dep,k}
+w_{slope}S_{slope,k}
}{1-\lambda_p}
```

总床变率：

```math
\dot z_b=\sum_k\dot z_{b,k}
```

网络累计床变满足：

```math
\frac{\partial\Delta z_b}{\partial t}\approx\dot z_b
```

并在 `t=0` 约束：

```math
\Delta z_b=0
```

全时域正式床变使用梯形积分：

```math
\Delta z_{b,k}(t_j)=
\sum_{i=1}^{j}
\frac{
\dot z_{b,k}(t_{i-1})+\dot z_{b,k}(t_i)
}{2}
(t_i-t_{i-1})
```

候选床面始终以初始 DEM 为基准：

```math
z_{b,candidate}=z_{b,initial}+\sum_k\Delta z_{b,k}
```

候选活动层级配：

```math
p_{k,candidate}=
Normalize\left(
p_{k,initial}+
\frac{\Delta z_{b,k}}{L_a}
\right)
```

## 12. 三阶段训练

### 12.1 阶段 1：水动力预训练

只训练 `FlowPINN`。

每个 epoch：

1. 遍历全部训练时间点。
2. 计算 SWE FVM 残差。
3. 计算入口流量、出口水位和固壁损失。
4. 将所有时间点的平均梯度累积。
5. 梯度裁剪后执行一次 Adam 更新。

水动力总损失：

```math
L_{flow}=
L_{continuity}
+L_{momentum,x}
+L_{momentum,y}
+0.5L_{boundary}
```

阶段结束保存 `phase1_flow_<timestamp>.pt`。

### 12.2 阶段 2：泥沙预训练

冻结 `FlowPINN` 参数，只训练 `SedimentPINN`。

泥沙单元按 `sediment_cell_batch_size` 分批，降低多粒径自动微分的显存占用。

总损失：

```math
L_{sed}=
L_{transport}
+w_cL_{capacity}
+w_iL_{initial}
+w_{in}L_{inlet}
+w_b(L_{Exner}+L_{\Delta z_b,0})
```

其中：

- `L_transport`：总输沙 PDE FVM 残差；
- `L_capacity`：浓度接近 Wu 平衡浓度；
- `L_initial`：仅在 `t=0` 约束初始浓度；
- `L_inlet`：入口浓度接近平衡来沙浓度；
- `L_Exner`：网络床变时间导数接近 Exner 床变率；
- `L_Δzb,0`：仅在 `t=0` 约束累计床变为零。

阶段结束保存 `phase2_sediment_<timestamp>.pt`。

### 12.3 阶段 3：全时域固定点耦合

阶段 3 不划分形态窗口。

`joint_epochs` 是所有耦合迭代合计的联合训练 epoch。代码将其尽量均匀分配到 `coupling_iterations` 次迭代。

每次耦合迭代：

1. 使用当前 DEM、当前级配和当前两个网络。
2. 在全部 `sample_dt` 时间点计算分粒径 Exner 床变率。
3. 对完整模拟时间域做梯形积分。
4. 从初始 DEM 和初始级配构造候选状态。
5. 使用 `coupling_relaxation` 松弛更新当前 DEM 和级配。
6. 在更新后的 DEM 上，对全部时间点联合训练两个网络。
7. 检查本轮最大 DEM 改变量是否不超过 `coupling_bed_tol`。

松弛更新：

```math
z_b^{new}=
(1-\omega)z_b^{old}
+\omega z_{b,candidate}
```

级配采用相同松弛方式并重新归一化。

每轮候选床面都由“初始 DEM + 当前模型预测的全时域累计床变”构造，因此不会把耦合迭代次数误当作物理时间重复累计。

联合训练中泥沙损失允许梯度传回 `FlowPINN`，两个网络参数由同一个 Adam 优化器更新。

阶段结束保存 `phase3_joint_<timestamp>.pt`。

## 13. 时间定义

```text
simulation_time：完整物理模拟时长
sample_dt：PDE 训练点和 Exner 积分采样间隔
output_dt：床面结果和诊断输出间隔
```

网络时间输入：

```math
t_{norm}=\frac{t}{simulation\_time}
```

当 `include_time_terms=true` 时，SWE 和输沙方程保留时间导数；设为 `false` 时，PDE 时间导数返回零，但网络仍接收时间坐标和不同时刻的边界条件。

正式结果积分节点取训练采样时刻与输出时刻的并集，避免 `output_dt` 大于 `sample_dt` 时降低积分精度。

## 14. 当前默认参数

当前 `config.yaml` 的关键设置：

```text
总模拟时间              2,937,600 s，约 34 天
训练采样间隔            86,400 s
输出间隔                86,400 s
粒径组                  0.25 至 128 mm，共 10 组
Manning n               0.01
典型水深                8 m
典型流速                1 m/s
适应长度                5 m
孔隙率                  0.4
活动层厚度              0.5 m
flow epochs             200
sediment epochs         100
joint epochs            50
耦合迭代                5
耦合松弛系数            0.3
床面收敛阈值            1e-5 m
泥沙单元 batch          1024
```

当前初始浓度配置为标量 `0`，会扩展到所有粒径组。SedimentPINN 初始化时会将用于 Softplus 偏置初始化的值至少截断到 `1e-6`，但初始条件损失目标仍为配置值 `0`。

## 15. 运行

建议使用包含以下依赖的 Python 环境：

```bash
pip install numpy torch pyyaml pillow matplotlib
```

从项目根目录运行：

```bash
python main.py
```

设备选择：

```text
CUDA 可用 -> cuda
否则      -> cpu
```

真实 DEM 和多粒径自动微分计算量较大，正式运行建议使用 GPU。

## 16. 输出文件

每次运行创建统一时间戳：

```text
YYYYMMDD-HHMMSS
```

同一轮训练的所有文件使用相同时间戳，避免覆盖旧结果。

### 16.1 阶段 checkpoint

```text
outputs/checkpoints/
├── phase1_flow_<timestamp>.pt
├── phase2_sediment_<timestamp>.pt
└── phase3_joint_<timestamp>.pt
```

checkpoint 包含：

- 两个模型参数；
- 优化器状态；
- 当前活动层级配；
- 当前绝对床面；
- 训练历史；
- 模拟时间与运行时间戳。

### 16.2 最终结果

```text
outputs/
├── flow_model_<timestamp>.pt
├── sediment_model_<timestamp>.pt
├── final_checkpoint_<timestamp>.pt
├── training_results_<timestamp>.npz
├── history_<timestamp>.json
├── config_used_<timestamp>.yaml
├── real_bed_<timestamp>.png
├── real_profiles_<timestamp>.png
├── real_losses_<timestamp>.png
└── time_points_<timestamp>/
```

`training_results_<timestamp>.npz` 包含：

- `bed_history`
- `active_layer_frac`
- `output_times`
- `integration_times`

床面历史以 `float64` 保存。

### 16.3 分时刻床面

`time_points_<timestamp>/` 中每个输出时刻保存一个 NPZ：

```text
bed_t_0000d_00h_0000000000s.npz
bed_t_0001d_00h_0000086400s.npz
...
```

每个文件包含：

- `time_seconds`
- `time_days`
- `bed_elevation`
- `bed_change`

目录内 `index.json` 记录模拟时间与文件名的对应关系。

## 17. 诊断信息

`history_<timestamp>.json` 和损失图记录：

- 水动力总损失；
- 连续方程与两个动量方程损失；
- 泥沙输移、容量、入口、初始条件和床变损失；
- 联合损失；
- 每轮耦合床面误差；
- 入口目标/模型流量；
- 出口目标/模型水位；
- 水深与平均流速；
- 浓度和能力浓度；
- 网络累计床变范围；
- 活动层级配范围；
- 固壁平均法向速度；
- 提前停止原因；
- 最终耦合 DEM 与最终重积分结果之间的投影误差。

## 18. 当前实现边界

当前模型仍有以下限制：

- 仅支持规则矩形 DEM 栅格。
- 河道通过绝对高程阈值划分，不支持矢量河道边界。
- 入口和出口方向固定为 top/bottom。
- `FlowPINN` 的 x 方向速度不能为负。
- 不包含干湿界面专用数值处理，水深仅通过下限保护计算。
- `beta_k` 和 `epsilon_k` 为常数。
- `alpha_active_layer` 当前会从配置读取，但尚未进入闭合或级配更新公式。
- 初始级配在空间上均匀。
- 活动层为单层近似，没有第二床层或多床层交换。
- 孔隙率和 Manning 糙率不随级配或床面变化。
- 未使用 HEC-RAS 或实测结果场做监督、校准和验证。
- 床面固定点收敛不代表水动力、泥沙质量守恒误差已经达到工程标准，需要结合诊断量判断。

## 19. 代码入口

程序主流程位于：

```python
run_real_case(config_path="config.yaml")
```

执行顺序：

```text
读取配置
-> 读取 DEM/Excel
-> 构建 FVM 网格
-> 创建两个 PINN
-> 创建物理损失
-> 水动力预训练
-> 泥沙预训练
-> 全时域 DEM-水动力-泥沙耦合
-> 最终 Exner 重积分
-> 诊断
-> 保存模型、结果和 PNG
```
