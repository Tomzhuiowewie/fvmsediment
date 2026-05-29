# 参数

## 时间

simulation_time：总模拟时间

window_dt：窗口步长（河床更新频率）

sample_dt：窗口采样间隔

output_dt：每个窗口输出

# PDE

## 水动力方程（浅水）

### 连续性方程

$\frac{\partial h}{\partial t} + \frac{\partial (hu)}{\partial x} + \frac{\partial (hv)}{\partial y} = 0$


### 动量方程
