# JAKA 键盘控制使用指南

| 文件 | 用途 |
|---|---|
| `jaka_keyboard_node.py` | 新版启动入口：模块化控制，支持周期日志 |
| `jaka_keyboard_servo.py` | 保留的原始控制脚本，适配新参数名称，控制逻辑不变 |
| `jaka_keyboard_servo_config.yaml` | 两个入口共用的默认参数 |

使用已有的 JAKA SDK Python 环境。下列命令在仓库根目录执行：

```bash
cd /home/a/replace_disk_robot
```

## 启动与退出

先启动机器人，再用只读模式检查力和键盘输入：

```bash
python examples/jaka_driver_tool/jaka_start.py
python src/replace_disk_robot/applications/jaka_keyboard/jaka_keyboard_node.py --dry-run
```

正式控制并记录日志：

```bash
python src/replace_disk_robot/applications/jaka_keyboard/jaka_keyboard_node.py --log-dir logs/jaka_trial_001
```

不需要日志时去掉 `--log-dir`。窗口内按 `Esc` 退出后，关闭机器人：

```bash
python examples/jaka_driver_tool/jaka_stop.py
```

使用原版本时，将启动命令中的 `jaka_keyboard_node.py` 换成 `jaka_keyboard_servo.py`；原版本不支持 `--log-dir`。两个入口选择一个运行。

## 键盘操作

先点击 **JAKA keyboard servo** 窗口，按住按键运动，松开停止名义运动。

| 按键 | 功能（默认 base 模式） |
|---|---|
| R / F | 沿当前 tool0 的 +Z / −Z 插入、退出 |
| W / S | 沿基座 +Z / −Z 移动 |
| A / D | 沿基座 −X / +X 移动 |
| Q / E、方向键 | 姿态旋转 |
| 空格 / Enter | 停止 / 尝试恢复 |
| Esc | 退出 |

加 `--command-frame tool` 可让 W/S、A/D 和旋转使用末端坐标系。R/F 始终沿末端 Z 轴。开启导纳时，松键后仍会响应外力。

## 参数设置

直接修改同目录的 `jaka_keyboard_servo_config.yaml`，下一次启动生效。命令行参数优先于 YAML，例如：

```bash
python src/replace_disk_robot/applications/jaka_keyboard/jaka_keyboard_node.py --linear-speed 0.001 --plot-wrench-enable false
```

常用参数：`linear_speed`（平移速度）、`adm_mass/damping/stiffness`（导纳 M/D/K）、`alpha`（滤波）、`deadband_force/torque`（死区）。导纳开启时需要启用重力补偿。当前默认标定 `tool/ft_gravity_samples_identified.json` 已用 `logs/force_direction_audit/candidate_tool0_repaired_bounded4.json` 的内容替换；默认开启重力补偿、关闭额外去皮，只对平移轴开启导纳，禁用轴列表为空。

从仓库根目录直接运行 `python3 src/replace_disk_robot/applications/jaka_keyboard/jaka_keyboard_node.py` 即使用上述配置。等价的显式参数为：

```bash
python3 src/replace_disk_robot/applications/jaka_keyboard/jaka_keyboard_node.py \
  --gravity-compensation-enable true \
  --gravity-json logs/force_direction_audit/candidate_tool0_repaired_bounded4.json \
  --tare-enable false --admittance --admittance-axes translation \
  --adm-disable-axes
```

`--adm-disable-axes` 后面不跟轴名表示不额外禁用任何平移轴。历史排查与验证边界见 [力方向排查报告](../../../../logs/force_direction_audit/report.md)。

用 `--config /path/to/config.yaml` 指定其他参数文件；用 `--help` 查看全部选项。

功能开关统一使用正向布尔值：`true` 开启，`false` 关闭。YAML 示例：

```yaml
gravity_compensation_enable: true  # 重力补偿
tare_enable: false                 # 当前默认关闭适配器去皮
plot_wrench_enable: false          # 六维力曲线窗口
```

命令行对应 `--gravity-compensation-enable true/false`、`--tare-enable true/false`、`--plot-wrench-enable true/false`。布尔值必须明确填写，YAML 中不要给 `true/false` 加引号。自定义旧配置需要改用新字段。关闭重力补偿时也需要在 YAML 中关闭导纳（`admittance: false`）。

## 日志分析

日志目录必须不存在。日志包含原始/处理后力、按键、导纳状态、关节命令、故障事件和参数快照。

```bash
python tool/analyze_jaka_control_log.py logs/jaka_trial_001 --plots
```

结果保存在日志目录的 `analysis/`。首先阅读 `report.md`：报告分为“力补偿”和“导纳控制”两部分，导纳进一步分为“外力与控制器输出”和“机械臂实际运动”。`report.json` 保留原有公式与完整性检查，并增加这两部分的统计。

| 输出 | 内容 |
|---|---|
| `report.md` | 六轴力补偿对比表、导纳外力/输出表、实际位移表及坐标系定义 |
| `force_control_samples.csv` | 逐周期原始力、补偿力及同坐标系下的差值；没有原始信号时省略原始/差值列 |
| `admittance_control_samples.csv` | 补偿后 TCP 外力、处理后外力、实际导纳输入、积分偏移/速度、目标与实测位移 |
| `cycle_metrics.csv` | 原有离散残差、同周期跟踪误差及命令状态 |
| `force_control_overview.png` | 原始与补偿后的六维力/力矩 |
| `admittance_control_overview.png` | 导纳外力、积分偏移及速度 |
| `admittance_tracking_overview.png` | 实测与目标位移、同周期误差、周期与命令状态 |

表格与 CSV 始终生成。`--plots` 生成三张图；`--plot-force-control` 只生成力补偿图，`--plot-admittance-control` 生成两张导纳图。CSV 缺失值留空，统计忽略缺失值，不补零。六轴顺序为 Fx/Fy/Fz/Tx/Ty/Tz，位移顺序为 x/y/z/rx/ry/rz；平移单位 mm、旋转单位 deg，速度单位 mm/s 或 deg/s。

力补偿在 `tcp_fts_sensor` 中对比原始与补偿后的数据；没有原始数据时只展示补偿数据。若没有传感器补偿阶段但有处理后 TCP 信号，报告明确标记降级数据来源及其坐标系。坐标系不同或未知的原始/补偿信号分别展示，不计算差值。均值、RMS、峰值绝对值为统计量，补偿前后差值不是补偿精度。

导纳输入/输出位于当前实测 TCP 的 `tool0` 轴。外力分为补偿后 TCP 外力（滤波前）、处理后外力（滤波/死区后）和轴屏蔽后的实际输入。积分偏移是位姿限幅前的输出；机械臂运动图使用 `jaka_base_link`，显示限幅后修正目标相对当周期名义位姿的偏移、实测相对名义位姿的偏移，以及实测/目标/名义位姿相对首个有效实测位姿的总位移。后一项包含键盘运动，实测相对名义位姿也包含跟踪滞后。

旋转位移采用 `log(R_current R_reference^T)` 的基座轴旋转向量，不作欧拉角相减。实测 TCP 来自实测关节的正运动学，反馈在本周期命令发送之前采集；同周期误差不能当作本周期命令执行后的响应。故障区间以红色阴影标记，CSV 保留 `command_sent`、`admittance_updated` 和 `fault`，积分输出存在不代表运动命令已经发送。目标与实测位姿坐标系不匹配或信号在日志中切换坐标系时，分析器报错，避免混轴比较。
