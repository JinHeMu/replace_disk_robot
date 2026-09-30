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

常用参数：`linear_speed`（平移速度）、`adm_mass/damping/stiffness`（导纳 M/D/K）、`alpha`（滤波）、`deadband_force/torque`（死区）。导纳开启时需要启用重力补偿；默认辨识文件为仓库中的 `tool/ft_gravity_samples_identified.json`。
用 `--config /path/to/config.yaml` 指定其他参数文件；用 `--help` 查看全部选项。

功能开关统一使用正向布尔值：`true` 开启，`false` 关闭。YAML 示例：

```yaml
gravity_compensation_enable: true  # 重力补偿
tare_enable: true                  # 非重力补偿路径的适配器去皮
plot_wrench_enable: false          # 六维力曲线窗口
```

命令行对应 `--gravity-compensation-enable true/false`、`--tare-enable true/false`、`--plot-wrench-enable true/false`。布尔值必须明确填写，YAML 中不要给 `true/false` 加引号。自定义旧配置需要改用新字段。关闭重力补偿时也需要在 YAML 中关闭导纳（`admittance: false`）。

## 日志分析

日志目录必须不存在。日志包含原始/处理后力、按键、导纳状态、关节命令、故障事件和参数快照。

```bash
python tool/analyze_jaka_control_log.py logs/jaka_trial_001 --plots
```

结果保存在日志目录的 `analysis/`：`report.json`（公式与完整性检查）、`cycle_metrics.csv`（周期指标）、`control_overview.png`（曲线）。
