# JAKA 使用说明

先按 [环境配置](01_环境配置.md) 安装并激活环境；下列命令均在仓库根目录执行。
当前使用 JAKA SDK，控制循环为 125 Hz。

## 仿真与实机

```bash
# 手柄输入检查，不连接机械臂
python src/replace_disk_robot/applications/jaka_joy/jaka_joy_node.py --probe --seconds 10

# 手柄 + JAKA MuJoCo 仿真
python src/replace_disk_robot/applications/jaka_joy/jaka_joy_node.py

# 键盘 + JAKA MuJoCo 仿真
python examples/keyboard_servo.py --model jaka --admittance
```

实机先确认 `robot_ip`、`local_ip` 和工具标定与实际安装一致，再按需执行：

```bash
# 上电使能；只查连接状态可加 --read-only
python examples/jaka_driver_tool/jaka_start.py

# 读取实机反馈和手柄，不发送运动
python src/replace_disk_robot/applications/jaka_joy/jaka_joy_node.py --backend real --dry-run --headless --seconds 10

# 实机手柄控制并记录日志
python src/replace_disk_robot/applications/jaka_joy/jaka_joy_node.py --backend real --log-dir logs/trial_001

# 退出控制程序后下使能、下电
python examples/jaka_driver_tool/jaka_stop.py
```

实机键盘入口为 `python src/replace_disk_robot/applications/jaka_keyboard/jaka_keyboard_node.py`，
支持 `--dry-run`、`--log-dir`；与手柄控制程序择一运行。日志目录必须是新目录。

## 操作与参数

手柄先松开 LB、摇杆和扳机回中，再按住 LB 运动。左摇杆平移，RT/LT 沿 `tool0` ±Z 插拔，
右摇杆和十字键左右调姿；RB 为精细速度，B 停止，松开 LB 并回中、卸载后 A 恢复，Back 退出。
松开 LB 保持关节目标。默认 `base` 模式使用基座 X/Z 平移与工具 Z 插拔；
`--command-frame tool` 切换为工具 X/Y/Z 平移。

键盘先点击控制窗口：W/S 升降、A/D 左右、R/F 沿工具 Z 插拔，Q/E 和方向键调姿；
空格停止、Enter 恢复、Esc 退出。开启导纳时，松键后仍可响应外力。

参数直接修改对应 YAML，命令行参数优先；完整选项使用 `--help` 查看：

- [手柄配置](../src/replace_disk_robot/applications/jaka_joy/jaka_joy_config.yaml)
- [键盘配置](../src/replace_disk_robot/applications/jaka_keyboard/jaka_keyboard_servo_config.yaml)

常用项为速度、`command_frame`、导纳 M/D/K、启用轴、滤波/死区和力限。
当前两套配置的速度与柔顺轴不同，以各自 YAML 为准。导纳需要重力补偿，
标定由 `gravity_json` 指定，默认是 `tool/ft_gravity_samples_identified.json`。

## 日志与标定

```bash
python tool/analyze_jaka_control_log.py logs/trial_001 --plots
```

结果写入该日志目录的 `analysis/`，包含报告、逐周期 CSV 和力控/导纳曲线。
分析保留完整采样与时间；同周期目标误差包含执行滞后，不能当作发送后立即达到的误差。
重力标定的采集、辨识和绘图见 [工具说明](../tool/README.md)。

## 代码分工

`applications/jaka_joy`、`jaka_keyboard` 各自负责输入和配置，`applications/common` 共用反馈、
力处理、导纳、伺服与日志；`adapters` 连接 SDK/仿真，`control` 和 `kinematics` 提供算法。

执行链：输入 → 名义位姿 → 导纳修正 → CartesianServo → 力限保护 → JAKA 关节命令。
关节为 `joint_1` 至 `joint_6`，基座为 `jaka_base_link`，TCP 为 `tool0`。
核心数据使用 m/rad/N/N·m、wxyz 四元数和显式坐标系；核心算法不导入 SDK 或 MuJoCo。
