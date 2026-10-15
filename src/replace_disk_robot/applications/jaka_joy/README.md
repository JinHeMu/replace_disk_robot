# JAKA 手柄应用

`jaka_joy_node.py` 默认运行 MuJoCo，`--backend real` 选择 JAKA 实机；
`--probe` 只读取手柄，`--dry-run` 不发送运动，`--log-dir` 记录控制日志。

参数见 [jaka_joy_config.yaml](jaka_joy_config.yaml)，运行和按键见
[JAKA 使用说明](../../../../docs/02_JAKA使用说明.md)。

- `joy_node.py`：输入映射、停止恢复、回中保持和运行调度。
- `joy_frontend.py`、`joy_simulation.py`：窗口与仿真组装。
- `control/joy_control.py`、`adapters/joy.py`：通用手柄映射与 SDL 读取。
- `applications/common`：共用力处理、导纳、伺服和日志。

`JoyControl.update(sample)` 更新输入，`command(now_s)` 输出 `CartesianJog`；
采样与控制时间使用同一时钟。完整命令行参数使用入口的 `--help` 查看。
