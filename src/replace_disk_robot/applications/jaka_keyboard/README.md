# JAKA 键盘应用

`jaka_keyboard_node.py` 是支持周期日志的实机入口，`--dry-run` 只读取反馈，
`--log-dir` 写入新日志目录。仿真使用 `examples/keyboard_servo.py --model jaka`。

参数见 [jaka_keyboard_servo_config.yaml](jaka_keyboard_servo_config.yaml)，
启动顺序、按键和日志分析见 [JAKA 使用说明](../../../../docs/02_JAKA使用说明.md)。

- `node.py`、`reference.py`、`frontend.py`：键盘应用、坐标映射和窗口。
- `applications/common`：共用反馈、力处理、导纳、伺服和日志。
- `jaka_keyboard_servo.py`：保留的兼容入口，不支持 `--log-dir`。

命令行参数覆盖 YAML；用 `--help` 查看完整选项。
