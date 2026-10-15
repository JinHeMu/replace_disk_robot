# JAKA SDK 工具

使用仓库内 `adapters/jaka` 和随附的 Linux x86_64 SDK，自动加载原生库。
先激活项目环境，运行顺序见 [JAKA 使用说明](../../docs/02_JAKA使用说明.md)。

| 脚本 | 用途 |
| --- | --- |
| `jaka_start.py` | 上电使能；`--read-only` 仅登录读状态 |
| `jaka_stop.py` | 关闭 Servo/EDG、下使能、下电 |
| `jaka_ft_test.py` | 读取六维力，支持 `--seconds` 和 `--csv` |
| `jaka_edg_servo.py` | 关节伺服测试；`--dry-run` 仅读取反馈 |
| `jaka_common.py` | 复用正式 SDK adapter 与会话工具 |

默认控制器 IP 为 `10.5.5.100`、本机 IP 为 `10.5.5.127`；
使用 `--robot-ip`、`--local-ip` 指定实际地址。各脚本的完整选项见 `--help`。
