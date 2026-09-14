# JAKA Python SDK 独立工具

本目录直接使用正式 adapter：

```text
src/replace_disk_robot/adapters/jaka/
├── __init__.py
└── interfaces.py
```

不依赖 ROS、MoveIt 或 MuJoCo。SDK 二进制仍在：

```text
src/replace_disk_robot/adapters/jaka/jaka_driver/x86_64-linux-gnu/
├── jkrc.so
└── libjakaAPI.so
```

`JakaEdgClient` 会自动用 `ctypes` 预加载 `libjakaAPI.so`，因此不需要手动设置
`LD_LIBRARY_PATH`。所有脚本都支持 `--help`。

## 文件说明

| 文件 | 功能 |
|---|---|
| `jaka_common.py` | 公共 CLI、EDG session、125 Hz `RateLoop`，并设置 `src` 路径 |
| `jaka_start.py` | 登录、上电、使能、设置关节 LPF 和力矩传感器模式 |
| `jaka_stop.py` | 关闭 servo、关闭 EDG、下使能、下电、logout |
| `jaka_ft_test.py` | 只读采集 EDG 力/力矩，可输出 CSV |
| `jaka_edg_servo.py` | 125 Hz EDG 关节 servo，默认保持当前位置，可选正弦测试 |

本地接口对应关系：

- `JakaRobotAdapter` 实现 `replace_disk_robot.core.ports.ArmPort`
- `JakaWristFTAdapter` 实现 `replace_disk_robot.core.ports.ForceTorquePort`
- 数据使用 `JointState`、`Wrench`

## 默认参数

```text
robot_ip:  10.5.5.100
local_ip:  10.5.5.127
EDG port:  10010
EDG mode:  0
rate:      125 Hz
servo step: 1   # 8 ms
```

## 使用顺序

### 1. 启动机器人

```bash
python examples/jaka_driver_tool/jaka_start.py
```

只做登录和状态读取，不上电/使能：

```bash
python examples/jaka_driver_tool/jaka_start.py --read-only
```

### 2. 测试力/力矩数据

```bash
# 终端打印 10 秒
python examples/jaka_driver_tool/jaka_ft_test.py --seconds 10

# 输出 CSV
python examples/jaka_driver_tool/jaka_ft_test.py \
    --seconds 10 \
    --csv simulation/mujoco/reports/jaka_ft.csv
```

CSV 列包括：

```text
time_s
joint0_rad ... joint5_rad
raw_fx_n raw_fy_n raw_fz_n raw_tx_nm raw_ty_nm raw_tz_nm
fx_n fy_n fz_n tx_nm ty_nm tz_nm
```

其中 `raw_*` 是 EDG 原始读数，`f/t` 是经过零偏、`R_sensor_to_tool`、
`r x F`、低通滤波和死区后的结果。

### 3. EDG servo 测试

默认保持当前位置，不主动运动：

```bash
python examples/jaka_driver_tool/jaka_edg_servo.py --seconds 5
```

只读 dry-run，不进入 servo 模式：

```bash
python examples/jaka_driver_tool/jaka_edg_servo.py --dry-run --seconds 5
```

关节 0 小幅正弦测试：

```bash
python examples/jaka_driver_tool/jaka_edg_servo.py \
    --sine-joint 0 \
    --sine-amplitude-deg 2 \
    --sine-frequency-hz 0.2 \
    --seconds 5
```

力/力矩或跟踪误差超限会自动停止：

```text
--max-force-n         20.0
--max-torque-nm        5.0
--max-joint-error-rad  0.5
```

servo 命令通过 `JakaRobotAdapter.command_joint_positions()` 发出，内部使用
EDG `step_num=1`，即 8 ms 周期。

### 4. 关闭机器人

```bash
python examples/jaka_driver_tool/jaka_stop.py
```

只关闭 EDG 和连接、不改电源/使能状态：

```bash
python examples/jaka_driver_tool/jaka_stop.py --read-only
```

## 注意

- 这些脚本直接驱动真机。第一次运行请使用 `--read-only` 或 `--dry-run`。
- `jaka_start.py` 不进入 servo 模式；真正发送 servo 命令的是
  `jaka_edg_servo.py`，并会在退出时自动 `servo_move_enable(False)`。
- `jaka_ft_test.py` 只读，不发送运动指令。
- 进入 EDG 前需要确保本机 IP、防火墙和 UDP 端口配置正确。
- `JakaWristFTAdapter` 默认使用 WBMM 硬件接口中的补偿矩阵和力臂参数；
  如有实测标定值，修改 adapter 模块中的 `DEFAULT_SENSOR_TO_TOOL_ROTATION`
  和 `DEFAULT_TOOL_ARM_M`，或通过构造函数传入。
