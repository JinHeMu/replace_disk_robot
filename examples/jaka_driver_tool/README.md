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
| [jaka_keyboard](../../src/replace_disk_robot/applications/jaka_keyboard/README.md) | 键盘控制入口与参数已移至 applications 功能包；使用指南见链接 |

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

### 4. 实机键盘 Cartesian servo

先确认机器人已经由 `jaka_start.py` 上电使能，再用只读模式检查 EDG/F/T 和键盘：

```bash
python src/replace_disk_robot/applications/jaka_keyboard/jaka_keyboard_servo.py --dry-run
```

没有显示服务时可用无界面只读模式：

```bash
python src/replace_disk_robot/applications/jaka_keyboard/jaka_keyboard_servo.py \
    --dry-run --headless --seconds 3
```

真正进入伺服时先用很小的速度和力限：

```bash
python src/replace_disk_robot/applications/jaka_keyboard/jaka_keyboard_servo.py \
    --linear-speed 0.002 \
    --angular-speed-deg 1 \
    --max-force-n 5 \
    --max-torque-nm 2
```

实机键盘脚本默认读取同目录的 `jaka_keyboard_servo_config.yaml`。速度、力限、死区、
重力补偿和导纳参数都可以直接改这个 JSON；命令行显式传入的参数会覆盖配置文件。
使用其他文件时加 `--config <file.yaml>`。

默认按 `jaka_base_link` 控制：W/S/A/D 和 Q/E/方向键使用 base 轴，
R/F 始终沿当前 `tool0` +Z/−Z（蓝色轴）插入/退出。初始姿态直接读取启动时的实测关节角。
需要 W/S/A/D 和姿态旋转相对末端 `tool0` 控制时加 `--command-frame tool`，
此时 R/F 仍然沿 `tool0` +Z/−Z，不会退回默认的 +X/−X。

默认会读取 `tool/ft_gravity_samples_identified.json`，在线扣除辨识出的
传感器零偏、末端负载重力和质心力矩；可用 `--gravity-json <file>` 指定其他
辨识结果，或用 `--gravity-compensation-enable false` 关闭。加 `--tare-compensated`
可以在启动时把当前**重力补偿后的六维力**作为零点，后续都减去该残余偏置。
默认同时打开非阻塞的六维力曲线窗口；不需要时加 `--plot-wrench-enable false`，
`--plot-wrench-enable true` 可显式保持打开。

运行后会打开一个名为 `JAKA keyboard servo` 的 GLFW 窗口；键盘按键必须在
该窗口内输入，不能在启动脚本的终端里按。终端里出现的 `s`、`w` 等字符
只表示终端获得了焦点，不会传给 JAKA。点击该窗口后再按住按键即可。

### 5. 导纳柔顺模式（可选）

#### 5.1 数据流

`--admittance` 在键盘和伺服之间插入一级六轴导纳：

```
按键 → 名义位姿积分 → 导纳偏移（tool0 轴）→ 偏移钳位 → 位姿跟踪伺服 → 关节命令
EDG 原始六维值 → 去负载重力/零偏 → 移到 tool0、滤波 → 导纳
```

按键只推动**名义位姿**；外力（用第 4 节的在线重力补偿去掉负载自重和零偏之后）产生
柔顺偏移。因此 `--admittance` **要求重力补偿处于打开状态**，否则负载自重会一直把偏移
顶跑；同时给出 `--admittance` 和 `--gravity-compensation-enable false` 时脚本会在联网之前直接
报错退出。

#### 5.2 前置条件：先做负载重力辨识

```bash
python tool/collect_ft_gravity_data.py --output tool/ft_gravity_samples.csv   # 采 12 个以上姿态
python tool/identify_ft_payload.py tool/ft_gravity_samples.csv                # 生成 _identified.json
```

默认使用 `tool/ft_gravity_samples_identified.json`，其他结果用 `--gravity-json <file>` 指定。
辨识要求控制器输出**未做重力补偿**的原始六维值；若报
`identified gravity load is nearly zero`，先换 `--torque-sensor-mode` 重采。详见
`tool/README.md`。

#### 5.3 第一步：只读验证（不使能 Servo、不发运动指令）

导纳链照常积分并打印，可以先把符号和量级确认清楚：

```bash
python src/replace_disk_robot/applications/jaka_keyboard/jaka_keyboard_servo.py --dry-run --admittance --plot-wrench-enable false
```

每秒两行状态，导纳字段追加在 `|T|` 之后：

```
[keyboard] t= 2.000s q=[...] cmd=[...] |F|= 0.041N |T|= 0.012Nm |Fext|= 0.04N dx= 0.3mm dth= 0.0deg axes=111000 tracking
```

| 字段 | 含义 |
| --- | --- |
| `|Fext|` | 去掉负载重力和零偏后的外力模长（导纳用的就是它） |
| `dx` | 当前柔顺平移偏移量 |
| `dth` | 当前柔顺旋转偏移量 |
| `axes` | 参与导纳的扳手轴掩码，`111000` = 只开平移 |

**判据**：机械臂静止且手没碰工具时 `|Fext|` 应接近 0（几个 0.01 N），`dx` 不随时间增长；
用手沿 base **+X** 推工具时 `dx` 为正，沿 **−X** 推则为负。若静止时 `|Fext|` 就有几牛且
`dx` 持续增大，说明重力补偿没有对上当前负载（`--gravity-json` 不是这次的辨识结果，或
负载换过），此时可用 `--tare-compensated` 把残余偏置作为零点，不要直接进入下一步。

#### 5.4 第二步：实机只开平移轴

确认符号正确后进入伺服，先用保守增益和小钳位：

```bash
python src/replace_disk_robot/applications/jaka_keyboard/jaka_keyboard_servo.py --admittance --plot-wrench-enable false \
  --adm-mass 2 2 2 .02 .02 .02 \
  --adm-damping 60 60 60 1.5 1.5 1.5 \
  --adm-stiffness 300 300 300 30 30 30 \
  --adm-max-velocity .05 .05 .05 .17 .17 .17 \
  --adm-max-offset-m 0.02 --adm-max-offset-deg 5 \
  --max-force-n 6 --max-torque-nm 8
```

用手轻推工具，机械臂应顺着推力让开、松手后回到名义位姿；按键仍然可以正常移动名义位姿。
确认稳定后再加大钳位或降低刚度。

#### 5.5 第三步：打开旋转柔顺

如果只想关闭某个轴的导纳，在 `--admittance-axes` 之后再加 `--adm-disable-axes`。
例如关闭 Z 轴竖直导纳、保留 X/Y 平移：

```bash
python src/replace_disk_robot/applications/jaka_keyboard/jaka_keyboard_servo.py --admittance --plot-wrench-enable false \
  --adm-disable-axes z
```

平移验证通过后加 `--admittance-axes all`，六个轴都参与导纳，此时 `--adm-max-offset-deg`
开始起作用：

```bash
python src/replace_disk_robot/applications/jaka_keyboard/jaka_keyboard_servo.py --admittance --plot-wrench-enable false \
  --admittance-axes all --adm-max-offset-m 0.03 --adm-max-offset-deg 8 \
  --max-force-n 6 --max-torque-nm 8
```

#### 5.6 参数速查

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--admittance` | 关 | 开启导纳；不开启时行为与原来完全一致 |
| `--admittance-axes` | `translation` | `translation` 只开平移，`all` 六轴全开 |
| `--adm-disable-axes` | 空 | 在 `--admittance-axes` 基础上禁用单轴，例如 `z` 关闭竖直导纳 |
| `--adm-mass` / `--adm-damping` / `--adm-stiffness` | 2/60/300（平移） | 各六个分量：kg、N·s/m、N/m；后三个是旋转的 kg·m²、N·m·s/rad、N·m/rad |
| `--adm-max-velocity` | 0.05 m/s、0.17 rad/s | 六个分量的速度上限 |
| `--adm-max-offset-m` / `--adm-max-offset-deg` | 0.03 / 8 | 柔顺偏移相对名义位姿的钳位，实机必设 |
| `--gravity-json` | `tool/ft_gravity_samples_identified.json` | 导纳输入所依赖的在线重力补偿辨识结果 |
| `--gravity-compensation-enable false` | 关 | 关闭重力补偿；与 `--admittance` 同时使用会直接报错 |
| `--tare-compensated` | 关 | 把当前补偿后残差作为零点，去掉残余偏置 |
| `--deadband-force` / `--deadband-torque` | 0 / 0 | 补偿后力/力矩死区；例如 `1.0` N、`0.5` N·m |

#### 5.7 安全要点与常见现象

- **偏移钳位**：`AdmittanceController` 本身不设上限，实机必须靠
  `--adm-max-offset-m/-deg` 把柔顺位移限制在名义位姿附近。
- **力矩限位**：力臂 0.4 m，工具末端几牛的接触就会在 tool0 上产生超过默认 2 N·m 的力矩，
  导纳模式下要显式放宽 `--max-torque-nm`，否则一碰就触发力限停机。
- **力限保护收到的是去重力后的外力**，负载自重不再占用 `--max-force-n` 的额度；
  `resume()` 的 80% 阈值判断同理，否则重载下 Enter 永远无法恢复。
- **停机即重新锚定**：`Space`/`Enter`/`Esc` 与无导纳模式一致；任何停机（含 `focus_lost`、
  力限、跟踪误差）都会把名义位姿和导纳偏移重新锚定到实测位姿，不会重放旧的柔顺偏移。
  故障期间导纳积分器不积分，避免 wind-up。
- `--tare-compensated` 只在重力补偿已打开时可用，二者叠加可以消掉辨识残差带来的稳态偏移。

| 现象 | 处理 |
| --- | --- |
| 静止时 `dx` 持续增大 | 重力补偿没对上当前负载：核对 `--gravity-json`，或加 `--tare-compensated` |
| 一接触就 `force_limit` 停机 | 放宽 `--max-torque-nm`（力臂长），或降低导纳刚度 |
| 打印 `tracking_error` | 柔顺偏移变化快于伺服跟随：降低 `--adm-max-velocity` 或导纳刚度，或放宽 `--max-joint-error-rad` |
| 打印 `joint_limit` | 名义位姿已接近关节限位，用按键把名义位姿挪回工作空间中间 |
| 手感太硬 / 太软 | 调 `--adm-stiffness`（越小越软）与 `--adm-damping`（抑制来回晃动） |

### 6. 关闭机器人

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

### 模块化节点与控制日志

键盘入口和参数已移至 `src/replace_disk_robot/applications/jaka_keyboard/`。
`jaka_keyboard_servo.py` 保留原控制逻辑，仅更新导入及资源路径。
新版入口为 `jaka_keyboard_node.py`，由
`replace_disk_robot.applications.jaka_keyboard.node` 调度。参数、力处理、参考生成和窗口职责已分离。
`tests/fixtures/jaka_keyboard_servo_before_refactor.py` 另保存原版本快照，用于离线回归。

将原启动命令中的脚本名换为 `jaka_keyboard_node.py`，保留原参数；追加
`--log-dir logs/jaka_trial_001` 可记录每个控制周期，目录必须不存在。
记录默认关闭；可同时加 `--dry-run` 先检查数据路径。日志写盘使用后台线程，丢样和写盘错误保存在结束摘要中。

```bash
python tool/analyze_jaka_control_log.py logs/jaka_trial_001 --plots
```

分析器只读日志，生成导纳离散公式校验、参考跟踪指标和曲线，不连接机器人。
模块职责、字段与验收边界见 [拆分与日志说明](../../docs/jaka_keyboard_refactor.md)。

简明使用指南见 [jaka_keyboard README](../../src/replace_disk_robot/applications/jaka_keyboard/README.md)。
