# 键盘与笛卡尔 Servo

## 启动与操作

在有桌面显示的终端中运行：

```bash
export CONDA_ENVS_PATH="$PWD/.conda_envs"
conda activate replace_disk_robot
python examples/check_isolated_environment.py
# UR5e（默认）
python examples/keyboard_servo.py
# JAKA ZU5；只控制六个机械臂关节
python examples/keyboard_servo.py --model jaka
```

同时打开最近 10 秒的腕部力/力矩动态曲线：

```bash
python examples/keyboard_servo.py --plot-wrench
```

曲线来自去皮后的 `Wrench`，上图为 `Fx/Fy/Fz`（N），下图为 `Tx/Ty/Tz`（N·m）。UR5e 数据表达在 `wrist_ft_site`，JAKA 数据表达在 `tcp_fts_site`。绘图在独立进程中按 10 Hz 刷新，控制进程只向有界队列提交样本，不等待 Matplotlib 布局或重绘。队列满时丢弃绘图样本，不延迟 Servo。为避免静止时约 `1e-11` 的浮点噪声被自动放大，力和力矩默认至少显示 ±1 N 与 ±0.1 N·m；实际数据超出后坐标轴会扩展，不会裁剪数据。

点击新打开的控制窗口，使其获得键盘焦点。按住连续运动，松开保持当前目标。默认平移速度 10 mm/s、角速度 5°/s；可用 `--linear-speed 0.005 --angular-speed-deg 2` 调低速度。鼠标左键拖动调整视角，滚轮缩放。

| 按键 | 动作 | 坐标约定 |
|---|---|---|
| W / S | 向上 / 向下平移 | UR5e `world`、JAKA `jaka_base_link` 的 +Z / −Z |
| A / D | 向左 / 向右平移 | 所选平移坐标系的 +Y / −Y |
| R / F | 向前 / 向后 | 所选平移坐标系的 +X / −X；UR5e 中对应插入/退出 |
| Q / E | 朝左 / 朝右偏转 | 当前 TCP +Z / −Z |
| ↑ / ↓ | 抬头 / 低头 | 当前 TCP −Y / +Y |
| ← / → | 末端自旋 | 当前 TCP +X / −X，按右手定则 |
| 空格 | 停止并锁住运动输入 | Enter 恢复 |
| Enter | 清空按键并恢复接收新输入 | 不重新去皮、不重置机器人姿态 |
| Esc | 退出窗口 | 仿真结束 |

“上、下、左、右、前、后”不随观察相机改变。UR5e 平移在 `world` 表达，初始 TCP X 沿硬盘长度朝插口、Y 为宽度方向、Z 向上，旋转中心是 `pinch`。JAKA 平移在机械臂安装座 `jaka_base_link` 表达，旋转中心是 URDF 的 `tool0`；`base_x/base_y/base_yaw`、车轮和底盘执行器不进入 Servo。两个模型的姿态旋转均为当前 TCP 自身轴。

同时按方向相反的键会抵消；同时按多个平移或旋转键分别归一化，不增加相应速度模长。R/F 沿所选平移坐标系的 X 轴前后运动：UR5e 相对 `world` 固定，JAKA 相对 `jaka_base_link` 固定，均不会随 TCP 转姿态而改变。

点击力曲线窗口时机械臂暂停；重新点击机械臂窗口后，仅失焦暂停会自动解除，必须重新按运动键，不会恢复之前按住的键。空格停止、限力、跟踪错误或循环超时仍需 Enter 恢复，切换窗口不会清除这些故障。控制窗口独立管理按键，方向键不会同时控制 MuJoCo 默认相机。

## 模块和接口

```text
KeyControl（按键状态，可换成手柄或其他输入）
    ↓ CartesianJog
CartesianServo（运动学、阻尼求解、限速、限位；可选碰撞检查）
    ↓ JointState
应用中的限力门
    ↓ ArmPort.command_joint_positions
MujocoRobotAdapter（显式启用偏置力前馈）
```

- `control/key_control.py`：`press(key)`、`release(key)`、`clear()`、`command()`。无运动学、无机器人命令、无窗口库依赖。
- `core.types.CartesianJog`：`base_frame`、`linear_m_s`、`angular_rad_s`。线速度在明确的 base_frame 表达；角速度在当前 TCP 表达，单位 m/s、rad/s。该混合约定显式固定，不是同一坐标系下的普通六维 twist。
- `control/servo.py`：`reset(measured)` 初始化保持位置；`submit(command, now_s)` 刷新命令；`update(measured, dt_s, now_s)` 返回命名 `JointState`；`halt(measured, reason)` 停止并锁存故障。时间使用单调递增秒数。
- `kinematics/tool.py`：`FixedToolKinematics` 包装现有 `KinematicsPort`，根据固定 TCP 偏置变换 FK、Jacobian 和 IK。未修改 URDF 或已有默认末端帧。
- `kinematics/jaka.py`：加载完整 Tracer + JAKA URDF，但 Pinocchio 模型只有 `joint_1`～`joint_6` 六个可动自由度；默认返回 `jaka_base_link → tool0` 的 FK、6×6 Jacobian 和局部 DLS IK。
- `adapters/mujoco/interfaces.py`：按所选模型配置机械臂关节、位置执行器和 F/T 传感器名称；JAKA 没有夹爪执行器，底盘自由度不会被伪装成机械臂关节。
- `examples/keyboard_servo.py`：连接输入、Servo、限力门和仿真，唯一负责给机械臂发送最终命令，同时提供 GLFW 窗口。
- `visual.LiveTypePlotter`：统一显示核心数据类型的滚动时序曲线；其 GUI 更新可能耗时，不直接放入实时控制循环。
- `visual.ProcessTypePlotter`：独立绘图进程，通过非阻塞、有界队列接收样本。关闭或绘图失败不会关闭机械臂窗口；程序退出时回收绘图进程。

Servo 不导入 MuJoCo、GLFW、Pinocchio 或 ROS；只面向现有运动学、碰撞查询端口和命名关节数据。MuJoCo 碰撞查询使用单独的 scratch data，不能修改实时仿真状态。

不经过键盘使用 Servo：

```python
from replace_disk_robot.core import CartesianJog

servo.reset(arm.read_joint_state())
# 控制循环中刷新命令；线速度在 world、角速度在当前 TCP。
servo.submit(CartesianJog("world", [0, 0, 0.005], [0, 0, 0]), now_s)
target = servo.update(arm.read_joint_state(), dt_s=0.01, now_s=now_s)
# 应用层执行安全过滤，再通过 ArmPort 输出 target。
```

## 求解与停止语义

每周期在上一个关节目标处计算 TCP Jacobian，把 TCP 角速度旋转到 base frame，然后使用阻尼最小二乘求解关节增量。默认 100 Hz 控制、1000 Hz 仿真、最多 60 Hz 显示；渲染与控制调度分开。

默认关节速度上限 0.2 rad/s。线/角速度先按范数限幅，关节增量再统一缩放。候选位置越过关节限位时拒绝这一周期。若调用方显式传入碰撞检查器，`CartesianServo` 会检查旧目标至候选目标、实测位置至候选目标两段，默认最大关节采样间隔 0.002 rad；本示例**未启用**该检查。

未刷新命令超过 150 ms 后保持最后的关节目标；默认实测与目标偏差超过 0.08 rad 会锁存 `tracking_error`，本示例把该阈值放大到 10 rad，避免接触卡住时早于 20 N 力阈值触发。窗口循环停顿超过 150 ms 或按空格也会锁存停止。失焦暂停在重新获得焦点时单独恢复，不清除其他故障。松键后保持已生成的目标，实际机械臂仍有有限的伺服跟踪/制动过程，不宣称瞬时停止。

`MujocoRobotAdapter(..., compensate_bias=True)` 显式启用当前位置伺服的重力/偏置力前馈；原有默认接口行为不变。手臂和夹爪模型、场景初始位置均保持原样。

示例不再启用 `CartesianServo` 的碰撞检查，因此工具可以接触障碍物并让接触力上升。启动静止去皮后，测得的腕部力范数超过 20 N 时锁存 `force_limit` 并停止；当前示例把力矩阈值设为无穷大，只按力判断。Enter 不会重新去皮掩盖载荷，此入口仍不是力控插入或完整实机安全控制器。

## 验证

```bash
python -m pytest -q
python examples/keyboard_servo.py --headless --output simulation/mujoco/reports/keyboard_servo.json
python examples/keyboard_servo.py --model jaka --headless --output /tmp/jaka_keyboard_servo.json
```

无界面入口使用固定默认速度，依次对全部十二个键执行 0.5 秒命令和松键保持，检查实际 TCP 位移/旋转方向、绕 TCP 旋转的平移误差、无碰撞以及停止后目标不累积。UR5e 默认从 `home` 开始；JAKA 的 `home` 接近运动学奇异位形，因此动力学 Servo 回归默认从非奇异的 `low` 开始。可通过 `--keyframe` 覆盖，但测试失败应被如实报告。它经过真实 MuJoCo 动力学，但使用程序生成的按键状态，不等于操作系统键盘端到端验证。

单元测试覆盖按键组合/抵消/清空、坐标系、超时、奇异 Jacobian、限速、限位、关节名称重排、故障锁存和路径中间碰撞；TCP 的 FK/Jacobian 与 MuJoCo `pinch` 对照，且检查偏置 TCP 的逆运动学。

当前执行环境不能连接 X11 桌面，因此真实 GLFW 窗口及物理按键尚未端到端验证；请在有显示服务的桌面终端运行交互入口。不能以无界面结果替代该项。

## 开启绘图后按键不响应的排查

运行命令不变：`python examples/keyboard_servo.py --plot-wrench`。按键必须在 **Cartesian servo 机械臂窗口** 中操作，曲线窗口不转发机器人按键。

原实现把 `LiveTypePlotter.update()` 放在控制循环中；内部的 `draw_idle()` / `flush_events()` 仍可能实际重绘并耗时，超过 150 ms 会触发 `loop_timeout`。只预热首次布局无法消除后续慢绘图。此外，曲线窗口获得焦点会触发 `focus_lost`，原来的恢复逻辑要求用户手动按 Enter，表现为运动键无响应。

修复后，终端、窗口标题和画面显示停止原因及恢复提示。若仍无响应，先看状态：

- `focus_lost`：点击机械臂窗口，再重新按运动键。
- `force_limit`：接触力超过 20 N，先反向离开接触，再按 Enter 恢复；不会自动放宽阈值。
- `joint_limit`：该方向受阻，尝试相反方向，不累积未执行目标。
- `loop_timeout`、`stopped`、`tracking_error`：检查窗口/控制循环状态后，在机械臂窗口按 Enter 恢复。
- `[plot] Plot disabled`：绘图进程启动或刷新失败，终端会打印错误；机械臂窗口继续运行。

回归测试包含阻塞绘图进程与满队列、失焦/恢复且不重放按键、保留显式故障，以及 Q/W/E/R 在失焦恢复后经过实际 MuJoCo 的运动检查。真实桌面焦点行为仍需本机窗口验证。
