# Xbox 手柄遥控 JAKA

`jaka_joy_node.py` 是新的手柄入口，`jaka_keyboard_node.py` 和原键盘配置继续保留。
手柄连续速度指令使用 `common.node.JakaServo` 的共用单循环，与键盘应用分别组合：

```text
Xbox → 死区/比例速度/坐标转换 → 名义位姿 → 导纳 → CartesianServo
     → ForceLimitGuard → JakaRobotAdapter → 实机 EDG 或 MuJoCo 位置执行器
```

默认后端是 `sim`，速度请求读取 `jaka_joy_config.yaml`（当前为 **0.2 m/s、4°/s**），导纳使用共用算法，参数由手柄自己的配置文件提供。
实际速度还受位姿伺服和关节速度限制；当前位姿平移速度上限为 0.05 m/s，继续放大请求不会等比例加速。
手柄参数独立放在 `jaka_joy_config.yaml`，命令行覆盖配置文件。
仿真用 MJCF 质量、质心、传感器位姿去除负载重力；实机仍使用原有标定文件。

## 模块分层与复用

| 文件 | 职责 |
|---|---|
| `control/joy_control.py` | `JoySample` 输入状态和 `JoyControl` 通用速度映射，与 `KeyControl` 同层；处理死区、LB 使能、RB 精细速度、输入有效期和速度范数限制 |
| `adapters/joy.py` | `XboxController` 读取 SDL 手柄并生成 `JoySample`，处理连接状态；创建读取器时才加载 pygame |
| `applications/jaka_joy/joy_node.py` | `JakaJoyServo` 应用：JAKA 坐标转换、沿工具轴插拔、导纳接入、回中重锚定、停止与恢复，以及运行调度 |
| `applications/jaka_joy/jaka_joy_node.py` | 可执行应用入口；手柄配置、窗口、仿真组装及验证留在此应用目录 |
| `applications/common/` | 共用伺服生命周期、力处理、导纳参考、日志、分析和 MuJoCo 反馈后端 |

`JoyControl` 的接口与键盘控制类似：`update(sample)` 更新输入，`command(now_s)` 输出 `CartesianJog`，`clear()` 清除输入。
输入时间戳和 `now_s` 必须使用同一时钟。通用模块不读取设备，也不连接机器人：

```python
from replace_disk_robot.control import JoyControl, JoySample

joy = JoyControl(linear_speed_m_s=0.01, angular_speed_rad_s=0.05,
                 base_frame="tool")
joy.update(JoySample(timestamp_s=1.0, axes=(0., -1., 0., 0., 0., 0.),
                     buttons=frozenset({"lb"})))
jog = joy.command(now_s=1.0)  # 工具 +X 方向，0.01 m/s
joy.clear()
```

默认平移映射是左摇杆控制 −Y/+X、扳机控制 +Z；可通过 `translation_axes` 指定三个方向向量。
输出平移和角速度均属于 `base_frame`，具体机器人需要的坐标转换由应用完成。
原应用目录中的 `joy.py`、`joy_input.py` 已分别移入上述控制模块和设备适配层，应用直接引用新模块。

## 运行

使用仓库的隔离环境，避免 ROS 的 Pinocchio 与 NumPy 混用：

```bash
conda activate /home/a/replace_disk_robot/.conda_envs/replace_disk_robot
python -m pip install -e '/home/a/replace_disk_robot[joy]' --no-build-isolation

# 只读手柄状态，不创建机器人连接
python /home/a/replace_disk_robot/src/replace_disk_robot/applications/jaka_joy/jaka_joy_node.py --probe --seconds 10

# 实物 Xbox 手柄 + MuJoCo 可视化
python /home/a/replace_disk_robot/src/replace_disk_robot/applications/jaka_joy/jaka_joy_node.py

# tool 模式提供三个独立的 TCP 平移方向
python /home/a/replace_disk_robot/src/replace_disk_robot/applications/jaka_joy/jaka_joy_node.py --command-frame tool

# 无窗口，仍读取真实手柄
python /home/a/replace_disk_robot/src/replace_disk_robot/applications/jaka_joy/jaka_joy_node.py --headless --seconds 10

# 自动闭环验证，不需要手柄，不连接实机
python /home/a/replace_disk_robot/src/replace_disk_robot/applications/jaka_joy/jaka_joy_node.py --validate --output /home/a/replace_disk_robot/simulation/mujoco/reports/jaka_joy_validation.json
```

## 按键和摇杆

先松开 LB，摇杆、扳机回中，然后按住 LB 操作。SDL 使用命名按键，避免依赖 Linux/Windows 的原始轴编号。

| 操作 | 默认 base 模式 | tool 模式 |
|---|---|---|
| 按住 LB | 允许名义运动和导纳 | 相同 |
| 左摇杆左右 | `jaka_base_link` −X/+X，与键盘 A/D 一致 | `tool0` +Y/−Y |
| 左摇杆上下 | `jaka_base_link` +Z/−Z，与键盘 W/S 一致 | `tool0` +X/−X |
| RT / LT | 当前工具 +Z 插入 / −Z 退回 | 相同 |
| 右摇杆左右 | 基座 −Rx/+Rx | 工具 −Rx/+Rx |
| 右摇杆上下 | 基座 −Ry/+Ry | 工具 −Ry/+Ry |
| 十字键左 / 右 | 基座 +Rz/−Rz | 工具 +Rz/−Rz |
| 同时按 RB | 25% 精细速度 | 相同 |
| B | 锁定软件停止 | 相同 |
| A | 松开 LB、输入回中、卸载后，重新锚定并恢复 | 相同 |
| Back | 停止并退出 | 相同 |

base 模式保持已有键盘的“升降、左右、沿工具轴插拔”语义；工具轴接近基座 Z 时，升降和插拔方向会接近平行。
需要完整三维 TCP 平移时使用 `--command-frame tool`，左摇杆控制工具 X/Y，扳机控制工具 Z。
`tool0` 是当前 URDF 的工具坐标系，不是相机或观察者坐标系。
工具角速度始终转换为现有伺服要求的 TCP 本体角速度。

摇杆默认每轴 15% 死区，死区外连续缩放到满速；多轴组合限制平移和旋转各自的速度向量范数。
摇杆回中时取消对应平移/旋转通道尚未完成的名义运动，并从当周期反馈重新锚定保持目标；无需松开 LB，可再次推动摇杆。
按住 LB 时导纳仍可响应外力；回中不清零柔顺偏移和速度，也不会取消另一通道仍在进行的运动。
**松开 LB 会冻结关节保持目标并重置柔顺参考**；再次运动须先松开 LB、输入回中，再按 LB。
B、断连、输入过期、力限位及其他故障需要 A 恢复，持续受力时恢复被拒绝。
断连后轮询同一 SDL GUID 的手柄，重新连接也不会自动运动。B 是软件停止键。
控制器采用后台轮询，切换窗口不会暂停；松开 LB 或按 B 停止。

## 实机入口

与键盘版本一样，先通过原有 `jaka_start.py` 流程完成上电使能。
`--backend real` 明确选择 EDG，入口本身不执行上电操作：

```bash
# 读取实机反馈、补偿力和手柄，不发送运动
python /home/a/replace_disk_robot/src/replace_disk_robot/applications/jaka_joy/jaka_joy_node.py --backend real --dry-run --headless --seconds 10

# 实机手柄控制；可选择力曲线与日志
python /home/a/replace_disk_robot/src/replace_disk_robot/applications/jaka_joy/jaka_joy_node.py --backend real --plot-wrench-enable true --log-dir /home/a/replace_disk_robot/logs/jaka_joy_trial_001
```

`--log-dir` 必须是新的目录。日志复用原记录器，并增加每周期手柄状态、按键事件和手柄模块源文件快照。
仿真与实机共用控制和限位链，但仿真通过不代表实机验证完成。本次未连接真实 JAKA。

## 2026-10-12 初次验证结果（2 mm/s、1°/s）

- 自动 MuJoCo 闭环 **45/45**：base/tool 下各运动方向正反向、直接伺服/导纳、比例速度、死区、LB、B/A、断连、输入过期、实际外力响应和力限恢复。
- 半幅速度比例约 0.510，RB 精细模式约 0.260。
- 实际向仿真工具施加基座 −Z 方向 3 N：测得约 −3.008 N，2 秒后移动约 −16.50 mm；卸载 40 秒后残余约 0.324 mm。30 N 触发力停止，持续加载时 A 无法恢复。
- 真实 USB 手柄与 MuJoCo 窗口运行 30 秒，3749 次有效采样，空闲关节漂移 0、日志无丢失。这段运行没有人工按键，运动方向通过上述自动样本验证。
- 手柄及相关当前控制测试 84 项通过。旧键盘重构/记录器的 31 项比较测试失败，在原始 HEAD 的独立副本中同样复现；没有修改这些旧测试或键盘控制算法。

报告：`simulation/mujoco/reports/jaka_joy_validation.json`、`jaka_joy_live.json`。

## 左摇杆回中后继续运动修复

0.2 m/s 请求超过 0.05 m/s 的位姿跟踪上限，名义位姿按请求持续积分，机械臂受速度/关节限制无法及时跟上。
旧代码在回中后仅把输入速度归零，仍追赶积累的位姿目标；右摇杆 4°/s 未超过 20°/s 的位姿旋转上限，所以现象较轻。

手柄节点现在在平移/旋转指令从非零变为零时，重新锚定对应通道，并清除旧关节目标的提前量。
使用当周期的 EDG/MuJoCo 关节反馈，保留导纳状态、持续外力响应及另一通道的名义目标。
键盘入口不调用新增的重锚定方法，原键盘行为保持不变；本次没有调整用户的速度配置。

- 0.2 m/s 下，base/tool 两种模式、左摇杆 X/Y 正反向共 8 个回归全部通过：运动 1 秒后回中，继续按住 LB 4 秒，再次推动可继续运动。
- 修复前上述场景在回中后仍走约 160–164 mm；修复后最多 0.704 mm 的仿真收敛位移，且无需松开 LB 就能再次运动。逐项数值见 `simulation/mujoco/reports/jaka_joy_centering.json`。
- 当前完整 MuJoCo 验证 **53/53** 通过；比例速度比较单独使用未饱和的 2 mm/s 工况，避免把执行器限速误判为映射错误。
- **106 项**手柄、重锚定、导纳、伺服及运动学测试通过，包括保留外力响应和右摇杆连续控制。

报告范围是仿真闭环，不是实机机械臂验证。运行中的旧进程需要退出后重新启动才能加载修复。

## JoyControl 分层重构验证

- **108 项**相关测试通过，覆盖通用模块独立导入、输入映射、设备重连、JAKA 坐标转换、回中停止、导纳和日志源文件快照。
- 重构前后 **2000 组**随机输入和姿态的速度映射一致，最大数值差约 `1.39e-16`。
- 完整 MuJoCo 闭环 **53/53** 通过，报告与重构前完全一致；包含保持 LB 的 8 个左摇杆回中场景。
- 新适配层成功读取已连接的 Microsoft X-Box 360 pad；此项为设备读取检查。

## 方案参考

- [Pygame joystick 官方文档](https://www.pygame.org/docs/ref/joystick.html)：Python 直接读取轴、按钮和十字键，原始编号依设备而异。
- [Pygame SDL controller 官方文档](https://www.pygame.org/docs/ref/sdl2_controller.html)：Xbox 等标准手柄的命名轴/按钮及连接状态；本模块选用此方式。
- [ROS 2 teleop_twist_joy](https://github.com/ros2/teleop_twist_joy)：使能按钮、轴到速度映射和速度缩放。当前项目直接复用 Python 控制链，无需新增 ROS 运行依赖。

上述资料提供接入方式和通用控制模式；本项目的具体按键布局及工具轴插拔语义由现有 JAKA 控制代码决定。
