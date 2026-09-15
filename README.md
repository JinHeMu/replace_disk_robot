# 硬盘精密插入实验

面向硬盘精密插入的 MuJoCo 仿真与实机 Jaka 工具集合。

当前仓库主要包含：

- UR5e + Robotiq 2F-85 + 腕部六维力传感器的 MuJoCo 插入场景；
- Tracer + JAKA ZU5 的 MuJoCo 模型、运动学与实机 Python SDK 适配；
- 键盘笛卡尔 Servo、力处理、导纳柔顺和限力安全链；
- Jaka 实机启动、关闭、F/T 读取、EDG servo 和键盘 Servo 工具；
- 一个不依赖 PyTorch 的 Residual RL 力控微调 Demo。

## 场景与当前能力

UR5e 场景中，机械臂启动时已经夹好硬盘：

- 硬盘：**177 × 106 × 26 mm** 长方体；
- 通道净口：**106.5 × 26.5 mm**，硬质仓壁每侧间隙 **0.25 mm**；
- 仓体深度：**190 mm**，无入口倒角、无后挡板；
- 内部两侧弹性摩擦导轨：低速对正插入时平均轴向阻力约 **7.5 N**；
- 初始硬盘前端距入口 **10 mm**，未来目标插深 **50 mm**，夹爪保持在口外。

当前已经实现：

- 场景/接触/摩擦信号链验证；
- 键盘名义位姿 Servo；
- `WrenchProcessor` 力坐标变换、滤波、死区；
- 导纳柔顺（默认仅平移轴，可开六轴）；
- Jaka EDG 状态读取与 125 Hz servo 接口；
- Jaka 实机启动、关闭、F/T 测试和键盘 Servo；
- Residual RL Demo 的环境、观测、动作、奖励、安全限幅与训练入口。

当前未实现：

- 完整的自主闭环插入任务和成功判据驱动状态机；
- 实机导纳插入控制；
- 摩擦夹持、打滑、视觉、ROS 2 实机输出；
- 生产级 SAC 训练；
- 仿真结果不能作为实机安全证明。

## 环境配置

当前仓库使用项目本地的隔离 Conda 环境：

```text
/home/a/replace_disk_robot/.conda_envs/replace_disk_robot
```

系统 Conda envs 目录是只读挂载，所以不要尝试把环境创建到
`/home/a/miniconda3/envs`。环境应当保留在仓库的 `.conda_envs/` 下，该目录已经被
`.gitignore` 忽略。

### 激活环境

最稳定、与当前目录无关的方式是直接使用绝对路径：

```bash
conda activate /home/a/replace_disk_robot/.conda_envs/replace_disk_robot
```

如果希望按名字 `replace_disk_robot` 激活，需要先把 `.conda_envs` 加入
`CONDA_ENVS_PATH`：

```bash
cd /home/a/replace_disk_robot
export CONDA_ENVS_PATH="$PWD/.conda_envs"
conda activate replace_disk_robot
```

注意：`CONDA_ENVS_PATH` 只对当前 shell 生效。`~/.bashrc` 和 `~/.condarc`
在当前工作区是只读的，因此无法替用户永久写入；每个新终端都需要重新 export，
或者直接使用上面的绝对路径激活。

### 首次创建或重建环境

```bash
cd /home/a/replace_disk_robot
CONDA_ENVS_PATH="$PWD/.conda_envs" conda env create -f environment.yml
```

如果只是重新安装当前仓库的 editable 包：

```bash
conda activate /home/a/replace_disk_robot/.conda_envs/replace_disk_robot
python -m pip install -e . --no-deps --no-build-isolation
```

### 环境内容与隔离

`environment.yml` 提供：

- Python 3.12；
- NumPy 2.x；
- MuJoCo 3.12.x；
- Pinocchio 4.x；
- Pillow、Matplotlib、pytest；
- 当前仓库 `pip install -e .`。

激活时会清空 `PYTHONPATH`、ROS、`LD_LIBRARY_PATH`、`CMAKE_PREFIX_PATH` 等变量，
避免宿主 ROS Humble 和 NumPy/Pinocchio ABI 污染。

运行隔离检查：

```bash
python examples/check_isolated_environment.py
```

预期输出包含：

```text
ISOLATION CHECK: PASS
```

VS Code 默认解释器配置在 `.vscode/settings.json`，指向：

```text
${env:HOME}/replace_disk_robot/.conda_envs/replace_disk_robot/bin/python
```

## 快速运行

```bash
# 环境检查
python examples/check_isolated_environment.py

# 基础动力学 smoke
python examples/smoke_test.py

# 全量测试
python -m pytest -q
```

## 场景、摩擦与接触验证

```bash
# 打开 UR5e 插入场景
python examples/scene_display_exp.py

# 入口近景，显示 site 坐标系
python examples/scene_display_exp.py --camera insertion_cam --show-sites

# 无界面保持初始状态 5 秒
python examples/scene_display_exp.py --headless --seconds 5

# 位置驱动摩擦验证，输出 friction_validation.json
python examples/validate_insertion_friction.py

# 接触测力验证
python examples/validate_insertion_contact.py \
    --output simulation/mujoco/reports/contact_validation.json

# 重新渲染截图
MUJOCO_GL=egl python examples/capture_scene.py
```

摩擦报告：

```text
simulation/mujoco/reports/friction_validation.json
```

和：

```text
simulation/mujoco/reports/contact_validation.json
```

7.5 N 是标定后的低速持续滑动阻力参考，不是恒定力目标；入口、换向和瞬态峰值可能更高。

## 键盘 Servo 与导纳

```bash
# 默认 UR5e
python examples/keyboard_servo.py

# 显示腕部 F/T 曲线
python examples/keyboard_servo.py --plot-wrench

# JAKA ZU5
python examples/keyboard_servo.py --model jaka

# JAKA 从实机启动记录姿态 up 开始
python examples/keyboard_servo.py --model jaka --keyframe up

# JAKA 所有按键改用当前 tool0 坐标系
python examples/keyboard_servo.py --model jaka --command-frame tool

# 键盘名义位姿 + 导纳柔顺
python examples/keyboard_servo.py --admittance
python examples/keyboard_servo.py --model jaka --admittance

# 六轴柔顺（谨慎使用）
python examples/keyboard_servo.py --admittance --admittance-axes all

# 无界面导纳回归
python examples/keyboard_servo.py --headless --admittance --model ur5e

# 无接触键盘动力学回归
python examples/keyboard_servo.py --model ur5e --headless
python examples/keyboard_servo.py --model jaka --headless
```

按键语义：

| 按键 | 动作 | 坐标约定 |
|---|---|---|
| W / S | 升高 / 降低 | base 模式下沿 kinematics base 的 ±Z；tool 模式下沿 TCP 的 ±Z |
| A / D | 左 / 右平移 | base 模式下沿 base 的 ±Y；tool 模式下沿 TCP 的 ±Y |
| R / F | 插入 / 退出 | UR5e：base 模式沿 `world` ±X；JAKA：始终沿当前 tool0 +Z / −Z（蓝色轴） |
| Q / E | 左右偏转 | base 模式按 base ±Z；tool 模式按 TCP ±Z |
| ↑ / ↓ | 抬头 / 低头 | base 模式按 base ∓Y；tool 模式按 TCP ∓Y |
| ← / → | 末端自旋 | base 模式按 base ±X；tool 模式按 TCP ±X |
| 空格 | 停止并锁存输入 | Enter 恢复 |
| Enter | 重置按键并恢复输入 | 不清除导纳状态/力零点 |
| Esc | 退出 | 仿真结束 |

UR5e 和 JAKA 默认都是 `base` 模式；只有显式传 `--command-frame tool` 时
W/S/A/D 和旋转键才改用当前 TCP/tool0 轴。JAKA 的 R/F 是例外，无论哪种模式都沿
当前 tool0 +Z/−Z。

导纳模式的数据链：

```text
KeyControl -> KeyboardAdmittanceController -> AdmittanceController
           -> CartesianServo.submit_pose()
           -> 限力门
           -> ArmPort.command_joint_positions()
```

外力进入导纳前会做：

1. F/T frame → 运动学 base frame 旋转；
2. 力矩参考点平移到 TCP；
3. 外部载荷符号约定；
4. 低通滤波与逐轴死区。

默认只对平移三轴开导纳；旋转按键仍可改变名义姿态。`--admittance-axes all`
才开放六轴柔顺。

## Jaka 实机工具

实机工具位于：

```text
examples/jaka_driver_tool/
```

推荐先只读验证：

```bash
# 登录、上电、使能、设置 LPF/torque mode
python examples/jaka_driver_tool/jaka_start.py

# 只登录读状态，不做电源/使能动作
python examples/jaka_driver_tool/jaka_start.py --read-only

# 读取 EDG F/T 并输出 CSV
python examples/jaka_driver_tool/jaka_ft_test.py \
    --seconds 10 \
    --csv simulation/mujoco/reports/jaka_ft.csv

# EDG servo dry-run，不进入 servo 模式
python examples/jaka_driver_tool/jaka_edg_servo.py --dry-run --seconds 5

# 实机键盘 Servo 的 dry-run
python examples/jaka_driver_tool/jaka_keyboard_servo.py --dry-run

# 关闭
python examples/jaka_driver_tool/jaka_stop.py
```

正式进入实机 Servo 前必须有人手边急停，并重新确认：

- 坐标系与 TCP；
- F/T 符号、零偏、工具重力；
- 速度、力/力矩限制；
- 机器人和工作空间边界。

实机 Jaka keyboard Servo 的默认 `base` 模式同样使用：

- W/S/A/D 和旋转键 → `jaka_base_link`；
- R/F → 当前 `tool0` +Z/−Z 插入/退出。

## Jaka 导纳实验

JAKA `low` 姿态下的虚拟外力实验：

```bash
# 默认会写报告，部分环境需要离屏渲染
MUJOCO_GL=egl python examples/jaka_admittance_experiment.py

# 不渲染 GIF/图片
python examples/jaka_admittance_experiment.py --no-render

# 指定输出目录
python examples/jaka_admittance_experiment.py --output-dir /tmp/jaka_adm
```

输出目录：

```text
simulation/mujoco/reports/jaka_admittance_experiment/
├── report.json
├── directions.png
├── force_sweep.png
├── loaded_poses.png
├── step_response_plus_x.png
└── step_response_plus_x.gif
```

这是确定性的虚拟外力实验，不是实机力控。

## Residual RL Demo

RL 方案说明见：

- [强化学习力控微调方案](docs/09_强化学习力控微调方案.md)
- [rl/README.md](rl/README.md)

运行 Demo：

```bash
# 零残差经典基线
python -m rl --policies zero --episodes 1 --seed 7

# zero + scripted 残差对比
python -m rl --policies zero scripted --episodes 1 --seed 7 \
    --record-history --output rl/artifacts/demo_report.json

# 固定 seed 评估
python -m rl.evaluate --policy zero --episodes 5 --seed 0
python -m rl.evaluate --policy scripted --episodes 5 --seed 0

# 依赖免安装 CEM Demo 训练
python -m rl.train --generations 3 --population 6 \
    --episodes-per-candidate 1 --seed 0

# RL 测试
python -m pytest rl/tests -q
```

`rl/requirements-rl.txt` 目前只包含可选的 `PyYAML`；核心 Demo 和测试不依赖
PyYAML、PyTorch 或 Gym。`rl/train.py` 是无额外依赖的 CEM Demo，不是生产 SAC
训练器。

## 代码结构

```text
src/replace_disk_robot/
├── adapters/
│   ├── jaka/          # Jaka Python SDK adapter、EDG 状态、125 Hz servo
│   ├── mujoco/        # MuJoCo 模型、机械臂/F-T/碰撞 adapter
│   └── ros2/          # 预留 ROS 2 边界，尚未实现
├── contact/           # 力处理与接触
├── control/           # Servo、键盘映射、导纳与导纳运动
├── core/              # Pose/Wrench/JointState/AdmittanceState 和端口协议
├── kinematics/        # UR5e、Jaka、平面臂、固定 TCP
├── planning/          # 碰撞、RRT、轨迹优化
├── safety/            # 限力门
├── task/              # 任务状态词表
└── visual/            # 实时力和类型曲线

rl/                    # Residual RL Demo：环境、观测、动作、安全、奖励、训练
examples/              # 仿真、Servo、验证、Jaka 工具
tests/                 # 核心、场景、Servo、导纳、Jaka 单元测试
docs/                  # 中文设计与验收文档
simulation/mujoco/     # 场景 XML、模型、截图、验证报告
```

## 接口与算法所有权

- `core.types`：`Pose`、`Wrench`、`JointState`、`AdmittanceState` 等显式单位和坐标系合同。
- `core.ports`：`ArmPort`、`ForceTorquePort`、`AdmittanceControllerPort`、`KinematicsPort` 等。
- `MujocoRobotAdapter` / `MujocoWristFTAdapter`：实现 MuJoCo 侧 `ArmPort` / `ForceTorquePort`。
- `JakaRobotAdapter` / `JakaWristFTAdapter`：实现 Jaka 侧 `ArmPort` / `ForceTorquePort`，使用 EDG 和本地 `core` 类型。
- `WrenchProcessor`：F/T 坐标变换、参考点平移、符号、滤波和死区。
- `KeyboardAdmittanceController`：键盘名义位姿积分、超前限幅和导纳状态。
- `AdmittanceController`：六轴导纳模型、限幅和状态复位。
- `CartesianServo`：笛卡尔速度到位姿/关节目标，含限速、限位、跟踪和可选碰撞检查。
- `ForceLimitGuard`：最终快速限力门。
- `rl/`：只输出有界残差；不直接写 MuJoCo `ctrl`，策略只依赖 NumPy 接口。

算法层不依赖 MuJoCo、ROS 或厂商 SDK；后端相关读写放在 `adapters/`。

## 文档

- [功能规格与验收标准](docs/01_功能规格与验收标准.md)
- [模型参数与假设](docs/02_模型参数与假设.md)
- [架构与接口](docs/03_架构与接口.md)
- [开发与维护标准](docs/04_开发与维护标准.md)
- [后续算法开发路线](docs/05_算法开发路线.md)
- [场景重建验收报告](docs/06_场景重建验收报告.md)
- [键盘与笛卡尔 Servo](docs/07_键盘与笛卡尔Servo.md)
- [加长硬盘仓与摩擦验证](docs/08_加长硬盘仓与摩擦验证.md)
- [强化学习力控微调方案](docs/09_强化学习力控微调方案.md)
- [Residual RL Demo](rl/README.md)
- [Jaka 实机工具](examples/jaka_driver_tool/README.md)

## 安全与证据边界

- 仿真通过不等于实机安全；真机运动前必须完成坐标系、TCP、工具重力、F/T 标定和只读验证。
- 7.5 N 是平均滑动阻力参考，不是恒定力控制目标；峰值可能更高。
- 导纳参数目前是仿真起步值，不是实机整定值。
- `rl/train.py` 是 Demo 优化器，不是生产 SAC 实现，也不能作为实机策略安全证明。
- 任何硬件动作都必须在有人值守、急停可用、低速小幅度和明确退出条件下进行。
