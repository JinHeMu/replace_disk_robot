# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目定位

固定式 UR5e + Robotiq 2F-85 + 腕部六维 F/T 的 2.5 英寸热插拔服务器硬盘更换工程。当前阶段以 MuJoCo 代替真实硬件，只搭建**可验证的框架**：算法由本项目掌控，MuJoCo / 未来 ROS 2 / 真机都是可替换适配器。当前验证等级为 **L2**（headless 动力学与信号链）；L0–L6 定义见 `docs/01_功能规格与验收标准.md`。规划/接触控制/感知/RL/完整状态机均未实现。

所有正式文档与规范在 `docs/`（中文）：功能规格(01)、模型参数与假设(02)、架构与接口(03)、开发与维护标准(04)、算法开发路线(05)。改代码前先读 03/04。

## 常用命令

本工程是独立 git 仓库（远程 `git@github.com:JinHeMu/hard_disk_robot.git`），以下命令在仓库根目录执行（脚本自身 `sys.path` 注入 `src/`，与 CWD 无关）：

```bash
pip install -e '.[mujoco]'   # 或 requirements.txt；仅需 mujoco>=3.1 numpy Pillow
# 如果当前用户 site-packages 只读导致安装失败，可跳过安装；下面脚本会自动把 src/ 加入 sys.path

python3 scripts/smoke_test.py                                # 冒烟：模型加载+信号链断言
python3 -m unittest discover -s tests -v                     # 全部 9 个单元测试
python3 scripts/run_scene.py                                 # 交互式查看器
python3 scripts/run_scene.py --headless --seconds 5          # 无显示器运行
MUJOCO_GL=egl python3 scripts/capture_scene.py               # 无显示器渲染固定相机截图
```

提交前检查（来自 `docs/04` 第 5 节）：跑 smoke_test + unittest、`git diff --check`；改动场景或碰撞几何时还要渲染截图检查初始穿模、错误接触、抽拉方向。

## 架构（六边形 / 端口-适配器）

```text
task / planning / contact / perception / safety
                    ↓ 只依赖
                core ports (core/types.py + core/ports.py)
                    ↑ 实现
       adapters/mujoco（现在）或 adapters/ros2（将来）
```

- **`core/`**：已实现。`types.py` 为不可变 dataclass 数据合同（`Pose`/`Wrench`/`JointState`/`TrajectoryPoint`），`__post_init__` 拒绝空坐标系、错误维度、NaN/Inf、零四元数；四元数一律 **wxyz**；`Wrench.as_vector()` 顺序为 force 在前、torque 在后。`ports.py` 为 Protocol（Arm/Gripper/ForceTorque/Mechanism/Kinematics/TrajectoryPlanner/ContactController），是算法层唯一可见的后端接口。
- **`safety/`**：已实现 `ForceLimitGuard`——确定性限力/限矩门，超限时冻结新关节目标（返回当前关节目标并置 trip 标志）。**持有最终命令过滤权**，RL/规划器/接触控制的输出都不得绕过；它是安全数据链检查，不是闭环力控。
- **`adapters/mujoco/`**：已实现 `MujocoRobotAdapter`、`MujocoWristFTAdapter`、`MujocoMechanismAdapter`、`load_model`/`reset_home`（依赖场景中 `home` keyframe）。**低层方法**（`command_arm`/`command_gripper`/`raw`/`tare`/`wrench`）只是 MuJoCo 调试接口，不属于跨后端合同；上层用 `read_joint_state`/`command_joint_positions`（按关节名交换）、`command_gripper_opening(opening_m)`（米制开口，适配器内部换算成 [255,0] 原始控制量）、`read_wrench`（带坐标系语义）。
- **`task/`**：只定义 `TaskPrimitive`/`TaskState` 枚举（稳定词表），无执行逻辑。
- **`kinematics/` `planning/` `contact/` `perception/` `baselines/moveit/` `adapters/ros2/`**：预留空目录（仅 README/占位）。新算法按 04 节 3 的流程落地到这些目录，不得塞进适配器或脚本。

### 依赖纪律（`docs/04` 第 2 节，有测试强制部分）

- 算法层与 `core` **不得 import** `mujoco`/`rclpy`/MoveIt/厂商 SDK；只有 `adapters/mujoco` 可以 import `mujoco`。包根 `__init__.py` 不加载任何后端——应用自行选择适配器。`tests/test_core_contracts.py::test_core_does_not_import_runtime_backends` 会检查这一点。
- 算法自研：IK 框架、轨迹生成、导纳/力位控制、任务状态机、Residual RL 与安全限制。成熟库（Pinocchio、FCL、优化器、ROS 2 通信）可用；MoveIt 只做基线对照与显式 fallback，主测试在未装 MoveIt 时须通过。

### 场景模型（已独立、无仓库外依赖）

- 场景由 `simulation/mujoco/scene.xml` 装配：`<include>` `models/ur5e/ur5e_with_gripper.xml`（UR5e+2F-85 MJCF，网格在本目录 `models/ur5e/assets/`）与 `models/server_2u_8bay.xml`。模型文件与网格解析均相对各自 XML 文件所在目录；通过 `hard_disk_robot.adapters.mujoco.load_model()` 加载时，底层使用仓库绝对路径，因此可从任意工作目录加载，工程本身不依赖仓库外的文件。
- `models/ur5e/` 是从旧工作区 `mujoco_sim/ur5e/` 工程 vendoring 进来的副本：若上游模型有更新，需同步 xml 与 `assets/`（已在文件头注释标明）。改动装配时不得再把路径指回仓库外。
- `simulation/mujoco/models/server_2u_8bay.xml`：2U 服务器、8×2.5 英寸盘位、19 英寸机架近似（均为简化几何，非厂商 CAD——替换真实参数前的优先级见 `docs/02`）。

## 稳定约定（`docs/03`）

坐标系与单位：`world` 为 MuJoCo 世界系、+Z 朝上；服务器深度沿世界 X，插入朝 **+X**、抽出朝 **-X**（`target_drive_slide` 正值=已抽出，行程 [0,0.16] m）；锁扣按压 `target_latch_press` [0,0.006] m；`wrist_ft_site` 为 F/T 局部系，局部 +Z 从法兰指向夹爪。单位：m、rad、N、N·m。

MuJoCo 稳定名称（模型层面不得改名）：

| 类型 | 名称 |
|---|---|
| 手臂执行器 | `shoulder_pan` … `wrist_3`（弧度位置目标） |
| 夹爪执行器 | `fingers_actuator`（0 张开 / 255 闭合，上层用米制开口） |
| F/T 传感器 | `wrist_force`、`wrist_torque`（局部系） |
| 机构关节 | `target_drive_slide`、`target_latch_press` |
| 新盘 | `replacement_drive_free`（7 维 free joint） |
| TCP site | `pinch` |

模型量级：`nu=9`（6 手臂 + 夹爪 + 2 机构执行器）、2 个传感器 6 维 sensordata、`home` keyframe。

F/T 语义（`docs/02` 第 3 节）：MuJoCo force/torque 传感器测量 site 所在子刚体与父刚体间的交互量、以 site 局部系表达，因此 `wrist_ft_link` 是无关节 dummy body；原始读数含夹爪与负载重力。`tare()` 只记录当前姿态偏置，重力补偿/滤波/噪声/标定矩阵均未实现。

## 验证等级与真实硬件门槛

- L0 静态合同 → L1 模型合同/单测 → L2 headless 动力学与信号链（**当前**）→ L4 接触闭环 → L5 真机只读/标定 → L6 风险评审后低速动作。只有完成该等级全部验收才能声称达到。
- 真机硬门槛（`docs/04` 第 6 节）：`command_output_enabled` 默认必须为 false；必须先完成外参/TCP/指尖/F-T 零点与重力标定；软治具→报废托架→真服务器的顺序；L5 成功不等于 L6 可批准。
