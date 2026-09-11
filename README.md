# 硬盘精密插入实验

固定 UR5e + Robotiq 2F-85 + 腕部六维力传感器的 MuJoCo 实验场景。启动时机械臂已经夹好硬盘，用于后续开发力控插入。旧服务器、托架、锁扣、暂存台和完整换盘示例已移除。

硬盘为 **177 × 106 × 26 mm** 长方体；沿 177 mm 长度方向水平插入 **106.5 × 26.5 mm** 的矩形通道，硬质仓壁每侧间隙 **0.25 mm**。仓体深 **190 mm**，无入口倒角、无后挡板；内部两侧弹性摩擦导轨会压紧硬盘，推进需要克服摩擦阻力，当前按实测目标标定为低速滑动时约 **7.5 N**（瞬时峰值可更高）。夹爪从后端 106 mm 短边的中间伸入，沿 26 mm 厚度夹紧，采用理想刚性夹持。

初始硬盘前端距入口 10 mm；未来目标插深为 50 mm，夹爪保持在口外。本次实现的是场景和接触测力验证，**没有实现闭环力控插入、抓取打滑或真实硬件控制**。

详见 [加长硬盘仓与摩擦验证](docs/08_加长硬盘仓与摩擦验证.md)。

## 运行

系统 Conda envs 目录只读；本仓库环境位于 `.conda_envs/replace_disk_robot`，使用前需要把该目录加入 Conda 搜索路径：

```bash
export CONDA_ENVS_PATH="$PWD/.conda_envs"
conda activate replace_disk_robot
python examples/check_isolated_environment.py
python examples/scene_display_exp.py
# 入口近景，或无界面保持初始状态 5 秒
python examples/scene_display_exp.py --camera insertion_cam --show-sites
python examples/scene_display_exp.py --headless --seconds 5
```

首次安装环境使用：

```bash
CONDA_ENVS_PATH="$PWD/.conda_envs" conda env create -f environment.yml
```

环境提供 MuJoCo、NumPy、Pinocchio 和 Pillow；激活时隔离宿主 ROS 路径。所有示例从自身路径寻找模型，可在任意工作目录运行。

```bash
python examples/smoke_test.py
python -m pytest -q
# 有界位置运动制造接触，验证腕部测力；不是力控插入
python examples/validate_insertion_friction.py
python examples/validate_insertion_contact.py --output simulation/mujoco/reports/contact_validation.json
MUJOCO_GL=egl python examples/capture_scene.py
```

![整体场景](simulation/mujoco/captures/overview_cam.png)

![初始夹持](simulation/mujoco/captures/grasp_cam.png)

## 键盘控制末端

```bash
python examples/keyboard_servo.py
# 同时显示腕部 Fx/Fy/Fz 和 Tx/Ty/Tz 动态曲线
python examples/keyboard_servo.py --plot-wrench
```

按住 W/S 升降、A/D 左右平移、R/F 向插口前进/后退、Q/E 左右偏转；↑/↓ 抬头低头、←/→ 自旋。松键保持，空格停止，Enter 恢复，Esc 退出。默认 10 mm/s、5°/s；平移按世界坐标，姿态按 TCP 自身轴，旋转中心为 `pinch`。

键盘与 Servo 独立，详见 [接口、按键和验证说明](docs/07_键盘与笛卡尔Servo.md)。该入口用于无接触手动点动，不是力控插入。

## 接口与算法所有权

- `load_model()` / `reset_home()`：加载场景、恢复已夹持初始状态；`home` 的控制量含静态重力平衡偏置。展示入口保持这些控制量，不重新发送未经补偿的关节位置。
- `MujocoRobotAdapter`：保留命名关节位置和夹爪控制接口。此场景以 `grasp_lock_*` 等式约束固定指关节；夹爪开口命令仍可写入，但不会释放理想抓持硬盘。
- `MujocoWristFTAdapter`：原始与去皮六维数据，坐标系 `wrist_ft_site`；静态去皮不等于任意姿态重力补偿。
- `MujocoCollisionChecker`：把夹持硬盘计入机器人，硬盘与插口接触不会被忽略。
- 通用运动学、规划、核心合同和限力门保留。算法层不依赖 MuJoCo；`insertion_validation` 中的局部姿态求解和位置运动仅供仿真验收，不是生产控制器。

## 文档

- [功能与验收](docs/01_功能规格与验收标准.md)
- [模型参数与假设](docs/02_模型参数与假设.md)
- [架构与接口](docs/03_架构与接口.md)
- [维护标准](docs/04_开发与维护标准.md)
- [后续算法路线](docs/05_算法开发路线.md)
- [本次验收结果](docs/06_场景重建验收报告.md)
