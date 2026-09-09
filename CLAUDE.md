# 项目维护说明

固定 UR5e + 2F-85 + 腕部六维力传感器的硬盘精密插入仿真。先读 `docs/03_架构与接口.md`、`docs/04_开发与维护标准.md`。所有正式说明在中文 `docs/` 中。

当前场景只有 177×106×26 mm 已夹持硬盘、106.5×26.5 mm 净口的 40 mm 直通通道及机械臂。服务器与换盘示例已删除。每侧间隙 0.25 mm；init 在入口外 10 mm；未来目标插深 50 mm。

核心类型、命名关节和 F/T 接口保留；`MujocoMechanismAdapter` 已删除。`home` 控制量包含重力平衡偏置，展示入口不要覆盖成裸关节位置或自动张爪。指关节理想锁定，硬盘固定在传感器下游，不能以张爪命令假装释放。

算法层不得依赖 MuJoCo/ROS/厂商 SDK。新增接触动作仅用于场景验收，当前无闭环力控插入。仿真数据不证明真实夹持或硬件安全。

```bash
conda activate hard_disk_robot
python examples/check_isolated_environment.py
python examples/smoke_test.py
python -m pytest -q
python examples/validate_insertion_contact.py --output simulation/mujoco/reports/contact_validation.json
MUJOCO_GL=egl python examples/capture_scene.py
git diff --check
```
