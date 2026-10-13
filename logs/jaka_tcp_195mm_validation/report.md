# 硬盘插入控制点统一为 195 mm

2026-10-13。按用户明确要求实施；全部验证离线完成，未连接或运动实机。

当前约定：传感器表面作为模型传感器原点，硬盘同轴安装，插入端在工具 +Z 前方 195 mm，横向偏移为零。用户测量是表面到插入端的距离，这一约定尚未加入传感器内部测量中心相对表面的偏移。

## 已同步的定义

| 位置 | 修改 |
|---|---|
| URDF `tool0_joint` | `xyz: [0, 0, 0.27] → [0, 0, 0.195] m` |
| MuJoCo XML `tool0` | `pos: [0, 0, 0.27] → [0, 0, 0.195] m` |
| 默认 EDG F/T 力臂 | `[0, -0.05, -0.4] → [0, 0, -0.195] m` |
| 当前在线重力标定 JSON | `tool_to_sensor_m: [0, +0.05, -0.4] → [0, 0, -0.195] m` |
| 标定中的工具坐标质心 | 按新力臂重新表达 `center_of_mass_tool_m/mm` |

仅调整 TCP 平移。工具轴方向、传感器旋转外参、质量、传感器坐标质心、重力向量和传感器零偏保留；标定 JSON 比较确认仅上述三个 TCP 相关字段发生变化。约 20° 旋转差异仍待方向验证，本次不混入旋转外参重辨识。

旧标定备份：[gravity_calibration_before_tcp_change.json](gravity_calibration_before_tcp_change.json)。历史采集数据、日志及候选标定报告未改写。旧打磨网格及碰撞外形未重建为硬盘外形，本次调整的是控制/力矩换算点。

## 验证

- 相关测试 **76 项通过**，详见 [related_tests.txt](related_tests.txt)。
- URDF/Pinocchio 与 XML/MuJoCo 多姿态 FK 和坐标方向一致，雅可比与数值差分一致。
- 三个不同姿态下，使用在线力臂从 TCP 反推的传感器原点与模型传感器原点一致。
- 使用生产实机力处理管线、在线标定和独立运动学原点，分别模拟在插入端施加三个轴向的 5 N 力，换算到插入 TCP 后附加力矩均为零，误差在 1e-12 内。
- MuJoCo 手柄、力处理、导纳、Servo、力保护全链路 **53 项全部通过**，见 [joy_mujoco_report.json](joy_mujoco_report.json)。

当前在线 JSON 同时供手柄/键盘节点读取；模型由运动学加载。退出并重新启动节点后生效，无需重装 editable 包。

```bash
env -u LD_LIBRARY_PATH PYTHONPATH=src .conda_envs/replace_disk_robot/bin/python -m pytest -q tests/test_jaka_kinematics.py tests/test_jaka_keyboard_admittance.py tests/test_jaka_driver_tool.py tests/test_force_processing.py tests/test_ft_gravity_collector.py tests/test_jaka_control_logging.py tests/test_jaka_joy.py
env -u LD_LIBRARY_PATH PYTHONPATH=src .conda_envs/replace_disk_robot/bin/python src/replace_disk_robot/applications/jaka_joy/jaka_joy_node.py --backend sim --validate --output logs/jaka_tcp_195mm_validation/joy_mujoco_report.json
```
