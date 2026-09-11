# ROS 2 adapter boundary

本目录将来只负责：机械臂/夹爪/F/T/相机驱动映射、TF、JointState、参数、时间戳、watchdog 和命令输出门。

不得在这里实现 IK、轨迹优化、导纳、Residual RL 或任务状态机。真机适配器必须实现 `replace_disk_robot.core.ports`，并保持与 MuJoCo 相同的单位和坐标系语义。
