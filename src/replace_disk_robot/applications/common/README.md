# JAKA 应用共用模块

`jaka_keyboard` 和 `jaka_joy` 分别负责输入及应用交互，共用本包的反馈、力处理、导纳和伺服控制链。
本包不导入这两个应用，也不创建键盘或手柄设备。

| 文件 | 职责 |
|---|---|
| `node.py` | `JakaServo`：初始化、停止、恢复、关闭和有序控制周期 |
| `config.py` | 共用参数、YAML/JSON 读取、命令行覆盖及校验；默认配置路径由各应用传入 |
| `wrench.py` | JAKA 传感器到 TCP 的变换、负载重力补偿、去皮和滤波 |
| `reference.py` | 名义位姿、导纳组合、轴屏蔽和柔顺偏移限制 |
| `log.py` | 同周期数据快照、异步写入、事件和源码快照 |
| `analysis.py` | 离线日志分析和绘图，由 `tool/analyze_jaka_control_log.py` 调用 |
| `simulation.py` | `MujocoJakaClient`、`MujocoWrenchPipeline`，提供 MuJoCo 反馈和执行器接口 |

控制周期保持：同一份反馈 → 力处理 → 周期检查 → 应用输入映射 → 名义参考与导纳 → CartesianServo → 力限保护 → 关节命令。
底层控制算法继续位于 `replace_disk_robot.control`，SDK 与 SDL 设备读取继续位于 `replace_disk_robot.adapters`。

应用继承 `JakaServo` 并实现 `_jog_to_servo()`；键盘使用 `_clear_input()` 清除按键。
键盘的 `ReferencePipeline` 在共用参考上添加 `KeyControl` 和按键坐标映射；手柄直接使用共用参考，通过 `JoyControl` 生成速度。
`input_snapshot()` 和 `input_source_paths()` 将各自的输入状态和驱动源文件接入共用日志。

键盘入口和参数路径保持原位置；手柄入口为：

```bash
python /home/a/replace_disk_robot/src/replace_disk_robot/applications/jaka_joy/jaka_joy_node.py
```

运行指南见 [键盘应用](../jaka_keyboard/README.md) 和 [手柄应用](../jaka_joy/README.md)。
两个应用各自加载配置文件，手柄速度仍为用户设置的 `0.2 m/s` 和 `4°/s`。

## 拆分验证

- 手柄 MuJoCo 自动闭环 **53/53** 通过，结果与拆分前完全一致，包括 8 个保持 LB 的左摇杆回中场景。
- 键盘在 base/tool、导纳开启/关闭四种组合下，与拆分前实际模块进行 **3000 周期** MuJoCo 对照；关节目标、反馈和外力数据最大差值为零。
- 真实 Xbox 空闲输入接入无窗口 MuJoCo 运行 2 秒，251 个周期，日志完整，无观察器错误；这项没有人工按键操作。
- 相关测试 **143 项通过**。另有拆分前已存在的 **31 项失败**，失败项目集合完全一致，没有新增失败。
- 键盘和手柄 YAML 内容均保持不变；两个入口、旧键盘入口和日志分析工具在仓库外执行 `--help` 均通过。

报告：`simulation/mujoco/reports/jaka_application_split_validation.json`、`jaka_joy_validation.json`。
上述为仿真和设备读取证据，未连接实机机械臂。
