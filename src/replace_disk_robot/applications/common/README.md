# JAKA 共用控制

供 `jaka_keyboard` 和 `jaka_joy` 复用，不导入具体输入应用或创建输入设备。

| 文件 | 职责 |
| --- | --- |
| `node.py` | 反馈、控制周期、停止/恢复和关闭 |
| `wrench.py` | 坐标变换、重力补偿、去皮与滤波 |
| `reference.py` | 名义位姿、导纳、轴屏蔽与偏移限制 |
| `config.py` | 参数加载与校验；配置路径由各应用提供 |
| `log.py` / `analysis.py` | 周期记录、源码快照、离线分析 |
| `simulation.py` | JAKA MuJoCo 反馈和执行接口 |

底层算法位于 `control`，设备读写位于 `adapters`；运行方式和坐标约定见
[JAKA 使用说明](../../../../docs/02_JAKA使用说明.md)。
