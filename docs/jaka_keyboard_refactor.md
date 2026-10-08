# JAKA 键盘导纳：职责拆分、同步日志与验收

2026-09-30。本轮保留当前实机配置及控制算法，迁移应用层职责，并新增可选日志。

## 入口和模块

原始脚本已迁移至 `src/replace_disk_robot/applications/jaka_keyboard/jaka_keyboard_servo.py`，保留原控制逻辑，适配迁移后的导入、路径和正向开关接口。
新版独立入口是 `src/replace_disk_robot/applications/jaka_keyboard/jaka_keyboard_node.py`，沿用原命令行参数和 YAML/JSON 配置。
完整重构前源码保存于 `tests/fixtures/jaka_keyboard_servo_before_refactor.py`，用于离线逐周期对照；不要将测试夹具作为新的实机入口。
迁移前源码快照 SHA-256：`8e6e1069dbd39acee79405d1bba54cf1ac822d675644f87de7f06dc895572902`。

新增应用目录 `src/replace_disk_robot/applications/jaka_keyboard/`：

| 文件 | 职责 |
|---|---|
| `config.py` | 参数定义、默认实机配置、命令行覆盖、校验 |
| `wrench.py` | 负载参数、传感器/TCP 变换、补偿、去皮、滤波；消费节点已读取的 EDG 状态 |
| `reference.py` | 按键坐标转换、名义参考、已有导纳的组合、输出偏移钳位 |
| `node.py` | 组装模块、启动/停止/恢复/关闭、单周期控制、唯一命令出口 |
| `frontend.py` | GLFW 按键、焦点、窗口和独立绘图进程 |
| `log.py` | 同周期前后快照、后台日志写入、事件及实验快照 |
| `analysis.py` | 完全离线的日志完整性、独立导纳公式和参考跟踪检查 |

`adapters/jaka/session.py` 成为 EDG 会话及单调时钟调度的实现位置，旧 `jaka_common.py` 保留兼容导出。
节点可显式注入测试运动学、反馈客户端和数据观察者。无窗口入口直接使用节点模块，无需导入 GLFW。

## 保留的控制行为

正常周期沿用：同一份 EDG 反馈 → 力处理 → 原更新前 `sample_callback` → 周期检查 → 键盘/导纳 → CartesianServo → 原力限保护 → SDK 关节命令。

- 控制参数数值及开关默认行为、CLI 覆盖规则、重力补偿及力矩参考点不变。
- 导纳积分、速度限幅、偏移输出钳位、轴屏蔽和伺服增益不变。
- 原停止/恢复/焦点事件逻辑不变；力限触发时原代码的两次保持命令均保留。
- `dry-run` 继续禁止 servo 使能和运动命令。
- 原采集回调仍在更新前运行，重力采集脚本继续支持。
- 日志默认关闭。日志开关不修改控制参数，不进行额外 SDK 反馈读取。

初次拆分未调整实机参数；后续文件迁移和正向开关命名仅改变位置及参数表达。旋转导纳稳定性参数和原控制保护策略保持不变。

## 正向开关

原脚本、新版节点和共用 YAML 统一使用 `gravity_compensation_enable`、`tare_enable`、`plot_wrench_enable`：`true` 开启，`false` 关闭。命令行使用对应的 `--gravity-compensation-enable true/false`、`--tare-enable true/false`、`--plot-wrench-enable true/false`。

当前三个开关默认均为 `true`，保持改名前的默认行为。适配器去皮开关只作用于非重力补偿路径；额外的补偿后去皮仍由 `tare_compensated` 控制。关闭重力补偿时须同时关闭导纳。旧接口已移除；自定义配置需使用新字段，布尔值填写为 YAML/JSON 布尔类型。

## 开启记录

将原来的实机启动命令中的脚本名替换为 `jaka_keyboard_node.py`，保留原参数，并追加：

```bash
--log-dir logs/jaka_trial_001
```

目录必须不存在，以免覆盖实验数据。所有实验均按实际控制周期记录，不按显示频率降采样。
可以先在原命令上同时加 `--dry-run` 与 `--log-dir`，验证数据路径。
`--log-queue-size` 默认为 4096，满队列丢弃新记录并计数，不阻塞控制线程。

每个实验目录产生：

- `metadata.json`：有效参数、坐标系/单位、运行时去皮偏置、模型和负载辨识文件内容/hash、控制相关源码内容/hash、Git 版本及工作区状态。
- `samples.jsonl`：每个控制周期一条结构化记录。
- `events.jsonl`：输入按键、焦点、参考重置、停止恢复、命令接受/失败、关闭事件。
- `summary.json`：接受/写入/丢弃/未写记录数量、写入采样数、观察者和写线程错误与完整性状态。

JSONL 可直接离线读取和转表，不是原生 rosbag 或 MCAP。序号相同的周期数据可以与事件关联；事件与采样分别保留单调时钟时间。
原始 SDK 字段中的非有限数保存为 `null`，同时保留对应故障；不生成不标准的 JSON NaN。

### 周期字段

| 字段 | 含义 |
|---|---|
| `sequence`, `elapsed_s`, `dt_s` | 周期编号、循环相对时间和真正用于积分的 dt |
| `*_monotonic_ns` | 主机读反馈/发命令/周期边界的单调时间 |
| `measured_q_rad`, `measured_qd_rad_s`, `measured_joint_torque` | 同一 EDG 反馈中的关节状态；关节力矩按 SDK 原始单位标记 |
| `sdk_cartesian_pose_raw` | SDK 原始位姿，平移 mm、姿态 rad；不能直接当作 URDF tool0 位姿 |
| `raw_sensor_wrench`, `wrench_stages` | 原始力、去偏置/补偿后传感器力、TCP 未滤波力、滤波后死区前状态和最终处理力 |
| `pressed_before/after`, `jog` | 按键状态及已转换坐标系的运动输入；角速度遵循原 TCP 内禀约定 |
| `admittance_before` | 积分前偏移/速度、轴掩码和名义位姿 |
| `admittance_input_wrench` | 轴屏蔽后真正输入积分器的六维力 |
| `admittance_integrated` | 积分后、保护重置前的状态 |
| `admittance_after` | 整个周期结束后的状态，可能已因故障重置 |
| `unclamped_pose`, `corrected_pose`, `measured_tcp_pose` | 输出钳位前/后的参考及实测关节 FK；都注明坐标系 |
| `servo_target_q_rad`, `final_command_q_rad` | 保护前和保护后的关节目标 |
| `send_attempts`, `command_sent` | 所有 SDK 发送尝试、时间、结果；接受不代表实机已执行 |
| `velocity_clipped_axes`, `*_offset_clamped` | 速度限幅/输出偏移钳位 |
| `fault`, `error`, `observer_errors` | 控制故障、抛出的错误及观察者异常 |

当前 EDG 封装没有设备采样时间和包序号。日志的时间是主机控制时间，不能据此声称测出了传感器到主机的完整延迟或证明数据包新鲜。

日志数组与控制状态分离，序列化及文件写入在后台进行。记录失败不改变运动命令；错误及丢样使记录验收不完整。停止退出时先按原逻辑关闭 servo，再等待写线程退出。进程被强制终止或磁盘故障可能留下部分日志，没有结束摘要时分析器按不完整处理。
自定义 `tick_callback` 在控制线程执行，只适合快速观察或入队；默认记录器不在该回调中同步写盘。

## 离线分析

```bash
python tool/analyze_jaka_control_log.py logs/jaka_trial_001 --plots
```

只读取日志，不加载 SDK、不连接或指挥机器人。生成 `analysis/report.md`（中文表格报告）、`report.json`、`cycle_metrics.csv`，以及 `force_control_samples.csv` 和 `admittance_control_samples.csv` 两份逐周期数据表。

报告分为两部分：

1. **力补偿**：在传感器坐标系下统计补偿后六维力/力矩；有原始数据时增加原始/补偿对比与配对差值，缺失值不补零。缺少传感器补偿阶段时可降级展示处理后 TCP 信号，明确来源与坐标系，不跨坐标系求差。
2. **导纳控制**：2.1 显示 `tool0` 下补偿外力（滤波前）、处理外力（滤波/死区后）、屏蔽后实际输入、积分偏移和速度；2.2 在 `jaka_base_link` 下显示限幅后柔顺目标与实测相对当周期名义位姿的位移，并显示相对首个有效实测位姿的总运动及同周期跟踪误差。

`--plots` 生成 `force_control_overview.png`、`admittance_control_overview.png` 和 `admittance_tracking_overview.png`。`--plot-force-control` 单独生成第一张，`--plot-admittance-control` 生成后两张。

积分偏移在当前实测 TCP 轴中，机械臂位移在基座轴中；不能直接按同名 x/y/z 比较。实测 TCP 由实测关节正运动学得到。总位移包含键盘运动，实测相对名义位姿包含跟踪滞后；旋转采用基座轴旋转向量，非欧拉角差。反馈在本周期命令发送前采集，因此同周期跟踪误差不代表该命令执行后的响应。故障与未发送命令周期在图表中保留明确标记。完整输出定义见 [jaka_keyboard README](../src/replace_disk_robot/applications/jaka_keyboard/README.md#日志分析)。

分析直接使用记录中的有效 M/D/K、真实 dt、积分前状态和屏蔽后的输入，独立计算：

\[
v_{k+1}=\operatorname{clip}\left(v_k+\Delta t_k\frac{F_k-Dv_k-Kx_k}{M},-v_{\max},v_{\max}\right),\qquad
x_{k+1}=x_k+\Delta t_kv_{k+1}.
\]

它对比积分后的状态，而非保护重置后的状态；不会对含有重置/限幅的整段日志套用未经修正的连续系统响应。
还检查屏蔽输入一致性、序号缺口、时间顺序、写盘完整性、周期抖动、限幅次数及参考/实测 FK 的位置和姿态误差。
图上原始传感器力与处理后 TCP 力分开显示，以免把不同坐标轴直接比较。

`discrete_equation_matches=true` 仅证明离散算式与记录相符。参考跟踪误差不是机器人内环模型辨识；整段回放也不能证明接触稳定。
换参数后沿用原来的实测力，只是相同输入下的导纳预测，不是新参数下的实机接触预测。

## 离线验收

原实现已由使用者在实机上验证；本轮所有测试使用模拟客户端/运动学，未连接实机。

- 基线对照包含 base/tool、导纳开/关、dry-run、当前实机 YAML、带/不带周期观察者，以及超力、跟踪误差、超时、无效数据、停止、焦点与恢复。
- 对照逐周期使用精确数组相等，检查处理力、偏移/速度、名义/修正位姿、最终命令、servo 使能调用与故障；也检查更新前回调的时机。
- 日志验收包含正常/只读记录、独立公式校验、故意篡改记录、队列溢出、异常观察者、SDK 发送失败、写盘失败、既有目录保护与不完整日志。
- `docs/jaka_keyboard_refactor_acceptance.json` 保存一次 1000 周期模拟对照的耗时和日志验收结果。耗时仅描述本机模拟负载，不证明实机 125 Hz 截止时间。

初次重构验收中，相关测试 71 项全部通过，其中原版本逐周期对照及入口检查 30 项。开启后台记录的 1000 周期模拟中，原/新版命令逐项一致，采样全部写入，丢样为 0，离散公式最大残差为 0。迁移的 5 个会话/调度定义与迁移前源码的 AST 完全一致。语法检查与差异格式检查通过。

拆分前有 5 个旧导纳测试失败：测试默认假设与当前实机 YAML 的开关/轴/速度不一致。测试现在显式使用空配置来固定原 CLI 默认参数；另有专门的原版本对照检查实机 YAML，实机配置没有被改动。
完整测试收集还发现两个既有规划测试引用不存在的 `replace_disk_robot.planning.optimize`。HEAD 已是同样的引用，而实现位于 `planning/optimization.py`；本轮不修改规划模块。
初次重构验收排除这两个收集错误后，其余仓库测试 201 项全部通过（176.29 秒）。当时的相关 71 项包含在这 201 项中，不额外累计。

入口与 YAML 迁入功能包时，相关测试 73 项通过；两种入口均可在仓库外目录执行 `--help`，共用同目录默认配置。当时原脚本控制定义的 AST 与原版本一致，YAML 内容逐字不变；重力采集工具导入检查通过，离线构建的 wheel 包含两个入口和 YAML。此次迁移未进行实机运行。

正向开关改名后，相关测试 93 项通过（5.09 秒），包含原版本逐周期行为对照、开关默认值、YAML 关闭、双向命令行覆盖、无效布尔值拒绝、去皮调用开关和日志参数快照。对照夹具保留历史源码，通过测试适配器翻译开关语义；运行代码只接受正向字段。此次改名也未连接实机。

提交前最终回归：排除上述两个既有规划测试收集错误后，其余 223 项通过（175.98 秒），包含 JAKA 运动学与 MuJoCo 帧对照。暂存差异格式检查通过，共享会话模块的类型注解可正常解析；未运行实机。

当前使用指南见 [jaka_keyboard README](../src/replace_disk_robot/applications/jaka_keyboard/README.md)。
