# JAKA 力方向与候选重力补偿：NUC 验证交接

本次验证回答两个问题：已知方向的手施力，是否被补偿和坐标变换正确识别；导纳是否正确使用这份外力计算柔顺输出。第一轮使用 `--dry-run`，机械臂应保持静止，不能据此声称候选参数已通过实际运动验证。

本地已完成三份日志的关节 FK、雅可比与离散导纳回放排查，详情见 [排查报告](../logs/force_direction_audit/report.md)。主要问题是 `tool0` 绕 Z 改向 180° 后，旧标定 JSON 的传感器外参覆盖了已修正的代码默认值。该改向会翻转工具 X/Y，保留 Z；现有日志主要激励 X/Y，尚不能确认 Z 或整个六轴符号。

## 1. 本次交付与验证范围

- 日志分析器：力补偿表、导纳外力/输出表、基座系实际运动表，及三个分开的图。
- 重力辨识：默认重力倾角上限 4°；同时保留自由拟合诊断。4° 来自底盘与安装误差的估计上限，不是末端水平仪直接测得的基座倾角。
- 候选文件：`logs/force_direction_audit/candidate_tool0_repaired_bounded4.json`。拟合质量约 0.672 kg，拟合倾角触及 4° 上界；修复后自由拟合仍约 13.8°，因此需要独立验证。
- 当前默认 YAML 与正式标定文件未替换。候选文件通过命令行显式加载。

NUC Codex 负责环境检查、启动下述 dry-run、检查记录和离线分析；人负责摆位、确认物理方向、轻推工具和记录时间。不要自行启动上电/使能脚本、关闭 dry-run、启用拖动模式、提高力限，或整体取反六轴信号。

`dry-run` 不使能 Servo、不发送运动目标，但会登录控制器、配置力传感器模式并启用 EDG 通信，不是完全不改变控制器状态。当前 dry-run 分支不经过运动分支的力限门控，不能把配置中的力限视作本轮施力保护。保持小力、避免强推，出现异常停止测试。确认没有其他程序或控制器任务在发送运动命令。

## 2. 在 NUC 准备环境

从仓库根目录执行。先获取本次提交，保留 NUC 本地改动，不用强制重置：

```bash
git status --short
git pull --ff-only origin main
git rev-parse HEAD
sha256sum logs/force_direction_audit/candidate_tool0_repaired_bounded4.json
python3 -c 'import sys, numpy, pinocchio; print(sys.executable); print(numpy.__version__); print(pinocchio.__version__)'
```

记录提交号、候选文件校验和、解释器路径、NumPy/Pinocchio/JAKA SDK 版本以及控制器模式。使用 NUC 已验证能读取 EDG 的 SDK 环境；本地 ROS Humble 的 Pinocchio 与 NumPy 2 不兼容，本地交叉核查使用系统 Python/NumPy 1 环境，NUC 不应盲目照搬解释器路径或安装新依赖。

核查当前基座/工具/传感器安装和标定时相同。人确认机器人已处于允许静止读数的状态，末端无接触、无明显拖链拉力；Codex 不自行改变其上电/使能状态。传感器模式应提供未重力补偿的原始值，避免重复补偿；仅 CLI 的 `torque_sensor_mode=1` 不能代替对 SDK/控制器实际数据语义的确认。

默认控制器 IP 为 `10.5.5.100`，本机 EDG IP 为 `10.5.5.127`；若 NUC 地址不同，在下述命令显式添加正确的 `--robot-ip`、`--local-ip`。

## 3. 采集不发送运动的方向日志

使用无窗口模式，避免旧日志中的 `focus_lost`。日志目录必须不存在；重复测试换成 `..._002`，不能覆盖记录。关闭额外去皮，保留绝对补偿误差；本轮只验证平移导纳，原始力矩仍保留在日志里。

```bash
python3 src/replace_disk_robot/applications/jaka_keyboard/jaka_keyboard_node.py \
  --dry-run --headless --seconds 120 --plot-wrench-enable false \
  --gravity-compensation-enable true \
  --gravity-json logs/force_direction_audit/candidate_tool0_repaired_bounded4.json \
  --tare-enable false --admittance --admittance-axes translation \
  --adm-disable-axes \
  --log-dir logs/jaka_direction_dryrun_001
```

检查有效参数中 `tare_compensated=false`、`identity_transform=false`；若 NUC 修改过默认 YAML，先生成专用测试配置，通过 `--config` 传入并保留副本。启动提示应包含 `dry-run: servo not enabled, no motion commands sent`。如果导纳因 `loop_timeout` 等故障停止更新，结束本轮，查明原因后换目录重启；不要把故障后的旧偏移当成新输出，dry-run 中 Enter 也不会恢复。

### 人的施力与时间标记

按 **机器人模型 `jaka_base_link` 的轴**标记方向，并记录基座轴在现场的指向。基座有 3–4° 倾斜时，地面竖直与基座 Z 近似但不完全相同。用模型/FK 的工具轴展示确认方向，不能只凭工具外壳判断 `tool0` 的 X/Y。手接触力传感器加载侧的工具，不推机械臂本体，不接触其他物体。

|阶段|建议时间，按日志 `elapsed_s`|操作|
|---|---|---|
|初始基线|0–10 s|完全松手、静止|
|第一组|10–70 s|依次 +X、-X、+Y、-Y、+Z、-Z；每方向轻推约 5 s，再松手约 5 s|
|补测|70–110 s|重复 Z 两个方向及可疑方向，记录实际时间|
|结束基线|110–120 s|完全松手、静止|

通常 2–4 N 的轻推即可辨认方向，但以现场可控的小力为准，不为达到数字强推。原配置有约 1 N 分量死区，信号低于死区时导纳输入可能为零；此时仍可检查滤波前外力，导纳结论标记为数据不足。松手后若状态未回稳，应延长间隔并换更长 `--seconds`。上述时间表只是参考，必须写下实际区间。

在日志目录新增 `test_notes.md`，写明接触位置、初始姿态、安装方向、实际每段开始/结束时间、是否混入其他方向、松手基线和异常。方向定义为“人施加给工具的力”，不能把工具对手的反力当成输入方向。建议再换一个明显不同的静止末端姿态重复一次；摆位由人完成，测试程序不发送摆位指令。

## 4. 先检查日志，再判断方向

保存 `metadata.json`、`samples.jsonl`、`events.jsonl`、`summary.json` 与人的记录。核查：

1. `metadata.effective_args.dry_run=true`，加载的是候选 JSON；运行时 `sensor_to_tool_rotation` 与候选 JSON 顶层 `sensor_to_tool_rotation` 相同，`tool_to_sensor_m` 约为 `[0,+0.05,-0.4]` m。核对模型、外参、候选校验和，不只看文件名。
2. 所有周期 `dry_run=true`、`command_sent=false`、`send_attempts` 为空；事件中没有 `command_accepted` 或 `command_rejected`。终端打印的 `cmd=` 是显示值，不能当成已发送命令。
3. 有足够 `admittance_updated=true` 周期；施力区间没有故障、观察器错误、缺失原始/补偿信号、日志丢行或明显采样停顿。`summary` 的丢弃/写入错误与分析报告的完整性检查均需查看。
4. 机械臂关节/FK 位移应接近静止，列出实际峰值漂移；若明显移动，先排查其他控制来源或物理松动。本轮不能用静止速度计算“顺应力运动”的通过率。

离线生成表格和三个图，不连接机器人：

```bash
python3 tool/analyze_jaka_control_log.py logs/jaka_direction_dryrun_001 --plots
```

## 5. 坐标系与判定方法

记 B=`jaka_base_link`、T=`tool0`、S=`tcp_fts_sensor`，`R_B_T` 是由实测关节 FK 得到的、把工具分量旋转到基座的矩阵。日志四元数顺序为 `wxyz`，不要按 `xyzw` 解读。

|信号|日志位置|坐标系/用途|
|---|---|---|
|原始力/力矩|`raw_sensor_wrench`|S；与补偿后信号同系对比|
|补偿后力/力矩|`wrench_stages.compensated_sensor`|S；保留无外力基线误差|
|TCP 补偿外力|`wrench_stages.external_tcp_unfiltered`|T；滤波前，优先检查物理方向|
|处理后力/实际导纳输入|`wrench_stages.processed` / `admittance_input_wrench`|T；滤波、死区、轴屏蔽会改变信号|
|导纳偏移/速度|`admittance_integrated.offset` / `.velocity`|T；积分后、位姿钳位前|
|目标/实测 TCP|`corrected_pose` / `measured_tcp_pose`|B；实测来自关节 FK|

### 5.1 补偿外力的方向

逐周期计算 `F_B = R_B_T @ external_tcp_unfiltered[:3]`，然后与人的基座方向标记比较。例如 +X 的方向向量为 `[1,0,0]`，-X 为 `[-1,0,0]`，其余同理；工具轴测试则先用 `R_B_T` 转换人的方向标记。不能直接拿 T 中 Fx 与 B 中位移 x 比较。

对每段用附近松手窗口的基座力均值作基线，计算增量 `delta_F_B`，单独报告未减基线的补偿均值/RMS及前后漂移，避免隐藏重力误差。剔除施力/松手过渡，只有信号超过基线噪声且方向稳定的区间可判定。推荐诊断标准：投影超过基线投影标准差的 3 倍，点积为正的有效样本比例 ≥95%，中值夹角余弦 ≥0.8。它们是本轮诊断阈值，不是硬件安全认证；施力不纯、信号太弱或姿态移动时标记 `INCONCLUSIVE`，补测。

输出每个 ±X/±Y/±Z 的样本数、基线、力增量均值、投影、正向比例、夹角余弦和 `PASS/FAIL/INCONCLUSIVE`。没有独立测力计时只能验证方向与相对变化，不能声称绝对力值已标定准确。至少在两个不同静止姿态检查无外力残余，明显随姿态变化时继续查重力补偿。

### 5.2 导纳输入与输出

先验证处理外力经轴屏蔽后等于 `admittance_input_wrench`，再按有效 M/D/K、真实 `dt_s` 与积分前状态独立重算：

```text
v1 = clip(v0 + dt * (F - D*v0 - K*x0) / M, -vmax, +vmax)
x1 = x0 + dt*v1
```

与 `admittance_integrated` 比较，正常残差应接近浮点误差，可用最大绝对残差 ≤1e-9 作离线一致性标准，同时报告单位、最大值、有效样本数。用 `R_B_T @ offset[:3]` 和 `R_B_T @ velocity[:3]` 展示基座方向下的预测输出，区分积分输出与位姿限幅后的目标。

导纳有惯性、阻尼和回正弹簧，换向或松手期间，当前速度/累计位移不一定与当前力同号。判断离散动力学是否正确，应检查上述净驱动力及积分残差；不要把每一帧 `F·v>0` 作为硬性标准。必要时重新启动单方向短测，使起始偏移与速度接近零，检查受力后的最初增量。

## 6. 历史日志离线复现（可先做，无 SDK）

以下脚本对已提交的三份历史日志，使用各自保存的 URDF 快照独立重算关节 FK/雅可比并对照 Pinocchio；会重新生成审计 CSV/NPZ/JSON。需要 NumPy 与 Pinocchio 可同时导入的环境，本地命令如下，NUC 可改为自己的兼容解释器：

```bash
PYTHONPATH=src:/opt/ros/humble/lib/python3.10/site-packages \
  /usr/bin/python3 -s logs/force_direction_audit/reproduce_audit.py
```

模型互相一致只能证明软件计算一致，不能证明物理传感器方向、TCP 或安装姿态正确。脚本对旧补偿残余力的重表达，也不能代替新候选标定下的实机施力记录。大体积审计 CSV/NPZ 和每日志 `analysis/` 可重新生成，本次交付保留报告、关键图、汇总、候选和复现脚本。

## 7. NUC 返回结果与后续边界

请将环境/提交记录、测试配置、原始日志、人的方向区间、分析报告/CSV/三个图与六方向结果表一并交回；明确区分“历史回放通过”“新施力方向通过”“重力补偿独立验证”“实际运动未测试”。只给截图无法完成方向核查。

若 X/Y 或 Z 单独反向，优先检查物理安装、运行时外参和基座转换；若六方向全部反向，再核查传感器给的是外力还是反力、SDK 原始模式及符号约定。不要直接改六轴负号或继续压小重力倾角。若方向通过但无外力基线随姿态明显变化，应重新采集被接受的静态窗口，并用未参与拟合的姿态验证；新采集器输出不使用历史 `--repair-stale-tool0-extrinsics` 修复开关。

本轮交接止于不发送运动的验证。六个平移方向通过也不代表扭矩通过，或候选标定可立即用于闭环。实际运动对比需另行安排小幅测试，再用关节 FK 速度、基座外力和转换到基座的导纳速度对比，并考虑响应滞后；同周期反馈是在本周期命令发送之前读取的。
