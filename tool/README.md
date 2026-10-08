# 六维力传感器末端负载重力辨识：使用方法

本目录三个脚本的实机使用流程：

- `collect_ft_gravity_data.py`：实机键盘 Servo + 5 Hz 自动采集静态姿态数据（**不需要按采集键**）；
- `identify_ft_payload.py`：离线辨识负载质量、质心、传感器零偏和受约束的重力方向；
- `plot_ft_gravity_data.py`：离线补偿并绘制补偿前后的六轴曲线。

以下命令假设仓库位于 `~/replace_disk_robot`。控制器 IP 不是默认值（`10.5.5.100` /
本机 `10.5.5.127`）时，所有脚本都加上 `--robot-ip <控制器IP> --local-ip <本机IP>`。
操作前把急停放在手边，并清空机械臂周围。

## 0. 前置条件

- 采集脚本用键盘窗口操作，必须在**有显示器**的机器上运行（或 `ssh -X` 转发）；
  没有显示会报 `cannot initialize GLFW display`。
- 采集脚本默认**不覆盖**已有 CSV；重跑时换 `--output`，或显式加 `--overwrite`。

## 1. 上电并使能 JAKA

采集脚本只负责使能 Servo，不做上电和使能，这一步必须先做：

```bash
cd ~/replace_disk_robot
python3 examples/jaka_driver_tool/jaka_start.py
```

## 2. 只读自检（不动机械臂）

```bash
# 2a. 先确认六维数据本身正常：只读、不使能 Servo、不发运动
python3 examples/jaka_driver_tool/jaka_ft_test.py --seconds 10

# 2b. 再确认采集脚本、窗口和自动采集逻辑正常
python3 tool/collect_ft_gravity_data.py --dry-run --seconds 25 \
  --output /tmp/ft_dry_run.csv --overwrite
```

2b 应在启动约 2 s 后出现

```
[capture] #0: still for 0.80s; recording 51 rows at 5 Hz
```

再过 10 s 出现

```
[capture] saved #0: 51 rows; total=1. Move the arm to a new orientation for the next pose.
```

看到这两行说明 5 Hz 自动采集链路已经通了。

## 3. 实机采集

```bash
python3 tool/collect_ft_gravity_data.py --output tool/ft_gravity_samples.csv
```

操作节奏：

1. 用键盘把末端移到一个新的、无碰撞的姿态：`W/S/A/D` 平移，`R/F` 沿当前 tool0 ±Z
   进退，`Q/E` 和方向键旋转。姿态不好摆时把速度调小，例如加 `--angular-speed-deg 2`。
2. **松开运动键并保持静止**：静止 `--settle-seconds`（默认 0.75 s）后自动开始记录，
   记录满 `--capture-seconds`（默认 10 s，5 Hz 下 51 行）后打印
   `[capture] saved #N: 51 rows; total=N.`。
3. 看到 `saved` 之后再移动机械臂，重复下一个姿态。建议采集 **12 个以上**姿态，并让工具
   坐标系的 X/Y/Z 轴相对重力方向都有明显变化（这决定底盘倾斜方向能否辨识准确）。
4. 若打印 `[capture] #N not saved: arm moved (...)`，说明窗口中间动了，属正常情况：
   停下来 0.75 s 后会自动重采，不需要任何按键。
5. 采集期间不要接触工件、不要让拖链明显拉扯、不要让人碰到机械臂。
6. 按键：`Space` 停止并保持，`Enter` 从当前实测关节位置重新锚定并恢复，`Esc` 退出
   （退出时自动关闭 Servo、关闭 EDG、logout，并打印本次采集统计）。

时间预算：每个姿态约 11 s 静止加摆位时间，12 个姿态约 3–5 分钟。

### 采集脚本常用参数

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--capture-rate-hz` | 5 | CSV 采样率；只有记录被抽稀，伺服环仍是 `--rate-hz`（125 Hz） |
| `--capture-seconds` | 10 | 每个静止窗口长度，5 Hz 下 51 行 |
| `--settle-seconds` | 0.75 | 静止多久后开始记录 |
| `--start-delay-seconds` | 2 | 启动后多久才允许开始第一个窗口 |
| `--output` / `--overwrite` | `tool/ft_gravity_samples.csv` | 输出路径；已存在时报错，需显式覆盖 |
| `--dry-run` | 关 | 只读采集，不使能 Servo、不发运动指令 |
| `--torque-sensor-mode` | 1 | 必须输出**未做重力补偿**的原始六维值 |

## 4. 采集后自检

```bash
python3 - <<'PY'
import csv, collections
rows = list(csv.DictReader(open("tool/ft_gravity_samples.csv")))
counts = collections.Counter(int(r["capture_id"]) for r in rows)
ok = {k: v for k, v in sorted(counts.items()) if k >= 0}
print("总行数:", len(rows), "| 有效姿态:", len(ok), "| 每窗行数:", sorted(set(ok.values())))
print("capture_id:", list(ok), "| 非窗口行:", counts.get(-1, 0))
PY
```

期望每窗 **51 行**，有效姿态数等于实际采集的姿态数。
默认辨识模式（全行）不依赖 `capture_id`；只有加 `--static-poses` 时才要求有效姿态数，且少于 6 个姿态无法辨识。

## 5. 离线辨识

```bash
python3 tool/identify_ft_payload.py tool/ft_gravity_samples.csv
```

默认行为是**忽略采集器的静态窗口标记，直接把 CSV 每一行当作一个准静态样本**。
只要运动速度足够慢，运动过程中的数据也会参与重力/质心/零偏拟合，不再要求
`capture_id >= 0`，也不要求每个姿态凑够 50 行。

如果仍然想使用采集器的 `capture_id` 窗口分割，并取每个窗口的中值作为一个姿态：

```bash
python3 tool/identify_ft_payload.py tool/ft_gravity_samples.csv --static-poses
```

结果默认写入 `tool/ft_gravity_samples_identified.json`（与 CSV 同名前缀），终端同时打印
`pose_source`、质量、base 系重力向量、重力倾角/方位角、质心和残差。
`--min-samples` 在默认模式下表示 CSV 总行数下限；在 `--static-poses` 模式下表示每个
窗口的最小行数（默认 50）。

- 默认使用 `--gravity-mode bounded`，重力物理向下方向限制在基座 -Z 周围 **4°** 内，同时重新拟合带符号的重力载荷、零偏和质心；这允许约 2° 的底盘倾斜和约 1.5° 的安装倾斜合计范围。约束倾角不是独立测量结果。`--gravity-tilt-limit-deg 2` 使用更紧的 2° 上限，`--gravity-tilt-limit-deg 0` 固定为基座竖直；
- `--gravity-mode free` 保留旧自由三维拟合，仅用于已确认的大倾角安装或诊断。输出始终保留 `unconstrained_gravity_tilt_deg` 和自由拟合残差；约束生效时仍会提示自由拟合的异常倾角，不能靠压小数字掩盖坐标错误；
- `mass_kg = |h| / g`，只取决于重力向量的模，与倾斜无关；
- `fit.force_rms_n`、`fit.torque_rms_nm` 越小越好；`fit.force_matrix_condition` 应远小于 100，
  否则说明姿态激励不够，需要补采更多不同朝向；
- 报 `identified gravity load is nearly zero` 时，说明控制器已对力做了重力补偿，必须换成
  未补偿的 `--torque-sensor-mode` 重新采集；
- 默认 10 s 窗口 = 51 行，对应辨识脚本默认的 `--min-samples 50`；若改小
  `--capture-seconds`，请把 `--min-samples` 调到 `N*5+1` 以内，否则该姿态会被跳过。

### 历史 tool0 坐标变换修复

2026-09-30 的三份 JAKA 日志及 `ft_gravity_samples.csv` 存在特定历史问题：URDF 的 tool0 已绕 Z 旋转 180°，采集/标定却仍使用旧 tool 轴中的传感器外参。修复需要同时重建每个样本的传感器朝向、修正输出旋转/力臂并重新拟合，不应只翻转控制力或单独修改重力向量。

```bash
python3 tool/identify_ft_payload.py tool/ft_gravity_samples.csv \
  --repair-stale-tool0-extrinsics --gravity-tilt-limit-deg 4 \
  --output logs/force_direction_audit/candidate_tool0_repaired_bounded4.json
```

输出已存在时请换名字，或在明确需要重新生成候选文件时加 `--overwrite`。新采集器已使用修正后的默认外参，正常的新 CSV **不加** `--repair-stale-tool0-extrinsics`；tool0 修改前的旧 CSV 也不能套用这个选项。只适用于“新 tool0 姿态 + 旧传感器外参”混用的数据。

在线 `gravity_json` 中的旋转与力臂会覆盖代码默认值，因此只改默认函数不能修复仍加载旧标定的运行。新候选标定应先通过只读的已知方向施力验证，再用于运动。不要直接覆盖当前标定。详细证据见 [三份日志排查报告](../logs/force_direction_audit/report.md)。

交给 NUC Codex 的六方向施力步骤、日志检查、坐标换算与判定标准见 [NUC 验证交接](../docs/jaka_force_direction_nuc_validation.md)。第一轮用 `--dry-run --headless`，不发送运动目标。

## 6. 绘图与独立交叉验证

```bash
python3 tool/plot_ft_gravity_data.py tool/ft_gravity_samples.csv --show
# 历史坐标混用数据的修复候选，绘图会按 JSON 修复标记同步重建传感器朝向：
python3 tool/plot_ft_gravity_data.py tool/ft_gravity_samples.csv \
  --identified logs/force_direction_audit/candidate_tool0_repaired_bounded4.json \
  --output logs/force_direction_audit/candidate_gravity_compensation.png
# → tool/ft_gravity_samples_gravity_comparison.png
```

用**没有参与辨识**的另一组数据做交叉验证：

```bash
python3 tool/collect_ft_gravity_data.py --output tool/ft_gravity_check.csv
python3 tool/plot_ft_gravity_data.py tool/ft_gravity_check.csv \
  --identified tool/ft_gravity_samples_identified.json
```

`--capture-id N` 只画某一个姿态；如果 CSV 全是 `capture_id=-1`，绘图脚本会自动改用全部数据，也可以显式加 `--include-noncapture`。

## 7. 把辨识结果用于在线补偿

`src/replace_disk_robot/applications/jaka_keyboard/jaka_keyboard_servo.py` 默认读取
`tool/ft_gravity_samples_identified.json`，在力限保护之前完成上述传感器零偏、
负载重力和质心力矩补偿，并默认打开补偿后的六维力曲线。下面的代码演示如何在
其他算法中手动接入同一套 `SensorWrenchCompensator`。

```python
import json
import numpy as np
from replace_disk_robot.contact.force_processing import (
    SensorCompensationConfig,
    SensorWrenchCompensator,
)

fit = json.load(open("tool/ft_gravity_samples_identified.json"))
config = SensorCompensationConfig(
    frame_id="tcp_fts_site",
    gravity_frame_id="jaka_base_link",                     # 传感器位姿所在的坐标系
    gravity_m_s2=np.asarray(fit["signed_gravity_force_base_n"]) / fit["mass_kg"],
    load_sign=1.0,                                         # 直接用带符号的重力向量
    payload_mass_kg=fit["mass_kg"],
    payload_com_sensor_m=fit["center_of_mass_sensor_m"],
)
compensator = SensorWrenchCompensator(config)
# 无接触状态下先做一次 tare()，之后用 compensate() 扣偏置和随姿态变化的重力
```

不要沿用默认的竖直重力 `(0, 0, -9.80665)`：以仓库自带的 `gravity_test01` 数据为例，
底盘倾斜 15.94°，漏掉的水平重力分量是 1.86 N，力残差会从 0.19 N 涨到 1.10 N。

## 8. 收尾

```bash
python3 examples/jaka_driver_tool/jaka_stop.py
```

## 常见问题速查

| 现象 | 处理 |
| --- | --- |
| `output CSV already exists` | 加 `--overwrite`，或换一个 `--output` |
| `cannot initialize GLFW display` | 到有桌面的机器上运行，或使用 `ssh -X` |
| 窗口一直不保存 | 机械臂没停稳或一直按着方向键；看窗口标题里的 `armed` / `settling` / `recording` 状态 |
| 有效姿态数少于采集次数 | 窗口中途被移动打断而作废，属正常，重采该姿态即可 |
| 辨识报姿态数不足 | 默认全行模式至少需要 `--min-samples` 行，且模型本身至少需要 6 个有效姿态；数据太少时降低 `--min-samples` 并补采 |
| 想改窗口长度 | 使用 `--static-poses` 时采集加 `--capture-seconds N`，辨识同步把 `--min-samples` 调到 `N*5+1` 以内 |
| 采集 CSV 全是 `capture_id=-1` | 默认全行模式可以直接辨识；若用 `--static-poses` 则会跳过这些行 |
| 重力向量解出来接近 0 | 控制器已做重力补偿，换成未补偿的 `--torque-sensor-mode` 重采 |
| 倾角告警 | 超过 `--max-tilt-deg`（默认 30°）才提示，用于发现坐标系/符号量级错误；正常底盘倾斜不会告警 |
