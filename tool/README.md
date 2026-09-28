# 六维力传感器末端负载重力辨识

本目录包含三个脚本：

- `collect_ft_gravity_data.py`：复用 `examples/jaka_driver_tool/jaka_keyboard_servo.py`
  的实机键盘笛卡尔 Servo，以 5 Hz 自动连续采集同步的关节状态、末端姿态和六维力传感器
  原始值（不需要按键触发采集）。
- `identify_ft_payload.py`：从多个静止姿态辨识负载质量、质心和六维传感器常值偏置。
- `plot_ft_gravity_data.py`：用辨识结果离线计算重力/零偏补偿，并绘制补偿前后的六轴曲线。

## 1. 采集

先按仓库原有流程启动并使能 JAKA，再建议先执行只读验证：

```bash
python3 tool/collect_ft_gravity_data.py --dry-run --seconds 20
```

确认网络、EDG 状态、六维数据和窗口均正常后，进行实机采集：

```bash
python3 tool/collect_ft_gravity_data.py \
  --output tool/ft_gravity_samples.csv
```

脚本默认不覆盖已有 CSV；确实需要替换时显式增加 `--overwrite`。

操作方式与原键盘 Servo 一致，但**不需要按采集键**：CSV 以 `--capture-rate-hz`
（默认 5 Hz）连续记录整个会话，是否采样由程序自己判断。把末端移动到一个无碰撞姿态、
松开运动键，机械臂静止 `--settle-seconds`（默认 0.75 s）后自动记录一个
`--capture-seconds`（默认 10 s，5 Hz 下 51 行）的静态窗口；再移动机械臂，下一个静止
姿态会自动记录。建议采集至少 12 个姿态，并让工具坐标系的 X/Y/Z 轴相对重力方向都有
明显变化。不要在接触工件、拖链拉扯明显或负载晃动时采样。

- `Space`：停止并保持
- `Enter`：从当前实测关节位置重新锚定并恢复
- `Esc`：安全退出并关闭 Servo 模式
- 静止窗口内一旦检测到运动或 Servo 故障，该窗口作废并按 `capture_id=-1` 留在 CSV 里，
  重新静止后自动重采，全程不需要按键；
- 运动过程中的采样也以 `capture_id=-1` 写入，所以 CSV 始终是完整、按时间递增的 5 Hz
  记录；`identify_ft_payload.py` 和 `plot_ft_gravity_data.py` 会自动忽略这些行；
- 只有 CSV 记录被抽稀到 5 Hz，伺服与安全控制环仍以 `--rate-hz`（默认 125 Hz）运行，
  因为 JAKA Servo 需要保持命令速率，且 `ServoConfig.max_dt_s` 不接受低于 20 Hz 的周期；
- 保存一个窗口后需要再次移动机械臂才会开始下一组，避免在同一个姿态上重复采集。

CSV 中 `raw_f*`/`raw_t*` 始终来自同一 EDG 包的原始 `torque_sensor`，启动 tare 只用于
在线力限位，不会修改辨识数据。辨识要求控制器输出的是**未做重力补偿**的原始六维值；
若当前 `--torque-sensor-mode` 已由控制器补偿重力，必须先切换为原始输出模式。

辨识脚本默认 `--min-samples 50`，5 Hz 采样时默认的 10 s 窗口刚好给出 51 行；如果把
`--capture-seconds` 调小，请同时把 `identify_ft_payload.py --min-samples` 调小，
否则该姿态会被跳过（采集脚本启动时会给出提示）。

## 2. 离线辨识

```bash
python3 tool/identify_ft_payload.py tool/ft_gravity_samples.csv
```

结果写入 `tool/ft_gravity_samples_identified.json`，主要字段为：

- `mass_kg`：负载质量，等于 `|h|/g`，只取决于重力向量的模，与重力方向无关；
- `signed_gravity_force_base_n`：拟合出的三维重力向量（在 `jaka_base_link` 表达）；
- `gravity_tilt_deg`、`gravity_tilt_azimuth_deg`、`gravity_down_base_unit`：重力方向相对
  base −Z 的倾角、倾斜方位的 base XY 方位角（0° 为 base +X，逆时针指向 +Y）和指向“下方”
  的单位向量，可与倾角仪读数交叉验证；
- `center_of_mass_sensor_mm`：相对传感器测量原点的质心；
- `center_of_mass_tool_mm`：按当前仓库传感器到 `tool0` 的固定变换换算的质心；
- `force_bias_sensor_n`、`torque_bias_sensor_nm`：原始传感器常值偏置；
- `fit`：力/力矩残差和矩阵条件数；
- `warnings`：坐标系、姿态激励或物理合理性警告。

**重力向量本来就是拟合量，不假定它沿 base −Z。** 模型里的 `h_b` 是自由三维向量，底盘倾斜、
地面坡度以及机械臂安装座的固定变换都会体现在 `signed_gravity_force_base_n` 里；质量只取它
的模除以 g，所以倾斜不会影响质量。`--max-tilt-deg`（默认 30°）只用来发现坐标系、符号或
补偿模式这一类量级错误，正常的底盘倾斜不会被判为故障。拟合本身与力的正负号约定无关，
`gravity_down_base_unit` 会按“底盘不可能倒过来”的假设把方向归一成真正的“下方”。

> 采集和辨识都在 `jaka_base_link` 下表达；若要与底盘 `base_link` 比较，还需乘上
> `base_link → jaka_base_link` 的固定变换。

把辨识结果送进在线补偿 `SensorWrenchCompensator` 时，必须使用这条带倾斜的重力向量，
否则会残留 `|h|·sin(tilt)` 的水平重力分量（对 `gravity_test01` 的 15.94°、6.78 N 就是
1.86 N，实测力残差会从 0.19 N 涨到 1.10 N）：

```python
config = SensorCompensationConfig(
    frame_id="tcp_fts_site",
    gravity_frame_id="jaka_base_link",                          # 传感器位姿所在的坐标系
    gravity_m_s2=np.asarray(fit["signed_gravity_force_base_n"]) / fit["mass_kg"],
    load_sign=1.0,                                              # 直接用带符号的重力向量
    payload_mass_kg=fit["mass_kg"],
    payload_com_sensor_m=fit["center_of_mass_sensor_m"],
)
```

## 3. 绘制补偿前后曲线

```bash
python3 tool/plot_ft_gravity_data.py tool/gravity_test01.csv --show
```

绘图脚本默认读取同目录、同文件名前缀的 `_identified.json`，并保存
`*_gravity_comparison.png`。默认只画 CSV 中已接受的静止窗口；可用 `--capture-id 3`
只看某一个姿态。曲线在采样窗口内连续绘制，窗口间用点线连接，以区分采样值与窗口之间的插值。补偿结果按传感器坐标系计算：从原始力/力矩中扣除辨识出的重力项和常值偏置。
CSV 的 `processed_*` 可能还包含坐标变换、滤波和死区，因此不用于这张重力补偿对比图。

模型假设采样期间机器人和负载完全静止、负载刚性固定、传感器比例因子和轴间耦合已由
厂家标定。本工具辨识负载与零偏，不替代六维传感器的比例/耦合标定。将结果写入机器人
控制器前，应先用未参与辨识的静态姿态做交叉验证，并低速、无接触地检查补偿后的残差。
