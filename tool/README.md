# 六维力传感器末端负载重力辨识

本目录包含两个脚本：

- `collect_ft_gravity_data.py`：复用 `examples/jaka_driver_tool/jaka_keyboard_servo.py`
  的实机键盘笛卡尔 Servo，采集同步的关节状态、末端姿态和六维力传感器原始值。
- `identify_ft_payload.py`：从多个静止姿态辨识负载质量、质心和六维传感器常值偏置。

## 1. 采集

先按仓库原有流程启动并使能 JAKA，再建议先执行只读验证：

```bash
python3 tool/collect_ft_gravity_data.py --dry-run --seconds 10
```

确认网络、EDG 状态、六维数据和窗口均正常后，进行实机采集：

```bash
python3 tool/collect_ft_gravity_data.py \
  --output tool/ft_gravity_samples.csv
```

脚本默认不覆盖已有 CSV；确实需要替换时显式增加 `--overwrite`。

操作方式与原键盘 Servo 一致。将末端移动到一个无碰撞姿态，松开运动键，按 `C`；
程序会等待机械臂稳定，然后保存一个静态窗口。建议采集至少 12 个姿态，并让工具坐标系的
X/Y/Z 轴相对重力方向都有明显变化。不要在接触工件、拖链拉扯明显或负载晃动时采样。

- `C`：稳定后采集一个姿态
- `Space`：停止并保持
- `Enter`：从当前实测关节位置重新锚定并恢复
- `Esc`：安全退出并关闭 Servo 模式

CSV 中 `raw_f*`/`raw_t*` 始终来自同一 EDG 包的原始 `torque_sensor`，启动 tare 只用于
在线力限位，不会修改辨识数据。辨识要求控制器输出的是**未做重力补偿**的原始六维值；
若当前 `--torque-sensor-mode` 已由控制器补偿重力，必须先切换为原始输出模式。

## 2. 离线辨识

```bash
python3 tool/identify_ft_payload.py tool/ft_gravity_samples.csv
```

结果写入 `tool/ft_gravity_samples_identified.json`，主要字段为：

- `mass_kg`：负载质量；
- `center_of_mass_sensor_mm`：相对传感器测量原点的质心；
- `center_of_mass_tool_mm`：按当前仓库传感器到 `tool0` 的固定变换换算的质心；
- `force_bias_sensor_n`、`torque_bias_sensor_nm`：原始传感器常值偏置；
- `fit`：力/力矩残差和矩阵条件数；
- `warnings`：坐标系、姿态激励或物理合理性警告。

模型假设采样期间机器人和负载完全静止、负载刚性固定、传感器比例因子和轴间耦合已由
厂家标定。本工具辨识负载与零偏，不替代六维传感器的比例/耦合标定。将结果写入机器人
控制器前，应先用未参与辨识的静态姿态做交叉验证，并低速、无接触地检查补偿后的残差。
