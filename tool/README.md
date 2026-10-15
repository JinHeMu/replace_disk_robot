# 标定与日志工具

先激活项目环境；实机启动和遥控见 [JAKA 使用说明](../docs/02_JAKA使用说明.md)。

| 脚本 | 用途 |
| --- | --- |
| `collect_ft_gravity_data.py` | 手柄切换姿态，自动采集静态原始 F/T |
| `identify_ft_payload.py` | 离线辨识质量、质心、零偏和重力方向 |
| `plot_ft_gravity_data.py` | 绘制重力补偿前后曲线 |
| `analyze_jaka_control_log.py` | 分析手柄/键盘控制日志 |

## 重力标定

先通过 `jaka_start.py` 上电使能，再运行采集。保持工具无外部接触，用手柄切换不同朝向；
松开 LB 并回中，静止后自动记录。采集器不加载已有重力标定或开启导纳。
`--dry-run --seconds 25` 可只读检查；输出路径应为新文件。

```bash
python tool/collect_ft_gravity_data.py --output logs/ft_samples.csv
python tool/identify_ft_payload.py logs/ft_samples.csv --output logs/ft_samples_identified.json
python tool/plot_ft_gravity_data.py logs/ft_samples.csv --identified logs/ft_samples_identified.json
```

检查结果后，通过控制应用的 `--gravity-json` 选用标定；运行时默认标定保存在
`tool/ft_gravity_samples_identified.json`。完整参数见各脚本的 `--help`。

## 控制日志

```bash
python tool/analyze_jaka_control_log.py logs/trial_001 --plots
```

报告、完整逐周期 CSV 和图像写入 `logs/trial_001/analysis/`。
