# JAKA ZU5 硬盘导轨插入

使用 **JAKA ZU5 + 六维力传感器 + Xbox 手柄 + RealSense D435**，让已夹持的硬盘进入固定机架导轨。
当前已有手柄/键盘遥控、导纳力控和日志分析；下一步接入同步视觉示教，再训练插入策略。

## 环境

Ubuntu 22.04/24.04 x86_64，在仓库根目录执行：

```bash
bash scripts/setup_env.sh && source scripts/activate_env.sh
```

统一安装 JAKA/MuJoCo、D435 和 PyTorch SAC 依赖。安装相机 USB 权限需要 sudo；
后续新终端执行 `source scripts/activate_env.sh`。

## 常用入口

```bash
# 软件与 GPU 自检，不连接机械臂
python examples/check_environment.py --require-cuda

# D435 RGB-D 取流检查
python examples/check_environment.py --camera

# Xbox 手柄控制 JAKA 仿真，默认不连接实机
python src/replace_disk_robot/applications/jaka_joy/jaka_joy_node.py
```

## 文档

- [环境配置](docs/01_环境配置.md)：安装、激活、D435 检查。
- [JAKA 使用说明](docs/02_JAKA使用说明.md)：遥控、力控参数、日志和代码位置。
- [RL 推进计划](docs/03_RL推进计划.md)：同步示教、BC、人在环 RL 与残差对照。

## 代码位置

| 目录 | 内容 |
| --- | --- |
| `src/replace_disk_robot/` | JAKA 适配、运动学、力控及手柄/键盘应用 |
| `examples/jaka_driver_tool/` | 上电、下电和 F/T 读取工具 |
| `tool/` | 重力标定、控制日志分析 |
| `simulation/mujoco/` | JAKA 模型及保留的 UR5e 示例 |
| `rl/` | 现有 UR5e 残差演示；JAKA 真机训练待实现 |

D435 目前提供独立取流检查，视觉与力觉同步采集尚未接入控制应用。
RL 依赖已配置，完整的 JAKA 训练环境、成功判定和在线训练入口仍待实现。
