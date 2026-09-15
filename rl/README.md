# Residual RL 力控微调 Demo

本目录是 `docs/09_强化学习力控微调方案.md` 的第一轮可运行 Demo，遵循
**Residual RL** 边界：

```text
位姿 / 腕部 F/T
      │
      ▼
WrenchProcessor（去偏、坐标变换、滤波）
      │
      ▼
InsertionController（确定性轴向推进 + Y/Z、pitch/yaw 顺应）
      │
      ├──────────────► ResidualPolicy（Zero / Scripted / Linear）
      │                         │ 5 维有界残差
      ▼                         ▼
          ResidualLimiter + 任务空间限幅
                        │
                        ▼
              CartesianServo（100 Hz）
                        │
                        ▼
              InsertionGuard（唯一最终命令出口）
                        │
                        ▼
                  MuJoCo 场景
```

本 Demo 没有直接写 `data.ctrl`，没有让策略访问 MuJoCo 接触对象或随机化真值，
也没有实现实机控制。

## 目录

```text
rl/
├── config.py          # 不可变配置对象；支持可选 YAML 目录
├── configs/           # insertion(含 observation)/residual/reward/randomization/sac
├── features.py        # 方案要求的固定 32 维 actor 观测 schema
├── env.py             # MuJoCo 20 Hz policy / 100 Hz baseline+servo 环境
├── insertion.py       # 确定性 InsertionController 基线
├── residual.py        # 5 维动作、缩放、变化率和累计偏移限制
├── safety.py          # InsertionGuard 硬阈值与命令锁存
├── reward.py          # 纯函数 reward 分项
├── policy.py          # ResidualPolicy 协议、Zero/Scripted/Linear/Torch 包装
├── runner.py          # episode 与固定 seed 评估
├── train.py           # 依赖免安装的 CEM Demo 训练入口；SAC 为后续可选实现
├── evaluate.py        # 固定 seed 策略评估入口
├── demo.py            # `python -m rl` 的薄演示入口
└── tests/             # 仅依赖 NumPy/MuJoCo/Pinocchio 的单元/冒烟测试
```

所有实验产物默认写到 `rl/artifacts/`，该目录已被 `rl/.gitignore` 忽略。

## 环境

仓库当前可用的隔离环境是：

```bash
/home/a/replace_disk_robot/.conda_envs/replace_disk_robot
```

用户提到的 `hard_replace_robot` 在已有 conda env 列表和仓库 `.conda_envs` 中不存在。
本 Demo 在现有 `replace_disk_robot` 隔离环境中验证通过；为读取可选 YAML 配置，
已在该环境安装 `PyYAML`。核心运行路径不依赖 PyYAML，配置默认值内置于
`rl/config.py`；可选依赖记录在 `rl/requirements-rl.txt`。

推荐运行方式：

```bash
cd /home/a/replace_disk_robot
env -u PYTHONPATH -u AMENT_PREFIX_PATH -u COLCON_PREFIX_PATH \
    -u CMAKE_PREFIX_PATH -u LD_LIBRARY_PATH -u PKG_CONFIG_PATH \
    -u ROS_DISTRO -u ROS_VERSION -u ROS_PYTHON_VERSION \
    .conda_envs/replace_disk_robot/bin/python -m rl \
    --policies zero scripted --episodes 1 --seed 7
```

如果当前 shell 已经正确激活该隔离环境，也可以直接：

```bash
python -m rl --policies zero scripted --episodes 1 --seed 7
```

## 快速验证

### 1. 零残差经典基线

```bash
python -m rl --policies zero --episodes 1 --seed 7
```

预期：零残差完成名义 50 mm 插入，且不触发 10 N / 1 Nm 硬安全阈值。

### 2. Demo 脚本残差对比

```bash
python -m rl --policies zero scripted --episodes 1 --seed 7 \
    --record-history --output rl/artifacts/demo_report.json
```

`scripted` 只用于验证残差融合与安全链，不代表学习效果。

### 3. 固定 seed 评估

```bash
python -m rl.evaluate --policy zero --episodes 5 --seed 0
python -m rl.evaluate --policy scripted --episodes 5 --seed 0
python -m rl.evaluate --policy linear --model rl/artifacts/cem_policy.npz
```

### 4. 依赖免安装 CEM Demo 训练

```bash
python -m rl.train --generations 3 --population 6 \
    --episodes-per-candidate 1 --seed 0
```

该训练器用一个 NumPy 线性 actor 跑通与 SAC 相同的环境/观测/动作/奖励接口。
真正训练时应实现方案中的 SAC 入口，或加载导出的 TorchScript `ResidualPolicy`。

### 5. 测试

```bash
python -m pytest rl/tests -q
```

## 关键合同

- **观测**：固定 32 维，顺序见 `features.FEATURE_NAMES`，schema 版本
  `residual-insertion-v1`，只在配置中给出物理尺度，不使用在线 running
  statistics。
- **动作**：固定 5 维 `[Δvx, Δvy, Δvz, Δwy, Δwz]`，归一化到 `[-1, 1]`，
  物理上限、变化率、累计横向/角度偏移限制见 `ResidualConfig`。
- **频率**：物理 `0.0005 s`，Servo/基线 `100 Hz`，RL `20 Hz`，每个策略动作保持
  5 个 Servo 周期。
- **任务坐标系**：所有策略观测、基线命令和残差均在 `socket_entry` 任务坐标系；
  函数在内部转换到 Servo 的历史混合约定，策略不直接处理关节空间。
- **安全**：`InsertionGuard` 锁存硬力/力矩、工作空间、跟踪和数据异常；故障后
  保持当前关节目标，不自动退回。
- **部署**：`ResidualPolicy` 只接收/返回 NumPy 数组；`env.py`、`safety.py`、
  `insertion.py` 不导入 PyTorch/Gym。

## 当前范围与限制

- 只做 MuJoCo UR5e + 当前理想夹持硬盘场景，不做视觉、抓取打滑、关节力矩控制、
  端到端策略、ROS 2 实机输出或在线学习。
- 随机化默认关闭，仅实现小范围初始位姿 C2 随机化；`--randomize` 是压力测试，
  当前 0.5 mm/0.4° 范围已可让零残差基线触发 10 N 硬安全门，这是有意的安全
  行为而非成功保证。C1/C3 的摩擦、刚度、阻尼、传感器噪声和时延随机化接口保留在
  `RandomizationConfig`，尚未接入场景参数。
- `train.py` 是依赖免安装的 CEM Demo，不是生产 SAC 实现；训练产物不能作为实机
  安全依据。
- 7.5 N 是当前场景的持续滑动阻力参考，不是必须精确跟踪的力目标；硬阈值始终在
  `safety.GuardConfig`，不被 reward 或训练配置覆盖。
