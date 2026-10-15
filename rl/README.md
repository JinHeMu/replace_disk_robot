# RL 模块

当前代码是 UR5e MuJoCo 残差演示：`env.py` 定义环境，`policy.py` 提供演示策略，
`train.py` 使用 CEM。JAKA 真机 SAC 入口尚未实现，后续工作见 [RL 推进计划](../docs/03_RL推进计划.md)。

与主项目共用 `.venv`，安装见 [环境配置](../docs/01_环境配置.md)。现有代码入口：

```bash
python -m rl --policies zero --episodes 1 --seed 7
python -m rl.train --help
python -m rl.evaluate --help
python -m pytest rl/tests -q
```

实验输出默认写入 `rl/artifacts/`。这些演示结果不代表 JAKA 真机插入能力。
