#!/usr/bin/env python3
"""Verify the project environment without connecting to or moving the robot.

--camera additionally captures D435 RGB-D frames; --require-cuda rejects CPU mode.
The SAC check uses Pendulum, not the unimplemented real-robot learning environment.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import os
from pathlib import Path
import site
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def check_packages() -> None:
    prefix = (ROOT / ".venv").resolve()
    require(Path(sys.prefix).resolve() == prefix, "请先 source scripts/activate_env.sh")
    require(not site.ENABLE_USER_SITE, "用户 site-packages 未隔离")
    for variable in ("PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH"):
        require(not os.environ.get(variable), f"{variable} 未清空，请重新激活环境")
    require(not any("/opt/ros/" in path for path in sys.path), "sys.path 混入了 ROS")
    print(f"Python {sys.version.split()[0]}: {sys.prefix}", flush=True)
    packages = {
        "numpy": "numpy", "mujoco": "mujoco", "pinocchio": "pin",
        "pygame": "pygame", "yaml": "PyYAML", "PIL": "Pillow",
        "matplotlib": "matplotlib", "pyrealsense2": "pyrealsense2",
        "cv2": "opencv-python-headless", "h5py": "h5py",
        "torch": "torch", "torchvision": "torchvision",
        "gymnasium": "gymnasium", "stable_baselines3": "stable-baselines3",
        "tensorboard": "tensorboard", "pytest": "pytest",
    }
    for module_name, distribution in packages.items():
        module = importlib.import_module(module_name)
        require(Path(module.__file__).resolve().is_relative_to(prefix),
                f"{module_name} 从项目环境外导入: {module.__file__}")
        print(f"  {distribution}=={importlib.metadata.version(distribution)}", flush=True)


def check_jaka() -> None:
    import mujoco
    import numpy as np

    from replace_disk_robot.adapters.jaka.interfaces import load_jkrc
    from replace_disk_robot.adapters.mujoco import load_model, reset_keyframe
    from replace_disk_robot.core import JointState
    from replace_disk_robot.kinematics.jaka import JakaKinematics

    require(hasattr(load_jkrc(), "RC"), "JAKA SDK 缺少 RC 接口")
    kinematics = JakaKinematics()
    pose = kinematics.forward(JointState(kinematics.joint_names, np.zeros(6)))
    require(kinematics.dof == 6 and np.isfinite(pose.position_m).all(), "JAKA FK 失败")
    model, data = load_model("jaka")
    reset_keyframe(model, data, "low")
    for _ in range(10):
        mujoco.mj_step(model, data)
    require(np.isfinite(data.qpos).all(), "JAKA MuJoCo 状态无效")
    print("PASS JAKA ZU5: SDK 导入、六轴 FK、MuJoCo 步进（未连接机械臂）", flush=True)


def check_learning(require_cuda: bool) -> None:
    import gymnasium as gym
    import numpy as np
    import torch
    from stable_baselines3 import SAC
    from torch.utils.tensorboard import SummaryWriter
    from torchvision.ops import nms

    torch.set_num_threads(1)
    cuda = torch.cuda.is_available()
    require(cuda or not require_cuda, "要求 CUDA，但 PyTorch 无法使用 NVIDIA GPU")
    device = "cuda" if cuda else "cpu"
    print(f"RL device: {torch.cuda.get_device_name(0) if cuda else 'CPU'}; "
          f"PyTorch CUDA runtime={torch.version.cuda}", flush=True)
    # Exercise CUDA kernels, autograd and the compiled torchvision extension.
    layer = torch.nn.Conv2d(3, 8, 3).to(device)
    loss = layer(torch.randn(2, 3, 32, 32, device=device)).square().mean()
    loss.backward()
    require(bool(torch.isfinite(layer.weight.grad).all()), "视觉网络梯度无效")
    boxes = torch.tensor([[0., 0., 1., 1.], [0., 0., 1., 1.]], device=device)
    require(nms(boxes, torch.tensor([.9, .8], device=device), .5).numel() == 1,
            "torchvision 原生算子失败")

    env = gym.make("Pendulum-v1")
    try:
        model = SAC("MlpPolicy", env, device=device, seed=7, verbose=0,
                    buffer_size=256, learning_starts=16, batch_size=16,
                    train_freq=4, gradient_steps=1, policy_kwargs={"net_arch": [32, 32]})
        before = [parameter.detach().clone() for parameter in model.actor.parameters()]
        model.learn(total_timesteps=64)
        require(model._n_updates > 0, "SAC 没有执行梯度更新")
        require(any(not torch.equal(old, new) for old, new in
                    zip(before, model.actor.parameters())), "SAC actor 参数没有变化")
        with tempfile.TemporaryDirectory(prefix="jaka-sac-check-") as temporary:
            path = Path(temporary) / "sac.zip"
            model.save(path)
            restored = SAC.load(path, env=env, device=device)
            observation, _ = env.reset(seed=7)
            action, _ = restored.predict(observation, deterministic=True)
            require(np.isfinite(action).all() and env.action_space.contains(action),
                    "SAC 恢复后的动作无效")
            with SummaryWriter(str(Path(temporary) / "tensorboard")) as writer:
                writer.add_scalar("check/updates", model._n_updates, 64)
            require(any((Path(temporary) / "tensorboard").glob("events.out.tfevents.*")),
                    "TensorBoard 日志未写入")
        print(f"PASS RL: 视觉网络反传、SAC 64 步 / {model._n_updates} 次更新、"
              "保存/加载、TensorBoard 日志", flush=True)
    finally:
        env.close()


def check_camera(capture: bool, serial: str | None) -> None:
    import numpy as np
    import pyrealsense2 as rs

    context = rs.context()
    devices = list(context.query_devices())
    candidates = []
    for device in devices:
        name = device.get_info(rs.camera_info.name)
        number = device.get_info(rs.camera_info.serial_number)
        print(f"RealSense: {name}, serial={number}", flush=True)
        if "D435" in name and (serial is None or number == serial):
            candidates.append(device)
    if not capture:
        print(f"PASS RealSense SDK; D435 数量={len(candidates)}；RGB-D 取流尚未检查", flush=True)
        return
    require(len(candidates) == 1,
            "需要一台可访问的 D435；检查 USB 3、udev 权限，多台时用 --serial 指定")
    device = candidates[0]
    number = device.get_info(rs.camera_info.serial_number)
    if device.supports(rs.camera_info.usb_type_descriptor):
        print(f"D435 USB: {device.get_info(rs.camera_info.usb_type_descriptor)}", flush=True)
    pipeline = rs.pipeline(context)
    configuration = rs.config()
    configuration.enable_device(number)
    configuration.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    configuration.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(configuration)
    try:
        align = rs.align(rs.stream.color)
        for _ in range(10):
            pipeline.wait_for_frames(5000)
        timestamps = []
        for _ in range(30):
            frames = align.process(pipeline.wait_for_frames(5000))
            color, depth = frames.get_color_frame(), frames.get_depth_frame()
            require(bool(color) and bool(depth), "缺少 RGB 或深度帧")
            rgb = np.asanyarray(color.get_data())
            depth_raw = np.asanyarray(depth.get_data())
            require(rgb.shape == (480, 640, 3) and depth_raw.shape == (480, 640),
                    "RGB-D 图像尺寸错误")
            timestamps.append((color.get_timestamp(), depth.get_timestamp()))
        require(bool((np.diff(np.asarray(timestamps), axis=0) > 0).all()),
                "RGB 或深度时间戳没有递增")
        scale = profile.get_device().first_depth_sensor().get_depth_scale()
        print(f"PASS D435: 30 对 RGB-D，640×480，对齐到 RGB；"
              f"深度单位={scale:g} m，末帧有效深度比例={np.mean(depth_raw > 0):.1%}", flush=True)
    finally:
        pipeline.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", action="store_true", help="要求 D435 并读取 30 对 RGB-D")
    parser.add_argument("--serial", help="指定 D435 序列号，配合 --camera")
    parser.add_argument("--require-cuda", action="store_true", help="CUDA 不可用时失败")
    arguments = parser.parse_args()
    if arguments.serial and not arguments.camera:
        parser.error("--serial 需要 --camera")
    check_packages()
    check_jaka()
    check_learning(arguments.require_cuda)
    check_camera(arguments.camera, arguments.serial)
    print("ENVIRONMENT CHECK: PASS", flush=True)


if __name__ == "__main__":
    main()
