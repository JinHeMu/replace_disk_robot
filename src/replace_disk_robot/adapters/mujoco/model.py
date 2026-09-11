"""MuJoCo model loading helpers shared by demos and tests."""

from pathlib import Path

import mujoco


PROJECT_ROOT = Path(__file__).resolve().parents[4]


def scene_path() -> Path:
    return PROJECT_ROOT / "simulation" / "mujoco" / "scene.xml"


def load_model() -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_path(str(scene_path()))
    return model, mujoco.MjData(model)


def reset_home(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id < 0:
        raise KeyError("scene is missing the required 'home' keyframe")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)
