"""MuJoCo model loading helpers shared by demos and tests."""

from pathlib import Path

import mujoco


PROJECT_ROOT = Path(__file__).resolve().parents[4]
JAKA_MODEL_PATH = (
    PROJECT_ROOT
    / "simulation"
    / "mujoco"
    / "models"
    / "jaka"
    / "tracer_jaka_zu5_robot.xml"
)

MODEL_PATHS = {
    "ur5e": PROJECT_ROOT / "simulation" / "mujoco" / "scene.xml",
    "jaka": JAKA_MODEL_PATH,
}


def scene_path(model_name: str = "ur5e") -> Path:
    """Return the selected MuJoCo model path.

    ``ur5e`` remains the default so existing scene validation keeps its exact
    behavior.  ``jaka`` selects the standalone Tracer + JAKA model.
    """
    try:
        return MODEL_PATHS[model_name]
    except KeyError as exc:
        choices = ", ".join(sorted(MODEL_PATHS))
        raise ValueError(f"unknown model {model_name!r}; expected one of: {choices}") from exc


def load_model(model_name: str = "ur5e") -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_path(str(scene_path(model_name)))
    return model, mujoco.MjData(model)


def reset_keyframe(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    keyframe_name: str,
) -> None:
    if not keyframe_name:
        raise ValueError("keyframe_name must not be empty")
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe_name)
    if key_id < 0:
        raise KeyError(f"model is missing the required {keyframe_name!r} keyframe")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)


def reset_home(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    reset_keyframe(model, data, "home")
