"""Assemble the joystick application with the shared MuJoCo backend."""
from copy import deepcopy
import numpy as np
from replace_disk_robot.core.rotation import rotation_matrix
from ..common.simulation import MujocoJakaClient, MujocoWrenchPipeline

def make_sim_app(args):
    from .joy_node import JakaJoyServo
    args = deepcopy(args)
    client = MujocoJakaClient(args.keyframe, args.rate_hz)
    # A simulated payload is known from MJCF. These settings select that
    # compensation and prevent real calibration/tare settings being applied.
    args.gravity_compensation_enable = True
    args.gravity_json = "model-known-simulation-payload"
    args.tare_enable = False
    app = JakaJoyServo(client, args)
    app.wrench = MujocoWrenchPipeline(client, args, app.kinematics)
    expected = app.kinematics.forward(client.robot.read_joint_state())
    base, tool = client.data.body("jaka_base_link"), client.data.body("tool0")
    rot = base.xmat.reshape(3, 3).T
    if (not np.allclose(expected.position_m, rot @ (tool.xpos - base.xpos), atol=2e-4)
            or not np.allclose(rotation_matrix(expected.quaternion_wxyz),
                               rot @ tool.xmat.reshape(3, 3), atol=2e-4)):
        raise RuntimeError("URDF and MuJoCo tool0 frames disagree")
    return app
