import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))

import mujoco
import numpy as np

from hard_disk_robot.core import JointState, Pose
from hard_disk_robot.core.rotation import rotation_matrix
from hard_disk_robot.kinematics.ur5e import UR5eKinematics
from hard_disk_robot.kinematics.tool import FixedToolKinematics
from hard_disk_robot.adapters.mujoco import load_model, reset_home


def test_offset_tcp_pose_jacobian_and_inverse_match_mujoco():
    model,data = load_model()
    reset_home(model,data)
    parent = UR5eKinematics(end_effector_frame='g_base')
    kin = FixedToolKinematics(parent,'g_base',Pose('g_base',[0,0,.145],[.5,-.5,-.5,-.5]))
    state = JointState(kin.joint_names, data.qpos[:6])
    pose = kin.forward(state)
    np.testing.assert_allclose(pose.position_m,data.site('pinch').xpos,atol=1e-9)
    np.testing.assert_allclose(rotation_matrix(pose.quaternion_wxyz),data.site('drive_center').xmat.reshape(3,3),atol=1e-9)
    jp,jr = np.zeros((3,model.nv)),np.zeros((3,model.nv))
    mujoco.mj_jacSite(model,data,jp,jr,model.site('pinch').id)
    np.testing.assert_allclose(kin.jacobian(state),np.vstack((jp[:,:6],jr[:,:6])),atol=1e-9)
    displaced = JointState(kin.joint_names,state.position_rad+[.01,-.01,.01,0,0,0])
    target = kin.forward(displaced)
    solution = kin.inverse(target,state)
    np.testing.assert_allclose(kin.forward(solution).position_m,target.position_m,atol=1e-5)
