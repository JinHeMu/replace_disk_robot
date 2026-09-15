#!/usr/bin/env python3
"""Hold-to-jog Cartesian servo for a selectable MuJoCo robot model.

Keyboard velocities can be expressed in the kinematics base frame or in the
current TCP/tool frame.  Both models start in base mode; tool mode is opt-in
with ``--command-frame tool``.  JAKA R/F always insert/retract along the
current tool0 +Z/-Z axis regardless of the selected command frame.

``--admittance`` inserts a compliant stage between the keyboard and the servo:

    keys -> nominal pose -> admittance correction -> pose-tracking servo -> robot

The measured F/T wrench is transformed to the kinematics base frame, shifted to
the TCP, filtered/deadbanded, and converted to an external load before being
fed to the admittance model.  The force limit guard stays on the raw sensor path.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from replace_disk_robot.adapters.mujoco import (
    load_model, reset_keyframe, MujocoRobotAdapter, MujocoWristFTAdapter,
)
from replace_disk_robot.contact import WrenchProcessor, WrenchProcessorConfig
from replace_disk_robot.control import (
    AdmittanceConfig, AdmittanceController, CartesianServo, KeyboardAdmittanceController,
    KeyControl, MotionReferenceConfig, ServoConfig,
)
from replace_disk_robot.core import CartesianJog, Pose, Wrench
from replace_disk_robot.core.rotation import quaternion_from_rotation_matrix, rotation_matrix
from replace_disk_robot.kinematics.jaka import JAKA_JOINT_NAMES, JakaKinematics
from replace_disk_robot.kinematics.ur5e import UR5eKinematics
from replace_disk_robot.kinematics.tool import FixedToolKinematics
from replace_disk_robot.safety import ForceLimitGuard


FORCE_STOP_N = 20.0
JAKA_ACTUATORS = tuple(f'joint_{index}_servo' for index in range(1, 7))
MODEL_CHOICES = ('ur5e', 'jaka')


class ServoDemo:
    """Application wiring: keyboard -> nominal pose/admittance -> safety gate -> ArmPort."""
    def __init__(self, linear_speed=.01, angular_speed=np.deg2rad(5),
                 model_name='ur5e', keyframe=None, command_frame='base',
                 admittance=False, admittance_axes='translation',
                 admittance_max_offset=.02, force_filter_alpha=.2,
                 force_deadband_n=.5, force_torque_deadband_nm=.05,
                 max_nominal_lead=.05, max_nominal_rotation_lead_deg=10.0):
        if model_name not in MODEL_CHOICES:
            raise ValueError(f'unknown model {model_name!r}; expected one of {MODEL_CHOICES}')
        if command_frame not in ('auto', 'base', 'tool'):
            raise ValueError(
                f"command_frame must be 'auto', 'base' or 'tool', got {command_frame!r}"
            )
        if admittance_axes not in ('translation', 'all'):
            raise ValueError(
                f"admittance_axes must be 'translation' or 'all', got {admittance_axes!r}"
            )
        if not np.isfinite(admittance_max_offset) or admittance_max_offset <= 0:
            raise ValueError('admittance_max_offset must be finite and positive')
        if not np.isfinite(max_nominal_lead) or max_nominal_lead <= 0:
            raise ValueError('max_nominal_lead must be finite and positive')
        if not np.isfinite(max_nominal_rotation_lead_deg) or max_nominal_rotation_lead_deg <= 0:
            raise ValueError('max_nominal_rotation_lead_deg must be finite and positive')
        self.model_name = model_name
        self.keyframe_name = keyframe or ('home' if model_name == 'ur5e' else 'low')
        self.model, self.data = load_model(model_name)
        reset_keyframe(self.model, self.data, self.keyframe_name)

        if model_name == 'ur5e':
            self.robot = MujocoRobotAdapter(
                self.model, self.data, compensate_bias=True,
            )
            parent = UR5eKinematics(end_effector_frame='g_base')
            # TCP at the existing pinch site; X forward matches disk length at init.
            self.kinematics = FixedToolKinematics(
                parent, 'g_base', Pose('g_base', [0,0,.145], [.5,-.5,-.5,-.5]),
            )
            joint_limits = parent.joint_limits_rad
            self.base_frame = parent.base_frame
            self.tool_frame = 'pinch'
            self.ft = MujocoWristFTAdapter(self.model, self.data)
        else:
            self.robot = MujocoRobotAdapter(
                self.model,
                self.data,
                compensate_bias=True,
                joint_names=JAKA_JOINT_NAMES,
                actuator_names=JAKA_ACTUATORS,
                gripper_actuator_name=None,
            )
            # The URDF's base and accessories are fixed.  Expressing FK in the
            # arm mount frame keeps base_x/base_y/base_yaw outside this Servo.
            self.kinematics = JakaKinematics()
            joint_limits = self.kinematics.joint_limits_rad
            self.base_frame = self.kinematics.base_frame
            self.tool_frame = self.kinematics.end_effector_frame
            self.ft = MujocoWristFTAdapter(
                self.model,
                self.data,
                force_sensor_name='tcp_fts_force',
                torque_sensor_name='tcp_fts_torque',
                frame_id='tcp_fts_site',
            )
        # No collision checker here: the tool may touch an obstacle and build force.
        # Tracking error is deliberately loose so contact cannot latch the servo
        # before the force gate; the stop condition is the measured force below.
        self.servo = CartesianServo(self.kinematics, joint_limits,
            ServoConfig(linear_speed_m_s=linear_speed, angular_speed_rad_s=angular_speed,
                        max_tracking_error_rad=10.0))
        if command_frame == 'auto':
            # Both supported models start in base mode; tool axes are opt-in.
            self.command_frame_mode = 'base'
        else:
            self.command_frame_mode = command_frame
        self.command_frame = (
            self.tool_frame if self.command_frame_mode == 'tool' else self.base_frame
        )
        self.keys = KeyControl(
            linear_speed, angular_speed, base_frame=self.command_frame,
        )
        # Only the measured force norm is used: stop above 20 N.
        # The torque threshold is disabled for this requested behavior.
        self.guard = ForceLimitGuard(force_limit_n=FORCE_STOP_N, torque_limit_nm=np.inf)
        self.admittance_enabled = bool(admittance)
        self.admittance_axes_mode = admittance_axes
        self.wrench_load_sign = -1.0  # MuJoCo F/T sensors report parent-on-child.
        if model_name == 'ur5e':
            self.force_sensor_site_name = 'wrist_ft_site'
            self.sensor_base_body_name = None
        else:
            self.force_sensor_site_name = 'tcp_fts_site'
            self.sensor_base_body_name = 'jaka_base_link'
        for _ in range(round(.5/self.model.opt.timestep)):
            self.robot.command_joint_positions(self.robot.read_joint_state())
            mujoco.mj_step(self.model, self.data)
        self.ft.tare()
        self.last_wrench = self.ft.read_wrench()
        self.servo.reset(self.robot.read_joint_state())
        self.steps_per_tick = round(.01/self.model.opt.timestep)
        self.dt = self.steps_per_tick*self.model.opt.timestep
        if self.admittance_enabled:
            enabled_axes = np.array(
                [True, True, True, False, False, False]
                if admittance_axes == 'translation' else [True] * 6
            )
            self.admittance = AdmittanceController(AdmittanceConfig(
                frame_id=self.base_frame,
                mass=[1.0, 1.0, 1.0, .01, .01, .01],
                damping=[50.0, 50.0, 50.0, 1.0, 1.0, 1.0],
                stiffness=[200.0, 200.0, 200.0, 20.0, 20.0, 20.0],
                max_offset=np.r_[
                    np.full(3, admittance_max_offset),
                    np.deg2rad(10.0) * np.ones(3),
                ],
                max_velocity=[.1, .1, .1, np.deg2rad(30.0), np.deg2rad(30.0), np.deg2rad(30.0)],
                max_dt_s=max(.01, self.dt),
            ))
            self.motion = KeyboardAdmittanceController(
                self.admittance,
                self.kinematics.forward(self.robot.read_joint_state()),
                MotionReferenceConfig(
                    max_lead_translation_m=max_nominal_lead,
                    max_lead_rotation_rad=np.deg2rad(max_nominal_rotation_lead_deg),
                    enabled_axes=enabled_axes,
                    max_dt_s=max(.01, self.dt),
                ),
            )
            self.wrench_processor = WrenchProcessor(WrenchProcessorConfig(
                frame_id=self.base_frame,
                load_sign=self.wrench_load_sign,
                filter_alpha=force_filter_alpha,
                force_deadband_n=force_deadband_n,
                torque_deadband_nm=force_torque_deadband_nm,
            ))
            self.last_external_wrench = Wrench(self.base_frame, np.zeros(3), np.zeros(3))
        self._verify_tcp()

    def _verify_tcp(self):
        pose = self.kinematics.forward(self.robot.read_joint_state())
        if self.model_name == 'ur5e':
            expected_position = self.data.site('pinch').xpos
            expected_rotation = self.data.site('drive_center').xmat.reshape(3,3)
            tolerance = 1e-7
        else:
            # MJCF omits the massless URDF tool0 body.  Its fixed transform is
            # 270 mm along tool0_and_camera_link +Z, with identical rotation.
            base = self.data.body('jaka_base_link')
            tool = self.data.body('tool0_and_camera_link')
            world_rotation_base = base.xmat.reshape(3, 3)
            world_rotation_tool = tool.xmat.reshape(3, 3)
            world_position_tool0 = (
                tool.xpos + world_rotation_tool @ np.array([0.0, 0.0, 0.27])
            )
            expected_position = world_rotation_base.T @ (
                world_position_tool0 - base.xpos
            )
            expected_rotation = world_rotation_base.T @ world_rotation_tool
            # The source URDF uses rounded RPY values while MJCF stores rounded
            # quaternions, leading to roughly 1e-4 rad orientation difference.
            tolerance = 2e-4
        if (
            not np.allclose(pose.position_m, expected_position, atol=tolerance)
            or not np.allclose(
                rotation_matrix(pose.quaternion_wxyz),
                expected_rotation,
                atol=tolerance,
            )
        ):
            raise RuntimeError(
                f'{self.model_name} TCP kinematics and MuJoCo model disagree'
            )

    def stop(self, reason='stopped'):
        self.keys.clear()
        self.servo.halt(self.robot.read_joint_state(), reason)

    def resume(self):
        self.keys.clear()
        # Resume does not retare and hide an existing load.
        measured = self.robot.read_joint_state()
        self.servo.reset(measured)
        if self.admittance_enabled:
            # Re-seed both references at the measured pose so a fault recovery
            # never replays a stale compliant offset.
            self.motion.reset(self.kinematics.forward(measured))
            self.wrench_processor.reset()
            self.last_external_wrench = Wrench(
                self.base_frame, np.zeros(3), np.zeros(3),
            )

    def _current_pose(self):
        target = self.servo.target
        if target is None:
            target = self.robot.read_joint_state()
        return self.kinematics.forward(target)

    def _jog_to_servo(self, pose=None) -> CartesianJog:
        """Map active keys to the mixed convention expected by Servo.

        ``CartesianServo`` consumes linear velocity in the kinematics base
        frame and angular velocity in the current TCP axes.

        * tool mode: W/S/A/D and rotations use the current tool frame.  For
          JAKA, R/F are the insertion pair and always follow tool0 +Z/−Z (the
          blue/flange axis), matching the physical insertion direction.
        * JAKA base mode: W/S/A/D translate in base, R/F still translate along
          the current tool0 +Z (blue axis), and Q/E/arrow rotations are
          specified in base axes then converted to intrinsic TCP axes.
        * UR5e base mode: preserve the previous base/tool-key behavior.

        ``pose`` lets the admittance path integrate keys in the nominal frame
        instead of the measured/servo-target frame.
        """
        if pose is None:
            pose = self._current_pose()
        if self.command_frame_mode == 'tool':
            forward_axis = np.array([0.0, 0.0, 1.0]) if self.model_name == 'jaka' else None
            command = self.keys.command(forward_axis=forward_axis)
            base_from_tool = rotation_matrix(pose.quaternion_wxyz)
            return CartesianJog(
                self.base_frame,
                base_from_tool @ command.linear_m_s,
                command.angular_rad_s,
            )
        if self.model_name == 'jaka':
            base_from_tool = rotation_matrix(pose.quaternion_wxyz)
            command = self.keys.command(forward_axis=base_from_tool[:, 2])
            return CartesianJog(
                self.base_frame,
                command.linear_m_s,
                base_from_tool.T @ command.angular_rad_s,
            )
        return self.keys.command()

    def _sensor_pose_in_base(self) -> Pose:
        """Return the MuJoCo F/T site pose in the kinematics base frame."""
        site = self.data.site(self.force_sensor_site_name)
        world_from_sensor = site.xmat.reshape(3, 3)
        position_world = np.asarray(site.xpos).copy()
        if self.sensor_base_body_name is None:
            return Pose(
                self.base_frame,
                position_world,
                quaternion_from_rotation_matrix(world_from_sensor),
            )
        base = self.data.body(self.sensor_base_body_name)
        world_from_base = base.xmat.reshape(3, 3)
        base_from_world = world_from_base.T
        return Pose(
            self.base_frame,
            base_from_world @ (position_world - np.asarray(base.xpos)),
            quaternion_from_rotation_matrix(base_from_world @ world_from_sensor),
        )

    def _submit_admittance_command(self, measured) -> None:
        """Run keyboard nominal pose + admittance and refresh the pose servo."""
        raw_wrench = self.ft.read_wrench()
        self.last_wrench = raw_wrench
        if not np.isfinite(raw_wrench.as_vector()).all():
            self.stop('invalid_wrench')
            return
        actual_pose = self.kinematics.forward(measured)
        external_wrench = self.wrench_processor.update(
            raw_wrench,
            self._sensor_pose_in_base(),
            actual_pose.position_m,
        )
        self.last_external_wrench = external_wrench
        jog = self._jog_to_servo(self.motion.nominal_pose)
        corrected = self.motion.update(jog, external_wrench, actual_pose, self.dt)
        self.servo.submit_pose(corrected, self.data.time)

    def tick(self, refresh=True):
        measured = self.robot.read_joint_state()
        if refresh and self.servo.fault is None:
            if self.admittance_enabled:
                self._submit_admittance_command(measured)
            else:
                self.servo.submit(self._jog_to_servo(), self.data.time)
        target = self.servo.update(measured, self.dt, self.data.time)
        for _ in range(self.steps_per_tick):
            self.last_wrench = self.ft.read_wrench()
            wrench = self.last_wrench.as_vector()
            if not np.isfinite(wrench).all():
                self.stop('invalid_wrench')
                target = self.servo.target
            else:
                _, tripped = self.guard.filter_arm_target(wrench, self.robot.arm_position(), target.position_rad)
                if tripped and not self.servo.fault:
                    self.stop('force_limit')
                    target = self.servo.target
            self.robot.command_joint_positions(target)
            mujoco.mj_step(self.model, self.data)
            if not np.isfinite(self.data.qpos).all():
                raise RuntimeError('Non-finite simulation state')


def headless_report(model_name='ur5e', keyframe=None, command_frame='base'):
    """Exercise all twelve actual key mappings through servo and MuJoCo dynamics."""
    from replace_disk_robot.control.key_control import KEY_AXES
    results = []
    app = None
    for key, (translation, rotation) in KEY_AXES.items():
        app = ServoDemo(
            model_name=model_name,
            keyframe=keyframe,
            command_frame=command_frame,
        )
        initial = app.kinematics.forward(app.robot.read_joint_state())
        r0 = rotation_matrix(initial.quaternion_wxyz)
        app.keys.press(key)
        for _ in range(50):
            app.tick()
        app.keys.release(key)
        for _ in range(50):
            app.tick()
        end = app.kinematics.forward(app.robot.read_joint_state())
        r1 = rotation_matrix(end.quaternion_wxyz)
        rot_delta = r0.T @ r1
        angular = np.array([rot_delta[2,1]-rot_delta[1,2], rot_delta[0,2]-rot_delta[2,0],
                            rot_delta[1,0]-rot_delta[0,1]])/2
        delta = end.position_m-initial.position_m
        translation, rotation = np.array(translation, dtype=float), np.array(rotation, dtype=float)
        if app.model_name == 'jaka':
            # JAKA R/F are always the insertion pair: current tool0 +Z/-Z.
            # In base mode other translations stay in base axes; in tool mode
            # other translations are specified in tool axes and rotated here.
            if key in ('r', 'f'):
                translation = r0 @ np.array([
                    0.0, 0.0, 1.0 if key == 'r' else -1.0,
                ])
            elif app.command_frame_mode == 'tool':
                translation = r0 @ translation
            if app.command_frame_mode == 'base':
                # The measured angular vector is expressed in the initial tool
                # frame, so base-frame rotation commands must be transformed.
                rotation = r0.T @ rotation
        elif app.command_frame_mode == 'tool':
            # delta is expressed in the base frame, while angular is the
            # rotation vector in the initial tool frame (log(r0.T @ r1)).
            # Therefore only the expected translation needs frame conversion.
            translation = r0 @ translation
        projection = float(delta@translation if translation.any() else angular@rotation)
        held = app.servo.target.position_rad.copy()
        for _ in range(20):
            app.tick()
        passed = bool(projection > (.002 if translation.any() else .01) and
                      not app.servo.fault and app.data.ncon == 0 and
                      np.array_equal(held, app.servo.target.position_rad) and
                      (translation.any() or np.linalg.norm(delta) < .001))
        results.append(dict(key=key, passed=passed, displacement_m=delta.tolist(),
                            rotation_tcp_rad=angular.tolist(), status=app.servo.status))
    return dict(
        passed=all(r['passed'] for r in results),
        model=model_name,
        keyframe=keyframe or ('home' if model_name == 'ur5e' else 'low'),
        command_frame=app.command_frame if app is not None else command_frame,
        tests=results,
        scope='Synthetic key events through actual servo/dynamics; not an OS keyboard test',
    )


def _synthetic_raw_wrench(app, external_force_base):
    """Return a sensor-frame reading whose external load is the given force."""
    sensor_pose = app._sensor_pose_in_base()
    rotation = rotation_matrix(sensor_pose.quaternion_wxyz)
    raw_force = rotation.T @ (np.asarray(external_force_base, dtype=float) / app.wrench_load_sign)
    return Wrench(app.ft.frame_id, raw_force, np.zeros(3))


def headless_admittance_report(model_name='ur5e', keyframe=None, command_frame='base'):
    """Validate keyboard tracking and directional compliance in MuJoCo."""
    app = ServoDemo(
        model_name=model_name,
        keyframe=keyframe,
        command_frame=command_frame,
        admittance=True,
    )
    tests = []

    # 1. No external load: held keys must still move the nominal/tracking chain.
    initial = app.kinematics.forward(app.robot.read_joint_state())
    app.keys.press('w')
    for _ in range(40):
        app.tick()
    app.keys.release('w')
    for _ in range(40):
        app.tick()
    after_key = app.kinematics.forward(app.robot.read_joint_state())
    key_displacement = after_key.position_m - initial.position_m
    tests.append(dict(
        name='keyboard_tracking_without_force',
        passed=bool(
            not app.servo.fault
            and np.linalg.norm(key_displacement) > 0.001
            and np.linalg.norm(app.motion.state().offset) < 1e-6
        ),
        displacement_m=key_displacement.tolist(),
        status=app.servo.status,
    ))

    # 2. Constant external load on the TCP: compliance must move along the load
    #    and release back toward the nominal pose once the load is removed.
    start = app.kinematics.forward(app.robot.read_joint_state()).position_m.copy()
    app.ft.read_wrench = lambda: _synthetic_raw_wrench(app, [0.0, 0.0, -5.0])
    for _ in range(150):
        app.tick()
    loaded = app.kinematics.forward(app.robot.read_joint_state()).position_m.copy()
    load_offset = app.motion.state().offset.copy()
    app.ft.read_wrench = lambda: Wrench(app.ft.frame_id, np.zeros(3), np.zeros(3))
    for _ in range(150):
        app.tick()
    released = app.kinematics.forward(app.robot.read_joint_state()).position_m.copy()
    tests.append(dict(
        name='translation_admittance_direction_and_release',
        passed=bool(
            not app.servo.fault
            and (loaded - start)[2] < -0.003
            and load_offset[2] < 0.0
            and np.linalg.norm(released - start) < 0.003
        ),
        loaded_displacement_m=(loaded - start).tolist(),
        offset_m=load_offset[:3].tolist(),
        released_residual_m=(released - start).tolist(),
        status=app.servo.status,
    ))

    return dict(
        passed=all(test['passed'] for test in tests),
        model=model_name,
        keyframe=app.keyframe_name,
        command_frame=app.command_frame,
        admittance_axes=app.admittance_axes_mode,
        tests=tests,
        scope='Synthetic F/T readings through keyboard/admittance/MuJoCo dynamics; not a hardware test',
    )


def handle_focus(app, focused):
    """Focus pauses never erase a force/explicit-stop fault or replay held keys."""
    app.keys.clear()
    if not focused and app.servo.fault is None:
        app.stop('focus_lost')
    elif focused and app.servo.fault == 'focus_lost':
        app.resume()


def control_status(app):
    status = app.servo.status
    if app.servo.fault == 'focus_lost':
        return 'focus_lost: click ROBOT window, then press a motion key'
    if app.servo.fault:
        return f'{status}: click ROBOT window and press Enter to resume'
    return status


def run_window(app, plot_wrench=False):
    import glfw
    if not glfw.init():
        raise RuntimeError('Cannot initialize GLFW display; use --headless for validation')
    window = None
    context = None
    plotter = None
    try:
        window = glfw.create_window(
            1200, 850, f'Cartesian servo | {app.model_name}', None, None,
        )
        if window is None:
            raise RuntimeError('Cannot create servo window')
        glfw.make_context_current(window)
        glfw.swap_interval(0)
        scene = mujoco.MjvScene(app.model, maxgeom=2000)
        context = mujoco.MjrContext(app.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        if app.model_name == 'ur5e':
            camera.lookat[:] = [.35, 0, .4]
            camera.distance = 1.25
        else:
            camera.lookat[:] = app.data.body('jaka_base_link').xpos + [0, 0, .35]
            camera.distance = 1.8
        camera.azimuth, camera.elevation = 135, -25
        option = mujoco.MjvOption()
        if plot_wrench:
            from replace_disk_robot.visual import ProcessTypePlotter
            plotter = ProcessTypePlotter(window_s=10.0, refresh_hz=10.0)
        keymap = {
            getattr(glfw, 'KEY_' + key.upper()): key
            for key in ('w', 's', 'a', 'd', 'r', 'f', 'q', 'e',
                        'up', 'down', 'left', 'right')
        }

        def on_key(window, key, scancode, action, mods):
            if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
                app.stop()
                glfw.set_window_should_close(window, True)
            elif key == glfw.KEY_SPACE and action == glfw.PRESS:
                app.stop()
            elif key == glfw.KEY_ENTER and action == glfw.PRESS:
                app.resume()
            elif key in keymap:
                if action == glfw.PRESS:
                    app.keys.press(keymap[key])
                elif action == glfw.RELEASE:
                    app.keys.release(keymap[key])

        def on_focus(window, focused):
            handle_focus(app, bool(focused))

        glfw.set_key_callback(window, on_key)
        glfw.set_window_focus_callback(window, on_focus)
        # Mouse drag changes view; the arrow keys belong exclusively to the robot.
        cursor = [0.,0.]
        def on_cursor(window, x, y):
            dx, dy = x-cursor[0], y-cursor[1]
            cursor[:] = [x,y]
            width,height = glfw.get_window_size(window)
            if height > 0 and glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
                mujoco.mjv_moveCamera(
                    app.model, mujoco.mjtMouse.mjMOUSE_ROTATE_V,
                    dx / height, dy / height, camera,
                )
        def on_scroll(window, xoffset, yoffset):
            mujoco.mjv_moveCamera(
                app.model, mujoco.mjtMouse.mjMOUSE_ZOOM,
                0, -.05 * yoffset, camera,
            )
        glfw.set_cursor_pos_callback(window, on_cursor)
        glfw.set_scroll_callback(window, on_scroll)
        # Enqueue only: GUI initialization and every redraw run in the child.
        if plotter is not None:
            plotter.update(app.last_wrench, app.data.time)
        deadline = time.monotonic()
        next_frame = deadline
        forward_label = (
            'insert/retract' if app.model_name in MODEL_CHOICES else 'forward/back'
        )
        print(f'Model: {app.model_name}; keyframe: {app.keyframe_name}; '
              f'command frame: {app.command_frame}')
        print(f'W/S up/down; A/D left/right; R/F {forward_label}; '
              'Q/E yaw; Up/Down pitch; Left/Right roll')
        print('Click ROBOT window to control. Plot window does not accept robot keys.')
        print('Hold to move; release to hold. Space stop; Enter resume; Esc exit.')
        print('Focus loss pauses; clicking ROBOT window resumes only that focus pause.')
        print(f'Obstacle contact is allowed; measured force > {FORCE_STOP_N:g} N stops the servo.')
        if app.admittance_enabled:
            print(f'Admittance ON: {app.admittance_axes_mode} axes; '
                  f'offset limit {app.admittance.config.max_offset[:3]} m; '
                  f'wrench filtered in {app.base_frame}.')
        if app.model_name == 'jaka' and app.command_frame_mode == 'base':
            print('JAKA base mode: R/F move along current tool0 +Z (blue axis); '
                  'other keys use base axes.')
        previous_status = None
        plot_error_reported = False
        while not glfw.window_should_close(window):
            glfw.poll_events()
            now = time.monotonic()
            if now-deadline > .15:
                app.stop('loop_timeout')
                deadline = now
            if now >= deadline:
                app.tick()
                if plotter is not None:
                    plotter.update(app.last_wrench, app.data.time)
                deadline += app.dt
            status = control_status(app)
            if status != previous_status:
                print(f'[servo] {status}', flush=True)
                glfw.set_window_title(
                    window, f'Cartesian servo | {app.model_name} | {status}',
                )
                previous_status = status
            if plotter is not None and plotter.error and not plot_error_reported:
                print(f'[plot] Plot disabled; robot window remains active:\n{plotter.error}', flush=True)
                plot_error_reported = True
            width,height = glfw.get_framebuffer_size(window)
            if width and height and now >= next_frame:
                next_frame = now + 1/60
                viewport = mujoco.MjrRect(0,0,width,height)
                mujoco.mjv_updateScene(app.model,app.data,option,None,camera,mujoco.mjtCatBit.mjCAT_ALL,scene)
                mujoco.mjr_render(viewport,scene,context)
                mode = ' | Admittance' if app.admittance_enabled else ''
                mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    viewport, f'W/S up/down | A/D left/right | R/F {forward_label}\nQ/E yaw | Arrows: pitch / roll\nSpace stop | Enter resume | Esc exit | Force stop: 20 N{mode}',
                    status, context)
                glfw.swap_buffers(window)
            time.sleep(.001)
    finally:
        app.stop()
        if plotter is not None:
            plotter.close()
        if context is not None:
            context.free()
        if window is not None:
            glfw.destroy_window(window)
        glfw.terminate()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--linear-speed', type=float, default=.01, help='m/s (default 0.01)')
    parser.add_argument('--angular-speed-deg', type=float, default=5., help='degrees/s (default 5)')
    parser.add_argument('--model', choices=MODEL_CHOICES, default='ur5e',
                        help='MuJoCo robot model (default: ur5e)')
    parser.add_argument('--keyframe',
                        help='initial keyframe (default: ur5e=home, jaka=low)')
    parser.add_argument(
        '--command-frame',
        choices=('auto', 'base', 'tool'),
        default='base',
        help=('keyboard velocity frame; base (default): UR5e world, JAKA '
              'jaka_base_link. tool (opt-in): W/S/A/D/rotations use the '
              'current TCP/tool0 axes. auto is an alias for base. '
              'JAKA R/F always insert/retract along current tool0 +Z/-Z.'),
    )
    parser.add_argument('--admittance', action='store_true',
                        help='enable keyboard nominal pose + six-axis admittance compliance')
    parser.add_argument('--admittance-axes', choices=('translation', 'all'),
                        default='translation',
                        help='wrench axes used by admittance (default: translation only)')
    parser.add_argument('--admittance-max-offset', type=float, default=.02,
                        help='translation admittance offset limit in m (default: 0.02)')
    parser.add_argument('--force-filter-alpha', type=float, default=.2,
                        help='F/T low-pass alpha in (0, 1] (default: 0.2)')
    parser.add_argument('--force-deadband-n', type=float, default=.5,
                        help='per-axis force deadband in N (default: 0.5)')
    parser.add_argument('--force-torque-deadband-nm', type=float, default=.05,
                        help='per-axis torque deadband in N*m (default: 0.05)')
    parser.add_argument('--max-nominal-lead', type=float, default=.05,
                        help='nominal reference lead over measured TCP in m (default: 0.05)')
    parser.add_argument('--max-nominal-rotation-lead-deg', type=float, default=10.,
                        help='nominal reference rotation lead over measured TCP in deg')
    parser.add_argument('--headless', action='store_true',
                        help='run scripted key checks (with --admittance: compliance checks)')
    parser.add_argument('--plot-wrench', action='store_true',
                        help='show live force and torque plots in a second window')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.headless:
        if args.admittance:
            report = headless_admittance_report(args.model, args.keyframe, args.command_frame)
        else:
            report = headless_report(args.model, args.keyframe, args.command_frame)
        text = json.dumps(report, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text+'\n')
        print(text)
        if not report['passed']:
            raise SystemExit(1)
    else:
        run_window(
            ServoDemo(
                args.linear_speed, np.deg2rad(args.angular_speed_deg),
                model_name=args.model, keyframe=args.keyframe,
                command_frame=args.command_frame,
                admittance=args.admittance,
                admittance_axes=args.admittance_axes,
                admittance_max_offset=args.admittance_max_offset,
                force_filter_alpha=args.force_filter_alpha,
                force_deadband_n=args.force_deadband_n,
                force_torque_deadband_nm=args.force_torque_deadband_nm,
                max_nominal_lead=args.max_nominal_lead,
                max_nominal_rotation_lead_deg=args.max_nominal_rotation_lead_deg,
            ),
            plot_wrench=args.plot_wrench,
        )


if __name__ == '__main__':
    main()
