#!/usr/bin/env python3
"""Hold-to-jog Cartesian servo in the MuJoCo window; contact stops at force > 20 N."""
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
    load_model, reset_home, MujocoRobotAdapter, MujocoWristFTAdapter,
)
from replace_disk_robot.control import CartesianServo, KeyControl, ServoConfig
from replace_disk_robot.core import Pose
from replace_disk_robot.kinematics.ur5e import UR5eKinematics
from replace_disk_robot.kinematics.tool import FixedToolKinematics
from replace_disk_robot.safety import ForceLimitGuard


FORCE_STOP_N = 20.0


class ServoDemo:
    """Application wiring: keyboard -> servo -> safety gate -> ArmPort adapter."""
    def __init__(self, linear_speed=.01, angular_speed=np.deg2rad(5)):
        self.model, self.data = load_model()
        reset_home(self.model, self.data)
        self.robot = MujocoRobotAdapter(self.model, self.data, compensate_bias=True)
        parent = UR5eKinematics(end_effector_frame='g_base')
        # TCP at the existing pinch site; X forward matches disk length at init.
        self.kinematics = FixedToolKinematics(parent, 'g_base',
            Pose('g_base', [0,0,.145], [.5,-.5,-.5,-.5]))
        # No collision checker here: the tool may touch an obstacle and build force.
        # Tracking error is deliberately loose so contact cannot latch the servo
        # before the force gate; the stop condition is the measured force below.
        self.servo = CartesianServo(self.kinematics, parent.joint_limits_rad,
            ServoConfig(linear_speed_m_s=linear_speed, angular_speed_rad_s=angular_speed,
                        max_tracking_error_rad=10.0))
        self.keys = KeyControl(linear_speed, angular_speed)
        self.ft = MujocoWristFTAdapter(self.model, self.data)
        # Only the measured force norm is used: stop above 20 N.
        # The torque threshold is disabled for this requested behavior.
        self.guard = ForceLimitGuard(force_limit_n=FORCE_STOP_N, torque_limit_nm=np.inf)
        for _ in range(round(.5/self.model.opt.timestep)):
            self.robot.command_joint_positions(self.robot.read_joint_state())
            mujoco.mj_step(self.model, self.data)
        self.ft.tare()
        self.last_wrench = self.ft.read_wrench()
        self.servo.reset(self.robot.read_joint_state())
        self.steps_per_tick = round(.01/self.model.opt.timestep)
        self.dt = self.steps_per_tick*self.model.opt.timestep
        self._verify_tcp()

    def _verify_tcp(self):
        pose = self.kinematics.forward(self.robot.read_joint_state())
        from replace_disk_robot.core.rotation import rotation_matrix
        if (not np.allclose(pose.position_m, self.data.site('pinch').xpos, atol=1e-7) or
                not np.allclose(rotation_matrix(pose.quaternion_wxyz),
                                self.data.site('drive_center').xmat.reshape(3,3), atol=1e-7)):
            raise RuntimeError('TCP kinematics and scene disagree')

    def stop(self, reason='stopped'):
        self.keys.clear()
        self.servo.halt(self.robot.read_joint_state(), reason)

    def resume(self):
        self.keys.clear()
        # Resume does not retare and hide an existing load.
        self.servo.reset(self.robot.read_joint_state())

    def tick(self, refresh=True):
        if refresh:
            self.servo.submit(self.keys.command(), self.data.time)
        target = self.servo.update(self.robot.read_joint_state(), self.dt, self.data.time)
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


def headless_report():
    """Exercise all twelve actual key mappings through servo and MuJoCo dynamics."""
    from replace_disk_robot.control.key_control import KEY_AXES
    from replace_disk_robot.core.rotation import rotation_matrix
    results = []
    for key, (translation, rotation) in KEY_AXES.items():
        app = ServoDemo()
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
        translation, rotation = np.array(translation), np.array(rotation)
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
    return dict(passed=all(r['passed'] for r in results), tests=results,
                scope='Synthetic key events through actual servo/dynamics; not an OS keyboard test')


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
        window = glfw.create_window(1200, 850, 'Cartesian servo', None, None)
        if window is None:
            raise RuntimeError('Cannot create servo window')
        glfw.make_context_current(window)
        glfw.swap_interval(0)
        scene = mujoco.MjvScene(app.model, maxgeom=2000)
        context = mujoco.MjrContext(app.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = [.35, 0, .4]
        camera.distance, camera.azimuth, camera.elevation = 1.25, 135, -25
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
        print('W/S up/down; A/D left/right; R/F insert/retract; Q/E yaw; Up/Down pitch; Left/Right roll')
        print('Click ROBOT window to control. Plot window does not accept robot keys.')
        print('Hold to move; release to hold. Space stop; Enter resume; Esc exit.')
        print('Focus loss pauses; clicking ROBOT window resumes only that focus pause.')
        print(f'Obstacle contact is allowed; measured force > {FORCE_STOP_N:g} N stops the servo.')
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
                glfw.set_window_title(window, f'Cartesian servo | {status}')
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
                mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    viewport, 'W/S up/down | A/D left/right | R/F insert/retract\nQ/E yaw | Arrows: pitch / roll\nSpace stop | Enter resume | Esc exit | Force stop: 20 N',
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
    parser.add_argument('--headless', action='store_true', help='run all twelve scripted key checks')
    parser.add_argument('--plot-wrench', action='store_true',
                        help='show live force and torque plots in a second window')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.headless:
        report = headless_report()
        text = json.dumps(report, indent=2)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text+'\n')
        print(text)
        if not report['passed']:
            raise SystemExit(1)
    else:
        run_window(
            ServoDemo(args.linear_speed, np.deg2rad(args.angular_speed_deg)),
            plot_wrench=args.plot_wrench,
        )


if __name__ == '__main__':
    main()
