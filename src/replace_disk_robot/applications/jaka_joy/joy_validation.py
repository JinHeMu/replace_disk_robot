"""Deterministic Xbox samples through the production node and MuJoCo dynamics."""
from copy import deepcopy
from contextlib import redirect_stdout
import io
import numpy as np

from replace_disk_robot.core.rotation import rotation_matrix, rotation_vector_from_matrix
from replace_disk_robot.control import JoySample
from .joy_simulation import make_sim_app


def validate(args):
    results = []
    apps = []

    def create(frame="base", admittance=True, linear_speed=None):
        profile = deepcopy(args)
        profile.command_frame, profile.admittance = frame, admittance
        if linear_speed is not None:
            profile.linear_speed = linear_speed
        profile.log_dir = None
        profile.dry_run = False
        app = make_sim_app(profile)
        apps.append(app)
        app._next_print = float("inf")
        app.initialize()
        app.validation_time = 0.
        app.max_contacts = 0
        step(app)
        return app

    def step(app, axes=(0.,)*6, buttons=(), *, connected=True, count=1):
        for _ in range(count):
            t = app.validation_time
            app.apply_sample(JoySample(t, tuple(axes), frozenset(buttons), connected), t)
            app.tick(t, app.client.dt)
            app.client.step()
            app.max_contacts = max(app.max_contacts, app.client.data.ncon)
            app.validation_time += app.client.dt

    def pose(app):
        base, tcp = app.client.data.body("jaka_base_link"), app.client.data.body("tool0")
        rot = base.xmat.reshape(3, 3).T
        return rot @ (tcp.xpos - base.xpos), rot @ tcp.xmat.reshape(3, 3)

    def check(name, passed, **metrics):
        results.append(dict(name=name, passed=bool(passed), **metrics))

    def move(frame, axis, sign=1., *, buttons=("lb",), admittance=True, linear_speed=None):
        app = create(frame, admittance, linear_speed)
        start, r0 = pose(app)
        axes = np.zeros(6)
        extra = set(buttons)
        if axis == "insert":
            axes[5 if sign > 0 else 4] = abs(sign)
        elif axis == "yaw":
            extra.add("dpleft" if sign > 0 else "dpright")
        else:
            index, polarity = {"horizontal": (0, 1.), "vertical": (1, -1.),
                               "roll": (2, -1.), "pitch": (3, 1.)}[axis]
            axes[index] = polarity * sign
        step(app, axes, extra, count=125)
        step(app, buttons=("lb",), count=75)
        end, r1 = pose(app)
        delta, angle = end - start, rotation_vector_from_matrix(r0.T @ r1)
        if axis == "insert":
            direction = r0[:, 2] * np.sign(sign)
        elif axis == "horizontal":
            direction = (np.array([1., 0., 0.]) if frame == "base" else -r0[:, 1]) * np.sign(sign)
        elif axis == "vertical":
            direction = (np.array([0., 0., 1.]) if frame == "base" else r0[:, 0]) * np.sign(sign)
        else:
            direction = np.eye(3)[{"roll": 0, "pitch": 1, "yaw": 2}[axis]] * np.sign(sign)
            if frame == "base":
                direction = r0.T @ direction
        translation = axis in ("horizontal", "vertical", "insert")
        projection = float((delta if translation else angle) @ direction)
        limits = app.kinematics.joint_limits_rad
        q = app.client.robot.arm_position()
        in_limits = bool(np.all(q >= limits[0]) and np.all(q <= limits[1]))
        check(f"{frame}_{axis}_{'positive' if sign > 0 else 'negative'}_"
              f"{'admittance' if admittance else 'servo'}_{abs(sign):g}_"
              f"{'precision' if 'rb' in buttons else 'normal'}_{app.args.linear_speed:g}m_s",
              projection > (.0004 if translation else .003) and app.servo.fault is None
              and app.max_contacts == 0 and in_limits,
              displacement_m=delta.tolist(), rotation_tcp_rad=angle.tolist(),
              projection=projection, fault=app.servo.fault, max_contacts=app.max_contacts)
        return app, projection

    with redirect_stdout(io.StringIO()):
        try:
            for frame in ("base", "tool"):
                for axis in ("horizontal", "vertical", "insert", "roll", "pitch", "yaw"):
                    for sign in (1., -1.):
                        move(frame, axis, sign)
            # Exercise direct velocity servo as well as the default admittance route.
            move("base", "horizontal", admittance=False)
            move("tool", "pitch", admittance=False)
            # Compare physical speed scaling below actuator/pose saturation.
            # The configured 0.2 m/s request can exceed the 0.05 m/s pose limit.
            full, full_distance = move("base", "horizontal", linear_speed=.002)
            half, half_distance = move("base", "horizontal", .575, linear_speed=.002)
            fine, fine_distance = move("base", "horizontal", buttons=("lb", "rb"), linear_speed=.002)
            check("proportional_half_deflection", .4 < half_distance / full_distance < .6,
                  ratio=half_distance / full_distance)
            check("rb_precision_speed", .18 < fine_distance / full_distance < .32,
                  ratio=fine_distance / full_distance)

            app = create()
            start, _ = pose(app)
            step(app, (1., 0., 0., 0., 0., 0.), count=100)
            check("no_lb_no_motion", np.linalg.norm(pose(app)[0] - start) < 1e-5)
            step(app)
            step(app, (.1, -.1, .1, -.1, .1, .1), ("lb",), count=100)
            check("deadzone_no_motion", np.linalg.norm(pose(app)[0] - start) < 1e-5)

            step(full, (1., 0., 0., 0., 0., 0.))  # release LB with stick still deflected
            held = full.servo.target.position_rad.copy()
            step(full, (1., 0., 0., 0., 0., 0.), count=50)
            check("lb_release_freezes_target", np.array_equal(held, full.servo.target.position_rad)
                  and not full._active)
            step(full, (1., 0., 0., 0., 0., 0.), ("lb",), count=20)
            check("cannot_rearm_until_neutral", not full._active
                  and np.array_equal(held, full.servo.target.position_rad))
            step(full)
            step(full, buttons=("lb",))
            check("lb_rearm_after_release_and_neutral", full._active and not full.servo.fault)
            step(full, buttons=("lb", "b"))
            step(full)
            step(full, buttons=("lb",))
            check("b_stop_latches", full.resume_required and not full._active)
            step(full, buttons=("a",))
            step(full, buttons=("lb",))
            check("a_reanchors_and_resumes", full._active and not full.servo.fault)
            step(full, connected=False)
            step(full, buttons=("lb",))
            check("disconnect_latches", full.resume_required and not full._active)
            step(full, buttons=("a",))
            step(full, buttons=("lb",))
            t = full.validation_time + full.args.joy_timeout_s + .001
            full.tick(t, full.client.dt)
            check("stale_input_stops", full.resume_required and not full._active)

            # Actual external force in MuJoCo, including sensor response and gravity
            # compensation. Express load and displacement in the same base frame.
            app = create()
            step(app, buttons=("lb",))
            start, r0 = pose(app)
            base_rot = app.client.data.body("jaka_base_link").xmat.reshape(3, 3).copy()
            body = app.client.data.body("tool0")
            app.client.data.xfrc_applied[body.id, :3] = base_rot @ np.array([0., 0., -3.])
            step(app, buttons=("lb",), count=250)
            loaded, _ = pose(app)
            force_tcp = app.last_wrench.force_n.copy()
            force_base = pose(app)[1] @ force_tcp
            check("physical_force_admittance_direction", loaded[2] - start[2] < -.001
                  and force_base[2] < -2. and not app.servo.fault,
                  displacement_m=(loaded-start).tolist(), force_base_n=force_base.tolist())
            app.client.data.xfrc_applied[:] = 0.
            step(app, buttons=("lb",), count=5000)
            residual = float(np.linalg.norm(pose(app)[0] - start))
            check("physical_force_release", residual < .001 and not app.servo.fault,
                  residual_m=residual)
            app.client.data.xfrc_applied[body.id, :3] = base_rot @ np.array([0., 0., -30.])
            step(app, buttons=("lb",), count=30)
            check("force_limit_stops", app.servo.fault == "force_limit" and app.resume_required,
                  fault=app.servo.fault)
            step(app, buttons=("a",))
            check("cannot_resume_under_load", app.servo.fault == "force_limit" and app.resume_required)
            app.client.data.xfrc_applied[:] = 0.
            step(app, count=100)
            step(app, buttons=("a",))
            step(app, buttons=("lb",))
            check("force_release_then_a_recovers", not app.servo.fault and app._active)

            # Reproduce reference backlog at the user's current 0.2 m/s speed.
            # LB remains held throughout centering and the subsequent restart.
            for frame in ("base", "tool"):
                for axis in (0, 1):
                    for sign in (-1., 1.):
                        app = create(frame, linear_speed=.2)
                        axes = np.zeros(6)
                        axes[axis] = sign
                        step(app, axes, ("lb",), count=125)
                        released, _ = pose(app)
                        backlog = float(np.linalg.norm(app.motion.nominal_pose.position_m - released))
                        maximum = 0.
                        for _ in range(500):
                            step(app, buttons=("lb",))
                            maximum = max(maximum, float(np.linalg.norm(pose(app)[0] - released)))
                        stopped = pose(app)[0].copy()
                        step(app, axes, ("lb",), count=25)
                        restart_distance = float(np.linalg.norm(pose(app)[0] - stopped))
                        check(f"center_left_axis{axis}_{sign:g}_{frame}_lb_held",
                              maximum < .001 and restart_distance > .002
                              and app.servo.fault is None and app._active,
                              speed_request_m_s=.2, backlog_at_release_m=backlog,
                              max_post_center_travel_m=maximum,
                              restart_distance_m=restart_distance, fault=app.servo.fault)
        finally:
            for app in apps:
                app.shutdown()
    return dict(passed=all(test["passed"] for test in results), tests=results,
                backend="MuJoCo JAKA", keyframe=args.keyframe,
                configured_linear_speed_m_s=args.linear_speed,
                configured_angular_speed_deg_s=args.angular_speed_deg,
                scope="Synthetic SDL-normalized samples through JakaJoyServo, measured MuJoCo "
                      "F/T, admittance, CartesianServo, force gate and position actuators. "
                      "External-force checks use physical xfrc_applied. Not a physical robot test.")
