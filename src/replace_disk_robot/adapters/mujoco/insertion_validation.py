"""Bounded position-driven fixtures for scene QA, NOT an insertion controller."""
from __future__ import annotations

import mujoco
import numpy as np

from .interfaces import MujocoWristFTAdapter
from .model import load_model, reset_home


def solve_center(model, seed, position, scratch=None):
    """Local numerical fixture IK; keep the drive aligned with world XYZ."""
    data = scratch if scratch is not None else mujoco.MjData(model)
    if scratch is None:
        reset_home(model, data)
    data.qpos[:6] = seed
    site = model.site('drive_center').id
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    for _ in range(100):
        mujoco.mj_forward(model, data)
        rotation = data.site_xmat[site].reshape(3, 3)
        error_r = sum((np.cross(rotation[:, i], np.eye(3)[:, i]) for i in range(3))) / 2
        error = np.r_[position-data.site_xpos[site], error_r]
        if np.linalg.norm(error) < 1e-10:
            return data.qpos[:6].copy()
        mujoco.mj_jacSite(model, data, jp, jr, site)
        jac = np.vstack((jp[:, :6], jr[:, :6]))
        dq = jac.T @ np.linalg.solve(jac@jac.T + 1e-8*np.eye(6), error)
        data.qpos[:6] += np.clip(dq, -.05, .05)
    raise RuntimeError('Local validation pose is unreachable')


def command_fixture(model, data, q):
    # Position-servo equilibrium feedforward, including gravity. No wrench feedback.
    data.ctrl[:6] = q + data.qfrc_bias[:6] / model.actuator_gainprm[:6, 0]


def contacts(model, data):
    return [dict(geom1=model.geom(c.geom1).name or str(c.geom1),
                 geom2=model.geom(c.geom2).name or str(c.geom2),
                 penetration_m=max(0., -float(c.dist))) for c in data.contact]


def observation(model, data, ft):
    mujoco.mj_forward(model, data)
    entry = data.site('socket_entry')
    relative = entry.xmat.reshape(3, 3).T @ (data.site('drive_front').xpos-entry.xpos)
    return dict(raw_wrench=ft.raw().tolist(), tared_wrench=ft.wrench().tolist(),
                frame='wrist_ft_site', insertion_depth_m=float(relative[0]),
                lateral_offset_m=relative[1:].tolist(), contacts=contacts(model, data))


def contact_reaction_at_sensor(model, data):
    """Opposite of environment-on-drive wrench, world axes at wrist sensor origin."""
    wrench = np.zeros(6)
    disk = model.geom('drive_collision').id
    origin = data.site('wrist_ft_site').xpos
    for i, c in enumerate(data.contact):
        if disk not in (c.geom1, c.geom2):
            continue
        local = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, local)
        rotation = c.frame.reshape(3, 3).T
        sign = 1 if c.geom1 == disk else -1
        force = sign * rotation @ local[:3]
        torque = sign * rotation @ local[3:] + np.cross(c.pos-origin, force)
        wrench += np.r_[force, torque]
    return wrench


def run_probe(kind='axial', timestep=.001):
    if kind not in ('axial', 'side'):
        raise ValueError('kind must be axial or side')
    model, data = load_model()
    model.opt.timestep = timestep
    reset_home(model, data)
    ft = MujocoWristFTAdapter(model, data)
    q = data.qpos[:6].copy()
    for _ in range(round(1/timestep)):
        command_fixture(model, data, q)
        mujoco.mj_step(model, data)
    ft.tare()
    baseline = observation(model, data, ft)
    initial = data.site('drive_center').xpos.copy()
    # Off-center entrance blocking, or a lateral nudge after entering the channel.
    offsets = ([np.array([0., .001, 0.]), np.array([.01005, .001, 0.])]
               if kind == 'axial' else
               [np.array([.015, 0., 0.]), np.array([.015, .00045, 0.])])
    scratch = mujoco.MjData(model)
    reset_home(model, scratch)
    start = initial.copy()
    peak_penetration = 0.
    peak_force = 0.
    all_pairs = set()
    for offset in offsets:
        end = initial + offset
        duration = max(.5, np.linalg.norm(end-start)/.005)
        steps = round(duration/timestep)
        for i in range(steps):
            t = (i+1)/steps
            blend = t*t*t*(10+t*(-15+6*t))
            q = solve_center(model, q, start + blend*(end-start), scratch)
            command_fixture(model, data, q)
            mujoco.mj_step(model, data)
            peak_force = max(peak_force, float(np.linalg.norm(ft.wrench()[:3])))
            for c in contacts(model, data):
                peak_penetration = max(peak_penetration, c['penetration_m'])
                all_pairs.add((c['geom1'], c['geom2']))
            if peak_force > 10 or np.linalg.norm(ft.wrench()[3:]) > 1:
                raise RuntimeError('Validation motion exceeded 10 N / 1 Nm bound')
        start = end
    samples = []
    expected_samples = []
    for _ in range(round(.5/timestep)):
        command_fixture(model, data, q)
        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        samples.append(ft.wrench())
        expected_samples.append(contact_reaction_at_sensor(model, data))
        peak_force = max(peak_force, float(np.linalg.norm(ft.wrench()[:3])))
        if peak_force > 10 or np.linalg.norm(ft.wrench()[3:]) > 1:
            raise RuntimeError('Validation hold exceeded 10 N / 1 Nm bound')
        for c in contacts(model, data):
            peak_penetration = max(peak_penetration, c['penetration_m'])
            all_pairs.add((c['geom1'], c['geom2']))
    result = observation(model, data, ft)
    mean = np.mean(samples[-round(.1/timestep):], axis=0)
    # Sensor reports the parent-on-child load, opposite the external contact load.
    world_force = data.site('wrist_ft_site').xmat.reshape(3,3) @ mean[:3]
    expected = np.mean(expected_samples[-round(.1/timestep):], axis=0)
    sensor_rotation = data.site('wrist_ft_site').xmat.reshape(3, 3)
    world_wrench = np.r_[sensor_rotation @ mean[:3], sensor_rotation @ mean[3:]]
    wrench_consistent = bool(np.allclose(world_wrench, expected, rtol=.1, atol=.03))
    result.update(expected_world_contact_reaction=expected.tolist(),
                  measured_world_wrench=world_wrench.tolist(),
                  contact_wrench_consistent=wrench_consistent, kind=kind, timestep_s=timestep, baseline=baseline,
                  peak_penetration_m=peak_penetration, peak_force_n=peak_force,
                  mean_tared_wrench=mean.tolist(), mean_world_force_n=world_force.tolist(),
                  contact_pairs=sorted(all_pairs))
    result['passed'] = bool(peak_penetration <= .00005 and
                            np.linalg.norm(mean[:3]) > .05 and wrench_consistent and
                            world_force[0 if kind == 'axial' else 1] > .05 and
                            all_pairs and all('drive_collision' in pair and
                                             any(n.startswith('socket_') for n in pair)
                                             for pair in all_pairs))
    return result


def run_friction_probe(timestep=.0005, friction=.6, force_limit_n=15., torque_limit_nm=2.):
    """Position-driven insertion/withdrawal QA, with real passive liner contact.

    Override liner friction only for the zero-friction comparison. The arm,
    aperture, disk dimensions, springs and collision masks remain identical.

    ``force_limit_n`` / ``torque_limit_nm`` bound the run so a parameter study
    can fail loudly instead of producing an invalid integration. They are not
    controller limits. The default fixture bound permits entry/reversal peaks
    above the 7.5 N sliding-load target.
    """
    model, data = load_model()
    model.opt.timestep = timestep
    # A stiffer position-driven test fixture keeps the narrow disk aligned
    # under the calibrated load. This does not change the scene arm gains,
    # CartesianServo, or inject any synthetic contact force.
    model.actuator_gainprm[:6, 0] *= 4
    model.actuator_biasprm[:6, 1] *= 4
    model.actuator_biasprm[:6, 2] *= 2
    for side in ('left', 'right'):
        model.geom('socket_liner_' + side).friction[0] = friction
    reset_home(model, data)
    ft = MujocoWristFTAdapter(model, data)
    q = data.qpos[:6].copy()
    for _ in range(round(.5/timestep)):
        command_fixture(model, data, q)
        mujoco.mj_step(model, data)
    ft.tare()
    initial = data.site('drive_center').xpos.copy()
    scratch = mujoco.MjData(model)
    reset_home(model, scratch)
    samples = {'insert': [], 'withdraw': []}
    max_penetration = 0.
    pairs = set()
    peak_force = 0.
    peak_torque = 0.
    start = 0.
    for phase, end, duration in [('insert', .07, 21.), ('withdraw', .04, 9.)]:
        for i in range(round(duration/timestep)):
            t = (i+1)*timestep/duration
            blend = t*t*(3-2*t)
            q = solve_center(model, q, initial + [start+(end-start)*blend, 0, 0], scratch)
            command_fixture(model, data, q)
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            if not np.isfinite(np.r_[data.qpos, data.qvel, data.sensordata]).all():
                raise RuntimeError('Non-finite friction probe state')
            force = data.site('wrist_ft_site').xmat.reshape(3,3) @ ft.wrench()[:3]
            peak_force = max(peak_force, float(np.linalg.norm(force)))
            peak_torque = max(peak_torque, float(np.linalg.norm(ft.wrench()[3:])))
            if peak_force > force_limit_n or np.linalg.norm(ft.wrench()[3:]) > torque_limit_nm:
                raise RuntimeError(
                    f'Friction probe exceeded {force_limit_n:g} N / {torque_limit_nm:g} Nm: '
                    f'phase={phase}, t={t}, force={force}, torque={ft.wrench()[3:]}, '
                    f'contacts={contacts(model, data)}'
                )
            for c in contacts(model, data):
                pairs.add((c['geom1'], c['geom2']))
                max_penetration = max(max_penetration, c['penetration_m'])
            depth = data.site('drive_front').xpos[0]-data.site('socket_entry').xpos[0]
            if .4 < t < .8 and depth > .025:
                reaction = contact_reaction_at_sensor(model, data)
                samples[phase].append([depth, force[0], reaction[0]])
        start = end
    result = dict(timestep_s=timestep, friction_coefficient=friction,
                  peak_torque_nm=peak_torque, fixture_position_gain_scale=4,
                  fixture_velocity_gain_scale=2, nominal_translation_speed_m_s=.0033333333333333335,
                  peak_force_n=peak_force, peak_penetration_m=max_penetration,
                  contact_pairs=sorted(pairs), final=observation(model, data, ft))
    for phase, values in samples.items():
        mean = np.mean(values, axis=0)
        result[phase] = dict(samples=len(values), mean_depth_m=float(mean[0]),
                             mean_sensor_axial_n=float(mean[1]), mean_contact_axial_n=float(mean[2]),
                             sensor_axial_std_n=float(np.std(np.asarray(values)[:, 1])))
    result['passed'] = bool(max_penetration < .00005 and all(samples.values()) and
        abs(result['final']['insertion_depth_m']-.03) < .003 and
        all('drive_collision' in pair and any(n.startswith('socket_') for n in pair) for pair in pairs) and
        all(abs(result[k]['mean_sensor_axial_n']-result[k]['mean_contact_axial_n']) < .15 for k in samples) and
        (friction == 0 or (result['insert']['mean_sensor_axial_n'] > .3 and
                           result['withdraw']['mean_sensor_axial_n'] < -.3)))
    return result
