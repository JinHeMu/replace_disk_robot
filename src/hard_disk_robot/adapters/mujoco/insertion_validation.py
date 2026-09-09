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
