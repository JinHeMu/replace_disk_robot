"""Backend-neutral Cartesian servo; no keyboard, renderer or simulator imports.

Two command flavours share the same damped differential IK, joint limits and
optional collision checks:

* :meth:`CartesianServo.submit` accepts a Cartesian velocity (the original
  keyboard interface);
* :meth:`CartesianServo.submit_pose` accepts a Cartesian pose and converts the
  pose error to a bounded tracking twist before solving IK.

Both APIs must be refreshed within ``command_timeout_s``; otherwise the servo
holds its last joint target.
"""

from dataclasses import dataclass

import numpy as np

from ..core.ports import KinematicsPort, CollisionCheckerPort
from ..core.types import CartesianJog, JointState, Pose, _vector
from ..core.rotation import rotation_matrix, rotation_vector_from_matrix


@dataclass(frozen=True)
class ServoConfig:
    # Velocity-command limits (KeyControl and other jog clients).
    linear_speed_m_s: float = .01
    angular_speed_rad_s: float = np.deg2rad(5)
    joint_speed_rad_s: float = .2
    damping: float = .02
    command_timeout_s: float = .15
    max_dt_s: float = .05
    max_tracking_error_rad: float = .08
    collision_step_rad: float = .002
    # Pose-command tracking gains and limits.  Kept separate so enabling
    # compliant pose tracking does not silently change keyboard jog speeds.
    pose_linear_gain_s_inv: float = 8.0
    pose_angular_gain_s_inv: float = 8.0
    pose_linear_speed_m_s: float = .05
    pose_angular_speed_rad_s: float = np.deg2rad(20)
    pose_deadband_m: float = 1e-5
    pose_deadband_rad: float = 1e-5

    def __post_init__(self):
        if any(not np.isfinite(v) or v <= 0 for v in self.__dict__.values()):
            raise ValueError('Servo limits must be finite and positive')


class CartesianServo:
    """Damped differential IK with named joints and persistent position hold.

    Caller supplies monotonically increasing seconds, actual joint state, and
    explicitly refreshed commands. update returns a JointState for ArmPort.
    A keyboard, joystick or autonomous client can all use the same submit API.
    """
    def __init__(self, kinematics: KinematicsPort, joint_limits_rad,
                 config: ServoConfig | None = None,
                 collision_checker: CollisionCheckerPort | None = None):
        self.kinematics = kinematics
        self.config = config or ServoConfig()
        self.collision_checker = collision_checker
        self.names = tuple(kinematics.joint_names)
        if len(set(self.names)) != len(self.names) or len(self.names) != kinematics.dof:
            raise ValueError('Kinematics must declare unique named joints')
        self.lower = _vector(joint_limits_rad[0], len(self.names), 'lower joint limits')
        self.upper = _vector(joint_limits_rad[1], len(self.names), 'upper joint limits')
        if np.any(self.lower >= self.upper):
            raise ValueError('Invalid joint limits')
        self.target = None
        self.command = None
        self.command_time = None
        self.pose_command = None
        self.pose_command_time = None
        self.last_time = None
        self.status = 'not_initialized'
        self.fault = None

    def _ordered(self, state):
        if (len(state.names) != len(self.names) or
                len(set(state.names)) != len(self.names) or set(state.names) != set(self.names)):
            raise ValueError('Each kinematics joint must be supplied exactly once')
        return JointState(self.names, [state.position_rad[state.names.index(n)] for n in self.names])

    def _check_command_time(self, now_s: float) -> None:
        if not np.isfinite(now_s):
            raise ValueError('Command time must be finite')
        latest_command_time = max(
            (value for value in (self.command_time, self.pose_command_time)
             if value is not None),
            default=None,
        )
        if ((latest_command_time is not None and now_s < latest_command_time) or
                (self.last_time is not None and now_s < self.last_time)):
            raise ValueError('Command time cannot go backwards')

    def reset(self, measured: JointState):
        state = self._ordered(measured)
        if np.any(state.position_rad < self.lower) or np.any(state.position_rad > self.upper):
            raise ValueError('Measured joints outside limits')
        self.target = state
        self.command = None
        self.command_time = None
        self.pose_command = None
        self.pose_command_time = None
        self.last_time = None
        self.fault = None
        self.status = 'holding'

    def submit(self, command: CartesianJog, now_s: float):
        """Submit (or refresh) a Cartesian velocity command."""
        if not isinstance(command, CartesianJog):
            raise TypeError('command must be a CartesianJog')
        self._check_command_time(now_s)
        self.command, self.command_time = command, now_s
        self.pose_command, self.pose_command_time = None, None

    def submit_pose(self, target: Pose, now_s: float):
        """Submit (or refresh) a Cartesian pose to track.

        The pose is integrated by the servo at the configured
        ``pose_*_speed`` limits.  A typical admittance loop refreshes it every
        control tick so that the reference follows the filtered force offset.
        """
        if not isinstance(target, Pose):
            raise TypeError('target must be a Pose')
        self._check_command_time(now_s)
        self.pose_command, self.pose_command_time = target, now_s
        self.command, self.command_time = None, None

    def halt(self, measured: JointState, reason='halted'):
        self.target = self._ordered(measured)
        self.command = None
        self.command_time = None
        self.pose_command = None
        self.pose_command_time = None
        self.fault = reason
        self.status = reason
        return self.target

    def update(self, measured: JointState, dt_s: float, now_s: float) -> JointState:
        actual = self._ordered(measured)
        if self.target is None:
            self.reset(actual)
        if (not np.isfinite(now_s) or not np.isfinite(dt_s) or
                not 0 < dt_s <= self.config.max_dt_s or
                (self.last_time is not None and now_s < self.last_time)):
            raise ValueError('Invalid servo time or dt')
        self.last_time = now_s
        if self.fault:
            return self.target
        if np.max(np.abs(actual.position_rad-self.target.position_rad)) > self.config.max_tracking_error_rad:
            return self.halt(actual, 'tracking_error')

        if self.pose_command is not None:
            if (self.pose_command_time is None or
                    now_s-self.pose_command_time > self.config.command_timeout_s):
                self.status = 'command_timeout'
                return self.target
            if self.pose_command_time > now_s:
                raise ValueError('Command is timestamped in the future')
            actual_pose = self.kinematics.forward(actual)
            if self.pose_command.frame_id != actual_pose.frame_id:
                raise ValueError('Pose command and kinematics base frames differ')
            linear, angular = self._pose_tracking_twist(self.pose_command, actual_pose)
            if not np.any(linear) and not np.any(angular):
                self.status = 'holding'
                return self.target
            return self._step(linear, angular, actual, dt_s, 'tracking')

        if self.command is None or self.command_time is None or now_s-self.command_time > self.config.command_timeout_s:
            self.status = 'command_timeout' if self.command is not None else 'holding'
            return self.target
        if self.command_time > now_s:
            raise ValueError('Command is timestamped in the future')
        pose = self.kinematics.forward(self.target)
        if self.command.base_frame != pose.frame_id:
            raise ValueError('Command and kinematics base frames differ')
        linear = self._limit(self.command.linear_m_s, self.config.linear_speed_m_s)
        angular = self._limit(self.command.angular_rad_s, self.config.angular_speed_rad_s)
        if not np.any(linear) and not np.any(angular):
            self.status = 'holding'
            return self.target
        angular_world = rotation_matrix(pose.quaternion_wxyz) @ angular
        return self._step(linear, angular_world, actual, dt_s, 'moving')

    def _pose_tracking_twist(self, reference: Pose, actual: Pose):
        """Return a bounded world-frame tracking twist [v; omega]."""
        position_error = reference.position_m - actual.position_m
        rotation_error = (
            rotation_matrix(reference.quaternion_wxyz)
            @ rotation_matrix(actual.quaternion_wxyz).T
        )
        rotation_error_vector = rotation_vector_from_matrix(rotation_error)
        if np.linalg.norm(position_error) < self.config.pose_deadband_m:
            position_error = np.zeros(3)
        if np.linalg.norm(rotation_error_vector) < self.config.pose_deadband_rad:
            rotation_error_vector = np.zeros(3)
        linear = self._limit(
            self.config.pose_linear_gain_s_inv * position_error,
            self.config.pose_linear_speed_m_s,
        )
        angular = self._limit(
            self.config.pose_angular_gain_s_inv * rotation_error_vector,
            self.config.pose_angular_speed_rad_s,
        )
        return linear, angular

    def _step(self, linear_world, angular_world, actual: JointState, dt_s: float,
              status: str) -> JointState:
        """Solve one damped-IK step for a world-frame twist and apply limits."""
        twist = np.r_[linear_world, angular_world]
        jac = np.asarray(self.kinematics.jacobian(self.target))
        if jac.shape != (6, len(self.names)) or not np.isfinite(jac).all():
            raise ValueError('Expected a finite [linear; angular] 6-by-dof Jacobian')
        dq = jac.T @ np.linalg.solve(jac @ jac.T + self.config.damping**2*np.eye(6), twist) * dt_s
        speed_scale = min(1., self.config.joint_speed_rad_s*dt_s/max(np.max(np.abs(dq)), 1e-15))
        dq *= speed_scale
        q = self.target.position_rad
        if np.any(q+dq < self.lower) or np.any(q+dq > self.upper):
            self.status = 'joint_limit'
            return self.target
        candidate = JointState(self.names, q+dq)
        if self.collision_checker is not None:
            # Check both the command segment and actual-to-candidate tracking gap.
            for start in (q, actual.position_rad):
                count = max(1, int(np.ceil(np.max(np.abs(candidate.position_rad-start))/self.config.collision_step_rad)))
                for alpha in np.linspace(0., 1., count+1):
                    if not self.collision_checker.is_collision_free(JointState(self.names, start+alpha*(candidate.position_rad-start))):
                        self.status = 'collision_blocked'
                        return self.target
        self.target = candidate
        self.status = status
        return candidate

    @staticmethod
    def _limit(vector, limit):
        return vector * min(1., limit/max(np.linalg.norm(vector), 1e-15))
