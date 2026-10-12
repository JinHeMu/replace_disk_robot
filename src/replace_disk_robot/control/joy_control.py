"""Pure joystick state and proportional jog mapping, parallel to KeyControl.

Device adapters supply normalized JoySample values. Robot applications own
frame transforms, reference integration, stopping and command dispatch.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..core.types import CartesianJog


@dataclass(frozen=True)
class JoySample:
    timestamp_s: float
    # SDL normalized axes: left X/Y, right X/Y, LT, RT.
    axes: tuple[float, ...] = (0., 0., 0., 0., 0., 0.)
    buttons: frozenset[str] = frozenset()
    connected: bool = True


class JoyControl:
    def __init__(self, linear_speed_m_s=.01, angular_speed_rad_s=np.deg2rad(5),
                 base_frame="world", *, deadzone=.15, precision_scale=.25, timeout_s=.15):
        values = [linear_speed_m_s, angular_speed_rad_s, deadzone, precision_scale, timeout_s]
        if not np.isfinite(values).all():
            raise ValueError("joy settings must be finite")
        if linear_speed_m_s <= 0 or angular_speed_rad_s <= 0 or timeout_s <= 0:
            raise ValueError("joy speeds and timeout must be positive")
        if not 0 <= deadzone < 1 or not 0 < precision_scale <= 1:
            raise ValueError("deadzone must be in [0,1); precision scale in (0,1]")
        if not base_frame:
            raise ValueError("base_frame must not be empty")
        self.linear_speed, self.angular_speed = linear_speed_m_s, angular_speed_rad_s
        self.base_frame = base_frame
        self.deadzone, self.precision_scale, self.timeout_s = deadzone, precision_scale, timeout_s
        self.clear()

    def update(self, sample: JoySample) -> None:
        """Replace the input state; disconnected/invalid samples yield zero jog."""
        if not isinstance(sample, JoySample):
            raise TypeError("sample must be a JoySample")
        self.sample = sample

    def clear(self) -> None:
        """Discard previous axes/buttons so they cannot be replayed."""
        self.sample = JoySample(0., connected=False)

    def valid(self, now):
        sample = self.sample
        axes = np.asarray(sample.axes, dtype=float)
        return bool(sample.connected and axes.shape == (6,) and np.isfinite(axes).all()
                    and (np.abs(axes[:4]) <= 1).all()
                    and ((axes[4:] >= 0) & (axes[4:] <= 1)).all()
                    and np.isfinite(sample.timestamp_s) and np.isfinite(now)
                    and 0 <= now - sample.timestamp_s <= self.timeout_s)

    def shape(self, value):
        return float(np.sign(value) * max(0., abs(value) - self.deadzone)
                     / (1. - self.deadzone))

    def neutral(self):
        sample = self.sample
        return (all(abs(v) <= self.deadzone for v in sample.axes)
                and not {"dpleft", "dpright"}.intersection(sample.buttons))

    def command(self, now_s: float, translation_axes=None) -> CartesianJog:
        """Return a velocity in ``base_frame``; LB enables, RB scales precision.

        Columns of translation_axes are directions for left-stick right,
        left-stick up and RT-minus-LT, expressed in base_frame. By default
        these are -Y, +X and +Z. Angular axes are roll, pitch and yaw in that
        same frame. The application transforms the jog for its robot servo.
        """
        sample = self.sample
        if not self.valid(now_s) or "lb" not in sample.buttons:
            return CartesianJog(self.base_frame, np.zeros(3), np.zeros(3))
        lx, ly, rx, ry, lt, rt = (self.shape(v) for v in sample.axes)
        axes = np.asarray(translation_axes if translation_axes is not None else
                          [[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]], dtype=float)
        if axes.shape != (3, 3) or not np.isfinite(axes).all():
            raise ValueError("translation_axes must be a finite 3x3 matrix")
        linear = axes @ np.array([lx, -ly, rt - lt])
        yaw = float("dpleft" in sample.buttons) - float("dpright" in sample.buttons)
        angular = np.array([-rx, ry, yaw])
        linear /= max(1., float(np.linalg.norm(linear)))
        angular /= max(1., float(np.linalg.norm(angular)))
        scale = self.precision_scale if "rb" in sample.buttons else 1.
        return CartesianJog(self.base_frame, linear * self.linear_speed * scale,
                            angular * self.angular_speed * scale)
