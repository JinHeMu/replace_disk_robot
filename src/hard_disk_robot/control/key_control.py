"""Pure key-state mapping. No IK, robot commands or window-system dependency."""
import numpy as np
from ..core.types import CartesianJog


# (base translation, intrinsic TCP rotation), with TCP X forward at init.
KEY_AXES = {
    'w': ([0,0,1], [0,0,0]), 's': ([0,0,-1], [0,0,0]),
    'a': ([0,1,0], [0,0,0]), 'd': ([0,-1,0], [0,0,0]),
    'r': ([1,0,0], [0,0,0]), 'f': ([-1,0,0], [0,0,0]),
    'q': ([0,0,0], [0,0,1]), 'e': ([0,0,0], [0,0,-1]),
    'up': ([0,0,0], [0,-1,0]), 'down': ([0,0,0], [0,1,0]),
    'left': ([0,0,0], [1,0,0]), 'right': ([0,0,0], [-1,0,0]),
}


class KeyControl:
    def __init__(self, linear_speed_m_s=.01, angular_speed_rad_s=np.deg2rad(5), base_frame='world'):
        if (not np.isfinite(linear_speed_m_s) or linear_speed_m_s <= 0 or
                not np.isfinite(angular_speed_rad_s) or angular_speed_rad_s <= 0):
            raise ValueError('Key speeds must be finite and positive')
        if not base_frame:
            raise ValueError('base_frame must not be empty')
        self.linear_speed = linear_speed_m_s
        self.angular_speed = angular_speed_rad_s
        self.base_frame = base_frame
        self.pressed = set()

    def press(self, key: str):
        key = key.lower()
        if key in KEY_AXES:
            self.pressed.add(key)

    def release(self, key: str):
        self.pressed.discard(key.lower())

    def clear(self):
        """Call on focus loss, stop or shutdown."""
        self.pressed.clear()

    def command(self) -> CartesianJog:
        linear, angular = np.zeros(3), np.zeros(3)
        for key in sorted(self.pressed):
            translation, rotation = KEY_AXES[key]
            linear += translation
            angular += rotation
        # Diagonal key combinations do not increase either speed norm.
        linear /= max(1., np.linalg.norm(linear))
        angular /= max(1., np.linalg.norm(angular))
        return CartesianJog(self.base_frame, linear*self.linear_speed, angular*self.angular_speed)
