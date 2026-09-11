"""Simple force/torque threshold gate used to validate the safety data path."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass
class ForceLimitGuard:
    """Trip when force norm or torque norm exceeds its configured limit.

    Set ``torque_limit_nm`` to ``numpy.inf`` to disable torque tripping while
    keeping the force threshold active.
    """
    force_limit_n: float = 10.0
    torque_limit_nm: float = 1.0

    def filter_arm_target(
        self,
        wrench: ArrayLike,
        current_q: ArrayLike,
        requested_q: ArrayLike,
    ) -> tuple[NDArray[np.float64], bool]:
        wrench_array = np.asarray(wrench, dtype=float)
        current = np.asarray(current_q, dtype=float)
        requested = np.asarray(requested_q, dtype=float)
        if wrench_array.shape != (6,) or current.shape != (6,) or requested.shape != (6,):
            raise ValueError("expected wrench(6), current_q(6), requested_q(6)")
        tripped = (
            np.linalg.norm(wrench_array[:3]) > self.force_limit_n
            or np.linalg.norm(wrench_array[3:]) > self.torque_limit_nm
        )
        return (current.copy() if tripped else requested.copy()), bool(tripped)
