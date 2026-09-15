"""Small NumPy-only wxyz quaternion and rotation-vector utilities."""
import numpy as np


def rotation_matrix(q):
    w, x, y, z = np.asarray(q, dtype=float)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def multiply(a, b):
    w, v = a[0], np.asarray(a[1:])
    s, u = b[0], np.asarray(b[1:])
    return np.r_[w*s-v@u, w*u+s*v+np.cross(v,u)]


def conjugate(q):
    return np.asarray(q)*[1,-1,-1,-1]


def quaternion_from_rotation_vector(rotation_vector):
    """Return a wxyz quaternion for an axis-angle vector in radians."""
    vector = np.asarray(rotation_vector, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError('rotation_vector must be a finite 3-vector')
    theta = float(np.linalg.norm(vector))
    if theta < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    half = 0.5 * theta
    return np.r_[np.cos(half), vector / theta * np.sin(half)]


def rotation_matrix_from_rotation_vector(rotation_vector):
    """Return the rotation matrix for an axis-angle vector in radians."""
    return rotation_matrix(quaternion_from_rotation_vector(rotation_vector))


def quaternion_from_rotation_matrix(matrix):
    """Return a normalized wxyz quaternion for a 3x3 rotation matrix.

    Uses the branch-stable Shepperd method.  ``w`` is chosen non-negative so
    that the quaternion behaves continuously for rotations below 180 degrees;
    the sign convention is irrelevant for the rotation itself.
    """
    m = np.asarray(matrix, dtype=float)
    if m.shape != (3, 3) or not np.all(np.isfinite(m)):
        raise ValueError('matrix must be a finite 3x3 rotation matrix')
    trace = float(np.trace(m))
    if trace > 0.0:
        s = 2.0 * np.sqrt(max(1e-15, trace + 1.0))
        q = np.array([
            0.25 * s,
            (m[2, 1] - m[1, 2]) / s,
            (m[0, 2] - m[2, 0]) / s,
            (m[1, 0] - m[0, 1]) / s,
        ])
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(max(1e-15, 1.0 + m[0, 0] - m[1, 1] - m[2, 2]))
        q = np.array([
            (m[2, 1] - m[1, 2]) / s,
            0.25 * s,
            (m[0, 1] + m[1, 0]) / s,
            (m[0, 2] + m[2, 0]) / s,
        ])
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(max(1e-15, 1.0 + m[1, 1] - m[0, 0] - m[2, 2]))
        q = np.array([
            (m[0, 2] - m[2, 0]) / s,
            (m[0, 1] + m[1, 0]) / s,
            0.25 * s,
            (m[1, 2] + m[2, 1]) / s,
        ])
    else:
        s = 2.0 * np.sqrt(max(1e-15, 1.0 + m[2, 2] - m[0, 0] - m[1, 1]))
        q = np.array([
            (m[1, 0] - m[0, 1]) / s,
            (m[0, 2] + m[2, 0]) / s,
            (m[1, 2] + m[2, 1]) / s,
            0.25 * s,
        ])
    if q[0] < 0.0:
        q = -q
    norm = float(np.linalg.norm(q))
    if norm < 1e-12 or not np.all(np.isfinite(q)):
        raise ValueError('matrix does not describe a valid rotation')
    return q / norm


def rotation_vector_from_matrix(matrix):
    """Return the axis-angle vector (rad) for a 3x3 rotation matrix."""
    m = np.asarray(matrix, dtype=float)
    if m.shape != (3, 3) or not np.all(np.isfinite(m)):
        raise ValueError('matrix must be a finite 3x3 rotation matrix')
    trace = float(np.trace(m))
    cosine = float(np.clip(0.5 * (trace - 1.0), -1.0, 1.0))
    angle = float(np.arccos(cosine))
    if angle < 1e-9:
        return 0.5 * np.array([
            m[2, 1] - m[1, 2],
            m[0, 2] - m[2, 0],
            m[1, 0] - m[0, 1],
        ])
    if np.pi - angle < 1e-6:
        diagonal = np.diag(m)
        axis_index = int(np.argmax(diagonal))
        axis = np.zeros(3)
        axis[axis_index] = np.sqrt(max(0.0, 0.5 * (diagonal[axis_index] + 1.0)))
        if axis[axis_index] < 1e-12:
            return np.zeros(3)
        if axis_index == 0:
            axis[1] = (m[0, 1] + m[1, 0]) / (4.0 * axis[0])
            axis[2] = (m[0, 2] + m[2, 0]) / (4.0 * axis[0])
        elif axis_index == 1:
            axis[0] = (m[0, 1] + m[1, 0]) / (4.0 * axis[1])
            axis[2] = (m[1, 2] + m[2, 1]) / (4.0 * axis[1])
        else:
            axis[0] = (m[0, 2] + m[2, 0]) / (4.0 * axis[2])
            axis[1] = (m[1, 2] + m[2, 1]) / (4.0 * axis[2])
        axis = axis / np.linalg.norm(axis)
        return angle * axis
    vector = np.array([
        m[2, 1] - m[1, 2],
        m[0, 2] - m[2, 0],
        m[1, 0] - m[0, 1],
    ])
    return vector * (angle / (2.0 * np.sin(angle)))
