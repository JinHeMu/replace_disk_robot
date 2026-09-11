"""Small NumPy-only wxyz quaternion utilities."""
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
