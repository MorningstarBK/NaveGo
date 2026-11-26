"""Attitude helpers (cf. NaveGo conversions/)."""
import numpy as np


def skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def euler2dcm(roll, pitch, yaw):
    """Body-to-NED DCM C_nb from ZYX Euler angles."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cp * cy, -cr * sy + sr * sp * cy, sr * sy + cr * sp * cy],
        [cp * sy, cr * cy + sr * sp * sy, -sr * cy + cr * sp * sy],
        [-sp, sr * cp, cr * cp],
    ])


def dcm2euler(C):
    """ZYX Euler angles (roll, pitch, yaw) from C_nb."""
    roll = np.arctan2(C[2, 1], C[2, 2])
    pitch = -np.arcsin(np.clip(C[2, 0], -1.0, 1.0))
    yaw = np.arctan2(C[1, 0], C[0, 0])
    return np.array([roll, pitch, yaw])


def rotvec2dcm(r):
    """Rodrigues formula: exp([r x])."""
    a = np.linalg.norm(r)
    S = skew(r)
    if a < 1e-10:
        return np.eye(3) + S + 0.5 * S @ S
    return np.eye(3) + np.sin(a) / a * S + (1.0 - np.cos(a)) / a**2 * S @ S


def dcm2rotvec(C):
    """Inverse of rotvec2dcm (matrix logarithm of a rotation)."""
    v = np.array([C[2, 1] - C[1, 2], C[0, 2] - C[2, 0], C[1, 0] - C[0, 1]])
    c = np.clip((np.trace(C) - 1.0) / 2.0, -1.0, 1.0)
    a = np.arccos(c)
    if a < 1e-8:
        return 0.5 * v
    return a / (2.0 * np.sin(a)) * v


def orthonormalize(C):
    U, _, Vt = np.linalg.svd(C)
    return U @ Vt


def wrap_angle(a):
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi
