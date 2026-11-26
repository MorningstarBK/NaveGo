"""Rotating IMU (rotation modulation) about the vehicle X axis, with a shaft encoder.

Frames: b = vehicle body, s = IMU sensor frame mounted on a turntable whose axis is b-X.
    C_bs = Rx(theta)          (sensor -> body)
    C_ns = C_nb @ C_bs        (sensor -> NED, what the INS/EKF estimates)
The encoder measures theta, so the vehicle attitude is recovered as
    C_nb = C_ns @ Rx(theta_enc)^T
"""
from dataclasses import dataclass, replace

import numpy as np

from ..attitude import dcm2euler, euler2dcm


def rot_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rotation_angle(t, rate_deg_s=36.0, mode="continuous", period=20.0):
    """Turntable angle theta(t) (rad).

    continuous: constant rate rotation.
    alternate:  rotate +360 deg then -360 deg every `period` s (smooth, sin^2 velocity profile).
    """
    if mode == "continuous":
        return np.deg2rad(rate_deg_s) * t
    if mode == "alternate":
        half = period / 2.0
        ph = np.mod(t, period)
        x = np.where(ph < half, ph / half, (ph - half) / half)
        prog = x - np.sin(2 * np.pi * x) / (2 * np.pi)          # 0 -> 1 smoothly
        return 2 * np.pi * np.where(ph < half, prog, 1.0 - prog)
    raise ValueError(mode)


def rotate_trajectory(ref, theta):
    """Sensor-frame trajectory: same position/velocity, attitude C_nb @ Rx(theta)."""
    C = np.einsum("kij,kjl->kil", ref.C_nb, np.array([rot_x(a) for a in theta]))
    euler = np.array([dcm2euler(c) for c in C])
    return replace(ref, C_nb=C, euler=euler)


@dataclass
class EncoderData:
    t: np.ndarray
    theta: np.ndarray     # measured angle (rad)
    params: dict


def simulate_encoder(t, theta, bits=16, noise_std_deg=0.002, offset_deg=0.0, seed=4):
    """Incremental/absolute encoder: quantized angle + noise + optional zero offset."""
    rng = np.random.default_rng(seed)
    lsb = 2 * np.pi / 2**bits
    meas = theta + np.deg2rad(offset_deg) + np.deg2rad(noise_std_deg) * rng.normal(size=len(theta))
    meas = np.round(meas / lsb) * lsb
    return EncoderData(t.copy(), meas, dict(bits=bits, lsb=lsb, noise_std_deg=noise_std_deg,
                                            offset_deg=offset_deg))


def vehicle_attitude(euler_sensor, theta_enc):
    """Vehicle roll/pitch/yaw (N,3) from sensor-frame Euler angles and encoder angle."""
    out = np.empty_like(euler_sensor)
    for k, (e, a) in enumerate(zip(euler_sensor, theta_enc)):
        out[k] = dcm2euler(euler2dcm(*e) @ rot_x(a).T)
    return out
