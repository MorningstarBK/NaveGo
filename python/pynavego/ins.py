"""Strapdown INS mechanization in NED (cf. NaveGo ins/att_update.m, vel_update.m, pos_update.m)."""
from dataclasses import dataclass, field

import numpy as np

from .attitude import rotvec2dcm
from .earth import earth_rate, gravity, radius, transport_rate


@dataclass
class NavState:
    lat: float                       # rad
    lon: float                       # rad
    h: float                         # m, positive up
    vel: np.ndarray                  # NED velocity (m/s)
    C_nb: np.ndarray = field(default_factory=lambda: np.eye(3))

    def copy(self):
        return NavState(self.lat, self.lon, self.h, self.vel.copy(), self.C_nb.copy())


def ins_step(s, wb, fb, dt):
    """Propagate one IMU interval.

    wb: compensated angular rate (rad/s, body); fb: compensated specific force (m/s^2, body).
    Returns (new_state, f_n) where f_n is the specific force resolved in NED (needed by the KF).
    The same discretization is inverted by sim.imu.true_imu_from_trajectory, so a perfect
    IMU reproduces the reference trajectory.
    """
    RM, RN = radius(s.lat)
    w_ie = earth_rate(s.lat)
    w_en = transport_rate(s.lat, s.vel, s.h, RM, RN)
    w_in = w_ie + w_en

    # Attitude: body rate relative to the navigation frame.
    w_nb_b = wb - s.C_nb.T @ w_in
    C_new = s.C_nb @ rotvec2dcm(w_nb_b * dt)

    # Velocity.
    f_n = 0.5 * (s.C_nb + C_new) @ fb
    vdot = f_n + gravity(s.lat, s.h) - np.cross(2.0 * w_ie + w_en, s.vel)
    v_new = s.vel + vdot * dt

    # Position (trapezoidal velocity).
    v_avg = 0.5 * (s.vel + v_new)
    lat_new = s.lat + v_avg[0] / (RM + s.h) * dt
    lon_new = s.lon + v_avg[1] / ((RN + s.h) * np.cos(s.lat)) * dt
    h_new = s.h - v_avg[2] * dt

    return NavState(lat_new, lon_new, h_new, v_new, C_new), f_n


def ins_step_batch(lat, lon, h, vel, C, wb, fb, dt):
    """Vectorized ins_step over M states (used by the UKF sigma points).

    lat, lon, h: (M,); vel, wb, fb: (M,3); C: (M,3,3).  Same discretization as ins_step.
    """
    from .earth import A, E2, OMEGA_IE
    s2 = np.sin(lat) ** 2
    den = 1.0 - E2 * s2
    RN = A / np.sqrt(den); RM = A * (1.0 - E2) / den ** 1.5
    cl, sl = np.cos(lat), np.sin(lat)
    w_ie = OMEGA_IE * np.stack((cl, np.zeros_like(cl), -sl), 1)
    w_en = np.stack((vel[:, 1] / (RN + h), -vel[:, 0] / (RM + h), -vel[:, 1] * np.tan(lat) / (RN + h)), 1)
    w_nb = wb - np.einsum("kji,kj->ki", C, w_ie + w_en)
    r = w_nb * dt
    a = np.linalg.norm(r, axis=1)[:, None, None]
    z = np.zeros(len(r))
    S = np.stack((np.stack((z, -r[:, 2], r[:, 1]), 1), np.stack((r[:, 2], z, -r[:, 0]), 1),
                  np.stack((-r[:, 1], r[:, 0], z), 1)), 1)
    s1 = np.where(a < 1e-10, 1.0, np.sin(a) / np.maximum(a, 1e-12))
    c1 = np.where(a < 1e-10, 0.5, (1 - np.cos(a)) / np.maximum(a, 1e-12) ** 2)
    C_new = C @ (np.eye(3) + s1 * S + c1 * S @ S)
    f_n = np.einsum("kij,kj->ki", 0.5 * (C + C_new), fb)
    g0 = 9.7803253359 * (1.0 + 0.001931853 * s2) / np.sqrt(den) * (1.0 - 2.0 * h / A)
    g = np.zeros_like(vel); g[:, 2] = g0
    v_new = vel + (f_n + g - np.cross(2.0 * w_ie + w_en, vel)) * dt
    va = 0.5 * (vel + v_new)
    return (lat + va[:, 0] / (RM + h) * dt, lon + va[:, 1] / ((RN + h) * cl) * dt,
            h - va[:, 2] * dt, v_new, C_new)
