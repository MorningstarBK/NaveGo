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
