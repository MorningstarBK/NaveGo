"""Reference trajectory generator (NaveGo ships none; this one is built for this package).

Velocity/attitude come from smooth analytic profiles (speed, turn rate, climb rate with a
coordinated-turn roll); position is integrated with exactly the same formulas as ins_step.
"""
from dataclasses import dataclass

import numpy as np

from ..attitude import euler2dcm
from ..earth import radius


@dataclass
class Trajectory:
    t: np.ndarray        # (N,)
    lat: np.ndarray      # (N,) rad
    lon: np.ndarray      # (N,) rad
    h: np.ndarray        # (N,) m
    vel: np.ndarray      # (N,3) NED m/s
    euler: np.ndarray    # (N,3) roll, pitch, yaw rad
    C_nb: np.ndarray     # (N,3,3)

    @property
    def freq(self):
        return 1.0 / (self.t[1] - self.t[0])


def _smoothstep(t, t0, t1):
    x = np.clip((t - t0) / (t1 - t0), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def generate_trajectory(duration=600.0, freq=100.0, lat0_deg=21.0285, lon0_deg=105.8542,
                        h0=20.0, yaw0_deg=30.0, speed=15.0, climb_amp=3.0, still=10.0):
    """Vehicle that stands still for `still` s, then cruises with turns and climbs/descents."""
    dt = 1.0 / freq
    t = np.arange(int(round(duration * freq)) + 1) * dt
    env = _smoothstep(t, still, still + 20.0)

    spd = env * (speed + 5.0 * np.sin(2 * np.pi * t / 200.0))
    yaw_rate = env * 0.04 * np.sin(2 * np.pi * t / 90.0)
    climb = env * climb_amp * np.sin(2 * np.pi * t / 150.0)

    yaw = np.deg2rad(yaw0_deg) + np.concatenate(
        ([0.0], np.cumsum(0.5 * (yaw_rate[1:] + yaw_rate[:-1]) * dt)))
    pitch = np.arctan2(climb, np.maximum(spd, 1e-6))
    roll = np.arctan(spd * yaw_rate / 9.80)

    vel = np.column_stack((spd * np.cos(yaw), spd * np.sin(yaw), -climb))

    N = len(t)
    lat = np.empty(N); lon = np.empty(N); h = np.empty(N)
    lat[0], lon[0], h[0] = np.deg2rad(lat0_deg), np.deg2rad(lon0_deg), h0
    for k in range(N - 1):
        RM, RN = radius(lat[k])
        va = 0.5 * (vel[k] + vel[k + 1])
        lat[k + 1] = lat[k] + va[0] / (RM + h[k]) * dt
        lon[k + 1] = lon[k] + va[1] / ((RN + h[k]) * np.cos(lat[k])) * dt
        h[k + 1] = h[k] - va[2] * dt

    euler = np.column_stack((roll, pitch, yaw))
    C = np.array([euler2dcm(*e) for e in euler])
    return Trajectory(t, lat, lon, h, vel, euler, C)
