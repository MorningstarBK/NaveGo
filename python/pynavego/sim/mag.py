"""Magnetometer simulation: m_s = C_ns^T m_n + hard-iron bias + white noise (uT).

The field is mounted on the sensor frame of the given trajectory (pass the rotated sensor
trajectory for a rotating IMU). m_n is a constant local Earth field from total intensity F,
inclination I (positive down) and declination D, e.g. from the WMM/IGRF for the site.
Default values are approximate for Hanoi.
"""
import numpy as np

from .baro import SensorData

MAG_PRESET = dict(freq=10.0, F=45.0, incl_deg=31.0, decl_deg=-1.3,
                  bias=(0.5, -0.3, 0.4), noise_std=0.2)


def earth_field_ned(F, incl_deg, decl_deg):
    I, D = np.deg2rad(incl_deg), np.deg2rad(decl_deg)
    return F * np.array([np.cos(I) * np.cos(D), np.cos(I) * np.sin(D), np.sin(I)])


def simulate_mag(traj, params=None, seed=5):
    p = dict(MAG_PRESET, **(params or {}))
    rng = np.random.default_rng(seed)
    step = int(round(traj.freq / p["freq"]))
    idx = np.arange(0, len(traj.t), step)
    m_n = earth_field_ned(p["F"], p["incl_deg"], p["decl_deg"])
    m = np.einsum("kji,j->ki", traj.C_nb[idx], m_n)          # C_nb^T m_n
    m = m + np.asarray(p["bias"]) + p["noise_std"] * rng.normal(size=m.shape)
    p["m_n"] = m_n
    return SensorData(traj.t[idx], {"m": m}, p)
