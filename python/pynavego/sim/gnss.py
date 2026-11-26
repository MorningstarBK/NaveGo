"""GNSS position/velocity simulation with optional outages (cf. NaveGo gnss_gen.m, gnss_outage.m)."""
import numpy as np

from ..earth import radius
from .baro import SensorData

GNSS_PRESET = dict(freq=1.0, stdm=(2.5, 2.5, 5.0), stdv=(0.1, 0.1, 0.15))


def simulate_gnss(ref, params=None, outages=(), seed=2):
    """outages: iterable of (t_start, t_end) intervals without GNSS."""
    p = dict(GNSS_PRESET, **(params or {}))
    rng = np.random.default_rng(seed)
    step = int(round(ref.freq / p["freq"]))
    idx = np.arange(0, len(ref.t), step)
    keep = np.ones(len(idx), bool)
    for t0, t1 in outages:
        keep &= ~((ref.t[idx] >= t0) & (ref.t[idx] <= t1))
    idx = idx[keep]
    stdm, stdv = np.asarray(p["stdm"]), np.asarray(p["stdv"])
    RM, RN = radius(ref.lat[idx])
    n = rng.normal(size=(len(idx), 3)) * stdm
    lat = ref.lat[idx] + n[:, 0] / (RM + ref.h[idx])
    lon = ref.lon[idx] + n[:, 1] / ((RN + ref.h[idx]) * np.cos(ref.lat[idx]))
    h = ref.h[idx] + n[:, 2]
    vel = ref.vel[idx] + rng.normal(size=(len(idx), 3)) * stdv
    return SensorData(ref.t[idx], {"lat": lat, "lon": lon, "h": h, "vel": vel}, p)
