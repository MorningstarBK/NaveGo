"""Line-of-sight (LOS) range sensor: slant distance from the first trajectory point (anchor).

r = |p - p0| + bias + white noise, with p in a local NED frame centred on the anchor
(curvilinear approximation, fine for trajectories of a few tens of km).
"""
import numpy as np

from ..earth import radius
from .baro import SensorData

LOS_PRESET = dict(freq=10.0, noise_std=0.5, bias=0.0, min_range=5.0)


def ned_from_anchor(lat, lon, h, lat0, lon0, h0):
    RM, RN = radius(lat0)
    return np.stack(((np.asarray(lat) - lat0) * (RM + h0),
                     (np.asarray(lon) - lon0) * (RN + h0) * np.cos(lat0),
                     -(np.asarray(h) - h0)), axis=-1)


def simulate_los(ref, params=None, seed=6):
    p = dict(LOS_PRESET, **(params or {}))
    rng = np.random.default_rng(seed)
    step = int(round(ref.freq / p["freq"]))
    idx = np.arange(0, len(ref.t), step)
    p["anchor"] = (float(ref.lat[0]), float(ref.lon[0]), float(ref.h[0]))
    ned = ned_from_anchor(ref.lat[idx], ref.lon[idx], ref.h[idx], *p["anchor"])
    r = np.linalg.norm(ned, axis=1) + p["bias"] + p["noise_std"] * rng.normal(size=len(idx))
    return SensorData(ref.t[idx], {"r": r, "r_true": np.linalg.norm(ned, axis=1)}, p)
