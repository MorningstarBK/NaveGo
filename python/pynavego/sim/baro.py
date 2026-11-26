"""Barometric altimeter simulation: h_baro = h_true + bias + Gauss-Markov drift + white noise."""
from dataclasses import dataclass

import numpy as np


@dataclass
class SensorData:
    t: np.ndarray
    data: dict          # arrays keyed by quantity
    params: dict


BARO_PRESET = dict(freq=10.0, noise_std=0.3, bias=4.0, drift_std=1.0, drift_corr=300.0)


def simulate_baro(ref, params=None, seed=1):
    p = dict(BARO_PRESET, **(params or {}))
    rng = np.random.default_rng(seed)
    step = int(round(ref.freq / p["freq"]))
    idx = np.arange(0, len(ref.t), step)
    dt = 1.0 / p["freq"]
    a = np.exp(-dt / p["drift_corr"])
    q = p["drift_std"] * np.sqrt(1.0 - a * a)
    drift = np.empty(len(idx)); d = 0.0
    for k in range(len(idx)):
        drift[k] = d
        d = a * d + q * rng.normal()
    bias = p["bias"] + drift
    h = ref.h[idx] + bias + p["noise_std"] * rng.normal(size=len(idx))
    return SensorData(ref.t[idx], {"h": h, "bias_true": bias}, p)
