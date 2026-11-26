"""Overlapping Allan deviation (cf. NaveGo allan-variance/allan_imu.m)."""
import numpy as np


def allan_deviation(x, freq, n_taus=60):
    """x: (N,) or (N,k) rate samples. Returns (taus, adev)."""
    x = np.asarray(x, float)
    if x.ndim == 1:
        x = x[:, None]
    N = x.shape[0]
    theta = np.vstack((np.zeros((1, x.shape[1])), np.cumsum(x, axis=0))) / freq
    m = np.unique(np.logspace(0, np.log10((N - 1) / 2), n_taus).astype(int))
    taus = m / freq
    adev = np.empty((len(m), x.shape[1]))
    for i, mi in enumerate(m):
        d = theta[2 * mi:] - 2 * theta[mi:-mi] + theta[:-2 * mi]
        adev[i] = np.sqrt(np.sum(d**2, axis=0) / (2 * taus[i] ** 2 * (N - 2 * mi)))
    return taus, adev.squeeze()
