"""Coning and sculling compensation for multi-subsample IMU increments.

The IMU delivers angle increments dtheta_i and velocity increments dv_i at a high rate; the INS
updates at a lower rate with N subsamples per update. The rotation vector and the body-frame
velocity increment of one update are

    phi = sum(dtheta) + dphi_coning
    dv  = sum(dv) + 1/2 theta x v + dv_sculling        (rotation compensation + sculling)

with the Ignagni/Savage optimal coefficients for pure coning / sculling:
    2-sample : dphi = 2/3  d1 x d2
    3-sample : dphi = 33/80 d1 x d3 + 57/80 d2 x (d3 - d1)
    4-sample : dphi = 736/945 (d1xd2 + d3xd4) + 334/945 (d1xd3 + d2xd4) + 526/945 d1xd4 + 654/945 d2xd3
    "1+1"    : dphi = 1/12 d_prev x d   (one sample per update, uses the previous update)
Sculling uses the same coefficients on (dtheta_i x dv_j + dv_i x dtheta_j).
References: Ignagni, JGCD 1996; Savage, JGCD 1998 (Parts 1 and 2); Groves 2013, sec. 5.5.

`compensate` returns equivalent constant rates (wb, fb) per update that make
pynavego.ins.ins_step reproduce phi and dv, so the EKF code is unchanged.
"""
import numpy as np

from .attitude import skew

_K = {
    2: [((0, 1), 2 / 3)],
    3: [((0, 2), 33 / 80), ((1, 2), 57 / 80), ((0, 1), 57 / 80)],   # d2 x (d3 - d1) = d2xd3 + d1xd2
    4: [((0, 1), 736 / 945), ((2, 3), 736 / 945), ((0, 2), 334 / 945), ((1, 3), 334 / 945),
        ((0, 3), 526 / 945), ((1, 2), 654 / 945)],
}
METHODS = ["mean", "1+1", "2-sample", "3-sample", "4-sample"]


def _split(x, n, m):
    """Group N=n*m high-rate samples into m updates of n subsamples, (m, n, 3)."""
    return x[: n * m].reshape(m, n, 3)


def _regroup(blocks, k):
    """Merge n subsamples into k equal groups (k must divide n)."""
    m, n, _ = blocks.shape
    return blocks.reshape(m, k, n // k, 3).sum(2)


def compensate(wb_hi, fb_hi, dt_hi, n, method="4-sample"):
    """wb_hi, fb_hi: (N,3) high-rate rates (each constant over its own dt_hi).
    n: subsamples per INS update.  Returns (wb, fb) of length m+1 at the update rate."""
    m = (len(wb_hi) - 1) // n
    dth = _split(wb_hi * dt_hi, n, m)
    dvl = _split(fb_hi * dt_hi, n, m)
    T = n * dt_hi
    theta = dth.sum(1)
    v = dvl.sum(1)
    dphi = np.zeros_like(theta)
    dvs = np.zeros_like(v)
    if method == "1+1":
        tp = np.vstack((theta[:1] * 0, theta[:-1]))
        vp = np.vstack((v[:1] * 0, v[:-1]))
        dphi = np.cross(tp, theta) / 12.0
        dvs = (np.cross(tp, v) + np.cross(vp, theta)) / 12.0
    elif method != "mean":
        k = int(method[0])
        if n % k:
            raise ValueError(f"{method} needs subsamples divisible by {k}, got {n}")
        a, b = _regroup(dth, k), _regroup(dvl, k)
        for (i, j), c in _K[k]:
            dphi += c * np.cross(a[:, i], a[:, j])
            dvs += c * (np.cross(a[:, i], b[:, j]) + np.cross(b[:, i], a[:, j]))
    phi = theta + dphi
    # ins_step applies f_n = 0.5 (C0 + C1) fb, i.e. C0 (I + 1/2 [phi x]) fb to first order;
    # choose fb so that this equals C0 (v + 1/2 theta x v + dvs).
    dv_tot = v + 0.5 * np.cross(theta, v) + dvs
    fb = np.array([np.linalg.solve(np.eye(3) + 0.5 * skew(p), d) for p, d in zip(phi, dv_tot)]) / T
    wb = phi / T
    return np.vstack((wb, wb[-1:])), np.vstack((fb, fb[-1:]))
