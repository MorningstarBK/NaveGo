"""Error-state Unscented Kalman Filter (UKF / cubature KF) for INS + baro + GNSS.

Same 16-state error vector and closed-loop feedback as fusion.run_fusion (EKF), but the
covariance is propagated by pushing 2n+1 sigma points through the full nonlinear INS
mechanization (ins_step_batch) instead of the linearized F matrix.

Each block of `step` IMU samples (default 10 = 0.1 s, the baro rate):
  1. draw sigma points from (0, P) around the nominal state; each is a candidate true state
  2. propagate all candidates through the nonlinear INS with their own bias estimates
  3. map candidates back to error vectors w.r.t. the propagated nominal (sigma point 0),
     form mean and covariance, add Qc * dt_block, fold the mean into the nominal
  4. redraw sigma points and run the unscented measurement update (baro and/or GNSS)
Scaled unscented transform (Wan & van der Merwe 2000): lambda = alpha^2 (n + kappa) - n.
alpha=1, beta=0, kappa=0 gives the cubature Kalman filter (Arasaratnam & Haykin 2009).
"""
import numpy as np

from .attitude import dcm2euler
from .earth import radius
from .fusion import FilterConfig, FusionResult
from .ins import ins_step_batch

NX = 16


def _rodrigues(r):
    a = np.linalg.norm(r, axis=1)[:, None, None]
    z = np.zeros(len(r))
    S = np.stack((np.stack((z, -r[:, 2], r[:, 1]), 1), np.stack((r[:, 2], z, -r[:, 0]), 1),
                  np.stack((-r[:, 1], r[:, 0], z), 1)), 1)
    s1 = np.where(a < 1e-12, 1.0, np.sin(a) / np.maximum(a, 1e-15))
    c1 = np.where(a < 1e-12, 0.5, (1 - np.cos(a)) / np.maximum(a, 1e-15) ** 2)
    return np.eye(3) + s1 * S + c1 * S @ S


def _logm(R):
    v = np.stack((R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]), 1)
    c = np.clip((np.trace(R, axis1=1, axis2=2) - 1) / 2, -1, 1)
    a = np.arccos(c)
    k = np.where(a < 1e-8, 0.5, a / (2 * np.sin(np.maximum(a, 1e-12))))
    return v * k[:, None]


class _Nom:
    """Nominal state + candidate (sigma point) states as arrays; index 0 = nominal."""

    def __init__(self, lat, lon, h, vel, C, gb, ab, bb):
        self.lat, self.lon, self.h, self.vel, self.C = lat, lon, h, vel, C
        self.gb, self.ab, self.bb = gb, ab, bb


def _candidates(nom0, X):
    """Candidate true states nom0 (+) x for each row of X (M,16)."""
    lat0, lon0, h0 = nom0.lat, nom0.lon, nom0.h
    RM, RN = radius(lat0)
    M = len(X)
    return _Nom(lat0 - X[:, 6] / (RM + h0), lon0 - X[:, 7] / ((RN + h0) * np.cos(lat0)), h0 + X[:, 8],
                nom0.vel[None] - X[:, 3:6], _rodrigues(X[:, 0:3]) @ nom0.C[None],
                nom0.gb[None] + X[:, 9:12], nom0.ab[None] + X[:, 12:15], nom0.bb + X[:, 15])


def _errors(c):
    """Error vectors of candidates w.r.t. candidate 0 (same convention as the EKF)."""
    RM, RN = radius(c.lat[0])
    E = np.zeros((len(c.lat), NX))
    E[:, 0:3] = _logm(c.C @ c.C[0].T)
    E[:, 3:6] = c.vel[0] - c.vel
    E[:, 6] = (c.lat[0] - c.lat) * (RM + c.h[0])
    E[:, 7] = (c.lon[0] - c.lon) * (RN + c.h[0]) * np.cos(c.lat[0])
    E[:, 8] = c.h - c.h[0]
    E[:, 9:12] = c.gb - c.gb[0]
    E[:, 12:15] = c.ab - c.ab[0]
    E[:, 15] = c.bb - c.bb[0]
    return E


def _single(c, i=0):
    return _Nom(c.lat[i], c.lon[i], c.h[i], c.vel[i].copy(), c.C[i].copy(), c.gb[i].copy(), c.ab[i].copy(),
                float(c.bb[i]))


def run_ukf(imu, init, baro=None, gnss=None, cfg=None, alpha=1.0, beta=2.0, kappa=0.0, step=10):
    cfg = cfg or FilterConfig()
    e = imu.err
    bp = baro.params if baro is not None else {}
    baro_noise = cfg.baro_noise_std or bp.get("noise_std", 0.5)
    baro_drift = cfg.baro_drift_std or bp.get("drift_std", 1.0)
    tau_b = cfg.baro_drift_corr or bp.get("drift_corr", 300.0)
    n = NX
    lam = alpha ** 2 * (n + kappa) - n
    Wm = np.full(2 * n + 1, 1 / (2 * (n + lam))); Wc = Wm.copy()
    Wm[0] = lam / (n + lam); Wc[0] = Wm[0] + 1 - alpha ** 2 + beta

    P = np.diag(np.concatenate((np.square(cfg.att0), np.full(3, cfg.vel0 ** 2), np.square(cfg.pos0),
                                np.full(3, e["gb_sta"] ** 2 + e["gb_dyn"] ** 2),
                                np.full(3, e["ab_sta"] ** 2 + e["ab_dyn"] ** 2), [cfg.baro_bias0 ** 2])))
    Qc = np.diag(np.concatenate((np.full(3, e["arw"] ** 2), np.full(3, e["vrw"] ** 2), np.zeros(3),
                                 np.full(3, 2 * e["gb_dyn"] ** 2 / e["gb_corr"]),
                                 np.full(3, 2 * e["ab_dyn"] ** 2 / e["ab_corr"]), [2 * baro_drift ** 2 / tau_b])))
    nom = _Nom(init.lat, init.lon, init.h, init.vel.copy(), init.C_nb.copy(), np.zeros(3), np.zeros(3), 0.0)

    t = imu.t; N = len(t)
    out = dict(pos=np.zeros((N, 3)), vel=np.zeros((N, 3)), euler=np.zeros((N, 3)), gb=np.zeros((N, 3)),
               ab=np.zeros((N, 3)), bb=np.zeros(N), P=np.zeros((N, NX)))

    def rec(k, c, Pd):
        out["pos"][k] = (c.lat[0], c.lon[0], c.h[0]); out["vel"][k] = c.vel[0]
        out["euler"][k] = dcm2euler(c.C[0]); out["gb"][k] = c.gb[0]; out["ab"][k] = c.ab[0]
        out["bb"][k] = c.bb[0]; out["P"][k] = Pd

    def sigma(P):
        try:
            L = np.linalg.cholesky((n + lam) * P)
        except np.linalg.LinAlgError:
            w, V = np.linalg.eigh(0.5 * (P + P.T))
            L = V @ np.diag(np.sqrt(np.maximum(w, 1e-18) * (n + lam)))
        return np.vstack((np.zeros(n), L.T, -L.T))

    def fold(nom, x):
        c = _candidates(nom, x[None])
        return _single(c)

    c = _candidates(nom, sigma(P))
    rec(0, c, np.diag(P))
    ag, aa, abb = np.exp(-(t[1] - t[0]) / e["gb_corr"]), np.exp(-(t[1] - t[0]) / e["ab_corr"]), np.exp(-(t[1] - t[0]) / tau_b)
    jb = jg = 0
    t_blk = t[0]
    for k in range(N - 1):
        dt = t[k + 1] - t[k]
        c.lat, c.lon, c.h, c.vel, c.C = ins_step_batch(c.lat, c.lon, c.h, c.vel, c.C,
                                                       imu.wb[k][None] - c.gb, imu.fb[k][None] - c.ab, dt)
        # residual biases follow their Gauss-Markov models (relative to the nominal estimate)
        c.gb = c.gb[0] + (c.gb - c.gb[0]) * ag
        c.ab = c.ab[0] + (c.ab - c.ab[0]) * aa
        c.bb = c.bb[0] + (c.bb - c.bb[0]) * abb
        tk = t[k + 1]
        if (k + 1) % step == 0 or k == N - 2:
            E = _errors(c)
            xm = Wm @ E
            D = E - xm
            P = (Wc[:, None] * D).T @ D + Qc * (tk - t_blk)
            nom = fold(_single(c), xm)
            t_blk = tk
            # measurement update with redrawn sigma points
            X = sigma(P)
            zs, zp, Rs = [], [], []
            if baro is not None:
                while jb < len(baro.t) and baro.t[jb] < tk - cfg.eps:
                    jb += 1
                if jb < len(baro.t) and abs(baro.t[jb] - tk) <= cfg.eps:
                    # candidate baro reading: h_i + bb_i = (h0 + dpD) + (bb0 + dbb)
                    zp.append((nom.h + X[:, 8]) + (nom.bb + X[:, 15]))
                    zs.append(baro.data["h"][jb]); Rs.append(baro_noise ** 2); jb += 1
            if gnss is not None:
                while jg < len(gnss.t) and gnss.t[jg] < tk - cfg.eps:
                    jg += 1
                if jg < len(gnss.t) and abs(gnss.t[jg] - tk) <= cfg.eps:
                    g = gnss.data
                    RM, RN = radius(nom.lat)
                    m = [(g["lat"][jg] - nom.lat) * (RM + nom.h), (g["lon"][jg] - nom.lon) * (RN + nom.h) * np.cos(nom.lat),
                         -(g["h"][jg] - nom.h)]
                    for i in range(3):                      # candidate NED offset = -dp
                        zp.append(-X[:, 6 + i]); zs.append(m[i]); Rs.append(gnss.params["stdm"][i] ** 2)
                    for i in range(3):
                        zp.append(nom.vel[i] - X[:, 3 + i]); zs.append(g["vel"][jg][i]); Rs.append(gnss.params["stdv"][i] ** 2)
                    jg += 1
            if zs:
                Z = np.array(zp).T                              # (2n+1, r)
                zh = Wm @ Z
                Dz = Z - zh
                Xm = Wm @ X
                Pzz = (Wc[:, None] * Dz).T @ Dz + np.diag(Rs)
                Pxz = (Wc[:, None] * (X - Xm)).T @ Dz
                K = np.linalg.solve(Pzz, Pxz.T).T
                x = Xm + K @ (np.array(zs) - zh)
                P = P - K @ Pzz @ K.T
                P = 0.5 * (P + P.T)
                nom = fold(nom, x)
            c = _candidates(nom, sigma(P))
        rec(k + 1, c, np.diag(P))

    pos = out["pos"]
    return FusionResult(t, pos[:, 0], pos[:, 1], pos[:, 2], out["vel"], out["euler"], out["gb"], out["ab"],
                        out["bb"], np.zeros((N, 3)), out["P"])
