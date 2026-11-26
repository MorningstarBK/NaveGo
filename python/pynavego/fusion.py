"""Loosely-coupled error-state EKF: INS aided by barometric altitude and/or GNSS.

Mirrors the structure of NaveGo ins-gnss/ins_gnss.m (closed-loop error-state filter run
at IMU rate, corrections fed back to the INS), extended with a barometer bias state.

Error state (16):
    0:3   phi   attitude error in NED  (C_est = (I - [phi x]) C_true)
    3:6   dv    velocity error NED     (est - true)
    6:9   dp    position error NED (m) (est - true)
    9:12  dgb   residual gyro bias     (true - estimated)
    12:15 dab   residual accel bias
    15    dbb   residual baro bias
Without aiding the vertical channel diverges (dvD_dot = 2g/R dpD); the barometer bounds it.
"""
from dataclasses import dataclass

import numpy as np

from .attitude import dcm2euler, orthonormalize, skew
from .earth import A, earth_rate, gravity_magnitude, radius, transport_rate
from .ins import NavState, ins_step

NX = 16


@dataclass
class FilterConfig:
    # Initial 1-sigma uncertainties.
    att0: tuple = (np.deg2rad(0.5), np.deg2rad(0.5), np.deg2rad(2.0))
    vel0: float = 0.2
    pos0: tuple = (3.0, 3.0, 5.0)
    baro_bias0: float = 5.0
    # Baro bias model (Gauss-Markov) and measurement noise; None -> taken from sensor params.
    baro_drift_std: float = None
    baro_drift_corr: float = None
    baro_noise_std: float = None
    eps: float = 1e-4             # time-matching tolerance (s), cf. gnss.eps in NaveGo


@dataclass
class FusionResult:
    t: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    h: np.ndarray
    vel: np.ndarray
    euler: np.ndarray
    gb: np.ndarray
    ab: np.ndarray
    baro_bias: np.ndarray
    P_diag: np.ndarray


def _F_matrix(s, f_n, tau_g, tau_a, tau_b):
    RM, RN = radius(s.lat)
    w_ie = earth_rate(s.lat)
    w_en = transport_rate(s.lat, s.vel, s.h, RM, RN)
    C = s.C_nb
    F = np.zeros((NX, NX))
    F[0:3, 0:3] = -skew(w_ie + w_en)
    F[0:3, 9:12] = -C
    F[3:6, 0:3] = skew(f_n)
    F[3:6, 3:6] = -skew(2.0 * w_ie + w_en)
    F[3:6, 12:15] = C
    F[5, 8] = 2.0 * gravity_magnitude(s.lat, s.h) / A
    F[6:9, 3:6] = np.eye(3)
    F[9:12, 9:12] = -np.eye(3) / tau_g
    F[12:15, 12:15] = -np.eye(3) / tau_a
    F[15, 15] = -1.0 / tau_b
    return F


def run_fusion(imu, init, baro=None, gnss=None, cfg=None):
    """Run the filter over all IMU samples.

    imu:  sim.ImuData (with .err SI parameters used for tuning)
    init: NavState at imu.t[0]
    baro: SensorData with data['h'] (optional)
    gnss: SensorData with lat/lon/h/vel (optional)
    """
    cfg = cfg or FilterConfig()
    e = imu.err
    bp = baro.params if baro is not None else {}
    baro_noise = cfg.baro_noise_std or bp.get("noise_std", 0.5)
    baro_drift = cfg.baro_drift_std or bp.get("drift_std", 1.0)
    tau_b = cfg.baro_drift_corr or bp.get("drift_corr", 300.0)

    t = imu.t
    N = len(t)
    s = init.copy()
    gb = np.zeros(3); ab = np.zeros(3); bb = 0.0

    P = np.diag(np.concatenate((
        np.square(cfg.att0), np.full(3, cfg.vel0**2), np.square(cfg.pos0),
        np.full(3, e["gb_sta"]**2 + e["gb_dyn"]**2),
        np.full(3, e["ab_sta"]**2 + e["ab_dyn"]**2),
        [cfg.baro_bias0**2])))
    Qc = np.diag(np.concatenate((
        np.full(3, e["arw"]**2), np.full(3, e["vrw"]**2), np.zeros(3),
        np.full(3, 2 * e["gb_dyn"]**2 / e["gb_corr"]),
        np.full(3, 2 * e["ab_dyn"]**2 / e["ab_corr"]),
        [2 * baro_drift**2 / tau_b])))
    I = np.eye(NX)

    out = {k: np.zeros((N, 3)) for k in ("vel", "euler", "gb", "ab", "pos")}
    out_bb = np.zeros(N); out_P = np.zeros((N, NX))

    def record(k):
        out["pos"][k] = (s.lat, s.lon, s.h)
        out["vel"][k] = s.vel
        out["euler"][k] = dcm2euler(s.C_nb)
        out["gb"][k], out["ab"][k] = gb, ab
        out_bb[k] = bb
        out_P[k] = np.diag(P)

    record(0)
    jb = jg = 0
    for k in range(N - 1):
        dt = t[k + 1] - t[k]
        s, f_n = ins_step(s, imu.wb[k] - gb, imu.fb[k] - ab, dt)
        if k % 1000 == 0:
            s.C_nb = orthonormalize(s.C_nb)

        F = _F_matrix(s, f_n, e["gb_corr"], e["ab_corr"], tau_b)
        Phi = I + F * dt
        P = Phi @ P @ Phi.T + Qc * dt

        tk = t[k + 1]
        H_rows, z, R = [], [], []
        RM, RN = radius(s.lat)

        if baro is not None:
            while jb < len(baro.t) and baro.t[jb] < tk - cfg.eps:
                jb += 1
            if jb < len(baro.t) and abs(baro.t[jb] - tk) <= cfg.eps:
                # z = h_ins - (h_baro - bb_hat) = -dpD - dbb
                Hb = np.zeros(NX); Hb[8] = -1.0; Hb[15] = -1.0
                H_rows.append(Hb); z.append(s.h - (baro.data["h"][jb] - bb)); R.append(baro_noise**2)
                jb += 1

        if gnss is not None:
            while jg < len(gnss.t) and gnss.t[jg] < tk - cfg.eps:
                jg += 1
            if jg < len(gnss.t) and abs(gnss.t[jg] - tk) <= cfg.eps:
                g = gnss.data
                dp = np.array([(s.lat - g["lat"][jg]) * (RM + s.h),
                               (s.lon - g["lon"][jg]) * (RN + s.h) * np.cos(s.lat),
                               -(s.h - g["h"][jg])])
                dv = s.vel - g["vel"][jg]
                for i in range(3):
                    Hp = np.zeros(NX); Hp[6 + i] = 1.0
                    H_rows.append(Hp); z.append(dp[i]); R.append(gnss.params["stdm"][i] ** 2)
                for i in range(3):
                    Hv = np.zeros(NX); Hv[3 + i] = 1.0
                    H_rows.append(Hv); z.append(dv[i]); R.append(gnss.params["stdv"][i] ** 2)
                jg += 1

        if H_rows:
            H = np.array(H_rows); z = np.array(z); Rm = np.diag(R)
            S = H @ P @ H.T + Rm
            K = np.linalg.solve(S, H @ P).T
            x = K @ z
            IKH = I - K @ H
            P = IKH @ P @ IKH.T + K @ Rm @ K.T
            # Closed-loop feedback.
            s.C_nb = (np.eye(3) + skew(x[0:3])) @ s.C_nb
            s.vel = s.vel - x[3:6]
            s.lat -= x[6] / (RM + s.h)
            s.lon -= x[7] / ((RN + s.h) * np.cos(s.lat))
            s.h += x[8]
            gb = gb + x[9:12]
            ab = ab + x[12:15]
            bb += x[15]

        record(k + 1)

    pos = out["pos"]
    return FusionResult(t, pos[:, 0], pos[:, 1], pos[:, 2], out["vel"], out["euler"],
                        out["gb"], out["ab"], out_bb, out_P)


def initial_state(ref, att_err_deg=(0.2, 0.2, 1.0), pos_err=(0.0, 0.0, 0.0), k=0):
    """NavState from a reference trajectory with optional initial errors."""
    from .attitude import euler2dcm
    RM, RN = radius(ref.lat[k])
    e = ref.euler[k] + np.deg2rad(att_err_deg)
    return NavState(ref.lat[k] + pos_err[0] / (RM + ref.h[k]),
                    ref.lon[k] + pos_err[1] / ((RN + ref.h[k]) * np.cos(ref.lat[k])),
                    ref.h[k] - pos_err[2], ref.vel[k].copy(), euler2dcm(*e))
