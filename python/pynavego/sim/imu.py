"""Inertial sensor simulation (cf. NaveGo simulation/acc_gen.m, gyro_gen.m, conversions/imu_si_errors.m).

Error model per axis: measured = true + static (turn-on) bias + Gauss-Markov dynamic bias
(bias instability) + white noise (ARW / VRW).
"""
from dataclasses import dataclass

import numpy as np

from ..attitude import dcm2rotvec
from ..earth import OMEGA_IE, earth_rate, gravity, gravity_magnitude, radius, transport_rate

G0 = 9.80665
D2R = np.pi / 180.0

# Datasheet-style units: arw deg/sqrt(h), vrw m/s/sqrt(h), gyro biases deg/h,
# accel biases mg, correlation times s.  Values are representative, not exact datasheets.
IMU_PRESETS = {
    # ring-laser / high-end FOG class (e.g. ~0.003 deg/h, 25 ug)
    "navigation": dict(arw=0.002, vrw=0.0012, gb_sta=0.01, ab_sta=0.025, gb_dyn=0.003, ab_dyn=0.005,
                       gb_corr=3600.0, ab_corr=3600.0),
    # Murata SCH16T-K30 datasheet Doc.No.14736 Rev.1, typical values (Tables 4 and 6).
    # Per-axis [X, Y, Z]. gb_sta = gyro offset typ (XY 0.1 deg/s, Z 0.01 deg/s, over -40..110 C),
    # ab_sta = accel offset typ 0.01 m/s^2 (~1.02 mg), bias instability 0.3 deg/h and 0.15 mm/s^2,
    # ARW XY 0.015 / Z 0.025 deg/sqrt(h), VRW 30 (mm/s)/sqrt(h), sensitivity error typ 0.05 %.
    # Correlation times are not in the datasheet (assumed 100 s).
    "sch16t": dict(arw=[0.015, 0.015, 0.025], vrw=0.03, gb_sta=[360.0, 360.0, 36.0], ab_sta=0.01 / G0 * 1e3,
                   gb_dyn=0.3, ab_dyn=0.15e-3 / G0 * 1e3, gb_corr=100.0, ab_corr=100.0, gsf=0.05, asf=0.05),
    # Same sensor without scale-factor error (to isolate its effect under IMU rotation).
    "sch16t_nosf": dict(arw=[0.015, 0.015, 0.025], vrw=0.03, gb_sta=[360.0, 360.0, 36.0], ab_sta=0.01 / G0 * 1e3,
                        gb_dyn=0.3, ab_dyn=0.15e-3 / G0 * 1e3, gb_corr=100.0, ab_corr=100.0),
    "tactical": dict(arw=0.15, vrw=0.05, gb_sta=1.0, ab_sta=1.0, gb_dyn=0.5, ab_dyn=0.05,
                     gb_corr=300.0, ab_corr=300.0),
    "industrial": dict(arw=0.3, vrw=0.03, gb_sta=10.0, ab_sta=5.0, gb_dyn=6.25, ab_dyn=0.1,
                       gb_corr=100.0, ab_corr=100.0),
    "consumer": dict(arw=0.6, vrw=0.3, gb_sta=200.0, ab_sta=30.0, gb_dyn=20.0, ab_dyn=1.0,
                     gb_corr=100.0, ab_corr=100.0),
}


def imu_errors_si(spec):
    """Convert datasheet units to SI (rad/s/sqrt(Hz), m/s^2/sqrt(Hz), rad/s, m/s^2)."""
    spec = {k: (np.asarray(v, float) if isinstance(v, (list, tuple)) else v) for k, v in spec.items()}
    return dict(
        gsf=spec.get("gsf", 0.0) * 1e-2,      # sensitivity (scale-factor) error, % -> fraction
        asf=spec.get("asf", 0.0) * 1e-2,
        arw=spec["arw"] * D2R / 60.0,
        vrw=spec["vrw"] / 60.0,
        gb_sta=spec["gb_sta"] * D2R / 3600.0,
        ab_sta=spec["ab_sta"] * 1e-3 * G0,
        gb_dyn=spec["gb_dyn"] * D2R / 3600.0,
        ab_dyn=spec["ab_dyn"] * 1e-3 * G0,
        gb_corr=float(spec["gb_corr"]),
        ab_corr=float(spec["ab_corr"]),
    )


@dataclass
class ImuData:
    t: np.ndarray    # (N,)
    wb: np.ndarray   # (N,3) rad/s; sample k covers [t_k, t_k+1)
    fb: np.ndarray   # (N,3) m/s^2
    err: dict        # SI error parameters (empty for a perfect IMU)

    @property
    def freq(self):
        return 1.0 / (self.t[1] - self.t[0])


def true_imu_from_trajectory(ref):
    """Error-free gyro/accelerometer outputs that make ins_step reproduce `ref`."""
    N = len(ref.t)
    dt = ref.t[1] - ref.t[0]
    wb = np.zeros((N, 3)); fb = np.zeros((N, 3))
    for k in range(N - 1):
        C0, C1 = ref.C_nb[k], ref.C_nb[k + 1]
        lat, h, v = ref.lat[k], ref.h[k], ref.vel[k]
        RM, RN = radius(lat)
        w_ie = earth_rate(lat)
        w_en = transport_rate(lat, v, h, RM, RN)
        wb[k] = dcm2rotvec(C0.T @ C1) / dt + C0.T @ (w_ie + w_en)
        f_n = (ref.vel[k + 1] - v) / dt - gravity(lat, h) + np.cross(2.0 * w_ie + w_en, v)
        fb[k] = np.linalg.solve(0.5 * (C0 + C1), f_n)
    wb[-1], fb[-1] = wb[-2], fb[-2]
    return ImuData(ref.t.copy(), wb, fb, {})


def gauss_markov(n, sigma, tau, dt, rng):
    """First-order Gauss-Markov process (n,3) with steady-state std `sigma`."""
    a = np.exp(-dt / tau)
    q = sigma * np.sqrt(1.0 - a * a)
    out = np.empty((n, 3))
    b = rng.normal(0.0, sigma, 3)
    for k in range(n):
        out[k] = b
        b = a * b + q * rng.normal(size=3)
    return out


def simulate_imu(ref, spec="industrial", seed=0):
    """Return (ImuData with errors, true ImuData). `spec` is a preset name or a datasheet dict."""
    if isinstance(spec, str):
        spec = IMU_PRESETS[spec]
    true = true_imu_from_trajectory(ref)
    return add_imu_errors(true, spec, seed), true


def add_imu_errors(true, spec="industrial", seed=0):
    """Add the error model of `spec` to error-free ImuData `true` (any sample rate)."""
    if isinstance(spec, str):
        spec = IMU_PRESETS[spec]
    e = imu_errors_si(spec)
    rng = np.random.default_rng(seed)
    N = len(true.t); dt = true.t[1] - true.t[0]; fs = 1.0 / dt

    gb_sta = e["gb_sta"] * rng.choice([-1.0, 1.0], 3)
    ab_sta = e["ab_sta"] * rng.choice([-1.0, 1.0], 3)
    gb_true = gb_sta + gauss_markov(N, e["gb_dyn"], e["gb_corr"], dt, rng)
    ab_true = ab_sta + gauss_markov(N, e["ab_dyn"], e["ab_corr"], dt, rng)
    gsf = e["gsf"] * rng.choice([-1.0, 1.0], 3)
    asf = e["asf"] * rng.choice([-1.0, 1.0], 3)
    e["gsf_vec"], e["asf_vec"] = gsf, asf
    wb = true.wb * (1.0 + gsf) + gb_true + e["arw"] * np.sqrt(fs) * rng.normal(size=(N, 3))
    fb = true.fb * (1.0 + asf) + ab_true + e["vrw"] * np.sqrt(fs) * rng.normal(size=(N, 3))
    e["gb_sta_vec"], e["ab_sta_vec"] = gb_sta, ab_sta
    e["gb_true"], e["ab_true"] = gb_true, ab_true      # total bias time series (sensor frame)
    return ImuData(true.t.copy(), wb, fb, e)


def true_imu_vectorized(t, lat, h, vel, C_nb):
    """Vectorized true_imu_from_trajectory for long high-rate trajectories (same discretization)."""
    dt = t[1] - t[0]
    C0, C1 = C_nb[:-1], C_nb[1:]
    R = np.einsum("kji,kjl->kil", C0, C1)
    v = np.stack((R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]), axis=1)
    c = np.clip((np.trace(R, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    a = np.arccos(c)
    k = np.where(a < 1e-8, 0.5, a / (2.0 * np.sin(np.maximum(a, 1e-12))))
    rotv = v * k[:, None]
    la, hh, v0 = lat[:-1], h[:-1], vel[:-1]
    RM, RN = radius(la)
    w_ie = OMEGA_IE * np.stack((np.cos(la), np.zeros_like(la), -np.sin(la)), axis=1)
    w_en = np.stack((v0[:, 1] / (RN + hh), -v0[:, 0] / (RM + hh), -v0[:, 1] * np.tan(la) / (RN + hh)), axis=1)
    wb = rotv / dt + np.einsum("kji,kj->ki", C0, w_ie + w_en)
    g = np.zeros_like(v0); g[:, 2] = gravity_magnitude(la, hh)
    f_n = (vel[1:] - v0) / dt - g + np.cross(2.0 * w_ie + w_en, v0)
    fb = np.linalg.solve(0.5 * (C0 + C1), f_n[:, :, None])[:, :, 0]
    wb = np.vstack((wb, wb[-1:])); fb = np.vstack((fb, fb[-1:]))
    return ImuData(t.copy(), wb, fb, {})
