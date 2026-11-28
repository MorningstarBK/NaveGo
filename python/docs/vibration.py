"""Vibration / shock effects on the rotating-IMU turntable: lateral wobble ("lắc ngang"),
longitudinal wobble ("lắc dọc"), once-per-revolution wobble (coning) and angle slip ("trượt góc")
between motor/encoder and the IMU table.

To capture coning/sculling errors honestly, the motion is generated at 500 Hz, converted to
exact IMU increments, then averaged to 100 Hz (an integrating IMU output). The 100 Hz INS then
uses the averaged rates, as a real system does. Wobble and slip are unknown to the navigation
software; the encoder measures the commanded motor angle only.

Run from python/:  python docs/vibration.py [n_seeds] [imu]  -> docs/vibration.json, figures/f16_vibration.png
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from pynavego.attitude import wrap_angle  # noqa: E402
from pynavego.fusion import initial_state, run_fusion  # noqa: E402
from pynavego.metrics import nav_errors, rmse  # noqa: E402
from pynavego.sim import (ImuData, add_imu_errors, generate_trajectory, simulate_encoder,  # noqa: E402
                          simulate_gnss, simulate_los, true_imu_vectorized, vehicle_attitude)

T, TG, FH, DEC, RATE = 600.0, 60.0, 400.0, 4, 36.0
D2R = np.pi / 180
# case: (lateral RMS deg, longitudinal RMS deg, once-per-rev amplitude deg, slip on)
CASES = {
    "Không rung": (0, 0, 0, False),
    "Lắc ngang 0,05°": (0.05, 0, 0, False),
    "Lắc dọc 0,05°": (0, 0.05, 0, False),
    "Lắc ngang + dọc 0,05°": (0.05, 0.05, 0, False),
    "Lắc ngang + dọc 0,2°": (0.2, 0.2, 0, False),
    "Lắc theo vòng quay 0,2°": (0, 0, 0.2, False),
    "Trượt góc": (0, 0, 0, True),
    "Tổng hợp": (0.05, 0.05, 0.2, True),
}
EXTRA_CASES = {"Lắc ngang + dọc 0,5°": (0.5, 0.5, 0, False), "Lắc ngang + dọc 1°": (1.0, 1.0, 0, False)}
# body axis indices perpendicular to the spin axis: (lateral wobble axis, longitudinal wobble axis)
WOB_AXES = {"x": (2, 1), "z": (0, 1)}
_REF = {}


def ref_high():
    if "r" not in _REF:
        _REF["r"] = generate_trajectory(duration=T, freq=FH)
    return _REF["r"]


def skew_vec(v):
    z = np.zeros(len(v))
    return np.stack((np.stack((z, -v[:, 2], v[:, 1]), 1), np.stack((v[:, 2], z, -v[:, 0]), 1),
                     np.stack((-v[:, 1], v[:, 0], z), 1)), 1)


def rodrigues_vec(r):
    a = np.linalg.norm(r, axis=1)[:, None, None]
    S = skew_vec(r)
    s1 = np.where(a < 1e-10, 1.0, np.sin(a) / np.maximum(a, 1e-12))
    s2 = np.where(a < 1e-10, 0.5, (1 - np.cos(a)) / np.maximum(a, 1e-12) ** 2)
    return np.eye(3) + s1 * S + s2 * S @ S


def band_noise(n, fs, f0, f1, rms, rng):
    x = np.fft.rfft(rng.normal(size=n))
    f = np.fft.rfftfreq(n, 1 / fs)
    x[(f < f0) | (f > f1)] = 0
    y = np.fft.irfft(x, n)
    return y / (y.std() + 1e-30) * rms


def dcm2euler_vec(C):
    return np.stack((np.arctan2(C[:, 2, 1], C[:, 2, 2]), -np.arcsin(np.clip(C[:, 2, 0], -1, 1)),
                     np.arctan2(C[:, 1, 0], C[:, 0, 0])), 1)


def build(seed, axis, case):
    lat_rms, lon_rms, rev, slip_on = {**CASES, **EXTRA_CASES}[case]
    ref = ref_high()
    n = len(ref.t); dt = 1 / FH
    rng = np.random.default_rng(700 + seed)
    th_cmd = RATE * D2R * ref.t
    slip = np.zeros(n)
    if slip_on:   # random-walk slip 0.05 deg/sqrt(s) + shocks every ~30 s of N(0, 0.5 deg)
        slip = np.cumsum(rng.normal(size=n)) * 0.05 * D2R * np.sqrt(dt)
        shocks = rng.random(n) < dt / 30.0
        slip += np.cumsum(shocks * rng.normal(0, 0.5 * D2R, n))
    th = th_cmd + slip
    alpha = np.zeros((n, 3))
    ia, ib = WOB_AXES[axis]
    if lat_rms:
        alpha[:, ia] += band_noise(n, FH, 5, 40, lat_rms * D2R, rng)
    if lon_rms:
        alpha[:, ib] += band_noise(n, FH, 5, 40, lon_rms * D2R, rng)
    if rev:       # once-per-revolution wobble: axis traces a cone (coning motion)
        alpha[:, ia] += rev * D2R * np.cos(th)
        alpha[:, ib] += rev * D2R * np.sin(th)
    i = "xyz".index(axis)
    c, s = np.cos(th), np.sin(th)
    R = np.zeros((n, 3, 3)); R[:, i, i] = 1
    j, k = [m for m in range(3) if m != i]
    sign = 1 if axis != "y" else -1
    R[:, j, j] = c; R[:, k, k] = c; R[:, j, k] = -s * sign; R[:, k, j] = s * sign
    Cs = ref.C_nb @ rodrigues_vec(alpha) @ R
    hi = true_imu_vectorized(ref.t, ref.lat, ref.h, ref.vel, Cs)
    m = (n - 1) // DEC
    wb = hi.wb[: m * DEC].reshape(m, DEC, 3).mean(1)
    fb = hi.fb[: m * DEC].reshape(m, DEC, 3).mean(1)
    idx = np.arange(0, m * DEC + 1, DEC)
    wb = np.vstack((wb, wb[-1:])); fb = np.vstack((fb, fb[-1:]))
    t = ref.t[idx]
    sl = lambda x: x[idx]
    ref100 = replace(ref, t=t, lat=sl(ref.lat), lon=sl(ref.lon), h=sl(ref.h), vel=sl(ref.vel),
                     euler=sl(ref.euler), C_nb=sl(ref.C_nb))
    sref100 = replace(ref100, C_nb=Cs[idx], euler=dcm2euler_vec(Cs[idx]))
    true = ImuData(t, wb, fb, {})
    extra = dict(alpha=alpha[idx], slip=slip[idx], hi=hi, ref_hi=ref, sC_hi=Cs)
    return ref100, sref100, true, th_cmd[idx], extra


def case(args):
    seed, axis, name, grade = args
    ref, sref, true, th_cmd, extra = build(seed, axis, name)
    imu = add_imu_errors(true, grade, seed)
    enc = simulate_encoder(ref.t, th_cmd, seed=100 + seed)
    gnss = simulate_gnss(ref, outages=[(TG + 1e-6, T + 1)], seed=400 + seed)
    los = simulate_los(ref, seed=500 + seed)
    out = ref.t > TG
    res = {}
    for aid, kw in (("none", {}), ("los", dict(los=los))):
        r = run_fusion(imu, initial_state(sref), gnss=gnss, **kw)
        e = nav_errors(r, ref)
        att = wrap_angle(vehicle_attitude(r.euler[::10], enc.theta[::10], axis) - ref.euler[::10])
        o = out[::10]
        res[aid] = dict(hor=rmse(e["horizontal"], out), h=rmse(e["h"], out),
                        tilt=float(np.rad2deg(rmse(np.hypot(att[:, 0], att[:, 1]), o))),
                        yaw=float(np.rad2deg(rmse(att[:, 2], o))))
    return args, res


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    grade = sys.argv[2] if len(sys.argv) > 2 else "sch16t"
    jobs = [(s, ax, c, grade) for ax in ("x", "z") for c in CASES for s in range(n)]
    with ProcessPoolExecutor() as ex:
        out = dict(ex.map(case, jobs, chunksize=4))
    data = {f"{ax}|{c}|{aid}": [out[(s, ax, c, grade)][aid] for s in range(n)]
            for ax in ("x", "z") for c in CASES for aid in ("none", "los")}
    json.dump(data, open(os.path.join(HERE, "vibration.json"), "w"), ensure_ascii=False)
    med = lambda ax, c, aid, k: float(np.median([d[k] for d in data[f"{ax}|{c}|{aid}"]]))
    print(f"IMU {grade}, Monte Carlo {n} seeds (median), 400 Hz motion -> 100 Hz IMU, GNSS until {TG:.0f} s")
    print("| Quay | Trường hợp | Ngang chỉ IMU (m) | Ngang IMU+LOS (m) | Độ cao chỉ IMU / +LOS (m) | Nghiêng xe chỉ IMU / +LOS (°) | Yaw xe chỉ IMU / +LOS (°) |")
    print("|---|---|---|---|---|---|---|")
    for ax in ("x", "z"):
        for c in CASES:
            print(f"| {ax.upper()} | {c} | {med(ax, c, 'none', 'hor'):.0f} | {med(ax, c, 'los', 'hor'):.1f} | "
                  f"{med(ax, c, 'none', 'h'):.1f} / {med(ax, c, 'los', 'h'):.1f} | "
                  f"{med(ax, c, 'none', 'tilt'):.3f} / {med(ax, c, 'los', 'tilt'):.3f} | "
                  f"{med(ax, c, 'none', 'yaw'):.3f} / {med(ax, c, 'los', 'yaw'):.3f} |")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2, figsize=(15, 8.5))
    names = list(CASES); y = np.arange(len(names))
    for i, ax in enumerate(("x", "z")):
        for j, (aid, col) in enumerate((("none", "#9aa3ad"), ("los", "#7b2cbf"))):
            axs[0, i].barh(y + (j - 0.5) * 0.38, [med(ax, c, aid, "hor") for c in names], 0.38, color=col,
                           label="Chỉ IMU" if aid == "none" else "IMU + LOS")
        axs[0, i].set_yticks(y); axs[0, i].set_yticklabels(names); axs[0, i].invert_yaxis()
        axs[0, i].set_xscale("log"); axs[0, i].set_xlabel("RMSE ngang (m)"); axs[0, i].set_title(f"Quay {ax.upper()}")
        axs[0, i].legend(fontsize=8); axs[0, i].grid(axis="x", alpha=0.3)
        axs[1, i].barh(y - 0.19, [med(ax, c, "los", "tilt") for c in names], 0.38, color="#1f77b4", label="nghiêng xe")
        axs[1, i].barh(y + 0.19, [med(ax, c, "los", "yaw") for c in names], 0.38, color="#d62728", label="yaw xe")
        axs[1, i].set_yticks(y); axs[1, i].set_yticklabels(names); axs[1, i].invert_yaxis()
        axs[1, i].set_xlabel("RMSE tư thế xe, IMU + LOS (°)"); axs[1, i].legend(fontsize=8); axs[1, i].grid(axis="x", alpha=0.3)
    fig.suptitle(f"Rung xóc trên trục quay ({grade}, trung vị {n} lần Monte Carlo)")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "figures", "f16_vibration.png"), dpi=110)

    # example time series of the disturbances (seed 0, Z, combined)
    ref, sref, true, th_cmd, ex = build(0, "z", "Tổng hợp")
    fig, a = plt.subplots(1, 3, figsize=(15, 3.4))
    k = ref.t < 2
    a[0].plot(ref.t[k], np.rad2deg(ex["alpha"][k, 0]), label="lắc ngang (quanh X)")
    a[0].plot(ref.t[k], np.rad2deg(ex["alpha"][k, 1]), label="lắc dọc (quanh Y)")
    a[0].set_title("Góc lắc trục quay, 2 s đầu (°)"); a[0].legend(fontsize=8)
    a[1].plot(ref.t, np.rad2deg(ex["slip"])); a[1].set_title("Trượt góc giữa động cơ và IMU (°)")
    a[2].plot(ref.t[k], np.rad2deg(true.wb[k])); a[2].set_title("Con quay thật (°/s), 2 s đầu")
    for x in a:
        x.set_xlabel("t (s)"); x.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "figures", "f16b_vibration_signals.png"), dpi=110)
    print("saved figures/f16_vibration.png, f16b_vibration_signals.png")
