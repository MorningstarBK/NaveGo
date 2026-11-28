"""EKF vs UKF / CKF on the rotating-IMU + baro problem (GNSS only for the first 60 s).

Run from python/:  python docs/ukf_eval.py [n_seeds] [imu]  -> docs/ukf_eval.json, figures/f18_ukf.png
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from pynavego.attitude import wrap_angle  # noqa: E402
from pynavego.fusion import FilterConfig, initial_state, run_fusion  # noqa: E402
from pynavego.metrics import nav_errors, rmse  # noqa: E402
from pynavego.sim import (generate_trajectory, rotate_trajectory, rotation_angle, simulate_baro,  # noqa: E402
                          simulate_encoder, simulate_gnss, simulate_imu, vehicle_attitude)
from pynavego.ukf import run_ukf  # noqa: E402

T, TG = 600.0, 60.0
MOUNTS = {"Quay X": "x", "Quay Z": "z", "Cố định": None}
INITS = {"Sai số đầu nhỏ (0,2/0,2/1°)": ((0.2, 0.2, 1.0), (0.5, 0.5, 2.0)),
         "Sai số đầu lớn (3/3/30°)": ((3.0, 3.0, 30.0), (5.0, 5.0, 45.0))}
FILTERS = {
    "EKF": None,
    "UKF α=1 β=2": dict(alpha=1.0, beta=2.0, kappa=0.0),
    "UKF α=0,5 β=2": dict(alpha=0.5, beta=2.0, kappa=0.0),
    "CKF (cubature)": dict(alpha=1.0, beta=0.0, kappa=0.0),
}


def job(args):
    seed, m, ini, grade = args
    ref = generate_trajectory(duration=T)
    ax = MOUNTS[m]
    th = rotation_angle(ref.t, 36.0)
    enc = simulate_encoder(ref.t, th, seed=100 + seed)
    sref = rotate_trajectory(ref, th, ax) if ax else ref
    eth = enc.theta if ax else np.zeros_like(ref.t)
    imu, _ = simulate_imu(sref, grade, seed=seed)
    baro = simulate_baro(ref, seed=200 + seed)
    gnss = simulate_gnss(ref, outages=[(TG + 1e-6, T + 1)], seed=400 + seed)
    err0, sig0 = INITS[ini]
    rng = np.random.default_rng(900 + seed)
    e0 = np.array(err0) * rng.choice([-1, 1], 3)
    init = initial_state(sref, att_err_deg=tuple(e0))
    cfg = FilterConfig(att0=tuple(np.deg2rad(sig0)))
    out = ref.t > TG
    res = {}
    for name, p in FILTERS.items():
        t0 = time.time()
        r = run_fusion(imu, init, baro=baro, gnss=gnss, cfg=cfg) if p is None else \
            run_ukf(imu, init, baro=baro, gnss=gnss, cfg=cfg, **p)
        dtc = time.time() - t0
        e = nav_errors(r, ref)
        att = wrap_angle(vehicle_attitude(r.euler[::10], eth[::10], ax or "x") - ref.euler[::10])
        o10 = out[::10]
        Pn, Pe, Pd = r.P_diag[:, 6], r.P_diag[:, 7], r.P_diag[:, 8]
        nees_hor = np.mean((e["north"][out] ** 2 / Pn[out] + e["east"][out] ** 2 / Pe[out]) / 2)
        nees_h = np.mean(e["h"][out] ** 2 / Pd[out])
        conv = ref.t <= TG
        res[name] = dict(hor=rmse(e["horizontal"], out), hor_end=float(e["horizontal"][-1]), h=rmse(e["h"], out),
                         yaw=float(np.rad2deg(rmse(att[:, 2], o10))),
                         tilt=float(np.rad2deg(rmse(np.hypot(att[:, 0], att[:, 1]), o10))),
                         yaw_60=float(np.rad2deg(abs(att[conv[::10]][-1, 2]))),
                         nees_hor=float(nees_hor), nees_h=float(nees_h), cpu=dtc,
                         ok=bool(np.all(np.isfinite(r.h))))
    return args, res


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    grade = sys.argv[2] if len(sys.argv) > 2 else "sch16t"
    jobs = [(s, m, i, grade) for m in MOUNTS for i in INITS for s in range(n)]
    with ProcessPoolExecutor() as ex:
        out = dict(ex.map(job, jobs))
    data = {f"{m}|{i}|{f}": [out[(s, m, i, grade)][f] for s in range(n)] for m in MOUNTS for i in INITS for f in FILTERS}
    json.dump(data, open(os.path.join(HERE, "ukf_eval.json"), "w"), ensure_ascii=False)
    med = lambda m, i, f, k: float(np.median([d[k] for d in data[f"{m}|{i}|{f}"]]))
    print(f"IMU {grade}, IMU + baro after GNSS loss at {TG:.0f} s, Monte Carlo {n} seeds (median)")
    print("| IMU | Sai số ban đầu | Bộ lọc | RMSE ngang (m) | Trôi cuối (m) | RMSE độ cao (m) | Nghiêng xe (°) | Yaw xe (°) | Yaw tại 60 s (°) | NEES ngang | NEES độ cao | CPU (s) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for m in MOUNTS:
        for i in INITS:
            for f in FILTERS:
                print(f"| {m} | {i} | {f} | {med(m, i, f, 'hor'):.0f} | {med(m, i, f, 'hor_end'):.0f} | {med(m, i, f, 'h'):.2f} | "
                      f"{med(m, i, f, 'tilt'):.3f} | {med(m, i, f, 'yaw'):.3f} | {med(m, i, f, 'yaw_60'):.3f} | "
                      f"{med(m, i, f, 'nees_hor'):.2f} | {med(m, i, f, 'nees_h'):.2f} | {med(m, i, f, 'cpu'):.1f} |")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.3))
    fl = list(FILTERS); cols = ["#9aa3ad", "#1f77b4", "#2ca02c", "#d62728"]
    labels = [f"{m}\n{i.split(' (')[0].replace('Sai số đầu ', '')}" for m in MOUNTS for i in INITS]
    x = np.arange(len(labels))
    for j, f in enumerate(fl):
        axs[0].bar(x + (j - 1.5) * 0.2, [med(m, i, f, "hor") for m in MOUNTS for i in INITS], 0.2, color=cols[j], label=f)
        axs[1].bar(x + (j - 1.5) * 0.2, [med(m, i, f, "yaw") for m in MOUNTS for i in INITS], 0.2, color=cols[j])
        axs[2].bar(x + (j - 1.5) * 0.2, [med(m, i, f, "nees_hor") for m in MOUNTS for i in INITS], 0.2, color=cols[j])
    axs[0].set_yscale("log"); axs[0].set_ylabel("RMSE ngang (m)"); axs[0].legend(fontsize=8)
    axs[1].set_ylabel("RMSE yaw xe (°)"); axs[1].set_yscale("log")
    axs[2].axhline(1, color="k", ls=":"); axs[2].set_yscale("log"); axs[2].set_ylabel("NEES ngang (lý tưởng = 1)")
    for a in axs:
        a.set_xticks(x); a.set_xticklabels(labels, fontsize=8); a.grid(axis="y", alpha=0.3)
    fig.suptitle(f"EKF so với UKF/CKF: IMU quay + baro ({grade}, trung vị {n} lần Monte Carlo)")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "figures", "f18_ukf.png"), dpi=110)
    print("saved figures/f18_ukf.png")
