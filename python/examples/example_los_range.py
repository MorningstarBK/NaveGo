"""Rotating IMU + encoder, INS aided by a LOS range sensor (distance to the start point).

GNSS only for the first 60 s, then lost to the end. No baro, no magnetometer.
Run from python/:  python examples/example_los_range.py [--mc 12] [--no-plot]
Horizontal error is also split into a radial part (along the LOS, observable by the range
sensor) and a tangential part (perpendicular, not observable from one anchor).
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pynavego.attitude import wrap_angle  # noqa: E402
from pynavego.fusion import initial_state, run_fusion  # noqa: E402
from pynavego.metrics import nav_errors, rmse  # noqa: E402
from pynavego.sim import (generate_trajectory, ned_from_anchor, rotate_trajectory,  # noqa: E402
                          rotation_angle, simulate_encoder, simulate_gnss, simulate_imu,
                          simulate_los, vehicle_attitude)

T, TG = 600.0, 60.0
MOUNTS = {"Quay X": "x", "Quay Y": "y", "Quay Z": "z", "Cố định": None}
AIDS = {"none": "Chỉ IMU", "los": "IMU + LOS"}


def run_case(seed, mount, imu_grade="industrial", los_std=0.5, keep=False):
    ref = generate_trajectory(duration=T)
    ax = MOUNTS[mount]
    th = rotation_angle(ref.t, 36.0)
    enc = simulate_encoder(ref.t, th, seed=100 + seed)
    sref = rotate_trajectory(ref, th, ax) if ax else ref
    eth = enc.theta if ax else np.zeros_like(ref.t)
    imu, _ = simulate_imu(sref, imu_grade, seed=seed)
    gnss = simulate_gnss(ref, outages=[(TG + 1e-6, T + 1)], seed=400 + seed)
    los = simulate_los(ref, dict(noise_std=los_std), seed=500 + seed)
    out = ref.t > TG
    anchor = los.params["anchor"]
    p_ref = ned_from_anchor(ref.lat, ref.lon, ref.h, *anchor)
    u = p_ref[:, :2] / np.maximum(np.linalg.norm(p_ref[:, :2], axis=1, keepdims=True), 1e-9)
    res_all = {}
    for aid in AIDS:
        r = run_fusion(imu, initial_state(sref), gnss=gnss, **({"los": los} if aid == "los" else {}))
        e = nav_errors(r, ref)
        dh = np.column_stack((e["north"], e["east"]))
        radial = np.sum(dh * u, axis=1)
        tang = u[:, 0] * dh[:, 1] - u[:, 1] * dh[:, 0]
        p_est = ned_from_anchor(r.lat, r.lon, r.h, *anchor)
        rng_err = np.linalg.norm(p_est, axis=1) - np.linalg.norm(p_ref, axis=1)
        att = wrap_angle(vehicle_attitude(r.euler[::10], eth[::10], ax or "x") - ref.euler[::10])
        o = out[::10]
        res_all[aid] = dict(
            hor_rmse=rmse(e["horizontal"], out), hor_max=float(e["horizontal"][out].max()),
            hor_end=float(e["horizontal"][-1]), rad_rmse=rmse(radial, out), tan_rmse=rmse(tang, out),
            rng_rmse=rmse(rng_err, out), h_rmse=rmse(e["h"], out), h_end=float(e["h"][-1]),
            tilt=float(np.rad2deg(rmse(np.hypot(att[:, 0], att[:, 1]), o))),
            yaw=float(np.rad2deg(rmse(att[:, 2], o))))
        if keep:
            res_all[aid]["series"] = dict(t=ref.t, hor=e["horizontal"], radial=radial, tang=tang,
                                          h=e["h"], rng=rng_err, n=e["north"], e=e["east"])
    if keep:
        res_all["ref"] = (ref, p_ref)
    return res_all


def _job(a):
    return a, run_case(*a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc", type=int, default=12, help="number of Monte Carlo seeds")
    ap.add_argument("--imu", default="industrial", choices=["navigation", "tactical", "industrial", "consumer"])
    ap.add_argument("--los-std", type=float, default=0.5)
    ap.add_argument("--json", default=None, help="save Monte Carlo results to this JSON file")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--out", default="los_results.png")
    a = ap.parse_args()

    jobs = [(s, m, a.imu, a.los_std) for s in range(a.mc) for m in MOUNTS]
    with ProcessPoolExecutor() as ex:
        out = dict(ex.map(_job, jobs))
    if a.json:
        import json
        data = {m: {aid: [out[(s, m, a.imu, a.los_std)][aid] for s in range(a.mc)] for aid in AIDS}
                for m in MOUNTS}
        json.dump(data, open(a.json, "w"), ensure_ascii=False)
    keys = [("hor_rmse", "RMSE ngang", 0), ("hor_max", "Ngang max", 0), ("hor_end", "Trôi cuối", 0),
            ("rad_rmse", "RMSE dọc LOS", 1), ("tan_rmse", "RMSE vuông góc LOS", 0),
            ("rng_rmse", "RMSE khoảng cách", 2), ("h_rmse", "RMSE độ cao", 1),
            ("tilt", "Nghiêng (°)", 3), ("yaw", "Yaw (°)", 3)]
    print(f"IMU {a.imu}, {T:.0f} s, GNSS until {TG:.0f} s, LOS range sigma {a.los_std} m, "
          f"Monte Carlo {a.mc} seeds (median; p10-p90 for horizontal RMSE)")
    hdr = "| IMU | Hỗ trợ | " + " | ".join(k[1] + (" (m)" if "°" not in k[1] else "") for k in keys) + " | p10–p90 ngang (m) |"
    print(hdr); print("|" + "---|" * (len(keys) + 3))
    for m in MOUNTS:
        for aid, lab in AIDS.items():
            vals = {k: [out[(s, m, a.imu, a.los_std)][aid][k] for s in range(a.mc)] for k, _, _ in keys}
            cells = " | ".join(f"{np.median(vals[k]):.{d}f}" for k, _, d in keys)
            h = vals["hor_rmse"]
            print(f"| {m} | {lab} | {cells} | {np.percentile(h, 10):.0f}–{np.percentile(h, 90):.0f} |")

    if a.no_plot:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 4, figsize=(15, 10))
    for i, m in enumerate(MOUNTS):
        R = run_case(0, m, a.imu, a.los_std, keep=True)
        ref, p_ref = R["ref"]
        for aid, c in (("none", "#9aa3ad"), ("los", "#7b2cbf")):
            S = R[aid]["series"]
            axs[0, i].semilogy(S["t"], np.maximum(S["hor"], 0.1), color=c, label=AIDS[aid])
            axs[1, i].plot(S["t"], S["radial"], color=c, ls="-", label=f"{AIDS[aid]} dọc LOS")
            axs[1, i].plot(S["t"], S["tang"], color=c, ls=":", label=f"{AIDS[aid]} vuông góc")
        S = R["los"]["series"]
        axs[2, i].plot(p_ref[:, 1] / 1000, p_ref[:, 0] / 1000, "k", label="thật")
        axs[2, i].plot((p_ref[:, 1] + S["e"]) / 1000, (p_ref[:, 0] + S["n"]) / 1000, color="#7b2cbf", label="IMU + LOS")
        Sn = R["none"]["series"]
        axs[2, i].plot((p_ref[:, 1] + Sn["e"]) / 1000, (p_ref[:, 0] + Sn["n"]) / 1000, color="#9aa3ad", label="Chỉ IMU")
        axs[2, i].plot(0, 0, "r*", ms=12, label="điểm neo LOS")
        axs[2, i].set_aspect("equal", adjustable="datalim"); axs[2, i].set_xlabel("Đông (km)")
        axs[0, i].set_title(m); axs[0, i].set_ylim(0.3, 3e4)
        axs[1, i].set_yscale("symlog", linthresh=10)
        for x in axs[:2, i]:
            x.axvspan(TG, T, color="0.92", zorder=0); x.grid(True, alpha=0.3); x.set_xlabel("t (s)")
        axs[2, i].grid(True, alpha=0.3)
    axs[0, 0].set_ylabel("sai số ngang (m)"); axs[1, 0].set_ylabel("sai số ngang (m)"); axs[2, 0].set_ylabel("Bắc (km)")
    axs[0, 0].legend(fontsize=8); axs[1, 0].legend(fontsize=7); axs[2, 0].legend(fontsize=7)
    fig.suptitle(f"IMU quay + encoder, hỗ trợ bằng khoảng cách LOS tới điểm xuất phát ({a.imu}, seed 0; vùng xám: mất GNSS)")
    fig.tight_layout(); fig.savefig(a.out, dpi=110)
    print("saved", a.out)


if __name__ == "__main__":
    main()
