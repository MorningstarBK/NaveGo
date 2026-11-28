"""Effect of motor/IMU misalignment on a rotating IMU (rotation modulation) + encoder.

Two misalignments, both unknown to the navigation software:
  mount     - IMU axes tilted w.r.t. the turntable (motor/encoder) axis: C_bs = R(theta) @ M
  axis_tilt - turntable axis tilted w.r.t. the vehicle body:             C_bs = T @ R(theta)
Sweeps the misalignment angle, Monte Carlo over sensor seeds (paired: same seed for every angle).
Run from python/:  python docs/misalignment.py [n_seeds] [imu]  -> docs/misalignment.json, figures/f15_misalignment.png
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from pynavego.attitude import wrap_angle  # noqa: E402
from pynavego.fusion import initial_state, run_fusion  # noqa: E402
from pynavego.metrics import nav_errors, rmse  # noqa: E402
from pynavego.sim import (generate_trajectory, rotate_trajectory, rotation_angle,  # noqa: E402
                          simulate_encoder, simulate_gnss, simulate_imu, simulate_los, vehicle_attitude)

T, TG = 600.0, 60.0
ANGLES = [0.0, 0.1, 0.5, 1.0, 2.0]          # deg
AXES = ["x", "z"]
KINDS = ["mount", "axis_tilt"]


def case(args):
    seed, axis, kind, ang, imu_grade = args
    ref = generate_trajectory(duration=T)
    th = rotation_angle(ref.t, 36.0)
    enc = simulate_encoder(ref.t, th, seed=100 + seed)
    # misalignment direction: fixed, about both axes perpendicular to the spin axis
    d = np.deg2rad(ang) / np.sqrt(2)
    vec = {"x": (0.0, d, d), "z": (d, d, 0.0)}[axis]
    sref = rotate_trajectory(ref, th, axis, **{kind: vec})
    imu, _ = simulate_imu(sref, imu_grade, seed=seed)
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
                        roll=float(np.rad2deg(rmse(att[:, 0], o))), pitch=float(np.rad2deg(rmse(att[:, 1], o))),
                        yaw=float(np.rad2deg(rmse(att[:, 2], o))),
                        att_max=float(np.rad2deg(np.abs(att[o]).max())))
    return args, res


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    grade = sys.argv[2] if len(sys.argv) > 2 else "sch16t"
    jobs = [(s, ax, k, a, grade) for ax in AXES for k in KINDS for a in ANGLES for s in range(n)]
    with ProcessPoolExecutor() as ex:
        out = dict(ex.map(case, jobs))
    data = {f"{ax}|{k}|{a}|{aid}": [out[(s, ax, k, a, grade)][aid] for s in range(n)]
            for ax in AXES for k in KINDS for a in ANGLES for aid in ("none", "los")}
    json.dump(data, open(os.path.join(HERE, "misalignment.json"), "w"))
    med = lambda ax, k, a, aid, key: float(np.median([d[key] for d in data[f"{ax}|{k}|{a}|{aid}"]]))

    KL = {"mount": "IMU lệch so với trục động cơ", "axis_tilt": "Trục động cơ lệch so với xe"}
    print(f"IMU {grade}, Monte Carlo {n} seeds (median), GNSS until {TG:.0f} s")
    print("| Quay | Loại lệch | Góc lệch (°) | Ngang chỉ IMU (m) | Ngang IMU+LOS (m) | Độ cao chỉ IMU / +LOS (m) | Roll / Pitch / Yaw xe, IMU+LOS (°) |")
    print("|---|---|---|---|---|---|---|")
    for ax in AXES:
        for k in KINDS:
            for a in ANGLES:
                print(f"| {ax.upper()} | {KL[k]} | {a} | {med(ax, k, a, 'none', 'hor'):.0f} | {med(ax, k, a, 'los', 'hor'):.1f} | "
                      f"{med(ax, k, a, 'none', 'h'):.1f} / {med(ax, k, a, 'los', 'h'):.1f} | "
                      f"{med(ax, k, a, 'los', 'roll'):.3f} / {med(ax, k, a, 'los', 'pitch'):.3f} / {med(ax, k, a, 'los', 'yaw'):.3f} |")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    for i, ax in enumerate(AXES):
        for k, ls in zip(KINDS, ("-", "--")):
            for aid, c in (("none", "#9aa3ad"), ("los", "#7b2cbf")):
                axs[0].plot(ANGLES, [med(ax, k, a, aid, "hor") for a in ANGLES], ls, marker="os"[i], color=c,
                            label=f"Quay {ax.upper()}, {KL[k]}, {'chỉ IMU' if aid == 'none' else '+LOS'}")
            v = [max(med(ax, k, a, "los", "roll"), med(ax, k, a, "los", "pitch")) for a in ANGLES]
            axs[1].plot(ANGLES, v, ls, marker="os"[i], color=f"C{i}", label=f"Quay {ax.upper()}, {KL[k]}")
            axs[2].plot(ANGLES, [med(ax, k, a, "los", "yaw") for a in ANGLES], ls, marker="os"[i], color=f"C{i}")
    axs[0].set_yscale("log"); axs[0].set_ylabel("RMSE ngang (m)"); axs[0].legend(fontsize=6)
    axs[1].plot(ANGLES, np.array(ANGLES) / np.sqrt(2), "k:", lw=1, label="= góc lệch / √2")
    axs[1].set_ylabel("max(RMSE roll, pitch) của xe (°)"); axs[1].legend(fontsize=7)
    axs[2].set_ylabel("RMSE yaw của xe (°)")
    for a in axs:
        a.set_xlabel("góc lệch (°)"); a.grid(alpha=0.3)
    fig.suptitle(f"Ảnh hưởng của lệch trục động cơ–IMU ({grade}, trung vị {n} lần Monte Carlo, IMU + LOS khi không ghi chú)")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "figures", "f15_misalignment.png"), dpi=110)
    print("saved figures/f15_misalignment.png")
