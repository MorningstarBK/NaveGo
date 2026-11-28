"""Coning/sculling algorithms on the vibrating turntable.

Part A - algorithm error only: perfect IMU (no sensor errors) at 400 Hz, free INS with no aiding,
         INS at 100 Hz with 4 subsamples per update, compared with a 400 Hz INS.
Part B - full system: SCH16T errors added at 400 Hz, EKF with GNSS for 60 s then (none | LOS).
Run from python/:  python docs/coning_test.py [n_seeds]  -> docs/coning.json, figures/f17_coning.png
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import vibration as V  # noqa: E402
from pynavego.attitude import euler2dcm, wrap_angle  # noqa: E402
from pynavego.coning import METHODS, compensate  # noqa: E402
from pynavego.fusion import initial_state, run_fusion  # noqa: E402
from pynavego.ins import NavState, ins_step  # noqa: E402
from pynavego.metrics import nav_errors, rmse  # noqa: E402
from pynavego.sim import (ImuData, add_imu_errors, simulate_encoder, simulate_gnss,  # noqa: E402
                          simulate_los, vehicle_attitude)

A_CASES = ["Lắc ngang + dọc 0,2°", "Lắc ngang + dọc 0,5°", "Lắc ngang + dọc 1°"]
B_CASES = ["Không rung", "Lắc ngang + dọc 0,2°", "Lắc ngang + dọc 0,5°", "Lắc ngang + dọc 1°"]
TA = 120.0   # seconds of free INS for part A


def free_ins(t, wb, fb, lat, lon, h, vel, C, n_steps):
    s = NavState(lat, lon, h, vel.copy(), C.copy())
    att = np.zeros((n_steps + 1, 3, 3)); pos = np.zeros((n_steps + 1, 3)); att[0] = C
    pos[0] = (lat, lon, h)
    for k in range(n_steps):
        s, _ = ins_step(s, wb[k], fb[k], t[k + 1] - t[k])
        att[k + 1] = s.C_nb; pos[k + 1] = (s.lat, s.lon, s.h)
    return att, pos


def att_err_deg(Ce, Ct):
    E = np.einsum("kij,klj->kil", Ce, Ct)
    v = np.stack((E[:, 2, 1] - E[:, 1, 2], E[:, 0, 2] - E[:, 2, 0], E[:, 1, 0] - E[:, 0, 1]), 1) / 2
    return np.rad2deg(np.linalg.norm(v, axis=1))


def part_a(args):
    axis, case = args
    ref100, sref100, true100, th, ex = V.build(0, axis, case)
    hi, rh, Chi = ex["hi"], ex["ref_hi"], ex["sC_hi"]
    nh = int(TA * V.FH); n = nh // V.DEC
    out = {}
    from pynavego.earth import radius
    RM, RN = radius(rh.lat[0])

    def errs(att, pos, idx):
        ea = att_err_deg(att, Chi[idx])
        dn = (pos[:, 0] - rh.lat[idx]) * (RM + rh.h[idx]); de = (pos[:, 1] - rh.lon[idx]) * (RN + rh.h[idx]) * np.cos(rh.lat[0])
        return dict(att_end=float(ea[-1]), att_rate=float(ea[-1] / TA * 3600), hor_end=float(np.hypot(dn, de)[-1]),
                    h_end=float(abs(pos[-1, 2] - rh.h[idx][-1])))
    att, pos = free_ins(rh.t, hi.wb, hi.fb, rh.lat[0], rh.lon[0], rh.h[0], rh.vel[0], Chi[0], nh)
    out["INS 400 Hz"] = errs(att, pos, np.arange(nh + 1))
    for mth in METHODS:
        if mth[0].isdigit() and V.DEC % int(mth[0]):
            continue                      # e.g. 3-sample needs a multiple of 3 subsamples
        wb, fb = compensate(hi.wb[: nh + 1], hi.fb[: nh + 1], 1 / V.FH, V.DEC, mth)
        t = rh.t[: nh + 1: V.DEC]
        att, pos = free_ins(t, wb, fb, rh.lat[0], rh.lon[0], rh.h[0], rh.vel[0], Chi[0], n)
        out[mth] = errs(att, pos, np.arange(0, nh + 1, V.DEC))
    return args, out


def part_b(args):
    seed, axis, case, mth = args
    ref, sref, true100, th, ex = V.build(seed, axis, case)
    hi = add_imu_errors(ex["hi"], "sch16t", seed)
    wb, fb = compensate(hi.wb, hi.fb, 1 / V.FH, V.DEC, mth)
    imu = ImuData(ref.t, wb, fb, hi.err)
    enc = simulate_encoder(ref.t, th, seed=100 + seed)
    gnss = simulate_gnss(ref, outages=[(V.TG + 1e-6, V.T + 1)], seed=400 + seed)
    los = simulate_los(ref, seed=500 + seed)
    out_m = ref.t > V.TG
    res = {}
    for aid, kw in (("none", {}), ("los", dict(los=los))):
        r = run_fusion(imu, initial_state(sref), gnss=gnss, **kw)
        e = nav_errors(r, ref)
        att = wrap_angle(vehicle_attitude(r.euler[::10], enc.theta[::10], axis) - ref.euler[::10])
        res[aid] = dict(hor=rmse(e["horizontal"], out_m), h=rmse(e["h"], out_m),
                        tilt=float(np.rad2deg(rmse(np.hypot(att[:, 0], att[:, 1]), out_m[::10]))))
    return args, res


if __name__ == "__main__":
    nseed = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    with ProcessPoolExecutor() as ex:
        A = dict(ex.map(part_a, [(ax, c) for ax in ("x", "z") for c in A_CASES]))
    print("PART A - algorithm error only (perfect IMU, free INS 120 s, 100 Hz update with 4 subsamples)")
    print("| Quay | Rung | Thuật toán | Sai số tư thế sau 120 s (°) | Tương đương trôi (°/h) | Sai số ngang 120 s (m) |")
    print("|---|---|---|---|---|---|")
    for ax in ("x", "z"):
        for c in A_CASES:
            for mth, v in A[(ax, c)].items():
                print(f"| {ax.upper()} | {c} | {mth} | {v['att_end']:.4f} | {v['att_rate']:.2f} | {v['hor_end']:.2f} |")
    best = {}
    for ax in ("x", "z"):
        for c in A_CASES:
            r = A[(ax, c)]
            best[(ax, c)] = min(r, key=lambda m: r[m]["att_end"])
    print("best per case:", best)

    jobs = [(s, ax, c, m) for ax in ("x", "z") for c in B_CASES for m in ("mean", "4-sample") for s in range(nseed)]
    with ProcessPoolExecutor() as ex:
        B = dict(ex.map(part_b, jobs, chunksize=2))
    med = lambda ax, c, m, aid, k: float(np.median([B[(s, ax, c, m)][aid][k] for s in range(nseed)]))
    print(f"\nPART B - SCH16T + EKF, Monte Carlo {nseed} seeds (median), GNSS until 60 s")
    print("| Quay | Rung | Thuật toán | Ngang chỉ IMU (m) | Ngang IMU+LOS (m) | Độ cao chỉ IMU / +LOS (m) | Nghiêng xe +LOS (°) |")
    print("|---|---|---|---|---|---|---|")
    for ax in ("x", "z"):
        for c in B_CASES:
            for m in ("mean", "4-sample"):
                print(f"| {ax.upper()} | {c} | {m} | {med(ax, c, m, 'none', 'hor'):.0f} | {med(ax, c, m, 'los', 'hor'):.1f} | "
                      f"{med(ax, c, m, 'none', 'h'):.1f} / {med(ax, c, m, 'los', 'h'):.1f} | {med(ax, c, m, 'los', 'tilt'):.3f} |")
    json.dump({"A": {f"{k[0]}|{k[1]}": v for k, v in A.items()},
               "B": {f"{k[0]}|{k[1]}|{k[2]}|{k[3]}": v for k, v in B.items()}},
              open(os.path.join(HERE, "coning.json"), "w"), ensure_ascii=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.3))
    labs = ["mean", "1+1", "2-sample", "4-sample", "INS 400 Hz"]
    for i, ax in enumerate(("x", "z")):
        for j, c in enumerate(A_CASES):
            vals = [max(A[(ax, c)][m]["att_rate"], 1e-3) for m in labs]
            axs[i].bar(np.arange(len(labs)) + (j - 1) * 0.27, vals, 0.27, label=c)
        axs[i].set_xticks(range(len(labs))); axs[i].set_xticklabels(labs, rotation=20)
        axs[i].set_yscale("log"); axs[i].set_ylabel("trôi tư thế do thuật toán (°/h)")
        axs[i].set_title(f"Quay {ax.upper()}: IMU lý tưởng, INS tự do 120 s"); axs[i].legend(fontsize=7); axs[i].grid(axis="y", alpha=0.3)
    x = np.arange(len(B_CASES))
    for j, (m, col) in enumerate((("mean", "#9aa3ad"), ("4-sample", "#2ca02c"))):
        axs[2].bar(x + (j - 0.5) * 0.38, [med("z", c, m, "none", "hor") for c in B_CASES], 0.38, color=col, label=f"Quay Z, chỉ IMU, {m}")
    axs[2].set_xticks(x); axs[2].set_xticklabels(B_CASES, rotation=15, fontsize=8); axs[2].set_yscale("log")
    axs[2].set_ylabel("RMSE ngang (m)"); axs[2].set_title("SCH16T + EKF, quay Z, chỉ IMU"); axs[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "figures", "f17_coning.png"), dpi=110)
    print("saved figures/f17_coning.png")
