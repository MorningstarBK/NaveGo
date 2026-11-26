"""Rotating IMU about the vehicle X axis + encoder, INS aided by baro; GNSS lost after 60 s.

Run from python/:  python examples/example_rotating_imu.py [--imu industrial] [--rate 36]
GNSS is available only during the first `--gnss-until` seconds (initial calibration), then it is
lost until the end. Compared: IMU only (pure INS after GNSS loss) vs IMU + baro, for a rotating
IMU and, as a reference, a fixed (strapped-down, non-rotating) IMU.
Vehicle roll/pitch/yaw are computed from the INS sensor attitude and the encoder angle.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pynavego.attitude import wrap_angle  # noqa: E402
from pynavego.fusion import initial_state, run_fusion  # noqa: E402
from pynavego.metrics import nav_errors, rmse  # noqa: E402
from pynavego.sim import (generate_trajectory, rotate_trajectory, rotation_angle,  # noqa: E402
                          simulate_baro, simulate_encoder, simulate_gnss, simulate_imu,
                          vehicle_attitude)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imu", default="industrial", choices=["tactical", "industrial", "consumer"])
    ap.add_argument("--duration", type=float, default=600.0)
    ap.add_argument("--gnss-until", type=float, default=60.0)
    ap.add_argument("--rate", type=float, default=36.0, help="turntable rate (deg/s)")
    ap.add_argument("--mode", default="continuous", choices=["continuous", "alternate"])
    ap.add_argument("--encoder-bits", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--out", default="rotating_imu_results.png")
    a = ap.parse_args()

    ref = generate_trajectory(duration=a.duration)               # vehicle trajectory
    theta = rotation_angle(ref.t, a.rate, a.mode)
    enc = simulate_encoder(ref.t, theta, bits=a.encoder_bits)
    baro = simulate_baro(ref, seed=1)
    gnss = simulate_gnss(ref, outages=[(a.gnss_until + 1e-6, a.duration + 1.0)], seed=2)
    outage = ref.t > a.gnss_until

    mounts = {
        "Rotating": (rotate_trajectory(ref, theta), enc.theta),
        "Fixed": (ref, np.zeros_like(ref.t)),
    }
    rows, curves = [], {}
    for mount, (sref, th) in mounts.items():
        imu, _ = simulate_imu(sref, a.imu, seed=a.seed)            # same error seed for both
        init = initial_state(sref)
        for aid, kw in (("IMU only", {}), ("IMU + Baro", dict(baro=baro))):
            res = run_fusion(imu, init, gnss=gnss, **kw)
            err = nav_errors(res, ref)                              # position/velocity
            att = wrap_angle(vehicle_attitude(res.euler, th) - ref.euler)
            hz = err["horizontal"]
            rows.append(dict(
                mount=mount, aid=aid,
                hor_end=hz[-1], hor_max=hz[outage].max(), hor_rmse=rmse(hz, outage),
                hor_rate=hz[-1] / ((a.duration - a.gnss_until) / 60.0),
                h_end=err["h"][-1], h_rmse=rmse(err["h"], outage),
                att=[np.rad2deg(rmse(att[:, i], outage)) for i in range(3)]))
            curves[f"{mount} / {aid}"] = (err, att)

    print(f"IMU {a.imu}, {a.duration:.0f} s, GNSS lost from t = {a.gnss_until:.0f} s to the end, "
          f"turntable {a.mode} {a.rate:.0f} deg/s about X, encoder {a.encoder_bits} bit")
    hdr = (f"| {'IMU mount':<9}| {'Aiding':<11}| {'Hor. drift end (m)':>18} | {'Hor. max (m)':>12} "
           f"| {'Hor. RMSE (m)':>13} | {'Hor. drift (m/min)':>18} | {'Alt. err end (m)':>16} "
           f"| {'Alt. RMSE (m)':>13} | {'Roll/Pitch/Yaw RMSE (deg)':>27} |")
    print(hdr)
    print("|" + "|".join("-" * (len(c)) for c in hdr.split("|")[1:-1]) + "|")
    for r in rows:
        print(f"| {r['mount']:<9}| {r['aid']:<11}| {r['hor_end']:>18.1f} | {r['hor_max']:>12.1f} "
              f"| {r['hor_rmse']:>13.1f} | {r['hor_rate']:>18.1f} | {r['h_end']:>16.2f} "
              f"| {r['h_rmse']:>13.2f} | {'%.3f / %.3f / %.3f' % tuple(r['att']):>27} |")
    print("(all error statistics are computed over the GNSS outage only)")

    if a.no_plot:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    for name, (err, att) in curves.items():
        ls = "-" if name.startswith("Rotating") else "--"
        ax[0].plot(ref.t, err["horizontal"], ls, label=name)
        ax[1].plot(ref.t, err["h"], ls, label=name)
        if name.startswith("Rotating"):
            ax[2].plot(ref.t, np.rad2deg(att[:, 2]), ls, label=name + " yaw")
            ax[2].plot(ref.t, np.rad2deg(att[:, 0]), ":", label=name + " roll")
    ax[0].set_ylabel("horizontal error (m)")
    ax[1].set_ylabel("altitude error (m)")
    ax[1].set_yscale("symlog", linthresh=10)
    ax[2].set_ylabel("vehicle attitude err (deg)")
    ax[3].plot(ref.t, np.rad2deg(wrap_angle(enc.theta)), lw=0.5)
    ax[3].set_ylabel("encoder angle (deg)")
    ax[3].set_xlabel("time (s)")
    for x in ax:
        x.axvspan(a.gnss_until, a.duration, color="0.9", zorder=0)
        x.grid(True, alpha=0.3)
    for x in ax[:3]:
        x.legend(fontsize=7, loc="upper left")
    fig.suptitle(f"Rotating IMU about X + encoder, {a.imu} IMU (shaded: no GNSS)")
    fig.tight_layout()
    fig.savefig(a.out, dpi=120)
    print("saved", a.out)


if __name__ == "__main__":
    main()
