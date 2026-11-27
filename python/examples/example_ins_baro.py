"""Demo: inertial sensor simulation + INS aided by barometric altitude (and GNSS).

Run from the python/ folder:  python examples/example_ins_baro.py [--imu industrial] [--no-plot]
Scenarios:
  1. INS only                -> vertical channel diverges
  2. INS + baro              -> altitude bounded (to within the baro bias), horizontal still drifts
  3. INS + GNSS (outage)     -> altitude drifts during the outage
  4. INS + GNSS + baro       -> GNSS calibrates the baro bias, baro holds altitude in the outage
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pynavego.fusion import initial_state, run_fusion  # noqa: E402
from pynavego.metrics import nav_errors, rmse  # noqa: E402
from pynavego.sim import generate_trajectory, simulate_baro, simulate_gnss, simulate_imu  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imu", default="industrial", choices=["tactical", "industrial", "consumer"])
    ap.add_argument("--duration", type=float, default=600.0)
    ap.add_argument("--outage", type=float, nargs=2, default=(300.0, 420.0))
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--out", default="ins_baro_results.png")
    a = ap.parse_args()

    ref = generate_trajectory(duration=a.duration)
    imu, _ = simulate_imu(ref, a.imu, seed=0)
    baro = simulate_baro(ref, seed=1)
    gnss = simulate_gnss(ref, outages=[tuple(a.outage)], seed=2)
    init = initial_state(ref)

    scenarios = {
        "INS": {},
        "INS+Baro": dict(baro=baro),
        "INS+GNSS": dict(gnss=gnss),
        "INS+GNSS+Baro": dict(gnss=gnss, baro=baro),
    }
    out_mask = (ref.t >= a.outage[0]) & (ref.t <= a.outage[1])
    results = {}
    print(f"IMU: {a.imu}, duration {a.duration:.0f} s, GNSS outage {a.outage[0]:.0f}-{a.outage[1]:.0f} s")
    print(f"{'scenario':<15}{'h RMSE':>10}{'h RMSE out':>12}{'hor RMSE':>11}{'hor RMSE out':>14}")
    for name, kw in scenarios.items():
        res = run_fusion(imu, init, **kw)
        err = nav_errors(res, ref)
        results[name] = (res, err)
        print(f"{name:<15}{rmse(err['h']):>10.2f}{rmse(err['h'], out_mask):>12.2f}"
              f"{rmse(err['horizontal']):>11.2f}{rmse(err['horizontal'], out_mask):>14.2f}")
    res4 = results["INS+GNSS+Baro"][0]
    print(f"baro bias: true {baro.data['bias_true'][-1]:.2f} m, estimated {res4.baro_bias[-1]:.2f} m")

    if a.no_plot:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    for name, (res, err) in results.items():
        if name != "INS":
            ax[0].plot(ref.t, err["h"], label=name)
        ax[1].semilogy(ref.t, np.maximum(err["horizontal"], 1e-3), label=name)
    ax[0].set_ylabel("altitude error (m)")
    ax[0].set_ylim(-15, 15)
    ax[1].set_ylabel("horizontal error (m)")
    ax[2].plot(baro.t, baro.data["bias_true"], "k", label="true baro bias")
    ax[2].plot(ref.t, res4.baro_bias, label="estimated (INS+GNSS+Baro)")
    ax[2].set_ylabel("baro bias (m)")
    ax[2].set_xlabel("time (s)")
    for x in ax:
        x.axvspan(*a.outage, color="0.85", zorder=0)
        x.grid(True, alpha=0.3)
        x.legend(loc="upper left", fontsize=8)
    fig.suptitle(f"pynavego - {a.imu} IMU, shaded = GNSS outage")
    fig.tight_layout()
    fig.savefig(a.out, dpi=120)
    print("saved", a.out)


if __name__ == "__main__":
    main()
