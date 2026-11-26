"""Monte Carlo over IMU/sensor error seeds: final horizontal drift and altitude error per case.

Run from python/:  python docs/monte_carlo.py [n_seeds]   -> docs/montecarlo.json
"""
import json, os, sys
import numpy as np
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from pynavego.fusion import initial_state, run_fusion  # noqa: E402
from pynavego.metrics import nav_errors, rmse  # noqa: E402
from pynavego.attitude import wrap_angle  # noqa: E402
from pynavego.sim import (generate_trajectory, rotate_trajectory, rotation_angle, simulate_baro,  # noqa: E402
                          simulate_encoder, simulate_gnss, simulate_imu, simulate_mag, vehicle_attitude)

T, TG = 600.0, 60.0
AX = {"Quay X": "x", "Quay Y": "y", "Quay Z": "z", "Cố định": None}


def one(args):
    seed, m = args
    ref = generate_trajectory(duration=T)
    th = rotation_angle(ref.t, 36.0)
    enc = simulate_encoder(ref.t, th, seed=100 + seed)
    ax = AX[m]
    sref = rotate_trajectory(ref, th, ax) if ax else ref
    eth = enc.theta if ax else np.zeros_like(ref.t)
    imu, _ = simulate_imu(sref, "industrial", seed=seed)
    baro = simulate_baro(ref, seed=200 + seed)
    mag = simulate_mag(sref, seed=300 + seed)
    gnss = simulate_gnss(ref, outages=[(TG + 1e-6, T + 1)], seed=400 + seed)
    out = ref.t > TG
    res = {}
    for aid, kw in (("none", {}), ("baro", dict(baro=baro)), ("mag", dict(mag=mag)),
                    ("both", dict(baro=baro, mag=mag))):
        r = run_fusion(imu, initial_state(sref), gnss=gnss, **kw)
        e = nav_errors(r, ref)
        att = wrap_angle(vehicle_attitude(r.euler[::10], eth[::10], ax or "x") - ref.euler[::10])
        o = out[::10]
        res[aid] = dict(hor=float(e["horizontal"][-1]), hor_rmse=float(rmse(e["horizontal"], out)),
                        hor_max=float(e["horizontal"][out].max()), h_rmse=float(rmse(e["h"], out)), h=float(abs(e["h"][-1])),
                        tilt=float(np.rad2deg(rmse(np.hypot(att[:, 0], att[:, 1]), o))),
                        yaw=float(np.rad2deg(rmse(att[:, 2], o))))
    return seed, m, res


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    jobs = [(s, m) for s in range(n) for m in AX]
    with ProcessPoolExecutor() as ex:
        out = list(ex.map(one, jobs))
    data = {}
    for s, m, r in out:
        for aid, v in r.items():
            data.setdefault(m, {}).setdefault(aid, []).append(v)
    json.dump(data, open(os.path.join(HERE, "montecarlo.json"), "w"), ensure_ascii=False)
    for m in AX:
        for aid in ("none", "baro", "mag", "both"):
            h = np.array([d["hor_rmse"] for d in data[m][aid]])
            print(f"{m:8s} {aid:5s} median {np.median(h):8.0f}  p10 {np.percentile(h,10):8.0f}  p90 {np.percentile(h,90):8.0f}")
