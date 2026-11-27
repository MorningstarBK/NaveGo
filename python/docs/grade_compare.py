"""Compare IMU grades (navigation / tactical / industrial / consumer): how much do rotation
modulation and LOS range aiding help each grade?  GNSS only for the first 60 s.

Run from python/:  python docs/grade_compare.py [n_seeds]  -> docs/grade_compare.json, figures/f14_grades.png
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "examples"))
sys.path.insert(0, os.path.join(HERE, ".."))
from example_los_range import MOUNTS, run_case  # noqa: E402

GRADES = ["navigation", "tactical", "industrial", "consumer"]


def job(a):
    s, m, g = a
    return a, run_case(s, m, g)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    if len(sys.argv) > 2:
        GRADES = sys.argv[2].split(",")
    tag = sys.argv[3] if len(sys.argv) > 3 else ""
    jobs = [(s, m, g) for g in GRADES for m in MOUNTS for s in range(n)]
    with ProcessPoolExecutor() as ex:
        out = dict(ex.map(job, jobs))
    data = {g: {m: {aid: [out[(s, m, g)][aid] for s in range(n)] for aid in ("none", "los")}
                for m in MOUNTS} for g in GRADES}
    json.dump(data, open(os.path.join(HERE, f"grade_compare{tag}.json"), "w"), ensure_ascii=False)
    med = lambda g, m, a, k: float(np.median([d[k] for d in data[g][m][a]]))

    print(f"Monte Carlo {n} seeds, median horizontal RMSE (m) during the GNSS outage")
    print("| Cấp IMU | IMU | Chỉ IMU | IMU + LOS | Lợi ích LOS (×) | Lợi ích quay (×, chỉ IMU) | Lợi ích quay (×, + LOS) | RMSE độ cao chỉ IMU / + LOS (m) | Yaw chỉ IMU / + LOS (°) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for g in GRADES:
        fn, fl = med(g, "Cố định", "none", "hor_rmse"), med(g, "Cố định", "los", "hor_rmse")
        for m in MOUNTS:
            a0, a1 = med(g, m, "none", "hor_rmse"), med(g, m, "los", "hor_rmse")
            rq0 = f"{fn / a0:.1f}" if m != "Cố định" else "–"
            rq1 = f"{fl / a1:.1f}" if m != "Cố định" else "–"
            print(f"| {g} | {m} | {a0:.1f} | {a1:.1f} | {a0 / a1:.1f} | {rq0} | {rq1} | "
                  f"{med(g, m, 'none', 'h_rmse'):.1f} / {med(g, m, 'los', 'h_rmse'):.1f} | "
                  f"{med(g, m, 'none', 'yaw'):.3f} / {med(g, m, 'los', 'yaw'):.3f} |")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.4))
    w = 0.2
    for j, m in enumerate(MOUNTS):
        x = np.arange(len(GRADES)) + (j - 1.5) * w
        v0 = [med(g, m, "none", "hor_rmse") for g in GRADES]
        v1 = [med(g, m, "los", "hor_rmse") for g in GRADES]
        ax[0].bar(x, v0, w, color=f"C{j}", alpha=0.35)
        ax[0].bar(x, v1, w, color=f"C{j}", label=m)
        ax[1].bar(x, np.array(v0) / np.array(v1), w, color=f"C{j}", label=m)
    for a in ax:
        a.set_xticks(range(len(GRADES))); a.set_xticklabels(GRADES); a.set_yscale("log"); a.grid(axis="y", alpha=0.3)
    ax[0].set_ylabel("RMSE ngang (m)"); ax[0].set_title("Nhạt: chỉ IMU; đậm: IMU + LOS")
    ax[1].set_ylabel("Chỉ IMU / IMU + LOS (lần)"); ax[1].set_title("Mức cải thiện nhờ LOS")
    ax[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "figures", f"f14_grades{tag}.png"), dpi=110)
    print(f"saved figures/f14_grades{tag}.png")
