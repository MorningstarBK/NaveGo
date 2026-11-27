"""Generate all figures and result tables for the study document (docs/index.html).

Run from python/:  python docs/make_figures.py
Scenario: industrial IMU, 600 s vehicle trajectory, GNSS only for the first 60 s,
IMU on a turntable rotating 36 deg/s about vehicle X, Y or Z (or fixed), encoder 16 bit.
Aiding after GNSS loss: none / baro / magnetometer / both.
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from pynavego.attitude import euler2dcm, skew, wrap_angle  # noqa: E402
from pynavego.earth import radius  # noqa: E402
from pynavego.fusion import initial_state, run_fusion  # noqa: E402
from pynavego.metrics import nav_errors, rmse  # noqa: E402
from pynavego.sim import (generate_trajectory, rotate_trajectory, rotation_angle,  # noqa: E402
                          simulate_baro, simulate_encoder, simulate_gnss, simulate_imu,
                          simulate_mag, vehicle_attitude)

FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
T, T_GNSS, RATE = 600.0, 60.0, 36.0
C_AID = {"none": "#9aa3ad", "baro": "#1f77b4", "mag": "#d62728", "both": "#2ca02c"}
L_AID = {"none": "Chỉ IMU", "baro": "IMU + Baro", "mag": "IMU + Từ kế", "both": "IMU + Từ kế + Baro"}
MOUNTS = ["Quay X", "Quay Y", "Quay Z", "Cố định"]
AXIS = {"Quay X": "x", "Quay Y": "y", "Quay Z": "z", "Cố định": None}
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 110, "savefig.bbox": "tight"})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=110)
    plt.close(fig)
    print("saved", name)


def shade(ax):
    ax.axvspan(T_GNSS, T, color="0.92", zorder=0)


ref = generate_trajectory(duration=T)
theta = rotation_angle(ref.t, RATE)
enc = simulate_encoder(ref.t, theta)
baro = simulate_baro(ref, seed=1)
gnss = simulate_gnss(ref, outages=[(T_GNSS + 1e-6, T + 1)], seed=2)
out = ref.t > T_GNSS
g = 9.79

runs, rows = {}, []
for m in MOUNTS:
    ax = AXIS[m]
    sref = rotate_trajectory(ref, theta, ax) if ax else ref
    th = enc.theta if ax else np.zeros_like(ref.t)
    imu, _ = simulate_imu(sref, "industrial", seed=0)
    mag = simulate_mag(sref, seed=5)
    init = initial_state(sref)
    for aid, kw in (("none", {}), ("baro", dict(baro=baro)), ("mag", dict(mag=mag)),
                    ("both", dict(baro=baro, mag=mag))):
        res = run_fusion(imu, init, gnss=gnss, **kw)
        err = nav_errors(res, ref)
        att = wrap_angle(vehicle_attitude(res.euler, th, ax or "x") - ref.euler)
        hz = err["horizontal"]
        # attitude error vector phi of the sensor frame (C_est = (I - [phi x]) C_true)
        k = np.arange(0, len(ref.t), 10)
        phi = np.array([-0.5 * np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]])
                        for E in (euler2dcm(*res.euler[i]) @ sref.C_nb[i].T for i in k)])
        dab_n = np.einsum("kij,kj->ki", sref.C_nb[k], imu.err["ab_true"][k] - res.ab[k])
        runs[(m, aid)] = dict(err=err, att=att, res=res, imu=imu, sref=sref, k=k, phi=phi,
                              dab_n=dab_n, mag=mag)
        rows.append(dict(mount=m, aid=L_AID[aid], hor_end=hz[-1], hor_max=hz[out].max(),
                         hor_rmse=rmse(hz, out), hor_rate=hz[-1] / ((T - T_GNSS) / 60),
                         h_end=err["h"][-1], h_rmse=rmse(err["h"], out),
                         roll=np.rad2deg(rmse(att[:, 0], out)), pitch=np.rad2deg(rmse(att[:, 1], out)),
                         yaw=np.rad2deg(rmse(att[:, 2], out))))
        print(m, aid, round(hz[-1], 1), round(err["h"][-1], 2))

json.dump(rows, open(os.path.join(HERE, "results.json"), "w"), ensure_ascii=False, indent=1)

# ---------- F1 trajectory ----------
RM, RN = radius(ref.lat[0])
N = (ref.lat - ref.lat[0]) * (RM + ref.h); E = (ref.lon - ref.lon[0]) * (RN + ref.h) * np.cos(ref.lat[0])
fig, a = plt.subplots(1, 3, figsize=(13, 3.8))
sc = a[0].scatter(E / 1000, N / 1000, c=ref.t, s=1, cmap="viridis"); a[0].set_aspect("equal")
a[0].plot(E[ref.t <= T_GNSS] / 1000, N[ref.t <= T_GNSS] / 1000, "k", lw=3, label="có GNSS (0–60 s)")
a[0].set_xlabel("Đông (km)"); a[0].set_ylabel("Bắc (km)"); a[0].legend(fontsize=8); a[0].set_title("Quỹ đạo phẳng")
fig.colorbar(sc, ax=a[0], label="t (s)")
a[1].plot(ref.t, ref.h, "k"); a[1].set_title("Độ cao (m)"); shade(a[1])
a[2].plot(ref.t, np.linalg.norm(ref.vel, axis=1), label="tốc độ (m/s)")
a[2].plot(ref.t, np.rad2deg(ref.euler[:, 0]), label="roll (°)")
a[2].plot(ref.t, np.rad2deg(ref.euler[:, 1]), label="pitch (°)"); a[2].legend(fontsize=8)
a[2].set_title("Động học xe"); shade(a[2])
for x in a[1:]: x.set_xlabel("t (s)")
save(fig, "f1_trajectory.png")

# ---------- F2 horizontal error ----------
fig, a = plt.subplots(1, 4, figsize=(14, 3.8), sharey=True)
for i, m in enumerate(MOUNTS):
    for aid in C_AID:
        a[i].semilogy(ref.t, np.maximum(runs[(m, aid)]["err"]["horizontal"], 0.1), color=C_AID[aid], label=L_AID[aid])
    a[i].set_title(m); shade(a[i]); a[i].set_xlabel("t (s)"); a[i].set_ylim(0.3, 3e4)
a[0].set_ylabel("Sai số ngang (m)"); a[0].legend(fontsize=7, loc="upper left")
save(fig, "f2_horizontal.png")

# ---------- F3 altitude error ----------
fig, a = plt.subplots(1, 4, figsize=(14, 3.6), sharey=True)
for i, m in enumerate(MOUNTS):
    for aid in C_AID:
        a[i].plot(ref.t, runs[(m, aid)]["err"]["h"], color=C_AID[aid], label=L_AID[aid])
    a[i].set_yscale("symlog", linthresh=2); a[i].set_title(m); shade(a[i]); a[i].set_xlabel("t (s)")
a[0].set_ylabel("Sai số độ cao (m)"); a[0].legend(fontsize=7, loc="lower left")
save(fig, "f3_altitude.png")

# ---------- F4 attitude ----------
fig, a = plt.subplots(3, 4, figsize=(14, 7.5), sharex=True)
for i, m in enumerate(MOUNTS):
    for aid in C_AID:
        for j in range(3):
            a[j, i].plot(ref.t, np.rad2deg(runs[(m, aid)]["att"][:, j]), color=C_AID[aid], lw=0.9, label=L_AID[aid])
    a[0, i].set_title(m)
    for j, n in enumerate(["roll", "pitch", "yaw"]):
        a[j, i].set_ylim(-1.5, 1.5); shade(a[j, i])
        if i == 0: a[j, i].set_ylabel(f"Sai số {n} (°)")
    a[2, i].set_xlabel("t (s)")
a[0, 0].legend(fontsize=7, ncol=2)
save(fig, "f4_attitude.png")

# ---------- F5 bars ----------
fig, a = plt.subplots(figsize=(10, 3.8))
w = 0.2
for j, aid in enumerate(C_AID):
    v = [runs[(m, aid)]["err"]["horizontal"][-1] for m in MOUNTS]
    b = a.bar(np.arange(4) + (j - 1.5) * w, v, w, color=C_AID[aid], label=L_AID[aid])
    a.bar_label(b, fmt="%.0f", fontsize=7)
a.set_yscale("log"); a.set_xticks(range(4)); a.set_xticklabels(MOUNTS)
a.set_ylabel("Trôi ngang sau 9 phút (m)"); a.legend(fontsize=8, ncol=4); a.grid(axis="x")
save(fig, "f5_bars.png")

# ---------- F6/F7 bias estimation ----------
def bias_fig(key, true_key, unit_scale, unit, name):
    fig, a = plt.subplots(2, 3, figsize=(13, 6), sharex=True)
    for r, m in enumerate(["Quay X", "Cố định"]):
        R = runs[(m, "both")]
        est = getattr(R["res"], key) * unit_scale
        tru = R["imu"].err[true_key] * unit_scale
        off = 9 if key == "gb" else 12
        sig = np.sqrt(R["res"].P_diag[:, off:off + 3]) * unit_scale
        for j in range(3):
            a[r, j].plot(ref.t, tru[:, j], "k", lw=1.2, label="thật")
            a[r, j].plot(ref.t, est[:, j], color="#d62728", label="ước lượng")
            a[r, j].fill_between(ref.t, est[:, j] - sig[:, j], est[:, j] + sig[:, j], color="#d62728", alpha=0.15, label="±1σ")
            a[r, j].set_title(f"{m} – trục {'XYZ'[j]} cảm biến"); shade(a[r, j])
        a[r, 0].set_ylabel(unit)
    a[0, 0].legend(fontsize=8)
    for x in a[1]: x.set_xlabel("t (s)")
    save(fig, name)


bias_fig("gb", "gb_true", 180 / np.pi * 3600, "bias con quay (°/h)", "f6_gyro_bias.png")
bias_fig("ab", "ab_true", 1000 / 9.80665, "bias gia tốc kế (mg)", "f7_accel_bias.png")

# ---------- F8 baro & mag bias ----------
fig, a = plt.subplots(1, 2, figsize=(13, 3.8))
R = runs[("Quay X", "both")]
a[0].plot(baro.t, baro.data["bias_true"], "k", label="thật")
a[0].plot(ref.t, R["res"].baro_bias, color="#1f77b4", label="ước lượng")
s = np.sqrt(R["res"].P_diag[:, 15])
a[0].fill_between(ref.t, R["res"].baro_bias - s, R["res"].baro_bias + s, color="#1f77b4", alpha=0.15)
a[0].set_title("Bias baro (Quay X, Từ kế + Baro)"); a[0].set_ylabel("m"); a[0].legend(fontsize=8); shade(a[0])
mb = np.array(R["mag"].params["bias"])
for j in range(3):
    a[1].plot(ref.t, R["res"].mag_bias[:, j], color=f"C{j}", label=f"ước lượng {'XYZ'[j]}")
    a[1].axhline(mb[j], color=f"C{j}", ls="--", lw=1)
a[1].set_title("Bias hard-iron từ kế (nét đứt = thật)"); a[1].set_ylabel("µT"); a[1].legend(fontsize=8); shade(a[1])
for x in a: x.set_xlabel("t (s)")
save(fig, "f8_baro_mag_bias.png")

# ---------- F9 rotation modulation ----------
b_s = np.array([1.0, 1.0, 1.0])
fig, a = plt.subplots(1, 4, figsize=(14, 3.4), sharey=True)
tt = ref.t[ref.t < 30]
for i, m in enumerate(MOUNTS):
    from pynavego.sim import rot_axis
    C = [rot_axis(AXIS[m], th) if AXIS[m] else np.eye(3) for th in theta[: len(tt)]]
    bn = np.array([c @ b_s for c in C])          # bias seen in vehicle frame (vehicle frozen)
    for j, n in enumerate("XYZ"):
        a[i].plot(tt, bn[:, j], color=f"C{j}", label=f"thành phần {n} xe")
        a[i].axhline(bn[:, j].mean(), color=f"C{j}", ls=":", lw=1)
    a[i].set_title(m); a[i].set_xlabel("t (s)")
a[0].set_ylabel("bias chiếu về khung xe"); a[0].legend(fontsize=7)
save(fig, "f9_modulation.png")

# ---------- F10 observability (1-sigma) ----------
fig, a = plt.subplots(1, 3, figsize=(14, 3.6))
for aid in C_AID:
    P = runs[("Quay X", aid)]["res"].P_diag
    a[0].semilogy(ref.t, np.rad2deg(np.sqrt(P[:, 2])), color=C_AID[aid], label=L_AID[aid])
    a[1].semilogy(ref.t, np.rad2deg(np.sqrt(P[:, 0])), color=C_AID[aid])
    a[2].semilogy(ref.t, np.sqrt(P[:, 8]), color=C_AID[aid])
a[0].set_title("σ yaw (°), Quay X"); a[1].set_title("σ nghiêng φ_N (°), Quay X"); a[2].set_title("σ độ cao (m), Quay X")
for x in a: shade(x); x.set_xlabel("t (s)")
a[0].legend(fontsize=7)
save(fig, "f10_sigma.png")

# ---------- F11 horizontal acceleration error budget ----------
fig, a = plt.subplots(1, 3, figsize=(14, 3.8), sharey=True)
for i, (m, aid) in enumerate([("Cố định", "mag"), ("Quay Z", "mag"), ("Quay X", "mag")]):
    R = runs[(m, aid)]; tk = ref.t[R["k"]]
    tilt = g * np.column_stack((R["phi"][:, 1], -R["phi"][:, 0]))   # f x phi, f=(0,0,-g)
    bias = R["dab_n"][:, :2]
    tot = tilt + bias
    win = 100
    # average the vector over 10 s first (what integrates into drift), then take the norm
    sm = lambda x: np.linalg.norm(np.column_stack([np.convolve(c, np.ones(win) / win, "same")
                                                   for c in x.T]), axis=1)
    a[i].plot(tk, sm(tilt) * 1e3 / 9.80665, label="|g·φ| (nghiêng)")
    a[i].plot(tk, sm(bias) * 1e3 / 9.80665, label="|C·δb_a| (bias còn lại)")
    a[i].plot(tk, sm(tot) * 1e3 / 9.80665, "k", lw=2, label="|tổng| → gây trôi")
    a[i].set_title(f"{m}, IMU + Từ kế"); shade(a[i]); a[i].set_xlabel("t (s)")
a[0].set_ylabel("sai số gia tốc ngang (mg)"); a[0].legend(fontsize=8)
save(fig, "f11_budget.png")
