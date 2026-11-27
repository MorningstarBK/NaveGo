"""Fill docs/template.html with numbers from results.json and montecarlo.json -> docs/index.html."""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
mc = json.load(open(os.path.join(HERE, "montecarlo.json")))
one = json.load(open(os.path.join(HERE, "results.json")))
MOUNTS = ["Quay X", "Quay Y", "Quay Z", "Cố định"]
AIDS = ["none", "baro", "mag", "both"]
LAB = {"none": "Chỉ IMU", "baro": "IMU + Baro", "mag": "IMU + Từ kế", "both": "IMU + Từ kế + Baro"}
COL = {"none": "var(--none)", "baro": "var(--baro)", "mag": "var(--mag)", "both": "var(--both)"}
NMC = len(mc["Quay X"]["none"])
los = json.load(open(os.path.join(HERE, "los_montecarlo.json")))
lmed = lambda m, a, k: float(np.median([d[k] for d in los[m][a]]))
lpct = lambda m, a, k, q: float(np.percentile([d[k] for d in los[m][a]], q))
LLAB = {"none": "Chỉ IMU", "los": "IMU + LOS"}


def med(m, a, k):
    return float(np.median([d[k] for d in mc[m][a]]))


def pct(m, a, k, q):
    return float(np.percentile([d[k] for d in mc[m][a]], q))


def f0(x):
    return f"{x:,.0f}".replace(",", ".")


lo, hi = 1.0, np.log10(20000)
pos = lambda v: max(0.0, (np.log10(max(v, 10)) - lo) / (hi - lo) * 100)

bars = []
for m in MOUNTS:
    rows = []
    for a in AIDS:
        v, p10, p90 = med(m, a, "hor_rmse"), pct(m, a, "hor_rmse", 10), pct(m, a, "hor_rmse", 90)
        rows.append(f'<div class="row"><span class="lab">{LAB[a]}</span><div class="trk">'
                    f'<i style="left:{pos(p10):.1f}%;width:{pos(p90) - pos(p10):.1f}%"></i>'
                    f'<s style="left:0;width:{pos(v):.1f}%;background:{COL[a]}"></s></div>'
                    f'<span class="v">{f0(v)} m</span></div>')
    bars.append(f'<div class="grp"><b>{m}</b><div>{"".join(rows)}</div></div>')

best_m = min(("Quay X", "Quay Y"), key=lambda m: med(m, "both", "hor_rmse"))
best = med(best_m, "both", "hor_rmse")
fix_none = med("Cố định", "none", "hor_rmse")
kpis = [
    (f"{f0(best)} m", f"RMSE ngang trung vị, {best_m} + từ kế + baro"),
    (f"{f0(fix_none)} m", "RMSE ngang trung vị, IMU cố định, không hỗ trợ"),
    (f"{med('Quay X', 'both', 'h_rmse'):.1f} m".replace(".", ","), "RMSE độ cao khi có baro"),
    (f"{med('Quay X', 'mag', 'yaw'):.2f}°".replace(".", ","), "RMSE yaw khi có từ kế"),
    (f"{f0(lmed('Quay X', 'los', 'hor_rmse'))} m", "RMSE ngang trung vị, Quay X + LOS (không baro, không từ kế)"),
]
kpi_html = "".join(f'<div class="kpi"><b>{a}</b><span>{b}</span></div>' for a, b in kpis)


def table_mc():
    h = ("<table><thead><tr><th>IMU</th><th>Hỗ trợ</th><th>RMSE ngang (m)</th><th>p10–p90 (m)</th>"
         "<th>Ngang max (m)</th><th>Trôi cuối (m)</th><th>m/phút</th><th>RMSE độ cao (m)</th>"
         "<th>|Độ cao cuối| (m)</th><th>RMSE nghiêng (°)</th><th>RMSE yaw (°)</th></tr></thead><tbody>")
    for m in MOUNTS:
        bestA = min(AIDS, key=lambda a: med(m, a, "hor_rmse"))
        for a in AIDS:
            cls = ' class="best"' if a == bestA else ""
            h += (f"<tr{cls}><td>{m}</td><td>{LAB[a]}</td><td>{f0(med(m, a, 'hor_rmse'))}</td>"
                  f"<td>{f0(pct(m, a, 'hor_rmse', 10))}–{f0(pct(m, a, 'hor_rmse', 90))}</td>"
                  f"<td>{f0(med(m, a, 'hor_max'))}</td><td>{f0(med(m, a, 'hor'))}</td>"
                  f"<td>{med(m, a, 'hor') / 9:.0f}</td><td>{med(m, a, 'h_rmse'):.2f}</td>"
                  f"<td>{med(m, a, 'h'):.2f}</td><td>{med(m, a, 'tilt'):.3f}</td><td>{med(m, a, 'yaw'):.3f}</td></tr>")
    return h + "</tbody></table>"


def table_one():
    h = ("<table><thead><tr><th>IMU</th><th>Hỗ trợ</th><th>Trôi ngang cuối (m)</th><th>Ngang max (m)</th>"
         "<th>RMSE ngang (m)</th><th>m/phút</th><th>Độ cao cuối (m)</th><th>RMSE độ cao (m)</th>"
         "<th>Roll / Pitch / Yaw RMSE (°)</th></tr></thead><tbody>")
    for r in one:
        h += (f"<tr><td>{r['mount']}</td><td>{r['aid']}</td><td>{r['hor_end']:.1f}</td><td>{r['hor_max']:.1f}</td>"
              f"<td>{r['hor_rmse']:.1f}</td><td>{r['hor_rate']:.1f}</td><td>{r['h_end']:.2f}</td>"
              f"<td>{r['h_rmse']:.2f}</td><td>{r['roll']:.3f} / {r['pitch']:.3f} / {r['yaw']:.3f}</td></tr>")
    return h + "</tbody></table>"




def table_los():
    h = ("<table><thead><tr><th>IMU</th><th>Hỗ trợ</th><th>RMSE ngang (m)</th><th>p10–p90 (m)</th>"
         "<th>Ngang max (m)</th><th>Trôi cuối (m)</th><th>Dọc LOS (m)</th><th>Vuông góc LOS (m)</th>"
         "<th>RMSE khoảng cách (m)</th><th>RMSE độ cao (m)</th><th>Nghiêng (°)</th><th>Yaw (°)</th></tr></thead><tbody>")
    for m in MOUNTS:
        for a in ("none", "los"):
            cls = ' class="best"' if a == "los" else ""
            h += (f"<tr{cls}><td>{m}</td><td>{LLAB[a]}</td><td>{f0(lmed(m, a, 'hor_rmse'))}</td>"
                  f"<td>{f0(lpct(m, a, 'hor_rmse', 10))}–{f0(lpct(m, a, 'hor_rmse', 90))}</td>"
                  f"<td>{f0(lmed(m, a, 'hor_max'))}</td><td>{f0(lmed(m, a, 'hor_end'))}</td>"
                  f"<td>{lmed(m, a, 'rad_rmse'):.1f}</td><td>{f0(lmed(m, a, 'tan_rmse'))}</td>"
                  f"<td>{lmed(m, a, 'rng_rmse'):.2f}</td><td>{lmed(m, a, 'h_rmse'):.1f}</td>"
                  f"<td>{lmed(m, a, 'tilt'):.3f}</td><td>{lmed(m, a, 'yaw'):.3f}</td></tr>")
    return h + "</tbody></table>"


def table_compare():
    h = ("<table><thead><tr><th>IMU</th><th>Chỉ IMU</th><th>Từ kế + Baro: ngang (m)</th><th>LOS: ngang (m)</th>"
         "<th>Từ kế + Baro: độ cao (m)</th><th>LOS: độ cao (m)</th></tr></thead><tbody>")
    for m in MOUNTS:
        h += (f"<tr><td>{m}</td><td>{f0(med(m, 'none', 'hor_rmse'))}</td><td>{f0(med(m, 'both', 'hor_rmse'))}</td>"
              f"<td><b>{f0(lmed(m, 'los', 'hor_rmse'))}</b></td><td><b>{med(m, 'both', 'h_rmse'):.1f}</b></td>"
              f"<td>{lmed(m, 'los', 'h_rmse'):.1f}</td></tr>")
    return h + "</tbody></table><p class=\"note\">RMSE trung vị trong lúc mất GNSS. Hai bộ Monte Carlo dùng cùng seed cho IMU và GNSS.</p>"


rep = {
    "{{BARS}}": "".join(bars), "{{KPIS}}": kpi_html, "{{NMC}}": str(NMC),
    "{{BEST}}": f0(best), "{{RATIO_FIX}}": f"{fix_none / best:.0f}",
    "{{FIX_NONE}}": f0(fix_none), "{{FIX_BARO}}": f0(med("Cố định", "baro", "hor_rmse")),
    "{{FIX_BOTH}}": f0(med("Cố định", "both", "hor_rmse")),
    "{{FIX_MAG_H}}": f0(med("Cố định", "mag", "h_rmse")),
    "{{YAW_MAG}}": f"{med('Cố định', 'mag', 'yaw'):.2f}–{med('Quay X', 'mag', 'yaw'):.2f}".replace(".", ","),
    "{{QX_BOTH}}": f0(med("Quay X", "both", "hor_rmse")), "{{QY_BOTH}}": f0(med("Quay Y", "both", "hor_rmse")),
    "{{QZ_NONE}}": f0(med("Quay Z", "none", "hor_rmse")), "{{QZ_BOTH}}": f0(med("Quay Z", "both", "hor_rmse")),
    "{{QZ_BARO}}": f0(med("Quay Z", "baro", "hor_rmse")),
    "{{LOS_NMC}}": str(len(los["Quay X"]["los"])), "{{LOS_TABLE}}": table_los(),
    "{{LOS_COMPARE}}": table_compare(),
    "{{LOS_QX}}": f0(lmed("Quay X", "los", "hor_rmse")), "{{LOS_QY}}": f0(lmed("Quay Y", "los", "hor_rmse")),
    "{{LOS_QZ}}": f0(lmed("Quay Z", "los", "hor_rmse")), "{{LOS_FIX}}": f0(lmed("Cố định", "los", "hor_rmse")),
    "{{LOS_QZ_YAW}}": f"{lmed('Quay Z', 'los', 'yaw'):.2f}".replace(".", ","),
    "{{LOS_QY_YAW}}": f"{lmed('Quay Y', 'los', 'yaw'):.2f}".replace(".", ","),
    "{{LOS_FIX_YAW}}": f"{lmed('Cố định', 'los', 'yaw'):.2f}".replace(".", ","),
    "{{TABLE_MC}}": table_mc(), "{{TABLE_ONE}}": table_one(),
}
s = open(os.path.join(HERE, "template.html"), encoding="utf-8").read()
for k, v in rep.items():
    s = s.replace(k, v)
assert "{{" not in s, s[s.index("{{"):s.index("{{") + 30]
open(os.path.join(HERE, "index.html"), "w", encoding="utf-8").write(s)
print("wrote docs/index.html")
