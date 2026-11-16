"""Redraw site/data/figs/e20_lines.png (scripts/run_e20.py, stage_fig()) from stored results only.

Inputs (as in run_e20.stage_fig): data/findings/e20.json (stars, line excess Z, continuum per window),
data/findings/e7_gate0.json (per-detector error scale s_d), data/ref/spherex_exposure_anomaly_log_20250810.csv and the
stored per-picture photometry data/phase2/e20/phot/<field>.json. star_arrays() and anomaly_set() are copied from
run_e20.py. Example star: the candidate with the largest Z (the original default; it matches the original figure).
Run from the repo root: .venv/bin/python figures/plot_e20_lines.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import INK, ROLES, WIDTH, panel_label, plt, save, use_style

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
LINES = {"Paa": (1.8751, 3), "Brg": (2.1655, 3), "Bra": (4.0523, 5)}
MIN_PIX, Z_DET = 18, 5.0


def anomaly_set():
    a = pd.read_csv(os.path.join(ROOT, "data/ref/spherex_exposure_anomaly_log_20250810.csv"))
    return {(o, d) for o, row in zip(a.OBSID, a.itertuples()) for d in range(1, 7) if getattr(row, f"BAND_{d}_AFFECTED") > 0}


def star_arrays(r, s_d, anom):
    n = len(r["flux"])
    f = np.array([np.nan if v is None else v for v in r["flux"]], float)
    e = np.array([np.nan if v is None else v for v in r["err"]], float)
    det = np.array(r["det"], int)
    e = e * np.array([s_d[str(d)] for d in det]) if n else e
    bad_anom = np.array([(o, d) in anom for o, d in zip(r["obs_id"], det)], bool)
    ok = np.isfinite(f) & np.isfinite(e) & (e > 0) & (np.array(r["npix"]) >= MIN_PIX) & ~np.array(r["sat"], bool) & ~bad_anom
    a = {k: np.array(r[k])[ok] for k in ("lam", "bw", "mjd", "obs_id")}
    a.update(flux=f[ok], err=e[ok], det=det[ok])
    return a


use_style()
o = json.load(open(os.path.join(ROOT, "data/findings/e20.json")))
s_d = json.load(open(os.path.join(ROOT, "data/findings/e7_gate0.json")))["s_d"]
anom = anomaly_set()
rows = o["stars"]
cand = [r for r in rows if r["role"] == "candidate"]
key = lambda r: max(r["lines"][L].get("Z", -99) for L in ("Paa", "Bra"))
ex = max(cand, key=key)
ctl = next(r for r in rows if r["field"] == ex["field"] and r["role"] == "control")
ph = json.load(open(os.path.join(ROOT, f"data/phase2/e20/phot/{ex['field']}.json")))["stars"]
Hh, Cc = ROLES["highlight"], ROLES["control"]

fig = plt.figure(figsize=(WIDTH, 5.6))
gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
for j, (L, lo, hi) in enumerate((("Paa", 1.62, 2.30), ("Bra", 3.80, 4.42))):
    ax = fig.add_subplot(gs[0, j])
    lamL, dL = LINES[L]
    for r, Q, lab, ms in ((ex, Hh, "Example candidate (a, b)", 6.5), (ctl, Cc, "Control, same pictures (a, b)", 4.5)):
        a = star_arrays(ph[r["source_id"]], s_d, anom)
        s = (a["det"] == dL) & (a["lam"] >= lo) & (a["lam"] <= hi)
        cont = r["lines"][L].get("windows", {})
        norm = np.nanmedian([v["cont_mJy"] for v in cont.values() if "cont_mJy" in v]) if any("cont_mJy" in v for v in cont.values()) else np.nanmedian(a["flux"][s])
        ax.errorbar(a["lam"][s], a["flux"][s] / norm, yerr=a["err"][s] / norm, fmt=Q["marker"], ms=ms, color=Q["color"],
                    mew=0.9 if Q is Cc else 0.3, mec=Q["color"] if Q is Cc else "white", elinewidth=0.8,
                    zorder=3 if Q is Hh else 2, label=lab)
        if r is ex:
            q = np.percentile(a["flux"][s] / norm, [2, 98])
            ax.set_ylim(q[0] - 0.25 * (q[1] - q[0]) - 0.05, q[1] + 0.35 * (q[1] - q[0]) + 0.05)
        print(L, "candidate" if r is ex else "control", r["source_id"], "pictures", int(s.sum()))
    for l2, n2 in ((1.8751, "Paα"), (2.1655, "Brγ"), (4.0523, "Brα")):
        if lo < l2 < hi:
            ax.axvline(l2, color=INK, lw=0.8, ls=":", zorder=1, label="Hydrogen line positions (a, b)")
    ax.set_xlabel("Wavelength (µm)")
    ax.set_ylabel("Flux ÷ local continuum (ratio)" if j == 0 else "")
    panel_label(ax, "ab"[j])
    print(L, "candidate Z", ex["lines"][L].get("Z"))

ax = fig.add_subplot(gs[1, :])
bins = np.linspace(-8, 12, 41)
for role, Q in (("control", Cc), ("candidate", Hh)):
    z = [max(r["lines"][L].get("Z", -99) for L in ("Paa", "Bra")) for r in rows if r["role"] == role and r["testable_main"]]
    z = np.clip(z, bins[0], bins[-1] - 1e-6)
    lab = f"{role.capitalize()}s (n = {len(z)})"
    if role == "control":
        ax.hist(z, bins=bins, histtype="stepfilled", color=Q["color"], alpha=0.5, lw=0, weights=np.full(len(z), 1 / len(z)), label=lab)
    else:
        ax.hist(z, bins=bins, histtype="step", color=Q["color"], lw=1.6, weights=np.full(len(z), 1 / len(z)), label=lab)
ax.axvline(Z_DET, color=INK, lw=1.0, ls="--", label=f"Threshold Z = {Z_DET:g}")
ax.set_xlabel("Line excess Z, larger of Paα and Brα (control spread)")
ax.set_ylabel("Fraction of stars")
ax.set_xlim(-8.5, 12.5)
panel_label(ax, "c")
ha, la = fig.axes[0].get_legend_handles_labels()
hb, lb = ax.get_legend_handles_labels()
la, ha = zip(*sorted(dict(zip(la, ha)).items(), key=lambda x: x[0].startswith("Hydrogen")))  # one line entry, listed last
fig.legend([*ha, *hb], [*la, *lb], loc="outside upper center", ncol=3, columnspacing=1.2, handletextpad=0.4)
save(fig, "e20_lines")
ac = [o[f"cand_{c}"] for c in ("HerbigStar", "beStar")]
print("example", ex["source_id"], ex["cls"], f"{sum(x['N_excess'] for x in ac)} of {sum(x['M_testable'] for x in ac)} candidates; "
      f"{o['controls']['n_pass']} of {o['controls']['n_testable']} controls")
