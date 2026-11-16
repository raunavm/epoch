"""Redraw site/data/figs/e9_rdra.png (scripts/e9_targets.py, _fig_rdra()) from stored results only.

Input: data/findings/e9.json (targets.rdra: light_curve, date_of_maximum, post_hoc_guard_off, a_dw_not_used).
Run from the repo root: .venv/bin/python figures/plot_e9_rdra.py
"""
import json
import os
import sys

import numpy as np
from astropy.time import Time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import INK, ROLES, WIDTH, panel_label, plt, save, use_style

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
BAND = "#d6d6d6"  # opaque tint of the control grey (comparison-star scatter)

use_style()
t = json.load(open(os.path.join(ROOT, "data/findings/e9.json")))["targets"]["rdra"]
S, R, M, X = ROLES["spherex"], ROLES["reference"], ROLES["model"], ROLES["secondary"]

lcs = {q["detector"]: q for q in t["light_curve"]}
tm = {q["detector"]: q for q in t["date_of_maximum"]["per_detector"]}
ph = t.get("post_hoc_guard_off", {})
lph = {q["detector"]: q for q in ph.get("light_curve", []) if q["detector"] not in lcs}
tph = {q["detector"]: q for q in ph.get("date_of_maximum", []) if q["detector"] in lph}
P, e0 = t["date_of_maximum"]["period_d"], t["date_of_maximum"]["catalogue_epoch_mjd"]
allm = np.concatenate([q["mjd"] for q in list(lcs.values()) + list(lph.values())])
ally = np.concatenate([q["rel_mag"] for q in list(lcs.values()) + list(lph.values())])

fig, axs = plt.subplots(2, 3, figsize=(WIDTH, 4.6), sharex=True, sharey=True)
ticks = Time([f"{y}-{m:02d}-01" for y, m in ((2025, 5), (2025, 8), (2025, 11), (2026, 2), (2026, 5))]).mjd
for d, ax in zip(range(1, 7), axs.ravel()):
    panel_label(ax, "abcdef"[d - 1])
    s = float(np.median(lcs[d]["rel_mag_err"])) / 1.0857 if d in lcs else np.nan
    if np.isfinite(s):
        ax.axhspan(-1.0857 * s, 1.0857 * s, color=BAND, lw=0, zorder=1)
    ax.axhline(0, color=INK, lw=0.6, zorder=2)
    for k in range(int(np.ceil((allm.min() - 20 - e0) / P)), int(np.floor((allm.max() + 20 - e0) / P)) + 1):
        ax.axvline(e0 + k * P, color=R["color"], lw=1.2, ls=R["ls"], zorder=2)
    if d in lcs or d in lph:
        q = lcs.get(d, lph.get(d))
        post_hoc = d not in lcs
        if d in tm or d in tph:
            tt = np.linspace(allm.min() - 10, allm.max() + 10, 300)
            c = tm.get(d, tph.get(d))
            phase = 2 * np.pi * (np.array(q["mjd"]) - c["mjd_max"]) / P
            off = np.mean(np.array(q["rel_mag"]) + c["semi_amplitude_mag"] * np.cos(phase))
            ax.plot(tt, off - c["semi_amplitude_mag"] * np.cos(2 * np.pi * (tt - c["mjd_max"]) / P), color=M["color"],
                    ls=M["ls"], lw=1.3, zorder=3)
        ax.errorbar(q["mjd"], q["rel_mag"], yerr=q["rel_mag_err"], fmt="D" if post_hoc else S["marker"],
                    ms=5 if post_hoc else 5.5, color=S["color"], elinewidth=1.2, mec="white", mew=0.7, zorder=4)
    else:
        nu = [q for q in t["a_dw_not_used"] if q["detector"] == d and q["a"] > -0.99]
        if nu:
            ax.plot([q["mjd"] for q in nu], [-2.5 * np.log10(1 + q["a"]) for q in nu], ls="none", marker=X["marker"],
                    ms=5.5, color=X["color"], mec="white", mew=0.6, zorder=3)
    ax.set_xticks(ticks)
    ax.set_xticklabels(["May\n2025", "Aug", "Nov", "Feb\n2026", "May"])
pad = 0.15 * (ally.max() - ally.min() + 0.1)
axs[0, 0].set_ylim(ally.max() + pad, ally.min() - pad)
axs[0, 0].set_xlim(allm.min() - 15, allm.max() + 15)
fig.supylabel("Brightness change (mag)", fontsize=11)
axs[1, 1].set_xlabel("Date")

h = [plt.Line2D([], [], marker=S["marker"], ls="none", ms=5.5, color=S["color"], mec="white", mew=0.7,
                label="SPHEREx, 30-day mean"),
     plt.Line2D([], [], marker="D", ls="none", ms=5, color=S["color"], mec="white", mew=0.7,
                label="SPHEREx, post hoc"),
     plt.Line2D([], [], marker=X["marker"], ls="none", ms=5.5, color=X["color"], mec="white", mew=0.6,
                label="SPHEREx, band not used"),
     plt.Rectangle((0, 0), 1, 1, color=BAND, lw=0, label="Comparison stars"),
     plt.Line2D([], [], color=R["color"], ls=R["ls"], lw=1.2, label="Catalogue maximum"),
     plt.Line2D([], [], color=M["color"], ls=M["ls"], lw=1.3, label=f"Sine, period {P:.0f} d")]
fig.legend(handles=[h[i] for i in (0, 3, 1, 4, 2, 5)], loc="outside lower center", ncol=3, columnspacing=1.2,
           handletextpad=0.4)
save(fig, "e9_rdra")
