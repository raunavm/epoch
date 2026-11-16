"""Redraw data/findings/e28_rdra.png (scripts/run_e28.py, stage_fig) from stored results only.

Input: data/findings/e28.json (rdra.detectors: binned light curve and fit), data/findings/e9.json (rdra.post_hoc_guard_off:
predicted amplitude and the earlier pictures), and the comparison-star band from run_e9's frozen noise model, as in the original.
Run from the repo root: OMP_NUM_THREADS=1 .venv/bin/python figures/plot_e28_rdra.py
"""
import json
import os
import sys

import numpy as np
from astropy.time import Time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from common import ROLES, WIDTH, panel_label, plt, save, use_style  # noqa: E402
import run_e9 as R  # noqa: E402

BAND = "#d6d6d6"
use_style()
r = json.load(open(os.path.join(ROOT, "data/findings/e28.json")))["rdra"]
ph = json.load(open(os.path.join(ROOT, "data/findings/e9.json")))["targets"]["rdra"]["post_hoc_guard_off"]
F = json.load(open(f"{R.OUT}/frozen.json"))["classes"]
model = {int(d): tuple(m) for d, m in F["rdra"]["sigma_model"].items()}
P, T_PRED = r["period_d"], r["predicted_max_mjd"]
A_pred = ph["date_of_maximum"][0]["semi_amplitude_mag"] / 1.0857
S, M = ROLES["spherex"], ROLES["model"]

fig, axs = plt.subplots(2, 1, figsize=(WIDTH, 4.8), sharex=True)
allm = [x["mjd"] for d in ("5", "6") for x in r["detectors"][d]["cells"]] + [m for q in ph["light_curve"] for m in q["mjd"]]
tt = np.linspace(min(allm) - 15, max(allm) + 15, 600)
for ax, d, letter in zip(axs, (6, 5), "ab"):
    panel_label(ax, letter)
    rec = r["detectors"][str(d)]
    k = r["k"][str(d)]
    band = R.cell_sigma(model, [d], [r["n_cap"]])[0] * k
    ax.axhspan(1 - band, 1 + band, color=BAND, lw=0, zorder=1)
    c0 = 0.0
    if "fit" in rec:
        c = rec["fit"]["c"]
        c0 = c[0]
        ax.plot(tt, np.exp(c[0] + c[1] * np.cos(2 * np.pi * tt / P) + c[2] * np.sin(2 * np.pi * tt / P)),
                color=S["color"], lw=1.4, zorder=3)
    ax.plot(tt, np.exp(c0 + A_pred * np.cos(2 * np.pi * (tt - T_PRED) / P)), color=M["color"], ls=M["ls"], lw=1.3,
            zorder=2)
    old = [x for x in ph["light_curve"] if x["detector"] == d]
    if old:
        ax.plot(old[0]["mjd"], 10 ** (-0.4 * np.array(old[0]["rel_mag"])), ls="none", marker="o", ms=5.5, mfc="white",
                mec=S["color"], mew=1.1, zorder=4)
    cells = rec["cells"]
    ax.errorbar([x["mjd"] for x in cells], [1 + x["a"] for x in cells], yerr=[x["sigma"] * k for x in cells],
                fmt=S["marker"], ms=5.5, color=S["color"], elinewidth=1.2, mec="white", mew=0.7, zorder=5)
    ax.set_ylabel("Relative flux\n" + ("4.4–5.0 µm" if d == 6 else "3.8–4.4 µm"))

ticks = Time([f"{y}-{m:02d}-01" for y, m in ((2025, 5), (2025, 8), (2025, 11), (2026, 2), (2026, 5))]).mjd
axs[1].set_xticks(ticks)
axs[1].set_xticklabels(["May\n2025", "Aug", "Nov", "Feb\n2026", "May"])
axs[1].set_xlim(tt[0], tt[-1])
axs[1].set_xlabel("Date")

h = [plt.Line2D([], [], marker=S["marker"], ls="none", ms=5.5, color=S["color"], mec="white", mew=0.7,
                label="Pictures measured for this test"),
     plt.Line2D([], [], marker="o", ls="none", ms=5.5, mfc="white", mec=S["color"], mew=1.1,
                label="Pictures used in the earlier test"),
     plt.Rectangle((0, 0), 1, 1, color=BAND, lw=0, label="Comparison stars, ±1σ"),
     plt.Line2D([], [], color=S["color"], lw=1.4, label=f"Fitted sine, period {P:.0f} d"),
     plt.Line2D([], [], color=M["color"], ls=M["ls"], lw=1.3, label="Predicted")]
fig.legend(handles=h, loc="outside lower center", ncol=3, columnspacing=1.2, handletextpad=0.4)
save(fig, "e28_rdra")
