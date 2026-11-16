"""Redraw site/data/figs/e9_rwaur.png (scripts/e9_targets.py, stage_figs()) from stored results only.

Inputs: data/findings/e9.json (targets.rwaur.colour) and data/phase2/e9/frozen.json (classes.rwaur.all_contrast_null).
Run from the repo root: .venv/bin/python figures/plot_e9_rwaur.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import INK, ROLES, WIDTH, panel_label, plt, save, use_style

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
WAVE_EDGES = {1: (0.74, 1.12), 2: (1.10, 1.65), 3: (1.64, 2.42), 4: (2.42, 3.82), 5: (3.81, 4.42), 6: (4.41, 5.00)}  # scripts/e9_lib.py

use_style()
t = json.load(open(os.path.join(ROOT, "data/findings/e9.json")))["targets"]["rwaur"]
null = 100 * np.array(json.load(open(os.path.join(ROOT, "data/phase2/e9/frozen.json")))["classes"]["rwaur"]["all_contrast_null"])
col = t["colour"]
infl = {q["detector"]: q["sigma"] for q in col.get("post_hoc_inflated", {}).get("contrast", [])}
S = ROLES["spherex"]
THR = dict(color=INK, lw=1.0, ls=(0, (4, 3)))
BAND = "#d6d6d6"  # opaque tint of the control grey, so neighbouring bands do not darken where they touch

fig, (ax, bx) = plt.subplots(1, 2, figsize=(WIDTH, 3.6), gridspec_kw=dict(width_ratios=[1.35, 1]))

# (a) change per band: horizontal bar = band width, vertical bar = error (widened by the pre-set factor);
#     grey band = +-1 sigma scatter of ordinary stars; dashed = +-5x RW Aur's error (the threshold)
for cdet in col["contrast"]:
    lo, hi = WAVE_EDGES[cdet["detector"]]
    s = 100 * cdet["sigma"]
    e = 100 * infl.get(cdet["detector"], cdet["sigma"])
    ax.fill_between([lo, hi], [-s, -s], [s, s], color=BAND, lw=0, zorder=1)
    ax.plot([lo, hi], [5 * e, 5 * e], zorder=2, **THR)
    ax.plot([lo, hi], [-5 * e, -5 * e], zorder=2, **THR)
    ax.errorbar([cdet["lam_um"]], [100 * cdet["c"]], yerr=[e], xerr=[[cdet["lam_um"] - lo], [hi - cdet["lam_um"]]],
                fmt=S["marker"], ms=6.5, color=S["color"], elinewidth=1.3, mec="white", mew=0.8, zorder=4)
ax.axhline(0, color=INK, lw=0.6, zorder=1)
ax.set_ylim(-95, 95)
ax.set_xlabel("Wavelength (µm)")
ax.set_ylabel("Brightness change (%)")
panel_label(ax, "a")

# (b) the same all-band change for the ordinary comparison stars, RW Aur marked
lim = max(np.abs(null).max(), abs(100 * col["all_detector"])) * 1.25
bx.hist(null, bins=np.linspace(-lim, lim, 25), color=BAND, edgecolor="white", linewidth=0.8)
bx.axvline(100 * col["all_detector"], color=S["color"], ls=S["ls"], lw=2.0)
bx.set_xlabel("All-band brightness change (%)")
bx.set_ylabel("Comparison stars (count)")
panel_label(bx, "b")

h = [plt.Line2D([], [], marker=S["marker"], ls="none", ms=6.5, color=S["color"], mec="white", mew=0.8,
                label="RW Aurigae (SPHEREx)"),
     plt.Line2D([], [], color=S["color"], ls=S["ls"], lw=2.0, label="RW Aurigae, all bands (b)"),
     plt.Rectangle((0, 0), 1, 1, color=BAND, lw=0, label=f"Comparison stars (n = {len(null)})"),
     plt.Line2D([], [], label="±5× RW Aurigae's error (threshold)", **THR)]
fig.legend(handles=h, loc="outside lower center", ncol=2, columnspacing=1.5, handletextpad=0.5)
save(fig, "e9_rwaur")
