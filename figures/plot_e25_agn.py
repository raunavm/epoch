"""AGN variability test (scripts/run_e25.py): (a) excess variance of 19 AGN over matched controls per wavelength bin,
(b) stacked cross-correlation of short- and long-wavelength light curves. Redrawn from stored results only.

Input: data/findings/e25.json -> wl_bin_edges_um, agn_minus_ctrl_z; ccf_lags_d, ccf_agn, ccf_ctrl, ccf_ctrl_draws_p1_p99,
null_p99 (99th percentile of the peak over time-shuffled AGN stacks).
Run from the repo root: .venv/bin/python figures/plot_e25_agn.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import INK, ROLES, WIDTH, panel_label, plt, save, use_style

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
BAND = "#e3e3e3"

use_style()
d = json.load(open(os.path.join(ROOT, "data/findings/e25.json")))
S, C = ROLES["spherex"], ROLES["control"]

fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 3.0))

edges = np.array(d["wl_bin_edges_um"], float)
mid, half = edges.mean(1), (edges[:, 1] - edges[:, 0]) / 2
for v in (-3, 3):
    a.axhline(v, color=INK, lw=0.9, ls=(0, (4, 3)), zorder=1, label="±3σ" if v > 0 else None)
a.axhline(0, color=INK, lw=0.6, zorder=2)
a.errorbar(mid, d["agn_minus_ctrl_z"], xerr=half, fmt=S["marker"], ms=4.5, color=S["color"], elinewidth=1, capsize=0,
           mec="white", mew=0.6, zorder=3)
a.set_xlabel("Wavelength (µm)")
a.set_ylabel("AGN excess over controls (σ)")
a.set_ylim(-4.5, 4.5)
panel_label(a, "a")

lag = np.array(d["ccf_lags_d"], float)
lo, hi = d["ccf_ctrl_draws_p1_p99"]
b.fill_between(lag, lo, hi, color=BAND, lw=0, zorder=1, label="Random control sets, 1–99%")
b.axhline(0, color=INK, lw=0.6, zorder=2)
b.axhline(d["null_p99"], color=INK, lw=0.9, ls=(0, (1, 2)), zorder=2, label="Time-shuffled AGN, 99%")
b.plot(lag, d["ccf_ctrl"], color=C["color"], lw=1.2, zorder=3, label="All controls")
b.plot(lag, d["ccf_agn"], color=S["color"], marker=S["marker"], ms=3, lw=1.2, mec="white", mew=0.4, zorder=4,
       label=f"AGN ({d['n_agn']})")
b.set_xlabel("Lag (d)")
b.set_ylabel("Stacked correlation")
b.set_xticks([-150, -75, 0, 75, 150])
panel_label(b, "b")

h1, l1 = a.get_legend_handles_labels()
h2, l2 = b.get_legend_handles_labels()
fig.legend(h1[::-1] + h2[::-1], l1[::-1] + l2[::-1], loc="outside lower center", ncol=3, columnspacing=1.2,
           handletextpad=0.4)
save(fig, "e25_agn")
