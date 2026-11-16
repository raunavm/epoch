"""Redraw site/data/figs/fig_sizes.png (scripts/make_figures.py, fig_sizes()) from stored results only.

Input: site/data/objects.json (D_truth_km, D_km, D_err_km, kind).
Run from the repo root: .venv/bin/python figures/plot_fig_sizes.py
"""
import json
import os
import sys

import numpy as np
from matplotlib.ticker import FixedLocator, NullLocator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import INK, ROLES, WIDTH, plt, save, use_style

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

use_style()
objs = json.load(open(os.path.join(ROOT, "site/data/objects.json")))
v = [o for o in objs if o.get("D_km") and o.get("D_truth_km")]
style = {"MBA": (ROLES["spherex"], "Main-belt asteroid"), "TJN": (ROLES["secondary"], "Jupiter Trojan")}

fig, ax = plt.subplots(figsize=(WIDTH, 4.0))
lo = min(o["D_truth_km"] for o in v) * 0.6
hi = max(o["D_km"] + o["D_err_km"] for o in v) * 1.1  # whole error bars inside the square axes
x = np.geomspace(lo, hi, 10)
ax.fill_between(x, x / 1.2, x * 1.2, color="#E6E6E6", lw=0, zorder=0, label="±20%")
ax.plot(x, x, color=INK, lw=0.9, zorder=1, label="Equal diameters")
for k in sorted(set(o["kind"] for o in v)):
    R, lab = style[k]
    s = [o for o in v if o["kind"] == k]
    ax.errorbar([o["D_truth_km"] for o in s], [o["D_km"] for o in s], yerr=[o["D_err_km"] for o in s],
                fmt=R["marker"], ms=7, color=R["color"], mec="white", mew=0.6, elinewidth=1.2, zorder=3, label=lab)
    print(k, len(s), [(o["name"], o["D_truth_km"], o["D_km"], o["D_err_km"], o["D_truth_source_short"]) for o in s])
ax.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi))
ax.set_aspect("equal")
for axis in (ax.xaxis, ax.yaxis):
    axis.set_major_locator(FixedLocator([20, 30, 50, 100, 200, 300]))
    axis.set_major_formatter(plt.FuncFormatter(lambda t, _: f"{t:g}"))
    axis.set_minor_locator(NullLocator())
ax.set_xlabel("Independent diameter (km)")
ax.set_ylabel("SPHEREx thermal diameter (km)")
h, l = ax.get_legend_handles_labels()
order = [2, 3, 1, 0]  # Main-belt, Trojan, equal diameters, +-20%
ax.legend([h[i] for i in order], [l[i] for i in order], loc="upper left", bbox_to_anchor=(1.02, 1.0))
save(fig, "fig_sizes")
