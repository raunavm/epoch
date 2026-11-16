"""Redraw site/data/figs/fig_reveal.png (scripts/make_figures.py, fig_reveal()) from stored results only.

Input: site/data/reveal.json (stack_pngs[].n/snr, frames[].snr, null_snrs).
Run from the repo root: .venv/bin/python figures/plot_fig_reveal.py
"""
import json
import os
import sys

import numpy as np
from matplotlib.ticker import FixedLocator, NullLocator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import INK, ROLES, WIDTH, panel_label, plt, save, use_style

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

use_style()
r = json.load(open(os.path.join(ROOT, "site/data/reveal.json")))
st = r["stack_pngs"]
SP, MO, C = ROLES["spherex"], ROLES["model"], ROLES["control"]

fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 3.3))

# (a) stack S/N against number of pictures combined; sqrt(N) curve scaled to the last step, as the original
n = np.array([s["n"] for s in st])
a.plot(n, [s["snr"] for s in st], marker=SP["marker"], ls=SP["ls"], color=SP["color"], ms=5, mec="white", mew=0.6,
       zorder=3, label="Stack along the orbit")
a.plot(n, st[-1]["snr"] * np.sqrt(n / n.max()), ls=MO["ls"], color=MO["color"], lw=1.4, zorder=2,
       label="√N expectation")
a.axhline(5, color=INK, lw=0.8, ls="--", zorder=1, label="S/N = 5")
a.set_xscale("log")
a.xaxis.set_major_locator(FixedLocator([1, 10, 100]))
a.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
a.xaxis.set_minor_locator(NullLocator())
a.set_xlabel("Pictures combined (count)")
a.set_ylabel("Signal-to-noise ratio")
a.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=1, handletextpad=0.5)
panel_label(a, "a")

# (b) single-picture S/N histogram (30 bins), off-orbit empty paths as dotted lines, 5 sigma line
b.hist([f["snr"] for f in r["frames"]], bins=30, color=SP["color"], edgecolor="white", linewidth=0.4,
       label="Single pictures", zorder=2)
for i, v in enumerate(r.get("null_snrs", [])):
    b.axvline(v, color=C["color"], lw=0.9, ls=C["ls"], zorder=3, label="Empty paths beside the orbit" if i == 0 else None)
b.axvline(5, color=INK, lw=0.8, ls="--", zorder=1, label="S/N = 5")
b.set_xlabel("Signal-to-noise ratio")
b.set_ylabel("Pictures (count)")
b.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=1, handletextpad=0.5)
panel_label(b, "b")
print("n_singles", len(r["frames"]), "n_nulls", len(r.get("null_snrs", [])), "stack_last", st[-1],
      "single max", max(f["snr"] for f in r["frames"]))
save(fig, "fig_reveal")
