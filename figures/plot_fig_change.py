"""Redraw site/data/figs/fig_change.png (scripts/make_figures.py, fig_change()) from stored results only.

Input: data/findings/f1.json (per_field: n_naive_all = plain blink, n_corrected = colour-aware view).
Run from the repo root: .venv/bin/python figures/plot_fig_change.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import INK, ROLES, WIDTH, plt, save, use_style

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

use_style()
pf = json.load(open(os.path.join(ROOT, "data/findings/f1.json")))["per_field"]
short = {"ecliptic_control": "Control patch (ecliptic)"}
names = [short.get(x["field"], x["name"].split("(")[0].strip()[:24]) for x in pf]
naive = np.array([x["n_naive_all"] for x in pf])
corr = np.array([x["n_corrected"] for x in pf])
y = np.arange(len(pf))
C, M = ROLES["control"], ROLES["model"]

fig, ax = plt.subplots(figsize=(WIDTH, 3.4))
ax.hlines(y, np.minimum(naive, corr), np.maximum(naive, corr), color=INK, lw=0.6, alpha=0.35, zorder=1)
ax.plot(naive, y, ls="none", marker=C["marker"], ms=8, mew=1.8, color=C["color"], zorder=3, label="Plain blink")
ax.plot(corr, y, ls="none", marker=M["marker"], ms=9, color=M["color"], mec="white", mew=0.6, zorder=4,
        label="Colour-aware view")
ax.set_yticks(y, names)
ax.tick_params(axis="y", length=0)
ax.spines["left"].set_visible(False)
ax.set_xlim(0, None)
ax.set_ylim(-0.6, len(pf) - 0.4)
ax.set_xlabel("Apparent changes per patch (count)")
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, columnspacing=2.0, handletextpad=0.4)
save(fig, "fig_change")
