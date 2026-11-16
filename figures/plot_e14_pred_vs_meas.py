"""Redraw site/data/figs/e14_pred_vs_meas.png (scripts/run_e14.py, figure()) from stored results only.

Input: data/findings/e14.json (results = E14 blind sample, e13_same_method = E13 asteroids; fields P_nom, P_cal, stack).
Run from the repo root: .venv/bin/python figures/plot_e14_pred_vs_meas.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import INK, ROLES, WIDTH, panel_label, plt, save, use_style

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
THR = dict(color=INK, lw=1.0, ls=(0, (4, 3)))     # 5 sigma detection threshold
EDGE = dict(color=INK, lw=0.8, ls=(0, (1, 2)))    # test-set boundary, expected >= 7

use_style()
o = json.load(open(os.path.join(ROOT, "data/findings/e14.json")))
res, e13 = o["results"], o["e13_same_method"]
S, X, M = ROLES["spherex"], ROLES["secondary"], ROLES["model"]

fig, axs = plt.subplots(1, 2, figsize=(WIDTH, 3.7))
for ax, key, xlabel, letter in [(axs[0], "P_nom", "Predicted S/N, per-picture noise (σ)", "a"),
                                (axs[1], "P_cal", "Predicted S/N, empty-path noise (σ)", "b")]:
    allv = [r[key] for r in res + e13] + [r["stack"] for r in res + e13]
    hi = max(allv) * 1.08 + 1
    lo = min(0, min(r["stack"] for r in res + e13)) - 1
    ax.plot([0, hi], [0, hi], color=M["color"], ls=M["ls"], lw=1.3, zorder=1, label="Measured = predicted")
    ax.axhline(5, zorder=2, label="5σ detection threshold", **THR)
    if key == "P_nom":
        ax.axvline(7, zorder=1, label="Test-set boundary (predicted 7σ)", **EDGE)
    ax.plot([r[key] for r in e13], [r["stack"] for r in e13], ls="none", marker=X["marker"], ms=6.5, color=X["color"],
            mec="white", mew=0.6, zorder=3, label=f"E13 asteroids, not blind (n = {len(e13)})")
    ax.plot([r[key] for r in res], [r["stack"] for r in res], ls="none", marker=S["marker"], ms=6, color=S["color"],
            mec="white", mew=0.7, zorder=4, label=f"E14 asteroids, predicted first (n = {len(res)})")
    ax.set_xlim(0, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Measured S/N of the stack (σ)")
    panel_label(ax, letter)

h, l = axs[0].get_legend_handles_labels()
order = [4, 3, 0, 1, 2]
fig.legend([h[i] for i in order], [l[i] for i in order], loc="outside lower center", ncol=2, columnspacing=1.5,
           handletextpad=0.5)
save(fig, "e14_pred_vs_meas")
