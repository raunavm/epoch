"""Redraw site/data/figs/e18b_floor.png (scripts/run_e18b.py, analyze(), first figure) from stored results only.

Input: data/findings/e18b.json -> injections.recall (per kind, amplitude A and 1-mag bin of SPHEREx mean AB
magnitude: n, recall, Wilson 95% interval). As in the original, only bins with n >= 5 are drawn, at the bin centre.
Run from the repo root: .venv/bin/python figures/plot_e18b_floor.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import INK, ROLES, SEQUENTIAL_CMAP, WIDTH, panel_label, plt, save, use_style

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
AMPS = [0.02, 0.05, 0.10, 0.20]

use_style()
rec = json.load(open(os.path.join(ROOT, "data/findings/e18b.json")))["injections"]["recall"]
cmap = plt.get_cmap(SEQUENTIAL_CMAP)
shade = {0.02: cmap(0.82), 0.05: cmap(0.55), 0.10: cmap(0.32), 0.20: cmap(0.08)}
mk = ROLES["spherex"]["marker"]

fig, axs = plt.subplots(1, 2, figsize=(WIDTH, 3.4), sharey=True)
for ax, kind, letter in zip(axs, ("sin", "ecl"), "ab"):
    for A in AMPS:
        q = [r for r in rec if r["kind"] == kind and r["A"] == A and r["n"] >= 5]
        x = [r["m_lo"] + 0.5 for r in q]
        ax.fill_between(x, [r["wilson95"][0] for r in q], [r["wilson95"][1] for r in q], color=shade[A], alpha=0.15,
                        lw=0, zorder=1)
        ax.plot(x, [r["recall"] for r in q], marker=mk, ls="-", ms=4.5, lw=1.4, color=shade[A], zorder=3,
                label=f"A = {A:.0%}")
    ax.axhline(0.5, color=INK, lw=0.6, ls=(0, (1, 2)), zorder=2, label="Half recovered")
    ax.set_ylim(-0.03, 1.03)
    panel_label(ax, letter)
axs[0].set_ylabel("Fraction recovered")
fig.supxlabel("SPHEREx mean AB magnitude (mag)", fontsize=11)
axs[1].legend(loc="center left", bbox_to_anchor=(1.0, 0.5), handlelength=2.0)
save(fig, "e18b_floor")
