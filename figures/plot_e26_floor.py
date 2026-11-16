"""Recovery of planted sinusoidal variations at the north ecliptic pole, without and with the shared-error correction
(scripts/run_e26.py; replaces the draft site/data/figs/e26_floor.png on the site).

Input: data/findings/e26.json -> detail.{before,after}.recall (kind, amplitude A, 1-mag bin of SPHEREx mean AB magnitude:
n, recall, Wilson 95% interval). Only bins with n >= 10 are drawn, the minimum run_e26 uses for its 50% floors.
Run from the repo root: .venv/bin/python figures/plot_e26_floor.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import INK, ROLES, SEQUENTIAL_CMAP, WIDTH, panel_label, plt, save, use_style

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
AMPS = [0.01, 0.02, 0.05, 0.10, 0.20]

use_style()
det = json.load(open(os.path.join(ROOT, "data/findings/e26.json")))["detail"]
cmap = plt.get_cmap(SEQUENTIAL_CMAP)
shade = {A: cmap(v) for A, v in zip(AMPS, (0.90, 0.70, 0.50, 0.30, 0.08))}
mk = ROLES["spherex"]["marker"]

fig, axs = plt.subplots(1, 2, figsize=(WIDTH, 3.4), sharey=True, sharex=True)
for ax, mode, letter in zip(axs, ("before", "after"), "ab"):
    rec = det[mode]["recall"]
    for A in AMPS:
        q = sorted((r for r in rec if r["kind"] == "sin" and r["A"] == A and r["n"] >= 10), key=lambda r: r["m_lo"])
        x = [r["m_lo"] + 0.5 for r in q]
        ax.fill_between(x, [r["wilson95"][0] for r in q], [r["wilson95"][1] for r in q], color=shade[A], alpha=0.15,
                        lw=0, zorder=1)
        ax.plot(x, [r["recall"] for r in q], marker=mk, ls="-", ms=4.5, lw=1.4, color=shade[A], zorder=3,
                label=f"{A:.0%}")
    ax.axhline(0.5, color=INK, lw=0.6, ls=(0, (1, 2)), zorder=2, label="Half recovered")
    ax.set_ylim(-0.03, 1.03)
    panel_label(ax, letter)
axs[0].set_ylabel("Fraction recovered")
fig.supxlabel("SPHEREx mean AB magnitude (mag)", fontsize=11)
axs[1].legend(title="Amplitude", loc="center left", bbox_to_anchor=(1.0, 0.5), handlelength=2.0)
save(fig, "e26_floor")
