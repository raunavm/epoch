"""Redraw site/data/figs/e21_completeness.png (scripts/run_e21_report.py, completeness_fig) from stored results only.

Input: data/findings/e21.json -> injection_grid_snr, injection_grid_au, completeness_k, completeness_n,
planted_ab_mag_by_snr (the same arrays the original heat map shows). Error bars: Wilson 68% intervals.
Run from the repo root: .venv/bin/python figures/plot_e21_completeness.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import INK, ROLES, SEQUENTIAL_CMAP, WIDTH, plt, save, use_style

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def wilson(k, n, z=1.0):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


use_style()
d = json.load(open(os.path.join(ROOT, "data/findings/e21.json")))
snr, au, ab = d["injection_grid_snr"], np.array(d["injection_grid_au"], float), d["planted_ab_mag_by_snr"]
K, N = np.array(d["completeness_k"], float), np.array(d["completeness_n"], float)
cmap = plt.get_cmap(SEQUENTIAL_CMAP)

fig, ax = plt.subplots(figsize=(WIDTH, 3.6))
for i in reversed(range(len(snr))):
    frac = K[i] / N[i]
    lo, hi = wilson(K[i], N[i])
    col = cmap(0.85 * (1 - i / (len(snr) - 1)))
    ax.errorbar(au, frac, yerr=[frac - lo, hi - frac], marker=ROLES["spherex"]["marker"], ms=4.5, lw=1.3,
                capsize=0, color=col, label=f"{snr[i]}σ per visit (AB {ab[i]:.1f})")
ax.axhline(0.5, color=INK, lw=0.6, ls=(0, (1, 2)), label="Half recovered")
ax.set_xticks(au, [f"{a:.0f}" for a in au])
ax.set_xlabel("Distance (AU)")
ax.set_ylabel("Fraction recovered")
ax.set_ylim(-0.03, 1.03)
h, l = ax.get_legend_handles_labels()
ax.legend(h[1:] + h[:1], l[1:] + l[:1], loc="center left", bbox_to_anchor=(1.0, 0.5), handlelength=2.0)
save(fig, "e21_completeness")
