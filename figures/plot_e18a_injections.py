"""Redraw site/data/figs/e18a_injections.png (scripts/run_e18a.py, figs() part 1) from stored results only.

Input: data/phase2/e18a/injections.parquet (one row per planted fake: class, strength, host SNR, flag).
Recall per host-SNR bin and 1-sigma Wilson intervals are computed exactly as recall_table() / wilson() in the
original script. The original faceted by strength with one colour per class; here each class gets its own panel
and strength is a sequential colour, so no categorical colour clashes with the fixed palette roles.
Run from the repo root: .venv/bin/python figures/plot_e18a_injections.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROLES, SEQUENTIAL_CMAP, WIDTH, panel_label, plt, save, use_style

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SNR_BINS = [5, 10, 20, 50, 100, 300, 1e9]
SNR_LAB = ["5-10", "10-20", "20-50", "50-100", "100-300", ">300"]
CLASSES = ["tdwarf", "redqso", "carbon", "hlines", "ices"]  # panels a-e
STRENGTHS = [("weak", "Weak"), ("medium", "Medium"), ("strong", "Strong")]


def wilson(k, n, z=1.0):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (c - h, c + h)


def recall(g):
    t = g.groupby("sb", observed=False).flag.agg(["sum", "size"])
    return t["sum"].values.astype(int), t["size"].values.astype(int)


use_style()
inj = pd.read_parquet(os.path.join(ROOT, "data/phase2/e18a/injections.parquet"))
inj = inj.assign(sb=pd.cut(inj.host_snr, SNR_BINS, labels=SNR_LAB))
cmap = plt.get_cmap(SEQUENTIAL_CMAP)
shade = {"weak": cmap(0.78), "medium": cmap(0.45), "strong": cmap(0.12)}
S, C = ROLES["spherex"], ROLES["control"]

k0, n0 = recall(inj[inj.cls == "null"])
nonempty = np.flatnonzero(n0 > 0)
xs = np.arange(len(nonempty))

fig, axs = plt.subplots(2, 3, figsize=(WIDTH, 4.9), sharex=True, sharey=True)
for ax, c, letter in zip(axs.ravel(), CLASSES, "abcde"):
    for al, lab in STRENGTHS:
        k, n = recall(inj[(inj.cls == c) & (inj.amp_label == al)])
        k, n = k[nonempty], n[nonempty]
        y = np.where(n > 0, k / np.maximum(n, 1), np.nan)
        lo_hi = np.array([wilson(a, b) for a, b in zip(k, n)])
        ax.errorbar(xs, y, yerr=[np.clip(y - lo_hi[:, 0], 0, None), np.clip(lo_hi[:, 1] - y, 0, None)],
                    marker=S["marker"], ls=S["ls"], color=shade[al], ms=4.5, lw=1.4, elinewidth=1.0, zorder=3,
                    label=f"{lab} fakes")
    ax.plot(xs, k0[nonempty] / n0[nonempty], marker=C["marker"], ls=C["ls"], color=C["color"], ms=4.5, lw=1.4,
            zorder=2, label="No injection")
    panel_label(ax, letter)
    ax.set_ylim(-0.04, 1.04)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlim(-0.5, len(xs) - 0.5)
    # bins are categories: ticks at the bin edges, so each point sits between its lower and upper edge
    ax.set_xticks(np.arange(len(xs)) - 0.5, [SNR_LAB[i].split("-")[0].lstrip(">") for i in nonempty])

leg_ax = axs[1, 2]
leg_ax.axis("off")
h, l = axs[0, 0].get_legend_handles_labels()
leg_ax.legend(h, l, loc="center left", bbox_to_anchor=(0.0, 0.5), handlelength=2.2)
axs[0, 2].tick_params(labelbottom=True)
fig.supxlabel("Host signal-to-noise per channel (ratio)", fontsize=11)
for ax in axs[:, 0]:
    ax.set_ylabel("Fraction flagged")
save(fig, "e18a_injections")
