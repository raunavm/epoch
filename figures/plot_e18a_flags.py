"""Redraw site/data/figs/e18a_flags.png (scripts/run_e18a.py, figs() part 3) from stored results only.

Inputs: data/phase2/e18a/features.npz (X: normalised 102-channel spectra, S: errors, M: valid mask, keep: channels),
data/phase2/e18a/triage.json (flagged sources, their 8 nearest ordinary neighbours, verdicts).
Channel centres are the frozen constants of run_e18a.py (CEN). Panel order follows the original
(survivors first, then the others in triage order; there are no survivors).
Run from the repo root: .venv/bin/python figures/plot_e18a_flags.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROLES, WIDTH, panel_label, plt, save, use_style

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "data/phase2/e18a")
DET_RANGES = [(0.74, 1.12), (1.10, 1.65), (1.64, 2.42), (2.42, 3.82), (3.81, 4.42), (4.41, 5.00)]
EDGES = [np.geomspace(lo, hi, 18) for lo, hi in DET_RANGES]
CEN = np.concatenate([np.sqrt(e[:-1] * e[1:]) for e in EDGES])

use_style()
T = pd.read_json(f"{OUT}/triage.json")
z = np.load(f"{OUT}/features.npz")
X, Sg, M, keep = z["X"], z["S"], z["M"], z["keep"]
lam = CEN[keep]
Sp, C = ROLES["spherex"], ROLES["control"]

order = np.r_[np.flatnonzero(T.verdict.values == "survivor"), np.flatnonzero(T.verdict.values != "survivor")][:12]
n = len(order)
nc = 2
nr = int(np.ceil((n + 1) / nc))  # one spare slot holds the legend
fig, axs = plt.subplots(nr, nc, figsize=(WIDTH, 2.0 * nr), sharex=True, squeeze=False)
for a, j, letter in zip(axs.ravel(), order, "abcdefghijkl"):
    t = T.iloc[j]
    i = int(t.idx)
    ref = X[np.array(t.neighbours, int)]
    rmed = np.median(ref, 0)
    for q in ref:
        a.plot(lam, q, color=C["color"], lw=0.5, alpha=0.35, zorder=1)
    a.plot(lam, rmed, color=C["color"], ls=C["ls"], lw=1.8, zorder=2, label="Median of the 8 most similar ordinary sources")
    m = M[i]
    a.errorbar(lam[m], X[i][m], yerr=Sg[i][m], fmt=Sp["marker"], ms=2.6, color=Sp["color"], elinewidth=0.8, zorder=3,
               label="Flagged source")
    yy = np.r_[X[i][m], rmed, ref.ravel()]
    a.set_ylim(min(0, np.nanmin(yy)) - 0.05 * np.nanmax(yy), 1.08 * np.nanmax(yy))
    a.set_xscale("log")
    a.set_xticks([0.8, 1, 1.5, 2, 3, 4, 5], ["0.8", "1", "1.5", "2", "3", "4", "5"])
    a.minorticks_off()
    panel_label(a, letter)
for a in axs.ravel()[n:]:
    a.axis("off")
h, l = axs[0, 0].get_legend_handles_labels()
h.insert(1, plt.Line2D([], [], color=C["color"], lw=0.5, alpha=0.5))
l.insert(1, "The 8 sources individually")
axs.ravel()[n].legend(h, l, loc="center left", bbox_to_anchor=(0.0, 0.5), handlelength=2.2)
for k in range(n):
    if k + nc >= n:  # nothing drawn below this panel
        axs.ravel()[k].tick_params(labelbottom=True)
fig.supxlabel("Wavelength (µm)", fontsize=11)
fig.supylabel("Flux density relative to the source's median (ratio)", fontsize=11)
save(fig, "e18a_flags")
