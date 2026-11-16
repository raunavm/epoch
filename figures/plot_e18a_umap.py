"""Redraw site/data/figs/e18a_umap.png (scripts/run_e18a.py, figs() part 2) from stored results only.

Inputs: data/phase2/e18a/umap.npy (columns: source index, UMAP x, UMAP y; the embedding the original figure drew,
saved by the same call), data/phase2/e18a/scores.parquet (set, known_class, flag), data/phase2/e18/sources.parquet
(kind), data/phase2/e18a/triage.json (verdicts). UMAP is not re-run.
Run from the repo root: .venv/bin/python figures/plot_e18a_umap.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import INK, ROLES, WIDTH, plt, save, use_style

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
E18, OUT = os.path.join(ROOT, "data/phase2/e18"), os.path.join(ROOT, "data/phase2/e18a")

use_style()
S = pd.read_parquet(f"{E18}/sources.parquet")
res = pd.read_parquet(f"{OUT}/scores.parquet")
T = pd.read_json(f"{OUT}/triage.json")
u = np.load(f"{OUT}/umap.npy")
idx, emb = u[:, 0].astype(int), u[:, 1:]
assert np.array_equal(idx, np.flatnonzero(res.set.isin(["population", "held_out"]).values))
Sp, R, H = ROLES["spherex"], ROLES["reference"], ROLES["highlight"]

fig, ax = plt.subplots(figsize=(WIDTH, 5.6))
kind = S.kind.values[idx]
ax.scatter(*emb[kind == "gaia"].T, s=5, c="#C4C4C4", lw=0, label=f"Star in Gaia ({(kind == 'gaia').sum()})")
ax.scatter(*emb[kind != "gaia"].T, s=5, c=Sp["color"], alpha=0.45, lw=0,
           label=f"Infrared-only source: galaxy or AGN ({(kind != 'gaia').sum()})")
kc = res.known_class.values[idx]
mk = {"quasar / AGN": ("s", "Quasar or AGN"), "planetary nebula": ("h", "Planetary nebula"),
      "white dwarf": ("<", "White dwarf"), "peculiar / emission-line star": ("P", "Peculiar or emission-line star"),
      "known variable star": ("D", "Known variable star")}
for c, (m, lab) in mk.items():
    w = kc == c
    if w.any():
        ax.scatter(*emb[w].T, s=34, marker=m, c=R["color"], edgecolor=INK, lw=0.5, zorder=3,
                   label=f"{lab}, catalogue ({w.sum()})")
flg = res.flag.values[idx]
ax.scatter(*emb[flg].T, s=150, marker=H["marker"], c=H["color"], edgecolor=INK, lw=0.6, zorder=4,
           label=f"Flagged as unusual ({flg.sum()})")
surv = set(T.idx[T.verdict == "survivor"]) if len(T) else set()
ws = np.isin(idx, list(surv))
if ws.any():  # none in the stored triage; kept so the figure follows the original logic
    ax.scatter(*emb[ws].T, s=150, marker=H["marker"], facecolor="white", edgecolor=H["color"], lw=1.2, zorder=5,
               label=f"Unusual, not explained by catalogues ({ws.sum()})")
ax.set_xticks([]), ax.set_yticks([])
ax.set_xlabel("UMAP axis 1 (arbitrary units)")
ax.set_ylabel("UMAP axis 2 (arbitrary units)")
ax.legend(loc="upper left", bbox_to_anchor=(0, -0.08), ncol=2, columnspacing=1.2, handletextpad=0.3, markerscale=1.0)
save(fig, "e18a_umap")
