"""Redraw site/data/figs/e15_methane.png (scripts/run_e15.py, fig_methane()) from stored results only.

Input: data/phase2/e15/analysis.json (calibration objects, candidates, threshold), as in `run_e15.py figs`.
Run from the repo root: .venv/bin/python figures/plot_e15_methane.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import INK, ROLES, SEQUENTIAL_CMAP, WIDTH, plt, save, use_style

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

use_style()
out = json.load(open(os.path.join(ROOT, "data/phase2/e15/analysis.json")))
H = ROLES["highlight"]

# Published spectral class is ordinal: discrete steps of the sequential map, plus one marker per class for greyscale.
cmap = plt.get_cmap(SEQUENTIAL_CMAP)
CLS = [("L0-L4.5", "o"), ("L5-L9.5", "s"), ("T0-T4.5", "^"), ("T5-T9.5", "D"), ("Y", "v")]
COL = {c: cmap(v) for (c, _), v in zip(CLS, np.linspace(0.0, 0.8, len(CLS)))}

fig, ax = plt.subplots(figsize=(WIDTH, 4.3))
th = out["threshold"]
ax.axvline(th, color=INK, lw=1.0, ls="--", zorder=1, label=f"Dividing line ({th:.2f})")

ok = lambda r: np.isfinite(r.get("ch4_index", np.nan)) and np.isfinite(r.get("colour_r45_over_j", np.nan))
for cls, mk in CLS:
    rr = [r for r in out["calibration"] if r["cls"] == cls and ok(r)]
    if rr:
        ax.errorbar([r["ch4_index"] for r in rr], [r["colour_r45_over_j"] for r in rr], xerr=[r["ch4_err"] for r in rr],
                    yerr=[r["colour_err"] for r in rr], fmt=mk, ms=6.5, color=COL[cls], mec="white", mew=0.7,
                    elinewidth=1.0, zorder=3, label=f"Known {cls.replace('-', '–')}")
cc = [r for r in out.get("candidates", []) if ok(r)]
if cc:
    ax.errorbar([r["ch4_index"] for r in cc], [r["colour_r45_over_j"] for r in cc], xerr=[r["ch4_err"] for r in cc],
                yerr=[r["colour_err"] for r in cc], fmt=H["marker"], ms=11, color=H["color"], mec="white", mew=0.6,
                elinewidth=1.0, zorder=4, label="Candidate, no\npublished spectrum")

ax.set_yscale("log")
ax.set_xlim(-1.0, 2.0)
ax.set_ylim(0.4, 600)
ax.set_xlabel("Methane index, 3.2–3.4 µm flux ÷ neighbouring flux (ratio)")
ax.set_ylabel("4.4–5.0 µm ÷ 1.0–1.3 µm flux (ratio)")
h, l = ax.get_legend_handles_labels()
order = list(range(1, len(h))) + [0]
ax.legend([h[i] for i in order], [l[i] for i in order], loc="upper left", bbox_to_anchor=(1.01, 1.0),
          handletextpad=0.4)
save(fig, "e15_methane")
n_off = sum(not ok(r) for r in out["calibration"])
print(f"known shown {len(out['calibration']) - n_off}/{len(out['calibration'])}, candidates shown "
      f"{len(cc)}/{len(out.get('candidates', []))}, threshold {th:.3f}")
