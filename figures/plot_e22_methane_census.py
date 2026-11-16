"""Methane index of brown-dwarf candidates against WISE W2 (scripts/run_e22.py), redrawn from stored results only.

Input: data/findings/e22.json -> objects (call, W2, ch4_index, ch4_err), control (known L0-L8.5 dwarfs);
data/findings/e22s.json -> threshold. Objects without a defined index are not drawn; the axis is clipped at -1.5 to 2.
Run from the repo root: .venv/bin/python figures/plot_e22_methane_census.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import INK, ROLES, WIDTH, plt, save, use_style

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
use_style()
d = json.load(open(os.path.join(ROOT, "data/findings/e22.json")))
thr = json.load(open(os.path.join(ROOT, "data/findings/e22s.json")))["threshold"]
S, R, C, H = ROLES["spherex"], ROLES["reference"], ROLES["control"], ROLES["highlight"]


def cols(rows):
    rows = [r for r in rows if r["ch4_index"] is not None and r["W2"] is not None]
    e = np.array([r["ch4_err"] if r["ch4_err"] is not None else np.nan for r in rows], float)
    return np.array([r["W2"] for r in rows], float), np.array([r["ch4_index"] for r in rows], float), e


fig, ax = plt.subplots(figsize=(WIDTH, 4.2))
groups = [  # (rows, label, style)
    ([o for o in d["objects"] if o["call"] == "undetermined"], "Too faint for a call",
     dict(fmt="o", ms=2.5, color=C["color"], alpha=0.35, elinewidth=0.6, zorder=1)),
    ([o for o in d["objects"] if o["call"] == "L-like"], "No methane dip (L-like)",
     dict(fmt=S["marker"], ms=3.5, color=S["color"], mec="white", mew=0.4, elinewidth=0.8, zorder=2)),
    (d["control"], "Known L dwarfs (control)",
     dict(fmt=R["marker"], ms=4, color=R["color"], mec="white", mew=0.5, elinewidth=0.8, zorder=3)),
    ([o for o in d["objects"] if o["call"] == "T or later"], "Methane dip (candidates)",
     dict(fmt=H["marker"], ms=9, color=H["color"], mec="white", mew=0.6, elinewidth=1.0, zorder=4)),
]
for rows, label, st in groups:
    x, y, e = cols(rows)
    ax.errorbar(x, y, yerr=np.where(np.isfinite(e), e, 0), capsize=0, label=label, **st)
ax.axhline(thr, color=INK, lw=1.0, ls=(0, (5, 3)), zorder=0, label=f"Dividing value ({thr})")
ax.set_ylim(-1.5, 2.0)
ax.set_xlabel("WISE W2 (mag)")
ax.set_ylabel("Methane index")
fig.legend(loc="outside lower center", ncol=3, handlelength=1.6, columnspacing=1.2)
save(fig, "e22_methane_census")
