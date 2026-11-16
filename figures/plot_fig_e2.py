"""Redraw site/data/figs/fig_e2.png (scripts/e2_counts.py, figure()) from stored results only.

Input: data/findings/e2.json (fields[].naive_all / naive_det / corrected / b / field; control_field).
Patches sorted by |b|; a zero count would be drawn at 0.5 as in the original (none is zero).
Run from the repo root: .venv/bin/python figures/plot_fig_e2.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROLES, WIDTH, plt, save, use_style

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

use_style()
e2 = json.load(open(os.path.join(ROOT, "data/findings/e2.json")))
recs = sorted(e2["fields"], key=lambda r: abs(r["b"]))
x = np.arange(len(recs))
series = [("naive_all", ROLES["control"], "Plain blink", dict(ms=6.5, mew=1.6)),
          ("naive_det", ROLES["spherex"], "One detector at a time", dict(ms=5.5, mec="white", mew=0.5)),
          ("corrected", ROLES["model"], "Colour-aware view (experimental)", dict(ms=6.5, mec="white", mew=0.5))]

fig, ax = plt.subplots(figsize=(WIDTH, 3.6))
for k, (key, R, lab, kw) in enumerate(series):
    ax.plot(x + (k - 1) * 0.22, [max(r[key], 0.5) for r in recs], ls="none", marker=R["marker"], color=R["color"],
            label=lab, zorder=3, **kw)
ax.set_yscale("log")
ax.set_ylim(15, 3000)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda t, _: f"{t:g}"))
ax.set_xticks(x, [f"{abs(r['b']):.0f}" for r in recs], fontsize=9)
ax.tick_params(axis="x", length=0)
ax.set_xlim(-0.6, len(recs) - 0.4)
ax.set_xlabel("Galactic latitude of patch, |b| (°)")
ax.set_ylabel("Changes, summed over pictures (count)")
ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, columnspacing=1.6, handletextpad=0.3)
c = e2["control_field"]
print("n_fields", len(recs), "control", c["naive_all"], c["naive_det"], c["corrected"],
      "frames", min(r["n_frames"] for r in recs), max(r["n_frames"] for r in recs), sum(r["n_frames"] for r in recs))
save(fig, "fig_e2")
