"""Redraw site/data/figs/e10_recall.png (scripts/run_e10.py, figure()) from stored results only.

Input: data/findings/e10.json (recall: by_amplitude, by_type_and_amplitude; rates and 95% Wilson intervals as stored).
The 28 saturated variables named in the caption are counted in data/findings/e10_stars.parquet (cls V, status not ok).
Run from the repo root: .venv/bin/python figures/plot_e10_recall.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROLES, WIDTH, plt, save, use_style

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
AMP_BINS = [(0.3, 0.5), (0.5, 1.0), (1.0, 99.0)]  # scripts/run_e10.py


def wilson(k, n, z=1.96):  # as scripts/run_e10.py, for the summed fast-variable bins
    if n == 0:
        return [None, None]
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def _sum(rs):  # as scripts/run_e10.py
    k, n = sum(r["k"] for r in rs), sum(r["n"] for r in rs)
    return dict(k=k, n=n, rate=k / n if n else None, ci95=wilson(k, n))


use_style()
rec = json.load(open(os.path.join(ROOT, "data/findings/e10.json")))["recall"]
S, X = ROLES["spherex"], ROLES["secondary"]
series = [("All testable variables", S["color"], S["marker"], lambda a: rec["by_amplitude"][a]["testable"]),
          ("Long-period variables", X["color"], X["marker"], lambda a: rec["by_type_and_amplitude"]["LPV"][a]),
          ("RR Lyrae, Cepheids, eclipsing", X["color"], "D",
           lambda a: _sum([rec["by_type_and_amplitude"][t][a] for t in ("RR", "CEP", "ECL")]))]

fig, ax = plt.subplots(figsize=(WIDTH, 3.8))
for j, (name, col, mk, get) in enumerate(series):
    for i, (lo, hi) in enumerate(AMP_BINS):
        r = get(f"{lo}-{hi}")
        if not r["n"]:
            continue
        x = i + (j - 1) * 0.27
        ax.errorbar(x, r["rate"], [[r["rate"] - r["ci95"][0]], [r["ci95"][1] - r["rate"]]], fmt=mk, color=col,
                    ms=7 if mk != "D" else 6, mec="white", mew=0.8, elinewidth=1.4)
ax.set_xticks(range(3))
ax.set_xticklabels(["0.3–0.5", "0.5–1", "> 1"])
ax.set_xlim(-0.6, 2.6)
ax.set_ylim(-0.05, 1.05)
ax.set_xlabel("Catalogue amplitude, Gaia G (mag)")
ax.set_ylabel("Fraction recovered")
h = [plt.Line2D([], [], marker=mk, ls="none", ms=7 if mk != "D" else 6, color=col, mec="white", mew=0.8, label=name)
     for name, col, mk, _ in series]
fig.legend(handles=h, loc="outside lower center", ncol=3, columnspacing=1.5, handletextpad=0.4)
save(fig, "e10_recall")
