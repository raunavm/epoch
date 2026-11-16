"""Recovery of injected sources against S/N at 0.03 false candidates per picture on the 20 test fields
(scripts/run_e30.py), redrawn from stored results only.

Input: data/findings/e30.json -> levels_snr_frame and results.<method>.r3.recall / recall_ci95 for CNN (the network),
M3BF (median reference with the archive's flags; reference line, not part of the rule), M3B and M2 (no flags).
Run from the repo root: .venv/bin/python figures/plot_e30_recall.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROLES, WIDTH, plt, save, use_style

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
use_style()
d = json.load(open(os.path.join(ROOT, "data/findings/e30.json")))
snr = np.array(d["levels_snr_frame"], float)
res = d["results"]
S, R, C = ROLES["spherex"], ROLES["reference"], ROLES["control"]
lines = [("CNN", "Network (no flags)", S, "-"), ("M3BF", "Median reference with flags", R, R["ls"]),
         ("M3B", "Median reference, no flags", C, C["ls"]), ("M2", "One detector at a time, no flags", ROLES["secondary"], ROLES["secondary"]["ls"])]

fig, ax = plt.subplots(figsize=(WIDTH, 3.3))
for key, label, st, ls in lines:
    r = res[key]["r3"]
    y = np.array(r["recall"], float); lo, hi = np.array(r["recall_ci95"], float).T
    ax.fill_between(snr, lo, hi, color=st["color"], alpha=0.15, lw=0)
    ax.plot(snr, y, ls=ls, marker=st["marker"], ms=4.5, lw=1.4, color=st["color"], mec="white", mew=0.5, label=label)
ax.set_xscale("log")
ax.set_xticks(snr, [f"{v:g}" for v in snr])
ax.minorticks_off()
ax.set_ylim(-0.03, 1.0)
ax.set_xlabel("Injected source S/N per picture")
ax.set_ylabel("Fraction recovered")
ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), handlelength=2.2)
save(fig, "e30_recall")
