"""Redraw site/data/figs/fig_e1_m6.png (scripts/run_e6.py, figure()) from stored results only.

Input: data/findings/e6.json (levels_snr_frame; per_method[M].efficiency, efficiency_ci95, k; M1-M5 copied from E1).
Run from the repo root: .venv/bin/python figures/plot_fig_e1_m6.py
"""
import json
import os
import sys

import numpy as np
from matplotlib.ticker import FixedLocator, NullLocator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import INK, ROLES, WIDTH, plt, save, use_style

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

# method -> (role, marker override, label); M3 and M5 share the secondary role, M5 as its second dataset ("D")
METHODS = {"M1": ("control", None, "M1 plain blink"), "M2": ("spherex", None, "M2 one detector at a time"),
           "M3": ("secondary", None, "M3 median, no wavelength term"), "M4": ("model", None, "M4 colour-aware view"),
           "M5": ("secondary", "D", "M5 wavelength-neighbour median"),
           "M6": ("highlight", None, "M6 native-pixel scene reference")}
ZORDER = {"M4": 6, "M6": 5, "M2": 4}

use_style()
out = json.load(open(os.path.join(ROOT, "data/findings/e6.json")))
L = out["levels_snr_frame"]

fig, ax = plt.subplots(figsize=(WIDTH, 4.0))
for m in sorted(out["per_method"]):
    v = out["per_method"][m]
    role, mk, lab = METHODS[m]
    R = ROLES[role]
    lo, hi = np.array(v["efficiency_ci95"]).T
    ax.fill_between(L, lo, hi, color=R["color"], alpha=0.18, lw=0, zorder=1)
    ax.plot(L, v["efficiency"], ls=R["ls"], marker=mk or R["marker"], ms=8 if (mk or R["marker"]) == "*" else 5.5,
            mew=1.4 if R["marker"] == "x" else 0.5, mec=R["color"] if R["marker"] == "x" else "white",
            color=R["color"], lw=1.8 if m in ("M4", "M6") else 1.4, zorder=ZORDER.get(m, 3),
            label=f"{lab}\nk = {v['k']:.2f}")
ax.axhline(0.5, color=INK, lw=0.6, alpha=0.5, zorder=0)
ax.set_xscale("log")
ax.xaxis.set_major_locator(FixedLocator(L))
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda t, _: f"{t:g}"))
ax.xaxis.set_minor_locator(NullLocator())
ax.set_ylim(-0.02, 1.02)
ax.set_xlabel("Planted signal-to-noise ratio per picture")
ax.set_ylabel("Fraction of planted sources found")
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), handletextpad=0.5, labelspacing=0.7)
print("n_fields", out["n_fields"], "n_frames", out["n_frames_total"],
      "M6 false alarm", out["M6_test_false_alarm"]["rate_per_frame"],
      {m: (v["n_trials"][0], v["n_trials"][-1]) for m, v in out["per_method"].items()})
save(fig, "fig_e1_m6")
