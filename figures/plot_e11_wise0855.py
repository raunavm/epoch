"""Redraw site/data/figs/e11_wise0855.png (scripts/run_e11.py, fig()) from stored results only.

Inputs: data/findings/e11.json (targets.wise0855) and the JWST PRISM table
data/phase2/e11/lit/wise0855/luhman2024_fig3_prism/dbf3.txt (Luhman et al. 2024, F_nu in Jy).
Run from the repo root: .venv/bin/python figures/plot_e11_wise0855.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROLES, WIDTH, plt, save, use_style

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
LIT = os.path.join(ROOT, "data/phase2/e11/lit")

use_style()
t = json.load(open(os.path.join(ROOT, "data/findings/e11.json")))["targets"]["wise0855"]
S, R = ROLES["spherex"], ROLES["reference"]

fig, ax = plt.subplots(figsize=(WIDTH, 4.0))

# JWST PRISM spectrum (thin) and its averages in the same 0.1 um steps (squares at the step midpoints)
d = [l.split() for l in open(f"{LIT}/wise0855/luhman2024_fig3_prism/dbf3.txt").read().splitlines()[13:]]
d = np.array([[float(x[0]), float(x[1]) * 1e3] for x in d if len(x) >= 2])
s = (d[:, 0] > 3.7) & (d[:, 0] < 5.1)
ax.plot(d[s, 0], d[s, 1], color=R["color"], lw=0.8, alpha=0.5, zorder=2, label="JWST NIRSpec spectrum")
mid = [(b["lo"] + b["hi"]) / 2 for b in t["bins"]]
ax.plot(mid, t["jwst"]["PRISM"]["binned_mJy"], marker=R["marker"], ls=R["ls"], color=R["color"], ms=5.5,
        lw=1.3, zorder=3, label="JWST, averaged in the same steps")

# Spitzer [4.5] (Luhman et al. 2024 Sec. 4.1); horizontal bar = filter half-width 0.5 um
r = next(x for x in t["comparison"] if x["band"] == "IRAC [4.5]" and "Luhman et al. 2024" in x["source"])
ax.errorbar(4.493, r["published_mJy"], yerr=r["published_err_mJy"], xerr=0.5, fmt="D", ms=6, color=R["color"],
            mfc="white", mec=R["color"], mew=1.3, elinewidth=1.0, zorder=4, label="Spitzer 4.5 µm filter")

# SPHEREx steps: filled if flux > 3x error, open otherwise; bars = stored error
det = [b for b in t["bins"] if b["detected_3sigma"]]
und = [b for b in t["bins"] if not b["detected_3sigma"]]
ax.errorbar([b["centre"] for b in det], [b["flux"] for b in det], yerr=[b["err"] for b in det], fmt=S["marker"],
            ms=6.5, color=S["color"], mec="white", mew=0.8, elinewidth=1.3, zorder=6, label="SPHEREx")
if und:
    ax.errorbar([b["centre"] for b in und], [b["flux"] for b in und], yerr=[b["err"] for b in und], fmt=S["marker"],
                ms=6.5, mfc="white", mec=S["color"], mew=1.4, color=S["color"], elinewidth=1.3, zorder=6,
                label="SPHEREx, below 3× error")

ax.set_xlim(3.72, 5.06), ax.set_ylim(-0.03, 0.85)
ax.set_xlabel("Wavelength (µm)")
ax.set_ylabel("Flux density (mJy)")
h, l = ax.get_legend_handles_labels()
order = [3, 4, 2, 1, 0]  # column 1: SPHEREx, SPHEREx open, Spitzer; column 2: JWST binned, JWST spectrum
ax.legend([h[i] for i in order], [l[i] for i in order], loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2,
          columnspacing=2.0, handletextpad=0.5)
save(fig, "e11_wise0855")
