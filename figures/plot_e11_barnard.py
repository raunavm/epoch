"""Redraw site/data/figs/e11_barnard.png (scripts/run_e11.py, fig()) from stored results only.

Input: data/findings/e11.json (targets.barnard: template, diagnostic_flagged_core.min4pix.bins, bins, comparison).
Run from the repo root: .venv/bin/python figures/plot_e11_barnard.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROLES, WIDTH, plt, save, use_style

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
# filter pivot wavelengths (um) and half-widths (um), as scripts/run_e11.py (ZP, BAND_HALF)
LAM = {"2MASS J": 1.235, "2MASS H": 1.662, "2MASS Ks": 2.159}
BAND_HALF = {"2MASS J": 0.081, "2MASS H": 0.1255, "2MASS Ks": 0.131}
PICK = [("2MASS J", "II/246"), ("2MASS H", "2MASS PSC"), ("2MASS Ks", "2MASS PSC")]

use_style()
t = json.load(open(os.path.join(ROOT, "data/findings/e11.json")))["targets"]["barnard"]
S, R, X = ROLES["spherex"], ROLES["reference"], ROLES["secondary"]

fig, ax = plt.subplots(figsize=(WIDTH, 3.9))
# Gl 213 (IRTF library) scaled to Barnard's Star, in the same colour steps (gaps where the template has no value)
tl, th = np.array(t["template"]["bin_lo"]), np.array(t["template"]["bin_hi"])
tv = np.array([np.nan if v is None else v for v in t["template"]["binned_mJy"]])
ax.plot(np.sqrt(tl * th), tv, marker=R["marker"], ls=R["ls"], color=R["color"], ms=4.5, mec="white", mew=0.6,
        lw=1.3, zorder=3, label="Similar star (Gl 213), scaled")
# SPHEREx from pictures with an overloaded core (not valid)
dg = t["diagnostic_flagged_core"]["min4pix"]["bins"]
ax.errorbar([x["centre"] for x in dg], [x["flux"] for x in dg], yerr=[x["err"] for x in dg],
            fmt=X["marker"], ms=6, color=X["color"], mec="white", mew=0.6, elinewidth=1.1, zorder=4,
            label="SPHEREx, overloaded pictures (not valid)")
# SPHEREx, the one picture that passed the rule
for x in t.get("bins", []):
    ax.errorbar(x["centre"], x["flux"], yerr=x["err"], fmt=S["marker"], ms=7.5, color=S["color"],
                mec="white", mew=0.8, elinewidth=1.3, zorder=6, label="SPHEREx, valid picture (n = 1)")
# published 2MASS brightness (horizontal bar = filter half-width)
first = True
for band, key in PICK:
    r = next(x for x in t["comparison"] if x["band"] == band and key in x["source"])
    ax.errorbar(LAM[band], r["published_mJy"], yerr=r["published_err_mJy"], xerr=BAND_HALF[band], fmt="D",
                ms=7, color=R["color"], mec="white", mew=0.8, elinewidth=1.3, zorder=7,
                label="Published filter brightness" if first else None)
    first = False
ax.set_xlim(0.7, 5.08), ax.set_ylim(0, 16000)
ax.set_xlabel("Wavelength (µm)")
ax.set_ylabel("Flux density (mJy)")
h, l = ax.get_legend_handles_labels()
order = [2, 1, 0, 3]  # valid SPHEREx, not-valid SPHEREx, template, filters
fig.legend([h[i] for i in order], [l[i] for i in order], loc="outside lower center", ncol=2, columnspacing=1.5,
           handletextpad=0.5)
save(fig, "e11_barnard")
