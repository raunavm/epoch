"""Redraw site/data/figs/e11_epsindib.png (scripts/run_e11.py, fig()) from stored results only.

Input: data/findings/e11.json (targets.epsindib: bins, band_tests, comparison).
Run from the repo root: .venv/bin/python figures/plot_e11_epsindib.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROLES, WIDTH, plt, save, use_style

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
# filter pivot wavelengths (um) and half-widths (um), as scripts/run_e11.py (ZP, BAND_HALF)
LAM = {"2MASS J": 1.235, "2MASS H": 1.662, "2MASS Ks": 2.159, "WISE W1": 3.3526, "IRAC [3.6]": 3.550, "IRAC [4.5]": 4.493}
BAND_HALF = {"2MASS J": 0.081, "2MASS H": 0.1255, "2MASS Ks": 0.131, "WISE W1": 0.33, "IRAC [3.6]": 0.375, "IRAC [4.5]": 0.5}
PICK = [("2MASS J", "II/246"), ("2MASS H", "2MASS PSC"), ("2MASS Ks", "2MASS PSC"), ("WISE W1", "All-Sky"),
        ("IRAC [3.6]", "Patten"), ("IRAC [4.5]", "Patten")]
STRIP = "#ebebeb"

use_style()
t = json.load(open(os.path.join(ROOT, "data/findings/e11.json")))["targets"]["epsindib"]
S, R = ROLES["spherex"], ROLES["reference"]
top = 65

fig, ax = plt.subplots(figsize=(WIDTH, 3.9))
# absorption bands that passed the pre-set test
for bt in t["band_tests"]:
    if bt["named"]:
        lo, hi = bt["band_um"]
        ax.axvspan(lo, hi, color=STRIP, lw=0, zorder=0)
# SPHEREx spectrum
b = t["bins"]
ax.plot([x["centre"] for x in b], [x["flux"] for x in b], ls=S["ls"], color=S["color"], lw=1.0, alpha=0.6, zorder=3)
ax.errorbar([x["centre"] for x in b], [x["flux"] for x in b], yerr=[x["err"] for x in b], fmt=S["marker"], ms=4.5,
            color=S["color"], mec="white", mew=0.6, elinewidth=1.1, zorder=4, label="SPHEREx")
# published filter brightness (horizontal bar = filter half-width) and SPHEREx through the same filter
first = True
for band, key in PICK:
    r = next(x for x in t["comparison"] if x["band"] == band and key in x["source"])
    lam = LAM[band]
    if r["ratio"] is not None:
        ax.errorbar(lam, r["spherex_mJy"], yerr=r["spherex_err_mJy"], fmt="D", ms=9, color=S["color"], mec="white",
                    mew=0.8, elinewidth=1.3, zorder=5, label="SPHEREx through the same filter" if first else None)
    ax.errorbar(lam, r["published_mJy"], yerr=r["published_err_mJy"], xerr=BAND_HALF[band], fmt="D", ms=5.5,
                color=R["color"], mec="white", mew=0.8, elinewidth=1.3, zorder=6,
                label="Published filter brightness" if first else None)
    first = False
ax.set_xlim(0.7, 5.08), ax.set_ylim(-3, top)
ax.set_xlabel("Wavelength (µm)")
ax.set_ylabel("Flux density (mJy)")
h, l = ax.get_legend_handles_labels()
h.append(plt.Rectangle((0, 0), 1, 1, color=STRIP, lw=0))
l.append("Absorption band passing the pre-set test")
fig.legend(h, l, loc="outside lower center", ncol=2, columnspacing=1.5, handletextpad=0.5)
save(fig, "e11_epsindib")
