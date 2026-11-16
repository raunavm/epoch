"""Redraw site/data/figs/e8_barnard.png (scripts/run_e8.py, figure()) from stored results only.

Inputs: data/findings/e8.json (windows, per_window_error_arcsec), data/phase2/e8/barnard_pictures.parquet,
data/phase2/e8/comp_windows_chosen.parquet, data/efields/e3x_barnard/field.json (field centre for the Gaia track).
Run from the repo root: .venv/bin/python figures/plot_e8_barnard.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from astropy.time import Time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROLES, WIDTH, panel_label, plt, save, use_style

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
# Gaia DR3 values for Barnard's Star, as in scripts/run_e8.py (epoch J2016.0)
BARNARD = dict(ra=269.44850252543836, dec=4.739420051112412, pmra=-801.5509783684709, pmdec=10362.394206546573)

use_style()
res = json.load(open(os.path.join(ROOT, "data/findings/e8.json")))
pic = pd.read_parquet(os.path.join(ROOT, "data/phase2/e8/barnard_pictures.parquet"))
cw = pd.read_parquet(os.path.join(ROOT, "data/phase2/e8/comp_windows_chosen.parquet"))
fj = json.load(open(os.path.join(ROOT, "data/efields/e3x_barnard/field.json")))
rows, sig = res["windows"], res["per_window_error_arcsec"]
S, R, C = ROLES["spherex"], ROLES["reference"], ROLES["control"]

fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 3.9), gridspec_kw=dict(width_ratios=[1, 1]))

# (a) positions: single pictures, Gaia track, visit averages (bars = comparison-star per-visit scatter), Gaia on those dates
tt = np.linspace(pic.mjd.min() - 15, pic.mjd.max() + 15, 200)
dt = Time(tt, format="mjd").jyear - 2016.0
ge = (BARNARD["ra"] - fj["ra"]) * np.cos(np.radians(fj["dec"])) * 3600 + BARNARD["pmra"] * dt / 1000.0
gn = (BARNARD["dec"] - fj["dec"]) * 3600 + BARNARD["pmdec"] * dt / 1000.0
u_ = pic[pic.used]
a.plot(u_.east, u_.north, ls="none", marker=S["marker"], ms=2.2, color=S["color"], alpha=0.25, mew=0, zorder=1,
       label="SPHEREx, single pictures")
a.plot(ge, gn, ls=R["ls"], lw=1.3, color=R["color"], zorder=2, label="Gaia predicted path")
for r in rows:
    a.plot(r["gaia_east"], r["gaia_north"], ls="none", marker=R["marker"], ms=6.5, color=R["color"], zorder=3)
    a.errorbar(r["east"], r["north"], xerr=sig, yerr=sig, fmt=S["marker"], ms=6.5, color=S["color"], mec="white",
               mew=0.8, elinewidth=1.3, zorder=4)
a.plot([], [], ls="none", marker=R["marker"], ms=6.5, color=R["color"], label="Gaia position on the visit dates")
a.errorbar([], [], xerr=[], yerr=[], fmt=S["marker"], ms=6.5, color=S["color"], mec="white", mew=0.8, elinewidth=1.3,
           label="SPHEREx, average of each visit")
a.invert_xaxis()
a.set_aspect("equal", adjustable="datalim")
a.set_xlabel("East–west position (arcsec)")
a.set_ylabel("North–south position (arcsec)")
panel_label(a, "a")

# (b) offsets from Gaia: comparison stars (one point per star per visit), their typical spread, Barnard's Star per visit
b.plot(cw.de - cw.de.mean(), cw.dn - cw.dn.mean(), ls="none", marker=C["marker"], ms=4.5, mew=1.0, color=C["color"],
       zorder=2, label=f"Comparison stars (n = {cw.source_id.nunique()}), each visit")
th = np.linspace(0, 2 * np.pi, 200)
b.plot(sig * np.cos(th), sig * np.sin(th), ls=C["ls"], lw=1.3, color=C["color"], zorder=1,
       label=f"Their typical spread ({sig:.2f} arcsec)")
for r in rows:
    b.plot(r["resid_east"], r["resid_north"], ls="none", marker=S["marker"], ms=7.5, color=S["color"], mec="white",
           mew=0.8, zorder=5)
b.plot([], [], ls="none", marker=S["marker"], ms=7.5, color=S["color"], mec="white", mew=0.8,
       label="Barnard's Star, SPHEREx minus Gaia")
b.axhline(0, color=C["color"], lw=0.6, zorder=0)
b.axvline(0, color=C["color"], lw=0.6, zorder=0)
b.invert_xaxis()
b.set_aspect("equal", adjustable="datalim")
b.set_xlabel("East–west offset from Gaia (arcsec)")
b.set_ylabel("North–south offset from Gaia (arcsec)")
panel_label(b, "b")

ha, la = a.get_legend_handles_labels()
hb, lb = b.get_legend_handles_labels()
order_a = [0, 3, 1, 2]  # single pictures, visit average, Gaia path, Gaia on dates
fig.legend([ha[i] for i in order_a] + hb, [la[i] for i in order_a] + lb, loc="outside lower center", ncol=2,
           columnspacing=2.0, handletextpad=0.5)
save(fig, "e8_barnard")
