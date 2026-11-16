"""Parallax offset from Gaia against colour for the comparison stars around Epsilon Indi B (scripts/run_e29.py),
redrawn from stored results only.

Input: data/findings/e29.json -> blue_stars, red_stars (bp_rp, d_parallax_mas, parallax_err_mas, used),
target (bp_rp, d_parallax_mas, sigma_fit_mas: the fit error only, as requested for this figure).
Run from the repo root: .venv/bin/python figures/plot_e29_colour.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import INK, ROLES, WIDTH, plt, save, use_style

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
use_style()
d = json.load(open(os.path.join(ROOT, "data/findings/e29.json")))
S, X, H = ROLES["spherex"], ROLES["secondary"], ROLES["highlight"]

fig, ax = plt.subplots(figsize=(WIDTH, 3.4))
ax.axhline(0, color=INK, lw=0.6, zorder=1)
for rows, st, label in ((d["blue_stars"], S, "Blue comparison stars"), (d["red_stars"], X, "Red comparison stars")):
    rows = [r for r in rows if r.get("used", True) and r.get("d_parallax_mas") is not None]
    ax.errorbar([r["bp_rp"] for r in rows], [r["d_parallax_mas"] for r in rows], yerr=[r["parallax_err_mas"] for r in rows],
                fmt=st["marker"], ms=5, color=st["color"], mec="white", mew=0.6, elinewidth=1, capsize=0, zorder=3, label=label)
tg = d["target"]
ax.errorbar([tg["bp_rp"]], [tg["d_parallax_mas"]], yerr=[tg["sigma_fit_mas"]], fmt=H["marker"], ms=11, color=H["color"],
            mec="white", mew=0.6, elinewidth=1.2, capsize=0, zorder=4, label="Epsilon Indi B (fit error only)")
ax.set_xlabel("Gaia BP − RP colour (mag)")
ax.set_ylabel("SPHEREx − Gaia parallax (mas)")
ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), handlelength=1.4)
save(fig, "e29_colour")
