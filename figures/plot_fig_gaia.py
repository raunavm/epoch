"""Redraw site/data/figs/fig_gaia.png (scripts/make_figures.py, fig_gaia()) from stored results only.

Inputs: site/data/objects.json and site/data/objects/<id>.json (spectrum.wl_um/refl/refl_err, spectrum.gaia,
spectrum.gaia_check). The five asteroids with a Gaia check, ordered by gaia_check.n as the original.
Run from the repo root: .venv/bin/python figures/plot_fig_gaia.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROLES, WIDTH, panel_label, plt, save, use_style

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
S = os.path.join(ROOT, "site/data")

use_style()
rows = []
for o in json.load(open(f"{S}/objects.json")):
    d = json.load(open(f"{S}/objects/{o['id']}.json"))
    if d["spectrum"].get("gaia") and d["spectrum"].get("gaia_check"):
        rows.append(d)
rows = sorted(rows, key=lambda d: -d["spectrum"]["gaia_check"]["n"])[:6]
SP, R = ROLES["spherex"], ROLES["reference"]

fig, axs = plt.subplots(2, 3, figsize=(WIDTH, 4.6), sharex=True)
for i, (ax, d) in enumerate(zip(axs.ravel(), rows)):
    sp = d["spectrum"]
    g, k = sp["gaia"], sp["gaia_check"]["k"]
    ax.errorbar(g["wl_um"], np.array(g["refl"]) * k, yerr=np.array(g["err"]) * k, fmt=R["marker"], ms=3.5,
                color=R["color"], elinewidth=1.0, zorder=3, label="Gaia DR3, scaled")
    wl, r, e = map(np.array, (sp["wl_um"], sp["refl"], sp["refl_err"]))
    m = wl < 1.1
    ax.errorbar(wl[m], r[m], yerr=e[m], fmt=SP["marker"], ms=3.5, color=SP["color"], elinewidth=0.9, zorder=2,
                label="SPHEREx")
    ax.set_xlim(0.35, 1.12)
    panel_label(ax, "abcdef"[i])
    c = sp["gaia_check"]
    print(d["name"], "n_plotted_spherex", int(m.sum()), "n_gaia", len(g["wl_um"]),
          "chi2/dof %.2f (%d dof)" % (c["chi2"] / max(c["dof"], 1), c["dof"]))
for ax in axs.ravel()[len(rows):]:
    ax.axis("off")
h, l = axs[0, 0].get_legend_handles_labels()
axs[1, 2].legend(h[::-1], l[::-1], loc="center left", bbox_to_anchor=(0.0, 0.55), handletextpad=0.4)
for ax in axs[1]:
    ax.set_xlabel("Wavelength (µm)")
axs[0, 2].tick_params(labelbottom=True)
for ax in axs[:, 0]:
    ax.set_ylabel("Relative reflectance")
save(fig, "fig_gaia")
