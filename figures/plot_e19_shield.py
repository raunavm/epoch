"""Redraw site/data/figs/e19_shield.png (scripts/run_e19.py, figure()) from stored results only.

Inputs: data/findings/e19.json "pictures" (the table run_e19.report() passes to figure(), incl. d, x, y, sigma_R and
the post-hoc pattern match `similarity`) and the stored cut-outs data/phase2/e19/cut/*.npy. Panel picks follow the
original rule. Run from the repo root: .venv/bin/python figures/plot_e19_shield.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMAGE_CMAP, INK, OKABE_ITO, ROLES, WIDTH, panel_label, plt, save, use_style

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SPOT = (1648.0, 1752.0)
NAMED = ["VY CMa", "eps Oct", "GY UMa"]
K26 = ["2025W46_1A_0770_1D1", "2025W43_2A_0490_1D1", "2025W43_1A_0318_2D1"]

use_style()
res = pd.DataFrame(json.load(open(os.path.join(ROOT, "data/findings/e19.json")))["pictures"])
nm = res[res.set == "named"]
vy_el = nm[(nm.star == "VY CMa") & (nm.d > 150)]
vy_el = vy_el.iloc[(vy_el.y - SPOT[1]).abs().argsort()]
cs, ce = res[res.set == "c_spot"], res[res.set == "c_else"]
near = cs.loc[cs.d.idxmin()]
panels = [(nm[nm.obs_id == oid[:-2]].iloc[0], f"{s_}, at the spot") for s_, oid in zip(NAMED, K26)]
panels += [(vy_el.iloc[0], "VY CMa, elsewhere"),
           (near, f"{' '.join(near.star.split())}, {near.d:.0f} px away"),
           (ce.iloc[(ce.similarity - ce.similarity.median()).abs().argsort().iloc[0]], None)]
panels[-1] = (panels[-1][0], f"{' '.join(panels[-1][0].star.split())}, elsewhere")

fig = plt.figure(figsize=(WIDTH, 8.0))
top, bot = fig.subfigures(2, 1, height_ratios=[5.0, 3.0], hspace=0.03)
axs = top.subplots(2, 3)
th = np.linspace(0, 2 * np.pi, 200)
for i, (a, (r, lab)) in enumerate(zip(axs.ravel(), panels)):
    R = np.load(os.path.join(ROOT, r.cutout))
    rr = np.hypot(*np.mgrid[:R.shape[0], :R.shape[1]] - (R.shape[0] - 1) / 2)
    D = (R - np.nanmedian(R[rr > 80])) / r.sigma_R
    a.imshow(np.arcsinh(D / 2), origin="lower", cmap=IMAGE_CMAP, vmin=-1.5, vmax=np.arcsinh(40), interpolation="nearest")
    c = (R.shape[0] - 1) / 2
    a.plot(c + 40 * np.cos(th), c + 40 * np.sin(th), ls=":", color=ROLES["highlight"]["color"], lw=1.2)
    a.set_xticks([]), a.set_yticks([])
    a.spines[:].set_visible(False)
    panel_label(a, "abcdef"[i])
    print("abcdef"[i], lab, r.obs_id, f"x,y {r.x:.0f},{r.y:.0f} match {r.similarity:+.2f}")

ax = bot.subplots()
groups = [("c_else", "Other bright stars, elsewhere", ROLES["control"], "x"),
          ("blank", "Spot with no bright star", ROLES["control"], "D"),
          ("c_spot", "Other bright stars, 36–60 px", ROLES["secondary"], None),
          ("k26", "Pictures listed by\nKarambelkar et al.", ROLES["reference"], None)]
for s_, lab, Q, mk in groups:
    d = res[res.set == s_]
    mk = mk or Q["marker"]
    kw = dict(facecolor="none", edgecolor=Q["color"], linewidths=0.9) if mk == "D" else dict(c=Q["color"], linewidths=0.9)
    ax.scatter(d.d + 1, d.similarity, s=18, marker=mk, label=f"{lab} ({len(d)})", alpha=0.9, zorder=2, **kw)
for s_, Q, mk, sz in (("VY CMa", ROLES["highlight"], "*", 70), ("eps Oct", ROLES["spherex"], "o", 26),
                      ("GY UMa", ROLES["spherex"], "D", 22)):
    d = nm[nm.star == s_]
    ax.scatter(d.d + 1, d.similarity, s=sz, marker=mk, color=Q["color"], edgecolor="white", linewidths=0.5,
               label=f"{s_} ({len(d)})", zorder=3)
ax.axvspan(0, 16, color=OKABE_ITO["grey"], alpha=0.2, lw=0, zorder=0, label="Within 15 px of the spot")
ax.axhline(0, color=INK, lw=0.6)
ax.set_xscale("log")
ax.set_xlim(0.8, 3000)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
ax.set_xlabel("Distance from the detector spot + 1 (px)")
ax.set_ylabel("Match to the shell pattern\n(correlation)")
h, l = ax.get_legend_handles_labels()
order = [4, 5, 6, 3, 2, 0, 1, 7]
ax.legend([h[i] for i in order], [l[i] for i in order], loc="upper left", bbox_to_anchor=(1.01, 1.02),
          handletextpad=0.3, labelspacing=0.45, fontsize=9)
panel_label(ax, "g")
save(fig, "e19_shield")
print({s_: int((res.set == s_).sum()) for s_ in res.set.unique()}, "n total", len(res))
