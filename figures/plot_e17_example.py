"""Redraw site/data/figs/e17_example.png (scripts/run_e17.py, fig_example()) from stored results only.

Inputs: data/findings/e17.json (known.rows, threshold), data/phase2/e17/fields.json, data/phase2/e17/<field>/real.json
and known.json (the search's stored candidate and the catalogue match), and the stored per-exposure S/N maps
data/phase2/e17/<field>/ex.npz. Object pick follows the original rule (faintest recovered known asteroid with S/N >= 8).
Run from the repo root: .venv/bin/python figures/plot_e17_example.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMAGE_CMAP, OKABE_ITO, ROLES, WIDTH, panel_label, plt, save, use_style
from e17_stamps import MARGIN, ROOT, field, load_ex, matches_known, rate_grid, stamps_along

use_style()
H = ROLES["highlight"]["color"]
R = json.load(open(os.path.join(ROOT, "..", "..", "findings", "e17.json")))
G = rate_grid()
rec = [r for r in R["known"]["rows"] if r["recovered"]]
rec.sort(key=lambda r: (r["snr"] < 8, -r["V"]))
r = rec[0]
nm = r["field"]
f = field(nm)
real = json.load(open(f"{ROOT}/{nm}/real.json"))
T = R["threshold"]["snr"]
ks = json.load(open(f"{ROOT}/{nm}/known.json"))["objects"]
c = [c for c in real["candidates"] if (c.get("snr_logo") or 0) > T and matches_known(c, ks) == r["name"]][0]
EX, t, grp = load_ex(nm)
singles, stack, static, (use, gx, gy) = stamps_along(EX, t, grp, f["t_ref"], c, G)
tx, ty = gx[use], gy[use]
e_mid = np.where(use)[0][np.argmin(np.abs(t[use] - f["t_ref"]))]
cx, cy = MARGIN + c["x"], MARGIN + c["y"]
h = int(max(25, np.max(np.abs(tx - cx)), np.max(np.abs(ty - cy))) + 8)
sl = (slice(cy - h, cy + h + 1), slice(cx - h, cx + h + 1))
w = EX[e_mid]["w"][sl]
big = np.where(w > 0, EX[e_mid]["Fw"][sl] / np.sqrt(np.where(w > 0, w, 1)), np.nan)
pick = np.linspace(0, len(singles) - 1, min(6, len(singles))).round().astype(int)

cmap = plt.get_cmap(IMAGE_CMAP).copy()
cmap.set_bad(OKABE_ITO["sky_blue"])
kw = dict(cmap=cmap, vmin=-3, vmax=6, origin="upper", interpolation="nearest")

fig = plt.figure(figsize=(WIDTH, 3.75))
top, bot = fig.subfigures(2, 1, height_ratios=[2.35, 1.0], hspace=0.02)
a0, a5, a6 = top.subplots(1, 3)
a0.imshow(big, **kw)
a0.plot(tx - (cx - h), ty - (cy - h), "-", color=H, lw=1.0)
a0.plot(tx - (cx - h), ty - (cy - h), "o", mfc="none", mec=H, ms=6, mew=1.0)
a5.imshow(static, **kw)
a6.imshow(stack, **kw)
for a in (a0, a5, a6):
    a.set_xticks([]), a.set_yticks([])
    a.spines[:].set_visible(False)
panel_label(a0, "a"), panel_label(a5, "c"), panel_label(a6, "d")
axs = bot.subplots(1, len(pick))
for k, (a, i) in enumerate(zip(axs, pick)):
    a.imshow(singles[i][1], **kw)
    a.add_patch(plt.Circle((10, 10), 3, fill=False, color=H, lw=1.0))
    a.set_xticks([]), a.set_yticks([])
    a.spines[:].set_visible(False)
panel_label(axs[0], "b")
print("b times (h):", " ".join(f"{singles[i][0] * 24:+.1f}" for i in pick))
save(fig, "e17_example")
print(f"{r['name']} V {r['V']:.1f} speed {r['speed']:.2f} deg/day, exposures {int(use.sum())}, stack S/N {c['snr']:.1f}, "
      f"cutout {2 * h + 1} px, track span {(t[use].max() - t[use].min()) * 24:.0f} h, singles {len(singles)}")
