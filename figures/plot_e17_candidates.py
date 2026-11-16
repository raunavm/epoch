"""Redraw site/data/figs/e17_candidates.png (scripts/run_e17.py, cmd_figs() + fig_strip()) from stored results only.

Inputs: data/findings/e17.json (unknown_survivors, candidates_above_T), data/phase2/e17/fields.json and the stored
per-exposure S/N maps data/phase2/e17/<field>/ex.npz. Same candidates (first 6 of the list) and the same first 12
exposures per strip as the original.
Run from the repo root: .venv/bin/python figures/plot_e17_candidates.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMAGE_CMAP, OKABE_ITO, ROLES, WIDTH, panel_label, plt, save, use_style
from e17_stamps import ROOT, field, load_ex, rate_grid, stamps_along

use_style()
H = ROLES["highlight"]["color"]
R = json.load(open(os.path.join(ROOT, "..", "..", "findings", "e17.json")))
G = rate_grid()
show = (R["unknown_survivors"] or R["candidates_above_T"])[:6]
cmap = plt.get_cmap(IMAGE_CMAP).copy()
cmap.set_bad(OKABE_ITO["sky_blue"])
kw = dict(cmap=cmap, vmin=-3, vmax=6, origin="upper", interpolation="nearest")

fig = plt.figure(figsize=(WIDTH, 2.25 * len(show)))
subs = fig.subfigures(len(show), 1, hspace=0.06)
cache = {}
for i, (sf, c) in enumerate(zip(subs if len(show) > 1 else [subs], show)):
    if c["field"] not in cache:
        cache = {c["field"]: load_ex(c["field"])}
    EX, t, grp = cache[c["field"]]
    singles, stack, static, (use, _, _) = stamps_along(EX, t, grp, field(c["field"])["t_ref"], c, G)
    singles = singles[:12]
    axs = sf.subplots(2, 7, gridspec_kw=dict(width_ratios=[1] * 6 + [1.25]))
    print("ab"[i], "times (h):", " ".join(f"{dt * 24:+.1f}" for dt, _ in singles))
    cells = [(axs[k // 6, k % 6], im, True) for k, (_, im) in enumerate(singles)]
    cells += [(axs[0, 6], static, False), (axs[1, 6], stack, False)]
    for a, im, ring in cells:
        a.imshow(im, **kw)
        if ring:
            a.add_patch(plt.Circle((10, 10), 3, fill=False, color=H, lw=0.9))
    for a in axs.ravel():
        a.set_xticks([]), a.set_yticks([])
        a.spines[:].set_visible(False)
    panel_label(axs[0, 0], "ab"[i])
    print("ab"[i], c["field"], c["x"], c["y"], f"stack S/N {c['snr']:.1f}, {c['speed']:.2f} deg/day, pass_all {c['checks']['pass_all']}, "
          f"exposures used {int(use.sum())}, shown {len(singles)}")
save(fig, "e17_candidates")
