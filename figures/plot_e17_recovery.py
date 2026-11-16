"""Redraw site/data/figs/e17_recovery.png (scripts/run_e17.py, fig_recovery()) from stored results only.

Input: data/findings/e17.json (injections.curve, known.rows, threshold, fields_complete).
Run from the repo root: .venv/bin/python figures/plot_e17_recovery.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import INK, ROLES, WIDTH, plt, save, use_style

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

use_style()
R = json.load(open(os.path.join(ROOT, "data/findings/e17.json")))
fig, ax = plt.subplots(figsize=(WIDTH, 4.0))

cv = R["injections"]["curve"]
xc = [np.mean(c["V"]) for c in cv]
for key_n, key_r, lab, role in (("n", "rec", "Planted, all speeds", "spherex"),
                                 ("n_trackable", "rec_trackable", "Planted, seen\nin ≥ 2 visits", "secondary")):
    Q = ROLES[role]
    n = np.array([c[key_n] for c in cv], float); k = np.array([c[key_r] for c in cv], float)
    fr = np.where(n > 0, k / np.maximum(n, 1), np.nan)
    err = np.sqrt(np.maximum(fr * (1 - fr), 0.25 / np.maximum(n, 1)) / np.maximum(n, 1))
    ax.errorbar(xc, fr, yerr=err, marker=Q["marker"], ls=Q["ls"], color=Q["color"], lw=1.3, ms=5.5, mec="white",
                mew=0.6, elinewidth=1.0, label=lab, zorder=3)

F = ROLES["reference"]
kr = R["known"]["rows"]
kv = np.array([r["V"] for r in kr]); kk = np.array([r["recovered"] for r in kr])
bins = np.arange(np.floor(kv.min()), np.ceil(kv.max()) + 1.01, 1.0)
for a, b in zip(bins[:-1], bins[1:]):
    s = (kv >= a) & (kv < b)
    if s.sum():
        ax.plot((a + b) / 2, kk[s].mean(), F["marker"], color=F["color"], ms=7, mec="white", mew=0.6, zorder=5)
        print(f"known V {a:.0f}-{b:.0f}: {int(kk[s].sum())}/{int(s.sum())}")
ax.plot([], [], F["marker"], color=F["color"], ms=7, mec="white", label="Catalogued\nasteroids\n(found/present)")

ax.axhline(0.5, color=INK, ls=":", lw=0.6, zorder=1)
ax.set_xlabel("Brightness (V magnitude)")
ax.set_ylabel("Fraction found (blind search)")
ax.set_ylim(-0.03, 1.1)
sec = ax.secondary_xaxis("top", functions=(lambda v: 3.636e6 * 10 ** (-0.4 * np.asarray(v)) * 1.297,
                                            lambda m: -2.5 * np.log10(np.maximum(np.asarray(m), 1e-9) / (3.636e6 * 1.297))))
sec.set_xticks([1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01])
sec.set_xticklabels(["1", "0.5", "0.2", "0.1", "0.05", "0.02", "0.01"])
sec.set_xlabel("Flux density at 1 µm, solar colour (mJy)")
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), handletextpad=0.5)
save(fig, "e17_recovery")
print(f"planted n {sum(c['n'] for c in cv)}, found {sum(c['rec'] for c in cv)}; known {len(kr)}, found {int(kk.sum())}; "
      f"threshold S/N {R['threshold']['snr']}, {R['threshold']['n_null_searches']} null searches, fields {len(R['fields_complete'])}")
