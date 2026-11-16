"""Tests of the change detector on real objects (scripts/run_e34.py), redrawn from stored results only.

Input: data/findings/e34.json ->
  real_stars.bins and real_stars.<method>.recall / ci95 (real stars moved to a new position in one picture);
  asteroids_fresh.by_V.<bin>.<method>.recall / wilson95 and .n (real asteroids in 16 new fields).
Methods: V2 (network, version 2), V1 (first network), M3BF / M3BK (median reference with the archive's flags,
at the detector thresholds / at the median-reference thresholds), M2K (one detector at a time).
Run from the repo root: .venv/bin/python figures/plot_e34_real.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROLES, WIDTH, panel_label, plt, save, use_style

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
use_style()
d = json.load(open(os.path.join(ROOT, "data/findings/e34.json")))
S, R, C, X = ROLES["spherex"], ROLES["reference"], ROLES["control"], ROLES["secondary"]

fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 3.6), layout="constrained", gridspec_kw=dict(width_ratios=[1.7, 1]))

st = d["real_stars"]
x = np.arange(len(st["bins"]))
for key, label, s in (("V2", "Network", S), ("V1", "First network", C), ("M3BF", "Median reference with flags", R)):
    lo, hi = np.array(st[key]["ci95"], float).T
    a.fill_between(x, lo, hi, color=s["color"], alpha=0.15, lw=0)
    a.plot(x, st[key]["recall"], ls=s["ls"], marker=s["marker"], ms=4.5, lw=1.4, color=s["color"], mec=s["color"] if s is C else "white", mew=0.8 if s is C else 0.5, label=label)
a.set_xticks(x, ["3–5", "5–8", "8–12", "12–20", "20–100", ">100"])
a.set_xlabel("Signal-to-noise ratio in one picture")
a.set_ylabel("Fraction recovered")
a.set_ylim(-0.02, 1.02)
panel_label(a, "a")

A = d["asteroids_fresh"]["by_V"]
bins = [("V16", "V ≤ 16"), ("V16_17p5", "V 16–17.5")]
meths = (("V2", S), ("V1", C), ("M3BK", R), ("M2K", X))
for j, (key, s) in enumerate(meths):
    for i, (bk, _) in enumerate(bins):
        r = A[bk][key]
        lo, hi = r["wilson95"]
        b.errorbar([i + (j - 1.5) * 0.14], [r["recall"]], yerr=[[r["recall"] - lo], [hi - r["recall"]]], fmt=s["marker"],
                   ms=5, color=s["color"], mec=s["color"] if s is C else "white", mew=0.8 if s is C else 0.5, elinewidth=1, capsize=0,
                   label={"V2": "Network", "V1": "First network", "M3BK": "Median reference with flags",
                          "M2K": "One detector at a time"}[key] if i == 0 and key == "M2K" else None)
b.set_xticks(range(len(bins)), [f"{l}\n({A[bk]['n']} sightings)" for bk, l in bins])
b.set_xlim(-0.6, len(bins) - 0.4)
b.set_ylim(-0.02, 1.02)
b.set_ylabel("Fraction of sightings marked")
panel_label(b, "b")
fig.legend(*a.get_legend_handles_labels(), loc="outside lower left", ncol=3, handlelength=2.2)
h, l = b.get_legend_handles_labels()
fig.legend(h, l, loc="outside lower right", handlelength=1.2)
save(fig, "e34_real")
