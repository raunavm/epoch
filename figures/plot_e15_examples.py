"""Redraw site/data/figs/e15_examples.png (scripts/run_e15.py, fig_examples()) from stored results only.

Inputs: data/phase2/e15/analysis.json (objects, error scale, methane index) and the stored per-picture forced fits
data/phase2/e15/forced/<obj>/*.json (same selection as run_e15.load_obj). Object picks follow the original rule.
Run from the repo root: .venv/bin/python figures/plot_e15_examples.py
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import INK, OKABE_ITO, ROLES, WIDTH, panel_label, plt, save, use_style

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
P = os.path.join(ROOT, "data/phase2/e15")
BINS = {"m33": (3.2, 3.4), "c30": (2.9, 3.1), "c40": (3.9, 4.1)}


def load_obj(o):  # copied from run_e15.load_obj
    rows = [json.load(open(f)) for f in glob.glob(f"{P}/forced/{o}/*.json") if not os.path.basename(f).startswith("_")]
    df = pd.DataFrame(rows)
    if not len(df) or "flux_mJy" not in df:
        return pd.DataFrame()
    df = df[df.on_frame & np.isfinite(df.flux_mJy) & np.isfinite(df.flux_err_mJy) & (df.flux_err_mJy > 0)]
    return df.reset_index(drop=True)


use_style()
out = json.load(open(f"{P}/analysis.json"))
S = ROLES["spherex"]

picks = []
for lo, hi in ((0, 10), (10, 20), (20, 30)):
    rr = [r for r in out["calibration"] if lo <= r["spt_code"] < hi and r["n_pictures"] > 0]
    picks.append(min(rr, key=lambda r: r["W2"]))                    # brightest in W2 (rule fixed in the protocol)
dec = [r for r in out.get("candidates", []) if r.get("decision") == "T or later"]
if dec:
    picks.append(max(dec, key=lambda r: r["z_from_threshold"]))

fig, axs = plt.subplots(len(picks), 1, figsize=(WIDTH, 1.75 * len(picks) + 0.6), sharex=True)
edges = np.arange(0.75, 5.0001, 0.1)
for k, (ax, r) in enumerate(zip(axs, picks)):
    df = load_obj(r["obj"])
    s = r["error_scale"]
    ax.axvspan(*BINS["c30"], color=OKABE_ITO["grey"], alpha=0.25, lw=0, zorder=0)
    ax.axvspan(*BINS["c40"], color=OKABE_ITO["grey"], alpha=0.25, lw=0, zorder=0)
    ax.axvspan(*BINS["m33"], color=ROLES["highlight"]["color"], alpha=0.25, lw=0, zorder=0, hatch="///",
               ec=ROLES["highlight"]["color"])
    ax.axhline(0, color=INK, lw=0.6, zorder=1)
    ax.plot(df.wl_um, df.flux_mJy, ".", color=S["color"], alpha=0.3, ms=3, mew=0, zorder=2, label="Single pictures")
    x, y, e = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        g = df[(df.wl_um >= lo) & (df.wl_um < hi)]
        if len(g):
            w = 1 / g.flux_err_mJy.values ** 2
            x.append(0.5 * (lo + hi)); y.append(np.sum(w * g.flux_mJy) / w.sum()); e.append(s / np.sqrt(w.sum()))
    ax.errorbar(x, y, yerr=e, fmt=S["marker"], color=S["color"], ms=4, mec="white", mew=0.5, elinewidth=1.0, zorder=3,
                label="0.1 µm bins")
    top = np.nanmax(y) if len(y) else 1
    ax.set_ylim(-0.25 * top, 1.2 * top)
    ax.set_ylabel("Flux (mJy)")
    panel_label(ax, "abcd"[k])
    print("abcd"[k], r["name"], r.get("spt", r.get("k21_photometric_type")),
          f"index {r.get('ch4_index', np.nan):.2f} ± {r.get('ch4_err', np.nan):.2f}", f"n = {len(df)}")
axs[-1].set_xlabel("Wavelength (µm)")
axs[-1].set_xlim(0.6, 5.1)
h, l = axs[0].get_legend_handles_labels()
h += [plt.Rectangle((0, 0), 1, 1, color=ROLES["highlight"]["color"], alpha=0.25, lw=0, hatch="///",
                    ec=ROLES["highlight"]["color"]),
      plt.Rectangle((0, 0), 1, 1, color=OKABE_ITO["grey"], alpha=0.25, lw=0)]
l += ["Methane window", "Reference windows"]
fig.legend(h, l, loc="outside upper center", ncol=4, handletextpad=0.4, columnspacing=1.5, markerscale=1.5)
save(fig, "e15_examples")
