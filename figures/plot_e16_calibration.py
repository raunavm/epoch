"""Redraw site/data/figs/e16_calibration.png (scripts/run_e16.py, figure()) from stored results only.

Inputs (as in run_e16.figure): data/phase2/e16/model.json (frozen model, validation, sub-pixel diagnostics),
data/phase2/e16/calib_stars.json + calib_meas/*.parquet (per-picture positions, held-out half B) and
data/phase2/e16/validation_windows_B.parquet (visit averages before/after). Helper functions below are copied from
run_e16.py / run_e8.py (usable, add_orientation, ecl_east_pa, basis, predict_px, rob, clip_mean); nothing is refitted.
Run from the repo root: .venv/bin/python figures/plot_e16_calibration.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import INK, OKABE_ITO, ROLES, WIDTH, panel_label, plt, save, use_style

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "data/phase2/e16")
SEED, PIX, SNR_MIN, CLIP_PX = 16, 6.15, 20.0, 0.025


def rob(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(1.4826 * np.median(np.abs(v - np.median(v)))) if len(v) else np.nan


def clip_mean(v, n_sig=5.0, floor=0.0):
    v = np.asarray(v, float)
    med, s = np.median(v), rob(v)
    keep = np.abs(v - med) <= n_sig * max(s, floor, 1e-9)
    return float(v[keep].mean()), int(keep.sum()), keep


def ecl_east_pa(ra, dec):
    from astropy.coordinates import SkyCoord, GeocentricTrueEcliptic
    import astropy.units as u
    c = SkyCoord(ra * u.deg, dec * u.deg).transform_to(GeocentricTrueEcliptic())
    c2 = SkyCoord(c.lon + 0.01 * u.deg, c.lat, frame=GeocentricTrueEcliptic()).icrs
    de, dn = (c2.ra.deg - ra) * np.cos(np.radians(dec)), c2.dec.deg - dec
    n = np.hypot(de, dn)
    return float(de / n), float(dn / n)


def basis(x, y, deg):
    u, v = (np.asarray(x, float) - 1019.5) / 1020.0, (np.asarray(y, float) - 1019.5) / 1020.0
    cols = [np.ones_like(u)]
    if deg >= 1:
        cols += [u, v]
    if deg >= 2:
        cols += [u * u, u * v, v * v]
    return np.stack(cols, 1)


def predict_px(model, det, x, y, season):
    det, x, y, season = (np.atleast_1d(np.asarray(a)) for a in (det, x, y, season))
    mx, my = np.zeros(len(det)), np.zeros(len(det))
    B = basis(x, y, model["deg"])
    for i in range(len(det)):
        k = f"{int(det[i])}" + (f"_{int(season[i])}" if model["season"] else "")
        cf = model["coef"].get(k)
        if cf is not None:
            mx[i], my[i] = B[i] @ np.array(cf["dx"]), B[i] @ np.array(cf["dy"])
    return mx, my


def load_calib():
    stars = {s["source_id"]: s for s in json.load(open(f"{OUT}/calib_stars.json"))["stars"]}
    m = pd.concat([pd.read_parquet(f"{OUT}/calib_meas/{f}") for f in sorted(os.listdir(f"{OUT}/calib_meas")) if f.endswith(".parquet")])
    m = m.dropna(subset=["source_id"])
    m = m[np.isfinite(m.dx) & np.isfinite(m.dy) & (m.snr > SNR_MIN) & ~m.edge.astype(bool)].copy()
    ee = {sid: ecl_east_pa(s["ra"], s["dec"]) for sid, s in stars.items()}
    e = np.array([ee[s] for s in m.source_id])
    m["season"] = ((m.j_ex.values * e[:, 0] + m.j_nx.values * e[:, 1]) > 0).astype(int)
    return m.reset_index(drop=True)


use_style()
S, X, M, C = ROLES["spherex"], ROLES["secondary"], ROLES["model"], ROLES["control"]
mres = json.load(open(f"{OUT}/model.json"))
mdl, v = mres["model"], mres["validation_on_B"]
m = load_calib()
B = m[m.half == "B"].copy()
mx, my = predict_px(mdl, B.det.values, B.x.values, B.y.values, B.season.values)
B["dx_after"] = B.dx - mx

fig, ax = plt.subplots(2, 2, figsize=(WIDTH, 6.0))
eb = dict(ms=5, elinewidth=1.0, mew=0.5)


def cells(q, col):
    cm = [clip_mean(g[col].values, floor=CLIP_PX)[0] for _, g in q.groupby("source_id") if len(g) >= 3]
    return (1000 * np.mean(cm), 1000 * np.std(cm, ddof=1) / np.sqrt(len(cm))) if len(cm) > 2 else (np.nan, np.nan)


# (a) offset along detector x per detector and orientation, held-out half B: measured / after the calibration
a = ax[0, 0]
for d in range(1, 7):
    for s in (0, 1):
        q = B[(B.det == d) & (B.season == s)]
        for col, R, off in (("dx", S, -0.06), ("dx_after", X, 0.06)):
            mu, se = cells(q, col)
            a.errorbar(d + (s - 0.5) * 0.36 + off, mu, se, fmt=R["marker"], color=R["color"], mec="white", **eb)
a.axhline(0, color=INK, lw=0.6)
a.set_xticks(range(1, 7))
a.set_xticks([d + (s - 0.5) * 0.36 for d in range(1, 7) for s in (0, 1)], ["W", "E"] * 6, minor=True)
a.tick_params(axis="x", which="major", length=0, pad=13)
a.tick_params(axis="x", which="minor", length=2, labelsize=9)
a.set_xlabel("Detector and orientation of detector x")
a.set_ylabel("Offset along detector x (10$^{-3}$ px)")

# (b) east-west offset of visit averages against orientation (half B)
a = ax[0, 1]
wb = pd.read_parquet(f"{OUT}/validation_windows_B.parquet")
bins = np.arange(-180, 181, 45)
for col, R, off in (("de", S, -3), ("de_after", X, 3)):
    xs_, mu, lo_, hi_ = [], [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        q = wb[col][(wb.ang >= lo) & (wb.ang < hi)].values * 1000
        if len(q) < 20:
            continue
        bs = [np.median(np.random.default_rng(SEED + i).choice(q, len(q))) for i in range(300)]
        xs_.append(0.5 * (lo + hi) + off); mu.append(np.median(q))
        lo_.append(np.median(q) - np.percentile(bs, 16)); hi_.append(np.percentile(bs, 84) - np.median(q))
    a.errorbar(xs_, mu, [lo_, hi_], fmt=R["marker"], color=R["color"], mec="white", zorder=3, **eb)
tt = np.linspace(-180, 180, 200)
a.plot(tt, 1000 * PIX * float(np.mean([mdl["coef"][k]["dx"][0] for k in mdl["coef"]])) * np.cos(np.radians(tt)),
       ls=M["ls"], color=M["color"], lw=1.3, zorder=2)
a.axhline(0, color=INK, lw=0.6)
a.set_xticks([-180, -90, 0, 90, 180])
a.set_xlabel("Direction of detector x on sky (° from east)")
a.set_ylabel("East–west offset from Gaia (mas)")

# (c) sub-pixel phase (all check stars), from the stored diagnostics
a = ax[1, 0]
ph = mres["diagnostics"]["by_subpixel_phase_x"]
order = np.argsort([float(p["group"].split(",")[0].strip("(")) for p in ph])
xc = np.array([np.mean([float(t) for t in ph[i]["group"].strip("(]").split(",")]) for i in order])
a.errorbar(xc - 0.01, [1000 * ph[i]["dx_mean"] for i in order], [1000 * ph[i]["dx_err"] for i in order],
           fmt=S["marker"], ls=S["ls"], color=S["color"], mec="white", lw=1.2, **eb)
a.errorbar(xc + 0.01, [1000 * ph[i]["rdx_mean"] for i in order], [1000 * ph[i]["rdx_err"] for i in order],
           fmt=C["marker"], ls=C["ls"], color=C["color"], lw=1.2, **eb)
a.axhline(0, color=INK, lw=0.6)
a.set_xlabel("Gaia position inside the pixel, along x (px)")
a.set_ylabel("Offset along detector x (10$^{-3}$ px)")

# (d) the pre-registered validation: visit-average offsets before / after
a = ax[1, 1]
r = np.r_[wb.de, wb.dn] * 1000
r2 = np.r_[wb.de_after, wb.dn_after] * 1000
bb = np.linspace(-800, 800, 65)
a.axvspan(-45, 45, color=OKABE_ITO["grey"], alpha=0.25, lw=0, zorder=0)
a.hist(np.clip(r, -799, 799), bb, color=S["color"], alpha=0.45, zorder=1)
a.hist(np.clip(r2, -799, 799), bb, histtype="step", lw=1.4, color=X["color"], zorder=2)
a.set_xlabel("Visit-average offset from Gaia (mas)")
a.set_ylabel("Star-visits (count)")

for k, a_ in enumerate(ax.ravel()):
    panel_label(a_, "abcd"[k])

handles = [Line2D([], [], marker=S["marker"], color=S["color"], mec="white", ls="none", ms=6),
           Line2D([], [], marker=X["marker"], color=X["color"], mec="white", ls="none", ms=6),
           Line2D([], [], ls=M["ls"], color=M["color"], lw=1.3),
           Line2D([], [], marker=C["marker"], ls=C["ls"], color=C["color"], ms=6),
           Patch(color=OKABE_ITO["grey"], alpha=0.25, lw=0)]
labels = ["Measured", "After subtracting the calibration", "Expected from a fixed +0.007 px offset",
          "Refined between grid points (c)", "Size of the correction (d)"]
fig.legend(handles, labels, loc="outside upper center", ncol=2, columnspacing=1.5, handletextpad=0.5)
save(fig, "e16_calibration")
print(f"stars measured {mres['n_stars_measured']}, half-B stars {v['n_stars']}, star-visits {v['n_star_windows']}, "
      f"RMS {1000 * v['rms_before_arcsec']:.0f} -> {1000 * v['rms_after_arcsec']:.0f} mas, reduction {100 * v['reduction']:.1f}%, "
      f"mean dx coef {np.mean([mdl['coef'][k]['dx'][0] for k in mdl['coef']]):.4f} px; half-B pictures {len(B)}")
