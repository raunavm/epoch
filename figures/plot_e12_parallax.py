"""Redraw site/data/figs/e12_parallax.png (scripts/run_e12.py, figure()) from stored results only.

Inputs: data/findings/e12.json (fit: params, params_no_parallax, ref_ra, ref_dec, ref_epoch_jyear, windows) and the
per-picture positions data/findings/motion_epsindib.parquet. No fitting is done here: the stored parameters are
evaluated on a time grid with the same design matrix as scripts/run_e12.py (Earth's barycentric position from astropy's
built-in ephemeris). The one picture the frozen clip removed is identified as the one whose removal reproduces the
stored window means.
Run from the repo root: .venv/bin/python figures/plot_e12_parallax.py
"""
import json
import os
import sys

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import get_body_barycentric
from astropy.time import Time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import ROLES, WIDTH, panel_label, plt, save, use_style

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def windows(mjd, gap=30.0):  # as scripts/run_e12.py
    o = np.argsort(mjd)
    lab = np.zeros(len(mjd), int)
    lab[o] = np.concatenate([[0], np.cumsum(np.diff(np.asarray(mjd)[o]) > gap)])
    return lab


def design(mjd, ra0, dec0, t0):  # as scripts/run_e12.py (with parallax column)
    t = Time(np.asarray(mjd), format="mjd")
    dt = t.jyear - t0
    X, Y, Z = get_body_barycentric("earth", t).xyz.to(u.au).value
    a, d = np.radians(ra0), np.radians(dec0)
    pa = X * np.sin(a) - Y * np.cos(a)
    pd_ = X * np.cos(a) * np.sin(d) + Y * np.sin(a) * np.sin(d) - Z * np.cos(d)
    o, z = np.ones_like(dt), np.zeros_like(dt)
    return np.block([[o[:, None], z[:, None], dt[:, None], z[:, None], pa[:, None]],
                     [z[:, None], o[:, None], z[:, None], dt[:, None], pd_[:, None]]])


use_style()
res = json.load(open(os.path.join(ROOT, "data/findings/e12.json")))
r = res["fit"]
ra0, dec0, t0 = r["ref_ra"], r["ref_dec"], r["ref_epoch_jyear"]
p = np.array([r["params"][k] for k in ("dra0_arcsec", "ddec0_arcsec", "pmra_arcsec_yr", "pmdec_arcsec_yr", "parallax_arcsec")])
p4 = np.array(r["params_no_parallax"])

# per-picture positions with the same selection as fit5()
d = pd.read_parquet(os.path.join(ROOT, "data/findings/motion_epsindib.parquet")).dropna(subset=["ra_meas"])
d = d[(d.snr > 20) & np.isfinite(d.err_arcsec)].sort_values("mjd").reset_index(drop=True)
assert np.isclose(float(np.median(d.ra_meas)), ra0) and np.isclose(float(np.median(d.dec_meas)), dec0) and len(d) == r["n_selected"]
x = (d.ra_meas.values - ra0) * 3600 * np.cos(np.radians(dec0))
y = (d.dec_meas.values - dec0) * 3600
w = windows(d.mjd.values)
keep = np.ones(len(d), bool)
for v in r["windows"]:
    idx = np.where(w == v["window"])[0]
    n_drop = len(idx) - v["n"]
    if n_drop == 0:
        continue
    assert n_drop == 1
    hit = [i for i in idx if np.isclose(np.delete(x, i)[np.delete(w, i) == v["window"]].mean(), v["x_arcsec"], rtol=0, atol=1e-9)
           and np.isclose(np.delete(y, i)[np.delete(w, i) == v["window"]].mean(), v["y_arcsec"], rtol=0, atol=1e-9)]
    assert len(hit) == 1
    keep[hit[0]] = False
assert (~keep).sum() == r["n_clipped"]

tt = np.linspace(d.mjd.min() - 20, d.mjd.max() + 20, 600)
A = design(tt, ra0, dec0, t0)
n = len(tt)
m5, m4, lin = A @ p, A[:, :4] @ p4, A[:, :4] @ p[:4]
lin_d = design(d.mjd.values, ra0, dec0, t0)[:, :4] @ p[:4]
nd = len(d)

S, M, C = ROLES["spherex"], ROLES["model"], ROLES["control"]
fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 3.9))

# (a) path on the sky
a.plot(x[keep], y[keep], ls="none", marker=S["marker"], ms=2.2, color=S["color"], alpha=0.3, mew=0, zorder=1,
       label="SPHEREx, single pictures")
a.plot(m4[:n], m4[n:], ls=C["ls"], lw=1.5, color=C["color"], zorder=2, label="Steady motion only")
a.plot(m5[:n], m5[n:], ls=M["ls"], lw=1.4, color=M["color"], zorder=3, label="Steady motion + yearly wobble")
for v in r["windows"]:
    a.errorbar(v["x_arcsec"], v["y_arcsec"], xerr=v["x_err"], yerr=v["y_err"], fmt=S["marker"], ms=5.5, color=S["color"],
               mec="white", mew=0.7, elinewidth=1.2, zorder=4)
a.invert_xaxis()
a.set_aspect("equal", adjustable="datalim")
a.set_xlabel("East–west position (arcsec)")
a.set_ylabel("North–south position (arcsec)")
panel_label(a, "a")

# (b) the yearly wobble: steady-motion part of the full model removed
b.axhline(0, color=C["color"], lw=0.6, zorder=0)
b.axvline(0, color=C["color"], lw=0.6, zorder=0)
b.plot((m5 - lin)[:n], (m5 - lin)[n:], ls=M["ls"], lw=1.4, color=M["color"], zorder=2)
for v in r["windows"]:
    mk = (w == v["window"]) & keep
    rx, ry = (x - lin_d[:nd])[mk].mean(), (y - lin_d[nd:])[mk].mean()
    b.errorbar(rx, ry, xerr=v["x_err"], yerr=v["y_err"], fmt=S["marker"], ms=5.5, color=S["color"], mec="white",
               mew=0.7, elinewidth=1.2, zorder=4)
b.invert_xaxis()
b.set_aspect("equal", adjustable="datalim")
b.set_xlabel("East–west shift (arcsec)")
b.set_ylabel("North–south shift (arcsec)")
panel_label(b, "b")

h, l = a.get_legend_handles_labels()
h.insert(1, plt.Line2D([], [], ls="none", marker=S["marker"], ms=5.5, color=S["color"], mec="white", mew=0.7))
l.insert(1, "SPHEREx, average of each visit")
fig.legend(h, l, loc="outside lower center", ncol=2, columnspacing=2.0, handletextpad=0.5)
save(fig, "e12_parallax")
