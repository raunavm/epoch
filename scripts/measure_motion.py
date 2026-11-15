"""Measure a nearby star's motion across SPHEREx DR1 and compare it with Gaia DR3.

Per frame (index footprint + exact WCS check; ≤ 250 frames spread in time): forced point-source fit with the
star's position free (orrery.fit.refine_position, ±1.5 px) jointly with Gaia DR3 neighbours and a background
plane. Hard-bad flags plus OVERFLOW (bit 1) are masked, so saturated cores are excluded and the unsaturated
wings carry the position. Positions are converted to sky offsets with the frame's own WCS.
Model: position(t) = Gaia DR3 position at J2016.0 + proper motion·(t − 2016.0) + parallax·(parallax factors from
Earth's barycentric position). Reported: per-pass mean offset from the Gaia prediction, and a proper motion
fitted to SPHEREx positions alone (parallax fixed to Gaia) vs Gaia's.
Gaia DR3 via the AIP mirror (Gaia Collaboration, Vallenari et al. 2023, A&A 674, A1).
Usage: .venv/bin/python scripts/measure_motion.py --name barnard --ra 269.452 --dec 4.6934 [--workers 2]
"""
import argparse, json, os, sys, warnings
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")
from astropy import log
log.setLevel("ERROR")
import astropy.units as u
from astropy.coordinates import get_body_barycentric
from astropy.time import Time
import pyvo

from orrery.data import Frame, BAD_BITS
from orrery.fit import fit_stamp, refine_position, flux_mJy
from orrery.forced import gaia_stars

MASK = BAD_BITS | (1 << 1)  # + OVERFLOW


def gaia_target(ra2000, dec2000, pmra, pmdec):
    ra16 = ra2000 + pmra * 16 / 3.6e6 / np.cos(np.radians(dec2000))
    dec16 = dec2000 + pmdec * 16 / 3.6e6
    s = pyvo.dal.TAPService("https://gaia.aip.de/tap")
    q = ("SELECT source_id, ra, dec, pmra, pmdec, parallax, ra_error, dec_error, pmra_error, pmdec_error, parallax_error, "
         "phot_g_mean_mag FROM gaiadr3.gaia_source WHERE 1=CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',"
         f"{ra16},{dec16},0.02))")
    t = s.run_sync(q).to_table().to_pandas()
    t["pm"] = np.hypot(t.pmra.fillna(0), t.pmdec.fillna(0))
    return t.sort_values("pm", ascending=False).iloc[0].to_dict()


def predict(g, mjd):
    """ICRS position at time t: J2016.0 + pm·dt + parallax displacement (standard parallax factors)."""
    t = Time(mjd, format="mjd")
    dt = t.jyear - 2016.0
    ra0, dec0 = np.radians(g["ra"]), np.radians(g["dec"])
    e = get_body_barycentric("earth", t).xyz.to(u.au).value  # X, Y, Z in au (ICRS)
    plx = g["parallax"] / 1000.0  # arcsec
    # parallax factors (e.g. Green, Spherical Astronomy): Δα* = π (X sinα − Y cosα), Δδ = π (X cosα sinδ + Y sinα sinδ − Z cosδ)
    dra_star = plx * (e[0] * np.sin(ra0) - e[1] * np.cos(ra0))
    ddec = plx * (e[0] * np.cos(ra0) * np.sin(dec0) + e[1] * np.sin(ra0) * np.sin(dec0) - e[2] * np.cos(dec0))
    ra = g["ra"] + (g["pmra"] * dt / 1000.0 + dra_star) / 3600.0 / np.cos(dec0)
    dec = g["dec"] + (g["pmdec"] * dt / 1000.0 + ddec) / 3600.0
    return ra, dec


def one(a):
    key, ra, dec, gid = a
    try:
        f = Frame(key)
        x, y = f.world2pix(ra, dec)
        x, y = float(x[0]), float(y[0])
        if not (20 <= x <= 2019 and 20 <= y <= 2019):
            return None
        st = f.stamp(x, y, half=15)
        xs, ys = x - st.x0, y - st.y0
        sra, sdec, sg = gaia_stars(ra, dec, 0.075, f.mjd, gmax=19.0)
        sxy = []
        if len(sra):
            gx, gy = f.world2pix(sra, sdec)
            for a_, b_ in zip(gx - st.x0, gy - st.y0):
                if -3 <= a_ <= 33 and -3 <= b_ <= 33 and np.hypot(a_ - xs, b_ - ys) > 1.5:  # exclude the target itself
                    sxy.append((a_, b_))
        sxy = np.array(sxy).reshape(-1, 2)
        xb, yb, surf, offs = refine_position(st.image, st.variance, st.flags, st.epsf, xs, ys, sxy, MASK, st.epsf_oversample, half=1.5, step=0.05)
        r = fit_stamp(st.image, st.variance, st.flags, st.epsf, xb, yb, sxy, MASK, st.epsf_oversample)
        # position error from the χ² surface curvature (Δχ² = 1), floored at the grid step
        c = surf - surf.min()
        okx = offs[np.where(c[np.argmin(surf) // surf.shape[1], :] <= 1)[0]]
        oky = offs[np.where(c[:, np.argmin(surf) % surf.shape[1]] <= 1)[0]]
        ex = max((okx.max() - okx.min()) / 2, 0.05) if len(okx) else 0.5
        ey = max((oky.max() - oky.min()) / 2, 0.05) if len(oky) else 0.5
        ram, decm = f.wcs.all_pix2world([[xb + st.x0, yb + st.y0]], 0)[0]
        # local scale for errors (arcsec per px)
        r1 = f.wcs.all_pix2world([[xb + st.x0 + 1, yb + st.y0]], 0)[0]
        scale = 3600 * np.hypot((r1[0] - ram) * np.cos(np.radians(decm)), r1[1] - decm)
        return dict(key=key, mjd=f.mjd, det=f.detector, wl=float(st.cwave[15, 15]), ra_pred=ra, dec_pred=dec,
                    ra_meas=float(ram), dec_meas=float(decm), err_arcsec=float(np.hypot(ex, ey) / np.sqrt(2) * scale),
                    snr=float(r.A / r.A_err) if r.A_err > 0 else np.nan, chi2_dof=float(r.chi2 / max(r.dof, 1)),
                    n_masked=int((~r.mask).sum()), flux_mJy=float(flux_mJy(r.A, st.omega_arcsec2)))
    except Exception as e:
        return dict(key=key, error=repr(e))


def main(a):
    g = gaia_target(a.ra, a.dec, a.pmra, a.pmdec)
    idx = pd.read_parquet("data/index/frames_dr1.parquet")
    idx = idx[idx.dqa == "Pass"]
    ra_m, dec_m = predict(g, np.full(len(idx), 61000.0))
    sep = np.degrees(np.arccos(np.clip(np.sin(np.radians(dec_m)) * np.sin(np.radians(idx.dec)) +
                                        np.cos(np.radians(dec_m)) * np.cos(np.radians(idx.dec)) * np.cos(np.radians(ra_m - idx.ra)), -1, 1)))
    cand = idx[sep < 2.6].sort_values("mjd")
    if len(cand) > a.cap:
        cand = cand.iloc[np.unique(np.linspace(0, len(cand) - 1, a.cap).round().astype(int))]
    ra_p, dec_p = predict(g, cand.mjd.values)
    args = [(k, float(r_), float(d_), int(g["source_id"])) for k, r_, d_ in zip(cand.key, ra_p, dec_p)]
    with ProcessPoolExecutor(a.workers) as ex:
        rows = [r for r in ex.map(one, args) if r]
    df = pd.DataFrame(rows)
    os.makedirs("data/findings", exist_ok=True)
    df.to_parquet(f"data/findings/motion_{a.name}.parquet")
    ok = df[df.get("error").isna() if "error" in df else slice(None)].dropna(subset=["ra_meas"])
    ok = ok[(ok.snr > 20) & np.isfinite(ok.err_arcsec)]
    # offsets from Gaia prediction, arcsec
    ok = ok.assign(dra=(ok.ra_meas - ok.ra_pred) * 3600 * np.cos(np.radians(ok.dec_pred)), ddec=(ok.dec_meas - ok.dec_pred) * 3600)
    gaps = np.where(np.diff(np.sort(ok.mjd.values)) > 30)[0]
    edges = [ok.mjd.min() - 1] + list(np.sort(ok.mjd.values)[gaps] + 1) + [ok.mjd.max() + 1]
    passes = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = ok[(ok.mjd > lo) & (ok.mjd <= hi)]
        if len(s) < 3:
            continue
        w = 1 / s.err_arcsec ** 2
        mr, md = np.sum(w * s.dra) / w.sum(), np.sum(w * s.ddec) / w.sum()
        # robust error: scatter-based (frames are not independent of PSF-model systematics)
        er = max(1 / np.sqrt(w.sum()), np.std(s.dra) / np.sqrt(len(s))); ed = max(1 / np.sqrt(w.sum()), np.std(s.ddec) / np.sqrt(len(s)))
        passes.append(dict(mjd_mean=float(s.mjd.mean()), n=int(len(s)), dra_arcsec=float(mr), dra_err=float(er), ddec_arcsec=float(md), ddec_err=float(ed)))
    # proper motion from SPHEREx alone: measured positions minus parallax term, linear in time
    t = Time(ok.mjd.values, format="mjd").jyear - 2016.0
    gp = dict(g, pmra=0.0, pmdec=0.0)
    ra_np, dec_np = predict(gp, ok.mjd.values)  # parallax only, no proper motion
    yra = (ok.ra_meas.values - ra_np) * 3600 * np.cos(np.radians(g["dec"]))
    ydec = (ok.dec_meas.values - dec_np) * 3600
    W = 1 / ok.err_arcsec.values ** 2
    A = np.vstack([np.ones_like(t), t]).T
    out = {}
    for name, y in (("pmra", yra), ("pmdec", ydec)):
        cov = np.linalg.inv(A.T @ (A * W[:, None]))
        c = cov @ (A.T @ (W * y))
        res = y - A @ c
        s2 = max(1.0, np.sum(W * res ** 2) / max(len(y) - 2, 1))  # inflate by reduced χ² if > 1
        out[name] = dict(spherex_mas_yr=float(c[1] * 1000), err_mas_yr=float(np.sqrt(cov[1, 1] * s2) * 1000), gaia_mas_yr=float(g[name]))
    res = dict(target=a.name, gaia_source_id=int(g["source_id"]), gaia=dict((k, float(g[k])) for k in ["ra", "dec", "pmra", "pmdec", "parallax", "phot_g_mean_mag"]),
               gaia_reference="Gaia Collaboration, Vallenari et al. 2023, A&A 674, A1 (Gaia DR3), via gaia.aip.de",
               n_frames_fit=int(len(ok)), n_frames_tried=int(len(df)), passes=passes, proper_motion=out,
               baseline_days=float(ok.mjd.max() - ok.mjd.min()), index="data/index/frames_dr1.parquet",
               method="scripts/measure_motion.py (free-position ePSF fit, saturated pixels masked; Gaia neighbours fitted)")
    json.dump(res, open(f"data/findings/motion_{a.name}.json", "w"), indent=1)
    print(json.dumps({k: res[k] for k in ["target", "n_frames_fit", "passes", "proper_motion"]}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--ra", type=float, required=True)
    ap.add_argument("--dec", type=float, required=True)
    ap.add_argument("--pmra", type=float, default=0.0)
    ap.add_argument("--pmdec", type=float, default=0.0)
    ap.add_argument("--cap", type=int, default=250)
    ap.add_argument("--workers", type=int, default=2)
    main(ap.parse_args())
