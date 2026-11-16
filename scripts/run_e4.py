"""E4 (docs/PROTOCOLS.md): is WISE 0855-0714 detected in SPHEREx DR1 when frames are combined along its path?

Stages (each resumable; partial results on disk):
  astrometry  literature astrometry -> reference position (data/findings/e4_astrometry.json)
  forced      per-frame forced fit at the predicted position, jointly with Gaia DR3 neighbours
              (orrery.forced.measure_frame); one JSON per frame in data/forced/wise0855_e4/,
              stamps in data/stamps/wise0855/
  analyze     primary statistic per pass + 24 null tracks; decision; data/findings/e4.json, site/data/e4/
  slider      rebuild only the picture-count slider (site/data/e4/reveal.json) from the Nov-Dec 2025 window
Usage: OMP_NUM_THREADS=1 .venv/bin/python scripts/run_e4.py astrometry|forced|analyze|slider [--workers 3]
"""
import argparse, glob, json, os, sys, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")
from astropy import log
log.setLevel("ERROR")
from astropy.time import Time
from astropy.coordinates import get_body_barycentric

INDEX = "data/index/frames_dr1.parquet"
ASTRO = "data/findings/e4_astrometry.json"
FDIR = "data/forced/wise0855_e4"
SDIR = "data/stamps/wise0855"
GAIA = "data/cache/gaia/wise0855_e4_r0.12.parquet"
AU_PER_PC = 206264.806

# Luhman & Esplin 2016, AJ 152, 78 (arXiv:1605.06655), Table 2: astrometry (J2000/ICRS deg, sigma arcsec, MJD, source)
LE16_T2 = [
    (133.7952573, 0.125, -7.2450910, 0.135, 55320.38, "WISE"), (133.7943232, 0.133, -7.2450719, 0.142, 55511.35, "WISE"),
    (133.7881778, 0.025, -7.2445255, 0.025, 56464.51, "Spitzer"), (133.7870855, 0.025, -7.2444659, 0.025, 56677.29, "Spitzer"),
    (133.7868470, 0.025, -7.2444642, 0.025, 56712.35, "Spitzer"), (133.7858416, 0.025, -7.2443262, 0.025, 56839.71, "Spitzer"),
    (133.7851503, 0.025, -7.2443041, 0.025, 56986.82, "HST"), (133.7847553, 0.025, -7.2442609, 0.025, 57051.25, "Spitzer"),
    (133.7843602, 0.020, -7.2442366, 0.020, 57084.82, "HST"), (133.7845153, 0.025, -7.2442655, 0.025, 57086.16, "Spitzer"),
    (133.7840688, 0.020, -7.2441796, 0.020, 57123.17, "HST"), (133.7835385, 0.025, -7.2441445, 0.025, 57209.43, "Spitzer"),
    (133.7819058, 0.020, -7.2440115, 0.020, 57474.62, "HST"), (133.7818923, 0.020, -7.2440157, 0.020, 57475.41, "HST"),
    (133.7818300, 0.020, -7.2440063, 0.020, 57484.38, "HST")]
# LE16 Table 4
PLX, PMRA, PMDEC = 0.449, -8.118, 0.680           # arcsec, arcsec/yr (pmra includes cos dec)
PLX_E, PMRA_E, PMDEC_E = 0.008, 0.008, 0.007
T0 = 57484.38                                      # reference epoch (MJD) = last HST epoch of LE16 Table 2


def earth_bary_au(mjd):
    e = get_body_barycentric("earth", Time(np.atleast_1d(mjd), format="mjd", scale="utc"))
    return np.stack([e.x.to("au").value, e.y.to("au").value, e.z.to("au").value], -1)


def predict(ra0, dec0, mjd, plx=PLX, pmra=PMRA, pmdec=PMDEC, t0=T0):
    """Apparent ICRS position seen from the Earth: barycentric linear motion + parallax (vector form)."""
    mjd = np.atleast_1d(np.asarray(mjd, float))
    dt = (mjd - t0) / 365.25
    ra = np.radians(ra0) + np.radians(pmra * dt / 3600) / np.cos(np.radians(dec0))
    dec = np.radians(dec0) + np.radians(pmdec * dt / 3600)
    u = np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], -1)
    v = u * (AU_PER_PC / plx) - earth_bary_au(mjd)      # star position relative to the observer, AU
    ra_o = np.degrees(np.arctan2(v[:, 1], v[:, 0])) % 360
    dec_o = np.degrees(np.arcsin(v[:, 2] / np.linalg.norm(v, axis=1)))
    return ra_o, dec_o


def stage_astrometry():
    t = pd.DataFrame(LE16_T2, columns=["ra", "sra", "dec", "sdec", "mjd", "src"])
    # Reference position: weighted 2-parameter fit to the six HST epochs (Earth-orbit observer, like SPHEREx),
    # with the LE16 proper motion and parallax fixed. Spitzer epochs are not used (Spitzer was ~1 AU from the
    # Earth by 2013-2015, so the Earth-based parallax factor does not apply to them).
    h = t[t.src == "HST"]
    r0, d0 = 133.7819, -7.2440
    for _ in range(3):
        pr, pd_ = predict(r0, d0, h.mjd.values)
        w = 1 / h.sra.values ** 2
        r0 += np.sum(w * (h.ra.values - pr)) / w.sum()
        d0 += np.sum(w * (h.dec.values - pd_)) / w.sum()
    out = []
    for _, row in t.iterrows():
        pr, pd_ = predict(r0, d0, row.mjd)
        out.append(dict(mjd=row.mjd, src=row.src, dra_arcsec=round(float((row.ra - pr[0]) * 3600 * np.cos(np.radians(d0))), 3),
                        ddec_arcsec=round(float((row.dec - pd_[0]) * 3600), 3)))
    hres = [o for o in out if o["src"] == "HST"]
    rms = float(np.sqrt(np.mean([o["dra_arcsec"] ** 2 + o["ddec_arcsec"] ** 2 for o in hres]) / 2))
    # challenge position (08h55m10.831s -07d14m42.53s) vs this solution at the first WISE epoch
    ch_ra, ch_dec = (8 + 55 / 60 + 10.831 / 3600) * 15, -(7 + 14 / 60 + 42.53 / 3600)
    pr, pd_ = predict(r0, d0, 55320.38)
    a = dict(object="WISE J085510.83-071442.5 (WISE 0855-0714)",
             citation="Luhman, K. L. & Esplin, T. L. 2016, AJ 152, 78, 'The Spectral Energy Distribution of the Coldest Known Brown Dwarf', arXiv:1605.06655 (text read: Sec. 3.1, Tables 2 and 4)",
             parallax_arcsec=PLX, parallax_err=PLX_E, pmra_cosdec_arcsec_yr=PMRA, pmra_err=PMRA_E,
             pmdec_arcsec_yr=PMDEC, pmdec_err=PMDEC_E,
             ref_epoch_mjd=T0, ref_ra_deg=round(r0, 7), ref_dec_deg=round(d0, 7), ref_frame="ICRS (LE16 'J2000' table positions)",
             ref_position_method="weighted fit of (RA, Dec) at MJD 57484.38 to the 6 HST epochs of LE16 Table 2 with LE16 Table 4 proper motion and parallax fixed; Earth-barycentric parallax (astropy get_body_barycentric, builtin ephemeris)",
             hst_residual_rms_arcsec_per_axis=round(rms, 4), residuals_all_epochs=out,
             residual_note="Spitzer residuals include Spitzer's own (non-Earth) parallax; WISE residuals have 0.13 arcsec errors",
             challenge_position=dict(ra_deg=round(ch_ra, 6), dec_deg=round(ch_dec, 6),
                                     offset_from_solution_at_mjd55320_arcsec=[round(float((ch_ra - pr[0]) * 3600 * np.cos(np.radians(d0))), 2),
                                                                             round(float((ch_dec - pd_[0]) * 3600), 2)]),
             other_values_checked="Luhman & Esplin 2014 (quoted in LE16 Sec. 3.1): pm (-8.10, +0.70) arcsec/yr +-0.02, parallax 0.433+-0.015; the challenge's rough values (-8.1, +0.7, 0.44) agree.",
             note="SPHEREx low-Earth-orbit offset ignored (< 1e-4 AU, < 0.05 mas at 2.2 pc). Uncertainty of the predicted position in 2025-26 from LE16 errors: ~0.008*10 yr = 0.08 arcsec = 0.013 px.")
    json.dump(a, open(ASTRO, "w"), indent=1)
    print(json.dumps({k: v for k, v in a.items() if k != "residuals_all_epochs"}, indent=1))


# ---------------------------------------------------------------- forced stage
def gaia_field():
    """One Gaia DR3 query (AIP mirror, as orrery.forced) covering the whole path; cached."""
    if os.path.exists(GAIA):
        return pd.read_parquet(GAIA)
    import pyvo
    from orrery.forced import GAIA_TAP
    a = json.load(open(ASTRO))
    q = ("SELECT ra, dec, pmra, pmdec, phot_g_mean_mag FROM gaiadr3.gaia_source WHERE "
         f"1=CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',{a['ref_ra_deg'] - 0.0025},{a['ref_dec_deg']},0.12)) AND phot_g_mean_mag<19.5")
    df = pyvo.dal.TAPService(GAIA_TAP).run_sync(q).to_table().to_pandas()
    df.to_parquet(GAIA)
    return df


def _gaia_stars_local(ra, dec, radius_deg, mjd, gmax=19.5, cache_dir=None):
    """Drop-in for orrery.forced.gaia_stars using the single cached field query (same J2016 propagation)."""
    df = pd.read_parquet(GAIA)
    cosd = np.cos(np.radians(dec))
    sep = np.hypot((df.ra.values - ra) * cosd, df.dec.values - dec)
    df = df[(sep < radius_deg) & (df.phot_g_mean_mag.values < gmax)]
    dt = Time(mjd, format="mjd").jyear - 2016.0
    d_dec = df["pmdec"].fillna(0).values * dt / 3.6e6
    d_ra = df["pmra"].fillna(0).values * dt / 3.6e6 / np.cos(np.radians(df["dec"].values))
    return df["ra"].values + d_ra, df["dec"].values + d_dec, df["phot_g_mean_mag"].values


def _job(args):
    key, ra, dec, out = args
    import orrery.forced as F
    F.gaia_stars = _gaia_stars_local
    try:
        r = F.measure_frame(key, {"RA": ra, "DEC": dec}, stamp_dir=SDIR)
        r = dict(key=key, on_frame=False) if r is None else dict(r, on_frame=True)
    except Exception as e:
        r = dict(key=key, error=repr(e))
    r.update(pred_ra=ra, pred_dec=dec)
    if "error" not in r:
        json.dump({k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()}, open(out, "w"))
    return r


def in_quad(cra, cdec, ra, dec):
    """Point inside the 4-corner footprint (gnomonic projection about the point)."""
    ra0, d0 = np.radians(ra), np.radians(dec)
    a, d = np.radians(np.asarray(cra, float)), np.radians(np.asarray(cdec, float))
    c = np.sin(d0) * np.sin(d) + np.cos(d0) * np.cos(d) * np.cos(a - ra0)
    x = np.cos(d) * np.sin(a - ra0) / c
    y = (np.cos(d0) * np.sin(d) - np.sin(d0) * np.cos(d) * np.cos(a - ra0)) / c
    s = [np.sign(x[i] * y[(i + 1) % 4] - y[i] * x[(i + 1) % 4]) for i in range(4)]  # origin vs each edge
    return all(v == s[0] for v in s)


def candidates():
    a = json.load(open(ASTRO))
    idx = pd.read_parquet(INDEX)
    n_all = len(idx)
    cosd = np.sin(np.radians(a["ref_dec_deg"])) * np.sin(np.radians(idx.dec.values)) + np.cos(np.radians(a["ref_dec_deg"])) * \
        np.cos(np.radians(idx.dec.values)) * np.cos(np.radians(idx.ra.values - a["ref_ra_deg"]))
    c = idx[np.degrees(np.arccos(np.clip(cosd, -1, 1))) < 2.6].copy()
    c["pred_ra"], c["pred_dec"] = predict(a["ref_ra_deg"], a["ref_dec_deg"], c.mjd.values, t0=a["ref_epoch_mjd"])
    c["in_fp"] = [in_quad(r.c_ra, r.c_dec, r.pred_ra, r.pred_dec) for r in c.itertuples()]
    c = c[c.in_fp]
    return c, n_all


def stage_forced(workers):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    os.makedirs(FDIR, exist_ok=True)
    gaia_field()
    c, n_all = candidates()
    n_fp = len(c)
    n_notpass = int((c.dqa != "Pass").sum())
    c = c[c.dqa == "Pass"]
    c = c.assign(o=c.detector.map({6: 0, 5: 1, 4: 2}).fillna(3)).sort_values(["o", "mjd"])  # 4.4-5.0 um detector first
    todo = [(r.key, float(r.pred_ra), float(r.pred_dec), f"{FDIR}/{r.obs_id}_D{r.detector}.json") for r in c.itertuples()]
    todo = [t for t in todo if not os.path.exists(t[3])]
    print(f"index rows {n_all}; footprint {n_fp} (non-Pass dqa excluded: {n_notpass}); to do {len(todo)}", flush=True)
    json.dump(dict(n_index=n_all, n_footprint=n_fp, n_dqa_not_pass=n_notpass, n_pass=len(c)), open(f"{FDIR}/_counts.json", "w"))
    with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(_job, t) for t in todo]
        for i, f in enumerate(as_completed(futs)):
            r = f.result()
            if "error" in r or i % 20 == 0:
                print(i, r.get("key", "")[-45:], r.get("error", ""), r.get("wl_um"), r.get("snr"), flush=True)


# ---------------------------------------------------------------- analysis stage
def load_rows():
    rows = [json.load(open(f)) for f in sorted(glob.glob(f"{FDIR}/*.json")) if not os.path.basename(f).startswith("_")]
    df = pd.DataFrame(rows)
    return df[df.on_frame].sort_values("mjd").reset_index(drop=True), len(df)


def passes_of(mjd, gap=30.0):
    k = np.concatenate([[0], np.cumsum(np.diff(mjd) > gap)])
    return k


HALF = 15
# 24 null tracks (PROTOCOLS E4): 12 at 6 px (angles k*30 deg), 12 at 9 px (k*30 + 15 deg), stamp pixel axes
NULLS = [(6 * np.cos(np.radians(30 * k)), 6 * np.sin(np.radians(30 * k))) for k in range(12)] + \
        [(9 * np.cos(np.radians(30 * k + 15)), 9 * np.sin(np.radians(30 * k + 15))) for k in range(12)]
# Secondary template: Spitzer/IRAC photometry of WISE 0855 (LE16 Table 3, Vega; means of the listed epochs)
# [3.6] = 17.34, [4.5] = 13.87; IRAC zero points 280.9 / 179.7 Jy and isophotal wavelengths 3.550 / 4.493 um
# (Reach et al. 2005, PASP 117, 978). Log-linear in flux between the two bands, held constant beyond 4.493 um.
F36, F45 = 280.9e3 * 10 ** (-0.4 * 17.34), 179.7e3 * 10 ** (-0.4 * 13.87)  # mJy
L36, L45 = 3.550, 4.493
W2_MJY = 171.787e3 * 10 ** (-0.4 * 14.02)  # WISE W2 = 14.02 (Wright et al. 2014 via LE16 Table 3); ZP Jarrett et al. 2011


def template_bd(wl):
    wl = min(float(wl), L45)
    return float(np.exp(np.log(F36) + (np.log(F45) - np.log(F36)) * (wl - L36) / (L45 - L36)) / F45)


def restored_star_xy(st, xs, ys, star_xy, mjd, ra, dec, half=HALF):
    """Gaia stars that measure_frame dropped as 'coincident with the target' (< 0.7 px), added back to the scene.
    Same star selection as orrery.forced.measure_frame otherwise. Returns (star_xy, n_added)."""
    from astropy.io import fits
    from astropy.wcs import WCS
    sra, sdec, _ = _gaia_stars_local(ra, dec, (half + 4) * 6.2 / 3600 * 1.42, mjd)
    if not len(sra):
        return star_xy, 0
    gx, gy = WCS(fits.Header.fromstring(st.wcs_header)).all_world2pix(sra, sdec, 0)
    gx, gy = gx - st.x0, gy - st.y0
    add = [(float(a), float(b)) for a, b in zip(gx, gy) if np.hypot(a - xs, b - ys) <= 0.7]
    if not add:
        return star_xy, 0
    return np.vstack([np.asarray(star_xy, float).reshape(-1, 2), np.array(add)]), len(add)


def load_frames(df, restore=False):
    from orrery.data import load_stamp
    fr = []
    for r in df.itertuples():
        b = f"{SDIR}/{r.obs_id}_D{r.detector}"
        z = np.load(b + "_fit.npz")
        st = load_stamp(b + ".npz")
        sxy, n_add = z["star_xy"], 0
        if restore:
            sxy, n_add = restored_star_xy(st, float(z["xs"]), float(z["ys"]), sxy, r.mjd, r.pred_ra, r.pred_dec)
        fr.append(dict(row=r, st=st, xs=float(z["xs"]), ys=float(z["ys"]), star_xy=sxy, n_restored=n_add))
    return fr


def items_for(frames, offset=(0.0, 0.0), template=lambda wl: 1.0):
    from orrery.stack import scene_residual, amp_from_mJy
    items = []
    for f in frames:
        r, st = f["row"], f["st"]
        xs, ys = f["xs"] + offset[0], f["ys"] + offset[1]
        res, w, fit = scene_residual(st, xs, ys, f["star_xy"])
        p = amp_from_mJy(template(r.wl_um), st.omega_arcsec2)  # p = amplitude of the template flux (mJy at 4.49 um or flat 1 mJy)
        items.append(dict(res=res, w=w, xs=xs, ys=ys, p=p, A=fit.A, A_err=fit.A_err, mjd=r.mjd, wl=r.wl_um, obs_id=r.obs_id,
                          det=int(r.detector), key=r.key, snr=fit.A / fit.A_err if fit.A_err > 0 else np.nan,
                          f_mJy=r.flux_mJy, f_err=r.flux_err_mJy, epsf=st.epsf))
    return items


def stat(frames, template, offset=(0.0, 0.0)):
    from orrery.stack import combine
    it = items_for(frames, offset, template)
    img, snr_map, a, ae = combine(it, HALF)
    return dict(a=float(a), ae=float(ae), snr=float(a / ae), img=img, items=it)


def evaluate(frames, template):
    s = stat(frames, template)
    nulls = [stat(frames, template, o)["snr"] for o in NULLS]
    m, sd = float(np.mean(nulls)), float(np.std(nulls, ddof=1))
    inner, outer = np.array(nulls[:12]), np.array(nulls[12:])
    scale = float(np.std(inner, ddof=1))
    sig = (s["snr"] - m) / sd
    sig_cal = s["ae"] * max(sd, 1.0)            # empirically calibrated 1-sigma flux error (never below nominal)
    ul95 = max(s["a"] - m * s["ae"], 0.0) + 1.645 * sig_cal
    out = dict(n_frames=len(frames), a_hat_mJy=round(s["a"], 4), sigma_a_mJy=round(s["ae"], 4), snr_nominal=round(s["snr"], 3),
               null_snrs=[round(x, 3) for x in nulls], null_mean=round(m, 3), null_std=round(sd, 3), significance=round(sig, 3),
               sigma_a_calibrated_mJy=round(sig_cal, 4), upper_limit_95_mJy=round(ul95, 4),
               e5_check=dict(scale_inner=round(scale, 3), outer_spread_after_rescale=round(float(np.std(outer, ddof=1) / scale), 3)))
    return out, s


def stage_analyze():
    import matplotlib
    matplotlib.use("Agg")
    from build_reveal import detection_image, png
    df, n_rows = load_rows()
    counts = json.load(open(f"{FDIR}/_counts.json"))
    a = json.load(open(ASTRO))
    prim = df[(df.wl_um >= 4.4) & (df.wl_um <= 5.0)].sort_values("mjd").reset_index(drop=True)
    prim["pass"] = passes_of(prim.mjd.values) + 1
    sec = df[(df.wl_um >= 3.5) & (df.wl_um <= 5.0)].sort_values("mjd").reset_index(drop=True)
    sec["pass"] = passes_of(sec.mjd.values) + 1
    os.makedirs("site/data/e4", exist_ok=True)
    res = dict(passes=[], secondary=[])
    for k, g in prim.groupby("pass"):
        fr = load_frames(g)
        o, s = evaluate(fr, lambda wl: 1.0)
        o.update(pass_id=int(k), start=Time(g.mjd.min(), format="mjd").iso[:10], end=Time(g.mjd.max(), format="mjd").iso[:10],
                 wl_um=[round(float(x), 3) for x in g.wl_um], detectors=sorted(set(int(x) for x in g.detector)),
                 frames=[dict(obs_id=r.obs_id, detector=int(r.detector), mjd=round(r.mjd, 5), wl_um=round(r.wl_um, 4),
                              flux_mJy=round(r.flux_mJy, 4), flux_err_mJy=round(r.flux_err_mJy, 4), snr=round(r.snr, 2),
                              chi2_local_per_dof=round(r.chi2_local / max(r.dof_local, 1), 2), n_stars=int(r.n_stars), s3_key=r.key)
                         for r in g.itertuples()],
                 png=f"e4/stack_pass{k}.png", expected_snr_at_IRAC45_flux=round(F45 / o["sigma_a_calibrated_mJy"], 2))
        png(detection_image(s["img"]), f"site/data/e4/stack_pass{k}.png", -3, 6, cmap="gray", mark=(HALF, HALF))
        frr = load_frames(g, restore=True)
        orr, sr = evaluate(frr, lambda wl: 1.0)
        png(detection_image(sr["img"]), f"site/data/e4/stack_pass{k}_neighbour_restored.png", -3, 6, cmap="gray", mark=(HALF, HALF))
        o["neighbour_restored"] = dict(rule="deviation/diagnostic: Gaia DR3 stars within 0.7 px of the predicted position, which orrery.forced.measure_frame "
                                            "drops as 'coincident with the target', are put back into the static scene (fitted jointly)",
                                       n_frames_with_star_restored=int(sum(f["n_restored"] > 0 for f in frr)), png=f"e4/stack_pass{k}_neighbour_restored.png",
                                       **{x: orr[x] for x in ("a_hat_mJy", "sigma_a_mJy", "snr_nominal", "null_mean", "null_std", "significance",
                                                              "sigma_a_calibrated_mJy", "upper_limit_95_mJy", "null_snrs")})
        gq = g[g.chi2_local / g.dof_local.clip(lower=1) <= 3]  # repo data-quality rule (build_reveal); NOT the declared statistic
        if 0 < len(gq) < len(g):
            oq, _ = evaluate(load_frames(gq), lambda wl: 1.0)
            o["sensitivity_quality_cut"] = dict(rule="drop frames with local chi2/dof > 3 (scripts/build_reveal.py rule); not the declared primary statistic",
                                                n_dropped=int(len(g) - len(gq)), **{x: oq[x] for x in ("n_frames", "a_hat_mJy", "sigma_a_mJy", "snr_nominal",
                                                "null_mean", "null_std", "significance", "upper_limit_95_mJy")})
        res["passes"].append(o)
        print("primary pass", k, {x: o[x] for x in ("n_frames", "a_hat_mJy", "sigma_a_mJy", "snr_nominal", "null_mean", "null_std", "significance", "upper_limit_95_mJy")}, flush=True)
    for k, g in sec.groupby("pass"):
        o, s = evaluate(load_frames(g, restore=True), template_bd)
        o.update(pass_id=int(k), wl_range_um=[3.5, 5.0], a_hat_meaning="flux density at 4.49 um implied by the template (mJy)",
                 neighbours="restored (see passes[].neighbour_restored)")
        res["secondary"].append(o)
        print("secondary pass", k, {x: o[x] for x in ("n_frames", "a_hat_mJy", "sigma_a_mJy", "snr_nominal", "null_std", "significance")}, flush=True)
    # control: 0.75-3.5 um, where WISE 0855 is expected < 0.01 mJy (LE16 Table 3: J = 25.0, H > 22.7, Ks > 18.6)
    ctl = df[df.wl_um < 3.5].sort_values("mjd").reset_index(drop=True)
    ctl["pass"] = passes_of(ctl.mjd.values) + 1
    res["control_short_wavelength"] = []
    for k, g in ctl.groupby("pass"):
        c0, c1 = stat(load_frames(g), lambda wl: 1.0), stat(load_frames(g, restore=True), lambda wl: 1.0)
        res["control_short_wavelength"].append(dict(pass_id=int(k), n_frames=len(g), wl_range_um=[0.75, 3.5],
            declared=dict(a_hat_mJy=round(c0["a"], 4), sigma_a_mJy=round(c0["ae"], 4), snr_nominal=round(c0["snr"], 2)),
            neighbour_restored=dict(a_hat_mJy=round(c1["a"], 4), sigma_a_mJy=round(c1["ae"], 4), snr_nominal=round(c1["snr"], 2))))
        print("control", res["control_short_wavelength"][-1], flush=True)
    # all passes together (not the declared statistic; context only)
    o, s = evaluate(load_frames(prim), lambda wl: 1.0)
    png(detection_image(s["img"]), "site/data/e4/stack_all.png", -3, 6, cmap="gray", mark=(HALF, HALF))
    res["all_passes_flat"] = o
    o, s = evaluate(load_frames(prim, restore=True), lambda wl: 1.0)
    png(detection_image(s["img"]), "site/data/e4/stack_all_neighbour_restored.png", -3, 6, cmap="gray", mark=(HALF, HALF))
    res["all_passes_flat_neighbour_restored"] = o
    print("all passes flat", {x: o[x] for x in ("n_frames", "a_hat_mJy", "sigma_a_mJy", "snr_nominal", "null_std", "significance")}, flush=True)
    res.update(counts=dict(counts, n_frames_measured_on_frame=int(len(df)), n_frames_attempted=n_rows,
                           n_frames_primary_4p4_5p0um=int(len(prim)), n_frames_secondary_3p5_5p0um=int(len(sec))))
    res["astrometry"] = {k: a[k] for k in a if k != "residuals_all_epochs"}
    return res


def jacobian(st, xs, ys):
    """Local d(RA*cosDec, Dec)/d(x, y) in arcsec per stamp pixel."""
    from astropy.io import fits
    from astropy.wcs import WCS
    w = WCS(fits.Header.fromstring(st.wcs_header))
    x, y = xs + st.x0, ys + st.y0
    r, d = w.all_pix2world([[x, y], [x + 1, y], [x, y + 1]], 0).T
    return np.array([[(r[1] - r[0]) * np.cos(np.radians(d[0])) * 3600, (r[2] - r[0]) * np.cos(np.radians(d[0])) * 3600],
                     [(d[1] - d[0]) * 3600, (d[2] - d[0]) * 3600]])


def centroid(items):
    """refine_position on the stacked residual (stars removed; per-pixel variance 1/Σ w p²; mean ePSF of the frames)."""
    from orrery.fit import refine_position
    from orrery.stack import centre
    num = np.zeros((2 * HALF + 1,) * 2)
    den = np.zeros_like(num)
    for it in items:
        if not (np.isfinite(it["A"]) and it["A_err"] > 0):
            continue
        w = centre(it["w"], it["xs"], it["ys"], HALF).clip(0)
        num += w * it["p"] * centre(it["res"], it["xs"], it["ys"], HALF)
        den += w * it["p"] ** 2
    ok = den > 0
    img = np.where(ok, num / np.where(ok, den, 1), 0.0)
    var = np.where(ok, 1 / np.where(ok, den, 1), np.inf)
    epsf = np.mean([it["epsf"] for it in items], axis=0)
    xb, yb, surf, offs = refine_position(img, var, np.zeros(img.shape, np.int32), epsf, HALF, HALF, np.zeros((0, 2)), 1, 5, half=1.0, step=0.05)
    return xb - HALF, yb - HALF


def centroid_pass(frames, n_boot=40, seed=20261115):
    s = stat(frames, lambda wl: 1.0)
    dx, dy = centroid(s["items"])
    rng = np.random.default_rng(seed)
    boot = [centroid([s["items"][i] for i in rng.integers(0, len(frames), len(frames))]) for _ in range(n_boot)]
    J = np.mean([jacobian(f["st"], f["xs"], f["ys"]) for f in frames], axis=0)
    w = np.array([1 / f["row"].flux_err_mJy ** 2 for f in frames])
    pra = np.sum(w * np.array([f["row"].pred_ra for f in frames])) / w.sum()
    pdec = np.sum(w * np.array([f["row"].pred_dec for f in frames])) / w.sum()
    off = J @ np.array([dx, dy])                              # arcsec (RA*cosDec, Dec)
    boff = np.array([J @ np.array(b) for b in boot])
    return dict(offset_px=[round(dx, 2), round(dy, 2)], offset_arcsec=[round(float(off[0]), 3), round(float(off[1]), 3)],
                offset_err_arcsec_bootstrap=[round(float(x), 3) for x in boff.std(0, ddof=1)],
                pred_ra_wmean=pra, pred_dec_wmean=pdec, mjd_wmean=float(np.sum(w * [f["row"].mjd for f in frames]) / w.sum()),
                meas_ra=pra + off[0] / 3600 / np.cos(np.radians(pdec)), meas_dec=pdec + off[1] / 3600, n_boot=n_boot)


SLIDER_PASS = 2  # the slider shows one visit window only: Nov-Dec 2025, the window behind the quoted significance


def build_slider(frames):
    """Reveal-style frame-count slider data (same format as site/data/reveal.json) for ONE visit window, frames as
    declared (no quality cut, no neighbour restoration), so the last step is that window's declared statistic.
    Order: expected S/N (flat template / noise); measured fluxes never choose the order. Every step n is calibrated on
    its own 24 empty paths (the same n pictures, same order, path displaced): (S/N - null mean) / max(null spread, 1).
    Returns (frames list, stacks list, per-window summary dict)."""
    from build_reveal import detection_image, png
    from orrery.stack import combine, centre
    out = "site/data/e4"
    on = stat(frames, lambda wl: 1.0)["items"]
    order = sorted(range(len(on)), key=lambda i: -on[i]["p"] / on[i]["A_err"])  # expected S/N order
    items = [on[i] for i in order]
    null_items = [[it[i] for i in order] for it in (items_for(frames, o) for o in NULLS)]
    single_std = max(float(np.std([i["snr"] for it in null_items for i in it if np.isfinite(i["snr"])])), 1.0)
    N = len(items)
    frs, stacks = [], []
    for k, it in enumerate(items):
        z = detection_image(centre(np.where(it["w"] > 0, it["res"], 0), it["xs"], it["ys"], HALF))
        png(z, f"{out}/single_{k:03d}.png", -3, 6, cmap="gray", mark=(HALF, HALF))
        frs.append(dict(obs_id=it["obs_id"], detector=it["det"], mjd=it["mjd"], wl_um=round(it["wl"], 4), single_png=f"e4/single_{k:03d}.png",
                        snr=round(float(it["snr"] / single_std), 2), snr_nominal=round(float(it["snr"]), 2), s3_key=it["key"]))
    for n in [n for n in sorted(set([1, 2, 3, 5, 8, 12, 18, N])) if n <= N]:
        img, _, a_, ae = combine(items[:n], HALF)
        nulls = []
        for ni in null_items:
            _, _, na, ne = combine(ni[:n], HALF)
            nulls.append(float(na / ne))
        m, sd = float(np.mean(nulls)), float(np.std(nulls, ddof=1))
        png(detection_image(img), f"{out}/stack_{n:03d}.png", -3, 6, cmap="gray", mark=(HALF, HALF))
        stacks.append(dict(n=n, png=f"e4/stack_{n:03d}.png", snr_nominal=round(float(a_ / ae), 2),
                           snr=round(float((a_ / ae - m) / max(sd, 1.0)), 2), null_mean=round(m, 3), null_std=round(sd, 3),
                           a_hat=round(float(a_), 3), a_err=round(float(ae), 3)))
    stacks[0]["snr"] = frs[0]["snr"]  # one picture: show the single-picture value, so the two numbers agree
    last = stacks[-1]
    summary = dict(n_frames=N, snr_nominal=last["snr_nominal"], null_mean=last["null_mean"], null_std=last["null_std"],
                   significance=last["snr"], single_std=round(single_std, 3), null_snrs=[round(x, 3) for x in nulls])
    return frs, stacks, summary


def stage_slider(prim, res):
    """Rebuild site/data/e4/reveal.json from the SLIDER_PASS window and record it in res['slider_stack']."""
    g = prim[prim["pass"] == SLIDER_PASS]
    p = next(x for x in res["passes"] if x["pass_id"] == SLIDER_PASS)
    for f in glob.glob("site/data/e4/single_*.png") + glob.glob("site/data/e4/stack_[0-9]*.png"):
        os.remove(f)  # stale slider pictures from an earlier build
    frs, stacks, sm = build_slider(load_frames(g))
    assert abs(sm["significance"] - p["significance"]) < 0.01, (sm["significance"], p["significance"])  # last step = declared statistic
    window = f"{p['start']} to {p['end']}"
    res["slider_stack"] = dict(frames=f"4.4-5.0 um, visit window {SLIDER_PASS} ({window}) only, frames as declared; every step "
                                      "calibrated on its own 24 empty paths, (S/N - null mean)/max(null spread, 1)",
                               pass_id=SLIDER_PASS, **{x: sm[x] for x in ("n_frames", "snr_nominal", "null_mean", "null_std", "significance", "single_std")},
                               a_hat_mJy=p["a_hat_mJy"], sigma_a_mJy=p["sigma_a_mJy"])
    cal = [round((x - sm["null_mean"]) / max(sm["null_std"], 1.0), 2) for x in sm["null_snrs"]]
    rev = dict(object="WISE 0855-0714", designation="WISE J085510.83-071442.5", n_frames_before_quality_cut=int(len(g)), meta=dict(name="WISE 0855-0714"),
               window=dict(pass_id=SLIDER_PASS, start=p["start"], end=p["end"], label="Nov–Dec 2025 visits"),
               frames=frs, stack_pngs=stacks, spectrum_points=[],
               noise_calibration=dict(stack_null_mean=sm["null_mean"], stack_null_std=sm["null_std"], single_std=sm["single_std"], n_null_tracks=24,
                                      per_step="each step: the same n pictures combined along 24 empty paths; (S/N - null mean)/max(null spread, 1)",
                                      singles="single pictures: S/N divided by the spread of single-picture S/N on the empty paths (floored at 1); no mean correction"),
               stack_snr=stacks[-1]["snr"], null_snrs=cal, null_snrs_nominal=sm["null_snrs"], n_frames=len(frs),
               hedge="Nov–Dec 2025 visits only; each step against the same pictures combined along 24 empty paths",
               order="pictures are combined in order of their expected signal-to-noise (flat template / noise; measured fluxes are not used to choose the order)",
               units="singles and stacks: Gaia stars removed, smoothed at the PSF scale, in units of each image's own background noise; frames 4.4-5.0 um, "
                     "Nov-Dec 2025 visits only, each centred on its predicted position (proper motion + parallax)",
               rerun="OMP_NUM_THREADS=1 .venv/bin/python scripts/run_e4.py forced && .venv/bin/python scripts/run_e4.py analyze  (slider only: run_e4.py slider)")
    json.dump(rev, open("site/data/e4/reveal.json", "w"), indent=1, default=float)
    print("slider stack", res["slider_stack"], flush=True)
    print("steps", [(s_["n"], s_["snr"]) for s_ in stacks], flush=True)


def slider_only():
    """Rebuild only the slider from the stamps on disk; data/findings/e4.json keeps every other key unchanged."""
    res = json.load(open("data/findings/e4.json"))
    df, _ = load_rows()
    prim = df[(df.wl_um >= 4.4) & (df.wl_um <= 5.0)].sort_values("mjd").reset_index(drop=True)
    prim["pass"] = passes_of(prim.mjd.values) + 1
    old = res.get("slider_stack", {})
    if old.get("pass_id") != SLIDER_PASS and "slider_stack_all_passes_superseded" not in res:
        res["slider_stack_all_passes_superseded"] = dict(old, note="no longer shown on the site: an all-window stack chosen after the fact "
                                                         "(quality cut, neighbour restored), not a declared statistic")
    stage_slider(prim, res)
    json.dump(res, open("data/findings/e4.json", "w"), indent=1, default=float)


NOTES = dict(
    statement="Candidate-level statement about a known object (not a discovery): a point source is measured at the predicted "
              "proper-motion + parallax position of WISE 0855-0714 in SPHEREx DR1 4.4-5.0 um frames.",
    citation="Luhman, K. L. & Esplin, T. L. 2016, AJ 152, 78 (arXiv:1605.06655): parallax 0.449+-0.008 arcsec, "
             "pm (RA*cosDec, Dec) = (-8.118+-0.008, +0.680+-0.007) arcsec/yr (Table 4); positions from Table 2.",
    expected_flux_mJy=dict(IRAC_4p5=round(F45, 3), IRAC_3p6=round(F36, 4), WISE_W2=round(W2_MJY, 3),
                           source="LE16 Table 3 Vega mags ([4.5] mean 13.87, [3.6] mean 17.34, W2 14.02); zero points 179.7/280.9 Jy "
                                  "(Reach et al. 2005) and 171.787 Jy (Jarrett et al. 2011). Not used in the primary statistic."),
    primary_definition="flat template (1 mJy in every frame), frames with central wavelength 4.4-5.0 um at the target, passes split at "
                       "gaps > 30 d; a_hat = sum(p A/s^2)/sum(p^2/s^2), p = amplitude of 1 mJy (orrery.stack.combine); nulls = same frames, "
                       "track displaced by 6 px (angles 0,30,..,330 deg) and 9 px (15,45,..,345 deg) in stamp pixel axes, static scene refitted "
                       "there; significance = (a_hat/sigma - null mean)/null std (ddof=1).",
    secondary_definition="SECONDARY (not the declared statistic): cold-brown-dwarf template p(lambda) = log-linear interpolation of the "
                         "LE16 Spitzer [3.6]/[4.5] fluxes (ratio ~16 between 3.55 and 4.49 um), constant beyond 4.49 um (assumption); "
                         "frames 3.5-5.0 um; neighbours restored.",
    deviations=["Neighbour confusion: a Gaia DR3 star (G = 16.31, RA 133.758183, Dec -7.241156 at J2016) lies 0.67-0.71 px from the "
                "predicted position in pass 3 (1.2-1.4 px in pass 2, 1.9 px in pass 1). orrery.forced.measure_frame drops Gaia stars "
                "< 0.7 px from the target as 'coincident', so in 31/100 pass-3 frames (9/21 primary frames) the star was not modelled "
                "and its flux entered the target amplitude. The short-wavelength control (0.75-3.5 um, WISE 0855 expected < 0.01 mJy) "
                "shows this (control_short_wavelength). passes[].significance is the declared statistic as frozen; "
                "passes[].neighbour_restored puts the star back (diagnostic). Centroids use neighbour_restored; the slider shows pass 2 only, as declared.",
                "Pass 1 (2025-05-05/06) has 3 primary frames; one (2025W19_1B_0138_1 D6) has a badly fitted background "
                "(local chi2/dof = 41, residuals 5-14 sigma across the stamp), which also inflates the null spread (std 57). "
                "The declared statistic for pass 1 is therefore uninformative; passes[0].sensitivity_quality_cut drops that frame.",
                "Upper limits are reported per pass for completeness although the decision branch is 'detected'.",
                "Astrometric reference position fitted by us to LE16 Table 2 HST epochs (LE16 gives no reference position)."],
    caveats=["Centroids: refine_position (0.05 px grid, +-1 px) on the per-pass stacked residual; frame orientation is constant to "
             "< 1.5 deg within a pass, so the pass-mean pixel-to-sky Jacobian is used; errors from 40 bootstrap resamples of frames. "
             "The G = 16.3 neighbour at 0.7-1.4 px can bias centroids even when modelled.",
             "SPHEREx band edges: 4.4-5.0 um uses detector-6 frames plus detector-5 frames at its long-wavelength edge (4.40-4.42 um); the measured flux is a flat-weighted mean over "
             "the channels sampled, not a synthetic IRAC/W2 magnitude."],
)


def main():
    res = stage_analyze()
    df, _ = load_rows()
    prim = df[(df.wl_um >= 4.4) & (df.wl_um <= 5.0)].sort_values("mjd").reset_index(drop=True)
    prim["pass"] = passes_of(prim.mjd.values) + 1
    det = [p for p in res["passes"] if p["significance"] >= 5]
    if det:
        res["decision"] = "detected: pass(es) %s at >= 5 vs nulls (primary statistic)" % [p["pass_id"] for p in det]
        from scipy.stats import chi2 as chi2d
        cen = {}
        for p in det:  # centroids only for passes detected at >= 5 vs nulls
            g = prim[prim["pass"] == p["pass_id"]]
            if len(g) >= 3:
                cen[p["pass_id"]] = c = centroid_pass(load_frames(g, restore=True))
                p["centroid"] = c
                print("centroid pass", p["pass_id"], c, flush=True)
        ids = [p["pass_id"] for p in det]
        comps = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a_, b_ = cen[ids[i]], cen[ids[j]]
                cd = np.cos(np.radians(a_["pred_dec_wmean"]))
                pred = [(b_["pred_ra_wmean"] - a_["pred_ra_wmean"]) * 3600 * cd, (b_["pred_dec_wmean"] - a_["pred_dec_wmean"]) * 3600]
                meas = [(b_["meas_ra"] - a_["meas_ra"]) * 3600 * cd, (b_["meas_dec"] - a_["meas_dec"]) * 3600]
                err = np.hypot(a_["offset_err_arcsec_bootstrap"], b_["offset_err_arcsec_bootstrap"])
                diff = np.subtract(meas, pred)
                comps.append(dict(passes=[ids[i], ids[j]], predicted_shift_arcsec=[round(x, 2) for x in pred],
                                  measured_shift_arcsec=[round(x, 2) for x in meas], measured_minus_predicted_arcsec=[round(x, 2) for x in diff],
                                  err_arcsec=[round(float(x), 2) for x in err], chi2_vs_prediction=round(float(np.sum((diff / err) ** 2)), 2),
                                  chi2_vs_stationary=round(float(np.sum((np.array(meas) / err) ** 2)), 2), dof=2))
                comps[-1].update(p_vs_prediction=float(chi2d.sf(comps[-1]["chi2_vs_prediction"], 2)),
                                 p_vs_stationary=float(chi2d.sf(comps[-1]["chi2_vs_stationary"], 2)))
                print("shift", comps[-1], flush=True)
        res["centroid_shift_vs_prediction"] = comps
        stage_slider(prim, res)
    else:
        res["decision"] = "not detected at >= 5 vs nulls in any pass (primary statistic); 95% upper limits given per pass"
    res["upper_limit_definition"] = "UL95 = max(a_hat - null_mean*sigma_a, 0) + 1.645*sigma_a*max(null_std, 1)  [mJy, flat in 4.4-5.0 um]"
    res["index"] = INDEX
    res.update(NOTES)
    json.dump(res, open("data/findings/e4.json", "w"), indent=1, default=float)
    print(res["decision"], flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["astrometry", "forced", "analyze", "slider"])
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    if a.stage == "astrometry":
        stage_astrometry()
    elif a.stage == "forced":
        stage_forced(min(a.workers, 3))
    elif a.stage == "slider":
        slider_only()
    else:
        main()
