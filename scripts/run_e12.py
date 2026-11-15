"""E12 — distance to Epsilon Indi B from SPHEREx positions alone (docs/PROTOCOLS.md PHASE 2; choices: docs/protocols/E12.md).

Stages (run in this order; `fit` only after docs/protocols/E12.md is frozen):
  select   null-star list from Gaia DR3 (rule + seed in E12.md)           -> data/phase2/e12/null_stars.json
  check    the stamp adapter reproduces the stored target positions        -> data/phase2/e12/adapter_check.json
  measure  per-picture positions of the null stars, same machinery         -> data/phase2/e12/null_<source_id>.parquet
  fit      5-parameter fit (target), bootstrap, null fits, verdict, figure -> data/findings/e12.json, site/data/figs/e12_parallax.png

Position machinery: scripts/measure_motion.py:one (unchanged; free-position ePSF fit, Gaia neighbours fitted). The
only difference for the null stars is how pixels are read: scripts/e2_cache.py:fast_stamp (documented bit-identical to
Frame.stamp) instead of Frame, and one Gaia table per star (async TAP; the sync endpoint timed out on 2026-11-15).
Parallax factors: Earth's barycentric position from astropy get_body_barycentric (built-in ephemeris); SPHEREx is in
low Earth orbit, <= 7,000 km from the geocentre = 5e-5 au, i.e. <= 0.013 mas of parallax for this star.
  OMP_NUM_THREADS=1 .venv/bin/python scripts/run_e12.py <stage>
"""
import json, os, sys, time, warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
warnings.filterwarnings("ignore")
from astropy import log  # noqa: E402
log.setLevel("ERROR")
import astropy.units as u  # noqa: E402
from astropy.coordinates import get_body_barycentric  # noqa: E402
from astropy.io import fits  # noqa: E402
from astropy.time import Time  # noqa: E402
from astropy.wcs import WCS  # noqa: E402

OUT = "data/phase2/e12"
TARGET_PARQUET = "data/findings/motion_epsindib.parquet"
GAIA_TARGET = dict(source_id="6412596012146801152", ra=331.07645259656397, dec=-56.79381206688221,
                   pmra=3981.97666296105, pmdec=-2466.8318147663504, parallax=270.6580324694526,
                   parallax_error=0.6897)  # Gaia DR3 (Vallenari et al. 2023), values as stored by measure_motion.py
FIELD = (331.0962, -56.8006)
SEED, N_NULL, N_BOOT = 12, 8, 1000
EPS_INDI_A_2026 = (330.8924, -56.8043)   # Gaia DR3 6412595290592307840 propagated to 2026.0
TAP = "https://gaia.aip.de/tap"


# ------------------------------------------------------------------ null-star selection
def select():
    import pyvo
    path = f"{OUT}/gaia_cone_0p30_G16p5.parquet"
    if not os.path.exists(path):
        q = ("SELECT source_id, ra, dec, pmra, pmdec, parallax, parallax_error, ruwe, phot_g_mean_mag, bp_rp FROM "
             "gaiadr3.gaia_source WHERE 1=CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',331.0962,-56.8006,0.30)) "
             "AND phot_g_mean_mag<16.5")
        t = pyvo.dal.TAPService(TAP).run_async(q).to_table().to_pandas()
        t["source_id"] = t["source_id"].astype("int64")
        t.to_parquet(path)
    t = pd.read_parquet(path)
    cd = np.cos(np.radians(FIELD[1]))

    def sep(ra, dec, r2, d2):
        return 3600 * np.hypot((ra - r2) * cd, dec - d2)
    ok = (t.phot_g_mean_mag >= 10.5) & (t.phot_g_mean_mag <= 13.0) & (t.ruwe < 1.4) & t.parallax.notna() & t.pmra.notna()
    ok &= sep(t.ra, t.dec, *EPS_INDI_A_2026) > 300
    for i in t.index[ok]:
        s = sep(t.ra, t.dec, t.ra[i], t.dec[i])
        if ((s > 0) & (s < 25) & (t.phot_g_mean_mag < t.phot_g_mean_mag[i] + 3)).any():
            ok[i] = False
        if ((s > 0) & (s < 120) & (t.phot_g_mean_mag < 9)).any():
            ok[i] = False
    el = t[ok].sort_values("source_id").reset_index(drop=True)
    pick = np.sort(np.random.default_rng(SEED).choice(len(el), min(N_NULL, len(el)), replace=False))
    stars = el.iloc[pick]
    rec = dict(n_cone=int(len(t)), n_eligible=int(len(el)), seed=SEED,
               stars=[dict(source_id=str(r.source_id), ra=r.ra, dec=r.dec, pmra=r.pmra, pmdec=r.pmdec, parallax=r.parallax,
                           parallax_error=r.parallax_error, G=r.phot_g_mean_mag, bp_rp=r.bp_rp, ruwe=r.ruwe,
                           sep_arcmin=float(sep(r.ra, r.dec, *FIELD) / 60)) for r in stars.itertuples()])
    json.dump(rec, open(f"{OUT}/null_stars.json", "w"), indent=1)
    print(json.dumps(rec, indent=1))


# ------------------------------------------------------------------ measurement (existing machinery, fast pixel reads)
class StampFrame:
    """Frame-like wrapper around one cut-out, enough for measure_motion.one."""

    def __init__(self, st):
        self.st, self.mjd, self.detector = st, st.mjd, st.detector
        self.wcs = WCS(fits.Header.fromstring(st.wcs_header))

    def world2pix(self, ra, dec):
        return self.wcs.all_world2pix(np.atleast_1d(ra), np.atleast_1d(dec), 0)

    def stamp(self, x, y, half=15):
        return self.st if self.st.image.shape == (2 * half + 1, 2 * half + 1) else None


_tab = {}


def _star_table(sid, ra, dec):
    """Gaia DR3 G < 19 within 0.085 deg of the star (same columns as orrery.forced.gaia_stars), one query per star."""
    p = f"{OUT}/gaia/{sid}.parquet"
    if sid not in _tab:
        if not os.path.exists(p):
            import pyvo
            os.makedirs(f"{OUT}/gaia", exist_ok=True)
            q = ("SELECT ra, dec, pmra, pmdec, phot_g_mean_mag FROM gaiadr3.gaia_source WHERE "
                 f"1=CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',{ra},{dec},0.085)) AND phot_g_mean_mag<19.0")
            pyvo.dal.TAPService(TAP).run_async(q).to_table().to_pandas().to_parquet(p)
        _tab[sid] = pd.read_parquet(p)
    return _tab[sid]


def measure_one(a):
    key, sid, g, mjd = a
    import measure_motion as mm
    import e2_cache
    try:
        ra, dec = mm.predict(g, np.array([mjd]))
        ra, dec = float(ra[0]), float(dec[0])
        st = e2_cache.fast_stamp(key, ra, dec, half=15)
        if st is None:
            return None
        if g.get("own_table"):
            df = _star_table(sid, g["ra"], g["dec"])

            def gs(ra_, dec_, radius, mjd, gmax=19.0):   # as orrery.forced.gaia_stars, from the per-star table
                dt = Time(mjd, format="mjd").jyear - 2016.0
                r = df["ra"].values + df["pmra"].fillna(0).values * dt / 3.6e6 / np.cos(np.radians(df["dec"].values))
                d = df["dec"].values + df["pmdec"].fillna(0).values * dt / 3.6e6
                k = (np.hypot((r - ra_) * np.cos(np.radians(dec_)), d - dec_) < radius) & (df["phot_g_mean_mag"].values < gmax)
                return r[k], d[k], df["phot_g_mean_mag"].values[k]
            mm.gaia_stars = gs
        mm.Frame = lambda k: StampFrame(st)
        r = mm.one((key, ra, dec, 0))
        if r is not None:
            r["source_id"] = sid
        return r
    except Exception as e:
        return dict(key=key, source_id=sid, error=repr(e))


def measure(workers=5):
    tp = pd.read_parquet(TARGET_PARQUET)
    stars = json.load(open(f"{OUT}/null_stars.json"))["stars"]
    for s in stars:
        _star_table(s["source_id"], s["ra"], s["dec"])     # one async query per star, before the workers start
        p = f"{OUT}/null_{s['source_id']}.parquet"
        if os.path.exists(p):
            continue
        g = dict(s, own_table=True)
        t = time.time()
        with ProcessPoolExecutor(workers) as ex:
            rows = [r for r in ex.map(measure_one, [(k, s["source_id"], g, m) for k, m in zip(tp.key, tp.mjd)]) if r]
        pd.DataFrame(rows).to_parquet(p)
        print(s["source_id"], len(rows), f"{time.time() - t:.0f}s", flush=True)


def check(n=4):
    """Adapter check: re-measure the target in n stored pictures; positions must equal the stored ones."""
    tp = pd.read_parquet(TARGET_PARQUET).dropna(subset=["ra_meas"])
    tp = tp.iloc[np.linspace(0, len(tp) - 1, n).round().astype(int)]
    out = []
    for r in tp.itertuples():
        g = dict(ra=r.ra_pred, dec=r.dec_pred, pmra=0.0, pmdec=0.0, parallax=0.0)   # stored prediction, used as is
        m = measure_one((r.key, "target", g, r.mjd))
        out.append(dict(key=r.key, d_ra_mas=(m["ra_meas"] - r.ra_meas) * 3.6e6, d_dec_mas=(m["dec_meas"] - r.dec_meas) * 3.6e6,
                        snr_new=m["snr"], snr_old=r.snr))
        print(out[-1], flush=True)
    json.dump(out, open(f"{OUT}/adapter_check.json", "w"), indent=1)


# ------------------------------------------------------------------ the 5-parameter fit
def windows(mjd, gap=30.0):
    o = np.argsort(mjd)
    lab = np.zeros(len(mjd), int)
    lab[o] = np.concatenate([[0], np.cumsum(np.diff(np.asarray(mjd)[o]) > gap)])
    return lab


def design(mjd, ra0, dec0, t0, with_plx=True):
    t = Time(np.asarray(mjd), format="mjd")
    dt = t.jyear - t0
    X, Y, Z = get_body_barycentric("earth", t).xyz.to(u.au).value
    a, d = np.radians(ra0), np.radians(dec0)
    pa = X * np.sin(a) - Y * np.cos(a)                                   # parallax factor, RA*cos(Dec)
    pd_ = X * np.cos(a) * np.sin(d) + Y * np.sin(a) * np.sin(d) - Z * np.cos(d)  # parallax factor, Dec
    o, z = np.ones_like(dt), np.zeros_like(dt)
    A = np.block([[o[:, None], z[:, None], dt[:, None], z[:, None], pa[:, None]],
                  [z[:, None], o[:, None], z[:, None], dt[:, None], pd_[:, None]]])
    return (A if with_plx else A[:, :4]), pa, pd_


def wls(A, y, sig):
    Aw, yw = A / sig[:, None], y / sig
    cov = np.linalg.inv(Aw.T @ Aw)
    p = cov @ (Aw.T @ yw)
    r = y - A @ p
    return p, cov, r, float(np.sum((r / sig) ** 2))


def fit5(df, n_boot=0, seed=SEED, sig_floor=0.0):
    """Frozen procedure (E12.md): selection, pass 1 (formal weights), 5-sigma clip per window, pass 2 (per-window
    empirical scatter as the per-picture error), covariance scaled by reduced chi2 if > 1, optional bootstrap."""
    d = df.dropna(subset=["ra_meas"])
    d = d[(d.snr > 20) & np.isfinite(d.err_arcsec)].sort_values("mjd").reset_index(drop=True)
    ra0, dec0 = float(np.median(d.ra_meas)), float(np.median(d.dec_meas))
    t0 = float(np.mean(Time(d.mjd.values, format="mjd").jyear))
    x = (d.ra_meas.values - ra0) * 3600 * np.cos(np.radians(dec0))
    y = (d.dec_meas.values - dec0) * 3600
    n = len(d)
    A, pa, pd_ = design(d.mjd.values, ra0, dec0, t0)
    Y = np.concatenate([x, y])
    p1, _, r1, _ = wls(A, Y, np.concatenate([d.err_arcsec.values] * 2))
    w = windows(d.mjd.values)
    keep = np.ones(n, bool)
    for k in np.unique(w):
        m = w == k
        for r in (r1[:n], r1[n:]):
            med = np.median(r[m])
            s = 1.4826 * np.median(np.abs(r[m] - med))
            keep[m] &= np.abs(r[m] - med) <= 5 * max(s, sig_floor, 1e-6)
    sig = np.zeros(2 * n)
    for k in np.unique(w):
        m = (w == k) & keep
        sig[:n][w == k] = np.std(r1[:n][m], ddof=1) if m.sum() > 2 else np.std(r1[:n][keep], ddof=1)
        sig[n:][w == k] = np.std(r1[n:][m], ddof=1) if m.sum() > 2 else np.std(r1[n:][keep], ddof=1)
    k2 = np.concatenate([keep, keep])
    p, cov, r, chi2 = wls(A[k2], Y[k2], sig[k2])
    dof = int(k2.sum() - 5)
    s2 = max(1.0, chi2 / dof)
    cov = cov * s2
    err = np.sqrt(np.diag(cov))
    A4 = A[k2][:, :4]
    p4, _, r4, chi2_4 = wls(A4, Y[k2], sig[k2])
    res = dict(n_selected=int(n), n_clipped=int((~keep).sum()), n_used=int(keep.sum()), ref_ra=ra0, ref_dec=dec0, ref_epoch_jyear=t0,
               params=dict(zip(["dra0_arcsec", "ddec0_arcsec", "pmra_arcsec_yr", "pmdec_arcsec_yr", "parallax_arcsec"], map(float, p))),
               errors=dict(zip(["dra0_arcsec", "ddec0_arcsec", "pmra_arcsec_yr", "pmdec_arcsec_yr", "parallax_arcsec"], map(float, err))),
               correlation=(cov / np.outer(err, err)).round(4).tolist(), chi2=chi2, dof=dof, chi2_red=chi2 / dof,
               chi2_no_parallax=chi2_4, params_no_parallax=list(map(float, p4)))
    wins = []
    idx = np.where(keep)[0]
    for k in np.unique(w):
        m = (w == k) & keep
        mm_ = np.concatenate([m, m])
        nk = int(m.sum())
        wins.append(dict(window=int(k), n=nk, mjd_mean=float(d.mjd.values[m].mean()), date=Time(d.mjd.values[m].mean(), format="mjd").iso[:10],
                         x_arcsec=float(x[m].mean()), y_arcsec=float(y[m].mean()),
                         x_err=float(sig[:n][m][0] / np.sqrt(nk)), y_err=float(sig[n:][m][0] / np.sqrt(nk)),
                         scatter_x=float(sig[:n][m][0]), scatter_y=float(sig[n:][m][0]),
                         plx_factor_ra=float(pa[m].mean()), plx_factor_dec=float(pd_[m].mean()),
                         resid5_x=float((Y - A @ p)[:n][m].mean()), resid5_y=float((Y - A @ p)[n:][m].mean()),
                         resid4_x=float((Y[:n] - A[:n, :4] @ p4)[m].mean()), resid4_y=float((Y[n:] - A[n:, :4] @ p4)[m].mean())))
    res["windows"] = wins
    res["chi2_window_means_5par"] = float(sum((v["resid5_x"] / v["x_err"]) ** 2 + (v["resid5_y"] / v["y_err"]) ** 2 for v in wins))
    res["chi2_window_means_no_parallax"] = float(sum((v["resid4_x"] / v["x_err"]) ** 2 + (v["resid4_y"] / v["y_err"]) ** 2 for v in wins))
    if n_boot:
        rng = np.random.default_rng(seed)
        bs = []
        while len(bs) < n_boot:
            j = rng.choice(idx, len(idx), replace=True)
            if min(np.bincount(w[j], minlength=w.max() + 1)) < 2:
                continue
            jj = np.concatenate([j, j + n])
            bs.append(wls(A[jj], Y[jj], sig[jj])[0])
        bs = np.array(bs)
        res["bootstrap"] = dict(n=n_boot, seed=seed, parallax_std=float(bs[:, 4].std(ddof=1)),
                                parallax_p16_p50_p84=[float(v) for v in np.percentile(bs[:, 4], [16, 50, 84])],
                                parallax_p2p5_p97p5=[float(v) for v in np.percentile(bs[:, 4], [2.5, 97.5])],
                                pm_std=[float(bs[:, 2].std(ddof=1)), float(bs[:, 3].std(ddof=1))])
    aux = dict(d=d, x=x, y=y, keep=keep, w=w, A=A, p=p, p4=p4, sig=sig, ra0=ra0, dec0=dec0, t0=t0)
    return res, aux


GRID_FLOOR = 0.155     # arcsec: half of the 0.05 px position grid of scripts/measure_motion.py (deviation 1, E12.md)
SYSTEM_PARALLAX = (274.8431, 0.0956)   # Gaia DR3, Epsilon Indi A (the system's distance; Ba's own Gaia value is orbit-affected)


def analyse(sig_floor):
    """Target fit + null-star fits + adopted error + verdict numbers for one clipping rule."""
    stars = json.load(open(f"{OUT}/null_stars.json"))["stars"]
    tp = pd.read_parquet(TARGET_PARQUET)
    t_ok = tp.dropna(subset=["ra_meas"])
    t_ok = t_ok[(t_ok.snr > 20) & np.isfinite(t_ok.err_arcsec)].sort_values("mjd")
    tw = windows(t_ok.mjd.values)
    lo = [t_ok.mjd.values[tw == k].min() - 5 for k in np.unique(tw)]
    hi = [t_ok.mjd.values[tw == k].max() + 5 for k in np.unique(tw)]
    null = []
    for s in stars:
        p = f"{OUT}/null_{s['source_id']}.parquet"
        df = pd.read_parquet(p) if os.path.exists(p) else pd.DataFrame()
        if "ra_meas" not in df:
            null.append(dict(source_id=s["source_id"], G=s["G"], used=False, reason="no measurements"))
            continue
        sel = df.dropna(subset=["ra_meas"])
        sel = sel[(sel.snr > 20) & np.isfinite(sel.err_arcsec)]
        per = [int(((sel.mjd > a) & (sel.mjd < b)).sum()) for a, b in zip(lo, hi)]
        rec = dict(source_id=s["source_id"], G=s["G"], sep_arcmin=s["sep_arcmin"], n_measured=int(len(df)), n_selected=int(len(sel)),
                   median_snr=float(sel.snr.median()) if len(sel) else None, gaia_parallax_mas=s["parallax"],
                   gaia_pmra_mas_yr=s["pmra"], gaia_pmdec_mas_yr=s["pmdec"], n_per_target_window=per)
        if len(sel) < 60 or min(per) < 5:      # eligibility: >= 60 selected pictures and >= 5 in each of the target's windows
            null.append(dict(rec, used=False, reason="too few selected pictures"))
            continue
        r, _ = fit5(df, sig_floor=sig_floor)
        rec.update(used=True, parallax_mas=1000 * r["params"]["parallax_arcsec"], parallax_err_mas=1000 * r["errors"]["parallax_arcsec"],
                   pmra_mas_yr=1000 * r["params"]["pmra_arcsec_yr"], pmdec_mas_yr=1000 * r["params"]["pmdec_arcsec_yr"],
                   pmra_err=1000 * r["errors"]["pmra_arcsec_yr"], pmdec_err=1000 * r["errors"]["pmdec_arcsec_yr"],
                   n_used=r["n_used"], chi2_red=r["chi2_red"],
                   window_scatter_arcsec=[[v["scatter_x"], v["scatter_y"]] for v in r["windows"]])
        rec["d_parallax_mas"] = rec["parallax_mas"] - s["parallax"]
        rec["d_pmra_mas_yr"], rec["d_pmdec_mas_yr"] = rec["pmra_mas_yr"] - s["pmra"], rec["pmdec_mas_yr"] - s["pmdec"]
        # mean offset from the Gaia prediction in each visit window (east, north; arcsec) - the seasonal shift, if any
        o = sel.assign(de=(sel.ra_meas - sel.ra_pred) * 3600 * np.cos(np.radians(sel.dec_pred)), dn=(sel.dec_meas - sel.dec_pred) * 3600)
        rec["window_offsets_from_gaia_arcsec"] = [[float(o.de[(o.mjd > a) & (o.mjd < b)].mean()), float(o.dn[(o.mjd > a) & (o.mjd < b)].mean())] for a, b in zip(lo, hi)]
        null.append(rec)
    used = [v for v in null if v.get("used")]
    dpl = np.array([v["d_parallax_mas"] for v in used])
    wo = np.array([v["window_offsets_from_gaia_arcsec"] for v in used])           # star x window x (east, north)
    null_sum = dict(n_used=len(used), d_parallax_mas=dpl.round(2).tolist(), n_negative=int((dpl < 0).sum()),
                    rms_about_zero_mas=float(np.sqrt(np.mean(dpl ** 2))), mean_mas=float(dpl.mean()), std_mas=float(dpl.std(ddof=1)),
                    mean_err_mas=float(dpl.std(ddof=1) / np.sqrt(len(dpl))),
                    max_abs_over_formal=float(np.max(np.abs(dpl) / np.array([v["parallax_err_mas"] for v in used]))),
                    d_pm_rms_mas_yr=[float(np.sqrt(np.mean(np.array([v["d_pmra_mas_yr"] for v in used]) ** 2))),
                                     float(np.sqrt(np.mean(np.array([v["d_pmdec_mas_yr"] for v in used]) ** 2)))],
                    mean_window_offset_from_gaia_arcsec=dict(windows=[Time((a + b) / 2, format="mjd").iso[:7] for a, b in zip(lo, hi)],
                                                             east=wo[:, :, 0].mean(0).round(4).tolist(), north=wo[:, :, 1].mean(0).round(4).tolist(),
                                                             east_sd_over_stars=wo[:, :, 0].std(0, ddof=1).round(4).tolist(),
                                                             north_sd_over_stars=wo[:, :, 1].std(0, ddof=1).round(4).tolist()))
    r, aux = fit5(tp, n_boot=N_BOOT, sig_floor=sig_floor)
    plx, e_fit, e_boot, e_null = 1000 * r["params"]["parallax_arcsec"], 1000 * r["errors"]["parallax_arcsec"], 1000 * r["bootstrap"]["parallax_std"], null_sum["rms_about_zero_mas"]
    e = max(e_fit, e_boot, e_null)
    gp, ge = GAIA_TARGET["parallax"], GAIA_TARGET["parallax_error"]
    sig0, sigg = plx / e, (plx - gp) / np.hypot(e, ge)
    ok = (abs(sig0) >= 3) and (abs(sigg) <= 2.5) and len(used) >= 3
    return dict(r=r, aux=aux, null=null, null_sum=null_sum, plx=plx, e_fit=e_fit, e_boot=e_boot, e_null=e_null, e=e, sig0=sig0, sigg=sigg,
                verdict="SUPPORTED" if ok else "FAILED")


def fit():
    assert os.path.exists("docs/protocols/E12.md"), "freeze docs/protocols/E12.md first"
    fz, a = analyse(0.0), analyse(GRID_FLOOR)       # exactly as frozen; and with the gridded-position clip fix (deviation 1)
    r, aux, plx, e = a["r"], a["aux"], a["plx"], a["e"]
    gp = GAIA_TARGET["parallax"]
    pc = lambda v: 1000.0 / v  # noqa: E731
    b = [1000 * v for v in r["bootstrap"]["parallax_p16_p50_p84"]]
    tp = pd.read_parquet(TARGET_PARQUET).dropna(subset=["ra_meas"])
    tp = tp[(tp.snr > 20) & np.isfinite(tp.err_arcsec)]
    tw = windows(tp.mjd.values)
    t_off = [dict(date=Time(tp.mjd.values[tw == k].mean(), format="mjd").iso[:10],
                  east=float(((tp.ra_meas - tp.ra_pred) * 3600 * np.cos(np.radians(tp.dec_pred)))[tw == k].mean()),
                  north=float(((tp.dec_meas - tp.dec_pred) * 3600)[tw == k].mean())) for k in np.unique(tw)]
    in_range = all(0.13 - 0.1 <= v["east"] <= 0.29 + 0.1 and -0.7 - 0.1 <= v["north"] <= -0.5 + 0.1 for v in t_off)
    res = dict(experiment="E12", protocol="docs/PROTOCOLS.md PHASE 2 E12; frozen choices docs/protocols/E12.md",
               target="Epsilon Indi B (Ba+Bb, unresolved blend)", data=TARGET_PARQUET,
               method="5-parameter least squares (offset x2, proper motion x2, parallax) on per-picture SPHEREx positions; "
                      "per-picture error = empirical scatter of its visit window; bootstrap over pictures; null = same fit on Gaia stars in the same pictures",
               observer="Earth barycentric position (astropy get_body_barycentric, built-in ephemeris); SPHEREx's offset from the geocentre changes the parallax by <= 0.013 mas",
               headline_basis="numbers at this level use deviation 1 (clip made safe for positions that come on a 0.05 px grid; it changes only null-star fits and "
                              "makes the adopted error larger); the run exactly as frozen is in frozen_procedure_as_run. The verdict is the same in both.",
               fit=r, parallax_mas=plx,
               parallax_err_mas=dict(fit_scaled=a["e_fit"], bootstrap=a["e_boot"], null_rms=a["e_null"], adopted=e, rule="largest of the three (E12.md)"),
               sigma_from_zero=a["sig0"], gaia_parallax_mas=gp, gaia_parallax_err_mas=GAIA_TARGET["parallax_error"], sigma_from_gaia=a["sigg"],
               secondary_reference=dict(label="secondary, not the frozen criterion: the system parallax, Gaia DR3 for Epsilon Indi A (Gaia's own value for Ba, 270.66, is affected by the 11-year orbit; Chen et al. 2022 give 274.99 +- 0.43 for the pair's barycentre)",
                                        parallax_mas=SYSTEM_PARALLAX[0], parallax_err_mas=SYSTEM_PARALLAX[1],
                                        sigma_from_reference=(plx - SYSTEM_PARALLAX[0]) / np.hypot(e, SYSTEM_PARALLAX[1]), distance_pc=pc(SYSTEM_PARALLAX[0])),
               distance_pc=pc(plx), distance_interval_pc_68=[pc(plx + e), pc(plx - e)] if plx - e > 0 else [pc(plx + e), None],
               distance_bootstrap_pc_68=[pc(b[2]), pc(b[0])] if b[0] > 0 else None, distance_ly=pc(plx) * 3.2616,
               distance_interval_ly_68=[pc(plx + e) * 3.2616, pc(plx - e) * 3.2616] if plx - e > 0 else None, gaia_distance_pc=pc(gp),
               proper_motion=dict(pmra_mas_yr=1000 * r["params"]["pmra_arcsec_yr"], pmra_err=1000 * r["errors"]["pmra_arcsec_yr"],
                                  pmdec_mas_yr=1000 * r["params"]["pmdec_arcsec_yr"], pmdec_err=1000 * r["errors"]["pmdec_arcsec_yr"],
                                  null_rms_mas_yr=a["null_sum"]["d_pm_rms_mas_yr"], gaia_pmra=GAIA_TARGET["pmra"], gaia_pmdec=GAIA_TARGET["pmdec"]),
               null=dict(headline="the decisive control: 8 ordinary Gaia stars measured in the same pictures with the same code and fitted the same way",
                         stars=a["null"], summary=a["null_sum"],
                         note="reported as found: every null star has a fitted parallax below Gaia's (see n_negative, mean_mas, std_mas). This is a shared, "
                              "season-dependent position shift that mimics a negative parallax; its cause was not established. No correction is applied; "
                              "the adopted error is the null stars' RMS about zero, so the shift is inside the error bar."),
               descriptive_not_frozen=dict(parallax_minus_null_mean_mas=plx - a["null_sum"]["mean_mas"],
                                           err_mas=float(np.hypot(a["null_sum"]["std_mas"], max(a["e_fit"], a["e_boot"]))),
                                           note="post hoc, not used for the verdict: target parallax after subtracting the null stars' mean offset; error = null-star scatter and fit error in quadrature"),
               offset_from_gaia_propagated=dict(per_window_arcsec=t_off, predicted_arcsec=dict(east=[0.13, 0.29], north=[-0.7, -0.5]),
                                                agrees_within_0p1_arcsec=bool(in_range),
                                                note="descriptive: SPHEREx position minus Gaia DR3 (Ba) propagated with Gaia's motion. Prediction from the published "
                                                     "Ba-Bb orbit (Chen et al. 2022; scripts/epsindi_orbit_check.py): Gaia's motion for Ba "
                                                     "contains orbital motion, and SPHEREx sees the light-weighted centre of the unresolved pair (0.69-0.84 arcsec apart)."),
               frozen_procedure_as_run=dict(parallax_mas=fz["plx"], parallax_err_mas=dict(fit_scaled=fz["e_fit"], bootstrap=fz["e_boot"], null_rms=fz["e_null"], adopted=fz["e"]),
                                            sigma_from_zero=fz["sig0"], sigma_from_gaia=fz["sigg"], verdict=fz["verdict"], null_summary=fz["null_sum"],
                                            null_stars=[{k: v.get(k) for k in ("source_id", "n_selected", "n_used", "d_parallax_mas", "parallax_err_mas", "window_scatter_arcsec")} for v in fz["null"]],
                                            defect="in 4 of 8 null stars one window's robust scatter collapsed to ~0 because the positions come on a 0.05 px grid "
                                                   "(more than half the pictures share the median value); the 5-sigma clip then dropped every other picture of that window "
                                                   "and the window received an error of 0.001-0.008 arcsec. The target's fit was not affected (1 picture clipped in both runs)."),
               criterion="parallax >= 3 sigma from zero AND within 2.5 sigma of Gaia DR3 (270.66 mas); null needs >= 3 usable stars",
               verdict=a["verdict"], verdict_frozen_procedure=fz["verdict"],
               context="Gagne et al. 2026 (ApJ, SPIFF; arXiv:2604.22012) do not model parallax in SPHEREx brown-dwarf photometry because it is 'negligible compared to a SPHEREx pixel even for the nearest brown dwarf'; "
                       "the amplitude here is 0.04 px. WISE/CatWISE astrometry shows season-dependent shifts of 80-600 mas that mimic parallax (Eisenhardt et al. 2020; Marocco et al. 2021), which is what the null stars test for.",
               limitations=["Ba and Bb (0.7-0.8 arcsec apart) are unresolved: SPHEREx measures the blended light, whose centre also shifts slowly with the 11-year orbit and with wavelength",
                            "positions are in each picture's own SPHEREx coordinate solution, which the mission ties to Gaia reference stars; 'SPHEREx alone' means no Gaia motion or distance of this object was used",
                            "four visit windows over 12.3 months; windows 1 and 4 are almost exactly a year apart, so the parallax rests mainly on the October 2025 window",
                            "all null stars came out with a parallax below Gaia's: a shared shift of several hundredths of an arcsecond between the spring and autumn visits, cause not established (the telescope's orientation on the sky reverses between them); it sets the adopted error"],
               gaia_reference="Gaia Collaboration, Vallenari et al. 2023, A&A 674, A1 (Gaia DR3)", gaia_source_id=GAIA_TARGET["source_id"],
               deviations=DEVIATIONS, freeze=FREEZE, written=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    json.dump(res, open("data/findings/e12.json", "w"), indent=1)
    json.dump(dict(frozen=dict(null=fz["null"], summary=fz["null_sum"]), deviation1=dict(null=a["null"], summary=a["null_sum"])), open(f"{OUT}/null_fits.json", "w"), indent=1)
    print(json.dumps({k: res[k] for k in ["parallax_mas", "parallax_err_mas", "sigma_from_zero", "sigma_from_gaia", "secondary_reference", "distance_pc",
                                           "distance_interval_pc_68", "distance_ly", "distance_interval_ly_68", "proper_motion", "verdict", "verdict_frozen_procedure",
                                           "descriptive_not_frozen", "offset_from_gaia_propagated"]}, indent=1))
    print(json.dumps(a["null_sum"], indent=1))
    print("corr", r["correlation"])
    figure(r, aux, res)


DEVIATIONS = [dict(n=1, time="2026-11-15 12:50 PST", what="clip: the robust sigma is floored at 0.155 arcsec (half of the 0.05 px position grid) for target and null stars alike",
                   why="found after the frozen run: positions from scripts/measure_motion.py come on a 0.05 px (0.31 arcsec) grid; in 4 of 8 null stars more than half "
                       "of one window's pictures shared the median value, the robust sigma was ~0 and the frozen 5-sigma clip discarded the rest of that window",
                   effect="target fit unchanged (242.6 mas, 1 picture clipped either way); null RMS about zero 43.3 -> 52.9 mas, so the adopted error grows; verdict unchanged")]
FREEZE = dict(file="docs/protocols/E12.md", sha256_at_freeze="54b2f57c80e23cb6ee6b20ca69ba77ce9a97f88b49d51bd6f6910d6ce95857fd",
              time="2026-11-15 12:08:47 PST (first target fit: 12:38:54 PST)", record="data/phase2/e12/freeze.txt",
              committed_in_run=False, note="the snapshot commit 022fff0 (12:34:10 PST, before the first target fit) holds the file with exactly this hash")


def figure(r, aux, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d, x, y, keep, w, p, p4 = aux["d"], aux["x"], aux["y"], aux["keep"], aux["w"], aux["p"], aux["p4"]
    tt = np.linspace(d.mjd.min() - 20, d.mjd.max() + 20, 600)
    A, _, _ = design(tt, aux["ra0"], aux["dec0"], aux["t0"])
    m5, m4 = A @ p, A[:, :4] @ p4
    n = len(tt)
    lin = A[:, :4] @ p[:4]                                   # straight-line part of the full model
    Ad, _, _ = design(d.mjd.values, aux["ra0"], aux["dec0"], aux["t0"])
    lin_d = Ad[:, :4] @ p[:4]
    nd = len(d)
    c5, c4, cp, cw = "#2a78d6", "#8a8f98", "#c9ced6", "#16181d"
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 5.6), dpi=160)
    a = ax[0]
    a.plot(x[keep], y[keep], ".", ms=3, color=cp, zorder=1, label="single pictures")
    a.plot(m4[:n], m4[n:], "--", lw=1.4, color=c4, zorder=2, label="straight-line motion only")
    a.plot(m5[:n], m5[n:], "-", lw=1.8, color=c5, zorder=3, label="motion + yearly wobble from Earth's orbit")
    for v in r["windows"]:
        a.errorbar(v["x_arcsec"], v["y_arcsec"], xerr=v["x_err"], yerr=v["y_err"], fmt="o", ms=5, color=cw, capsize=2, zorder=4)
        a.annotate(Time(v["mjd_mean"], format="mjd").strftime("%b %Y"), (v["x_arcsec"], v["y_arcsec"]), textcoords="offset points",
                   xytext=(8, 6), fontsize=9, color=cw)
    a.invert_xaxis()
    a.set_aspect("equal", adjustable="datalim")
    a.set_xlabel("position east–west on the sky (arcseconds; east is left)")
    a.set_ylabel("position north–south on the sky (arcseconds)")
    a.set_title("Path across the sky over one year", fontsize=11)
    a.legend(fontsize=8, frameon=False, loc="upper left")
    b = ax[1]
    b.axhline(0, color=c4, ls="--", lw=1.2)
    b.axvline(0, color=c4, ls="--", lw=1.2)
    b.plot((m5 - lin)[:n], (m5 - lin)[n:], "-", lw=1.8, color=c5, label="expected loop for the fitted distance")
    for i, v in enumerate(r["windows"]):
        mk = (w == v["window"]) & keep
        rx, ry = (x - lin_d[:nd])[mk].mean(), (y - lin_d[nd:])[mk].mean()
        b.errorbar(rx, ry, xerr=v["x_err"], yerr=v["y_err"], fmt="o", ms=6, color=cw, capsize=2, zorder=4)
        b.annotate(Time(v["mjd_mean"], format="mjd").strftime("%b %Y"), (rx, ry), textcoords="offset points",
                   xytext=[(10, 12), (10, -16), (10, 10), (-58, -18)][i % 4], fontsize=9)
    b.plot([], [], "o", color=cw, label="average of each visit (bars: scatter between pictures)")
    b.plot(0, 0, "+", ms=10, color=c4, label="where it would stay with no wobble")
    b.invert_xaxis()
    b.set_aspect("equal", adjustable="datalim")
    b.set_xlabel("east–west shift, steady motion removed (arcseconds)")
    b.set_ylabel("north–south shift, steady motion removed (arcseconds)")
    b.set_title("The yearly wobble that gives the distance", fontsize=11)
    b.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    e = res["parallax_err_mas"]["adopted"]
    fig.suptitle(f"Epsilon Indi B: wobble {res['parallax_mas']:.0f} ± {e:.0f} thousandths of an arcsecond → distance "
                 f"{res['distance_ly']:.1f} light-years (Gaia: {1000 / res['gaia_parallax_mas'] * 3.2616:.1f}) — {res['verdict'].lower()}",
                 fontsize=11.5)
    for a_ in ax:
        a_.spines[["top", "right"]].set_visible(False)
        a_.grid(alpha=0.2, lw=0.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs("site/data/figs", exist_ok=True)
    fig.savefig("site/data/figs/e12_parallax.png")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    dict(select=select, check=check, measure=measure, fit=fit)[sys.argv[1]]()
