"""E8 — Barnard's Star motion with saturation-robust positions (docs/PROTOCOLS.md PHASE 2; choices: docs/protocols/E8.md).

Stages, in order (the last two only after docs/protocols/E8.md is frozen):
  select    comparison stars (Gaia DR3, predicted K within 1 mag of Barnard's, slow, |b| > 15)  -> data/phase2/e8/comp_stars.json
  cache     41 x 41 native cut-outs of every comparison star                                  -> data/phase2/e8/stamps/<source_id>/
  measure   all position estimators on the comparison stars                                   -> data/phase2/e8/comp_meas.parquet
  evaluate  per-estimator scatter and window offsets on the comparison stars                  -> data/phase2/e8/comp_eval.json
  barnard   frozen estimator on Barnard's Star, fit, controls, verdict, figure                -> data/findings/e8.json, site/data/figs/e8_barnard.png
  OMP_NUM_THREADS=1 .venv/bin/python scripts/run_e8.py <stage>
"""
import json, os, sys, time, warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
warnings.filterwarnings("ignore")
from astropy import log  # noqa: E402
log.setLevel("ERROR")
import astropy.units as u  # noqa: E402
from astropy.coordinates import SkyCoord, get_body_barycentric  # noqa: E402
from astropy.io import fits  # noqa: E402
from astropy.time import Time  # noqa: E402
from astropy.wcs import WCS  # noqa: E402

from orrery.data import BAD_BITS, load_stamp, save_stamp  # noqa: E402

OUT = "data/phase2/e8"
TAP = "https://gaia.aip.de/tap"
SEED, N_CONES, PER_CONE, CAP, HALF = 8, 10, 3, 60, 20
K_BARNARD = 4.524          # 2MASS Ks (Cutri et al. 2003)
SAT_BITS = (1 << 1) | (1 << 15)                 # OVERFLOW, NONLINEAR
MASK = BAD_BITS | (1 << 1)                      # hard-bad (incl. NONLINEAR) + OVERFLOW, as scripts/measure_motion.py
MASK_SAT = MASK & ~((1 << 2) | (1 << 26) | (1 << 28))   # the same without SUR_ERROR, BLOOM, HALO (the block around a bright core)
BARNARD = dict(source_id="4472832130942575872", ra=269.44850252543836, dec=4.739420051112412, pmra=-801.5509783684709,
               pmdec=10362.394206546573, parallax=546.975939730948, G=8.193974494934082)   # Gaia DR3, epoch J2016.0


def k_pred(G, bp_rp):
    """Predicted 2MASS Ks from Gaia G and BP-RP (Gaia EDR3 photometric relation, Riello et al. 2021, valid for
    -0.5 < BP-RP < 2.5). Used because the 2MASS service (VizieR TAP) was unreachable on 2026-11-15."""
    return G - (-0.0981 + 2.089 * bp_rp - 0.1579 * bp_rp ** 2)


def windows(mjd, gap=30.0):
    mjd = np.asarray(mjd, float)
    o = np.argsort(mjd)
    w = np.zeros(len(mjd), int)
    w[o] = np.concatenate([[0], np.cumsum(np.diff(mjd[o]) > gap)])
    return w


def tap_async(q):
    import pyvo
    for i in range(4):
        try:
            return pyvo.dal.TAPService(TAP).run_async(q).to_table().to_pandas()
        except Exception as e:
            print("  TAP retry", repr(e)[:120], flush=True)
            time.sleep(5 * (i + 1))
    raise RuntimeError("Gaia query failed")


# ------------------------------------------------------------------ comparison-star selection
def select():
    from build_field import candidates, load_index
    rng = np.random.default_rng(SEED)
    cen = []
    while len(cen) < N_CONES:
        ra, dec = float(rng.uniform(0, 360)), float(np.degrees(np.arcsin(rng.uniform(-1, 1))))
        c = SkyCoord(ra, dec, unit="deg")
        if abs(c.galactic.b.deg) > 17 and 15 < abs(c.barycentrictrueecliptic.lat.deg) < 45:
            cen.append((ra, dec))

    def cone(i):
        p = f"{OUT}/cone_{i:02d}.parquet"
        if not os.path.exists(p):
            ra, dec = cen[i]
            t = tap_async("SELECT source_id, ra, dec, pmra, pmdec, parallax, ruwe, phot_g_mean_mag, bp_rp FROM gaiadr3.gaia_source "
                          f"WHERE 1=CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',{ra},{dec},2.0)) AND phot_g_mean_mag<10")
            t["source_id"] = t["source_id"].astype("int64")
            t.to_parquet(p)
        return pd.read_parquet(p)
    with ThreadPoolExecutor(4) as ex:
        cones = list(ex.map(cone, range(N_CONES)))
    df, _ = load_index()
    stars, log_ = [], []
    for i, t in enumerate(cones):
        t = t.dropna(subset=["pmra", "pmdec", "bp_rp"]).copy()
        t["K_pred"] = k_pred(t.phot_g_mean_mag, t.bp_rp)
        b = SkyCoord(t.ra.values, t.dec.values, unit="deg").galactic.b.deg
        ok = (np.abs(t.K_pred - K_BARNARD) <= 1.0) & (t.bp_rp >= 0.8) & (t.bp_rp <= 2.5) & (np.hypot(t.pmra, t.pmdec) < 50) & (np.abs(b) > 15)
        for j in t.index[ok]:
            s = 3600 * np.hypot((t.ra - t.ra[j]) * np.cos(np.radians(t.dec[j])), t.dec - t.dec[j])
            if ((s > 0) & (s < 120)).any():                      # another G < 10 star within 2 arcmin
                ok[j] = False
        el = t[ok].sort_values("source_id").reset_index(drop=True)
        pick = np.sort(rng.choice(len(el), min(PER_CONE, len(el)), replace=False))
        log_.append(dict(cone=i, ra=cen[i][0], dec=cen[i][1], n_G10=int(len(t)), n_eligible=int(len(el))))
        for r in el.iloc[pick].itertuples():
            c = candidates(df, r.ra, r.dec).sort_values("mjd", kind="stable")
            w = windows(c.mjd.values) if len(c) else np.array([])
            nw = [int((w == k).sum()) for k in np.unique(w)]
            good = sum(n >= 8 for n in nw) >= 2
            if len(c) > CAP:
                c = c.iloc[np.linspace(0, len(c) - 1, CAP).round().astype(int)]
            rec = dict(source_id=str(r.source_id), cone=i, ra=r.ra, dec=r.dec, pmra=r.pmra, pmdec=r.pmdec, parallax=r.parallax,
                       G=r.phot_g_mean_mag, bp_rp=r.bp_rp, K_pred=float(r.K_pred), ruwe=r.ruwe,
                       b=float(SkyCoord(r.ra, r.dec, unit="deg").galactic.b.deg), n_pictures_per_window=nw, used=bool(good),
                       keys=list(c.key) if good else [])
            stars.append(rec)
            print(rec["source_id"], f"G={r.phot_g_mean_mag:.2f} Kp={r.K_pred:.2f}", nw, good, flush=True)
    rec = dict(seed=SEED, rule="see docs/protocols/E8.md", K_barnard=K_BARNARD, cones=log_, stars=stars, n_used=sum(s["used"] for s in stars))
    json.dump(rec, open(f"{OUT}/comp_stars.json", "w"), indent=1)
    print("used", rec["n_used"], "of", len(stars))


# ------------------------------------------------------------------ cut-out cache
def predict(g, mjd):
    """Gaia DR3 position at the picture time: J2016.0 + proper motion + parallax (Earth barycentric position)."""
    t = Time(np.atleast_1d(mjd), format="mjd")
    dt = t.jyear - 2016.0
    a, d = np.radians(g["ra"]), np.radians(g["dec"])
    X, Y, Z = get_body_barycentric("earth", t).xyz.to(u.au).value
    plx = (g.get("parallax") or 0.0) / 1000.0
    da = g["pmra"] * dt / 1000.0 + plx * (X * np.sin(a) - Y * np.cos(a))
    dd = g["pmdec"] * dt / 1000.0 + plx * (X * np.cos(a) * np.sin(d) + Y * np.sin(a) * np.sin(d) - Z * np.cos(d))
    return g["ra"] + da / 3600.0 / np.cos(d), g["dec"] + dd / 3600.0


def _fetch(a):
    import e2_cache
    sid, key, ra, dec = a
    name = key.split("/")[-1].replace(".fits", "")
    p = f"{OUT}/stamps/{sid}/{name}.npz"
    if os.path.exists(p):
        return "ok"
    try:
        st = e2_cache.fast_stamp(key, ra, dec, half=HALF)
        if st is None or st.image.shape != (2 * HALF + 1, 2 * HALF + 1):
            return "edge"
        save_stamp(st, p[:-4] + ".tmp.npz")
        os.replace(p[:-4] + ".tmp.npz", p)
        return "ok"
    except Exception as e:
        return "error " + repr(e)[:80]


def cache():
    stars = [s for s in json.load(open(f"{OUT}/comp_stars.json"))["stars"] if s["used"]]
    idx = pd.read_parquet("data/index/frames_dr1.parquet", columns=["key", "mjd"]).set_index("key").mjd

    def gaia_nb(s):
        p = f"{OUT}/gaia/{s['source_id']}.parquet"
        if not os.path.exists(p):
            os.makedirs(f"{OUT}/gaia", exist_ok=True)
            tap_async("SELECT source_id, ra, dec, pmra, pmdec, phot_g_mean_mag FROM gaiadr3.gaia_source WHERE "
                      f"1=CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',{s['ra']},{s['dec']},0.06)) AND phot_g_mean_mag<19.0").to_parquet(p)
    jobs = []
    for s in stars:
        os.makedirs(f"{OUT}/stamps/{s['source_id']}", exist_ok=True)
        ra, dec = predict(s, idx[s["keys"]].values)
        jobs += [(s["source_id"], k, float(r), float(d)) for k, r, d in zip(s["keys"], ra, dec)]
    t = time.time()
    with ThreadPoolExecutor(16) as ex:
        g = [ex.submit(gaia_nb, s) for s in stars]
        st = list(ex.map(_fetch, jobs))
        [f.result() for f in g]
    print(pd.Series(st).value_counts().to_string(), f"{time.time() - t:.0f}s", flush=True)


# ------------------------------------------------------------------ position estimators
# All work on one cut-out and a FIXED starting point (xs, ys) in cut-out pixels; every search is a +-1.5 px box
# around that point. Returned: (x, y) in cut-out pixels (NaN if no solution) plus a quality number.
BOX, STEP_A, STEP_W = 1.5, 0.05, 0.1
RIN, ROUT, FRAC = 2.0, 8.0, 0.05


def _valid(st, bits=MASK):
    return ((st.flags & bits) == 0) & np.isfinite(st.image) & np.isfinite(st.variance) & (st.variance > 0)


def _quad_refine(surf, j, i, step):
    """Sub-step minimum from the 3 x 3 neighbourhood of the grid minimum (separable parabolas)."""
    out = []
    for line in (surf[j, i - 1:i + 2], surf[j - 1:j + 2, i]):
        den = line[0] - 2 * line[1] + line[2]
        out.append(float(np.clip(0.5 * (line[0] - line[2]) / den, -1, 1) * step) if np.all(np.isfinite(line)) and den > 0 else 0.0)
    return out


def est_epsf(st, xs, ys, nb_xy, bits=MASK, h=15):
    """(a) the earlier method: ePSF + Gaia neighbours + plane, chi2 grid +-1.5 px step 0.05 on the central 31 x 31
    pixels, hard-bad + OVERFLOW masked. Same chi2 as orrery.fit.fit_stamp (static columns projected out once)."""
    from orrery.psf import psf_model
    c = (st.image.shape[0] - 1) // 2
    sl = (slice(c - h, c + h + 1), slice(c - h, c + h + 1))
    img, var = st.image[sl].astype(float), st.variance[sl].astype(float)
    m = _valid(st, bits)[sl]
    x0, y0 = xs - (c - h), ys - (c - h)
    H = W = 2 * h + 1
    yy, xx = np.indices((H, W))
    nb = [(a - (c - h), b - (c - h)) for a, b in nb_xy]
    nb = [(a, b) for a, b in nb if -3 <= a <= W + 2 and -3 <= b <= H + 2 and np.hypot(a - x0, b - y0) > 1.5]
    cols = [psf_model(st.epsf, a, b, (H, W), st.epsf_oversample, box=3) for a, b in nb] + [np.ones((H, W)), xx - h, yy - h]
    sw = 1.0 / np.sqrt(var[m])
    Xs = np.stack([c_[m] for c_ in cols], 1) * sw[:, None]
    Xs = Xs[:, np.any(Xs != 0, axis=0)]
    U, S, _ = np.linalg.svd(Xs, full_matrices=False)
    U = U[:, S > S.max() * 1e-12]
    d = img[m] * sw
    d = d - U @ (U.T @ d)
    dd = d @ d
    offs = np.linspace(-BOX, BOX, int(round(2 * BOX / STEP_A)) + 1)
    surf = np.full((len(offs), len(offs)), dd)
    amp = np.zeros_like(surf)
    den_ = np.zeros_like(surf)
    for j, oy in enumerate(offs):
        for i, ox in enumerate(offs):
            p = psf_model(st.epsf, x0 + ox, y0 + oy, (H, W), st.epsf_oversample, box=3)[m] * sw
            p = p - U @ (U.T @ p)
            den = p @ p
            if den > 1e-30:
                surf[j, i] = dd - (p @ d) ** 2 / den
                amp[j, i], den_[j, i] = (p @ d) / den, den
    j, i = np.unravel_index(np.argmin(surf), surf.shape)
    edge = j in (0, len(offs) - 1) or i in (0, len(offs) - 1)
    snr = amp[j, i] * np.sqrt(den_[j, i])
    return xs + offs[i], ys + offs[j], dict(snr=float(snr), edge=bool(edge), chi2_dof=float(surf[j, i] / max(m.sum() - U.shape[1] - 1, 1)))


def _wing_setup(st, bits=MASK):
    from scipy.ndimage import map_coordinates
    img = np.where(np.isfinite(st.image), st.image, 0.0).astype(float)
    good = _valid(st, bits)
    var = np.where(good, st.variance, 0.0).astype(float)
    b = np.ones_like(good)
    b[2:-2, 2:-2] = False
    bkg = float(np.median(img[b & good])) if (b & good).sum() > 10 else 0.0
    yy, xx = np.indices(img.shape)
    return map_coordinates, img - bkg, good, var, yy, xx


def _grid_min(cost, xs, ys):
    offs = np.linspace(-BOX, BOX, int(round(2 * BOX / STEP_W)) + 1)
    surf = np.array([[cost(xs + ox, ys + oy) for ox in offs] for oy in offs])
    if not np.isfinite(surf).any():
        return np.nan, np.nan, dict(edge=True, cost=np.nan)
    j, i = np.unravel_index(np.nanargmin(surf), surf.shape)
    edge = j in (0, len(offs) - 1) or i in (0, len(offs) - 1)
    dx, dy = (0.0, 0.0) if edge else _quad_refine(surf, j, i, STEP_W)
    return xs + offs[i] + dx, ys + offs[j] + dy, dict(edge=bool(edge), cost=float(surf[j, i]))


def est_sym(st, xs, ys, bits=MASK):
    """(c) centre of symmetry: minimise the weighted mean squared difference between the image and its 180-degree
    rotation about the trial centre, over unflagged pixels at RIN..ROUT px whose mirror point touches no flagged pixel."""
    mc, img, good, var, yy, xx = _wing_setup(st, bits)
    bad = (~good).astype(float)

    def cost(cx, cy):
        r = np.hypot(xx - cx, yy - cy)
        sel = good & (r >= RIN) & (r <= ROUT)
        co = [2 * cy - yy[sel], 2 * cx - xx[sel]]
        ok = mc(bad, co, order=1, mode="constant", cval=1.0) < 1e-6
        if ok.sum() < 40:
            return np.nan
        im = mc(img, co, order=1, mode="constant", cval=0.0)[ok]
        vm = mc(var, co, order=1, mode="constant", cval=0.0)[ok]
        a = img[sel][ok]
        w = 1.0 / (var[sel][ok] + vm + (FRAC * 0.5 * (np.abs(a) + np.abs(im))) ** 2)
        return float(np.sum(w * (a - im) ** 2) / ok.sum())
    return _grid_min(cost, xs, ys)


KNOTS = np.array([2.0, 3.0, 4.0, 5.0, 6.5, 8.0])


def est_radial(st, xs, ys, bits=MASK):
    """(d) centre of circular symmetry: minimise the weighted scatter of unflagged wing pixels (RIN..ROUT px) about a
    free radial profile (piecewise-linear in radius, 6 knots), refitted at every trial centre."""
    _, img, good, var, yy, xx = _wing_setup(st, bits)

    def cost(cx, cy):
        r = np.hypot(xx - cx, yy - cy)
        sel = good & (r >= RIN) & (r <= ROUT)
        if sel.sum() < 60:
            return np.nan
        rr, v = r[sel], img[sel]
        X = np.maximum(0, 1 - np.abs(rr[:, None] - KNOTS[None, :]) / np.where(rr[:, None] < KNOTS[None, :],
                       np.r_[1, np.diff(KNOTS)][None, :], np.r_[np.diff(KNOTS), 1][None, :]))
        sw = 1.0 / np.sqrt(var[sel] + (FRAC * np.abs(v)) ** 2)
        beta, *_ = np.linalg.lstsq(X * sw[:, None], v * sw, rcond=None)
        return float(np.sum(((v - X @ beta) * sw) ** 2) / (sel.sum() - len(KNOTS)))
    return _grid_min(cost, xs, ys)


def est_core(st, xs, ys):
    """(e) centre of the flagged core: unweighted centroid of the connected group (8-connectivity) of OVERFLOW /
    NONLINEAR pixels that has a member within 2 px of the starting point."""
    from scipy.ndimage import label
    sat = (st.flags & SAT_BITS) != 0
    lab, n = label(sat, structure=np.ones((3, 3)))
    yy, xx = np.indices(sat.shape)
    near = sat & (np.hypot(xx - xs, yy - ys) <= 2.0)
    if not near.any():
        return np.nan, np.nan, dict(n_core=0)
    ids = np.unique(lab[near])
    g = np.isin(lab, ids)
    return float(xx[g].mean()), float(yy[g].mean()), dict(n_core=int(g.sum()))


def run_est(k, st, xs, ys, nb_xy):
    """a, c, d: protocol estimators (full mask); e: flagged-core centroid; f, g, h = a, c, d with MASK_SAT."""
    if k in "af":
        return est_epsf(st, xs, ys, nb_xy, MASK if k == "a" else MASK_SAT)
    if k in "cg":
        return est_sym(st, xs, ys, MASK if k == "c" else MASK_SAT)
    if k in "dh":
        return est_radial(st, xs, ys, MASK if k == "d" else MASK_SAT)
    return est_core(st, xs, ys)


def measure_stamp(st, xs, ys, nb_xy, which="acdefgh"):
    """Run the estimators on one cut-out; returns a flat dict (pixel offsets from the starting point + sky offsets)."""
    wcs = WCS(fits.Header.fromstring(st.wcs_header))
    yy, xx = np.indices(st.image.shape)
    r = np.hypot(xx - xs, yy - ys)
    sat = (st.flags & SAT_BITS) != 0
    bad = ~_valid(st)
    out = dict(mjd=st.mjd, det=st.detector, obs_id=st.obs_id, x_full=xs + st.x0, y_full=ys + st.y0, wl=float(st.cwave[HALF, HALF]),
               n_sat_1p5=int((sat & (r <= 1.5)).sum()), n_sat=int((sat & (r <= 10)).sum()), n_bad_8=int((bad & (r <= 8)).sum()),
               r_sat_max=float(r[sat & (r <= 10)].max()) if (sat & (r <= 10)).any() else 0.0,
               n_free_box=int((~bad & (np.abs(xx - xs) <= 3) & (np.abs(yy - ys) <= 3)).sum()))
    ra0, dec0 = wcs.all_pix2world([[xs + st.x0, ys + st.y0]], 0)[0]
    for k in which:
        try:
            x, y, q = run_est(k, st, xs, ys, nb_xy)
        except Exception as e:
            x, y, q = np.nan, np.nan, dict(error=repr(e)[:60])
        out[f"{k}_dx"], out[f"{k}_dy"] = x - xs, y - ys
        for kk, v in q.items():
            out[f"{k}_{kk}"] = v
        if np.isfinite(x):
            ra, dec = wcs.all_pix2world([[x + st.x0, y + st.y0]], 0)[0]
            out[f"{k}_ra"], out[f"{k}_dec"] = float(ra), float(dec)
        else:
            out[f"{k}_ra"] = out[f"{k}_dec"] = np.nan
    out["ra_start"], out["dec_start"] = float(ra0), float(dec0)
    # local Jacobian (arcsec east / north per pixel step), used for the detector-frame offset correction
    e = wcs.all_pix2world([[xs + st.x0 + 1, ys + st.y0], [xs + st.x0, ys + st.y0 + 1]], 0)
    cd = np.cos(np.radians(dec0))
    out.update(j_ex=float((e[0][0] - ra0) * cd * 3600), j_nx=float((e[0][1] - dec0) * 3600),
               j_ey=float((e[1][0] - ra0) * cd * 3600), j_ny=float((e[1][1] - dec0) * 3600))
    return out


def _measure_comp(a):
    s, path = a
    st = load_stamp(path)
    wcs = WCS(fits.Header.fromstring(st.wcs_header))
    ra, dec = predict(s, st.mjd)
    x, y = wcs.all_world2pix([[float(ra[0]), float(dec[0])]], 0)[0]
    g = pd.read_parquet(f"{OUT}/gaia/{s['source_id']}.parquet")
    dt = Time(st.mjd, format="mjd").jyear - 2016.0
    gr = g.ra.values + g.pmra.fillna(0).values * dt / 3.6e6 / np.cos(np.radians(g.dec.values))
    gd = g.dec.values + g.pmdec.fillna(0).values * dt / 3.6e6
    gx, gy = wcs.all_world2pix(np.c_[gr, gd], 0).T
    r = measure_stamp(st, x - st.x0, y - st.y0, list(zip(gx - st.x0, gy - st.y0)))
    r.update(source_id=s["source_id"], file=os.path.basename(path), ra_pred=float(ra[0]), dec_pred=float(dec[0]))
    return r


def measure(workers=5):
    stars = [s for s in json.load(open(f"{OUT}/comp_stars.json"))["stars"] if s["used"]]
    jobs = []
    for s in stars:
        d = f"{OUT}/stamps/{s['source_id']}"
        s2 = {k: s[k] for k in ("source_id", "ra", "dec", "pmra", "pmdec", "parallax")}
        jobs += [(s2, f"{d}/{f}") for f in sorted(os.listdir(d)) if f.endswith(".npz") and ".tmp" not in f]
    t = time.time()
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(_measure_comp, jobs, chunksize=8))
    pd.DataFrame(rows).to_parquet(f"{OUT}/comp_meas.parquet")
    print(len(rows), f"{time.time() - t:.0f}s")


# ------------------------------------------------------------------ evaluation on the comparison stars
EST = "acdefgh"
EST_NAME = dict(a="(a) ePSF fit, all flagged pixels masked (earlier method)", b="(b) = (a) on detectors with < 10% flagged cores",
                c="(c) centre of symmetry of the wings, all flagged pixels masked", d="(d) centre of circular symmetry of the wings, all flagged pixels masked",
                e="(e) centroid of the flagged core", f="(f) ePSF fit, only OVERFLOW/NONLINEAR/dead pixels masked",
                g="(g) centre of symmetry, only OVERFLOW/NONLINEAR/dead pixels masked", h="(h) centre of circular symmetry, only OVERFLOW/NONLINEAR/dead pixels masked")


def rob(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return float(1.4826 * np.median(np.abs(v - np.median(v)))) if len(v) else np.nan


def clip_mean(v, n_sig=5.0, floor=0.0):
    """Mean after one round of n_sig robust-sigma clipping about the median; returns (mean, n kept, keep mask)."""
    v = np.asarray(v, float)
    med, s = np.median(v), rob(v)
    keep = np.abs(v - med) <= n_sig * max(s, floor, 1e-9)
    return float(v[keep].mean()), int(keep.sum()), keep


def det_offsets(m, k, exclude=None):
    """Median detector-frame offset (dx, dy) in pixels per detector, from comparison-star pictures."""
    q = m[(m.source_id != exclude)] if exclude is not None else m
    q = q[np.isfinite(q[f"{k}_dx"]) & ~q[f"{k}_edge"].fillna(False).astype(bool)] if f"{k}_edge" in q else q[np.isfinite(q[f"{k}_dx"])]
    return {int(d): (float(g[f"{k}_dx"].median()), float(g[f"{k}_dy"].median())) for d, g in q.groupby("det")}


def sky_offsets(m, k, cal=None):
    """East/north offsets (arcsec) of estimator k from the starting point, optionally after subtracting the
    per-detector pixel-frame offset `cal`; NaN where the estimator has no solution or sits on the box edge."""
    dx, dy = m[f"{k}_dx"].values.astype(float).copy(), m[f"{k}_dy"].values.astype(float).copy()
    if f"{k}_edge" in m:
        bad = m[f"{k}_edge"].fillna(True).astype(bool).values
        dx[bad] = np.nan
    if cal is not None:
        cx = np.array([cal.get(int(d), (np.nan, np.nan))[0] for d in m.det])
        cy = np.array([cal.get(int(d), (np.nan, np.nan))[1] for d in m.det])
        dx, dy = dx - cx, dy - cy
    return m.j_ex.values * dx + m.j_ey.values * dy, m.j_nx.values * dx + m.j_ny.values * dy


def window_table(m, k, cal_mode, dets=None):
    """Per star and window: clipped mean east/north offset from Gaia. cal_mode: None | 'loo' (leave-one-star-out)."""
    rows, pic = [], []
    for sid, g in m.groupby("source_id"):
        g = g.sort_values("mjd")
        cal = det_offsets(m, k, exclude=sid) if cal_mode else None
        de, dn = sky_offsets(g, k, cal)
        ok = np.isfinite(de) & np.isfinite(dn)
        if dets is not None:
            ok &= g.det.isin(dets).values
        w = windows(g.mjd.values)
        for wi in np.unique(w):
            s = ok & (w == wi)
            if s.sum() < 5:
                continue
            me, _, ke = clip_mean(de[s])
            mn, _, kn = clip_mean(dn[s])
            kk = ke & kn
            rows.append(dict(source_id=sid, window=int(wi), mjd=float(g.mjd.values[s].mean()), n=int(kk.sum()), n_valid=int(s.sum()),
                             n_all=int((w == wi).sum()), de=float(de[s][kk].mean()), dn=float(dn[s][kk].mean()),
                             sd_e=float(de[s][kk].std(ddof=1)), sd_n=float(dn[s][kk].std(ddof=1))))
            pic.append(pd.DataFrame(dict(det=g.det.values[s], re=de[s] - np.median(de[s]), rn=dn[s] - np.median(dn[s]))))
    return pd.DataFrame(rows), (pd.concat(pic) if pic else pd.DataFrame(columns=["det", "re", "rn"]))


def metrics(m, k, cal_mode, dets=None):
    wt, pic = window_table(m, k, cal_mode, dets)
    if len(wt) < 10:
        return dict(n_star_windows=int(len(wt)))
    off = np.r_[wt.de - wt.de.mean(), wt.dn - wt.dn.mean()]
    dif = []
    for sid, g in wt.groupby("source_id"):
        g = g.sort_values("window")
        for i in range(len(g) - 1):
            dif += [g.de.values[i + 1] - g.de.values[i], g.dn.values[i + 1] - g.dn.values[i]]
    dif = np.array(dif)
    return dict(n_star_windows=int(len(wt)), n_stars=int(wt.source_id.nunique()), frac_valid=float(wt.n_valid.sum() / wt.n_all.sum()),
                per_picture_scatter_arcsec=rob(np.r_[pic.re, pic.rn]),
                per_picture_scatter_by_detector={int(d): rob(np.r_[g.re, g.rn]) for d, g in pic.groupby("det")},
                mean_offset_arcsec=[float(wt.de.mean()), float(wt.dn.mean())],
                window_scatter_arcsec=float(np.sqrt(np.mean(off ** 2))), window_scatter_robust_arcsec=rob(off),
                window_to_window_arcsec=float(np.sqrt(np.mean(dif ** 2)) / np.sqrt(2)) if len(dif) else None,
                window_to_window_robust_arcsec=rob(dif) / np.sqrt(2) if len(dif) else None, n_window_pairs=int(len(dif) // 2))


def evaluate():
    m = pd.read_parquet(f"{OUT}/comp_meas.parquet")
    sat_frac = {int(d): float((g.n_sat_1p5 > 0).mean()) for d, g in m.groupby("det")}
    dets_b = [d for d, f in sat_frac.items() if f < 0.10]
    res = dict(n_pictures=int(len(m)), n_stars=int(m.source_id.nunique()),
               frac_pictures_with_flagged_core_by_detector=sat_frac, detectors_for_b=dets_b,
               median_unmasked_pixels_in_epsf_box_of_49=float(m.n_free_box.median()),
               median_saturated_pixels=float(m.n_sat.median()), median_masked_within_8px=float(m.n_bad_8.median()), candidates={})
    for k in EST:
        for cm in (None, "loo"):
            res["candidates"][k + ("+cal" if cm else "")] = metrics(m, k, cm)
    for cm in (None, "loo"):
        res["candidates"]["b" + ("+cal" if cm else "")] = metrics(m, "a", cm, dets_b) if dets_b else dict(n_star_windows=0, note="no detector has < 10% flagged cores")
    # why (a) fails: its error against the number of unmasked pixels left under the ePSF box
    ea = np.hypot(*sky_offsets(m, "a"))
    res["a_error_vs_unmasked_box_pixels"] = [dict(n_free_lo=int(lo), n_free_hi=int(hi), n=int(s.sum()), median_error_arcsec=float(np.nanmedian(ea[s])))
                                             for lo, hi in ((0, 5), (5, 15), (15, 30), (30, 50)) for s in [((m.n_free_box >= lo) & (m.n_free_box < hi)).values] if s.sum()]
    res["a_fraction_on_box_edge"] = float(m.a_edge.mean())
    ok = {k: v for k, v in res["candidates"].items() if v.get("n_star_windows", 0) >= 20 and v.get("frac_valid", 0) >= 0.8}
    best = min(ok, key=lambda k: ok[k]["window_scatter_arcsec"])
    res["choice"] = dict(estimator=best, rule="smallest comparison-star per-window scatter among candidates with >= 80% usable pictures and >= 20 star-windows",
                         window_scatter_arcsec=ok[best]["window_scatter_arcsec"], window_to_window_arcsec=ok[best]["window_to_window_arcsec"],
                         per_window_error_arcsec=max(ok[best]["window_scatter_arcsec"], ok[best]["window_to_window_arcsec"]))
    k0 = best[0] if best[0] != "b" else "a"
    res["detector_offsets_px"] = {k: det_offsets(m, k) for k in EST}
    wt, _ = window_table(m, k0, "loo" if best.endswith("+cal") else None, dets_b if best[0] == "b" else None)
    wt.to_parquet(f"{OUT}/comp_windows_chosen.parquet")
    # the comparison stars most like the target in brightness (predicted K <= 4.7; Barnard's Star has K = 4.52)
    kp = {s_["source_id"]: s_["K_pred"] for s_ in json.load(open(f"{OUT}/comp_stars.json"))["stars"]}
    br = wt[wt.source_id.map(kp) <= 4.7]
    res["choice"]["n_bright_star_windows"] = int(len(br))
    res["choice"]["window_scatter_bright_arcsec"] = float(np.sqrt(np.mean(np.r_[br.de - wt.de.mean(), br.dn - wt.dn.mean()] ** 2)))
    res["choice"]["per_window_error_arcsec"] = max(res["choice"]["per_window_error_arcsec"], res["choice"]["window_scatter_bright_arcsec"])
    # the same motion fit on every comparison star (they do not move at this precision): empirical null for the motion
    pm = []
    for sid, g in m.groupby("source_id"):
        g = g.sort_values("mjd")
        de, dn = sky_offsets(g, k0, det_offsets(m, k0, exclude=sid) if best.endswith("+cal") else None)
        okp = np.isfinite(de) & np.isfinite(dn) & (g.det.isin(dets_b).values if best[0] == "b" else True)
        rows, _ = window_means(g.mjd.values[okp], de[okp], dn[okp])
        if len(rows) >= 2:
            tt = np.array([r["t"] for r in rows])
            f_ = linfit_windows(tt, np.array([r["east"] for r in rows]), np.array([r["north"] for r in rows]), np.ones(len(rows)))
            pm.append(dict(source_id=sid, n_windows=len(rows), baseline_yr=float(tt.max() - tt.min()),
                           pm_east_arcsec_yr=f_["east"]["pm_arcsec_yr"], pm_north_arcsec_yr=f_["north"]["pm_arcsec_yr"]))
    v = np.array([[q["pm_east_arcsec_yr"], q["pm_north_arcsec_yr"]] for q in pm])
    res["comparison_star_motion_null"] = dict(stars=pm, rms_arcsec_yr=float(np.sqrt(np.mean(v ** 2))), max_abs_arcsec_yr=float(np.abs(v).max()),
                                              median_baseline_yr=float(np.median([q["baseline_yr"] for q in pm])))
    json.dump(res, open(f"{OUT}/comp_eval.json", "w"), indent=1)
    print("comparison-star motion null:", {kk: res["comparison_star_motion_null"][kk] for kk in ("rms_arcsec_yr", "max_abs_arcsec_yr", "median_baseline_yr")})
    print(f"{'cand':7s} {'n_sw':>5s} {'valid':>6s} {'pic':>6s} {'win':>6s} {'win_rob':>7s} {'w2w':>6s} {'w2w_rob':>7s}  mean offset")
    for k, v in res["candidates"].items():
        if "window_scatter_arcsec" in v:
            print(f"{k:7s} {v['n_star_windows']:5d} {v['frac_valid']:6.2f} {v['per_picture_scatter_arcsec']:6.3f} {v['window_scatter_arcsec']:6.3f} "
                  f"{v['window_scatter_robust_arcsec']:7.3f} {v['window_to_window_arcsec']:6.3f} {v['window_to_window_robust_arcsec']:7.3f}  "
                  f"{v['mean_offset_arcsec'][0]:+.3f} {v['mean_offset_arcsec'][1]:+.3f}")
        else:
            print(k, v)
    print({k: res[k] for k in ("frac_pictures_with_flagged_core_by_detector", "median_unmasked_pixels_in_epsf_box_of_49", "a_error_vs_unmasked_box_pixels", "a_fraction_on_box_edge", "choice")})


# ------------------------------------------------------------------ Barnard's Star (run once, after the freeze)
BFIELD = "data/efields/e3x_barnard"        # 77 x 77 native cut-outs of every indexed picture covering the field centre
N_CONTROL = 4
GRID_FLOOR = 0.155     # arcsec, half of the 0.05 px grid of estimator (a): floor of the robust sigma in the controls' clip (deviation 1)
DEVIATIONS = [dict(n=1, time="2026-11-15 12:55 PST", what="control stars only: the robust sigma used by the 5-sigma clip is floored at 0.155 arcsec (half of estimator (a)'s 0.05 px grid)",
                   why="found on the first run: estimator (a) returns positions on a 0.05 px (0.31 arcsec) grid, more than half of a window's pictures share the median value, "
                       "the robust sigma is 0 and the clip dropped every other picture; 7 of 8 controls came out as exactly 0.000 +- 0.001 arcsec/yr, which is an artefact",
                   effect="Barnard's Star itself is untouched (its estimator is continuous; identical numbers in both runs); only the control values change. "
                          "First-run output kept as data/phase2/e8/e8_first_run_control_bug.json")]
FREEZE = dict(file="docs/protocols/E8.md", sha256_at_freeze="db7e58a4fca87e6bde1239dd27ae20cd91b307a4add96216708d21f6760ec0d7",
              time="2026-11-15 12:44:28 PST (target run started 12:44:45 PST)", record="data/phase2/e8/freeze.txt", frozen_copy="data/phase2/e8/E8_frozen_copy.md",
              committed_in_run=False, note="the hash is the pre-registration record until the file is committed")


def controls():
    """Control stars for the Barnard pictures (Gaia table only): faint, isolated Gaia neighbours. 'near' = inside the cached
    cut-outs (62-130 arcsec from the field centre, G 11-15); 'far' = 130-220 arcsec, G 11-14.5, own cut-outs fetched."""
    fj = json.load(open(f"{BFIELD}/field.json"))
    g = pd.read_parquet(f"{BFIELD}/gaia.parquet")
    g = g[g.source_id.astype("int64") != int(BARNARD["source_id"])].reset_index(drop=True)
    dt = 2026.0 - 2016.0
    ra = g.ra + g.pmra.fillna(0) * dt / 3.6e6 / np.cos(np.radians(g.dec))
    dec = g.dec + g.pmdec.fillna(0) * dt / 3.6e6
    cd = np.cos(np.radians(fj["dec"]))
    sep = 3600 * np.hypot((ra - fj["ra"]) * cd, dec - fj["dec"])
    iso = np.ones(len(g), bool)
    for i in g.index:
        s = 3600 * np.hypot((ra - ra[i]) * cd, dec - dec[i])
        iso[i] = not ((s > 0) & (s < 15) & (g.phot_g_mean_mag < g.phot_g_mean_mag[i] + 2)).any()
    rng = np.random.default_rng(SEED)
    out, n_el = [], {}
    for name, ok in (("near", (g.phot_g_mean_mag >= 11) & (g.phot_g_mean_mag <= 15) & (sep >= 62) & (sep <= 130)),
                     ("far", (g.phot_g_mean_mag >= 11) & (g.phot_g_mean_mag <= 14.5) & (sep > 130) & (sep <= 220))):
        el = g[ok & iso & g.pmra.notna()].assign(sep_arcsec=sep[ok & iso & g.pmra.notna()]).sort_values("source_id").reset_index(drop=True)
        n_el[name] = int(len(el))
        pick = np.sort(rng.choice(len(el), min(N_CONTROL, len(el)), replace=False))
        out += [dict(source_id=str(int(r.source_id)), ra=r.ra, dec=r.dec, pmra=r.pmra, pmdec=r.pmdec, parallax=float(np.nan_to_num(r.parallax)),
                     G=r.phot_g_mean_mag, sep_arcsec=float(r.sep_arcsec), far=(name == "far")) for r in el.iloc[pick].itertuples()]
    json.dump(dict(n_eligible=n_el, seed=SEED, stars=out), open(f"{OUT}/controls.json", "w"), indent=1)
    print(n_el, [(c["source_id"], round(c["G"], 2), round(c["sep_arcsec"]), c["far"]) for c in out])


def _fetch_ctl(a):
    import e2_cache
    sid, key, ra, dec = a
    p = f"{OUT}/ctl_stamps/{sid}/{key.split('/')[-1].replace('.fits', '')}.npz"
    if os.path.exists(p):
        return "ok"
    try:
        st = e2_cache.fast_stamp(key, ra, dec, half=15)
        if st is None or st.image.shape != (31, 31):
            return "edge"
        save_stamp(st, p[:-4] + ".tmp.npz")
        os.replace(p[:-4] + ".tmp.npz", p)
        return "ok"
    except Exception as e:
        return "error " + repr(e)[:80]


def cache_controls():
    fj = json.load(open(f"{BFIELD}/field.json"))
    jobs = []
    for c in json.load(open(f"{OUT}/controls.json"))["stars"]:
        if c["far"]:
            os.makedirs(f"{OUT}/ctl_stamps/{c['source_id']}", exist_ok=True)
            ra, dec = predict(c, np.array([f["mjd"] for f in fj["frame_list"]]))
            jobs += [(c["source_id"], f["key"], float(r), float(d)) for f, r, d in zip(fj["frame_list"], ra, dec)]
    t = time.time()
    with ThreadPoolExecutor(6) as ex:
        st = list(ex.map(_fetch_ctl, jobs))
    print(pd.Series(st).value_counts().to_string(), f"{time.time() - t:.0f}s", flush=True)


def _startcheck_one(a):
    s, path, k, ox, oy = a
    st = load_stamp(path)
    wcs = WCS(fits.Header.fromstring(st.wcs_header))
    ra, dec = predict(s, st.mjd)
    x, y = wcs.all_world2pix([[float(ra[0]), float(dec[0])]], 0)[0]
    xs, ys = x - st.x0 + ox, y - st.y0 + oy
    nb = []
    if k in "af":
        g = pd.read_parquet(f"{OUT}/gaia/{s['source_id']}.parquet")
        dt = Time(st.mjd, format="mjd").jyear - 2016.0
        gx, gy = wcs.all_world2pix(np.c_[g.ra.values + g.pmra.fillna(0).values * dt / 3.6e6 / np.cos(np.radians(g.dec.values)),
                                         g.dec.values + g.pmdec.fillna(0).values * dt / 3.6e6], 0).T
        nb = list(zip(gx - st.x0, gy - st.y0))
    xm, ym, q = run_est(k, st, xs, ys, nb)
    return dict(source_id=s["source_id"], file=os.path.basename(path), dx2=xm - (x - st.x0), dy2=ym - (y - st.y0), edge2=bool(q.get("edge", False)))


def startcheck(workers=5):
    """Does the chosen estimator return the same position when the search box is started up to 0.8 px away from the
    star (as it will be for Barnard's Star, whose box is fixed on the sky)? Comparison stars only."""
    ev = json.load(open(f"{OUT}/comp_eval.json"))
    k = ev["choice"]["estimator"][0].replace("b", "a")
    m = pd.read_parquet(f"{OUT}/comp_meas.parquet")
    stars = {s["source_id"]: {kk: s[kk] for kk in ("source_id", "ra", "dec", "pmra", "pmdec", "parallax")}
             for s in json.load(open(f"{OUT}/comp_stars.json"))["stars"] if s["used"]}
    rng = np.random.default_rng(SEED)
    r_, th = 0.8 * np.sqrt(rng.uniform(0, 1, len(m))), rng.uniform(0, 2 * np.pi, len(m))
    jobs = [(stars[sid], f"{OUT}/stamps/{sid}/{f}", k, float(a * np.cos(b)), float(a * np.sin(b))) for sid, f, a, b in zip(m.source_id, m.file, r_, th)]
    with ProcessPoolExecutor(workers) as ex:
        d = pd.DataFrame(list(ex.map(_startcheck_one, jobs, chunksize=16)))
    d = d.merge(m[["source_id", "file", f"{k}_dx", f"{k}_dy", f"{k}_edge"]], on=["source_id", "file"])
    ok = ~d.edge2 & ~d[f"{k}_edge"].astype(bool) & np.isfinite(d.dx2) & np.isfinite(d[f"{k}_dx"])
    diff = np.hypot(d.dx2 - d[f"{k}_dx"], d.dy2 - d[f"{k}_dy"])[ok] * 6.2
    ev["start_point_check"] = dict(estimator=k, displacement="uniform in a disc of radius 0.8 px, default_rng(8)", n=int(ok.sum()),
                                   frac_on_edge=float(d.edge2.mean()), median_shift_arcsec=float(diff.median()), p95_shift_arcsec=float(diff.quantile(0.95)),
                                   max_shift_arcsec=float(diff.max()), frac_shift_gt_0p3_arcsec=float((diff > 0.3).mean()))
    json.dump(ev, open(f"{OUT}/comp_eval.json", "w"), indent=1)
    print(ev["start_point_check"])


def freeze():
    """Write the frozen choice (estimator, per-window error, control stars) that the `barnard` stage reads."""
    ev = json.load(open(f"{OUT}/comp_eval.json"))
    fz = dict(estimator=ev["choice"]["estimator"], per_window_error_arcsec=ev["choice"]["per_window_error_arcsec"],
              controls=json.load(open(f"{OUT}/controls.json"))["stars"], written=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    json.dump(fz, open(f"{OUT}/frozen.json", "w"), indent=1)
    print(fz["estimator"], fz["per_window_error_arcsec"], len(fz["controls"]))


def substamp(st, x, y, half):
    """Square sub-cut-out of a cached cut-out, centred on the pixel nearest full-frame (x, y); None if it does not fit."""
    from dataclasses import replace
    ix, iy = int(round(x)) - st.x0, int(round(y)) - st.y0
    H, W = st.image.shape
    if ix - half < 0 or iy - half < 0 or ix + half + 1 > W or iy + half + 1 > H:
        return None
    sl = (slice(iy - half, iy + half + 1), slice(ix - half, ix + half + 1))
    return replace(st, x0=st.x0 + ix - half, y0=st.y0 + iy - half, image=st.image[sl], variance=st.variance[sl], zodi=st.zodi[sl],
                   flags=st.flags[sl], cwave=st.cwave[sl], cband=st.cband[sl])


def plx_factors(ra, dec, mjd):
    t = Time(np.asarray(mjd), format="mjd")
    X, Y, Z = get_body_barycentric("earth", t).xyz.to(u.au).value
    a, d = np.radians(ra), np.radians(dec)
    return X * np.sin(a) - Y * np.cos(a), X * np.cos(a) * np.sin(d) + Y * np.sin(a) * np.sin(d) - Z * np.cos(d)


def _measure_barnard(a):
    f, k, fj, controls = a
    st = load_stamp(f"{BFIELD}/{f['file']}")
    wcs = WCS(fits.Header.fromstring(st.wcs_header))
    g = pd.read_parquet(f"{BFIELD}/gaia.parquet")
    dt = Time(st.mjd, format="mjd").jyear - 2016.0
    gr = g.ra.values + g.pmra.fillna(0).values * dt / 3.6e6 / np.cos(np.radians(g.dec.values))
    gd = g.dec.values + g.pmdec.fillna(0).values * dt / 3.6e6
    gx, gy = wcs.all_world2pix(np.c_[gr, gd], 0).T
    out = dict(file=f["file"], key=f["key"])
    x, y = wcs.all_world2pix([[fj["ra"], fj["dec"]]], 0)[0]           # FIXED starting point: the field centre
    s2 = substamp(st, x, y, HALF)
    if s2 is not None:
        out.update(measure_stamp(s2, x - s2.x0, y - s2.y0, list(zip(gx - s2.x0, gy - s2.y0)), which=k))
    ctl = []
    for c in controls:
        ra, dec = predict(c, st.mjd)
        cx, cy = wcs.all_world2pix([[float(ra[0]), float(dec[0])]], 0)[0]
        if c.get("far"):
            pth = f"{OUT}/ctl_stamps/{c['source_id']}/{f['key'].split('/')[-1].replace('.fits', '')}.npz"
            s3 = load_stamp(pth) if os.path.exists(pth) else None
        else:
            s3 = substamp(st, cx, cy, 15)
        if s3 is None:
            continue
        xa, ya, q = est_epsf(s3, cx - s3.x0, cy - s3.y0, list(zip(gx - s3.x0, gy - s3.y0)), MASK)
        ram, decm = wcs.all_pix2world([[xa + s3.x0, ya + s3.y0]], 0)[0]
        ctl.append(dict(source_id=c["source_id"], mjd=st.mjd, det=st.detector, snr=q["snr"], edge=q["edge"],
                        de=float((ram - ra[0]) * np.cos(np.radians(dec[0])) * 3600), dn=float((decm - dec[0]) * 3600)))
    out["controls"] = ctl
    return out


def linfit_windows(t, E, N, sig):
    """Offset + proper motion from window means (weighted least squares; errors scaled by reduced chi2 if > 1)."""
    t0 = float(np.mean(t))
    A = np.c_[np.ones_like(t), t - t0]
    out = {}
    for name, y in (("east", E), ("north", N)):
        w = 1 / np.asarray(sig) ** 2
        cov = np.linalg.inv(A.T @ (A * w[:, None]))
        p = cov @ (A.T @ (w * y))
        chi2 = float(np.sum(w * (y - A @ p) ** 2))
        dof = len(t) - 2
        s2 = max(1.0, chi2 / dof) if dof > 0 else 1.0
        out[name] = dict(offset_arcsec=float(p[0]), pm_arcsec_yr=float(p[1]), pm_err=float(np.sqrt(cov[1, 1] * s2)), chi2=chi2, dof=dof)
    out["t0_jyear"] = t0
    return out


def window_means(mjd, E, N, sig_floor=0.0):
    """Per visit window (gaps > 30 d): 5-robust-sigma clip about the window median of the residuals from a straight
    line fitted to all pictures, then plain means of time and position."""
    t = Time(mjd, format="mjd").jyear
    A = np.c_[np.ones_like(t), t - t.mean()]
    rE, rN = E - A @ np.linalg.lstsq(A, E, rcond=None)[0], N - A @ np.linalg.lstsq(A, N, rcond=None)[0]
    w = windows(mjd)
    rows, keep_all = [], np.zeros(len(t), bool)
    for wi in np.unique(w):
        s = np.where(w == wi)[0]
        if len(s) < 5:
            continue
        _, _, ke = clip_mean(rE[s], floor=sig_floor)
        _, _, kn = clip_mean(rN[s], floor=sig_floor)
        k = s[ke & kn]
        keep_all[k] = True
        rows.append(dict(window=int(wi), n=int(len(k)), n_clipped=int(len(s) - len(k)), mjd=float(mjd[k].mean()), t=float(t[k].mean()),
                         east=float(E[k].mean()), north=float(N[k].mean()),
                         internal_err_east=float(rE[k].std(ddof=1) / np.sqrt(len(k))), internal_err_north=float(rN[k].std(ddof=1) / np.sqrt(len(k))),
                         scatter_east=float(rob(rE[k] - np.median(rE[k]))), scatter_north=float(rob(rN[k] - np.median(rN[k])))))
    return rows, keep_all


def barnard(workers=5):
    assert os.path.exists("docs/protocols/E8.md"), "freeze docs/protocols/E8.md first"
    ev = json.load(open(f"{OUT}/comp_eval.json"))
    fz = json.load(open(f"{OUT}/frozen.json"))
    best, sig_w = fz["estimator"], fz["per_window_error_arcsec"]
    k = best[0] if best[0] != "b" else "a"
    cal = {int(d): v for d, v in ev["detector_offsets_px"][k].items()} if best.endswith("+cal") else None
    fj = json.load(open(f"{BFIELD}/field.json"))
    controls = fz["controls"]
    p = f"{OUT}/barnard_meas.parquet"
    if not os.path.exists(p):
        with ProcessPoolExecutor(workers) as ex:
            rows = list(ex.map(_measure_barnard, [(f, k, dict(ra=fj["ra"], dec=fj["dec"]), controls) for f in fj["frame_list"]], chunksize=4))
        ctl = pd.DataFrame([c for r in rows for c in r.pop("controls")])
        ctl.to_parquet(f"{OUT}/barnard_controls.parquet")
        pd.DataFrame(rows).to_parquet(p)
    m, ctl = pd.read_parquet(p), pd.read_parquet(f"{OUT}/barnard_controls.parquet")
    n_all = len(m)
    m = m.dropna(subset=["mjd"]).reset_index(drop=True)
    if best[0] == "b":
        m = m[m.det.isin(ev["detectors_for_b"])].reset_index(drop=True)
    de, dn = sky_offsets(m, k, cal)                       # arcsec east / north of the fixed starting point (field centre)
    ok = np.isfinite(de) & np.isfinite(dn)
    mjd = m.mjd.values[ok]
    pa, pd_ = plx_factors(BARNARD["ra"], BARNARD["dec"], mjd)
    plx = BARNARD["parallax"] / 1000.0
    E, N = de[ok] - plx * pa, dn[ok] - plx * pd_          # parallax (Gaia value, held fixed) removed
    rows, keep = window_means(mjd, E, N)
    for r in rows:
        r["err"] = sig_w
    t = np.array([r["t"] for r in rows])
    fit = linfit_windows(t, np.array([r["east"] for r in rows]), np.array([r["north"] for r in rows]), np.full(len(rows), sig_w)) if len(rows) >= 2 else None
    # Gaia prediction in the same frame (offsets from the field centre, parallax removed) for the residuals and the figure
    def gaia_track(mj):
        dt = Time(mj, format="mjd").jyear - 2016.0
        return ((BARNARD["ra"] - fj["ra"]) * np.cos(np.radians(fj["dec"])) * 3600 + BARNARD["pmra"] * dt / 1000.0,
                (BARNARD["dec"] - fj["dec"]) * 3600 + BARNARD["pmdec"] * dt / 1000.0)
    for r in rows:
        ge, gn = gaia_track(r["mjd"])
        r.update(gaia_east=float(ge), gaia_north=float(gn), resid_east=float(r["east"] - ge), resid_north=float(r["north"] - gn),
                 date=Time(r["mjd"], format="mjd").iso[:10])
    res = dict(experiment="E8", protocol="docs/PROTOCOLS.md PHASE 2 E8; frozen choices docs/protocols/E8.md", target="Barnard's Star",
               gaia=dict(source_id=BARNARD["source_id"], pmra_mas_yr=BARNARD["pmra"], pmdec_mas_yr=BARNARD["pmdec"], parallax_mas=BARNARD["parallax"],
                         reference="Gaia Collaboration, Vallenari et al. 2023, A&A 674, A1 (Gaia DR3)"),
               method=dict(estimator=best, description=EST_NAME[best[0]] + ("; per-detector pixel-frame offset from the comparison stars subtracted" if cal else ""),
                           start_point="fixed sky point (field centre RA %.6f Dec %.6f) in every picture; the +-1.5 px search box does not follow the star" % (fj["ra"], fj["dec"]),
                           parallax="Gaia DR3 parallax held fixed; parallax factors from Earth's barycentric position (astropy get_body_barycentric, built-in ephemeris); SPHEREx's low Earth orbit changes them by < 0.03 mas",
                           per_window_error="comparison-star per-window scatter, floored at their window-to-window spread", detector_offsets_px=cal),
               pictures=dict(cached=n_all, measured=int(len(m)), with_solution=int(ok.sum()), used=int(keep.sum())),
               comparison=dict(n_stars=ev["n_stars"], n_pictures=ev["n_pictures"], candidates=ev["candidates"], choice=ev["choice"],
                               a_error_vs_unmasked_box_pixels=ev["a_error_vs_unmasked_box_pixels"],
                               median_unmasked_pixels_in_epsf_box_of_49=ev["median_unmasked_pixels_in_epsf_box_of_49"],
                               frac_pictures_with_flagged_core_by_detector=ev["frac_pictures_with_flagged_core_by_detector"],
                               stars=[{kk: s[kk] for kk in ("source_id", "ra", "dec", "G", "bp_rp", "K_pred", "pmra", "pmdec", "b", "n_pictures_per_window")}
                                      for s in json.load(open(f"{OUT}/comp_stars.json"))["stars"] if s["used"]]),
               windows=rows, per_window_error_arcsec=sig_w, deviations=DEVIATIONS, freeze=FREEZE,
               earlier_failure=dict(result="data/findings/motion_barnard.json: pmdec +5.6 +- 1.5, pmra -5.0 +- 1.9 arcsec/yr (FAILED)",
                                    causes_tested_on_comparison_stars=[
                                        "main: the Level-2 flags mark a block of 30-50 pixels (BLOOM, SUR_ERROR around 2-6 OVERFLOW pixels) around a star this bright; with that mask a median of 7 of the 49 pixels under the ePSF model remain, and the earlier estimator's median error is 9.4 arcsec when fewer than 5 remain (0.43 arcsec when 30 or more do)",
                                        "second: the centre of the wings sits 0.09-0.20 px from the Gaia position along detector x in every detector; the detector's orientation on the sky reverses between visits half a year apart, so without a per-detector correction this becomes a jump of about 1 arcsec between visits (window-to-window spread 0.82 arcsec uncorrected, 0.11 arcsec corrected)",
                                        "not supported as main causes: asymmetric flag masks (flagged-core centroid within 0.1 px of Gaia), detector edges (scatter 0.077 vs 0.056 px); persistence not tested"]),
               start_point_check=ev.get("start_point_check"), comparison_star_motion_null=ev.get("comparison_star_motion_null"),
               written=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    if fit:
        gE, gN = BARNARD["pmra"] / 1000.0, BARNARD["pmdec"] / 1000.0
        mE, mN, sE, sN = fit["east"]["pm_arcsec_yr"], fit["north"]["pm_arcsec_yr"], fit["east"]["pm_err"], fit["north"]["pm_err"]
        tot = float(np.hypot(mE, mN))
        stot = float(np.sqrt((mE * sE) ** 2 + (mN * sN) ** 2) / tot)
        res.update(fit=fit, pmra_arcsec_yr=mE, pmra_err=sE, pmdec_arcsec_yr=mN, pmdec_err=sN, total_arcsec_yr=tot, total_err=stot,
                   total_sigma=tot / stot, sigma_from_gaia=dict(pmra=(mE - gE) / sE, pmdec=(mN - gN) / sN),
                   baseline_yr=float(t.max() - t.min()), n_windows=len(rows))
    # controls: faint Gaia neighbours in the same pictures, estimator (a), offsets from their own Gaia prediction
    cres = []
    for c in controls:
        q = ctl[(ctl.source_id == c["source_id"]) & (ctl.snr > 10) & ~ctl.edge] if len(ctl) else ctl
        if len(q) < 20:
            cres.append(dict(source_id=c["source_id"], G=c["G"], used=False, n=int(len(q))))
            continue
        cr, _ = window_means(q.mjd.values, q.de.values, q.dn.values, sig_floor=GRID_FLOOR)
        if len(cr) < 2:
            cres.append(dict(source_id=c["source_id"], G=c["G"], used=False, n=int(len(q)), reason="< 2 windows"))
            continue
        tt = np.array([r["t"] for r in cr])
        fe = linfit_windows(tt, np.array([r["east"] for r in cr]), np.array([r["north"] for r in cr]),
                            np.array([np.hypot(r["internal_err_east"], r["internal_err_north"]) / np.sqrt(2) for r in cr]))
        # with 2 windows the fit has no spare freedom: error from the windows' own scatter
        cres.append(dict(source_id=c["source_id"], G=c["G"], sep_arcsec=c["sep_arcsec"], used=True, n=int(len(q)), n_windows=len(cr),
                         gaia_pmra_mas_yr=c["pmra"], gaia_pmdec_mas_yr=c["pmdec"],
                         d_pmra_arcsec_yr=fe["east"]["pm_arcsec_yr"], d_pmra_err=fe["east"]["pm_err"],
                         d_pmdec_arcsec_yr=fe["north"]["pm_arcsec_yr"], d_pmdec_err=fe["north"]["pm_err"],
                         sigma=[fe["east"]["pm_arcsec_yr"] / fe["east"]["pm_err"], fe["north"]["pm_arcsec_yr"] / fe["north"]["pm_err"]],
                         windows=[{kk: r[kk] for kk in ("mjd", "n", "east", "north", "internal_err_east", "internal_err_north")} for r in cr]))
    used = [c for c in cres if c.get("used")]
    ctl_ok = len(used) >= 3 and all(max(abs(c["sigma"][0]), abs(c["sigma"][1])) <= 2.5 for c in used)
    res["controls"] = dict(stars=cres, n_used=len(used), consistent_with_gaia=bool(ctl_ok),
                           note="measured with estimator (a) (they have no flagged core); value = fitted minus Gaia proper motion; error from each window's own picture-to-picture scatter")
    if fit:
        crit = abs(res["sigma_from_gaia"]["pmra"]) <= 2.5 and abs(res["sigma_from_gaia"]["pmdec"]) <= 2.5 and res["total_sigma"] >= 5
        res["criterion"] = "both components within 2.5 sigma of Gaia DR3 AND total motion >= 5 sigma; control: >= 3 faint neighbours consistent with Gaia"
        res["verdict"] = "SUPPORTED" if (crit and ctl_ok) else "FAILED"
        res["criterion_parts"] = dict(components_within_2p5_sigma=bool(abs(res["sigma_from_gaia"]["pmra"]) <= 2.5 and abs(res["sigma_from_gaia"]["pmdec"]) <= 2.5),
                                      total_motion_ge_5_sigma=bool(res["total_sigma"] >= 5), controls_consistent=bool(ctl_ok))
    else:
        res["verdict"] = "FAILED"
        res["criterion_parts"] = dict(note="fewer than 2 usable windows")
    json.dump(res, open("data/findings/e8.json", "w"), indent=1)
    pd.DataFrame(rows).to_parquet("data/findings/e8_windows.parquet")
    pd.DataFrame(dict(mjd=mjd, det=m.det.values[ok], east=E, north=N, used=keep)).to_parquet(f"{OUT}/barnard_pictures.parquet")
    print(json.dumps({kk: res.get(kk) for kk in ("pictures", "windows", "pmra_arcsec_yr", "pmra_err", "pmdec_arcsec_yr", "pmdec_err", "total_sigma",
                                                  "sigma_from_gaia", "verdict", "criterion_parts")}, indent=1))
    control_diag()
    figure()


def control_diag():
    """Post hoc, descriptive (not part of the frozen analysis): is the control stars' shared east-west drift a fixed
    offset in detector pixels that flips on the sky when the telescope's orientation reverses between visits?"""
    res = json.load(open("data/findings/e8.json"))
    fj = json.load(open(f"{BFIELD}/field.json"))
    rows = []
    for f in fj["frame_list"]:
        st = load_stamp(f"{BFIELD}/{f['file']}")
        w = WCS(fits.Header.fromstring(st.wcs_header))
        x, y = w.all_world2pix([[fj["ra"], fj["dec"]]], 0)[0]
        p = w.all_pix2world([[x, y], [x + 1, y], [x, y + 1]], 0)
        cd = np.cos(np.radians(fj["dec"]))
        J = np.array([[(p[1][0] - p[0][0]) * cd * 3600, (p[2][0] - p[0][0]) * cd * 3600], [(p[1][1] - p[0][1]) * 3600, (p[2][1] - p[0][1]) * 3600]])
        rows.append(dict(mjd=st.mjd, det=st.detector, Ji=np.linalg.inv(J), ang=float(np.degrees(np.arctan2(J[1, 0], J[0, 0])))))
    c = pd.read_parquet(f"{OUT}/barnard_controls.parquet")
    c = c[(c.snr > 10) & ~c.edge].merge(pd.DataFrame(rows), on=["mjd", "det"])
    d = np.array([Ji @ np.array([a, b]) for Ji, a, b in zip(c.Ji, c.de, c.dn)])
    c["dx"], c["dy"], c["win"] = d[:, 0], d[:, 1], windows(c.mjd.values)
    ps = c.groupby(["source_id", "win"])[["dx", "dy", "de", "dn"]].mean()
    tab = []
    for wi in sorted(c.win.unique()):
        q = ps.xs(wi, level="win")
        tab.append(dict(window=int(wi), detector_x_axis_angle_from_east_deg=float(c.ang[c.win == wi].median()),
                        sky_east_arcsec=float(q.de.mean()), sky_east_sd_over_stars=float(q.de.std(ddof=1)), sky_north_arcsec=float(q.dn.mean()),
                        sky_north_sd_over_stars=float(q.dn.std(ddof=1)), pixel_dx=float(q.dx.mean()), pixel_dx_sd_over_stars=float(q.dx.std(ddof=1)),
                        pixel_dy=float(q.dy.mean()), pixel_dy_sd_over_stars=float(q.dy.std(ddof=1)), n_stars=int(len(q))))
    used = [s for s in res["controls"]["stars"] if s.get("used")]
    dra = np.array([s["d_pmra_arcsec_yr"] for s in used])
    ddec = np.array([s["d_pmdec_arcsec_yr"] for s in used])
    e_t = res["pmra_err"]
    res["controls"]["summary"] = dict(mean_d_pmra_arcsec_yr=float(dra.mean()), sd_d_pmra=float(dra.std(ddof=1)), mean_d_pmdec_arcsec_yr=float(ddec.mean()),
                                      sd_d_pmdec=float(ddec.std(ddof=1)), n_beyond_2p5_sigma=int(sum(max(abs(s["sigma"][0]), abs(s["sigma"][1])) > 2.5 for s in used)),
                                      fraction_of_barnard_motion=float(abs(dra.mean()) / res["total_arcsec_yr"]))
    res["post_hoc_not_frozen"] = dict(
        note="descriptive, added after the control clause failed; none of this changes the verdict",
        control_offsets_by_window=tab,
        reading="the controls' mean offset from Gaia has the same sign in detector pixels in both visits (about +0.005 to +0.010 px along detector x) while the "
                "detector's orientation on the sky reverses, so on the sky it changes sign between visits: a small fixed detector-frame offset of the ePSF positions, "
                "not a motion. Barnard's own estimator has its per-detector offset removed with the comparison stars, the controls' estimator (a) does not.",
        controls_against_target_error=dict(error_arcsec_yr=e_t, max_abs_sigma=float(max(np.abs(dra).max(), np.abs(ddec).max()) / e_t),
                                           note="the same control values compared with the target's frozen error bar instead of their own picture-to-picture scatter"))
    res["verdict_note"] = ("Frozen criterion as written in docs/protocols/E8.md: the two motion clauses are met (both components within 0.2 sigma of Gaia; total motion "
                           "27.7 sigma), the control clause is not: the 8 faint neighbours share an apparent east-west motion of about -0.17 arcsec/yr "
                           "(2-6 sigma against their own picture-to-picture scatter; 1.7% of Barnard's motion, under half of its error bar). Hence FAILED as registered.")
    json.dump(res, open("data/findings/e8.json", "w"), indent=1)
    print(json.dumps(dict(summary=res["controls"]["summary"], post_hoc=res["post_hoc_not_frozen"]), indent=1))


VERDICT_LINE = {("SUPPORTED", True): "registered test: passed",
                ("FAILED", False): "the motion matches Gaia, but the registered check on faint neighbour stars did not pass, so the registered verdict is: failed",
                ("FAILED", True): "registered test: failed"}


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = json.load(open("data/findings/e8.json"))
    pic = pd.read_parquet(f"{OUT}/barnard_pictures.parquet")
    cw = pd.read_parquet(f"{OUT}/comp_windows_chosen.parquet")
    fj = json.load(open(f"{BFIELD}/field.json"))
    rows, sig = res["windows"], res["per_window_error_arcsec"]
    c5, c4, cp, ck = "#2a78d6", "#8a8f98", "#c9ced6", "#16181d"
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 6.3), dpi=160, gridspec_kw=dict(width_ratios=[1.15, 1]))
    a = ax[0]
    tt = np.linspace(pic.mjd.min() - 15, pic.mjd.max() + 15, 200)
    dt = Time(tt, format="mjd").jyear - 2016.0
    ge = (BARNARD["ra"] - fj["ra"]) * np.cos(np.radians(fj["dec"])) * 3600 + BARNARD["pmra"] * dt / 1000.0
    gn = (BARNARD["dec"] - fj["dec"]) * 3600 + BARNARD["pmdec"] * dt / 1000.0
    u_ = pic[pic.used]
    a.plot(u_.east, u_.north, ".", ms=3, color=cp, label="single SPHEREx pictures", zorder=1)
    a.plot(ge, gn, "-", lw=1.8, color=c5, label="where Gaia says the star should be", zorder=2)
    for r in rows:
        a.errorbar(r["east"], r["north"], xerr=sig, yerr=sig, fmt="o", ms=6, color=ck, capsize=3, zorder=4)
        a.plot(r["gaia_east"], r["gaia_north"], "s", ms=7, mfc="none", mec=c5, mew=1.6, zorder=3)
        a.annotate(Time(r["mjd"], format="mjd").strftime("%b %Y"), (r["east"], r["north"]), textcoords="offset points", xytext=(10, -4), fontsize=9)
    a.plot([], [], "o", color=ck, label="SPHEREx average of each visit (bars: spread of the comparison stars)")
    a.plot([], [], "s", mfc="none", mec=c5, mew=1.6, label="Gaia's position on the same dates")
    a.invert_xaxis()
    a.set_aspect("equal", adjustable="datalim")
    a.set_xlabel("position east–west on the sky (arcseconds; east is left)")
    a.set_ylabel("position north–south on the sky (arcseconds)")
    a.set_title("Barnard's Star between two SPHEREx visits", fontsize=11)
    a.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    b = ax[1]
    b.plot(cw.de - cw.de.mean(), cw.dn - cw.dn.mean(), "o", ms=4, color=cp, label=f"{cw.source_id.nunique()} comparison stars that do not move (one point per visit)")
    th = np.linspace(0, 2 * np.pi, 200)
    b.plot(sig * np.cos(th), sig * np.sin(th), "--", lw=1.2, color=c4, label=f"their typical spread ({sig:.2f} arcseconds)")
    for i, r in enumerate(rows):
        b.plot(r["resid_east"], r["resid_north"], "o", ms=8, color=c5, zorder=5)
        b.annotate(Time(r["mjd"], format="mjd").strftime("%b %Y"), (r["resid_east"], r["resid_north"]), textcoords="offset points",
                   xytext=[(10, 8), (-62, -16)][i % 2], fontsize=9)
    b.plot([], [], "o", color=c5, ms=8, label="Barnard's Star: SPHEREx minus Gaia, each visit")
    b.axhline(0, color=c4, lw=0.6)
    b.axvline(0, color=c4, lw=0.6)
    b.invert_xaxis()
    b.set_aspect("equal", adjustable="datalim")
    b.set_xlabel("difference from Gaia, east–west (arcseconds)")
    b.set_ylabel("difference from Gaia, north–south (arcseconds)")
    b.set_title("How far each measurement lands from Gaia", fontsize=11)
    b.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    if "pmdec_arcsec_yr" in res:
        fig.suptitle(f"Barnard's Star motion per year: north {res['pmdec_arcsec_yr']:+.1f} ± {res['pmdec_err']:.1f}, east {res['pmra_arcsec_yr']:+.1f} ± {res['pmra_err']:.1f} arcseconds "
                     f"(Gaia: {BARNARD['pmdec'] / 1000:+.1f}, {BARNARD['pmra'] / 1000:+.1f})\n" + VERDICT_LINE[(res["verdict"], res["criterion_parts"].get("controls_consistent", False))], fontsize=11)
    for a_ in ax:
        a_.spines[["top", "right"]].set_visible(False)
        a_.grid(alpha=0.2, lw=0.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    os.makedirs("site/data/figs", exist_ok=True)
    fig.savefig("site/data/figs/e8_barnard.png")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    globals()[sys.argv[1]]()
