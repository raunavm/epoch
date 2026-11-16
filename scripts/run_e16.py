"""E16 — detector-frame astrometric calibration of the ePSF positions, then E12 and E8 re-tested
(docs/PROTOCOLS.md PHASE 3 E16; frozen choices: docs/protocols/E16.md). The registered E8/E12 verdicts stand.

Stages:
  select    calibration stars from the cached fields (not E8/E12)        -> data/phase2/e16/calib_stars.json
  measure   per-picture ePSF positions of the calibration stars           -> data/phase2/e16/calib_meas/<field>.parquet
  check     estimator identity: run_e8.est_epsf vs measure_motion.one      -> data/phase2/e16/estimator_check.json
  model     diagnostics, model selection on half A, validation on half B  -> data/phase2/e16/model.json
  retest    (only after the freeze) E12 and E8 with corrected positions   -> data/findings/e16.json
  figure                                                                  -> site/data/figs/e16_calibration.png
  OMP_NUM_THREADS=1 .venv/bin/python scripts/run_e16.py <stage>
"""
import json, os, sys, time, warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from dataclasses import replace
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
warnings.filterwarnings("ignore")
from astropy import log  # noqa: E402
log.setLevel("ERROR")
from astropy.io import fits  # noqa: E402
from astropy.time import Time  # noqa: E402
from astropy.wcs import WCS  # noqa: E402

from orrery.data import load_stamp  # noqa: E402
import run_e8 as e8  # noqa: E402  (est_epsf, substamp, predict, windows, clip_mean, rob, MASK)

OUT = "data/phase2/e16"
EF = "data/efields"
TAP = "https://gaia.aip.de/tap"
SEED = 16
G_MIN, G_MAX, RUWE_MAX = 11.0, 16.0, 1.4
R_MAX_PX = 34.0            # star within 34 px (209") of the field centre: its 31 x 31 sub-cut-out fits in the 101 x 101 cut-out (HALF = 50)
HALF = 50                  # new cut-outs, read per picture, measured on the fly, not stored
N_PIC = 120                # pictures per field: rows linspace(0, n-1, 120).round() of the MJD-sorted cached frame list
PIX = 6.15                 # arcsec per pixel (nominal)
GRID_FLOOR = 0.155         # arcsec: floor of the robust sigma in every 5-sigma clip (half the 0.05 px grid; E8/E12 deviation 1)
SNR_MIN = 20.0             # picture used if ePSF signal-to-noise > 20 (E12 rule) and the solution is off the box edge
EXCLUDE_FIELDS = ("e3x_barnard", "e3x_epsindib", "e7_epsindib", "e7_rdra", "e7_rwaur")   # E8/E12 fields; e7_* duplicate e3x_*
AVOID = [(269.448, 4.739, "Barnard's Star (E8)"), (331.096, -56.801, "Epsilon Indi B (E12)")]


def sep_deg(ra1, dec1, ra2, dec2):
    a1, d1, a2, d2 = map(np.radians, (ra1, dec1, ra2, dec2))
    return np.degrees(np.arccos(np.clip(np.sin(d1) * np.sin(d2) + np.cos(d1) * np.cos(d2) * np.cos(a1 - a2), -1, 1)))


# ------------------------------------------------------------------ calibration sample
def select():
    import pyvo
    avoid = list(AVOID) + [(c["ra"], c["dec"], f"E8 comparison cone {c['cone']}") for c in json.load(open("data/phase2/e8/comp_stars.json"))["cones"]]
    used_ids = {s["source_id"] for s in json.load(open("data/phase2/e8/comp_stars.json"))["stars"]}
    used_ids |= {s["source_id"] for s in json.load(open("data/phase2/e8/controls.json"))["stars"]}
    used_ids |= {s["source_id"] for s in json.load(open("data/phase2/e12/null_stars.json"))["stars"]}
    fields, rej = [], []
    for name in sorted(os.listdir(EF)):
        d = f"{EF}/{name}"
        if not os.path.exists(f"{d}/READY") or not os.path.exists(f"{d}/field.json"):
            continue
        fj = json.load(open(f"{d}/field.json"))
        near = [lab for ra, dec, lab in avoid if sep_deg(fj["ra"], fj["dec"], ra, dec) < 3.0]
        if name in EXCLUDE_FIELDS or near:
            rej.append(dict(field=name, reason="E8/E12 field or duplicate" if name in EXCLUDE_FIELDS else f"within 3 deg of {near}"))
            continue
        fields.append(dict(field=name, ra=fj["ra"], dec=fj["dec"], n_frames=fj["n_frames"]))
    rows = []
    for f in fields:
        g = pd.read_parquet(f"{EF}/{f['field']}/gaia.parquet")
        cd = np.cos(np.radians(f["dec"]))
        ra26 = g.ra + g.pmra.fillna(0) * 10 / 3.6e6 / np.cos(np.radians(g.dec))
        de26 = g.dec + g.pmdec.fillna(0) * 10 / 3.6e6
        r = 3600 * np.hypot((ra26 - f["ra"]) * cd, de26 - f["dec"]) / PIX
        ok = (g.phot_g_mean_mag >= G_MIN) & (g.phot_g_mean_mag <= G_MAX) & (r <= R_MAX_PX) & g.pmra.notna() & g.parallax.notna()
        for i in g.index[ok]:
            s = 3600 * np.hypot((ra26 - ra26[i]) * cd, de26 - de26[i])
            iso = not ((s > 0) & (s < 15) & (g.phot_g_mean_mag < g.phot_g_mean_mag[i] + 2)).any()
            nobright = not ((s > 0) & (s < 120) & (g.phot_g_mean_mag < 9)).any()
            rows.append(dict(field=f["field"], source_id=str(int(g.source_id[i])), ra=float(g.ra[i]), dec=float(g.dec[i]), pmra=float(g.pmra[i]),
                             pmdec=float(g.pmdec[i]), parallax=float(g.parallax[i]), G=float(g.phot_g_mean_mag[i]), r_px=float(r[i]),
                             isolated=bool(iso), no_bright=bool(nobright)))
    c = pd.DataFrame(rows)
    p = f"{OUT}/ruwe.parquet"
    if not os.path.exists(p):
        ids = ",".join(c.source_id)
        t = pyvo.dal.TAPService(TAP).run_async(f"SELECT source_id, ruwe, bp_rp FROM gaiadr3.gaia_source WHERE source_id IN ({ids})").to_table().to_pandas()
        t["source_id"] = t.source_id.astype("int64").astype(str)
        t.to_parquet(p)
    c = c.merge(pd.read_parquet(p), on="source_id", how="left")
    c["eligible"] = c.isolated & c.no_bright & (c.ruwe < RUWE_MAX) & ~c.source_id.isin(used_ids)
    el = c[c.eligible].sort_values("source_id").reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    half = np.empty(len(el), object)
    half[rng.permutation(len(el))] = np.where(np.arange(len(el)) < (len(el) + 1) // 2, "A", "B")
    el["half"] = half
    rec = dict(seed=SEED, rule=dict(fields=f"every READY cached field in {EF} except {list(EXCLUDE_FIELDS)} and any field within 3 deg of Barnard's Star, "
                                           "Epsilon Indi B or an E8 comparison-star cone",
                                    stars=f"Gaia DR3 G {G_MIN}-{G_MAX}, RUWE < {RUWE_MAX}, pmra/pmdec/parallax present, within {R_MAX_PX} px of the field "
                                          "centre (J2026 position), no other source with G < G_star + 2 within 15 arcsec, no G < 9 source within 2 arcmin, "
                                          "not an E8 comparison/control star or E12 null star",
                                    split="eligible stars sorted by source_id; default_rng(16).permutation -> first half of the permutation = half A, rest = half B"),
               fields=fields, rejected_fields=rej, n_candidates=int(len(c)), n_eligible=int(len(el)),
               n_A=int((el.half == "A").sum()), n_B=int((el.half == "B").sum()), stars=el.to_dict("records"))
    json.dump(rec, open(f"{OUT}/calib_stars.json", "w"), indent=1)
    print(len(fields), "fields;", len(c), "candidates;", len(el), "eligible", el.groupby("field").size().to_dict())


# ------------------------------------------------------------------ per-picture positions
def epsf_grid(st, xs, ys, nb_xy, bits=e8.MASK, h=15):
    """Line-for-line the computation of run_e8.est_epsf (estimator (a) = the measure_motion.one fit: ePSF + Gaia
    neighbours + plane, chi2 grid +-1.5 px step 0.05, hard-bad + OVERFLOW masked). Returns the grid minimum (identical
    to est_epsf; checked in stage `check`) and, for diagnostics only, a parabola-refined off-grid minimum."""
    from orrery.psf import psf_model
    c = (st.image.shape[0] - 1) // 2
    sl = (slice(c - h, c + h + 1), slice(c - h, c + h + 1))
    img, var = st.image[sl].astype(float), st.variance[sl].astype(float)
    m = e8._valid(st, bits)[sl]
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
    offs = np.linspace(-e8.BOX, e8.BOX, int(round(2 * e8.BOX / e8.STEP_A)) + 1)
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
    rx, ry = (0.0, 0.0) if edge else e8._quad_refine(surf, j, i, e8.STEP_A)
    return xs + offs[i], ys + offs[j], dict(snr=float(snr), edge=bool(edge), chi2_dof=float(surf[j, i] / max(m.sum() - U.shape[1] - 1, 1)),
                                            refined_dx=float(rx), refined_dy=float(ry), n_nb=len(nb))


def jacobian(wcs, x, y):
    """Arcsec east / north per +1 px step along detector x and y at full-frame pixel (x, y)."""
    p = wcs.all_pix2world([[x, y], [x + 1, y], [x, y + 1]], 0)
    cd = np.cos(np.radians(p[0][1]))
    return ((p[1][0] - p[0][0]) * cd * 3600, (p[1][1] - p[0][1]) * 3600, (p[2][0] - p[0][0]) * cd * 3600, (p[2][1] - p[0][1]) * 3600)


def ecl_east_pa(ra, dec):
    """Unit vector (east, north) of increasing ecliptic longitude at (ra, dec)."""
    from astropy.coordinates import SkyCoord, GeocentricTrueEcliptic
    import astropy.units as u
    c = SkyCoord(ra * u.deg, dec * u.deg).transform_to(GeocentricTrueEcliptic())
    c2 = SkyCoord(c.lon + 0.01 * u.deg, c.lat, frame=GeocentricTrueEcliptic()).icrs
    de, dn = (c2.ra.deg - ra) * np.cos(np.radians(dec)), c2.dec.deg - dec
    n = np.hypot(de, dn)
    return float(de / n), float(dn / n)


def epsf_zone(release, det, x, y):
    """orrery.data.Frame.epsf_at(round(x), round(y)): the ePSF zone nearest the star's pixel."""
    from orrery.data import _calib
    c = _calib(release, det)
    i = np.argmin((c["xc"] - int(round(x))) ** 2 + (c["yc"] - int(round(y))) ** 2)
    return np.asarray(c["epsf"][i], float)


def measure_picture(a):
    """One cached-field picture: read a 101 x 101 cut-out at the field centre, measure every calibration star in it."""
    import e2_cache
    field, fra, fdec, frow, stars = a
    try:
        st = e2_cache.fast_stamp(frow["key"], fra, fdec, half=HALF)
    except Exception as e:
        return [dict(field=field, key=frow["key"], error=repr(e)[:120])]
    if st is None:
        return []
    wcs = WCS(fits.Header.fromstring(st.wcs_header))
    g = pd.read_parquet(f"{EF}/{field}/gaia.parquet")
    dt = Time(st.mjd, format="mjd").jyear - 2016.0
    gx, gy = wcs.all_world2pix(np.c_[g.ra.values + g.pmra.fillna(0).values * dt / 3.6e6 / np.cos(np.radians(g.dec.values)),
                                     g.dec.values + g.pmdec.fillna(0).values * dt / 3.6e6], 0).T
    rows = []
    for s in stars:
        ra, dec = e8.predict(s, st.mjd)                       # Gaia DR3 at the picture time, with parallax (Earth barycentric)
        x, y = wcs.all_world2pix([[float(ra[0]), float(dec[0])]], 0)[0]
        s3 = e8.substamp(st, x, y, 15)
        if s3 is None or not (20 <= x <= 2019 and 20 <= y <= 2019):
            continue
        s3 = replace(s3, epsf=epsf_zone(st.release, st.detector, x, y))   # ePSF zone at the star, as Frame.stamp / fast_stamp give it
        try:
            xa, ya, q = epsf_grid(s3, x - s3.x0, y - s3.y0, list(zip(gx - s3.x0, gy - s3.y0)))
        except Exception as e:
            rows.append(dict(field=field, key=frow["key"], source_id=s["source_id"], error=repr(e)[:120]))
            continue
        jex, jnx, jey, jny = jacobian(wcs, x, y)
        dx, dy = xa - (x - s3.x0), ya - (y - s3.y0)
        rows.append(dict(field=field, key=frow["key"], obs_id=st.obs_id, source_id=s["source_id"], G=s["G"], half=s["half"], mjd=st.mjd,
                         det=st.detector, x=float(x), y=float(y), wl=float(s3.cwave[15, 15]), dx=float(dx), dy=float(dy),
                         rdx=float(dx + q["refined_dx"]), rdy=float(dy + q["refined_dy"]), snr=q["snr"], edge=q["edge"], chi2_dof=q["chi2_dof"],
                         n_nb=q["n_nb"], j_ex=jex, j_nx=jnx, j_ey=jey, j_ny=jny,
                         de=jex * dx + jey * dy, dn=jnx * dx + jny * dy))
    return rows


def measure(workers=6):
    cs = json.load(open(f"{OUT}/calib_stars.json"))
    os.makedirs(f"{OUT}/calib_meas", exist_ok=True)
    stars = pd.DataFrame(cs["stars"])
    for f in cs["fields"]:
        p = f"{OUT}/calib_meas/{f['field']}.parquet"
        ss = stars[stars.field == f["field"]]
        if os.path.exists(p) or not len(ss):
            continue
        fl = sorted(json.load(open(f"{EF}/{f['field']}/field.json"))["frame_list"], key=lambda r: r["mjd"])
        if len(fl) > N_PIC:
            fl = [fl[i] for i in np.unique(np.linspace(0, len(fl) - 1, N_PIC).round().astype(int))]
        sl = [{k: r[k] for k in ("source_id", "ra", "dec", "pmra", "pmdec", "parallax", "G", "half")} for r in ss.to_dict("records")]
        t = time.time()
        with ProcessPoolExecutor(workers) as ex:
            rows = [r for rr in ex.map(measure_picture, [(f["field"], f["ra"], f["dec"], fr, sl) for fr in fl]) for r in rr]
        df = pd.DataFrame(rows)
        df.to_parquet(p)
        print(f["field"], len(fl), "pictures", len(sl), "stars", len(df), "rows", int(df.get("error", pd.Series(dtype=str)).notna().sum()), "errors",
              f"{time.time() - t:.0f}s", flush=True)


def repair(workers=6):
    """Re-read pictures whose pixel read failed (transient network errors) and replace their error rows."""
    cs = json.load(open(f"{OUT}/calib_stars.json"))
    stars = pd.DataFrame(cs["stars"])
    for f in cs["fields"]:
        p = f"{OUT}/calib_meas/{f['field']}.parquet"
        if not os.path.exists(p):
            continue
        df = pd.read_parquet(p)
        if "error" not in df or not df.error.notna().any():
            continue
        bad = sorted(df.key[df.error.notna()].unique())
        fl = {r["key"]: r for r in json.load(open(f"{EF}/{f['field']}/field.json"))["frame_list"]}
        sl = [{k: r[k] for k in ("source_id", "ra", "dec", "pmra", "pmdec", "parallax", "G", "half")} for r in stars[stars.field == f["field"]].to_dict("records")]
        with ProcessPoolExecutor(workers) as ex:
            rows = [r for rr in ex.map(measure_picture, [(f["field"], f["ra"], f["dec"], fl[k], sl) for k in bad]) for r in rr]
        df = pd.concat([df[~df.key.isin(bad)], pd.DataFrame(rows)], ignore_index=True)
        df.to_parquet(p)
        print(f["field"], len(bad), "pictures re-read;", int(df.error.notna().sum()) if "error" in df else 0, "errors left", flush=True)


def check(n_star=4, n_pic=6):
    """The calibration positions must be the E12 and E8 estimators: re-measure some calibration star-pictures with
    run_e12.measure_one (= measure_motion.one, E12's pipeline, its own pixel read and Gaia table) and with run_e8.est_epsf."""
    import run_e12 as e12
    e12.OUT = f"{OUT}/check"                    # its per-star Gaia tables go here, nothing of E12 is touched
    os.makedirs(f"{OUT}/check/gaia", exist_ok=True)
    m = pd.concat([pd.read_parquet(f"{OUT}/calib_meas/{f}") for f in sorted(os.listdir(f"{OUT}/calib_meas"))[:6]])
    m = m.dropna(subset=["dx"])
    stars = {s["source_id"]: s for s in json.load(open(f"{OUT}/calib_stars.json"))["stars"]}
    rng = np.random.default_rng(SEED)
    out = []
    for sid in rng.choice(sorted(m.source_id.unique()), n_star, replace=False):
        q = m[m.source_id == sid]
        for r in q.iloc[np.sort(rng.choice(len(q), min(n_pic, len(q)), replace=False))].itertuples():
            s = stars[sid]
            g = dict(ra=s["ra"], dec=s["dec"], pmra=s["pmra"], pmdec=s["pmdec"], parallax=s["parallax"], own_table=True)
            a = e12.measure_one((r.key, sid, g, r.mjd))
            de12 = (a["ra_meas"] - a["ra_pred"]) * 3600 * np.cos(np.radians(a["dec_pred"]))
            dn12 = (a["dec_meas"] - a["dec_pred"]) * 3600
            import e2_cache
            st = e2_cache.fast_stamp(r.key, a["ra_pred"], a["dec_pred"], half=15)
            wcs = WCS(fits.Header.fromstring(st.wcs_header))
            x, y = wcs.all_world2pix([[a["ra_pred"], a["dec_pred"]]], 0)[0]
            gt = pd.read_parquet(f"{EF}/{r.field}/gaia.parquet")
            dt = Time(st.mjd, format="mjd").jyear - 2016.0
            gx, gy = wcs.all_world2pix(np.c_[gt.ra.values + gt.pmra.fillna(0).values * dt / 3.6e6 / np.cos(np.radians(gt.dec.values)),
                                             gt.dec.values + gt.pmdec.fillna(0).values * dt / 3.6e6], 0).T
            xa, ya, _ = e8.est_epsf(st, x - st.x0, y - st.y0, list(zip(gx - st.x0, gy - st.y0)))
            out.append(dict(source_id=sid, key=r.key, e16_dx=r.dx, e16_dy=r.dy, e8a_dx=float(xa - (x - st.x0)), e8a_dy=float(ya - (y - st.y0)),
                            e16_de=r.de, e16_dn=r.dn, e12_de=float(de12), e12_dn=float(dn12)))
            print(out[-1], flush=True)
    d = pd.DataFrame(out)
    res = dict(n=len(d), max_abs_diff_px_e16_vs_e8a=float(np.nanmax(np.abs(np.r_[d.e16_dx - d.e8a_dx, d.e16_dy - d.e8a_dy]))),
               max_abs_diff_arcsec_e16_vs_e12=float(np.nanmax(np.abs(np.r_[d.e16_de - d.e12_de, d.e16_dn - d.e12_dn]))), rows=out)
    json.dump(res, open(f"{OUT}/estimator_check.json", "w"), indent=1)
    print({k: v for k, v in res.items() if k != "rows"})


# ------------------------------------------------------------------ model: detector-frame offset, polynomial in detector (x, y)
CANDIDATES = [(0, False), (0, True), (1, False), (1, True), (2, False), (2, True)]   # (degree, per orientation class)
N_FOLD = 5
CLIP_PX = 0.025            # floor of the robust sigma (px) in the model fit's one 5-sigma clip (half the 0.05 px grid)
MIN_WIN = 5                # pictures needed for a window mean


def basis(x, y, deg):
    u, v = (np.asarray(x, float) - 1019.5) / 1020.0, (np.asarray(y, float) - 1019.5) / 1020.0
    cols = [np.ones_like(u)]
    if deg >= 1:
        cols += [u, v]
    if deg >= 2:
        cols += [u * u, u * v, v * v]
    return np.stack(cols, 1)


def n_params(deg, seas):
    return 6 * 2 * (1, 3, 6)[deg] * (2 if seas else 1)


def usable(m):
    return m[np.isfinite(m.dx) & np.isfinite(m.dy) & (m.snr > SNR_MIN) & ~m.edge.astype(bool)].copy()


def add_orientation(m, stars):
    """Orientation class of each picture: 1 if detector +x points within 90 deg of increasing ecliptic longitude at the
    star, else 0 (SPHEREx's orientation on the sky reverses between visits half a year apart)."""
    ee = {sid: ecl_east_pa(s["ra"], s["dec"]) for sid, s in stars.items()}
    e = np.array([ee[s] for s in m.source_id])
    m["season"] = ((m.j_ex.values * e[:, 0] + m.j_nx.values * e[:, 1]) > 0).astype(int)
    m["ang_east_deg"] = np.degrees(np.arctan2(m.j_nx.values, m.j_ex.values))
    return m


def fit_model(m, deg, seas):
    """Per detector (and orientation class if seas): least squares of the pixel offsets dx, dy on the polynomial,
    one 5-robust-sigma clip (floor CLIP_PX), refit."""
    model = dict(deg=deg, season=bool(seas), coef={})
    for d in range(1, 7):
        for s in ((0, 1) if seas else (None,)):
            q = m[(m.det == d) & ((m.season == s) if s is not None else True)]
            if len(q) < 20:
                continue
            B = basis(q.x, q.y, deg)
            cf = {}
            for c in ("dx", "dy"):
                v = q[c].values
                p = np.linalg.lstsq(B, v, rcond=None)[0]
                r = v - B @ p
                keep = np.abs(r - np.median(r)) <= 5 * max(e8.rob(r), CLIP_PX)
                cf[c] = np.linalg.lstsq(B[keep], v[keep], rcond=None)[0].tolist()
            model["coef"][f"{d}" + ("" if s is None else f"_{s}")] = cf
    return model


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


def corrected_sky(m, model):
    if model is None:
        return m.de.values, m.dn.values
    mx, my = predict_px(model, m.det.values, m.x.values, m.y.values, m.season.values)
    dx, dy = m.dx.values - mx, m.dy.values - my
    return m.j_ex.values * dx + m.j_ey.values * dy, m.j_nx.values * dx + m.j_ny.values * dy


def window_means(m, model):
    """Per star and visit window (gaps > 30 d): mean east/north offset from Gaia after one 5-robust-sigma clip per
    coordinate (floor 0.155 arcsec), pictures kept in both coordinates; windows with >= MIN_WIN usable pictures."""
    de, dn = corrected_sky(m, model)
    m = m.assign(cde=de, cdn=dn)
    rows = []
    for sid, g in m.groupby("source_id"):
        w = e8.windows(g.mjd.values)
        for wi in np.unique(w):
            s = w == wi
            if s.sum() < MIN_WIN:
                continue
            _, _, ke = e8.clip_mean(g.cde.values[s], floor=GRID_FLOOR)
            _, _, kn = e8.clip_mean(g.cdn.values[s], floor=GRID_FLOOR)
            k = ke & kn
            rows.append(dict(source_id=sid, window=int(wi), n=int(k.sum()), mjd=float(g.mjd.values[s][k].mean()), G=float(g.G.values[0]),
                             de=float(g.cde.values[s][k].mean()), dn=float(g.cdn.values[s][k].mean()),
                             season_frac=float(g.season.values[s][k].mean()), ang=float(np.median(g.ang_east_deg.values[s][k]))))
    return pd.DataFrame(rows)


def rms_wm(wm):
    return float(np.sqrt(np.mean(np.r_[wm.de, wm.dn] ** 2)))


def w2w(wm):
    dif = []
    for _, g in wm.groupby("source_id"):
        g = g.sort_values("window")
        dif += list(np.diff(g.de.values)) + list(np.diff(g.dn.values))
    return float(np.sqrt(np.mean(np.array(dif) ** 2)) / np.sqrt(2)) if dif else np.nan


def load_calib():
    cs = json.load(open(f"{OUT}/calib_stars.json"))
    stars = {s["source_id"]: s for s in cs["stars"]}
    m = pd.concat([pd.read_parquet(f"{OUT}/calib_meas/{f}") for f in sorted(os.listdir(f"{OUT}/calib_meas")) if f.endswith(".parquet")])
    m = usable(m.dropna(subset=["source_id"]))
    m = add_orientation(m, stars)
    m["px"], m["py"] = m.x - np.round(m.x), m.y - np.round(m.y)   # sub-pixel phase of the Gaia position, -0.5..0.5
    return cs, stars, m.reset_index(drop=True)


def cell_stats(m, by, cols=("dx", "dy")):
    """Mean of star-level cell means (clipped) per group, error = SD / sqrt(number of stars)."""
    out = []
    for key, g in m.groupby(by):
        cm = [[e8.clip_mean(q[c].values, floor=CLIP_PX)[0] for c in cols] for _, q in g.groupby("source_id") if len(q) >= 3]
        if len(cm) < 3:
            continue
        cm = np.array(cm)
        out.append(dict(group=key if not isinstance(key, tuple) else list(key), n_stars=len(cm), n_pictures=int(len(g)),
                        **{f"{c}_mean": float(cm[:, i].mean()) for i, c in enumerate(cols)},
                        **{f"{c}_err": float(cm[:, i].std(ddof=1) / np.sqrt(len(cm))) for i, c in enumerate(cols)}))
    return out


def diagnostics(m):
    m = m.copy()
    m["gbin"] = pd.cut(m.G, [11, 13, 14, 15, 16.01], right=False).astype(str)
    m["pxbin"] = pd.cut(m.px, np.linspace(-0.5, 0.5, 6)).astype(str)
    m["pybin"] = pd.cut(m.py, np.linspace(-0.5, 0.5, 6)).astype(str)
    m["xbin"] = pd.cut(m.x, [0, 680, 1360, 2040]).astype(str)
    m["ybin"] = pd.cut(m.y, [0, 680, 1360, 2040]).astype(str)
    m["wlbin"] = m.groupby("det").wl.transform(lambda v: pd.qcut(v, 3, labels=["short", "mid", "long"]).astype(str))
    m["grid_minus_refined_x"], m["grid_minus_refined_y"] = m.dx - m.rdx, m.dy - m.rdy
    return dict(
        by_detector=cell_stats(m, "det"), by_detector_orientation=cell_stats(m, ["det", "season"]),
        by_detector_x_third=cell_stats(m, ["det", "xbin"]), by_detector_y_third=cell_stats(m, ["det", "ybin"]),
        by_detector_wavelength_third=cell_stats(m, ["det", "wlbin"]),
        wavelength_ranges_um={int(d): [float(g.wl.min()), float(g.wl.max())] for d, g in m.groupby("det")},
        by_G=cell_stats(m, "gbin"), by_G_and_orientation=cell_stats(m, ["gbin", "season"]),
        by_subpixel_phase_x=cell_stats(m, "pxbin", ("dx", "rdx")), by_subpixel_phase_y=cell_stats(m, "pybin", ("dy", "rdy")),
        grid_vs_refined=dict(mean_dx_grid=float(m.dx.mean()), mean_dx_refined=float(m.rdx.mean()), mean_dy_grid=float(m.dy.mean()),
                             mean_dy_refined=float(m.rdy.mean()), sd_grid_minus_refined_px=float(np.r_[m.grid_minus_refined_x, m.grid_minus_refined_y].std()),
                             note="refined = parabola through the 3 x 3 chi2 values around the grid minimum (diagnostic only; the estimator is the grid minimum)"),
        per_picture_scatter_px=dict(dx=float((m.dx - m.groupby("source_id").dx.transform("mean")).std()), dy=float((m.dy - m.groupby("source_id").dy.transform("mean")).std()),
                                    note="standard deviation about each star's mean (the robust MAD collapses to 0 on the 0.05 px grid)"),
        orientation_angles_deg={int(s): [float(np.percentile(g.ang_east_deg, 5)), float(np.median(g.ang_east_deg)), float(np.percentile(g.ang_east_deg, 95))]
                                for s, g in m.groupby("season")})


def model():
    cs, stars, m = load_calib()
    A, B = m[m.half == "A"], m[m.half == "B"]
    n_pic_det = {int(d): int(n) for d, n in m.groupby("det").size().items()}
    res = dict(n_stars_measured=int(m.source_id.nunique()), n_pictures_used=int(len(m)), n_distinct_pictures=int(m.key.nunique()),
               n_distinct_pictures_per_detector={int(d): int(g.key.nunique()) for d, g in m.groupby("det")}, n_star_pictures_per_detector=n_pic_det,
               n_stars_A=int(A.source_id.nunique()), n_stars_B=int(B.source_id.nunique()), n_fields=int(m.field.nunique()))
    res["diagnostics"] = diagnostics(m)
    json.dump(res, open(f"{OUT}/model.json", "w"), indent=1)
    # model selection on half A: 5-fold cross-validation over A's stars
    ids = np.array(sorted(A.source_id.unique()))
    fold = dict(zip(ids[np.random.default_rng(SEED).permutation(len(ids))], np.arange(len(ids)) % N_FOLD))
    fA = A.source_id.map(fold).values
    cv = {"none": rms_wm(window_means(A, None))}
    for deg, seas in CANDIDATES:
        wm = pd.concat([window_means(A[fA == k], fit_model(A[fA != k], deg, seas)) for k in range(N_FOLD)])
        cv[f"deg{deg}" + ("_per_orientation" if seas else "")] = rms_wm(wm)
    order = sorted(CANDIDATES, key=lambda c: (n_params(*c), c[0]))
    best = order[0]
    name = lambda c: f"deg{c[0]}" + ("_per_orientation" if c[1] else "")  # noqa: E731
    for c in order[1:]:
        if cv[name(c)] < 0.99 * cv[name(best)]:
            best = c
    res["selection_on_A"] = dict(rule=f"{N_FOLD}-fold cross-validation over half-A stars (folds from default_rng({SEED}) permutation of the sorted ids); "
                                      "metric = RMS of held-out residual window means; candidates in order of parameter count, a larger model replaces "
                                      "the current choice only if it lowers the metric by > 1%",
                                 cv_rms_arcsec=cv, chosen=name(best))
    mdl = fit_model(A, *best)
    # star-bootstrap uncertainty of the coefficients (half A, 200 draws)
    rng = np.random.default_rng(SEED)
    gA = {s: g for s, g in A.groupby("source_id")}
    bs = []
    for _ in range(200):
        pick = rng.choice(ids, len(ids), replace=True)
        bs.append(fit_model(pd.concat([gA[s] for s in pick]), *best)["coef"])
    mdl["coef_err"] = {k: {c: np.std([b[k][c] for b in bs if k in b], axis=0, ddof=1).tolist() for c in ("dx", "dy")} for k in mdl["coef"]}
    res["model"] = mdl
    # validation on half B
    wb0, wb1 = window_means(B, None), window_means(B, mdl)
    r0, r1 = rms_wm(wb0), rms_wm(wb1)
    red = 1 - r1 / r0
    res["validation_on_B"] = dict(n_star_windows=int(len(wb0)), n_stars=int(wb0.source_id.nunique()), rms_before_arcsec=r0, rms_after_arcsec=r1,
                                  reduction=red, threshold=0.20, passed=bool(red >= 0.20),
                                  window_to_window_before_arcsec=w2w(wb0), window_to_window_after_arcsec=w2w(wb1),
                                  mean_offset_before=[float(wb0.de.mean()), float(wb0.dn.mean())], mean_offset_after=[float(wb1.de.mean()), float(wb1.dn.mean())],
                                  all_candidates_on_B_descriptive={name(c): 1 - rms_wm(window_means(B, fit_model(A, *c))) / r0 for c in CANDIDATES})
    res["verdict_calibration"] = "PASSED" if red >= 0.20 else "FAILED"
    wb0.assign(de_after=wb1.de.values, dn_after=wb1.dn.values).to_parquet(f"{OUT}/validation_windows_B.parquet")
    json.dump(res, open(f"{OUT}/model.json", "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("selection_on_A", "validation_on_B", "verdict_calibration")}, indent=1))
    print(json.dumps(res["model"], indent=1)[:3000])


# ------------------------------------------------------------------ re-tests (only after docs/protocols/E16.md is frozen)
def frozen_model():
    r = json.load(open(f"{OUT}/model.json"))
    assert r["verdict_calibration"] == "PASSED", "calibration failed validation: not applied"
    return r["model"]


def season_of(jex, jnx, ra, dec):
    e = ecl_east_pa(ra, dec)
    return (np.asarray(jex) * e[0] + np.asarray(jnx) * e[1] > 0).astype(int)


def correct_radec(wcs, det, ra, dec, mdl, ra_star, dec_star):
    """Measured sky position -> detector pixel -> subtract the model offset -> back to the sky (same WCS)."""
    x, y = wcs.all_world2pix([[ra, dec]], 0)[0]
    jex, jnx, _, _ = jacobian(wcs, x, y)
    s = season_of(jex, jnx, ra_star, dec_star)
    mx, my = predict_px(mdl, [det], [x], [y], [s])
    r2, d2 = wcs.all_pix2world([[x - mx[0], y - my[0]]], 0)[0]
    return float(r2), float(d2), float(x), float(y), int(s), float(mx[0]), float(my[0])


def wcs_for_keys(keys, field_dir, cache_dir):
    """Full-frame WCS of each picture: from the cached field cut-outs when present, else the FITS header (2 range reads)."""
    import e2_cache
    from orrery.data import BUCKET
    os.makedirs(cache_dir, exist_ok=True)
    fj = json.load(open(f"{field_dir}/field.json"))
    have = {f["key"]: f"{field_dir}/{f['file']}" for f in fj["frame_list"]}
    out = {}
    for k in keys:
        p = f"{cache_dir}/{k.split('/')[-1].replace('.fits', '')}.hdr"
        if not os.path.exists(p):
            if k in have:
                txt = load_stamp(have[k]).wcs_header
            else:
                url = f"{BUCKET}/{k}"
                _, o = e2_cache._header(url, 0)
                h, _ = e2_cache._header(url, o)
                txt = WCS(h).to_header(relax=True).tostring()
            open(p, "w").write(txt)
        out[k] = WCS(fits.Header.fromstring(open(p).read()))
    return out


def retest_e12(mdl):
    import shutil
    import run_e12 as e12
    d = f"{OUT}/e12"
    os.makedirs(d, exist_ok=True)
    shutil.copy("data/phase2/e12/null_stars.json", f"{d}/null_stars.json")
    tables = {"target": pd.read_parquet(e12.TARGET_PARQUET)}
    stars = json.load(open("data/phase2/e12/null_stars.json"))["stars"]
    for s in stars:
        tables[s["source_id"]] = pd.read_parquet(f"data/phase2/e12/null_{s['source_id']}.parquet")
    keys = sorted(set().union(*[set(t.key) for t in tables.values()]))
    W = wcs_for_keys(keys, "data/efields/e3x_epsindib", f"{OUT}/e12_wcs")
    pos = {"target": (e12.GAIA_TARGET["ra"], e12.GAIA_TARGET["dec"])} | {s["source_id"]: (s["ra"], s["dec"]) for s in stars}
    corr = []
    for name, t in tables.items():
        t = t.copy()
        for i in t.index[t.ra_meas.notna()]:
            r2, d2, x, y, s, mx, my = correct_radec(W[t.key[i]], int(t.det[i]), t.ra_meas[i], t.dec_meas[i], mdl, *pos[name])
            corr.append(dict(star=name, key=t.key[i], mjd=float(t.mjd[i]), det=int(t.det[i]), x=x, y=y, season=s, mx=mx, my=my,
                             d_east_arcsec=(r2 - t.ra_meas[i]) * 3600 * np.cos(np.radians(d2)), d_north_arcsec=(d2 - t.dec_meas[i]) * 3600))
            t.loc[i, "ra_meas"], t.loc[i, "dec_meas"] = r2, d2
        t.to_parquet(f"{d}/target_corrected.parquet" if name == "target" else f"{d}/null_{name}.parquet")
    pd.DataFrame(corr).to_parquet(f"{OUT}/e12_corrections.parquet")
    e12.OUT, e12.TARGET_PARQUET = d, f"{d}/target_corrected.parquet"
    out = {}
    for lab, floor in (("headline_clip_floor_0p155", e12.GRID_FLOOR), ("clip_as_frozen", 0.0)):
        a = e12.analyse(floor)
        ns = a["null_sum"]
        gp, ge = e12.GAIA_TARGET["parallax"], e12.GAIA_TARGET["parallax_error"]
        crit = dict(null_mean_within_2sigma_of_zero=bool(abs(ns["mean_mas"]) <= 2 * ns["mean_err_mas"]),
                    target_ge_3sigma_from_zero=bool(a["sig0"] >= 3), target_within_2p5sigma_of_gaia=bool(abs(a["sigg"]) <= 2.5))
        out[lab] = dict(parallax_mas=a["plx"], parallax_err_mas=dict(fit_scaled=a["e_fit"], bootstrap=a["e_boot"], null_rms=a["e_null"], adopted=a["e"]),
                        sigma_from_zero=a["sig0"], sigma_from_gaia=a["sigg"], distance_pc=1000 / a["plx"],
                        proper_motion_mas_yr=[1000 * a["r"]["params"]["pmra_arcsec_yr"], 1000 * a["r"]["params"]["pmdec_arcsec_yr"]],
                        proper_motion_err_mas_yr=[1000 * a["r"]["errors"]["pmra_arcsec_yr"], 1000 * a["r"]["errors"]["pmdec_arcsec_yr"]],
                        fit_windows=a["r"]["windows"], bootstrap=a["r"]["bootstrap"],
                        null_summary=ns, null_stars=[{k: v.get(k) for k in ("source_id", "G", "used", "n_used", "parallax_mas", "parallax_err_mas",
                                                                             "d_parallax_mas", "d_pmra_mas_yr", "d_pmdec_mas_yr", "window_offsets_from_gaia_arcsec")}
                                                     for v in a["null"]],
                        criteria=crit, verdict="PASSED" if all(crit.values()) and ns["n_used"] >= 3 else "FAILED")
    c = pd.DataFrame(corr)
    out["correction_applied"] = dict(n_positions=int(len(c)), median_abs_shift_arcsec=float(np.median(np.hypot(c.d_east_arcsec, c.d_north_arcsec))),
                                     by_orientation={int(s): dict(n=int(len(g)), mean_d_east_arcsec=float(g.d_east_arcsec.mean()), mean_d_north_arcsec=float(g.d_north_arcsec.mean()))
                                                     for s, g in c.groupby("season")})
    return out


def retest_e8(mdl):
    ev = json.load(open("data/phase2/e8/comp_eval.json"))
    fz = json.load(open("data/phase2/e8/frozen.json"))
    assert fz["estimator"] == "d+cal"
    comp = {s["source_id"]: s for s in json.load(open("data/phase2/e8/comp_stars.json"))["stars"]}
    B = e8.BARNARD
    # comparison stars (estimator d): pixel offsets corrected, then d+cal and the per-window error re-derived as in E8 evaluate()
    m = pd.read_parquet("data/phase2/e8/comp_meas.parquet")
    ee = {sid: ecl_east_pa(comp[sid]["ra"], comp[sid]["dec"]) for sid in m.source_id.unique()}
    e_ = np.array([ee[s] for s in m.source_id])
    seas = ((m.j_ex.values * e_[:, 0] + m.j_nx.values * e_[:, 1]) > 0).astype(int)
    mx, my = predict_px(mdl, m.det.values, m.x_full.values, m.y_full.values, seas)
    m["d_dx"], m["d_dy"] = m.d_dx - mx, m.d_dy - my
    met = e8.metrics(m, "d", "loo")
    wt, _ = e8.window_table(m, "d", "loo")
    br = wt[wt.source_id.map({k: v["K_pred"] for k, v in comp.items()}) <= 4.7]
    sc_br = float(np.sqrt(np.mean(np.r_[br.de - wt.de.mean(), br.dn - wt.dn.mean()] ** 2)))
    sig_w = max(met["window_scatter_arcsec"], met["window_to_window_arcsec"], sc_br)
    cal = e8.det_offsets(m, "d")
    comp_res = dict(window_scatter_arcsec=met["window_scatter_arcsec"], window_to_window_arcsec=met["window_to_window_arcsec"],
                    window_scatter_bright_arcsec=sc_br, per_window_error_arcsec=sig_w, detector_offsets_px=cal,
                    registered=dict(per_window_error_arcsec=fz["per_window_error_arcsec"], detector_offsets_px=ev["detector_offsets_px"]["d"]),
                    model_shift_px_mean=[float(mx.mean()), float(my.mean())])
    # Barnard's Star (estimator d, fixed start point), identical fit
    bm = pd.read_parquet("data/phase2/e8/barnard_meas.parquet").dropna(subset=["mjd"]).reset_index(drop=True)
    sb = season_of(bm.j_ex.values, bm.j_nx.values, B["ra"], B["dec"])
    bx, by = predict_px(mdl, bm.det.values, bm.x_full.values, bm.y_full.values, sb)
    bm["d_dx"], bm["d_dy"] = bm.d_dx - bx, bm.d_dy - by
    de, dn = e8.sky_offsets(bm, "d", cal)
    ok = np.isfinite(de) & np.isfinite(dn)
    mjd = bm.mjd.values[ok]
    pa, pd_ = e8.plx_factors(B["ra"], B["dec"], mjd)
    E, N = de[ok] - B["parallax"] / 1000 * pa, dn[ok] - B["parallax"] / 1000 * pd_
    rows, keep = e8.window_means(mjd, E, N)
    t = np.array([r["t"] for r in rows])
    fit = e8.linfit_windows(t, np.array([r["east"] for r in rows]), np.array([r["north"] for r in rows]), np.full(len(rows), sig_w))
    gE, gN = B["pmra"] / 1000, B["pmdec"] / 1000
    mE, mN, sE, sN = fit["east"]["pm_arcsec_yr"], fit["north"]["pm_arcsec_yr"], fit["east"]["pm_err"], fit["north"]["pm_err"]
    tot = float(np.hypot(mE, mN))
    stot = float(np.sqrt((mE * sE) ** 2 + (mN * sN) ** 2) / tot)
    bar = dict(n_used=int(keep.sum()), windows=rows, fit=fit, pmra_arcsec_yr=mE, pmra_err=sE, pmdec_arcsec_yr=mN, pmdec_err=sN, total_arcsec_yr=tot,
               total_err=stot, total_sigma=tot / stot, sigma_from_gaia=dict(pmra=(mE - gE) / sE, pmdec=(mN - gN) / sN))
    # controls (estimator a): sky offsets -> detector pixels (each picture's WCS at the control) -> corrected -> sky
    fj = json.load(open(f"{e8.BFIELD}/field.json"))
    W = {}
    for f in fj["frame_list"]:
        st = load_stamp(f"{e8.BFIELD}/{f['file']}")
        W[(round(st.mjd, 6), st.detector)] = WCS(fits.Header.fromstring(st.wcs_header))
    ctl = pd.read_parquet("data/phase2/e8/barnard_controls.parquet")
    cinfo = {c["source_id"]: c for c in fz["controls"]}
    new, diag = [], []
    for r in ctl.itertuples():
        c = cinfo[r.source_id]
        w = W[(round(r.mjd, 6), int(r.det))]
        ra, dec = e8.predict(c, r.mjd)
        x, y = w.all_world2pix([[float(ra[0]), float(dec[0])]], 0)[0]
        jex, jnx, jey, jny = jacobian(w, x, y)
        J = np.array([[jex, jey], [jnx, jny]])
        dx, dy = np.linalg.solve(J, [r.de, r.dn])
        s = int(season_of(jex, jnx, c["ra"], c["dec"]))
        cx, cy = predict_px(mdl, [r.det], [x], [y], [s])
        dx2, dy2 = dx - cx[0], dy - cy[0]
        new.append(dict(r._asdict(), de=float(jex * dx2 + jey * dy2), dn=float(jnx * dx2 + jny * dy2)))
        diag.append(dict(source_id=r.source_id, mjd=r.mjd, det=int(r.det), snr=r.snr, edge=r.edge, season=s, ang=float(np.degrees(np.arctan2(jnx, jex))),
                         dx_before=float(dx), dy_before=float(dy), dx_after=float(dx2), dy_after=float(dy2), de_before=r.de, dn_before=r.dn,
                         de_after=new[-1]["de"], dn_after=new[-1]["dn"]))
    ctl2 = pd.DataFrame(new).drop(columns=["Index"])
    pd.DataFrame(diag).to_parquet(f"{OUT}/e8_controls_corrected.parquet")
    cres = []
    for c in fz["controls"]:
        q = ctl2[(ctl2.source_id == c["source_id"]) & (ctl2.snr > 10) & ~ctl2.edge]
        if len(q) < 20:
            cres.append(dict(source_id=c["source_id"], G=c["G"], used=False, n=int(len(q))))
            continue
        cr, _ = e8.window_means(q.mjd.values, q.de.values, q.dn.values, sig_floor=e8.GRID_FLOOR)
        if len(cr) < 2:
            cres.append(dict(source_id=c["source_id"], G=c["G"], used=False, n=int(len(q)), reason="< 2 windows"))
            continue
        tt = np.array([r["t"] for r in cr])
        fe = e8.linfit_windows(tt, np.array([r["east"] for r in cr]), np.array([r["north"] for r in cr]),
                               np.array([np.hypot(r["internal_err_east"], r["internal_err_north"]) / np.sqrt(2) for r in cr]))
        sg = [fe["east"]["pm_arcsec_yr"] / fe["east"]["pm_err"], fe["north"]["pm_arcsec_yr"] / fe["north"]["pm_err"]]
        cres.append(dict(source_id=c["source_id"], G=c["G"], used=True, n=int(len(q)), n_windows=len(cr), d_pmra_arcsec_yr=fe["east"]["pm_arcsec_yr"],
                         d_pmra_err=fe["east"]["pm_err"], d_pmdec_arcsec_yr=fe["north"]["pm_arcsec_yr"], d_pmdec_err=fe["north"]["pm_err"], sigma=sg,
                         consistent=bool(max(abs(sg[0]), abs(sg[1])) <= 2.5),
                         windows=[{kk: r[kk] for kk in ("mjd", "n", "east", "north", "internal_err_east", "internal_err_north")} for r in cr]))
    used = [c for c in cres if c.get("used")]
    n_ok = sum(c["consistent"] for c in used)
    dra = np.array([c["d_pmra_arcsec_yr"] for c in used])
    ddec = np.array([c["d_pmdec_arcsec_yr"] for c in used])
    crit = dict(controls_ge_6_of_8_within_2p5sigma=bool(n_ok >= 6),
                barnard_components_within_2p5sigma=bool(abs(bar["sigma_from_gaia"]["pmra"]) <= 2.5 and abs(bar["sigma_from_gaia"]["pmdec"]) <= 2.5),
                barnard_total_ge_5sigma=bool(bar["total_sigma"] >= 5))
    return dict(comparison_stars=comp_res, barnard=bar,
                controls=dict(stars=cres, n_used=len(used), n_consistent=int(n_ok), mean_d_pmra_arcsec_yr=float(dra.mean()), sd_d_pmra=float(dra.std(ddof=1)),
                              mean_d_pmdec_arcsec_yr=float(ddec.mean()), sd_d_pmdec=float(ddec.std(ddof=1))),
                criteria=crit, verdict="PASSED" if all(crit.values()) else "FAILED")


def retest():
    fz = open("docs/protocols/E16.md").read()
    assert "FROZEN" in fz, "freeze docs/protocols/E16.md first"
    mdl = frozen_model()
    t0 = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    r12 = retest_e12(mdl)
    json.dump(r12, open(f"{OUT}/retest_e12.json", "w"), indent=1, default=float)
    r8 = retest_e8(mdl)
    json.dump(r8, open(f"{OUT}/retest_e8.json", "w"), indent=1, default=float)
    print("started", t0)
    print(json.dumps({k: r12[k]["criteria"] | dict(verdict=r12[k]["verdict"], plx=r12[k]["parallax_mas"], err=r12[k]["parallax_err_mas"]["adopted"],
                                                     null_mean=r12[k]["null_summary"]["mean_mas"], null_err=r12[k]["null_summary"]["mean_err_mas"])
                      for k in ("headline_clip_floor_0p155", "clip_as_frozen")}, indent=1, default=float))
    print(json.dumps(dict(criteria=r8["criteria"], verdict=r8["verdict"], controls={k: v for k, v in r8["controls"].items() if k != "stars"},
                          barnard={k: r8["barnard"][k] for k in ("pmra_arcsec_yr", "pmra_err", "pmdec_arcsec_yr", "pmdec_err", "total_sigma", "sigma_from_gaia")},
                          per_window_error=r8["comparison_stars"]["per_window_error_arcsec"]), indent=1, default=float))


def findings():
    """data/findings/e16.json: calibration sample, model, validation, re-tests, verdicts, comparison with the registered runs."""
    cs = json.load(open(f"{OUT}/calib_stars.json"))
    mres = json.load(open(f"{OUT}/model.json"))
    e8j, e12j = json.load(open("data/findings/e8.json")), json.load(open("data/findings/e12.json"))
    res = dict(experiment="E16", label="re-test after calibration (the registered E8 and E12 verdicts are unchanged)",
               protocol="docs/PROTOCOLS.md PHASE 3 E16; frozen choices docs/protocols/E16.md", code="scripts/run_e16.py",
               calibration_sample=dict(rule=cs["rule"], seed=cs["seed"], n_fields=len(cs["fields"]), n_eligible=cs["n_eligible"],
                                       fields=[f["field"] for f in cs["fields"]], rejected_fields=cs["rejected_fields"],
                                       **{k: mres[k] for k in ("n_stars_measured", "n_stars_A", "n_stars_B", "n_pictures_used", "n_distinct_pictures",
                                                               "n_distinct_pictures_per_detector", "n_star_pictures_per_detector")},
                                       stars_file=f"{OUT}/calib_stars.json", measurements=f"{OUT}/calib_meas/"),
               estimator_check=json.load(open(f"{OUT}/estimator_check.json")) | dict(rows=None),
               diagnostics=mres["diagnostics"], model_selection_on_A=mres["selection_on_A"], model=mres["model"],
               validation_on_B=mres["validation_on_B"], calibration_verdict=mres["verdict_calibration"], written=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    if os.path.exists(f"{OUT}/retest_e12.json"):
        r12, r8 = json.load(open(f"{OUT}/retest_e12.json")), json.load(open(f"{OUT}/retest_e8.json"))
        h = r12["headline_clip_floor_0p155"]
        res["e12_retest"] = r12
        res["e8_retest"] = r8
        res["verdicts_E16"] = dict(E16_E12=h["verdict"], E16_E12_clip_as_frozen=r12["clip_as_frozen"]["verdict"], E16_E8=r8["verdict"])
        res["comparison_with_registered"] = dict(
            E12=dict(registered_verdict=e12j["verdict"], parallax_mas=[e12j["parallax_mas"], h["parallax_mas"]],
                     adopted_err_mas=[e12j["parallax_err_mas"]["adopted"], h["parallax_err_mas"]["adopted"]],
                     null_mean_mas=[e12j["null"]["summary"]["mean_mas"], h["null_summary"]["mean_mas"]],
                     null_mean_err_mas=[e12j["null"]["summary"]["mean_err_mas"], h["null_summary"]["mean_err_mas"]],
                     null_rms_mas=[e12j["null"]["summary"]["rms_about_zero_mas"], h["null_summary"]["rms_about_zero_mas"]],
                     n_null_negative=[e12j["null"]["summary"]["n_negative"], h["null_summary"]["n_negative"]], order="[registered, after calibration]"),
            E8=dict(registered_verdict=e8j["verdict"], barnard_pm_arcsec_yr=[[e8j["pmra_arcsec_yr"], e8j["pmdec_arcsec_yr"]], [r8["barnard"]["pmra_arcsec_yr"], r8["barnard"]["pmdec_arcsec_yr"]]],
                    barnard_err=[e8j["pmra_err"], r8["barnard"]["pmra_err"]], controls_mean_d_pmra=[e8j["controls"]["summary"]["mean_d_pmra_arcsec_yr"], r8["controls"]["mean_d_pmra_arcsec_yr"]],
                    controls_n_beyond_2p5sigma=[e8j["controls"]["summary"]["n_beyond_2p5_sigma"], r8["controls"]["n_used"] - r8["controls"]["n_consistent"]],
                    order="[registered, after calibration]"))
    else:
        res["verdicts_E16"] = dict(note="re-tests not run: the calibration failed validation, so by the frozen rule it is not applied")
    json.dump(res, open("data/findings/e16.json", "w"), indent=1, default=float)
    print(json.dumps(res["verdicts_E16"], indent=1))


def figure():
    """Calibration result (the re-tests were not run: the calibration failed its validation)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 14, "axes.titlesize": 15, "axes.labelsize": 14, "legend.fontsize": 12, "figure.facecolor": "white", "axes.facecolor": "white"})
    cb, ca = "#8a8f98", "#2a78d6"
    mres = json.load(open(f"{OUT}/model.json"))
    mdl, v = mres["model"], mres["validation_on_B"]
    _, _, m = load_calib()
    B = m[m.half == "B"].copy()
    mx, my = predict_px(mdl, B.det.values, B.x.values, B.y.values, B.season.values)
    B["dx_after"] = B.dx - mx
    fig, ax = plt.subplots(2, 2, figsize=(16, 12), dpi=110)

    def cells(q, col):
        cm = [e8.clip_mean(g[col].values, floor=CLIP_PX)[0] for _, g in q.groupby("source_id") if len(g) >= 3]
        return (1000 * np.mean(cm), 1000 * np.std(cm, ddof=1) / np.sqrt(len(cm))) if len(cm) > 2 else (np.nan, np.nan)
    # (a) offset along detector x, per detector and orientation (held-out half B), before / after
    a = ax[0, 0]
    for d in range(1, 7):
        for s, mk in ((0, "s"), (1, "o")):
            q = B[(B.det == d) & (B.season == s)]
            for col, c, off in (("dx", cb, -0.06), ("dx_after", ca, 0.06)):
                mu, se = cells(q, col)
                a.errorbar(d + (s - 0.5) * 0.36 + off, mu, se, fmt=mk, color=c, ms=8, capsize=3)
    a.axhline(0, color="k", lw=0.8)
    a.set_xticks(range(1, 7))
    a.set_xlabel("detector (square: detector x toward ecliptic west; circle: toward east)")
    a.set_ylabel("star position minus Gaia along detector x\n(thousandths of a pixel; 1 px = 6.15 arcsec)")
    a.set_title("Held-out check stars: a small fixed offset in the\ndetector's own frame, the same for both orientations")
    a.plot([], [], "s", color=cb, label="measured"); a.plot([], [], "s", color=ca, label="after subtracting the calibration")
    a.legend(frameon=False, loc="upper left")
    # (b) east-west offset of visit averages vs orientation (half B)
    a = ax[0, 1]
    wb = pd.read_parquet(f"{OUT}/validation_windows_B.parquet")
    bins = np.arange(-180, 181, 45)
    cen = 0.5 * (bins[1:] + bins[:-1])
    for col, c, lab, off in (("de", cb, "measured", -3), ("de_after", ca, "after subtracting the calibration", 3)):
        xs_, mu, lo_, hi_ = [], [], [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            q = wb[col][(wb.ang >= lo) & (wb.ang < hi)].values * 1000
            if len(q) < 20:
                continue
            bs = [np.median(np.random.default_rng(SEED + i).choice(q, len(q))) for i in range(300)]
            xs_.append(0.5 * (lo + hi) + off); mu.append(np.median(q)); lo_.append(np.median(q) - np.percentile(bs, 16)); hi_.append(np.percentile(bs, 84) - np.median(q))
        a.errorbar(xs_, mu, [lo_, hi_], fmt="o", color=c, ms=9, capsize=3, label=lab)
    tt = np.linspace(-180, 180, 200)
    a.plot(tt, 1000 * PIX * float(np.mean([mdl["coef"][k]["dx"][0] for k in mdl["coef"]])) * np.cos(np.radians(tt)), "--", color="#c0392b", lw=1.5,
           label="expected from a fixed +0.007 px offset along detector x")
    a.axhline(0, color="k", lw=0.8)
    a.set_xlabel("direction of the detector's x axis on the sky (degrees from east)")
    a.set_ylabel("east-west offset from Gaia, median of visit\naverages (thousandths of an arcsecond)")
    a.set_title("Held-out check stars: east-west offset against the\ntelescope's orientation (bins with >= 20 visits)")
    a.legend(frameon=False, loc="best")
    # (c) sub-pixel phase
    a = ax[1, 0]
    ph = mres["diagnostics"]["by_subpixel_phase_x"]
    order = np.argsort([float(p["group"].split(",")[0].strip("(")) for p in ph])
    xc = [np.mean([float(t) for t in ph[i]["group"].strip("(]").split(",")]) for i in order]
    a.errorbar(np.array(xc) - 0.01, [1000 * ph[i]["dx_mean"] for i in order], [1000 * ph[i]["dx_err"] for i in order], fmt="s-", color=cb, ms=8, capsize=3,
               label="estimator as used (0.05 px grid)")
    a.errorbar(np.array(xc) + 0.01, [1000 * ph[i]["rdx_mean"] for i in order], [1000 * ph[i]["rdx_err"] for i in order], fmt="o-", color=ca, ms=8, capsize=3,
               label="same, refined between grid points")
    a.axhline(0, color="k", lw=0.8)
    a.set_xlabel("where Gaia puts the star inside a pixel (fraction of a pixel along x)")
    a.set_ylabel("offset along detector x\n(thousandths of a pixel)")
    a.set_title("All check stars: the offset depends on where the star falls\ninside a pixel; the 0.05 px grid itself makes no difference")
    a.legend(frameon=False, loc="best")
    # (d) the validation: visit averages are dominated by star-to-star offsets
    a = ax[1, 1]
    r = np.r_[wb.de, wb.dn] * 1000
    r2 = np.r_[wb.de_after, wb.dn_after] * 1000
    bb = np.linspace(-800, 800, 65)
    a.hist(np.clip(r, -799, 799), bb, color=cb, alpha=0.6, label=f"measured: RMS {1000 * v['rms_before_arcsec']:.0f}")
    a.hist(np.clip(r2, -799, 799), bb, histtype="step", lw=2.2, color=ca, label=f"after calibration: RMS {1000 * v['rms_after_arcsec']:.0f}")
    a.axvspan(-45, 45, color="#f2b134", alpha=0.25, label="size of the correction (about 45)")
    a.set_xlabel("visit-average offset from Gaia (thousandths of an arcsec)")
    a.set_ylabel("number of star-visits (held-out half)")
    a.set_title(f"Pre-registered test: RMS must drop by at least 20%;\nit dropped by {100 * v['reduction']:.1f}% -> calibration FAILED, not applied")
    a.legend(frameon=False, loc="upper right", fontsize=11)
    for a_ in ax.ravel():
        a_.spines[["top", "right"]].set_visible(False)
        a_.grid(alpha=0.25, lw=0.5)
    fig.suptitle("E16: calibrating a tiny detector-frame position offset on 471 independent stars\n"
                 "the offset is real (about +0.007 pixel along detector x) but the pre-registered validation failed; "
                 "E8/E12 re-tests not run, registered verdicts unchanged", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    os.makedirs("site/data/figs", exist_ok=True)
    fig.savefig("site/data/figs/e16_calibration.png", facecolor="white")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    globals()[sys.argv[1]](*map(int, sys.argv[2:]))
