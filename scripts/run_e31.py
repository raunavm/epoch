"""E31 (docs/protocols/E31.md): our forced-ePSF fluxes against IRSA's published SPHEREx aperture-photometry recipe
(IRSA SDT tutorial, aperture_photometry.py) on the same Level-2 pictures, and the live site's 3x3 quick look
run as the site's own JavaScript in node. Stages (each writes to data/phase2/e31/):
  select   5 E18 control stars + picture draw            -> sample.json, ours_e18.parquet
  fetch    41x41 stamps of the E18 pictures, SAPM files   -> stamps/<id>/*.npz, ref/
  measure  IRSA-SDT aperture photometry on every picture  -> irsa.parquet
  ql       site quick look (node, site/js/live/grid.js)    -> ql.parquet
  analyze  binned comparison, quick-look loss table       -> data/findings/e31.json, e31s.json
  fig      data/findings/e31_compare.png/.svg
Usage: OMP_NUM_THREADS=1 PYTHONPATH=<photutils dir> .venv/bin/python scripts/run_e31.py <stage>
"""
import glob, json, os, re, shutil, subprocess, sys, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
warnings.filterwarnings("ignore")

D = "data/phase2/e31"
SEED = 2026111531
N_PER_DET = 24
J2016_MJD = 57388.5
SAPM = "dr1/solid_angle_pixel_map/cal-sapm-v3-2026-191/{d}/solid_angle_pixel_map_D{d}_spx_cal-sapm-v3-2026-191.fits"
ARCSEC2_TO_SR = (np.pi / (180.0 * 3600.0)) ** 2
RADII = [3.0, 4.0, 5.0]
R_PRIMARY = 5.0
BAD_FLAGS = {"TRANSIENT": 0, "OVERFLOW": 1, "SUR_ERROR": 2, "NONFUNC": 6, "DICHROIC": 7, "MISSING_DATA": 9, "HOT": 10,
             "COLD": 11, "NONLINEAR": 15, "CROSSTALK": 20, "GHOST": 22, "GHOST_FPA": 23, "GHOST_EXT": 24,
             "STREAK": 25, "PERSIST": 17, "BLOOM": 26, "SNOWBALL": 27, "HALO": 28, "SATELLITE_HALO": 29}
LOG_EDGES = (0.75 * (5.0 / 0.75) ** (np.arange(39) / 38))[::2]
W_EDGES = np.round(3.8 + 0.1 * np.arange(13), 1)
DET_RANGE = {1: (0.75, 1.10), 2: (1.10, 1.64), 3: (1.64, 2.42), 4: (2.42, 3.82), 5: (3.82, 4.42), 6: (4.42, 5.00)}
JS_SRC = ["site/js/live/grid.js", "site/js/live/wcs.js"]


def propagate(ra, dec, pmra, pmdec, mjd):
    dt = (mjd - J2016_MJD) / 365.25
    pmra, pmdec = np.nan_to_num(pmra), np.nan_to_num(pmdec)
    return ra + pmra * dt / 3.6e6 / np.cos(np.radians(dec)), dec + pmdec * dt / 3.6e6


def parse_key(key):
    m = re.search(r"level2_(.+)D(\d)_spx", key)
    return m.group(1), int(m.group(2))


# ------------------------------------------------------------------ select
def stage_select():
    import pyarrow.dataset as ds
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    S = pd.read_parquet("data/phase2/e18/sources.parquet")
    const = S[(S.kind == "gaia") & (S.G < 19.5) & (S.phot_variable_flag != "VARIABLE") & S.vari_class.isna()
              & (S.ruwe < 1.4) & (S.d_nearest_arcsec.fillna(99) > 15) & (S.blend_n == 1) & S.in_region]
    c = const[(const.G >= 12) & (const.G <= 15) & (const.d_nearest_arcsec.fillna(99) > 35)
              & const.d_brighter_arcsec.isna() & (const.neowise_var_kim2026 != True) & const.milliquas_name.isna()]
    V = pd.read_parquet("data/phase2/e18a/vsx_raw.parquet")
    sc, vc = SkyCoord(c.ra.values * u.deg, c.dec.values * u.deg), SkyCoord(V.RAJ2000.values * u.deg, V.DEJ2000.values * u.deg)
    sep = sc.match_to_catalog_sky(vc)[1].arcsec
    elig = c[sep > 5].copy()
    elig = elig.iloc[np.argsort(elig.id.astype(np.int64).values, kind="stable")].reset_index(drop=True)
    perm = np.random.default_rng(SEED).permutation(len(elig))
    pick = elig.iloc[perm[:5]].reset_index(drop=True)
    print(len(const), len(c), len(elig), pick[["id", "G", "bp_rp", "d_nearest_arcsec"]].to_string())
    ids = pick.id.tolist()
    t = ds.dataset("data/phase2/e18/phot", format="parquet").to_table(filter=ds.field("id").isin(ids)).to_pandas()
    usable = np.isfinite(t.flux_mJy) & (t.n_masked_box <= 3) & (t.flags_core >= 0) & ((t.flags_core & 2) == 0)
    t = t[usable & (t.edge_px >= 22)]
    F = pd.read_parquet("data/phase2/e18/frames.parquet", columns=["key", "obs_id", "detector"])
    t = t.merge(F, on=["obs_id", "detector"], how="left")
    rows = []
    for k, sid in enumerate(ids):
        g = t[t.id == sid].sort_values(["detector", "mjd", "obs_id"], kind="stable")
        rng = np.random.default_rng(SEED + k)
        for det, gd in g.groupby("detector"):
            sel = gd.iloc[rng.permutation(len(gd))[:N_PER_DET]]
            rows.append(sel)
    R = pd.concat(rows).reset_index(drop=True)
    os.makedirs(D, exist_ok=True)
    R.to_parquet(f"{D}/ours_e18.parquet")
    json.dump(dict(seed=SEED, n_const=int(len(const)), n_after_cuts=int(len(c)), n_eligible=int(len(elig)),
                   stars=pick[["id", "ra", "dec", "pmra", "pmdec", "G", "bp_rp", "ruwe", "d_nearest_arcsec", "Jmag", "Kmag",
                               "W1mag", "W2mag"]].to_dict("records"),
                   n_pictures={sid: int((R.id == sid).sum()) for sid in ids}, n_usable_rows={sid: int((t.id == sid).sum()) for sid in ids}),
              open(f"{D}/sample.json", "w"), indent=1, default=float)
    print(R.groupby(["id", "detector"]).size().unstack())


# ------------------------------------------------------------------ fetch
def stage_fetch():
    from concurrent.futures import ThreadPoolExecutor
    import requests
    from e2_cache import fast_stamp
    from orrery.data import save_stamp, BUCKET
    os.makedirs(f"{D}/ref", exist_ok=True)
    for d in range(1, 7):
        p = f"{D}/ref/sapm_D{d}.fits"
        if not os.path.exists(p):
            r = requests.get(f"{BUCKET}/{SAPM.format(d=d)}", timeout=120)
            r.raise_for_status()
            open(p + ".tmp", "wb").write(r.content)
            os.replace(p + ".tmp", p)
    R = pd.read_parquet(f"{D}/ours_e18.parquet")
    star = {s["id"]: s for s in json.load(open(f"{D}/sample.json"))["stars"]}

    def one(r):
        p = f"{D}/stamps/{r.id}/{r.obs_id}_D{r.detector}.npz"
        if os.path.exists(p):
            return 0
        s = star[r.id]
        ra, dec = propagate(s["ra"], s["dec"], s["pmra"], s["pmdec"], r.mjd)
        st = fast_stamp(r.key, ra, dec, half=20)
        if st is None:
            return 1
        os.makedirs(os.path.dirname(p), exist_ok=True)
        save_stamp(st, p + ".tmp.npz")
        os.replace(p + ".tmp.npz", p)
        return 0
    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(one, R.itertuples()))
    print("fetched; off-frame:", sum(res))


# ------------------------------------------------------------------ target lists (stamp path, sky position, ours)
def targets():
    """Rows: target, stamp, ra, dec, mjd, det, obs_id, f_ours, e_ours, wl_ours."""
    out = []
    smp = json.load(open(f"{D}/sample.json"))
    star = {s["id"]: s for s in smp["stars"]}
    qc = json.load(open("data/phase2/e18/qc.json"))["a_error_scale"]["two_param_model"]["per_detector"]
    R = pd.read_parquet(f"{D}/ours_e18.parquet")
    for r in R.itertuples():
        s = star[r.id]
        ra, dec = propagate(s["ra"], s["dec"], s["pmra"], s["pmdec"], r.mjd)
        m = qc[str(int(r.detector))]
        e = float(np.hypot(m["a"] * r.flux_err_mJy, m["b_frac"] * r.flux_mJy))
        out.append(dict(target=f"gaia{r.id}", stamp=f"{D}/stamps/{r.id}/{r.obs_id}_D{r.detector}.npz", ra=ra, dec=dec,
                        mjd=r.mjd, det=int(r.detector), obs_id=r.obs_id, f_ours=r.flux_mJy, e_ours=e, wl_ours=r.wl_um))
    # Epsilon Indi B (E7)
    e7 = json.load(open("data/findings/e7_runs/e7_epsindib.json"))["phot"]["target"]
    G = pd.read_parquet("data/efields/e3x_epsindib/gaia.parquet")
    g = G[G.source_id == 6412596012146801152].iloc[0]
    for i, key in enumerate(e7["key"]):
        if e7["sat"][i] or e7["npix"][i] < 18 or not np.isfinite(e7["flux"][i]):
            continue
        obs, det = parse_key(key)
        ra, dec = propagate(g.ra, g.dec, g.pmra, g.pmdec, e7["mjd"][i])
        out.append(dict(target="epsindib", stamp=f"data/efields/e3x_epsindib/frames/{obs}_D{det}.npz", ra=ra, dec=dec,
                        mjd=e7["mjd"][i], det=det, obs_id=obs, f_ours=e7["flux"][i], e_ours=e7["err"][i], wl_ours=e7["lam"][i]))
    # WISE 0855 (E4 cache, E11 rows)
    for r in json.load(open("data/phase2/e11/wise0855_frames.json")):
        if r["flux"] is None or not np.isfinite(r["flux"]) or not (r["chi2_local_dof"] <= 3):
            continue
        e4 = json.load(open(f"data/forced/wise0855_e4/{r['obs_id']}_D{r['det']}.json"))
        out.append(dict(target="wise0855", stamp=f"data/efields/e3x_wise0855/frames/{r['obs_id']}_D{r['det']}.npz",
                        ra=e4["pred_ra"], dec=e4["pred_dec"], mjd=r["mjd"], det=r["det"], obs_id=r["obs_id"],
                        f_ours=r["flux"], e_ours=r["err"], wl_ours=r["wl"]))
    T = pd.DataFrame(out)
    return T[T.stamp.map(os.path.exists)].reset_index(drop=True), len(T)


# ------------------------------------------------------------------ measure: IRSA SDT recipe
def irsa_phot(st, ra, dec, sapm):
    """aperture_photometry.py build_phot_table + extract_photometry + extract_dqf for one source in one picture."""
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.coordinates import SkyCoord
    from astropy.stats import sigma_clipped_stats
    from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
    from scipy.ndimage import map_coordinates
    wcs = WCS(fits.Header.fromstring(st.wcs_header))
    xf, yf = wcs.world_to_pixel(SkyCoord(ra, dec, unit="deg", frame="icrs"))
    x, y = float(xf) - st.x0, float(yf) - st.y0
    H, W = st.image.shape
    scale = sapm[st.y0:st.y0 + H, st.x0:st.x0 + W] * ARCSEC2_TO_SR * 1e6 * 1e3
    image = st.image.astype(float) * scale
    var = st.variance.astype(float) * scale ** 2
    pos = [(x, y)]
    bad = ~np.isfinite(image)
    m = CircularAnnulus(pos, r_in=7, r_out=16).to_mask(method="center")[0]
    cut = m.multiply(image)
    vals = cut[m.data > 0]
    vals = vals[np.isfinite(vals)]
    _, med, _ = sigma_clipped_stats(vals, sigma=3, maxiters=5)
    out = dict(x=x, y=y, bkg_med=float(med))
    for r in RADII:
        ap = CircularAperture(pos, r=r)
        s = aperture_photometry(image, ap, mask=bad, method="subpixel", subpixels=5)["aperture_sum"][0]
        out[f"f_r{r:.0f}"] = float(s - med * ap.area)
        out[f"e_r{r:.0f}"] = float(np.sqrt(aperture_photometry(var, ap, mask=~np.isfinite(var), method="subpixel",
                                                               subpixels=5)["aperture_sum"][0]))
        nb = 0
        for b in BAD_FLAGS.values():
            nb += aperture_photometry(((st.flags & (1 << b)) > 0).astype(float), ap, method="center")["aperture_sum"][0]
        out[f"nbad_r{r:.0f}"] = float(nb)
    out["wl"] = float(map_coordinates(st.cwave.astype(float), [[y], [x]], order=1)[0])
    return out


def stage_measure():
    from astropy.io import fits
    from orrery.data import load_stamp
    T, n_all = targets()
    print("pictures with stamps:", len(T), "of", n_all, T.groupby("target").size().to_dict())
    sapm = {d: fits.getdata(f"{D}/ref/sapm_D{d}.fits", "IMAGE").astype(float) for d in range(1, 7)}
    rows = []
    for i, r in enumerate(T.itertuples()):
        st = load_stamp(r.stamp)
        o = irsa_phot(st, r.ra, r.dec, sapm[r.det])
        rows.append({**r._asdict(), **o, "omega": st.omega_arcsec2})
        if i % 200 == 0:
            print(i, flush=True)
    pd.DataFrame(rows).drop(columns=["Index"]).to_parquet(f"{D}/irsa.parquet")


# ------------------------------------------------------------------ quick look: the site's JS in node
NODE_MAIN = """import { resample, aperture, toMJy, gridSize } from './grid.js';
import { makeWCS } from './wcs.js';
import fs from 'fs';
const out = [];
for (const L of fs.readFileSync(process.argv[2], 'utf8').trim().split('\\n')) {
  const r = JSON.parse(L), wcs = makeWCS(r.hdr);
  const rows = { y0: 0, y1: r.H - 1, nx: r.W, data: Float32Array.from(r.data.map(x => x === null ? NaN : x)) };
  const res = { i: r.i };
  for (const fov of [2, 4]) {
    const S = gridSize(fov), c = (S - 1) / 2, { v } = resample(wcs, rows, null, S, r.ra, r.dec);
    const a = aperture(v, S, Math.round(c), Math.round(c));
    res['ql' + fov] = toMJy(a.flux, r.omega); res['ql' + fov + '_err'] = toMJy(a.err, r.omega); res['S' + fov] = S;
  }
  out.push(res);
}
fs.writeFileSync(process.argv[3], JSON.stringify(out));
"""


def stage_ql():
    from astropy.io import fits
    from orrery.data import load_stamp
    jd = f"{D}/js"
    os.makedirs(jd, exist_ok=True)
    for s in JS_SRC:
        shutil.copyfile(s, f"{jd}/{os.path.basename(s)}")
    json.dump({"type": "module"}, open(f"{jd}/package.json", "w"))
    open(f"{jd}/main.js", "w").write(NODE_MAIN)
    I = pd.read_parquet(f"{D}/irsa.parquet")
    with open(f"{D}/ql_in.jsonl", "w") as f:
        for i, r in enumerate(I.itertuples()):
            st = load_stamp(r.stamp)
            h = fits.Header.fromstring(st.wcs_header)
            hdr = {k: h[k] for k in h if isinstance(h[k], (int, float)) and not isinstance(h[k], bool)}
            hdr["CRPIX1"] -= st.x0
            hdr["CRPIX2"] -= st.y0
            H, W = st.image.shape
            data = [None if not np.isfinite(v) else float(v) for v in st.image.ravel()]
            f.write(json.dumps(dict(i=i, hdr=hdr, H=H, W=W, data=data, ra=r.ra, dec=r.dec, omega=st.omega_arcsec2)) + "\n")
    subprocess.run(["node", f"{jd}/main.js", f"{D}/ql_in.jsonl", f"{D}/ql_out.json"], check=True)
    Q = pd.DataFrame(json.load(open(f"{D}/ql_out.json"))).set_index("i").sort_index()
    for c in Q.columns:
        I[c] = pd.to_numeric(Q[c], errors="coerce").values
    I.to_parquet(f"{D}/ql.parquet")
    os.remove(f"{D}/ql_in.jsonl")


# ------------------------------------------------------------------ analyze
def rmad(x):
    x = np.asarray(x, float)
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) else np.nan


def compare(g, edges, fcol="f_r5", ecol="e_r5"):
    bins, chi2, chi2p, ratios = [], 0.0, 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = g[(g.wl >= lo) & (g.wl < hi)]
        n = len(b)
        if n < 3:
            continue
        fo, fi = b.f_ours.values, b[fcol].values
        mo, mi = fo.mean(), fi.mean()
        so, si = fo.std(ddof=1) / np.sqrt(n), fi.std(ddof=1) / np.sqrt(n)
        sp = (fo - fi).std(ddof=1) / np.sqrt(n)
        c2, c2p = (mo - mi) ** 2 / (so ** 2 + si ** 2), (mo - mi) ** 2 / sp ** 2 if sp > 0 else np.nan
        chi2 += c2
        chi2p += c2p
        det_ok = mi >= 5 * si
        R = mo / mi if det_ok else np.nan
        if det_ok:
            ratios.append(R)
        bins.append(dict(lo=float(lo), hi=float(hi), centre=float(b.wl.mean()), n=n, f_ours=float(mo), e_ours=float(so),
                         f_irsa=float(mi), e_irsa=float(si), e_paired=float(sp), ratio=float(R) if det_ok else None,
                         ratio_err=float(abs(R) * np.hypot(so / mo, si / mi)) if det_ok else None, chi2=float(c2)))
    nb = len(bins)
    med = float(np.median(ratios)) if ratios else np.nan
    sc = rmad(ratios)
    res = dict(n_pictures=int(len(g)), n_bins=nb, n_ratio_bins=len(ratios), median_ratio=med, robust_scatter=sc,
               median_ratio_err=float(1.2533 * sc / np.sqrt(len(ratios))) if ratios else None,
               chi2=float(chi2), dof=nb, chi2_dof=float(chi2 / nb) if nb else None,
               chi2_paired=float(chi2p), chi2_dof_paired=float(chi2p / nb) if nb else None,
               per_picture_median_ratio=float(np.median(g.f_ours / g[fcol])), bins=bins)
    res["agrees"] = bool(nb > 0 and len(ratios) > 0 and abs(med - 1) <= 0.05 and chi2 / nb < 2)
    return res


def stage_analyze():
    Q = pd.read_parquet(f"{D}/ql.parquet")
    smp = json.load(open(f"{D}/sample.json"))
    ok = Q[np.isfinite(Q.f_r5) & (Q.nbad_r5 == 0) & np.isfinite(Q.f_ours)].copy()
    out = dict(experiment="E31", protocol="docs/protocols/E31.md", generated=pd.Timestamp.now(tz="US/Pacific").strftime("%Y-%m-%d %H:%M %Z"),
               irsa_access=dict(
                   tool="IRSA SPHEREx Spectrophotometry Tool (Tractor forced photometry), SPHEREx Data Explorer web app",
                   programmatic_api_found=False, irsa_tap_status_2026_11_15="HTTP 502 at 20:12 PST",
                   used="IRSA SDT tutorial aperture-photometry recipe (irsa-tutorials, aperture_photometry.py, "
                        "commit 6615d7b 2026-10-28), run by us on the same DR1 Level-2 pictures; r=5 px, annulus 7-16 px, DR1 SAPM"),
               units=dict(wavelength="um", flux="mJy"), sample=smp, targets={})
    loss_cut = {}
    for tgt, g in Q.groupby("target"):
        gg = ok[ok.target == tgt]
        edges = W_EDGES if tgt == "wise0855" else LOG_EDGES
        res = compare(gg, edges)
        res["n_pictures_measured"] = int(len(g))
        res["n_rejected_irsa_flags"] = int((np.isfinite(g.f_r5) & (g.nbad_r5 > 0)).sum())
        res["radius_r3"] = {k: v for k, v in compare(gg, edges, "f_r3", "e_r3").items() if k != "bins"}
        res["radius_r4"] = {k: v for k, v in compare(gg, edges, "f_r4", "e_r4").items() if k != "bins"}
        res["max_abs_wl_diff_um"] = float(np.nanmax(np.abs(gg.wl - gg.wl_ours))) if len(gg) else None
        out["targets"][tgt] = res
    stars = [t for t in out["targets"] if t.startswith("gaia")]
    out["verdict"] = "PASS" if all(out["targets"][t]["agrees"] for t in stars) else "FAILED"
    out["n_stars_agree"] = int(sum(out["targets"][t]["agrees"] for t in stars))
    # quick-look loss table (stars only)
    S_ = Q[Q.target.str.startswith("gaia") & np.isfinite(Q.f_r5) & (Q.nbad_r5 == 0) & (Q.f_r5 >= 10 * Q.e_r5)].copy()
    for fov in (2, 4):
        tab = {}
        for d, gd in S_.groupby("det"):
            ent = {}
            for ref, col in (("irsa_r5", "f_r5"), ("ours", "f_ours")):
                r = (gd[f"ql{fov}"] / gd[col]).replace([np.inf, -np.inf], np.nan).dropna()
                lo, hi = DET_RANGE[int(d)]
                thirds = []
                for k in range(3):
                    a, b = lo + (hi - lo) * k / 3, lo + (hi - lo) * (k + 1) / 3
                    rr = r[(gd.wl.loc[r.index] >= a) & (gd.wl.loc[r.index] < b)]
                    thirds.append(dict(lo=round(a, 3), hi=round(b, 3), n=int(len(rr)), median=float(rr.median()) if len(rr) else None,
                                       rscatter=rmad(rr), correction=float(1 / rr.median()) if len(rr) else None))
                ent[ref] = dict(n=int(len(r)), median=float(r.median()), rscatter=rmad(r),
                                median_err=float(1.2533 * rmad(r) / np.sqrt(len(r))), correction=float(1 / r.median()), thirds=thirds)
            tab[str(int(d))] = ent
        out[f"quicklook_fov{fov}"] = dict(S=int(Q[f"S{fov}"].dropna().iloc[0]), per_detector=tab)
    # ours/IRSA per detector (stars), descriptive
    pdet = {}
    for d, gd in ok[ok.target.str.startswith("gaia") & (ok.f_r5 >= 10 * ok.e_r5)].groupby("det"):
        r = gd.f_ours / gd.f_r5
        pdet[str(int(d))] = dict(n=int(len(r)), median=float(r.median()), rscatter=rmad(r), median_err=float(1.2533 * rmad(r) / np.sqrt(len(r))))
    out["stars_ours_over_irsa_per_detector"] = pdet
    json.dump(out, open("data/findings/e31.json", "w"), indent=1, default=lambda o: None if o is None else float(o))
    flat = {}

    def put(k, v):
        k = re.sub(r"[^A-Za-z0-9_]", "_", k)
        flat[k] = None if (isinstance(v, float) and not np.isfinite(v)) else v
    put("verdict", out["verdict"])
    put("n_stars_agree", out["n_stars_agree"])
    for t, r in out["targets"].items():
        for k in ("n_pictures", "n_bins", "median_ratio", "median_ratio_err", "robust_scatter", "chi2", "dof", "chi2_dof",
                  "chi2_dof_paired", "agrees"):
            put(f"{t}_{k}", r[k])
        put(f"{t}_median_ratio_r3", r["radius_r3"]["median_ratio"])
        put(f"{t}_median_ratio_r4", r["radius_r4"]["median_ratio"])
    for fov in (2, 4):
        for d, e in out[f"quicklook_fov{fov}"]["per_detector"].items():
            for ref in ("irsa_r5", "ours"):
                put(f"ql_fov{fov}_D{d}_{ref}_ratio", e[ref]["median"])
                put(f"ql_fov{fov}_D{d}_{ref}_correction", e[ref]["correction"])
                put(f"ql_fov{fov}_D{d}_{ref}_rscatter", e[ref]["rscatter"])
    for d, e in pdet.items():
        put(f"stars_ours_over_irsa_D{d}", e["median"])
        put(f"stars_ours_over_irsa_D{d}_err", e["median_err"])
    json.dump(flat, open("data/findings/e31s.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in flat.items()}, indent=0)[:4000])


# ------------------------------------------------------------------ figure
def stage_fig():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = json.load(open("data/findings/e31.json"))
    Q = pd.read_parquet(f"{D}/ql.parquet")
    ok = Q[np.isfinite(Q.f_r5) & (Q.nbad_r5 == 0) & np.isfinite(Q.f_ours)]
    order = [t for t in out["targets"] if t.startswith("gaia")] + ["epsindib", "wise0855"]
    names = {t: f"Gaia DR3 {t[4:]}" for t in order if t.startswith("gaia")}
    names.update(epsindib="Epsilon Indi B (Ba+Bb)", wise0855="WISE 0855−0714")
    fig, axs = plt.subplots(4, 2, figsize=(9, 10.5), sharex=True)
    axs = axs.ravel()
    for ax, t in zip(axs, order):
        g = ok[ok.target == t]
        sn = g.f_r5 / g.e_r5
        gg = g[sn >= 5]
        ax.scatter(gg.wl, gg.f_ours / gg.f_r5, s=4, color="#9aa0a6", alpha=0.6, lw=0)
        b = [x for x in out["targets"][t]["bins"] if x["ratio"] is not None]
        ax.errorbar([x["centre"] for x in b], [x["ratio"] for x in b], yerr=[x["ratio_err"] for x in b], fmt="o", ms=3.5,
                    color="#1f5fa8", ecolor="#1f5fa8", elinewidth=1, capsize=0)
        ax.axhline(1, color="k", lw=0.7)
        ax.set_ylim(0.7, 1.3) if t != "wise0855" else ax.set_ylim(0.0, 2.0)
        ax.set_title(names[t], fontsize=9)
        ax.set_xscale("log")
    ax = axs[7]
    S_ = Q[Q.target.str.startswith("gaia") & np.isfinite(Q.f_r5) & (Q.nbad_r5 == 0) & (Q.f_r5 >= 10 * Q.e_r5)]
    ax.scatter(S_.wl, S_.ql2 / S_.f_r5, s=4, color="#c26a00", alpha=0.6, lw=0, label="2\u2032 view (default)")
    ax.scatter(S_.wl, S_.ql4 / S_.f_r5, s=4, color="#5b3a8e", alpha=0.6, lw=0, label="4\u2032 view")
    ax.legend(fontsize=8, markerscale=3, loc="lower left", frameon=False)
    ax.axhline(1, color="k", lw=0.7)
    ax.set_ylim(0.0, 1.3)
    ax.set_title("Site 3\u00d73 quick look, 5 stars", fontsize=9)
    for a in axs:
        a.set_xticks([0.75, 1, 1.5, 2, 3, 4, 5])
        a.set_xticklabels(["0.75", "1", "1.5", "2", "3", "4", "5"])
        a.tick_params(labelsize=8)
    for a in axs[6:]:
        a.set_xlabel("Wavelength (µm)", fontsize=9)
    for a in axs[::2]:
        a.set_ylabel("Ours / IRSA recipe", fontsize=9)
    axs[7].set_ylabel("Quick look / IRSA recipe", fontsize=9)
    fig.tight_layout()
    fig.savefig("data/findings/e31_compare.png", dpi=150)
    fig.savefig("data/findings/e31_compare.svg")


if __name__ == "__main__":
    dict(select=stage_select, fetch=stage_fetch, measure=stage_measure, ql=stage_ql, analyze=stage_analyze, fig=stage_fig)[sys.argv[1]]()
