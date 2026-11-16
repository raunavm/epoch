"""E18-B: SPHEREx variability in the NEP deep field with the colour-aware static model (docs/protocols/E18.md,
section "E18-B frozen choices"). Run from the repo root:
  .venv/bin/python scripts/run_e18b.py catalogs   # Gaia DR3 variability tables + VSX in the region (catalogue facts only)
  .venv/bin/python scripts/run_e18b.py prep       # consolidate data/phase2/e18/phot -> data/phase2/e18b/phot.parquet
  .venv/bin/python scripts/run_e18b.py gate0      # per-detector error scale s_d and systematic floor f_d (constant stars)
  .venv/bin/python scripts/run_e18b.py stats      # statistics: all sources, shuffled null, injections
  .venv/bin/python scripts/run_e18b.py analyze    # thresholds, recall, candidates + vetoes, figures, data/findings/e18b.json
"""
import os, sys, time, json, glob

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

E18 = "data/phase2/e18"
D = "data/phase2/e18b"
OUT = "data/findings/e18b.json"
FIGS = "site/data/figs"
RA0, DEC0, RCAT = 270.0, 66.56, 0.47


# ---------------------------------------------------------------- catalogues (facts only, no SPHEREx statistics)
def catalogs():
    from run_e10 import tap, vizier_cone
    os.makedirs(D, exist_ok=True)
    s = pd.read_parquet(f"{E18}/sources.parquet")
    g = s[(s.kind == "gaia") & s.in_region & (s.phot_variable_flag == "VARIABLE")]   # vari_* tables hold only these
    ids = ",".join(g.id.astype(str))
    out = {}
    qs = {"ecl": "SELECT source_id, frequency, global_ranking FROM gaiadr3.vari_eclipsing_binary",
          "rr": "SELECT source_id, pf, p1_o FROM gaiadr3.vari_rrlyrae",
          "rot": "SELECT source_id, best_rotation_period FROM gaiadr3.vari_rotation_modulation",
          "mso": "SELECT source_id, frequency1, amplitude_g_freq1 FROM gaiadr3.vari_ms_oscillator",
          "sts": "SELECT source_id, frequency, amplitude_estimate FROM gaiadr3.vari_short_timescale",
          "lpv": "SELECT source_id, frequency, amplitude FROM gaiadr3.vari_long_period_variable",
          "summ": "SELECT source_id, num_selected_g_fov, std_dev_mag_g_fov FROM gaiadr3.vari_summary"}
    for k, q in qs.items():
        try:
            t = tap(f"{q} WHERE source_id IN ({ids})")
            t["source_id"] = t.source_id.astype("int64").astype(str)
            out[k] = t
            print(k, len(t), flush=True)
        except Exception as e:
            print(k, "failed", repr(e)[:100], flush=True)
    m = g[["id"]].rename(columns={"id": "source_id"})
    for k, t in out.items():
        m = m.merge(t.rename(columns={c: f"{k}_{c}" for c in t.columns if c != "source_id"}), on="source_id", how="left")
    m.to_parquet(f"{D}/gaia_vari.parquet", index=False)
    v = vizier_cone("B/vsx/vsx", RA0, DEC0, RCAT, "OID,Name,Type,max,min,Period,RAJ2000,DEJ2000")
    if v is None:
        print("VSX unreachable", flush=True)
    else:
        v.to_parquet(f"{D}/vsx.parquet", index=False)
        print("VSX", len(v), flush=True)


# ---------------------------------------------------------------- frozen choices (docs/protocols/E18.md "E18-B frozen choices")
SEED = 2026111518
MAX_MASKED, MIN_EDGE = 3, 8                       # usable picture: n_masked_box <= 3, >= 8 px from the picture edge
EXCL_BITS = (1 << 1) | (1 << 20) | (1 << 22) | (1 << 23) | (1 << 24) | (1 << 25)   # OVERFLOW, CROSSTALK, GHOST*, STREAK in the 3x3 core
CHI2LOC_FAC, CHI2LOC_MIN = 10.0, 10.0             # drop picture if local chi2/dof > max(10, 10 x source median)
MIN_PICS, MIN_DET = 100, 30                       # source: >= 100 usable pictures; detector term: >= 30
N_KNOTS = 4                                        # cubic B-spline per detector, 4 equally spaced interior knots
CLIP = 5.0                                         # static fit only: points with |pull| > 5 excluded, 3 iterations
BIN_D, MIN_BIN = 1.0, 3                            # time bins of 1 d, >= 3 pictures per bin
FMIN, FMAX, OVERS = 1 / 100.0, 1 / 0.1, 5          # LS 0.1-100 d, grid 1/(5 T)
P_ORB = 98.0 / 1440
ALIAS_P = [1.0, 0.5, 1 / 3, 182.6, 365.25] + [k * P_ORB for k in range(1, 5)]
ALIAS_TOL, N_WIN_PEAKS = 0.05, 3
STATS = ["chi2r", "chi2r_det", "s2", "ls"]         # detection statistics (amp descriptive)
TIME_STATS = ["s2", "ls", "amp"]
FAR = 0.001
N_SHUF = 5
AMPS = [0.02, 0.05, 0.10, 0.20]
P_INJ = (0.2, 50.0)
ECL_DUTY = 0.08
BRIGHT_W1, BRIGHT_G = 9.0, 10.0                    # bright sources for crosstalk / persistence vetoes
BLOOM_W1, BLOOM_G, BLOOM_R = 8.0, 9.0, 60.0        # bloom veto: such a star within 60"
AIRGLOW = (1.053, 1.113)
NGC6543 = (269.6392, 66.6331, 3.5)                 # deviation D-B1: Cat's Eye nebula, veto radius (arcmin)
WORKERS = 4


def flags_ok(fc):
    return (fc >= 0) & ((fc & EXCL_BITS) == 0)


def anomaly_set():
    """(OBSID, band) pairs the SPHEREx exposure anomaly log (data/ref, 2025-08-10, incomplete) marks as affected
    (code 1: > 90% or 2: 10-90% of the band)."""
    a = pd.read_csv("data/ref/spherex_exposure_anomaly_log_20250810.csv")
    return {(o, b) for b in range(1, 7) for o in a.OBSID[a[f"BAND_{b}_AFFECTED"] >= 1]}


def prep():
    """One pass over the photometry parts (each part holds whole pictures): usable rows (sorted by source, time) ->
    phot.parquet; positions of bright sources in every picture (crosstalk / persistence vetoes) -> bright_xy.parquet;
    picture orientation PA from all sources in the picture."""
    s = pd.read_parquet(f"{E18}/sources.parquet")
    sc = s.set_index("idx")
    bright = set(s.idx[(s.W1mag < BRIGHT_W1) | (s.G < BRIGHT_G)])
    bad = anomaly_set()
    cols = ["src", "x", "y", "n_masked_box", "flags_core", "flux_mJy", "flux_err_mJy", "wl_um", "chi2dof_local", "mjd",
            "detector", "obs_id", "edge_px"]
    keep, bxy, cnt = [], [], dict(rows=0, nonfinite=0, masked=0, edge=0, flags=0, anomaly=0)
    parts = sorted(glob.glob(f"{E18}/phot/part-*.parquet"))
    npic, pics_det = 0, {}
    for p in parts:
        q = pd.read_parquet(p, columns=cols)
        cnt["rows"] += len(q)
        bxy.append(q.loc[q.src.isin(bright), ["src", "x", "y", "mjd", "detector", "flux_mJy"]])
        key = q.obs_id.astype(str) + "_" + q.detector.astype(str)
        uk, pic = np.unique(key.values, return_inverse=True)
        q["pic"] = (pic + npic).astype(np.int32)
        npic += len(uk)
        # orientation of the pixel axes relative to the tangent plane, per picture
        ra, de = sc.ra.reindex(q.src.values).values, sc.dec.reindex(q.src.values).values
        xi = np.cos(np.radians(de)) * np.sin(np.radians(ra - RA0))
        eta = np.sin(np.radians(de)) * np.cos(np.radians(DEC0)) - np.cos(np.radians(de)) * np.sin(np.radians(DEC0)) * np.cos(np.radians(ra - RA0))
        g = pd.DataFrame(dict(k=q.pic.values, x=q.x.values, y=q.y.values, xi=xi, eta=eta))
        for c in ("x", "y", "xi", "eta"):
            g[c] = g[c] - g.groupby("k")[c].transform("mean")
        g["c"] = g.xi * g.x + g.eta * g.y
        g["s"] = g.xi * g.y - g.eta * g.x
        ang = g.groupby("k")[["c", "s"]].sum()
        q["pa"] = (np.degrees(np.arctan2(ang.s, ang.c)) % 360).reindex(q.pic.values).values.astype(np.float32)
        ok = np.isfinite(q.flux_mJy) & np.isfinite(q.flux_err_mJy) & (q.flux_err_mJy > 0)
        cnt["nonfinite"] += int((~ok).sum())
        m = q.n_masked_box <= MAX_MASKED
        cnt["masked"] += int((ok & ~m).sum()); ok &= m
        m = q.edge_px >= MIN_EDGE
        cnt["edge"] += int((ok & ~m).sum()); ok &= m
        m = flags_ok(q.flags_core.values)
        cnt["flags"] += int((ok & ~m).sum()); ok &= m
        m = ~np.array([(o, int(d)) in bad for o, d in zip(q.obs_id.values, q.detector.values)])
        cnt["anomaly"] += int((ok & ~m).sum()); ok &= m
        q = q[ok].drop(columns=["obs_id", "edge_px", "n_masked_box", "flags_core"])
        for d_ in np.unique(q.detector.values):
            pics_det[int(d_)] = pics_det.get(int(d_), 0) + int(np.unique(q.pic.values[q.detector.values == d_]).size)
        keep.append(q.astype(dict(src=np.int32, x=np.float32, y=np.float32, detector=np.int8)))
    q = pd.concat(keep, ignore_index=True)
    del keep
    med = q.groupby("src").chi2dof_local.transform("median")
    m = q.chi2dof_local <= np.maximum(CHI2LOC_MIN, CHI2LOC_FAC * med)
    cnt["chi2_local"] = int((~m).sum())
    q = q[m].sort_values(["src", "mjd"]).reset_index(drop=True)
    q.to_parquet(f"{D}/phot.parquet", index=False, row_group_size=200000)
    pd.concat(bxy, ignore_index=True).to_parquet(f"{D}/bright_xy.parquet", index=False)
    cnt.update(n_parts=len(parts), n_pictures=npic, kept=len(q), n_pictures_with_usable_rows_per_detector=pics_det,
               mjd_range=[float(q.mjd.min()), float(q.mjd.max())], time=time.strftime("%H:%M:%S"))
    json.dump(cnt, open(f"{D}/prep.json", "w"), indent=1)
    print(cnt, flush=True)


# ---------------------------------------------------------------- the colour-aware static model and statistics
def _spline_design(lam, det):
    from scipy.interpolate import BSpline
    dets = np.unique(det)
    nb = N_KNOTS + 4
    A = np.zeros((len(lam), nb * len(dets)))
    for j, d in enumerate(dets):
        s = det == d
        a, b = lam[s].min() - 1e-6, lam[s].max() + 1e-6
        t = np.r_[[a] * 4, np.linspace(a, b, N_KNOTS + 2)[1:-1], [b] * 4]
        A[np.flatnonzero(s), nb * j:nb * (j + 1)] = BSpline.design_matrix(lam[s], t, 3).toarray()
    return A


def static_fit(f, e, lam, det, s_d, f_d):
    """Robust WLS per-detector cubic spline; sigma^2 = (s_d e)^2 + (f_d M)^2. Returns model, sigma, npar."""
    A = _spline_design(lam, det)
    sd = np.array([s_d[int(d)] for d in det]) * e
    fd = np.array([f_d[int(d)] for d in det])
    use = np.ones(len(f), bool)
    sig = sd
    for _ in range(3):
        w = 1 / sig[use]
        c, *_ = np.linalg.lstsq(A[use] * w[:, None], f[use] * w, rcond=None)
        M = A @ c
        sig = np.sqrt(sd ** 2 + (fd * np.abs(M)) ** 2)
        use = np.abs((f - M) / sig) < CLIP
    npar = int(np.linalg.matrix_rank(A[use]))
    return M, sig, npar


def alias_mask(freq, t):
    """Allowed frequencies: not within ALIAS_TOL (in period) of the fixed alias periods or of the N_WIN_PEAKS
    strongest peaks of this source's spectral window."""
    from astropy.timeseries import LombScargle
    bad = np.zeros(len(freq), bool)
    P = 1 / freq
    win = LombScargle(t, np.ones(len(t)), fit_mean=False, center_data=False).power(freq, method="fast")
    pk = []
    o = np.argsort(win)[::-1]
    for i in o:
        if all(abs(P[i] / p - 1) > ALIAS_TOL for p in pk):
            pk.append(P[i])
        if len(pk) == N_WIN_PEAKS:
            break
    for p in list(ALIAS_P) + pk:
        bad |= np.abs(P / p - 1) < ALIAS_TOL
    return ~bad, pk


def binned(t, a, w, t0):
    b = np.floor((t - t0) / BIN_D).astype(int)
    ub, inv, nb = np.unique(b, return_inverse=True, return_counts=True)
    sw = np.bincount(inv, w)
    ab = np.bincount(inv, w * a) / sw
    ok = nb >= MIN_BIN
    return ab[ok], 1 / np.sqrt(sw[ok]), ub[ok]


def s2_of(t, a, w, t0):
    ab, sb, _ = binned(t, a, w, t0)
    if len(ab) < 5:
        return np.nan, np.nan, 0
    z2 = (ab / sb) ** 2
    exc = z2 - 1
    tot = exc[exc > 0].sum()
    return float(z2.mean()), float(exc.max() / tot) if tot > 0 else 0.0, len(ab)


def source_stats(d, s_d, f_d, freq=None, allowed=None, inject=None, shuffle_rng=None, full=False):
    """d: dict of arrays t, f, e, lam, det, pa. Returns statistics dict (and arrays if full)."""
    from astropy.timeseries import LombScargle
    t, f, e, lam, det = d["t"], d["f"], d["e"], d["lam"], d["det"]
    if inject is not None:
        f = f * (1 + inject(t))
    M, sig, npar = static_fit(f, e, lam, det, s_d, f_d)
    if shuffle_rng is not None:                        # time labels permuted within each detector
        t = t.copy()
        for dd in np.unique(det):
            i = np.flatnonzero(det == dd)
            t[i] = t[shuffle_rng.permutation(i)]
    p = (f - M) / sig
    n = len(f)
    out = dict(n=n, chi2r=float((p ** 2).sum() / max(1, n - npar)))
    exc = p ** 2 - 1
    tot = exc[exc > 0].sum()
    out["frac1"] = float(exc.max() / tot) if tot > 0 else 0.0
    cd = {}
    for dd in np.unique(det):
        s = det == dd
        if s.sum() >= MIN_DET:
            cd[int(dd)] = float((p[s] ** 2).sum() / (s.sum() - min(N_KNOTS + 4, s.sum() - 1)))
    out["chi2r_det"] = max(cd.values()) if cd else np.nan
    out["chi2r_by_det"] = cd
    v = M > 0
    a, w = (f[v] - M[v]) / M[v], (M[v] / sig[v]) ** 2
    tv, t0 = t[v], d["t"].min()
    out["s2"], out["fracbin"], out["n_bins"] = s2_of(tv, a, w, t0)
    ab, sb, ub = binned(tv, a, w, t0)
    out["amp"] = float(np.percentile(ab, 95) - np.percentile(ab, 5)) if len(ab) >= 5 else np.nan
    sdet = {}
    for dd in np.unique(det[v]):
        s = det[v] == dd
        sdet[int(dd)] = s2_of(tv[s], a[s], w[s], t0)[0] if s.sum() >= MIN_DET else np.nan
    out["s2_by_det"] = sdet
    g = (d["pa"][v] >= 180).astype(int)
    out["s2_by_pa"] = {k: s2_of(tv[g == k], a[g == k], w[g == k], t0)[0] if (g == k).sum() >= MIN_DET else np.nan
                       for k in (0, 1)}
    if freq is None:
        T = t.max() - t.min()
        freq = np.arange(FMIN, FMAX, 1 / (OVERS * T))
        allowed, pk = alias_mask(freq, d["t"])
        out["window_peaks_d"] = pk
    pw = LombScargle(tv, a, 1 / np.sqrt(w)).power(freq, method="fast")
    pa_ = np.where(allowed, pw, -1)
    i = int(np.argmax(pa_))
    out["ls"], out["P_best"] = float(pw[i]), float(1 / freq[i])
    out["m_ab"] = float(-2.5 * np.log10(np.median(np.abs(M)) / 3.631e6))
    if full:
        out["_arr"] = dict(t=t, f=f, M=M, sig=sig, a_all=(f - M) / np.where(M > 0, M, np.nan), det=det, lam=lam,
                           freq=freq, power=pw, allowed=allowed, ab=ab, sb=sb, ub=ub, t0=t0)
    return out


def sine_amp(t, a, w, P):
    """WLS amplitude (and error) of a sinusoid with period P fitted to residuals a (mean free)."""
    X = np.stack([np.ones_like(t), np.sin(2 * np.pi * t / P), np.cos(2 * np.pi * t / P)], 1)
    sw = np.sqrt(w)
    c, *_ = np.linalg.lstsq(X * sw[:, None], a * sw, rcond=None)
    cov = np.linalg.inv((X * w[:, None]).T @ X)
    A = np.hypot(c[1], c[2])
    eA = np.sqrt((c[1] ** 2 * cov[1, 1] + c[2] ** 2 * cov[2, 2]) / max(A * A, 1e-30))
    return float(A), float(eA)


# ---------------------------------------------------------------- populations
def populations():
    """Sources table with catalogue memberships, null flag and calibration/test split (fixed seed)."""
    s = pd.read_parquet(f"{E18}/sources.parquet")
    s = s[s.in_region & s.is_primary].reset_index(drop=True)
    s["vsx"] = None
    s["vsx_type"] = None
    s["vsx_period"] = np.nan
    if os.path.exists(f"{D}/vsx.parquet"):
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        v = pd.read_parquet(f"{D}/vsx.parquet")
        cv = SkyCoord(v.RAJ2000.astype(float).values * u.deg, v.DEJ2000.astype(float).values * u.deg)
        i, d2, _ = SkyCoord(s.ra.values * u.deg, s.dec.values * u.deg).match_to_catalog_sky(cv)
        m = d2.arcsec < 5.0
        s.loc[m, "vsx"] = v.Name.values[i[m]]
        s.loc[m, "vsx_type"] = v.Type.values[i[m]]
        s.loc[m, "vsx_period"] = pd.to_numeric(v.Period.values[i[m]], errors="coerce")
    s["gaia_var"] = (s.phot_variable_flag == "VARIABLE") | s.vari_class.notna()
    s["kim"] = s.neowise_var_kim2026.fillna(False).astype(bool)
    s["mq"] = s.milliquas_name.notna()
    s["catalogued_var"] = s.gaia_var | s.kim | s.vsx.notna()
    s["null"] = (s.kind == "gaia") & ~s.catalogued_var & ~s.mq & ~(s.classprob_quasar.fillna(0) > 0.5)
    rng = np.random.default_rng(SEED)
    nid = np.sort(s.id[s.null].values)
    cal = set(nid[rng.permutation(len(nid))[: len(nid) // 2]])
    s["half"] = np.where(s.null, np.where(s.id.isin(cal), "cal", "test"), "")
    return s


def load_src(srcs):
    q = pd.read_parquet(f"{D}/phot.parquet", filters=[("src", "in", list(map(int, srcs)))],
                        columns=["src", "mjd", "flux_mJy", "flux_err_mJy", "wl_um", "detector", "pa", "x", "y", "pic"])
    out = {}
    for k, g in q.groupby("src"):
        out[int(k)] = dict(t=g.mjd.values.astype(float), f=g.flux_mJy.values.astype(float),
                           e=g.flux_err_mJy.values.astype(float), lam=g.wl_um.values.astype(float),
                           det=g.detector.values.astype(int), pa=g.pa.values.astype(float), x=g.x.values, y=g.y.values,
                           pic=g.pic.values)
    return out


def drop_thin(d):
    """Detectors with < MIN_DET usable pictures are dropped for this source."""
    u, c = np.unique(d["det"], return_counts=True)
    keep = np.isin(d["det"], u[c >= MIN_DET])
    return {k: v[keep] for k, v in d.items()}


def gate0():
    """s_d: sqrt(median reduced chi2 of the spline static model) over constant stars 15 <= G < 17 (f_d = 0);
    f_d: per-detector fractional floor so that the median reduced chi2 of constant stars G < 13 is 1."""
    s = populations()
    c = s[s.null & (s.ruwe < 1.4) & (s.d_nearest_arcsec > 15) & (s.blend_n == 1) & (s.G < 17)]
    data = load_src(c.idx.values)
    one = {d: 1.0 for d in range(1, 7)}
    zero = {d: 0.0 for d in range(1, 7)}

    def red(idx, s_d, f_d):
        r = {d: [] for d in range(1, 7)}
        for i in idx:
            if i not in data:
                continue
            d = drop_thin(data[i])
            if len(d["f"]) < MIN_PICS:
                continue
            M, sig, _ = static_fit(d["f"], d["e"], d["lam"], d["det"], s_d, f_d)
            p = (d["f"] - M) / sig
            for dd in np.unique(d["det"]):
                m = d["det"] == dd
                r[int(dd)].append((p[m] ** 2).sum() / (m.sum() - (N_KNOTS + 4)))
        return r

    faint = c.idx[(c.G >= 15)].values
    r0 = red(faint, one, zero)
    s_d = {d: float(np.sqrt(np.median(v))) for d, v in r0.items() if v}
    bright = c.idx[c.G < 13].values
    lo, hi = {d: 0.0 for d in s_d}, {d: 0.2 for d in s_d}
    for _ in range(12):                                  # bisection on f_d, all detectors at once
        mid = {d: (lo[d] + hi[d]) / 2 for d in s_d}
        rb = red(bright, s_d, mid)
        for d in s_d:
            if rb[d] and np.median(rb[d]) > 1:
                lo[d] = mid[d]
            else:
                hi[d] = mid[d]
    f_d = {d: (lo[d] + hi[d]) / 2 for d in s_d}
    rb1 = red(bright, s_d, zero)
    out = dict(rule="s_d from constant stars 15<=G<17 (median reduced chi2 of the per-detector cubic-spline static "
                    "model = 1); f_d fractional floor added in quadrature so that constant stars G<13 have median "
                    "reduced chi2 = 1", s_d=s_d, f_d=f_d, n_faint=len(faint), n_bright=len(bright),
               bright_median_red_chi2_without_floor={d: float(np.median(v)) for d, v in rb1.items() if v},
               es_v2_1_irreducible_scatter=0.03, time=time.strftime("%H:%M:%S"))
    json.dump(out, open(f"{D}/gate0.json", "w"), indent=1)
    print(json.dumps(out, indent=1), flush=True)


# ---------------------------------------------------------------- statistics (parallel)
def _inj_fn(kind, A, P, ph):
    if kind == "sin":
        return lambda t: A * np.sin(2 * np.pi * t / P + ph)
    return lambda t: np.where(((t / P + ph / (2 * np.pi)) % 1.0) < ECL_DUTY, -A, 0.0)


def _flat(r):
    o = {k: v for k, v in r.items() if not isinstance(v, (dict, list)) and k != "_arr"}
    for k in ("chi2r_by_det", "s2_by_det", "s2_by_pa"):
        for kk, vv in r.get(k, {}).items():
            o[f"{k}_{kk}"] = vv
    return o


def _work(task):
    chunk, roles, g0 = task
    s_d = {int(k): v for k, v in g0["s_d"].items()}
    f_d = {int(k): v for k, v in g0["f_d"].items()}
    data = load_src(chunk)
    real, shuf, inj = [], [], []
    for i in chunk:
        if i not in data:
            continue
        d = drop_thin(data[i])
        if len(d["f"]) < MIN_PICS:
            continue
        rng = np.random.default_rng(SEED + int(i))
        freq = np.arange(FMIN, FMAX, 1 / (OVERS * (d["t"].max() - d["t"].min())))
        allowed, pk = alias_mask(freq, d["t"])
        try:
            r = source_stats(d, s_d, f_d, freq, allowed)
        except Exception as e:
            print("fail", i, repr(e)[:100], flush=True)
            continue
        real.append(dict(src=int(i), **_flat(r), window_peaks=json.dumps(pk)))
        if roles.get(i) == "cal":
            for k in range(N_SHUF):
                shuf.append(dict(src=int(i), k=k, **_flat(source_stats(d, s_d, f_d, freq, allowed, shuffle_rng=rng))))
        if roles.get(i) == "test":
            for kind in ("sin", "ecl"):
                for A in AMPS:
                    P = float(np.exp(rng.uniform(np.log(P_INJ[0]), np.log(P_INJ[1]))))
                    ph = float(rng.uniform(0, 2 * np.pi))
                    rr = source_stats(d, s_d, f_d, freq, allowed, inject=_inj_fn(kind, A, P, ph))
                    inj.append(dict(src=int(i), kind=kind, A=A, P=P, phase=ph, **_flat(rr)))
    return real, shuf, inj


def stats():
    from concurrent.futures import ProcessPoolExecutor, as_completed
    g0 = json.load(open(f"{D}/gate0.json"))
    s = populations()
    roles = dict(zip(s.idx, s.half))
    idx = s.idx.values
    chunks = [idx[i:i + 25] for i in range(0, len(idx), 25)]
    os.makedirs(f"{D}/stats", exist_ok=True)
    for p in glob.glob(f"{D}/stats/*.parquet"):
        os.remove(p)
    t0 = time.time()
    with ProcessPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(_work, (c, {int(k): roles[k] for k in c}, g0)): n for n, c in enumerate(chunks)}
        for k, fu in enumerate(as_completed(futs)):
            n = futs[fu]
            real, shuf, inj = fu.result()
            for name, rows in (("real", real), ("shuf", shuf), ("inj", inj)):
                if rows:
                    pd.DataFrame(rows).to_parquet(f"{D}/stats/{name}-{n:04d}.parquet", index=False)
            if k % 10 == 0:
                print(f"{time.strftime('%H:%M:%S')} {k + 1}/{len(chunks)} chunks, {time.time() - t0:.0f}s", flush=True)
    print("stats done", time.strftime("%H:%M:%S"), flush=True)


# ---------------------------------------------------------------- analysis
def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


class Norm:
    """z = (log10 X - median(m_AB)) / scale(m_AB) from the calibration-half real null (10 quantile bins in m_AB)."""
    def __init__(self, m, x):
        ok = np.isfinite(m) & np.isfinite(x) & (x > 0)
        m, y = m[ok], np.log10(x[ok])
        e = np.quantile(m, np.linspace(0, 1, 11))
        b = np.clip(np.searchsorted(e, m, side="right") - 1, 0, 9)
        self.c = np.array([np.median(m[b == k]) for k in range(10)])
        self.med = np.array([np.median(y[b == k]) for k in range(10)])
        self.sc = np.array([1.4826 * np.median(np.abs(y[b == k] - np.median(y[b == k]))) for k in range(10)])

    def z(self, m, x):
        x = np.asarray(x, float)
        y = np.log10(np.where(x > 0, x, np.nan))
        return (y - np.interp(m, self.c, self.med)) / np.interp(m, self.c, self.sc)


def _read(name):
    fs = sorted(glob.glob(f"{D}/stats/{name}-*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def calibrate(real, shuf, pop):
    cal = real[real.src.isin(pop.idx[pop.half == "cal"])]
    norms, thr = {}, {}
    for st in STATS + ["amp"]:
        norms[st] = Norm(cal.m_ab.values, cal[st].values)
        zr = norms[st].z(cal.m_ab.values, cal[st].values)
        q = [float(np.nanquantile(zr, 1 - FAR))]
        if st in TIME_STATS and len(shuf):
            sm = shuf.merge(cal[["src", "m_ab"]], on="src", suffixes=("_s", ""))
            q.append(float(np.nanquantile(norms[st].z(sm.m_ab.values, sm[st].values), 1 - FAR)))
        thr[st] = dict(Z=max(q), from_real=q[0], from_shuffled=q[1] if len(q) > 1 else None)
    # per-detector / per-orientation s2 (consistency checks V5, V6): 99th percentile of z of the null
    sub = {}
    for col in [c for c in real.columns if c.startswith("s2_by_det_") or c.startswith("s2_by_pa_")]:
        ok = np.isfinite(cal[col].values) & (cal[col].values > 0)
        if ok.sum() < 50:
            continue
        n = Norm(cal.m_ab.values[ok], cal[col].values[ok])
        sub[col] = (n, float(np.nanquantile(n.z(cal.m_ab.values[ok], cal[col].values[ok]), 0.99)))
    return norms, thr, sub


def zscores(df, norms):
    for st in STATS + ["amp"]:
        df[f"z_{st}"] = norms[st].z(df.m_ab.values, df[st].values)
    return df


def above(df, thr):
    hit = np.zeros(len(df), bool)
    for st in STATS:
        df[f"hit_{st}"] = df[f"z_{st}"].values > thr[st]["Z"]
        hit |= df[f"hit_{st}"].values
    df["above"] = hit
    return df


def zrow(r, norms, thr):
    z = {st: float(norms[st].z(np.array([r["m_ab"]]), np.array([r[st]]))[0]) for st in STATS}
    return z, any(z[st] > thr[st]["Z"] for st in STATS)


def vetoes(i, pop, norms, thr, sub, g0, bxy):
    """All E18-B vetoes and consistency checks for one source (index i)."""
    s_d = {int(k): v for k, v in g0["s_d"].items()}
    f_d = {int(k): v for k, v in g0["f_d"].items()}
    d = drop_thin(load_src([i])[i])
    freq = np.arange(FMIN, FMAX, 1 / (OVERS * (d["t"].max() - d["t"].min())))
    allowed, _ = alias_mask(freq, d["t"])
    base = source_stats(d, s_d, f_d, freq, allowed, full=True)
    zb, _ = zrow(base, norms, thr)
    res = dict(z=zb)

    def redo(mask, name):
        dd = {k: v[~mask] for k, v in d.items()}
        dd = drop_thin(dd)
        if len(dd["f"]) < MIN_PICS:
            res[name] = dict(n_removed=int(mask.sum()), passed=False, note="too few pictures left")
            return
        r = source_stats(dd, s_d, f_d, freq, allowed)
        z, hit = zrow(r, norms, thr)
        res[name] = dict(n_removed=int(mask.sum()), passed=bool(hit), z={k: round(v, 2) for k, v in z.items()})

    # V1 crosstalk, V2 persistence
    b = bxy[bxy.detector.isin(np.unique(d["det"]))]
    b = b[b.src != i]
    xt = np.zeros(len(d["t"]), bool)
    pe = np.zeros(len(d["t"]), bool)
    bk = {k: g for k, g in b.groupby(["detector", "mjd"])}
    bd = {k: g.sort_values("mjd") for k, g in b.groupby("detector")}
    for j in range(len(d["t"])):
        g = bk.get((d["det"][j], d["t"][j]))
        if g is not None:
            dx, dy = g.x.values - d["x"][j], g.y.values - d["y"][j]
            t1 = (np.abs(dx) <= 2) & (np.abs(np.abs(dy) - 32) <= 2)
            k = np.round(np.abs(dx) / 64)
            t2 = (np.abs(dy) <= 2) & (k >= 1) & (np.abs(np.abs(dx) - 64 * k) <= 2)
            xt[j] = bool((t1 | t2).any())
        g = bd.get(d["det"][j])
        if g is not None:
            lo, hi = np.searchsorted(g.mjd.values, [d["t"][j] - 0.25, d["t"][j]])
            gg = g.iloc[lo:hi]
            gg = gg[gg.mjd < d["t"][j]]
            pe[j] = bool((np.hypot(gg.x.values - d["x"][j], gg.y.values - d["y"][j]) <= 3).any())
    redo(xt, "V1_crosstalk")
    redo(pe, "V2_persistence")
    redo((d["lam"] >= AIRGLOW[0]) & (d["lam"] <= AIRGLOW[1]), "V3_airglow")
    # V4 leave-one-out
    A = base["_arr"]
    v4 = dict(frac1=round(base["frac1"], 3), fracbin=round(base["fracbin"], 3))
    if len(A["ab"]):
        kb = A["ub"][int(np.argmax((A["ab"] / A["sb"]) ** 2))]
        m = np.floor((d["t"] - A["t0"]) / BIN_D).astype(int) == kb
        redo(m, "_V4_drop_bin")
        v4.update(drop_top_bin=res.pop("_V4_drop_bin"))
    v4["passed"] = bool(v4["frac1"] <= 0.3 and v4["fracbin"] <= 0.3 and v4.get("drop_top_bin", {}).get("passed", False))
    res["V4_leave_one_out"] = v4
    # V5 detectors, V6 orientations
    v = A["M"] > 0
    t, a, w, det = d["t"][v], ((d["f"] - A["M"]) / A["M"])[v], (A["M"] / A["sig"])[v] ** 2, d["det"][v]
    pa = d["pa"][v]
    P = base["P_best"]
    zdet = {}
    for dd in range(1, 7):
        col = f"s2_by_det_{dd}"
        val = base["s2_by_det"].get(dd, np.nan)
        if col in sub and np.isfinite(val):
            zdet[dd] = float(sub[col][0].z(np.array([base["m_ab"]]), np.array([val]))[0]) > sub[col][1]
    sin_det = {}
    for dd in np.unique(det):
        m = det == dd
        if m.sum() >= MIN_DET:
            A_, e_ = sine_amp(t[m], a[m], w[m], P)
            sin_det[int(dd)] = (round(A_, 4), round(e_, 4))
    n_s2 = int(sum(zdet.values()))
    n_sin = int(sum(1 for A_, e_ in sin_det.values() if A_ > 3 * e_))
    res["V5_detectors"] = dict(n_det_s2_above_null99=n_s2, sine_at_P_best_per_det=sin_det, n_det_sine_3sigma=n_sin,
                               passed=bool(n_s2 >= 2 or n_sin >= 2))
    zpa, sin_pa = {}, {}
    for k in (0, 1):
        col = f"s2_by_pa_{k}"
        val = base["s2_by_pa"].get(k, np.nan)
        if col in sub and np.isfinite(val):
            zpa[k] = float(sub[col][0].z(np.array([base["m_ab"]]), np.array([val]))[0]) > sub[col][1]
        m = (pa >= 180) == bool(k)
        if m.sum() >= MIN_DET:
            sin_pa[k] = sine_amp(t[m], a[m], w[m], P)
    ok_s2 = len(zpa) == 2 and all(zpa.values())
    ok_sin = len(sin_pa) == 2 and all(A_ > 3 * e_ for A_, e_ in sin_pa.values()) and \
        abs(sin_pa[0][0] - sin_pa[1][0]) < 3 * np.hypot(sin_pa[0][1], sin_pa[1][1])
    res["V6_orientations"] = dict(s2_both_above_null99=bool(ok_s2), sine_both=bool(ok_sin),
                                  sine=[[round(x, 4) for x in sin_pa[k]] for k in sorted(sin_pa)], passed=bool(ok_s2 or ok_sin))
    # V7 bloom
    s = pop.set_index("idx").loc[i]
    allc = pd.read_parquet(f"{E18}/sources.parquet")
    br = allc[((allc.W1mag < BLOOM_W1) | (allc.G < BLOOM_G)) & (allc.idx != i)]
    sep = np.hypot((br.ra - s.ra) * np.cos(np.radians(s.dec)), br.dec - s.dec) * 3600
    res["V7_bloom"] = dict(nearest_bright_arcsec=round(float(sep.min()), 1) if len(sep) else None,
                           passed=bool(len(sep) == 0 or sep.min() > BLOOM_R))
    sep_n = float(np.hypot((s.ra - NGC6543[0]) * np.cos(np.radians(s.dec)), s.dec - NGC6543[1]) * 60)
    res["V8_nebula"] = dict(sep_ngc6543_arcmin=round(sep_n, 2), passed=bool(sep_n > NGC6543[2]))   # deviation D-B1
    res["survives"] = bool(all(res[k]["passed"] for k in res if k.startswith("V")))
    res["P_best"] = P
    res["amp"] = base["amp"]
    return res, base, d


def lc_plot(ax, base, d, title, P=None, ax2=None):
    A = base["_arr"]
    cols = ["#0072B2", "#009E73", "#56B4E9", "#E69F00", "#D55E00", "#CC79A7"]
    v = A["M"] > 0
    t, a, det = d["t"][v], ((d["f"] - A["M"]) / A["M"])[v], d["det"][v]
    w = (A["M"] / A["sig"])[v] ** 2
    allab = []
    for dd in np.unique(det):
        m = det == dd
        ab, sb, ub = binned(t[m], a[m], w[m], A["t0"])
        ax.errorbar(A["t0"] + ub + 0.5 - 60800, ab, sb, fmt=".", ms=3, lw=0.5, color=cols[dd - 1], label=f"D{dd}")
        allab.append(np.abs(ab))
    if allab and np.concatenate(allab).size:
        lim = max(0.05, 1.5 * np.nanpercentile(np.concatenate(allab), 95))
        ax.set_ylim(-lim, lim)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("MJD − 60800 (1-day bins)")
    ax.set_ylabel("flux / static spectrum − 1")
    ax.set_title(title, fontsize=7)
    if ax2 is not None and P:
        ph = (t / P) % 1
        nb = 20
        for dd in np.unique(det):
            m = det == dd
            k = np.floor(ph[m] * nb).astype(int)
            sw = np.bincount(k, w[m], nb)
            mu = np.bincount(k, w[m] * a[m], nb) / np.where(sw > 0, sw, np.nan)
            ax2.plot((np.arange(nb) + 0.5) / nb, mu, ".", ms=3, color=cols[dd - 1], alpha=0.6)
        k = np.floor(ph * nb).astype(int)
        sw = np.bincount(k, w, nb)
        mu = np.bincount(k, w * a, nb) / np.where(sw > 0, sw, np.nan)
        er = 1 / np.sqrt(np.where(sw > 0, sw, np.nan))
        for off in (0, 1):
            ax2.errorbar((np.arange(nb) + 0.5) / nb + off, mu, er, fmt="o", ms=4, color="k", lw=0.8)
        lim = max(0.02, 4 * np.nanmax(np.abs(mu)))
        ax2.set_ylim(-lim, lim)
        ax2.axhline(0, color="k", lw=0.5)
        ax2.set_xlim(0, 2)
        ax2.set_xlabel(f"phase (P = {P:.4g} d)")


def analyze():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g0 = json.load(open(f"{D}/gate0.json"))
    pop = populations()
    real, shuf, inj = _read("real"), _read("shuf"), _read("inj")
    real = real.merge(pop[["idx", "id", "kind", "G", "W1mag", "half", "null", "catalogued_var", "gaia_var", "kim", "mq",
                           "vsx", "vsx_type", "vsx_period", "vari_class", "phot_variable_flag", "milliquas_type",
                           "kim2026_name", "ra", "dec"]].rename(columns={"idx": "src"}), on="src")
    norms, thr, sub = calibrate(real, shuf, pop)
    real = above(zscores(real, norms), thr)
    test = real[real.half == "test"]
    far = {st: dict(k=int(test[f"hit_{st}"].sum()), n=len(test), rate=round(float(test[f"hit_{st}"].mean()), 5),
                    wilson95=wilson(int(test[f"hit_{st}"].sum()), len(test))) for st in STATS}
    far["any"] = dict(k=int(test.above.sum()), n=len(test), rate=round(float(test.above.mean()), 5),
                      wilson95=wilson(int(test.above.sum()), len(test)))
    aw = real[(real.kind == "allwise_only") & ~real.catalogued_var & ~real.mq]
    # ---- injections
    inj = inj.merge(real[["src", "m_ab"]].rename(columns={"m_ab": "m_ab_src"}), on="src")
    inj["m_ab"] = inj.m_ab_src
    inj = above(zscores(inj, norms), thr)
    inj["P_ok"] = np.abs(inj.P_best / inj.P - 1) < 0.02
    mb = np.arange(np.floor(inj.m_ab.min()), np.ceil(inj.m_ab.max()) + 1)
    rec = []
    for kind in ("sin", "ecl"):
        for A in AMPS:
            for lo in mb[:-1]:
                q = inj[(inj.kind == kind) & (inj.A == A) & (inj.m_ab >= lo) & (inj.m_ab < lo + 1)]
                if len(q) == 0:
                    continue
                rec.append(dict(kind=kind, A=A, m_lo=float(lo), n=len(q), recall=round(float(q.above.mean()), 4),
                                wilson95=wilson(int(q.above.sum()), len(q)),
                                by_stat={st: round(float(q[f"hit_{st}"].mean()), 3) for st in STATS},
                                period_recovered=round(float(q.P_ok.mean()), 3) if kind == "sin" else None))
    floor = {}
    for kind in ("sin", "ecl"):
        for lo in mb[:-1]:
            ok = [r["A"] for r in rec if r["kind"] == kind and r["m_lo"] == lo and r["recall"] >= 0.5 and r["n"] >= 10]
            floor.setdefault(kind, {})[f"{lo:.0f}-{lo + 1:.0f}"] = min(ok) if ok else None
    # ---- known variables
    gv = pd.read_parquet(f"{D}/gaia_vari.parquet")
    real = real.merge(gv.rename(columns={"source_id": "id"}), on="id", how="left")
    P_cat = np.full(len(real), np.nan)
    src_cat = np.array([None] * len(real), dtype=object)
    for col, f, lab in (("ecl_frequency", True, "Gaia EB"), ("rr_pf", False, "Gaia RR"), ("sts_frequency", True, "Gaia STS"),
                        ("rot_best_rotation_period", False, "Gaia rotation"), ("vsx_period", False, "VSX")):
        if col in real:
            v = real[col].values.astype(float)
            v = 1 / v if f else v
            m = np.isnan(P_cat) & np.isfinite(v) & (v > 0)
            P_cat[m], src_cat[m] = v[m], lab
    real["P_cat"], real["P_cat_src"] = P_cat, src_cat
    classes = {}

    def cls(name, m):
        q = real[m]
        classes[name] = dict(n=len(q), n_above=int(q.above.sum()), recall=round(float(q.above.mean()), 3) if len(q) else None,
                             wilson95=wilson(int(q.above.sum()), len(q)),
                             by_stat={st: int(q[f"hit_{st}"].sum()) for st in STATS},
                             median_m_ab=round(float(q.m_ab.median()), 2) if len(q) else None)
    for c in sorted(real.vari_class.dropna().unique()):
        cls(f"Gaia vari_class {c}", real.vari_class == c)
    cls("Gaia VARIABLE, no class", (real.phot_variable_flag == "VARIABLE") & real.vari_class.isna())
    for c in sorted(real.vsx_type.dropna().unique()):
        cls(f"VSX {c}", real.vsx_type == c)
    cls("Kim+2026 NEOWISE NEP variable", real.kim)
    cls("Milliquas AGN (not a variability catalogue)", real.mq)
    cls("all catalogued variables", real.catalogued_var)
    cls("null test half (false-alarm reference)", real.half == "test")
    # ---- period comparisons and vetoes on known variables
    bxy = pd.read_parquet(f"{D}/bright_xy.parquet")
    s_d = {int(k): v for k, v in g0["s_d"].items()}
    f_d = {int(k): v for k, v in g0["f_d"].items()}
    per = []
    examples = []
    for _, r in real[np.isfinite(real.P_cat)].iterrows():
        d = drop_thin(load_src([r.src])[r.src])
        freq = np.arange(FMIN, FMAX, 1 / (OVERS * (d["t"].max() - d["t"].min())))
        allowed, _ = alias_mask(freq, d["t"])
        b = source_stats(d, s_d, f_d, freq, allowed, full=True)
        A = b["_arr"]
        rank = None
        powc = None
        if FMIN <= 1 / r.P_cat <= FMAX:
            k = int(np.argmin(np.abs(A["freq"] - 1 / r.P_cat)))
            win = slice(max(0, k - 2), k + 3)
            powc = float(A["power"][win].max())
            rank = int((A["power"][A["allowed"]] > powc).sum()) + 1
        match = any(abs(b["P_best"] / (r.P_cat * f) - 1) < 0.02 for f in (1, 0.5, 2))
        v = A["M"] > 0
        sa = sine_amp(d["t"][v], A["a_all"][v], (A["M"] / A["sig"])[v] ** 2, r.P_cat)
        per.append(dict(id=r.id, src=int(r.src), cls=r.vari_class if isinstance(r.vari_class, str) else r.vsx_type,
                        P_cat=float(r.P_cat), P_cat_src=r.P_cat_src, P_best=b["P_best"], match=bool(match),
                        power_at_P_cat=powc, rank_at_P_cat=rank, ls_peak=b["ls"], above=bool(r.above),
                        sine_amp_at_P_cat=[round(sa[0], 4), round(sa[1], 4)], m_ab=round(r.m_ab, 2), G=float(r.G)))
        examples.append((per[-1], b, d))
    per_df = pd.DataFrame(per)
    known_veto = []
    for _, r in real[real.catalogued_var & real.above].iterrows():
        vr, _, _ = vetoes(int(r.src), pop, norms, thr, sub, g0, bxy)
        known_veto.append(dict(id=r.id, survives=vr["survives"], **{k: vr[k]["passed"] for k in vr if k.startswith("V")}))
    # ---- candidates
    cand = real[real.above & ~real.catalogued_var]
    cands, cand_plot = [], []
    for _, r in cand.iterrows():
        vr, b, d = vetoes(int(r.src), pop, norms, thr, sub, g0, bxy)
        row = dict(id=r.id, src=int(r.src), kind=r.kind, ra=r.ra, dec=r.dec, G=None if np.isnan(r.G) else float(r.G),
                   W1=None if np.isnan(r.W1mag) else float(r.W1mag), m_ab=round(r.m_ab, 2), milliquas=r.milliquas_type
                   if isinstance(r.milliquas_type, str) else None, null_half=r.half or None,
                   z={st: round(float(r[f"z_{st}"]), 2) for st in STATS}, hits=[st for st in STATS if r[f"hit_{st}"]],
                   amp_5_95=round(float(r.amp), 4), P_best=round(float(r.P_best), 5), n=int(r.n), checks=vr)
        cands.append(row)
        if vr["survives"]:
            cand_plot.append((row, b, d))
    n_uncat = int((~real.catalogued_var).sum())
    exp_false = dict(nominal=round(len(STATS) * FAR * n_uncat, 2),
                     realised_any_rate_x_n=round(far["any"]["rate"] * n_uncat, 2),
                     n_noncatalogued_tested=n_uncat)
    # ---- figures
    os.makedirs(FIGS, exist_ok=True)
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    cols = {0.02: "#999999", 0.05: "#56B4E9", 0.10: "#0072B2", 0.20: "#D55E00"}
    for ax, kind, lab in zip(axs, ("sin", "ecl"), ("sinusoid (semi-amplitude A)", "box eclipse (depth A, 8% duty)")):
        for A in AMPS:
            q = [r for r in rec if r["kind"] == kind and r["A"] == A and r["n"] >= 5]
            x = [r["m_lo"] + 0.5 for r in q]
            ax.plot(x, [r["recall"] for r in q], "o-", color=cols[A], label=f"A = {A:.0%}")
            ax.fill_between(x, [r["wilson95"][0] for r in q], [r["wilson95"][1] for r in q], color=cols[A], alpha=0.15)
        ax.set_title(lab, fontsize=9)
        ax.set_xlabel("SPHEREx mean AB magnitude")
        ax.axhline(0.5, color="k", lw=0.5, ls=":")
        ax.spines[["top", "right"]].set_visible(False)
    axs[0].set_ylabel("recall at 0.1% false-alarm per statistic")
    axs[0].legend(fontsize=7, frameon=False)
    fig.suptitle("E18-B: injected variability recovered in real NEP photometry (P = 0.2–50 d)", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{FIGS}/e18b_floor.png", dpi=150)
    plt.close(fig)
    ex = [e for e in examples if (e[0]["rank_at_P_cat"] or 1e9) <= 3 or (e[0]["match"] and e[0]["cls"] == "ECL")]
    ex = sorted(ex, key=lambda e: (e[0]["rank_at_P_cat"] or 1e9))[:4]
    if ex:
        fig, axs = plt.subplots(len(ex), 2, figsize=(9, 2.4 * len(ex)), squeeze=False)
        for (pr, b, d), ax in zip(ex, axs):
            lc_plot(ax[0], b, d, f"Gaia DR3 {pr['id']} ({pr['cls']}), G = {pr['G']:.1f}\n"
                    f"catalogue P = {pr['P_cat']:.5g} d ({pr['P_cat_src']}); SPHEREx LS peak {pr['P_best']:.5g} d", P=pr["P_cat"], ax2=ax[1])
            ax[1].set_title(f"folded at catalogue P (black: all detectors, 20 bins)\nSPHEREx LS rank at catalogue P = "
                            f"{pr['rank_at_P_cat']}; sine amp {pr['sine_amp_at_P_cat'][0]:.3f} ± {pr['sine_amp_at_P_cat'][1]:.3f}", fontsize=7)
        axs[0][0].legend(fontsize=6, ncol=6, frameon=False)
        fig.tight_layout()
        fig.savefig(f"{FIGS}/e18b_examples.png", dpi=140)
        plt.close(fig)
    if cand_plot:
        fig, axs = plt.subplots(len(cand_plot), 2, figsize=(9, 2.4 * len(cand_plot)), squeeze=False)
        for (row, b, d), ax in zip(cand_plot, axs):
            lc_plot(ax[0], b, d, f"{row['id']} (SPHEREx m_AB {row['m_ab']})\nabove threshold in: {', '.join(row['hits'])}",
                    P=row["P_best"], ax2=ax[1])
            ax[1].set_title(f"folded at SPHEREx LS peak P = {row['P_best']:.5g} d (black: all detectors)", fontsize=7)
        fig.tight_layout()
        fig.savefig(f"{FIGS}/e18b_candidates.png", dpi=140)
        plt.close(fig)
    out = dict(experiment="E18-B", protocol="docs/protocols/E18.md, section 'E18-B frozen choices' (14:32 PST)",
               time=time.strftime("%Y-%m-%d %H:%M:%S %Z"), prep=json.load(open(f"{D}/prep.json")), gate0=g0,
               n_sources_analysed=len(real), n_null=dict(cal=int((real.half == "cal").sum()), test=int((real.half == "test").sum())),
               thresholds=thr, false_alarm_test_half=far,
               allwise_only_noncatalogued_exceedance=dict(n=len(aw), k=int(aw.above.sum()), wilson95=wilson(int(aw.above.sum()), len(aw))),
               null_distributions={st: dict(cal_quantiles_z=[round(float(x), 3) for x in np.nanquantile(
                   real[real.half == "cal"][f"z_{st}"], [0.5, 0.9, 0.99, 0.999])],
                   norm_bins_m_ab=[round(float(x), 2) for x in norms[st].c], median_log10=[round(float(x), 4) for x in norms[st].med],
                   scale_log10=[round(float(x), 4) for x in norms[st].sc]) for st in STATS + ["amp"]},
               amp_null_median_by_m_ab={f"{c:.2f}": round(10 ** m, 4) for c, m in zip(norms["amp"].c, norms["amp"].med)},
               injections=dict(n=len(inj), recall=rec, floor_smallest_A_with_recall_ge_50pct=floor),
               known_variables=dict(classes=classes, periods=per, vetoes_on_known_above=known_veto,
                                    n_period_match=int(per_df.match.sum()) if len(per_df) else 0,
                                    n_with_catalogue_period=len(per_df)),
               candidates=dict(n_above_noncatalogued=len(cands), n_survivors=int(sum(c["checks"]["survives"] for c in cands)),
                               expected_false=exp_false, list=cands))
    json.dump(out, open(OUT + ".tmp", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    os.replace(OUT + ".tmp", OUT)
    real.drop(columns=[c for c in real.columns if c.startswith("s2_by") or c.startswith("chi2r_by")]).to_parquet(
        f"{D}/e18b_sources.parquet", index=False)
    print(json.dumps(dict(thr={k: round(v["Z"], 2) for k, v in thr.items()}, far=far["any"], floor=floor,
                          classes={k: (v["n"], v["n_above"]) for k, v in classes.items()},
                          per=f"{out['known_variables']['n_period_match']}/{len(per_df)}",
                          cands=out["candidates"]["n_above_noncatalogued"], surv=out["candidates"]["n_survivors"],
                          exp=exp_false), indent=1, default=str), flush=True)


# ---------------------------------------------------------------- independent checks (survivors only)
def _lc_check(t, mag, err, P_sx):
    from scipy.stats import chi2 as _chi2
    from astropy.timeseries import LombScargle
    t, mag, err = (np.asarray(x, float) for x in (t, mag, err))
    ok = np.isfinite(t) & np.isfinite(mag) & np.isfinite(err) & (err > 0)
    t, mag, err = t[ok], mag[ok], err[ok]
    if len(t) < 20:
        return dict(n_epochs=int(len(t)), confirmed=False, reason="fewer than 20 good epochs")
    w = 1 / err ** 2
    mu = (w * mag).sum() / w.sum()
    c2 = float((w * (mag - mu) ** 2).sum())
    pv = float(_chi2.sf(c2, len(t) - 1))
    amp = float(np.percentile(mag, 95) - np.percentile(mag, 5))
    me = float(np.median(err))
    fr, pw = LombScargle(t, mag, err).autopower(minimum_frequency=1 / 100, maximum_frequency=10, samples_per_peak=5)
    P = float(1 / fr[int(np.argmax(pw))])
    rank = None
    if P_sx and FMIN <= 1 / P_sx <= FMAX:
        k = int(np.argmin(np.abs(fr - 1 / P_sx)))
        rank = int((pw > pw[max(0, k - 2):k + 3].max()).sum()) + 1
    same_P = bool(P_sx and any(abs(P / (P_sx * f) - 1) < 0.02 for f in (1, 0.5, 2)))
    amp_ok = bool(amp > 5 * me and amp > 0.1)
    return dict(n_epochs=int(len(t)), chi2_red=round(c2 / (len(t) - 1), 3), p_constant=pv,
                amplitude_p5_p95_mag=round(amp, 4), median_error_mag=round(me, 4), ls_peak_period_d=round(P, 5),
                ls_peak_power=round(float(pw.max()), 3), same_period_as_spherex=same_P, ls_rank_at_spherex_period=rank, amplitude_criterion=amp_ok,
                confirmed=bool(pv < 0.01 and (same_P or amp_ok)))


def check():
    import io, requests
    from run_e10 import tap
    out = json.load(open(OUT))
    os.makedirs(f"{D}/ztf", exist_ok=True)
    UA = {"User-Agent": "python-client"}
    for c in out["candidates"]["list"]:
        if not c["checks"]["survives"]:
            continue
        rec = []
        cp = f"{D}/ztf/{c['src']}.csv"
        try:
            if not os.path.exists(cp):
                r = requests.get("https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves",
                                 params={"POS": f"CIRCLE {c['ra']:.6f} {c['dec']:.6f} 0.0004", "FORMAT": "csv",
                                         "BAD_CATFLAGS_MASK": 32768}, headers=UA, timeout=(15, 300))
                r.raise_for_status()
                open(cp, "w").write(r.text)
            z = pd.read_csv(cp)
            z = z[z.catflags == 0] if len(z) else z
            if len(z) == 0:
                rec.append(dict(survey="ZTF (IRSA)", status="no good epochs", confirmed=False))
            for fc, q in (z.groupby("filtercode") if len(z) else []):
                q = q[q.oid == q.oid.value_counts().index[0]]
                rec.append(dict(survey="ZTF (IRSA)", filter=str(fc), status="ok", median_mag=round(float(q.mag.median()), 3),
                                **_lc_check(q.mjd, q.mag, q.magerr, c["P_best"])))
        except Exception as e:
            rec.append(dict(survey="ZTF (IRSA)", status="unreachable: " + repr(e)[:100], confirmed=False))
        if c["kind"] == "gaia":
            try:
                h = tap(f"SELECT has_epoch_photometry FROM gaiadr3.gaia_source WHERE source_id = {c['id']}")
                if len(h) and str(h.iloc[0, 0]).lower() in ("true", "1", "t"):
                    g = tap(f"SELECT g_transit_time, g_transit_mag, g_transit_flux_over_error FROM gaiadr3.epoch_photometry WHERE source_id = {c['id']}")
                    arr = lambda v: np.array(str(v).strip("[]{}()").replace(",", " ").split(), float) if isinstance(v, str) else np.asarray(v, float)
                    t_, m_, soe = (arr(g.iloc[0, k]) for k in range(3))
                    rec.append(dict(survey="Gaia DR3 epoch photometry", status="ok", **_lc_check(t_, m_, 1.0857 / soe, c["P_best"])))
                else:
                    rec.append(dict(survey="Gaia DR3 epoch photometry", status="not published for this source", confirmed=False))
            except Exception as e:
                rec.append(dict(survey="Gaia DR3 epoch photometry", status="failed: " + repr(e)[:100], confirmed=False))
        if c.get("G") is not None and c["G"] < 15:
            try:
                from astroquery.mast import Observations
                from astropy.coordinates import SkyCoord
                import astropy.units as u
                obs = Observations.query_criteria(coordinates=SkyCoord(c["ra"] * u.deg, c["dec"] * u.deg), radius=0.003 * u.deg,
                                                  dataproduct_type="timeseries", obs_collection=["TESS", "HLSP"])
                rec.append(dict(survey="TESS (MAST)", status=f"{len(obs)} time-series products listed; light curves not analysed",
                                provenance=sorted(set(str(x) for x in obs["provenance_name"]))[:10] if len(obs) else [],
                                confirmed=False))
            except Exception as e:
                rec.append(dict(survey="TESS (MAST)", status="failed: " + repr(e)[:100], confirmed=False))
        else:
            rec.append(dict(survey="TESS (MAST)", status="not attempted: G >= 15 or no Gaia match (too faint/blended for 21\" pixels)", confirmed=False))
        c["independent"] = rec
        c["independently_confirmed_in"] = sorted(set(r["survey"] for r in rec if r.get("confirmed")))
        print(c["id"], [(r["survey"], r.get("filter"), r.get("status"), r.get("p_constant"), r.get("confirmed")) for r in rec], flush=True)
    json.dump(out, open(OUT + ".tmp", "w"), indent=1, default=str)
    os.replace(OUT + ".tmp", OUT)


def posthoc():
    """Descriptive, after the frozen analysis (labelled post hoc in e18b.json; changes no verdict)."""
    out = json.load(open(OUT))
    r = pd.read_parquet(f"{D}/e18b_sources.parquet")
    vc = r.P_best.round(2).value_counts()
    ph = dict(note="descriptive checks made after the frozen analysis; they change no frozen verdict")
    ph["common_mode_ls_periods"] = dict(
        top_P_best_counts={f"{k:.2f}": int(v) for k, v in vc.head(8).items()},
        n_sources_P_best_79p36_or_95p06=int(((r.P_best - 79.36).abs() < 0.05).sum() + ((r.P_best - 95.06).abs() < 0.05).sum()),
        note="many unrelated sources share LS peaks at 79.4 d and 95.1 d: a common-mode long-period systematic "
             "(not in the frozen alias list); it inflates the null tail of ls")
    if os.path.exists(f"{D}/posthoc_ztf.json"):
        ph["ztf_on_detected_known_and_threshold_limited"] = json.load(open(f"{D}/posthoc_ztf.json"))
    if os.path.exists(f"{D}/posthoc_tess.json"):
        ph["tess"] = json.load(open(f"{D}/posthoc_tess.json"))
    near = r[(np.hypot((r.ra - NGC6543[0]) * np.cos(np.radians(r.dec)), r.dec - NGC6543[1]) * 60 <= NGC6543[2])]
    ph["ngc6543_affected"] = dict(n_sources_within_3p5arcmin=len(near), n_above=int(near.above.sum()),
                                  ids_above=near.id[near.above].tolist())
    from scipy.stats import mannwhitneyu
    far_ = r[np.hypot((r.ra - NGC6543[0]) * np.cos(np.radians(r.dec)), r.dec - NGC6543[1]) * 60 > NGC6543[2]]
    aw = far_[(far_.kind == "allwise_only") & ~far_.catalogued_var & ~far_.mq]
    pop = {}
    for name, g in (("Kim+2026 NEOWISE variables", far_[far_.kim]), ("Milliquas AGN", far_[far_.mq]),
                    ("Kim or Milliquas", far_[far_.kim | far_.mq])):
        c = aw[(aw.m_ab > g.m_ab.min() - 0.3) & (aw.m_ab < g.m_ab.max() + 0.3)]
        pop[name] = dict(n=len(g), n_control=len(c), **{st: dict(
            median_z=round(float(g[f"z_{st}"].median()), 2), control_median_z=round(float(c[f"z_{st}"].median()), 2),
            mann_whitney_p_greater=float(mannwhitneyu(g[f"z_{st}"].dropna(), c[f"z_{st}"].dropna(), alternative="greater").pvalue))
            for st in ("chi2r", "s2", "ls", "amp")})
    ph["population_test_agn_vs_allwise_only"] = dict(
        note="known AGN / NEOWISE variables vs uncatalogued AllWISE-only sources in the same m_AB range (NGC 6543 "
             "region excluded); one-sided Mann-Whitney on the normalised statistics", results=pop)
    out["post_hoc"] = ph
    json.dump(out, open(OUT + ".tmp", "w"), indent=1, default=str)
    os.replace(OUT + ".tmp", OUT)
    print(json.dumps({k: v for k, v in ph.items() if k != "ztf_on_detected_known_and_threshold_limited"}, indent=1, default=str))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    globals()[sys.argv[1]]()
