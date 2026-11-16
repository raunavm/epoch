"""E25: AGN variability spectrum (A) and short/long simultaneous-band cross-correlation (B) in the NEP deep field
(docs/protocols/E25.md). Run from the repo root:
  .venv/bin/python scripts/run_e25.py samples   # catalogue-only sample definition -> data/phase2/e25/samples.parquet
  .venv/bin/python scripts/run_e25.py load      # usable photometry of the sample sources only -> phot_sel.parquet
  .venv/bin/python scripts/run_e25.py persource # static model, per-(source, wavelength bin) excess variance, day LCs
  .venv/bin/python scripts/run_e25.py analyze   # population spectra, stacked CCF, nulls, injections, figures, json
"""
import os, sys, time, json, glob

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_e18b import (populations, static_fit, drop_thin, flags_ok, anomaly_set, NGC6543, MAX_MASKED, MIN_EDGE,
                      CHI2LOC_FAC, CHI2LOC_MIN, MIN_PICS, AIRGLOW)

E18 = "data/phase2/e18"
D = "data/phase2/e25"
G0 = "data/phase2/e18b/gate0.json"
OUT = "data/findings/e25.json"
FIGS = "site/data/figs"
SEED = 2026111525
WORKERS = 4
T0 = 60789.0                                         # day index = floor(mjd - T0)

# wavelength bins: each detector split at the geometric midpoint of its band range (12 bins)
DET_RANGE = {1: (0.74, 1.12), 2: (1.10, 1.65), 3: (1.64, 2.42), 4: (2.42, 3.82), 5: (3.81, 4.42), 6: (4.41, 5.00)}
R_DET = {1: 41, 2: 41, 3: 41, 4: 35, 5: 110, 6: 130}  # resolving power (fine-template bins of width lambda / (2R))
MIN_DAY_PICS, CLIP_DAY, MIN_DAYS_A = 3, 5.0, 10
TPL_MIN_DAYS = 5
MIN_DAYS_B = 30
LAG_W, LAG_MAX, LAG_MIN = 5.0, 150.0, 10.0
N_BOOT, N_SHUF, N_CTRL_DRAW = 2000, 1000, 2000
N_GAIA_PER_AGN = 20


def binedges():
    out = []
    for d, (lo, hi) in DET_RANGE.items():
        mid = np.sqrt(lo * hi)
        out += [(d, lo, mid), (d, mid, hi)]
    return out


BINS = binedges()


# ---------------------------------------------------------------- samples (catalogue columns only)
def samples():
    os.makedirs(D, exist_ok=True)
    s = populations()
    s["w12"] = s.W1mag - s.W2mag
    s["d_ngc6543_arcmin"] = np.hypot((s.ra - NGC6543[0]) * np.cos(np.radians(s.dec)), s.dec - NGC6543[1]) * 60
    far = s.d_ngc6543_arcmin > NGC6543[2]
    kim_agn = s.kim & ((s.kind == "allwise_only") | (s.classprob_quasar.fillna(0) > 0.5) | (s.w12 >= 0.8))
    s["group"] = ""
    agn = (s.mq | kim_agn) & far
    s.loc[agn, "group"] = "agn"
    lo, hi = s.W1mag[agn].min() - 0.3, s.W1mag[agn].max() + 0.3
    aw = (s.kind == "allwise_only") & ~s.catalogued_var & ~s.mq & far & s.W1mag.between(lo, hi)
    s.loc[aw & (s.w12 < 0.8), "group"] = "ctrl_aw"
    s.loc[aw & (s.w12 >= 0.8), "group"] = "aw_red"            # descriptive only (colour-selected AGN candidates)
    g = s[s.null & far & s.W1mag.between(lo, hi)]
    used = set()
    for w1 in s.W1mag[agn].dropna().sort_values().values:   # W1-matched Gaia stars, nearest first, no reuse
        c = g[~g.idx.isin(used)]
        used |= set(c.idx.values[np.argsort(np.abs(c.W1mag.values - w1), kind="stable")[:N_GAIA_PER_AGN]])
    s.loc[s.idx.isin(used), "group"] = "ctrl_gaia"
    s = s[s.group != ""]
    cols = ["idx", "id", "kind", "group", "ra", "dec", "G", "W1mag", "W2mag", "w12", "milliquas_name", "milliquas_type",
            "milliquas_z", "kim", "kim2026_name", "classprob_quasar", "d_ngc6543_arcmin"]
    s[cols].to_parquet(f"{D}/samples.parquet", index=False)
    print(s.group.value_counts().to_dict(), "W1 range", lo, hi)
    print(s[s.group == "agn"][["idx", "id", "W1mag", "w12", "milliquas_name", "milliquas_type", "milliquas_z",
                               "kim2026_name"]].to_string())


# ---------------------------------------------------------------- photometry of the sample only (E18-B cuts)
def _load_part(args):
    p, srcs, bad = args
    cols = ["src", "n_masked_box", "flags_core", "flux_mJy", "flux_err_mJy", "wl_um", "chi2dof_local", "mjd",
            "detector", "obs_id", "edge_px"]
    q = pd.read_parquet(p, columns=cols)
    q = q[q.src.isin(srcs)]
    ok = np.isfinite(q.flux_mJy) & np.isfinite(q.flux_err_mJy) & (q.flux_err_mJy > 0)
    ok &= (q.n_masked_box <= MAX_MASKED) & (q.edge_px >= MIN_EDGE) & flags_ok(q.flags_core.values)
    key = q.obs_id.astype(str) + "_" + q.detector.astype(str)
    ok &= ~key.isin(bad)
    q = q[ok].drop(columns=["n_masked_box", "flags_core", "edge_px"])
    return q.astype(dict(src=np.int32, detector=np.int8))


def load():
    from concurrent.futures import ProcessPoolExecutor
    s = pd.read_parquet(f"{D}/samples.parquet")
    srcs = set(s.idx.astype(int))
    bad = {f"{o}_{b}" for o, b in anomaly_set()}
    parts = sorted(glob.glob(f"{E18}/phot/part-*.parquet"))
    t0 = time.time()
    with ProcessPoolExecutor(WORKERS) as ex:
        out = list(ex.map(_load_part, [(p, srcs, bad) for p in parts], chunksize=8))
    q = pd.concat(out, ignore_index=True)
    n0 = len(q)
    med = q.groupby("src").chi2dof_local.transform("median")
    q = q[q.chi2dof_local <= np.maximum(CHI2LOC_MIN, CHI2LOC_FAC * med)]
    q["obs_id"] = q.obs_id.astype("category")
    q = q.sort_values(["src", "mjd"]).reset_index(drop=True)
    q.to_parquet(f"{D}/phot_sel.parquet", index=False, row_group_size=500000)
    info = dict(n_parts=len(parts), rows_before_chi2_local=n0, rows=len(q), n_src=int(q.src.nunique()),
                sec=round(time.time() - t0), time=time.strftime("%H:%M:%S"),
                mjd_per_obs_id_max_spread_d=float(q.groupby("obs_id", observed=True).mjd.agg(np.ptp).max()))
    json.dump(info, open(f"{D}/load.json", "w"), indent=1)
    print(info, flush=True)


# ---------------------------------------------------------------- per source
def day_bins(day, a, sa):
    """Within-day 5-sigma clip about the day median, >= 3 pictures: weighted mean, its error, within-day chi2 and dof."""
    o = np.argsort(day, kind="stable")
    day, a, sa = day[o], a[o], sa[o]
    ud, st, cnt = np.unique(day, return_index=True, return_counts=True)
    rd, ra, rs, c2, dof = [], [], [], 0.0, 0
    for u, i0, n in zip(ud, st, cnt):
        if n < MIN_DAY_PICS:
            continue
        x, e = a[i0:i0 + n], sa[i0:i0 + n]
        k = np.abs(x - np.median(x)) <= CLIP_DAY * e
        if k.sum() < MIN_DAY_PICS:
            continue
        x, e = x[k], e[k]
        w = 1 / e ** 2
        m = (w * x).sum() / w.sum()
        rd.append(u); ra.append(m); rs.append(1 / np.sqrt(w.sum()))
        c2 += float((w * (x - m) ** 2).sum()); dof += len(x) - 1
    return np.array(rd), np.array(ra), np.array(rs), c2, dof


def fine_template(lam, det, a, day):
    """Time-averaged residual spectrum: median of a in bins of width lambda/(2R) per detector; bins sampled on
    fewer than TPL_MIN_DAYS distinct days get 0. Returns template per picture and the fraction of pictures in such bins."""
    tpl = np.zeros(len(a))
    thin = 0
    for d in np.unique(det):
        s = np.flatnonzero(det == d)
        lo, hi = DET_RANGE[int(d)]
        nb = int(np.ceil(np.log(hi / lo) * 2 * R_DET[int(d)]))
        b = np.clip(np.floor(np.log(lam[s] / lo) * 2 * R_DET[int(d)]).astype(int), -1, nb)
        for k in np.unique(b):
            i = s[b == k]
            if np.unique(day[i]).size >= TPL_MIN_DAYS:
                tpl[i] = np.median(a[i])
            else:
                thin += len(i)
    return tpl, thin / len(a)


def nxs(ab, sb, scale2=1.0):
    n = len(ab)
    if n < MIN_DAYS_A:
        return np.nan, np.nan
    noise = scale2 * np.mean(sb ** 2)
    return float(np.var(ab, ddof=1) - noise), float(np.sqrt(2 / n) * noise)


def _one(args):
    i, g, s_d, f_d = args
    d = dict(t=g.mjd.values.astype(float), f=g.flux_mJy.values.astype(float), e=g.flux_err_mJy.values.astype(float),
             lam=g.wl_um.values.astype(float), det=g.detector.values.astype(int),
             obs=g.obs_id.astype(str).values)
    d = drop_thin(d)
    if len(d["f"]) < MIN_PICS:
        return None, None, None
    M, sig, _ = static_fit(d["f"], d["e"], d["lam"], d["det"], s_d, f_d)
    v = (M > 0) & ~((d["lam"] >= AIRGLOW[0]) & (d["lam"] <= AIRGLOW[1]))
    t, lam, det, obs = d["t"][v], d["lam"][v], d["det"][v], d["obs"][v]
    a0, sa = (d["f"][v] - M[v]) / M[v], sig[v] / M[v]
    day = np.floor(t - T0).astype(int)
    tpl, thin = fine_template(lam, det, a0, day)
    a = a0 - tpl
    m_ab = float(-2.5 * np.log10(np.median(M) / 3.631e6))
    # (A)
    rowsA = []
    for b, (dd, lo, hi) in enumerate(BINS):
        k = (det == dd) & (lam >= lo) & (lam < hi)
        if k.sum() < MIN_DAY_PICS * MIN_DAYS_A:
            continue
        ud, ab, sb, c2, dof = day_bins(day[k], a[k], sa[k])
        x, ex = nxs(ab, sb)
        k2 = c2 / dof if dof > 0 else np.nan             # within-day empirical noise scale (squared)
        xe, _ = nxs(ab, sb, k2) if np.isfinite(k2) else (np.nan, np.nan)
        _, ab0, sb0, _, _ = day_bins(day[k], a0[k], sa[k])
        x0, _ = nxs(ab0, sb0)
        rowsA.append(dict(src=i, bin=b, n_pics=int(k.sum()), n_days=len(ab), nxs=x, nxs_err=ex, nxs_withinday=xe,
                          withinday_scale2=k2, nxs_spline_only=x0, mean_sb2=float(np.mean(sb ** 2)) if len(sb) else np.nan,
                          M_med=float(np.median(M[v][k])), m_ab=m_ab))
    # (B): simultaneous pairs only (same obs_id, detector d and d+3 both usable for this source)
    short, long_ = det <= 3, det >= 4
    paired = np.isin(obs, np.intersect1d(obs[short], obs[long_]))
    rowsB = None
    ds, s_, ss, _, _ = day_bins(day[paired & short], a[paired & short], sa[paired & short])
    dl, l_, sl, _, _ = day_bins(day[paired & long_], a[paired & long_], sa[paired & long_])
    com, i1, i2 = np.intersect1d(ds, dl, return_indices=True)
    if len(com) >= MIN_DAYS_B:
        rowsB = pd.DataFrame(dict(src=i, day=com, s=s_[i1], s_err=ss[i1], l=l_[i2], l_err=sl[i2]))
    diag = dict(src=i, n_pics=len(a), n_paired=int(paired.sum()), thin_template_frac=thin, m_ab=m_ab,
                n_days_B=len(com))
    return rowsA, rowsB, diag


def persource():
    from concurrent.futures import ProcessPoolExecutor
    g0 = json.load(open(G0))
    s_d = {int(k): v for k, v in g0["s_d"].items()}
    f_d = {int(k): v for k, v in g0["f_d"].items()}
    q = pd.read_parquet(f"{D}/phot_sel.parquet")
    tasks = [(int(i), g, s_d, f_d) for i, g in q.groupby("src")]
    del q
    A, B, Dg = [], [], []
    t0 = time.time()
    with ProcessPoolExecutor(WORKERS) as ex:
        for n, (ra, rb, dg) in enumerate(ex.map(_one, tasks, chunksize=4)):
            if ra:
                A += ra
            if rb is not None:
                B.append(rb)
            if dg:
                Dg.append(dg)
            if n % 100 == 0:
                print(f"{time.strftime('%H:%M:%S')} {n}/{len(tasks)} {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(A).to_parquet(f"{D}/varA.parquet", index=False)
    pd.concat(B, ignore_index=True).to_parquet(f"{D}/lcB.parquet", index=False)
    pd.DataFrame(Dg).to_parquet(f"{D}/diag.parquet", index=False)
    print("persource done", time.strftime("%H:%M:%S"), flush=True)


# ---------------------------------------------------------------- population (A)
def popA(V, srcs, col="nxs", weighted=True):
    """Per bin: weighted mean of col over sources (weights 1/nxs_err^2) or unweighted median."""
    out = np.full(len(BINS), np.nan)
    v = V[V.src.isin(srcs) & np.isfinite(V[col]) & (V.nxs_err > 0)]
    for b, g in v.groupby("bin"):
        if weighted:
            w = 1 / g.nxs_err.values ** 2
            out[b] = (w * g[col].values).sum() / w.sum()
        else:
            out[b] = np.median(g[col].values)
    return out


def bootA(V, srcs, rng, col="nxs", weighted=True, n=N_BOOT):
    srcs = np.asarray(srcs)
    # precompute per-bin sums per source for speed
    v = V[V.src.isin(srcs) & np.isfinite(V[col]) & (V.nxs_err > 0)]
    pos = {s: k for k, s in enumerate(srcs)}
    W, X = np.zeros((len(srcs), len(BINS))), np.zeros((len(srcs), len(BINS)))
    k = v.src.map(pos).values
    w = 1 / v.nxs_err.values ** 2 if weighted else np.ones(len(v))
    W[k, v.bin.values] = w
    X[k, v.bin.values] = v[col].values
    out = np.empty((n, len(BINS)))
    for i in range(n):
        c = np.bincount(rng.integers(0, len(srcs), len(srcs)), minlength=len(srcs))
        if weighted:
            out[i] = (c[:, None] * W * X).sum(0) / np.where((c[:, None] * W).sum(0) > 0, (c[:, None] * W).sum(0), np.nan)
        else:
            m = W > 0
            out[i] = [np.median(np.repeat(X[m[:, b], b], c[m[:, b]])) if (c[m[:, b]] > 0).any() else np.nan
                      for b in range(len(BINS))]
    return out


def fvar(x):
    return np.sign(x) * np.sqrt(np.abs(x))


# ---------------------------------------------------------------- (B) discrete correlation
NK = int(LAG_MAX / LAG_W)
LAGS = np.arange(-NK, NK + 1) * LAG_W


def prep_B(L, inj=None):
    """Per source: standardised s, l and the lag-bin index matrix. inj: dict src -> (s_add, l_add) arrays."""
    out = {}
    for i, g in L.groupby("src"):
        s, l = g.s.values.copy(), g.l.values.copy()
        if inj is not None and i in inj:
            s, l = s + inj[i][0], l + inj[i][1]
        s = (s - s.mean()) / s.std()
        l = (l - l.mean()) / l.std()
        day = g.day.values
        tau = day[None, :] - day[:, None]                  # [i short, j long]: day_j - day_i
        k = np.rint(tau / LAG_W).astype(int)
        m = np.abs(k) <= NK
        out[int(i)] = (s, l, (k + NK)[m], m)
    return out


def ccf_sums(s, l, km, m):
    P = (s[:, None] * l[None, :])[m]
    return np.bincount(km, P, 2 * NK + 1), np.bincount(km, minlength=2 * NK + 1).astype(float)


def stats_C(C):
    pos = (LAGS >= LAG_MIN) & (LAGS <= LAG_MAX)
    neg = (LAGS <= -LAG_MIN) & (LAGS >= -LAG_MAX)
    ip = np.flatnonzero(pos)
    j = ip[np.nanargmax(C[ip])]
    cp = C[j]
    lo = hi = j
    while lo - 1 >= ip[0] and C[lo - 1] >= 0.8 * cp:
        lo -= 1
    while hi + 1 <= ip[-1] and C[hi + 1] >= 0.8 * cp:
        hi += 1
    cc = C[lo:hi + 1]
    cent = float((LAGS[lo:hi + 1] * cc).sum() / cc.sum()) if cc.sum() > 0 else float(LAGS[j])
    return dict(C_pos=float(cp), tau_peak=float(LAGS[j]), tau_cent=cent, C_neg=float(np.nanmax(C[neg])),
                C0=float(C[NK]))


def drw(rng, n, tau, sig):
    x = np.empty(n)
    x[0] = rng.normal(0, sig)
    a = np.exp(-1 / tau)
    e = rng.normal(0, sig * np.sqrt(1 - a * a), n)
    for k in range(1, n):
        x[k] = a * x[k - 1] + e[k]
    return x


def analyze():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(SEED)
    S = pd.read_parquet(f"{D}/samples.parquet")
    V = pd.read_parquet(f"{D}/varA.parquet")
    L = pd.read_parquet(f"{D}/lcB.parquet")
    Dg = pd.read_parquet(f"{D}/diag.parquet")
    grp = {g: S.idx[S.group == g].astype(int).values for g in ("agn", "ctrl_aw", "ctrl_gaia", "aw_red")}
    wl = np.array([np.sqrt(lo * hi) for _, lo, hi in BINS])
    res = dict(experiment="E25", protocol="docs/protocols/E25.md", time=time.strftime("%Y-%m-%d %H:%M:%S"))
    inA = {g: sorted(set(V.src[V.src.isin(v)])) for g, v in grp.items()}
    res["n_agn"], res["n_control"] = len(inA["agn"]), len(inA["ctrl_aw"])
    res["n_control_gaia"], res["n_aw_red"] = len(inA["ctrl_gaia"]), len(inA["aw_red"])
    # ------------------------------ (A)
    A = {}
    for g in ("agn", "ctrl_aw", "ctrl_gaia", "aw_red"):
        if not inA[g]:
            continue
        val = popA(V, inA[g])
        bs = bootA(V, inA[g], rng)
        A[g] = dict(val=val, err=np.nanstd(bs, 0), bs=bs)
    d = A["agn"]["val"] - A["ctrl_aw"]["val"]
    dz = d / np.hypot(A["agn"]["err"], A["ctrl_aw"]["err"])
    nge3 = int(np.nansum(dz >= 3))
    ratio = np.where(A["ctrl_aw"]["val"] > 2 * A["ctrl_aw"]["err"], A["agn"]["val"] / A["ctrl_aw"]["val"], np.nan)
    rnd = lambda x, k=6: [None if not np.isfinite(v) else round(float(v), k) for v in x]
    res.update(wl_um=rnd(wl, 3), wl_bin_edges_um=[[round(lo, 3), round(hi, 3)] for _, lo, hi in BINS],
               agn=rnd(A["agn"]["val"]), agn_err=rnd(A["agn"]["err"]),
               ctrl=rnd(A["ctrl_aw"]["val"]), ctrl_err=rnd(A["ctrl_aw"]["err"]),
               ctrl_gaia=rnd(A["ctrl_gaia"]["val"]), ctrl_gaia_err=rnd(A["ctrl_gaia"]["err"]),
               agn_fvar=rnd(fvar(A["agn"]["val"]), 4), agn_fvar_p16_p84=[rnd(np.nanpercentile(fvar(A["agn"]["bs"]), q, 0), 4) for q in (16, 84)],
               ctrl_fvar=rnd(fvar(A["ctrl_aw"]["val"]), 4),
               agn_minus_ctrl=rnd(d), agn_minus_ctrl_z=rnd(dz, 2), ratio_agn_over_ctrl=rnd(ratio, 3),
               n_bins_ge3sigma=nge3, A_passed=bool(nge3 >= 3),
               var_spectrum_units="sigma^2_NXS (fractional excess variance of 1-d binned residuals); *_fvar = signed sqrt")
    if "aw_red" in A:
        res["aw_red"], res["aw_red_err"] = rnd(A["aw_red"]["val"]), rnd(A["aw_red"]["err"])
    rob = {}
    for name, col, wt in (("withinday_noise", "nxs_withinday", True), ("spline_only", "nxs_spline_only", True),
                          ("unweighted_median", "nxs", False)):
        a_, c_ = popA(V, inA["agn"], col, wt), popA(V, inA["ctrl_aw"], col, wt)
        ea = np.nanstd(bootA(V, inA["agn"], rng, col, wt, 500), 0)
        ec = np.nanstd(bootA(V, inA["ctrl_aw"], rng, col, wt, 500), 0)
        z = (a_ - c_) / np.hypot(ea, ec)
        rob[name] = dict(agn=rnd(a_), agn_err=rnd(ea), ctrl=rnd(c_), ctrl_err=rnd(ec), z=rnd(z, 2),
                         n_bins_ge3sigma=int(np.nansum(z >= 3)))
    zg = (A["agn"]["val"] - A["ctrl_gaia"]["val"]) / np.hypot(A["agn"]["err"], A["ctrl_gaia"]["err"])
    rob["vs_ctrl_gaia"] = dict(z=rnd(zg, 2), n_bins_ge3sigma=int(np.nansum(zg >= 3)))
    loo = {}
    for s in inA["agn"]:
        keep = [x for x in inA["agn"] if x != s]
        a_ = popA(V, keep)
        ea = np.nanstd(bootA(V, keep, rng, n=300), 0)
        z = (a_ - A["ctrl_aw"]["val"]) / np.hypot(ea, A["ctrl_aw"]["err"])
        loo[int(s)] = int(np.nansum(z >= 3))
    rob["leave_one_agn_out_n_bins_ge3sigma"] = loo
    va = V[V.src.isin(inA["agn"]) & np.isfinite(V.nxs) & (V.nxs_err > 0)].copy()
    va["w"] = 1 / va.nxs_err ** 2
    va["wfrac"] = va.w / va.groupby("bin").w.transform("sum")
    top = va.loc[va.groupby("bin").wfrac.idxmax()]
    rob["top_weight_agn_per_bin"] = [dict(bin=int(r.bin), src=int(r.src), weight_frac=round(float(r.wfrac), 3),
                                          nxs=round(float(r.nxs), 6), nxs_err=round(float(r.nxs_err), 6))
                                     for r in top.itertuples()]
    rob["per_agn_nxs_over_err"] = {int(s): rnd((g.set_index("bin").nxs / g.set_index("bin").nxs_err)
                                                .reindex(range(len(BINS))).values, 2) for s, g in va.groupby("src")}
    rob["n_agn_per_bin"] = va.groupby("bin").src.nunique().reindex(range(len(BINS))).fillna(0).astype(int).tolist()
    rob["withinday_scale2_median"] = {g: rnd(V[V.src.isin(inA[g])].groupby("bin").withinday_scale2.median()
                                             .reindex(range(len(BINS))).values, 3) for g in ("agn", "ctrl_aw")}
    res["A_robustness"] = rob
    print("A:", nge3, "bins >= 3 sigma; z =", rnd(dz, 2), flush=True)
    # ------------------------------ (B)
    PB = prep_B(L)
    agnB = [s for s in grp["agn"] if s in PB]
    ctlB = [s for s in grp["ctrl_aw"] if s in PB]
    gaiB = [s for s in grp["ctrl_gaia"] if s in PB]
    sums = {s: ccf_sums(*PB[s]) for s in PB}

    def stack(srcs, w=None):
        num = sum(sums[s][0] for s in srcs)
        den = sum(sums[s][1] for s in srcs)
        return num / np.where(den > 0, den, np.nan)

    Ca = stack(agnB)
    st = stats_C(Ca)
    # N1 shuffles
    n1 = np.empty(N_SHUF)
    for r in range(N_SHUF):
        num, den = 0, 0
        for s in agnB:
            sv, lv, km, m = PB[s]
            a_, b_ = ccf_sums(sv, rng.permutation(lv), km, m)
            num, den = num + a_, den + b_
        n1[r] = stats_C(num / den)["C_pos"]
    shuf_curves_q = None
    # N2 control draws
    nA = len(agnB)
    Snum = np.array([sums[s][0] for s in ctlB]); Sden = np.array([sums[s][1] for s in ctlB])
    n2, n2c = np.empty(N_CTRL_DRAW), []
    for r in range(N_CTRL_DRAW):
        k = rng.choice(len(ctlB), nA, replace=False)
        C = Snum[k].sum(0) / Sden[k].sum(0)
        n2[r] = stats_C(C)["C_pos"]
        n2c.append(C)
    n2c = np.array(n2c)
    Cc, Cg = stack(ctlB), stack(gaiB)
    p99_1, p99_2 = float(np.quantile(n1, 0.99)), float(np.quantile(n2, 0.99))
    B_pass = bool(st["C_pos"] > p99_1 and st["C_pos"] > p99_2 and st["C_pos"] > st["C_neg"])
    # bootstrap lag
    An = np.array([sums[s][0] for s in agnB]); Ad = np.array([sums[s][1] for s in agnB])
    bt = []
    for r in range(N_BOOT):
        c = np.bincount(rng.integers(0, nA, nA), minlength=nA)
        bt.append(stats_C((c[:, None] * An).sum(0) / (c[:, None] * Ad).sum(0)))
    bc = np.array([b["tau_cent"] for b in bt]); bp = np.array([b["tau_peak"] for b in bt])
    lag_err = float((np.percentile(bc, 84) - np.percentile(bc, 16)) / 2)
    loo = {int(s): {k: round(v, 4) for k, v in stats_C(stack([x for x in agnB if x != s])).items()} for s in agnB}
    # common-mode corrected (descriptive)
    Lc = L[L.src.isin(ctlB)]
    cm = Lc.assign(ws=1 / Lc.s_err ** 2, wl_=1 / Lc.l_err ** 2)
    cm = cm.assign(xs=cm.ws * cm.s, xl=cm.wl_ * cm.l).groupby("day")[["ws", "wl_", "xs", "xl"]].sum()
    cm = pd.DataFrame(dict(s=cm["xs"] / cm["ws"], l=cm["xl"] / cm["wl_"]))
    L2 = L.copy()
    L2["s"] = L2.s - cm.s.reindex(L2.day).fillna(0).values
    L2["l"] = L2.l - cm.l.reindex(L2.day).fillna(0).values
    PB2 = prep_B(L2)
    sums2 = {s: ccf_sums(*PB2[s]) for s in PB2}
    st_cm = stats_C(sum(sums2[s][0] for s in agnB) / sum(sums2[s][1] for s in agnB))
    n2cm = []
    S2n = np.array([sums2[s][0] for s in ctlB]); S2d = np.array([sums2[s][1] for s in ctlB])
    for r in range(500):
        k = rng.choice(len(ctlB), nA, replace=False)
        n2cm.append(stats_C(S2n[k].sum(0) / S2d[k].sum(0))["C_pos"])
    # injections (controls only)
    inj = {}
    Lc_by = {int(i): g for i, g in L[L.src.isin(ctlB)].groupby("src")}
    for sig in (0.05, 0.10, 0.20):
        for lag in (50, 100):
            rec, taus = 0, []
            for r in range(200):
                k = rng.choice(len(ctlB), nA, replace=False)
                add = {}
                for j in k:
                    s = ctlB[j]
                    g = Lc_by[s]
                    x = drw(rng, 400 + lag + 1, 200.0, sig)
                    add[s] = (x[g.day.values + lag], x[g.day.values])     # long(t) = short(t - lag)
                Pi = prep_B(L[L.src.isin([ctlB[j] for j in k])], add)
                num = sum(ccf_sums(*Pi[s])[0] for s in Pi)
                den = sum(ccf_sums(*Pi[s])[1] for s in Pi)
                si = stats_C(num / den)
                taus.append(si["tau_cent"])
                rec += int(si["C_pos"] > p99_2 and si["C_pos"] > si["C_neg"] and abs(si["tau_cent"] - lag) <= 25)
            inj[f"sigma{sig:.2f}_lag{lag}"] = dict(recovered_frac=rec / 200, median_tau_cent=float(np.median(taus)))
            print("inj", sig, lag, rec / 200, flush=True)
    res.update(n_agn_B=nA, n_control_B=len(ctlB), n_control_gaia_B=len(gaiB),
               lag_days=round(st["tau_cent"], 1), lag_err=round(lag_err, 1), lag_peak_days=st["tau_peak"],
               lag_boot_median=float(np.median(bc)), lag_peak_boot_p16_p84=[float(np.percentile(bp, 16)), float(np.percentile(bp, 84))],
               ccf_peak=round(st["C_pos"], 5), ccf_neg_side_max=round(st["C_neg"], 5), ccf_zero_lag=round(st["C0"], 5),
               null_p99=round(p99_1, 5), null_p99_ctrl_draws=round(p99_2, 5),
               p_shuffle=float((1 + (n1 >= st["C_pos"]).sum()) / (1 + N_SHUF)),
               p_ctrl_draws=float((1 + (n2 >= st["C_pos"]).sum()) / (1 + N_CTRL_DRAW)),
               B_passed=B_pass, ccf_lags_d=LAGS.tolist(), ccf_agn=rnd(Ca, 5), ccf_ctrl=rnd(Cc, 5), ccf_ctrl_gaia=rnd(Cg, 5),
               ccf_ctrl_draws_p1_p99=[rnd(np.nanpercentile(n2c, q, 0), 5) for q in (1, 99)],
               ctrl_zero_lag=round(float(Cc[NK]), 5), ctrl_gaia_zero_lag=round(float(Cg[NK]), 5),
               B_leave_one_out=loo, B_common_mode_corrected=dict(**{k: round(v, 4) for k, v in st_cm.items()},
                                                                  ctrl_draws_p99=round(float(np.quantile(n2cm, 0.99)), 5)),
               B_injections=inj, B_days_per_agn={int(s): int(len(PB[s][0])) for s in agnB},
               diag_thin_template_frac_median=float(Dg.thin_template_frac.median()),
               diag_n_days_B_median={g: float(Dg[Dg.src.isin(grp[g])].n_days_B.median()) for g in ("agn", "ctrl_aw")})
    print("B:", st, "p99 shuffle", p99_1, "p99 ctrl", p99_2, "pass", B_pass, "lag err", lag_err, flush=True)
    json.dump(res, open(f"{D}/analyze_raw.json", "w"), indent=1, default=float)
    # ------------------------------ figures
    os.makedirs(FIGS, exist_ok=True)
    ph = json.load(open(f"{D}/posthoc.json")) if os.path.exists(f"{D}/posthoc.json") else None
    fig, axs = plt.subplots(2 if ph else 1, 1, figsize=(7, 7.2 if ph else 4.2), sharex=True, squeeze=False)
    ax = axs[0, 0]
    for g, c, lab, off in (("agn", "C3", f"AGN (n={res['n_agn']})", 1.0), ("ctrl_aw", "C0", f"AllWISE-only controls (n={res['n_control']})", 0.985),
                           ("ctrl_gaia", "0.5", f"Gaia stars, W1-matched (n={res['n_control_gaia']})", 1.015)):
        y = fvar(A[g]["val"])
        lo_, hi_ = np.nanpercentile(fvar(A[g]["bs"]), 16, 0), np.nanpercentile(fvar(A[g]["bs"]), 84, 0)
        ax.errorbar(wl * off, y, yerr=[y - lo_, hi_ - y], fmt="o-" if g == "agn" else "s--", color=c, ms=4, lw=1,
                    capsize=2, label=lab)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xscale("log")
    ax.set_xticks([0.8, 1, 1.5, 2, 3, 4, 5]); ax.set_xticklabels(["0.8", "1", "1.5", "2", "3", "4", "5"])
    ax.set_ylabel(r"$F_{\rm var}$ (signed $\sqrt{\sigma^2_{\rm NXS}}$), weighted mean")
    ax.set_title("SPHEREx NEP deep field: fractional excess variability vs wavelength")
    ax.legend(frameon=False, fontsize=8)
    if ph:
        from run_e18b import wilson
        ax = axs[1, 0]
        r = ph["A_per_source_withinday_z_gt3"]
        for g, c, lab, off, fmt in (("agn", "C3", "AGN", 1.0, "o-"), ("ctrl", "C0", "AllWISE-only controls", 0.985, "s--")):
            k = np.array([r[str(b)][f"{g}_k"] for b in range(len(BINS))]); n = np.array([r[str(b)][f"{g}_n"] for b in range(len(BINS))])
            ci = np.array([wilson(int(kk), int(nn)) for kk, nn in zip(k, n)], float)
            ax.errorbar(wl * off, k / n, yerr=[k / n - ci[:, 0], ci[:, 1] - k / n], fmt=fmt, color=c, ms=4, lw=1, capsize=2, label=lab)
        ax.set_ylabel("fraction of sources with excess\nvariance > 3σ (within-day noise; post hoc)")
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel("observed wavelength (µm)")
    fig.tight_layout(); fig.savefig(f"{FIGS}/e25_var_spectrum.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    q1, q99 = np.nanpercentile(n2c, 1, 0), np.nanpercentile(n2c, 99, 0)
    ax.fill_between(LAGS, q1, q99, color="C0", alpha=0.2, lw=0, label=f"random {nA}-control stacks, 1–99%")
    ax.plot(LAGS, Cc, color="C0", lw=1, label=f"all AllWISE-only controls (n={len(ctlB)})")
    ax.plot(LAGS, Ca, "o-", color="C3", ms=3, lw=1.2, label=f"AGN stack (n={nA})")
    ax.axhline(p99_1, color="C3", ls=":", lw=1, label="AGN time-shuffle null, 99th pct of peak")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("lag (d); positive = 2.4–5 µm lags 0.75–2.4 µm")
    ax.set_ylabel("stacked discrete correlation")
    ax.set_title("Simultaneous short/long-band residual light curves")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(f"{FIGS}/e25_ccf.png", dpi=150); plt.close(fig)
    return res


def posthoc():
    """Descriptive checks decided after the frozen results were seen (labelled post hoc; no verdict changes)."""
    from scipy.stats import fisher_exact
    rng = np.random.default_rng(SEED + 1)
    S = pd.read_parquet(f"{D}/samples.parquet")
    V = pd.read_parquet(f"{D}/varA.parquet")
    L = pd.read_parquet(f"{D}/lcB.parquet")
    grp = {g: S.idx[S.group == g].astype(int).values for g in ("agn", "ctrl_aw")}
    rnd = lambda x, k=5: [None if not np.isfinite(v) else round(float(v), k) for v in x]
    ph = dict(note="post hoc, after the frozen (A) and (B) results were seen; descriptive only, no verdict changed")
    # ---- A1: per-source significance with the within-day noise scale
    V["z_wd"] = V.nxs_withinday / (V.nxs_err * V.withinday_scale2)
    a1 = {}
    for b in range(len(BINS)):
        va, vc = V[(V.bin == b) & V.src.isin(grp["agn"])], V[(V.bin == b) & V.src.isin(grp["ctrl_aw"])]
        ka, kc = int((va.z_wd > 3).sum()), int((vc.z_wd > 3).sum())
        a1[b] = dict(agn_k=ka, agn_n=len(va), ctrl_k=kc, ctrl_n=len(vc),
                     fisher_p_greater=float(fisher_exact([[ka, len(va) - ka], [kc, len(vc) - kc]], "greater")[1]))
    ph["A_per_source_withinday_z_gt3"] = a1
    ph["A_per_agn_withinday_z"] = {int(s): rnd(g.set_index("bin").z_wd.reindex(range(len(BINS))).values, 1)
                                   for s, g in V[V.src.isin(grp["agn"])].groupby("src")}
    # ---- A2: unweighted median of the within-day-noise sigma2_NXS, bootstrap over sources
    def med(srcs):
        v = V[V.src.isin(srcs) & np.isfinite(V.nxs_withinday)]
        return v.groupby("bin").nxs_withinday.median().reindex(range(len(BINS))).values
    out = {}
    for g in ("agn", "ctrl_aw"):
        s = np.array(sorted(set(V.src[V.src.isin(grp[g])])))
        bs = np.array([med(rng.choice(s, len(s))) for _ in range(500)])
        out[g] = (med(s), np.nanstd(bs, 0))
    z = (out["agn"][0] - out["ctrl_aw"][0]) / np.hypot(out["agn"][1], out["ctrl_aw"][1])
    ph["A_median_withinday"] = dict(agn=rnd(out["agn"][0], 6), agn_err=rnd(out["agn"][1], 6), ctrl=rnd(out["ctrl_aw"][0], 6),
                                    ctrl_err=rnd(out["ctrl_aw"][1], 6), z=rnd(z, 2), n_bins_ge3sigma=int(np.nansum(z >= 3)))
    # ---- B: global-peak centroid, asymmetry, per-source drivers
    PB = prep_B(L)
    agnB = [s for s in grp["agn"] if s in PB]
    ctlB = [s for s in grp["ctrl_aw"] if s in PB]
    sums = {s: ccf_sums(*PB[s]) for s in PB}
    pos = (LAGS >= LAG_MIN) & (LAGS <= LAG_MAX)
    neg = pos[::-1]

    def gcent(C):
        j = int(np.nanargmax(C))
        lo = hi = j
        while lo - 1 >= 0 and C[lo - 1] >= 0.8 * C[j]:
            lo -= 1
        while hi + 1 < len(C) and C[hi + 1] >= 0.8 * C[j]:
            hi += 1
        return float((LAGS[lo:hi + 1] * C[lo:hi + 1]).sum() / C[lo:hi + 1].sum())

    def asym(C):                                   # mean over tau of C(+tau) - C(-tau)
        return float(np.mean(C[pos] - C[np.flatnonzero(neg)[::-1]]))
    An = np.array([sums[s][0] for s in agnB]); Ad = np.array([sums[s][1] for s in agnB])
    Cn = np.array([sums[s][0] for s in ctlB]); Cd = np.array([sums[s][1] for s in ctlB])
    Ca = An.sum(0) / Ad.sum(0)
    nA = len(agnB)
    bt_c, bt_a = [], []
    for _ in range(N_BOOT):
        c = np.bincount(rng.integers(0, nA, nA), minlength=nA)
        C = (c[:, None] * An).sum(0) / (c[:, None] * Ad).sum(0)
        bt_c.append(gcent(C)); bt_a.append(asym(C))
    sh = []
    for _ in range(500):
        num, den = 0, 0
        for s in agnB:
            sv, lv, km, m = PB[s]
            a_, b_ = ccf_sums(sv, rng.permutation(lv), km, m)
            num, den = num + a_, den + b_
        sh.append(asym(num / den))
    cd = []
    for _ in range(N_CTRL_DRAW):
        k = rng.choice(len(ctlB), nA, replace=False)
        cd.append(asym(Cn[k].sum(0) / Cd[k].sum(0)))
    # zero-lag red-noise null: DRW (tau 200 d, rms 0.10) added identically to both bands of random control draws
    Lc_by = {int(i): g for i, g in L[L.src.isin(ctlB)].groupby("src")}
    rn, rn_c = [], []
    for _ in range(300):
        k = rng.choice(len(ctlB), nA, replace=False)
        add = {}
        for j in k:
            g = Lc_by[ctlB[j]]
            x = drw(rng, 401, 200.0, 0.10)
            add[ctlB[j]] = (x[g.day.values], x[g.day.values])
        Pi = prep_B(L[L.src.isin([ctlB[j] for j in k])], add)
        ss = [ccf_sums(*Pi[s]) for s in Pi]
        C = sum(x[0] for x in ss) / sum(x[1] for x in ss)
        rn.append(asym(C)); rn_c.append(gcent(C))
    A0 = asym(Ca)
    pct = lambda x, q: float(np.percentile(x, q))
    ph["B_global_peak_centroid"] = dict(lag_days=round(gcent(Ca), 2), boot_median=pct(bt_c, 50),
                                        boot_err=(pct(bt_c, 84) - pct(bt_c, 16)) / 2,
                                        zero_lag_rednoise_null_p1_p99=[pct(rn_c, 1), pct(rn_c, 99)],
                                        rule="centroid of the contiguous bins around the global maximum of the stacked CCF with C >= 0.8 Cmax")
    ph["B_asymmetry"] = dict(statistic="mean over tau = 10..150 d of C(+tau) - C(-tau)", value=round(A0, 5),
                             boot_err=round(float(np.std(bt_a)), 5),
                             shuffle_p99=round(pct(sh, 99), 5), p_shuffle=float((1 + np.sum(np.array(sh) >= A0)) / 501),
                             ctrl_draws_p99=round(pct(cd, 99), 5), p_ctrl_draws=float((1 + np.sum(np.array(cd) >= A0)) / (1 + len(cd))),
                             zero_lag_rednoise_p99=round(pct(rn, 99), 5), p_zero_lag_rednoise=float((1 + np.sum(np.array(rn) >= A0)) / 301))
    # per-source: broad correlation |tau| <= 30 d and asymmetry, percentile among controls
    near = np.abs(LAGS) <= 30
    def per(s):
        n, d = sums[s]
        C = n / np.where(d > 0, d, np.nan)
        return float(n[near].sum() / d[near].sum()), asym(C)
    ctl = np.array([per(s) for s in ctlB])
    rows = {}
    for s in agnB:
        b, a = per(s)
        rows[int(s)] = dict(C_near=round(b, 4), pct_vs_ctrl=round(float((ctl[:, 0] < b).mean() * 100), 1),
                            asym=round(a, 4), asym_pct_vs_ctrl=round(float((ctl[:, 1] < a).mean() * 100), 1),
                            loo_asym=round(asym(np.delete(An, agnB.index(s), 0).sum(0) / np.delete(Ad, agnB.index(s), 0).sum(0)), 5))
    ph["B_per_agn"] = rows
    ph["B_ctrl_C_near_p50_p95_p99"] = [pct(ctl[:, 0], q) for q in (50, 95, 99)]
    ph["B_n_agn_C_near_above_ctrl_p99"] = int(sum(r["pct_vs_ctrl"] > 99 for r in rows.values()))
    json.dump(ph, open(f"{D}/posthoc.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in ph.items() if k not in ("A_per_agn_withinday_z",)}, indent=1))


def finalize():
    """Write data/findings/e25.json from analyze_raw.json plus notes."""
    res = json.load(open(f"{D}/analyze_raw.json"))
    res["notes"] = open(f"{D}/notes.txt").read().strip() if os.path.exists(f"{D}/notes.txt") else ""
    if os.path.exists(f"{D}/posthoc.json"):
        res["post_hoc"] = json.load(open(f"{D}/posthoc.json"))
    json.dump(res, open(OUT, "w"), indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    globals()[sys.argv[1]]()
