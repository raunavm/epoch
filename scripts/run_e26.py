"""E26: learned systematics correction (cotrending) for the SPHEREx NEP deep-field photometry
(docs/protocols/E26.md). Run from the repo root:
  .venv/bin/python scripts/run_e26.py prep      # E18-B usable rows -> data/phase2/e26/phot.parquet (same rule as run_e18b.prep)
  .venv/bin/python scripts/run_e26.py resid     # static-model residuals of the calibration-half null stars (training data)
  .venv/bin/python scripts/run_e26.py select    # model selection + final correction, calibration half only
  .venv/bin/python scripts/run_e26.py stats before   # E18-B statistics, uncorrected (E18-B + 1% injections)
  .venv/bin/python scripts/run_e26.py gate0     # E18-B Gate-0 rule on corrected photometry
  .venv/bin/python scripts/run_e26.py stats after    # same statistics on corrected photometry
  .venv/bin/python scripts/run_e26.py analyze   # thresholds, false alarms, recall/floors, known variables, json, figure
"""
import os, sys, time, json, glob

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_e18b as B

D = "data/phase2/e26"
G0_E18B = "data/phase2/e18b/gate0.json"
B.D = D                                            # run_e18b.load_src reads f"{B.D}/phot.parquet"


def prep():
    os.makedirs(D, exist_ok=True)
    B.prep()


def g0_dicts(g0):
    return {int(k): v for k, v in g0["s_d"].items()}, {int(k): v for k, v in g0["f_d"].items()}


def static_fit_slope(f, e, lam, det, s_d, f_d):
    """Same fit as run_e18b.static_fit, also returning dM/dlambda of the fitted spline."""
    from scipy.interpolate import BSpline
    A = B._spline_design(lam, det)
    sd = np.array([s_d[int(d)] for d in det]) * e
    fd = np.array([f_d[int(d)] for d in det])
    use = np.ones(len(f), bool)
    sig = sd
    for _ in range(3):
        w = 1 / sig[use]
        c, *_ = np.linalg.lstsq(A[use] * w[:, None], f[use] * w, rcond=None)
        M = A @ c
        sig = np.sqrt(sd ** 2 + (fd * np.abs(M)) ** 2)
        use = np.abs((f - M) / sig) < B.CLIP
    dM = np.zeros(len(f))
    nb = B.N_KNOTS + 4
    for j, d in enumerate(np.unique(det)):
        s = det == d
        a, b = lam[s].min() - 1e-6, lam[s].max() + 1e-6
        t = np.r_[[a] * 4, np.linspace(a, b, B.N_KNOTS + 2)[1:-1], [b] * 4]
        dM[s] = BSpline(t, c[nb * j:nb * (j + 1)], 3).derivative()(lam[s])
    return M, sig, dM


def _resid_work(args):
    chunk, g0 = args
    s_d, f_d = g0_dicts(g0)
    data = B.load_src(chunk)
    out = []
    for i in chunk:
        if i not in data:
            continue
        d = B.drop_thin(data[i])
        if len(d["f"]) < B.MIN_PICS:
            continue
        M, sig, dM = static_fit_slope(d["f"], d["e"], d["lam"], d["det"], s_d, f_d)
        v = M > 0
        out.append(pd.DataFrame(dict(src=np.int32(i), pic=d["pic"][v], det=d["det"][v].astype(np.int8),
                                     t=d["t"][v], x=d["x"][v], y=d["y"][v], lam=d["lam"][v].astype(np.float32),
                                     M=M[v].astype(np.float32), a=((d["f"][v] - M[v]) / M[v]).astype(np.float32),
                                     sa=(sig[v] / M[v]).astype(np.float32), slope=(dM[v] / M[v]).astype(np.float32))))
    return pd.concat(out, ignore_index=True) if out else None


def resid():
    """Training data: static-model residuals of calibration-half null stars only (no injections exist in this set)."""
    from concurrent.futures import ProcessPoolExecutor
    g0 = json.load(open(G0_E18B))
    s = B.populations()
    idx = s.idx[s.half == "cal"].values
    chunks = [idx[i:i + 40] for i in range(0, len(idx), 40)]
    with ProcessPoolExecutor(3) as ex:
        parts = [p for p in ex.map(_resid_work, [(c, g0) for c in chunks]) if p is not None]
    r = pd.concat(parts, ignore_index=True)
    r.to_parquet(f"{D}/resid_cal.parquet", index=False)
    print(len(r), r.src.nunique(), time.strftime("%H:%M:%S"), flush=True)


# ---------------------------------------------------------------- the correction model (per picture + per-detector map)
SEED = 2026111526
NPIX, NCELL = 2040, 16                             # per-detector map: 16 x 16 cells of 127.5 px
MIN_TRAIN = 30                                     # picture terms only for pictures with >= 30 training points
CLIP_T = 5.0                                       # training points: |pull| < 5 (before and after the first fit)
N_BACKFIT = 3
L_CLIP = (-1.9, 1.5)                               # log10(M/mJy) clipped to the 1st-99th percentile of training rows
MODELS = {"M0": ([], False), "M1": (["z"], False), "M2": (["z"], True), "M3": (["z", "u", "v"], True),
          "M4": (["z", "u", "v", "s"], True), "M5": (["z", "u", "v", "s", "b"], True),
          "M6": (["z", "l"], True), "M7": (["z", "l", "s"], True)}


def features(r, terms):
    cols = {"z": lambda: np.ones(len(r)), "u": lambda: (r.x.values - NPIX / 2) / (NPIX / 2),
            "v": lambda: (r.y.values - NPIX / 2) / (NPIX / 2), "s": lambda: r.slope.values.astype(float),
            "b": lambda: 1.0 / r.M.values.astype(float),
            "l": lambda: np.clip(np.log10(r.M.values.astype(float)), *L_CLIP) - 0.5}
    return np.stack([cols[t]() for t in terms], 1) if terms else np.zeros((len(r), 0))


def cell(r):
    cx = np.clip((r.x.values / NPIX * NCELL).astype(int), 0, NCELL - 1)
    cy = np.clip((r.y.values / NPIX * NCELL).astype(int), 0, NCELL - 1)
    return (r.det.values.astype(int) - 1) * NCELL * NCELL + cy * NCELL + cx


def _pic_solve(pic, X, y, w, npic):
    K = X.shape[1]
    A = np.zeros((npic, K, K))
    b = np.zeros((npic, K))
    for j in range(K):
        b[:, j] = np.bincount(pic, w * X[:, j] * y, npic)
        for k in range(j, K):
            A[:, j, k] = A[:, k, j] = np.bincount(pic, w * X[:, j] * X[:, k], npic)
    n = np.bincount(pic, minlength=npic)
    ok = n >= MIN_TRAIN
    th = np.zeros((npic, K))
    if K:
        tr = np.trace(A, axis1=1, axis2=2)[ok]
        th[ok] = np.linalg.solve(A[ok] + 1e-10 * tr[:, None, None] * np.eye(K), b[ok][..., None])[..., 0]
    return th


def fit_model(r, name, npic):
    """Learn the correction from training rows r (columns pic, det, x, y, slope, M, a, sa). Returns params dict."""
    terms, use_map = MODELS[name]
    X = features(r, terms)
    pic, c = r.pic.values, cell(r)
    a, w = r.a.values.astype(float), 1 / r.sa.values.astype(float) ** 2
    keep = np.abs(a * np.sqrt(w)) < CLIP_T
    th, mp = np.zeros((npic, len(terms))), np.zeros(6 * NCELL * NCELL)
    for it in range(2):                              # pass 2 re-clips on the corrected pull
        for _ in range(N_BACKFIT if use_map else 1):
            k = keep
            if terms:
                th = _pic_solve(pic[k], X[k], a[k] - mp[c[k]], w[k], npic)
            if use_map:
                res = a[k] - (X[k] * th[pic[k]]).sum(1)
                mp = np.bincount(c[k], w[k] * res, len(mp)) / np.maximum(np.bincount(c[k], w[k], len(mp)), 1e-30)
        pred = (X * th[pic]).sum(1) + mp[c]
        keep = np.abs((a - pred) * np.sqrt(w)) < CLIP_T
    return dict(name=name, terms=terms, theta=th, map=mp)


def predict(P, r):
    X = features(r, P["terms"])
    return (X * P["theta"][r.pic.values]).sum(1) + P["map"][cell(r)]


T0 = 60789.880277196                               # global day grid: floor(mjd - T0) (first usable picture)
NDAY = 386
KMAX = 15                                          # SysRem components tried (K = 0..15)
K_TOL = 0.01                                       # K = smallest K with CV score within 1% of the minimum
SR_ITER = 30
SR_CLIP, SR_STAR_MAX = 8.0, 10.0                   # CBV training: day bins |z| < 8; stars with day-binned chi2/bin < 10


def day_bins(src, t, a, w):
    """Per (source, global day) weighted mean residual; bins with >= 3 pictures (as run_e18b.binned)."""
    day = np.floor(t - T0).astype(int)
    key = pd.MultiIndex.from_arrays([src, day])
    g = pd.DataFrame(dict(sw=w, swa=w * a, n=1), index=key).groupby(level=[0, 1]).sum()
    g = g[g.n >= B.MIN_BIN]
    return g.swa.values / g.sw.values, 1 / np.sqrt(g.sw.values), g.index.get_level_values(0).values, \
        g.index.get_level_values(1).values


def sysrem(src, day, ab, sb, K):
    """SysRem (Tamuz et al. 2005) on the (star x day) matrix of day-binned fractional residuals, weights 1/sb^2.
    Returns V (K x NDAY), each component normalised to unit rms over the days it covers."""
    us, si = np.unique(src, return_inverse=True)
    w = 1 / sb ** 2
    w = np.where(np.abs(ab / sb) < SR_CLIP, w, 0.0)
    chi = pd.Series((ab / sb) ** 2).groupby(si).mean().values
    w = np.where(chi[si] < SR_STAR_MAX, w, 0.0)
    r = ab.copy()
    V = np.zeros((K, NDAY))
    for k in range(K):
        av = np.ones(NDAY)
        for _ in range(SR_ITER):
            c = np.bincount(si, w * r * av[day], len(us)) / np.maximum(np.bincount(si, w * av[day] ** 2, len(us)), 1e-30)
            av = np.bincount(day, w * r * c[si], NDAY) / np.maximum(np.bincount(day, w * c[si] ** 2, NDAY), 1e-30)
        cov = np.bincount(day, w, NDAY) > 0
        av = np.where(cov, av, 0.0)
        av /= np.sqrt(np.mean(av[cov] ** 2))
        c = np.bincount(si, w * r * av[day], len(us)) / np.maximum(np.bincount(si, w * av[day] ** 2, len(us)), 1e-30)
        r = r - c[si] * av[day]
        V[k] = av
    return V


def cbv_coef(V, day, ab, sb):
    """WLS coefficients of one star's day-binned residuals on the CBVs."""
    if len(V) == 0:
        return np.zeros(0)
    X = V[:, day].T / sb[:, None]
    c, *_ = np.linalg.lstsq(X, ab / sb, rcond=None)
    return c


def folds(ids):
    """Two folds of the calibration-half stars (seeded permutation of sorted ids)."""
    ids = np.sort(np.asarray(ids))
    p = np.random.default_rng(SEED).permutation(len(ids))
    return {int(i): int(k < len(ids) // 2) for i, k in zip(ids, np.argsort(p))}


def _cbv_cv(te, P, V):
    """Held-out score for K = 0..KMAX: each held-out star's CBV coefficients fitted on its even (odd) days,
    day-binned chi2 evaluated on its odd (even) days."""
    ab, sb, s_, d_ = day_bins(te.src.values, te.t.values, te.a.values - predict(P, te), te.w.values)
    out = np.zeros(KMAX + 1)
    for i0 in np.unique(s_):
        m = s_ == i0
        for par in (0, 1):
            fit, ev = m & (d_ % 2 == par), m & (d_ % 2 != par)
            for K in range(KMAX + 1):
                c = cbv_coef(V[:K], d_[fit], ab[fit], sb[fit])
                pr = V[:K][:, d_[ev]].T @ c if K else 0.0
                out[K] += (((ab[ev] - pr) / sb[ev]) ** 2).sum()
    return out


def _train_rows(r):
    return r[np.abs(r.a.values * np.sqrt(r.w.values)) < CLIP_T]


def learn(tr, name, K, npic):
    """Full correction learned from training rows tr: picture model + K CBVs from its day-binned residuals."""
    P = fit_model(tr, name, npic)
    q = _train_rows(tr)
    ab, sb, s_, d_ = day_bins(q.src.values, q.t.values, q.a.values - predict(P, q), q.w.values)
    return P, sysrem(s_, d_, ab, sb, K)


def select():
    """Frozen model selection, calibration half only (docs/protocols/E26.md). Stage 1: picture model by held-out
    per-picture chi2 (2 folds of stars). Stage 2: number of CBVs K by held-out even/odd-day day-binned chi2."""
    r = pd.read_parquet(f"{D}/resid_cal.parquet")
    npic = int(r.pic.max()) + 1
    fo = folds(r.src.unique())
    r["fold"] = r.src.map(fo).values
    r["w"] = 1 / r.sa.values.astype(float) ** 2
    held = np.abs(r.a.values * np.sqrt(r.w.values)) < CLIP_T
    s1 = {}
    for name in MODELS:
        tot = 0.0
        for k in (0, 1):
            te = (r.fold.values == 1 - k) & held
            P = fit_model(r[r.fold == k], name, npic)
            q = r[te]
            tot += float((r.w.values[te] * (q.a.values - predict(P, q)) ** 2).sum())
        s1[name] = tot
        print("stage1", name, f"{tot:.7g}", time.strftime("%H:%M:%S"), flush=True)
    best = min(s1, key=s1.get)
    s2 = np.zeros(KMAX + 1)
    for k in (0, 1):
        P, V = learn(r[r.fold == k], best, KMAX, npic)
        s2 += _cbv_cv(r[(r.fold.values == 1 - k) & held], P, V)
    K = int(np.flatnonzero(s2 <= s2.min() * (1 + K_TOL))[0])
    print("stage2", np.round(s2 / s2[0], 4), "K =", K, flush=True)
    models = {}
    for k in (0, 1):
        P, V = learn(r[r.fold == k], best, K, npic)
        models[f"theta_{k}"], models[f"map_{k}"], models[f"V_{k}"] = P["theta"], P["map"], V
    np.savez(f"{D}/correction.npz", **models)
    out = dict(stage1_heldout_chi2=s1, picture_model=best, picture_terms=MODELS[best][0], map=MODELS[best][1],
               stage2_heldout_daybin_chi2=[float(x) for x in s2], stage2_relative=[round(float(x / s2[0]), 5) for x in s2],
               n_components=K, n_train_stars=int(r.src.nunique()), fold_sizes=[int((pd.Series(fo) == k).sum()) for k in (0, 1)],
               time=time.strftime("%H:%M:%S"))
    json.dump(out, open(f"{D}/selection.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("stage2_heldout")}, indent=1), flush=True)


# ---------------------------------------------------------------- applying the correction
AMPS = [0.01] + list(B.AMPS)                       # E18-B grid plus 1%
WORKERS = 3


def load_correction():
    z = np.load(f"{D}/correction.npz")
    sel = json.load(open(f"{D}/selection.json"))
    terms = sel["picture_terms"]
    return {k: (dict(terms=terms, theta=z[f"theta_{k}"], map=z[f"map_{k}"]), z[f"V_{k}"]) for k in (0, 1)}


def model_index(pop):
    """Which fold's correction each source gets: a calibration star in fold k gets the one learned on fold 1 - k
    (never its own); every other source gets fold 0 or 1 by a seeded draw."""
    cal = pop.idx[pop.half == "cal"].values
    # populations() depends on vsx.parquet in run_e18b.D; without it the split changes silently (E26 post-run note)
    assert len(cal) == 992, f"calibration half has {len(cal)} stars, not E26's 992: is vsx.parquet in {B.D}?"
    fo = folds(cal)
    out = {int(i): 1 - fo[int(i)] for i in cal}
    other = np.sort(pop.idx[pop.half != "cal"].values)
    for i, k in zip(other, np.random.default_rng(SEED + 1).integers(0, 2, len(other))):
        out[int(i)] = int(k)
    return out


def correct(d, P, V, s_d, f_d):
    """Corrected copy of one source's data dict d (flux possibly already injected)."""
    f, e = d["f"], d["e"]
    M, sig, dM = static_fit_slope(f, e, d["lam"], d["det"], s_d, f_d)
    v = M > 0
    r = pd.DataFrame(dict(pic=d["pic"][v], det=d["det"][v], x=d["x"][v], y=d["y"][v], M=M[v], slope=dM[v] / M[v]))
    a, w = (f[v] - M[v]) / M[v], (M[v] / sig[v]) ** 2
    pp = predict(P, r)
    r1 = a - pp
    k = np.abs(r1 * np.sqrt(w)) < CLIP_T
    pc = np.zeros(v.sum())
    if len(V):
        ab, sb, _, dd = day_bins(np.zeros(k.sum(), int), d["t"][v][k], r1[k], w[k])
        c = cbv_coef(V, dd, ab, sb)
        pc = V[:, np.clip(np.floor(d["t"][v] - T0).astype(int), 0, NDAY - 1)].T @ c
    pred = np.zeros(len(f))
    pred[v] = pp + pc
    out = dict(d)
    out["f"], out["e"] = f / (1 + pred), e / (1 + pred)
    return out


def _work(task):
    chunk, roles, mi, g0, mode = task
    s_d, f_d = g0_dicts(g0)
    s0, f0 = g0_dicts(json.load(open(G0_E18B)))
    C = load_correction() if mode == "after" else None
    data = B.load_src(chunk)
    real, shuf, inj = [], [], []
    for i in chunk:
        if i not in data:
            continue
        d = B.drop_thin(data[i])
        if len(d["f"]) < B.MIN_PICS:
            continue
        rng = np.random.default_rng(B.SEED + int(i))   # same streams as run_e18b._work
        freq = np.arange(B.FMIN, B.FMAX, 1 / (B.OVERS * (d["t"].max() - d["t"].min())))
        allowed, pk = B.alias_mask(freq, d["t"])
        fix = (lambda dd: correct(dd, *C[mi[int(i)]], s0, f0)) if C else (lambda dd: dd)
        try:
            r = B.source_stats(fix(d), s_d, f_d, freq, allowed)
        except Exception as ex:
            print("fail", i, repr(ex)[:100], flush=True)
            continue
        real.append(dict(src=int(i), **B._flat(r)))
        if roles.get(i) == "cal":
            dc = fix(d)
            for k in range(B.N_SHUF):
                shuf.append(dict(src=int(i), k=k, **B._flat(B.source_stats(dc, s_d, f_d, freq, allowed, shuffle_rng=rng))))
        if roles.get(i) == "test":
            pars = []
            for kind in ("sin", "ecl"):                # E18-B draw order first, then the added 1% amplitude
                for A in B.AMPS:
                    pars.append((kind, A, float(np.exp(rng.uniform(np.log(B.P_INJ[0]), np.log(B.P_INJ[1])))),
                                 float(rng.uniform(0, 2 * np.pi))))
            for kind in ("sin", "ecl"):
                pars.append((kind, 0.01, float(np.exp(rng.uniform(np.log(B.P_INJ[0]), np.log(B.P_INJ[1])))),
                             float(rng.uniform(0, 2 * np.pi))))
            for kind, A, P, ph in pars:
                di = dict(d)
                di["f"] = d["f"] * (1 + B._inj_fn(kind, A, P, ph)(d["t"]))     # injected BEFORE the correction
                rr = B.source_stats(fix(di), s_d, f_d, freq, allowed)
                inj.append(dict(src=int(i), kind=kind, A=A, P=P, phase=ph, **B._flat(rr)))
    return real, shuf, inj


def stats():
    """stats before|after: E18-B statistics on all sources, shuffled calibration null, test-half injections."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    mode = sys.argv[2]
    g0 = json.load(open(G0_E18B if mode == "before" else f"{D}/gate0_after.json"))
    pop = B.populations()
    roles = dict(zip(pop.idx, pop.half))
    mi = model_index(pop)
    assert not (set(pop.idx[pop.half == "test"]) & set(pd.read_parquet(f"{D}/resid_cal.parquet", columns=["src"]).src))
    idx = pop.idx.values
    if len(sys.argv) > 3:                              # smoke test on the first n calibration stars only
        idx = pop.idx[pop.half == "cal"].values[:int(sys.argv[3])]
    chunks = [idx[i:i + 25] for i in range(0, len(idx), 25)]
    od = f"{D}/stats_{mode}"
    os.makedirs(od, exist_ok=True)
    for p in glob.glob(f"{od}/*.parquet"):
        os.remove(p)
    t0 = time.time()
    with ProcessPoolExecutor(WORKERS) as ex:
        futs = {ex.submit(_work, (c, {int(k): roles[k] for k in c}, mi, g0, mode)): n for n, c in enumerate(chunks)}
        for k, fu in enumerate(as_completed(futs)):
            n = futs[fu]
            for name, rows in zip(("real", "shuf", "inj"), fu.result()):
                if rows:
                    pd.DataFrame(rows).to_parquet(f"{od}/{name}-{n:04d}.parquet", index=False)
            if k % 10 == 0:
                print(f"{time.strftime('%H:%M:%S')} {k + 1}/{len(chunks)} chunks, {time.time() - t0:.0f}s", flush=True)
    print("stats done", mode, time.strftime("%H:%M:%S"), flush=True)


def gate0():
    """E18-B Gate-0 rule (run_e18b.gate0, unchanged) on corrected photometry -> gate0_after.json."""
    pop = B.populations()
    mi = model_index(pop)
    C = load_correction()
    s0, f0 = g0_dicts(json.load(open(G0_E18B)))
    raw = B.load_src

    def load_corrected(srcs):
        out = {}
        for i, d in raw(srcs).items():
            d = B.drop_thin(d)
            if len(d["f"]) >= B.MIN_PICS:
                out[i] = correct(d, *C[mi[int(i)]], s0, f0)
        return out
    B.load_src = load_corrected
    B.gate0()                                          # writes f"{B.D}/gate0.json"
    B.load_src = raw
    os.replace(f"{D}/gate0.json", f"{D}/gate0_after.json")


# ---------------------------------------------------------------- evaluation (identical to run_e18b.analyze for these numbers)
OUT = "data/findings/e26.json"
FIG = "site/data/figs/e26_floor.png"
CM_P = (79.36, 95.06)                              # E18-B post-hoc common-mode periods (|P_best - P| < 0.05 d)
STEPS = AMPS + [None]


def _read(mode, name):
    fs = sorted(glob.glob(f"{D}/stats_{mode}/{name}-*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)


def evaluate(mode, pop, m_bin):
    real, shuf, inj = _read(mode, "real"), _read(mode, "shuf"), _read(mode, "inj")
    real = real.merge(pop[["idx", "id", "kind", "half", "catalogued_var", "kim", "mq"]].rename(columns={"idx": "src"}), on="src")
    norms, thr, _ = B.calibrate(real, shuf, pop)
    real = B.above(B.zscores(real, norms), thr)
    test = real[real.half == "test"]
    far = {st: dict(k=int(test[f"hit_{st}"].sum()), n=len(test), wilson95=B.wilson(int(test[f"hit_{st}"].sum()), len(test)))
           for st in B.STATS}
    far["any"] = dict(k=int(test.above.sum()), n=len(test), rate=round(float(test.above.mean()), 5),
                      wilson95=B.wilson(int(test.above.sum()), len(test)))
    inj = inj.drop(columns="m_ab").merge(real[["src", "m_ab"]], on="src")     # host m_AB, as run_e18b.analyze
    inj = B.above(B.zscores(inj, norms), thr)
    inj["m_bin"] = np.floor(inj.src.map(m_bin)).astype(int)
    rec = []
    for (kind, A, lo), q in inj.groupby(["kind", "A", "m_bin"]):
        rec.append(dict(kind=kind, A=float(A), m_lo=int(lo), n=len(q), recall=round(float(q.above.mean()), 4),
                        wilson95=B.wilson(int(q.above.sum()), len(q))))
    floor = {}
    for kind in ("sin", "ecl"):
        for lo in sorted(set(inj.m_bin)):
            ok = [r["A"] for r in rec if r["kind"] == kind and r["m_lo"] == lo and r["recall"] >= 0.5 and r["n"] >= 10]
            nmin = min([r["n"] for r in rec if r["kind"] == kind and r["m_lo"] == lo] or [0])
            floor.setdefault(kind, {})[f"{lo}-{lo + 1}"] = (min(ok) if ok else None) if nmin >= 10 else "n<10"
    cls = {}
    for name, m in (("catalogued", real.catalogued_var), ("kim", real.kim), ("milliquas", real.mq)):
        q = real[m]
        cls[name] = dict(n=len(q), n_above=int(q.above.sum()), wilson95=B.wilson(int(q.above.sum()), len(q)),
                         ids_above=q.id[q.above].tolist())
    cm = int(sum(((real.P_best - p).abs() < 0.05).sum() for p in CM_P))
    cmc = int(sum(((real[real.half == "cal"].P_best - p).abs() < 0.05).sum() for p in CM_P))
    return dict(thresholds={k: round(v["Z"], 3) for k, v in thr.items()}, far=far, recall=rec, floor=floor, classes=cls,
                common_mode=cm, common_mode_cal=cmc, n_P_60_120=int(real.P_best.between(60, 120).sum()),
                n_sources=len(real), n_inj=len(inj),
                null_cal_z_quantiles={st: [round(float(x), 2) for x in np.nanquantile(real[real.half == "cal"][f"z_{st}"],
                                                                                       [0.5, 0.99, 0.999])] for st in B.STATS})


def _step(a):
    return STEPS.index(a)


def analyze():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pop = B.populations()
    rb = _read("before", "real")
    m_bin = dict(zip(rb.src, rb.m_ab))              # recall binned by the uncorrected m_AB in both runs
    ev = {m: evaluate(m, pop, m_bin) for m in ("before", "after")}
    fb, fa = ev["before"]["floor"]["sin"], ev["after"]["floor"]["sin"]
    bins = [b for b in fb if fb[b] != "n<10"]
    lowered = [b for b in bins if _step(fa[b]) < _step(fb[b])]
    raised = [b for b in bins if _step(fa[b]) > _step(fb[b])]
    fa_after = ev["after"]["far"]["any"]
    ok_fa = fa_after["k"] / fa_after["n"] <= 0.004
    improved = len(lowered) >= 2 and ok_fa
    verdict = ("IMPROVED" if improved else "NOT IMPROVED") + \
        f": sinusoid 50%-recall floor lowered by >= 1 amplitude step in {len(lowered)} m_AB bin(s) {lowered}" + \
        (f", raised in {raised}" if raised else "") + \
        f"; test-half false alarms {fa_after['k']}/{fa_after['n']} ({'<=' if ok_fa else '>'} 0.4%)"
    cmb, cma = ev["before"]["common_mode"], ev["after"]["common_mode"]
    sel = json.load(open(f"{D}/selection.json"))
    out = dict(experiment="E26", protocol="docs/protocols/E26.md", time=time.strftime("%Y-%m-%d %H:%M:%S %Z"),
               verdict=verdict, criterion_met=bool(improved),
               n_components=sel["n_components"], picture_model=sel["picture_model"], picture_terms=sel["picture_terms"],
               per_detector_map=sel["map"])
    for kind in ("sin", "ecl"):
        for b in ev["before"]["floor"][kind]:
            out[f"floor_{kind}_before_{b}"] = ev["before"]["floor"][kind][b]
            out[f"floor_{kind}_after_{b}"] = ev["after"]["floor"][kind][b]
    for m in ("before", "after"):
        f = ev[m]["far"]["any"]
        out[f"fa_{m}"] = f"{f['k']}/{f['n']}"
        out[f"fa_{m}_rate"] = f["rate"]
        out[f"fa_{m}_wilson95"] = f["wilson95"]
        for c, lab in (("catalogued", "catalogued"), ("kim", "kim")):
            q = ev[m]["classes"][c]
            out[f"recall_{lab}_{m}"] = f"{q['n_above']}/{q['n']}"
            out[f"recall_{lab}_{m}_wilson95"] = q["wilson95"]
        out[f"common_mode_{m}"] = ev[m]["common_mode"]
    out["common_mode_disappears"] = bool(cma <= 0.1 * cmb)
    out["detail"] = ev
    json.dump(out, open(OUT + ".tmp", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    os.replace(OUT + ".tmp", OUT)
    # figure: recall vs amplitude per m_AB bin, before vs after
    shown = [b for b in bins if int(b.split("-")[0]) >= 12][:8]
    fig, axs = plt.subplots(2, len(shown), figsize=(2.0 * len(shown), 4.6), sharex=True, sharey=True, squeeze=False)
    for row, (kind, lab) in enumerate((("sin", "sinusoid"), ("ecl", "box eclipse"))):
        for ax, b in zip(axs[row], shown):
            lo = int(b.split("-")[0])
            for m, col, ls in (("before", "#888888", "--"), ("after", "#0072B2", "-")):
                q = sorted([r for r in ev[m]["recall"] if r["kind"] == kind and r["m_lo"] == lo], key=lambda r: r["A"])
                x = [100 * r["A"] for r in q]
                ax.plot(x, [r["recall"] for r in q], ls, marker="o", ms=3, color=col, label=m)
                ax.fill_between(x, [r["wilson95"][0] for r in q], [r["wilson95"][1] for r in q], color=col, alpha=0.12, lw=0)
            ax.axhline(0.5, color="k", lw=0.5, ls=":")
            ax.set_xscale("log")
            ax.set_xticks([1, 2, 5, 10, 20])
            ax.set_xticklabels(["1", "2", "5", "10", "20"])
            ax.set_ylim(-0.02, 1.02)
            ax.spines[["top", "right"]].set_visible(False)
            if row == 0:
                ax.set_title(f"m_AB {b}", fontsize=8)
            if row == 1:
                ax.set_xlabel("amplitude (%)", fontsize=8)
            ax.tick_params(labelsize=7)
        axs[row][0].set_ylabel(f"recall, {lab}", fontsize=8)
    axs[0][-1].legend(fontsize=7, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG, dpi=150)
    plt.close(fig)
    print(json.dumps({k: v for k, v in out.items() if k != "detail"}, indent=1, default=str), flush=True)


if __name__ == "__main__":
    globals()[sys.argv[1]]()
