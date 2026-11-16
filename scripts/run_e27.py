"""E27 - median reference (M3, M3b) versus one-detector blink (M2) at three equal false-alarm rates
(docs/protocols/E27.md). Methods and rules are the E1 code paths under deviation D1 (orrery.e1, unchanged);
M3b is defined here (m3b_maps). Raw cutouts in data/phase2/e27/efields (scripts/e2_cache.build format).

Steps (resumable; OMP_NUM_THREADS=1; at most 3 processes):
  draw          field positions (calibration, test, ecliptic) -> data/phase2/e27/fields.json
  cache <grp>   raw cutouts for grp in {calib, test, ecl}
  check         M3b with c = 1 reproduces orrery.e1.m3_maps (on calibration field c00)
  calib         thresholds on the calibration fields (nothing planted) -> data/phase2/e27/thresholds.json
  test          per test field: false components (nothing planted) + planted kmax records -> runs/
  ecl           per ecliptic field: SkyBoT asteroid sightings, kmax records (OUTLIER kept) -> runs/
  summary       data/findings/e27.json, e27s.json, e27_recall.png/.svg
"""
import contextlib, glob, hashlib, json, os, sys, time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from orrery import e1  # noqa: E402
from orrery.change import change_model  # noqa: E402
from orrery.data import BAD_BITS  # noqa: E402
import run_e1  # noqa: E402

BASE = "data/phase2/e27"
ROOT = f"{BASE}/efields"
run_e1.ROOT = ROOT                        # load_field / ready read the E27 cache
FIELDS = f"{BASE}/fields.json"
THR = f"{BASE}/thresholds.json"
RUNS = f"{BASE}/runs"
OUT, OUTS = "data/findings/e27.json", "data/findings/e27s.json"
FIG = "data/findings/e27_recall"
METHODS = ("M2", "M3", "M3b")
RATES = (0.01, 0.03, 0.10)                # false components per frame (map)
OPS = ("r1", "r3", "r10")
SEED_CALIB, SEED_TEST, SEED_ECL, SEED_PLANT, SEED_BOOT = 2026111527, 2026111528, 2026111529, 2026111530, 2026111531
N_CALIB, N_ECL = 8, 10
QUOTA = {"high": 10, "mid": 5, "low": 5}
MIN_SEP_DEG = 0.5
C_MIN_FRAMES = 5
R_PLANT, R_AST, EDGE, VMAX, S = e1.DET_RADIUS, 2.0, 3, 19.0, 48
WORKERS = 2                               # + 1 download process = 3


@contextlib.contextmanager
def outlier_kept():
    """E3b switch (scripts/run_e3b.py): OUTLIER (bit 19) not masked; hard-bad bits still masked."""
    old = e1.MASK_BITS
    e1.MASK_BITS = BAD_BITS
    try:
        yield
    finally:
        e1.MASK_BITS = old


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


# ---------------------------------------------------------------- M3b
def m3b_maps(vs, sigma, det, f, scale=True):
    """M3 (orrery.e1.m3_maps, same median and rejection) divided by a per-pixel empirical scale
    c(p) = max(1, 1.4826 * median_o |z_o(p)|), z_o = (v_o - med) / sqrt(sigma_o^2 + (f med)^2) over the other frames o
    of the same detector kept after rejection (leave-one-out); c = 1 where fewer than C_MIN_FRAMES such frames."""
    N, S_, _ = vs.shape
    out = np.full_like(vs, np.nan)
    for d in np.unique(det):
        E = np.flatnonzero(det == d)
        V = vs[E].reshape(len(E), -1)
        s2 = (sigma[E] ** 2)[:, None]
        cols = np.arange(V.shape[1])
        for i, e in enumerate(E):
            if len(E) < 2:
                break
            o = np.delete(np.arange(len(E)), i)
            Vo, s2o = V[o], s2[o]
            M = np.isfinite(Vo)
            with np.errstate(all="ignore"):
                med = np.nanmedian(np.where(M, Vo, np.nan), 0)
                for _ in range(2):
                    z = np.where(M, np.abs(Vo - med) / np.sqrt(s2o + (f * med) ** 2), -1)
                    worst = np.argmax(z, 0)
                    drop = (z[worst, cols] > 4) & (M.sum(0) > 1)
                    if not drop.any():
                        break
                    M[worst[drop], cols[drop]] = False
                    med = np.nanmedian(np.where(M, Vo, np.nan), 0)
                m = (V[i] - med) / np.sqrt(sigma[e] ** 2 + (f * med) ** 2)
                if scale:
                    za = np.where(M, np.abs(Vo - med) / np.sqrt(s2o + (f * med) ** 2), np.nan)
                    c = np.maximum(1.0, 1.4826 * np.nanmedian(za, 0))
                    c = np.where(M.sum(0) >= C_MIN_FRAMES, c, 1.0)
                    m = m / c
                out[e] = m.reshape(S_, S_)
    return out


def maps_for(v, lam, det, mjd, f):
    """{M2, M3, M3b: list of despiked maps or None}, as orrery.e1.method_maps for M2/M3 (D1 rules)."""
    r = change_model(v, lam, det, mjd, f=f)
    maps = {"M2": e1.blink_d1(r["vs"], r["sigma"], r["prev_det"], f),
            "M3": list(e1.m3_maps(r["vs"], r["sigma"], det, f)),
            "M3b": list(m3b_maps(r["vs"], r["sigma"], det, f))}
    return {k: [None if m is None else e1.despike(m) for m in ms] for k, ms in maps.items()}


def field_maps(name, f, keep_outlier=False):
    ctx = outlier_kept() if keep_outlier else contextlib.nullcontext()
    with ctx:
        fd = run_e1.load_field(name)
        v, lam, det, mjd = e1.frame_arrays(fd)
        maps = maps_for(v, lam, det, mjd, f)
    return fd, v, lam, det, mjd, maps


def kmax(m, x, y, radius):
    """Largest grid k at which a positive >= 2 px component lies within `radius` px of (x, y); 0 if none at 0.05."""
    S_ = m.shape[0]
    jj, ii = np.mgrid[0:S_, 0:S_]
    near = np.hypot(ii - x, jj - y) <= radius

    def hit(k):
        return bool((e1.components(m, k)[near] > 0).any())
    g = e1.K_GRID
    if not hit(g[0]):
        return 0.0
    lo, hi = 0, min(len(g) - 1, int(np.searchsorted(g, np.nanmax(m))))
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if hit(g[mid]):
            lo = mid
        else:
            hi = mid - 1
    return float(g[lo])


# ---------------------------------------------------------------- draw + cache
def _sep(ra1, dec1, ra2, dec2):
    a1, d1, a2, d2 = map(np.radians, (ra1, dec1, ra2, dec2))
    return np.degrees(2 * np.arcsin(np.sqrt(np.sin((d1 - d2) / 2) ** 2 + np.cos(d1) * np.cos(d2) * np.sin((a1 - a2) / 2) ** 2)))


def used_centres():
    """Every earlier 48 px field centre in data/efields (calib, e2_*, e3_*, e3b_*, e3x_*)."""
    out = []
    for p in glob.glob("data/efields/*/field.json"):
        j = json.load(open(p))
        out.append((j["ra"], j["dec"]))
    return out


def draw():
    if os.path.exists(FIELDS):
        return json.load(open(FIELDS))
    import e2_cache
    from build_field import candidates
    from astropy.coordinates import SkyCoord, BarycentricMeanEcliptic
    import astropy.units as u
    df, src = e2_cache.load_index()
    avoid = used_centres()

    def far(ra, dec):
        return all(_sep(ra, dec, a, b) >= MIN_SEP_DEG for a, b in avoid)

    def sky(seed, quota, prefix):
        rng, left, acc, rej, n = np.random.default_rng(seed), dict(quota), [], [], 0
        while sum(left.values()):
            ra, dec = float(rng.uniform(0, 360)), float(np.degrees(np.arcsin(rng.uniform(-1, 1))))
            n += 1
            b = float(SkyCoord(ra, dec, unit="deg").galactic.b.deg)
            bd = e2_cache.band(abs(b))
            rec = dict(draw=n, ra=ra, dec=dec, b=b, band=bd)
            if left.get(bd, 0) == 0:
                rej.append(dict(rec, reason="band quota full")); continue
            if not far(ra, dec):
                rej.append(dict(rec, reason=f"< {MIN_SEP_DEG} deg from an earlier field")); continue
            nf = len(candidates(df, ra, dec))
            rec["n_indexed"] = nf
            if nf < e2_cache.MIN_FRAMES:
                rej.append(dict(rec, reason=f"< {e2_cache.MIN_FRAMES} indexed frames")); continue
            left[bd] -= 1
            acc.append(dict(rec, field=f"{prefix}{len(acc):02d}"))
            avoid.append((ra, dec))
        return dict(seed=seed, quota=quota, n_draws=n, fields=acc, rejected=rej)

    cal = sky(SEED_CALIB, {"high": N_CALIB}, "e27c")
    tst = sky(SEED_TEST, QUOTA, "e27t")
    rng, ecl = np.random.default_rng(SEED_ECL), []
    while len(ecl) < N_ECL:
        lon, lat = rng.uniform(0, 360), rng.uniform(-2, 2)
        c = SkyCoord(lon * u.deg, lat * u.deg, frame=BarycentricMeanEcliptic()).icrs
        ra, dec = float(c.ra.deg), float(c.dec.deg)
        if not far(ra, dec):
            continue
        ecl.append(dict(field=f"e27e{len(ecl):02d}", ecl_lon=float(lon), ecl_lat=float(lat), ra=ra, dec=dec))
        avoid.append((ra, dec))
    out = dict(protocol="docs/protocols/E27.md", index=src, n_index_frames=int(len(df)),
               sky_rule="per draw ra = rng.uniform(0, 360) then dec = degrees(arcsin(rng.uniform(-1, 1))); band by "
                        "|b| (high > 30, mid 15-30, low < 15); accepted if the band has quota left, the centre is >= "
                        "0.5 deg from every earlier field, and >= 40 indexed frames cover it (scripts/e2_cache.py rule)",
               ecl_rule="lon = U(0, 360) then lat = U(-2, 2) deg (BarycentricMeanEcliptic J2000), as E3/E3b; draws "
                        "< 0.5 deg from an earlier field skipped",
               calib=cal, test=tst, ecliptic=dict(seed=SEED_ECL, fields=ecl))
    os.makedirs(BASE, exist_ok=True)
    json.dump(out, open(FIELDS, "w"), indent=1)
    return out


def names(group):
    fl = draw()
    return [x["field"] for x in {"calib": fl["calib"], "test": fl["test"], "ecl": fl["ecliptic"]}[group]["fields"]]


def cache(group):
    import shutil
    import e2_cache
    from concurrent.futures import ThreadPoolExecutor
    e2_cache.ROOT = ROOT
    free = shutil.disk_usage(".").free / 1e9
    print(f"free disk {free:.1f} GB before caching {group}", flush=True)
    if free < 20:
        sys.exit("less than 20 GB free; refusing")
    fl = draw()
    grp = {"calib": fl["calib"]["fields"], "test": fl["test"]["fields"], "ecl": fl["ecliptic"]["fields"]}[group]
    df, src = e2_cache.load_index()
    with ThreadPoolExecutor(e2_cache.WORKERS) as ex:
        for x in grp:
            e2_cache.build(x["field"], x["ra"], x["dec"], df, ex, dict(role=f"E27 {group} field"))


# ---------------------------------------------------------------- calibration
def _curve(ms, kmax_grid):
    """Components per map for k = K_GRID[:kmax_grid] (one row per map)."""
    g = e1.K_GRID[:kmax_grid]
    out = np.zeros((len(ms), len(g)), np.int32)
    for i, m in enumerate(ms):
        top = np.nanmax(m) if np.isfinite(m).any() else 0
        for j, k in enumerate(g):
            if k >= top:
                break
            out[i, j] = e1.n_components(m, k)
    return out


def calib_field(args):
    name, keep = args
    p = f"{RUNS}/calib_{name}_{'kept' if keep else 'masked'}.npz"
    if os.path.exists(p):
        return p
    f = run_e1.f_frozen()
    fd, v, lam, det, mjd, maps = field_maps(name, f, keep)
    G = int(np.searchsorted(e1.K_GRID, 200.0)) + 1            # curve up to k = 200 per map
    rec = {}
    for m in METHODS:
        ms = [x for x in maps[m] if x is not None]
        rec[m] = _curve(ms, G)
        rec[m + "_top"] = np.array([np.nanmax(x) if np.isfinite(x).any() else 0 for x in ms])
    np.savez_compressed(p + ".tmp.npz", **rec)
    os.replace(p + ".tmp.npz", p)
    print(f"calib {name} {'kept' if keep else 'masked'} done", flush=True)
    return p


def calib():
    if os.path.exists(THR):
        sys.exit(f"{THR} exists (frozen)")
    cal = names("calib")
    assert all(run_e1.ready(n) for n in cal), "calibration fields not all READY"
    os.makedirs(RUNS, exist_ok=True)
    with Pool(WORKERS) as p:
        list(p.imap_unordered(calib_field, [(n, k) for k in (False, True) for n in cal]))
    out = dict(protocol="docs/protocols/E27.md", fields=cal, f=run_e1.f_frozen(), rates=list(RATES),
               rule="smallest k on the 0.05 grid with pooled (positive components, 8-connected, >= 2 px, D1 despike) / "
                    "(maps) <= rate over all calibration fields; nothing planted (E1 rule, pooled)",
               modes={}, frozen_local=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               code_sha256={x: sha(x) for x in ("src/orrery/e1.py", "src/orrery/change.py", "scripts/run_e27.py")})
    for mode in ("masked", "kept"):
        out["modes"][mode] = {}
        for m in METHODS:
            Z = [np.load(f"{RUNS}/calib_{n}_{mode}.npz") for n in cal]
            C = np.concatenate([z[m] for z in Z])
            per_field = [z[m] for z in Z]
            rate = C.sum(0) / len(C)
            ks = {}
            for op, r in zip(OPS, RATES):
                ok = np.flatnonzero(rate <= r)
                assert len(ok), f"no k <= 200 reaches {r} for {m} {mode}"
                ks[op] = float(e1.K_GRID[ok[0]])
            # field-bootstrap spread of the calibrated k (diagnostic)
            rng = np.random.default_rng(SEED_BOOT)
            kb = {op: [] for op in OPS}
            for _ in range(500):
                idx = rng.integers(0, len(cal), len(cal))
                Cb = np.concatenate([per_field[i] for i in idx])
                rb = Cb.sum(0) / len(Cb)
                for op, r in zip(OPS, RATES):
                    ok = np.flatnonzero(rb <= r)
                    kb[op].append(float(e1.K_GRID[ok[0]]) if len(ok) else np.inf)
            out["modes"][mode][m] = dict(
                k=ks, n_maps=int(len(C)),
                rate_at_k={op: float(rate[int(np.searchsorted(e1.K_GRID, ks[op] - 1e-9))]) for op in OPS},
                k_boot95={op: [e1.q(kb[op], .025), e1.q(kb[op], .975)] for op in OPS},
                max_map_value_p99=float(np.percentile(np.concatenate([z[m + "_top"] for z in Z]), 99)))
            print(mode, m, ks, out["modes"][mode][m]["k_boot95"], flush=True)
    json.dump(out, open(THR, "w"), indent=1)


def check():
    """M3b code with the scale switched off must equal orrery.e1.m3_maps exactly (calibration field 0)."""
    n = names("calib")[0]
    f = run_e1.f_frozen()
    fd = run_e1.load_field(n)
    v, lam, det, mjd = e1.frame_arrays(fd)
    r = change_model(v, lam, det, mjd, f=f)
    a = e1.m3_maps(r["vs"], r["sigma"], det, f)
    b = m3b_maps(r["vs"], r["sigma"], det, f, scale=False)
    c = m3b_maps(r["vs"], r["sigma"], det, f)
    print("identical:", np.array_equal(a, b, equal_nan=True))
    ratio = a / c
    print("scale c percentiles 50/90/99/99.9:", np.nanpercentile(ratio, [50, 90, 99, 99.9]).round(3))


# ---------------------------------------------------------------- test fields
def test_field(name):
    p = f"{RUNS}/test_{name}.json"
    if os.path.exists(p):
        return name
    thr = json.load(open(THR))
    K = {m: thr["modes"]["masked"][m]["k"] for m in METHODS}
    f = thr["f"]
    fi = names("test").index(name)
    t = time.time()
    fd = run_e1.load_field(name)
    # nothing planted: false components at each threshold
    v, lam, det, mjd = e1.frame_arrays(fd)
    maps = maps_for(v, lam, det, mjd, f)
    fa = {m: dict(n_maps=sum(x is not None for x in maps[m]),
                  components={op: int(sum(e1.n_components(x, K[m][op]) for x in maps[m] if x is not None)) for op in OPS})
          for m in METHODS}
    levels = []
    old = e1.SEED0
    e1.SEED0 = SEED_PLANT                     # E1 planting rule with a fresh base seed
    try:
        for li, snr in enumerate(e1.LEVELS):
            pos = e1.draw_positions(fd, li, fi)
            imgs = [e1.plant(fr, x, y, snr)[0] for fr, (x, y) in zip(fd["frames"], pos)]
            v, lam, det, mjd = e1.frame_arrays(fd, imgs)
            mp = maps_for(v, lam, det, mjd, f)
            trials = []
            for e, (x, y) in enumerate(pos):
                ix, iy = int(np.clip(round(x), 0, S - 1)), int(np.clip(round(y), 0, S - 1))
                if not np.isfinite(v[e, iy, ix]) or any(mp[m][e] is None for m in METHODS):
                    continue
                trials.append([e] + [kmax(mp[m][e], x, y, R_PLANT) for m in METHODS])
            levels.append(dict(snr=snr, seed=SEED_PLANT + 1000 * li + fi, columns=["frame"] + list(METHODS),
                               trials=trials))
    finally:
        e1.SEED0 = old
    rec = dict(field=name, field_index=fi, n_frames=len(fd["frames"]), frames_dropped=len(fd["dropped"]),
               thresholds=K, false_alarms=fa, levels=levels, seconds=round(time.time() - t, 1))
    json.dump(rec, open(p + ".tmp", "w"), separators=(",", ":"))
    os.replace(p + ".tmp", p)
    print(name, {m: fa[m]["components"] for m in METHODS}, f"{time.time() - t:.0f}s", flush=True)
    return name


def ecl_field(name):
    import build_field as bf
    p = f"{RUNS}/ecl_{name}.json"
    if os.path.exists(p):
        return name
    thr = json.load(open(THR))
    f = thr["f"]
    fd, v, lam, det, mjd, maps = field_maps(name, f, keep_outlier=True)
    ast, nq = bf.known_asteroids(mjd, S, fd["ra0"], fd["dec0"])
    rows = []
    for e in range(len(mjd)):
        for o in ast[e]:
            if not (o["vmag"] < VMAX and EDGE <= o["x"] <= S - 1 - EDGE and EDGE <= o["y"] <= S - 1 - EDGE):
                continue
            if maps["M2"][e] is None:
                continue
            rows.append(dict(frame=e, name=o["name"], vmag=o["vmag"], x=round(o["x"], 2), y=round(o["y"], 2),
                             detector=int(det[e]), **{m: kmax(maps[m][e], o["x"], o["y"], R_AST) for m in METHODS}))
    fa = {m: dict(n_maps=sum(x is not None for x in maps[m])) for m in METHODS}
    json.dump(dict(field=name, n_frames=len(mjd), skybot_queries=nq, rows=rows, maps=fa), open(p, "w"))
    print(name, len(rows), "sightings", flush=True)
    return name


def run(group):
    assert os.path.exists(THR), "thresholds not frozen"
    todo = [n for n in names(group) if run_e1.ready(n)]
    with Pool(WORKERS) as p:
        list(p.imap_unordered(test_field if group == "test" else ecl_field, todo))


# ---------------------------------------------------------------- summary
def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(max(0.0, float(c - h)), 4), round(min(1.0, float(c + h)), 4)]


def summary():
    thr = json.load(open(THR))
    fl = draw()
    T = [json.load(open(f"{RUNS}/test_{n}.json")) for n in names("test") if os.path.exists(f"{RUNS}/test_{n}.json")]
    nf, nl = len(T), len(e1.LEVELS)
    K = {m: thr["modes"]["masked"][m]["k"] for m in METHODS}
    # D[m][op]: [field, level] detections; N: [field, level] trials (common trial set)
    N = np.array([[len(L["trials"]) for L in t["levels"]] for t in T], float)
    D = {m: {op: np.array([[sum(r[1 + METHODS.index(m)] >= K[m][op] for r in L["trials"]) for L in t["levels"]]
                           for t in T], float) for op in OPS} for m in METHODS}
    FA = {m: {op: np.array([t["false_alarms"][m]["components"][op] for t in T], float) for op in OPS} for m in METHODS}
    NM = {m: np.array([t["false_alarms"][m]["n_maps"] for t in T], float) for m in METHODS}
    rng = np.random.default_rng(SEED_BOOT)
    idx = rng.integers(0, nf, size=(2000, nf))
    i20 = e1.LEVELS.index(20)
    res, verdict = {}, {}
    for m in METHODS:
        res[m] = {}
        for op, rate in zip(OPS, RATES):
            rec = D[m][op].sum(0) / N.sum(0)
            rb = D[m][op][idx].sum(1) / N[idx].sum(1)
            fa = FA[m][op].sum() / NM[m].sum()
            fab = FA[m][op][idx].sum(1) / NM[m][idx].sum(1)
            res[m][op] = dict(k=K[m][op], target_rate=rate,
                              recall=[round(float(x), 4) for x in rec],
                              recall_ci95=[[round(e1.q(rb[:, l], .025), 4), round(e1.q(rb[:, l], .975), 4)] for l in range(nl)],
                              n_detected=D[m][op].sum(0).astype(int).tolist(),
                              realised_fa_rate=round(float(fa), 5),
                              realised_fa_ci95=[round(e1.q(fab, .025), 5), round(e1.q(fab, .975), 5)],
                              realised_fa_over_target=round(float(fa / rate), 3),
                              false_components=int(FA[m][op].sum()), n_maps=int(NM[m].sum()))
    for ch in ("M3", "M3b"):
        verdict[ch] = {}
        wins = 0
        for op, rate in zip(OPS, RATES):
            d = (D[ch][op] - D["M2"][op])
            db = d[idx].sum(1)[:, i20] / N[idx].sum(1)[:, i20]
            dp = float(d.sum(0)[i20] / N.sum(0)[i20])
            ci = [e1.q(db, .025), e1.q(db, .975)]
            dl = d[idx].sum(1) / N[idx].sum(1)
            fa_ok = res[ch][op]["realised_fa_rate"] <= 1.5 * rate
            win = bool(dp > 0 and ci[0] > 0 and fa_ok)
            wins += win
            verdict[ch][op] = dict(diff_snr20=round(dp, 4), diff_snr20_ci95=[round(x, 4) for x in ci],
                                   diff_all_levels=[round(float(x), 4) for x in d.sum(0) / N.sum(0)],
                                   diff_all_levels_ci95=[[round(e1.q(dl[:, l], .025), 4), round(e1.q(dl[:, l], .975), 4)]
                                                         for l in range(nl)],
                                   fa_within_1p5x_target=bool(fa_ok), qualifies=win)
        verdict[ch]["n_qualifying"] = wins
        lose = sum(verdict[ch][op]["diff_snr20_ci95"][1] < 0 for op in OPS)
        verdict[ch]["result"] = "BEATS M2" if wins >= 2 else ("LOSS" if lose >= 2 else "TIE")
    # per-band realised false-alarm rate (secondary)
    band = {t["field"]: next(x["band"] for x in fl["test"]["fields"] if x["field"] == t["field"]) for t in T}
    fa_band = {b: {m: {op: round(float(sum(t["false_alarms"][m]["components"][op] for t in T if band[t["field"]] == b) /
                                     max(1, sum(t["false_alarms"][m]["n_maps"] for t in T if band[t["field"]] == b))), 5)
                       for op in OPS} for m in METHODS} for b in QUOTA}
    # ecliptic asteroid secondary (OUTLIER kept)
    KK = {m: thr["modes"]["kept"][m]["k"] for m in METHODS}
    E = [json.load(open(f"{RUNS}/ecl_{n}.json")) for n in names("ecl") if os.path.exists(f"{RUNS}/ecl_{n}.json")]
    rows = [dict(r, field=e["field"]) for e in E for r in e["rows"]]
    ast = {}
    for m in METHODS:
        ast[m] = {}
        for op in OPS:
            h = sum(r[m] >= KK[m][op] for r in rows)
            objs = {}
            for r in rows:
                objs[(r["field"], r["name"])] = objs.get((r["field"], r["name"]), False) or r[m] >= KK[m][op]
            ast[m][op] = dict(k=KK[m][op], hits=int(h), n=len(rows), recall=round(h / len(rows), 4) if rows else None,
                              wilson95=wilson(h, len(rows)), objects_detected=int(sum(objs.values())), n_objects=len(objs))
    out = dict(experiment="E27", protocol="docs/protocols/E27.md", question="median reference (M3/M3b) vs one-detector blink (M2) at equal false-alarm rate",
               index=fl["index"], n_index_frames=fl["n_index_frames"], f=thr["f"], thresholds_file=THR,
               thresholds_sha256=sha(THR), thresholds=thr["modes"],
               calibration_fields=thr["fields"], test_fields=[t["field"] for t in T], n_test_fields=nf,
               test_frames=int(sum(t["n_frames"] for t in T)), levels_snr_frame=list(e1.LEVELS),
               n_trials=N.sum(0).astype(int).tolist(),
               trial_rule="planted frame whose nearest grid pixel has data and that has an M2 map (so M3, M3b too); "
                          "the same trials for every method",
               detection="positive >= 2 px 8-connected component above k within 1.5 px of the planted position (D1 rules)",
               bootstrap=dict(n=2000, seed=SEED_BOOT, unit="test field", recall="pooled detections / pooled trials"),
               operating_points=dict(zip(OPS, RATES)), results=res, verdict=verdict,
               decision_rule="challenger beats M2 iff at >= 2 of 3 operating points: recall difference at S/N 20 > 0 with "
                             "the 95% field-bootstrap interval excluding zero AND the challenger's realised test "
                             "false-alarm rate <= 1.5 x target at that operating point; LOSS if the interval lies "
                             "below zero at >= 2 operating points; otherwise TIE",
               realised_fa_by_band=fa_band, n_test_fields_by_band={b: sum(band[t["field"]] == b for t in T) for b in QUOTA},
               asteroids=dict(fields=[e["field"] for e in E], rule="SkyBoT V < 19, >= 3 px from edge, frames with an M2 "
                              "map; component within 2 px; OUTLIER not masked; thresholds from the calibration fields "
                              "with OUTLIER not masked", n_sightings=len(rows), per_method=ast,
                              skybot_queries=int(sum(e["skybot_queries"] for e in E))),
               code_sha256={x: sha(x) for x in ("src/orrery/e1.py", "src/orrery/change.py", "scripts/run_e27.py")},
               generated_local=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    json.dump(out, open(OUT, "w"), indent=1)
    flat(out)
    figure(out)
    for m in METHODS:
        for op in OPS:
            r = res[m][op]
            print(m, op, "k", r["k"], "recall", r["recall"], "FA", r["realised_fa_rate"], r["realised_fa_ci95"])
    print(json.dumps(verdict, indent=1))
    print(json.dumps(ast, indent=1))


def flat(out):
    s = dict(source=OUT, n_test_fields=out["n_test_fields"], test_frames=out["test_frames"],
             n_calib_fields=len(out["calibration_fields"]), n_sightings=out["asteroids"]["n_sightings"],
             n_ecl_fields=len(out["asteroids"]["fields"]))
    i20 = out["levels_snr_frame"].index(20)
    for m in METHODS:
        for op in OPS:
            r = out["results"][m][op]
            s[f"k_{m}_{op}"] = r["k"]
            s[f"recall20_{m}_{op}"] = r["recall"][i20]
            s[f"recall20_{m}_{op}_lo"], s[f"recall20_{m}_{op}_hi"] = r["recall_ci95"][i20]
            s[f"recall12_{m}_{op}"] = r["recall"][e1.LEVELS.index(12)]
            s[f"fa_{m}_{op}"] = r["realised_fa_rate"]
            s[f"fa_{m}_{op}_lo"], s[f"fa_{m}_{op}_hi"] = r["realised_fa_ci95"]
            a = out["asteroids"]["per_method"][m][op]
            s[f"ast_{m}_{op}_hits"], s[f"ast_{m}_{op}_recall"] = a["hits"], a["recall"]
    for ch in ("M3", "M3b"):
        v = out["verdict"][ch]
        s[f"verdict_{ch}"] = v["result"]
        for op in OPS:
            s[f"diff20_{ch}_{op}"] = v[op]["diff_snr20"]
            s[f"diff20_{ch}_{op}_lo"], s[f"diff20_{ch}_{op}_hi"] = v[op]["diff_snr20_ci95"]
    s = {k.replace("M3b", "M3B"): v for k, v in s.items()}
    assert all(not any(c in k for c in ".- ") for k in s)
    json.dump(s, open(OUTS, "w"), indent=1)


def figure(out, op="r3"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"M2": "#56B4E9", "M3": "#009E73", "M3b": "#D55E00"}
    sty = {"M2": "--", "M3": "-.", "M3b": "-"}
    lab = {"M2": "M2 blink, same detector", "M3": "M3 median of other frames", "M3b": "M3b median, per-pixel scale"}
    L = out["levels_snr_frame"]
    fig, ax = plt.subplots(figsize=(5.6, 3.9))
    for m in METHODS:
        r = out["results"][m][op]
        lo, hi = np.array(r["recall_ci95"]).T
        ax.fill_between(L, lo, hi, color=col[m], alpha=0.15, lw=0)
        ax.plot(L, r["recall"], sty[m], marker="o", ms=4, color=col[m], lw=1.6, label=f"{lab[m]} (k = {r['k']:.2f})")
    ax.set_xscale("log")
    ax.set_xticks(L)
    ax.set_xticklabels([str(x) for x in L])
    ax.minorticks_off()
    ax.set_xlabel("planted S/N per frame")
    ax.set_ylabel(f"recall at {out['operating_points'][op]:.2f} false components per frame")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG + ".png", dpi=200)
    fig.savefig(FIG + ".svg")
    plt.close(fig)


if __name__ == "__main__":
    a = sys.argv[1]
    if a == "cache":
        cache(sys.argv[2])
    else:
        {"draw": draw, "check": check, "calib": calib, "test": lambda: run("test"), "ecl": lambda: run("ecl"),
         "summary": summary}[a]()
