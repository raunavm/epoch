"""E30 - a learned scorer of real changes on top of M3b candidates (docs/protocols/E30.md).

Steps (resumable; OMP_NUM_THREADS=1; `--workers n` where a step uses a pool):
  draw          fresh field positions -> data/phase2/e30/fields.json
  fetch         Gaia prefetch (serial, back-off), then raw cutouts: e30r, e30c, e30t, e30e
  split         train / validation split of the 49-field pool -> data/phase2/e30/split.json
  build         training passes (12 planted + 1 unplanted, augmented) and validation passes -> data/phase2/e30/sets/
  train         CNN and LR, 5 seeds each, early stopping, learning curve -> data/phase2/e30/train/
  sim2real      injected vs real asteroid candidates on the 21 earlier ecliptic fields (diagnostic)
  calib         thresholds on the 12 calibration fields -> data/phase2/e30/thresholds.json, site/data/model/e30.json
  test | ecl    frozen test fields / ecliptic fields -> data/phase2/e30/runs/
  summary       data/findings/e30.json, e30s.json, e30_recall.png/.svg, e30_examples.png
  fixture       site/tests/fixtures/e30_parity.json
"""
import glob, json, os, sys, time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE = "data/phase2/e30"
ROOT = f"{BASE}/efields"
FIELDS = f"{BASE}/fields.json"
SEED_SPLIT, SEED_CAL, SEED_TEST, SEED_ECL, SEED_TRX = 2026111540, 2026111541, 2026111542, 2026111543, 2026111546
SEED_PLANT, SEED_BOOT = 2026111544, 2026111545
SEED_TRPOS, SEED_TRSNR, SEED_AUG, SEED_VAL = 2026111547, 2026111548, 2026111549, 2026111550
SEED_NET, SEED_CALBOOT, SEED_LC = 2026111551, 2026111552, 2026111553
MIN_SEP_DEG = 0.5
SETS, TRAIN, RUNS = f"{BASE}/sets", f"{BASE}/train", f"{BASE}/runs"
SPLIT, CAND, THR = f"{BASE}/split.json", f"{BASE}/train/candidate_models.json", f"{BASE}/thresholds.json"
MODEL = "site/data/model/e30.json"
PROTOCOL = "docs/protocols/E30.md"
E27_DONE = "data/findings/e27.json"
OUT, OUTS, FIG, FIGEX = "data/findings/e30.json", "data/findings/e30s.json", "data/findings/e30_recall", \
    "data/findings/e30_examples.png"
RATES, OPS = (0.01, 0.03, 0.10), ("r1", "r3", "r10")
R_PLANT, R_AST, EDGE, VMAX, S = 1.5, 2.0, 3, 19.0, 48
KMIN = 2.0                                      # component curves are exact for k >= KMIN
SCORERS = ("CNN", "LR", "Z")
COMP = ("M2", "M3", "M3B", "M2D", "M3BD", "M3BF")
COMP_DESC = dict(M2="M2 one-detector blink, live (no flags, no despike)", M3="M3 median of other frames, live",
                 M3B="M3b median with per-pixel scale, live", M2D="M2 live with despike (secondary)",
                 M3BD="M3b live with despike (secondary)", M3BF="M3b with flags masked and despike (E27 primary, reference line)")
CROP = slice(14, 34)
POOL = ["calib"] + [f"e2_{i:02d}" for i in range(20)]   # D2: e30r dropped; D3: e27c not read before e27.json exists
OLD_ECL = [f"e3_ecl{i:02d}" for i in range(10)] + [f"e3b_ecl{i:02d}" for i in range(10)] + ["e3x_ecliptic_control"]
NEPOCH, BATCH, LR0, WD, PATIENCE, NSEED = 30, 512, 2e-3, 1e-4, 5, 2   # D2: 2 seeds
N_TRAIN_PASSES = 12                             # D3: back to the design (D2 had cut it to 6)


# ---------------------------------------------------------------- draw + cache
def _sep(ra1, dec1, ra2, dec2):
    a1, d1, a2, d2 = map(np.radians, (ra1, dec1, ra2, dec2))
    return np.degrees(2 * np.arcsin(np.sqrt(np.sin((d1 - d2) / 2) ** 2 + np.cos(d1) * np.cos(d2) * np.sin((a1 - a2) / 2) ** 2)))


def used_centres():
    out = [(j["ra"], j["dec"]) for j in (json.load(open(p)) for p in glob.glob("data/efields/*/field.json"))]
    e27 = json.load(open("data/phase2/e27/fields.json"))
    for g in ("calib", "test", "ecliptic"):
        out += [(x["ra"], x["dec"]) for x in e27[g]["fields"]]
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
    n_avoid0 = len(avoid)

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
            print(prefix, len(acc), bd, flush=True)
        return dict(seed=seed, quota=quota, n_draws=n, fields=acc, rejected=rej)

    cal = sky(SEED_CAL, {"high": 6, "mid": 3, "low": 3}, "e30c")
    tst = sky(SEED_TEST, {"high": 10, "mid": 5, "low": 5}, "e30t")
    rng, ecl, nd = np.random.default_rng(SEED_ECL), [], 0
    while len(ecl) < 10:
        lon, lat = rng.uniform(0, 360), rng.uniform(-2, 2)
        nd += 1
        c = SkyCoord(lon * u.deg, lat * u.deg, frame=BarycentricMeanEcliptic()).icrs
        ra, dec = float(c.ra.deg), float(c.dec.deg)
        if not far(ra, dec):
            continue
        ecl.append(dict(field=f"e30e{len(ecl):02d}", ecl_lon=float(lon), ecl_lat=float(lat), ra=ra, dec=dec))
        avoid.append((ra, dec))
    trx = sky(SEED_TRX, {"high": 10, "mid": 5, "low": 5}, "e30r")
    out = dict(protocol="docs/protocols/E30.md", index=src, n_index_frames=int(len(df)), n_avoided_earlier=n_avoid0,
               sky_rule="as run_e27.draw: ra = U(0, 360), dec = degrees(arcsin(U(-1, 1))); band quota; >= 0.5 deg from "
                        "every centre in data/efields/*/field.json, data/phase2/e27/fields.json and each other; >= 40 "
                        "indexed frames",
               calib=cal, test=tst, ecliptic=dict(seed=SEED_ECL, n_draws=nd, fields=ecl), train_extra=trx)
    os.makedirs(BASE, exist_ok=True)
    json.dump(out, open(FIELDS, "w"), indent=1)
    return out


def group(name):
    fl = draw()
    return {"calib": fl["calib"], "test": fl["test"], "ecl": fl["ecliptic"], "trx": fl["train_extra"]}[name]["fields"]


def gaia_prefetch(fields):
    """Serial Gaia cone queries (e2_cache.gaia query) with exponential back-off (HTTP 429 seen on the AIP mirror)."""
    import e2_cache
    for x in fields:
        d = f"{ROOT}/{x['field']}"
        os.makedirs(d, exist_ok=True)
        p = f"{d}/gaia.parquet"
        for i in range(8):
            if e2_cache.gaia(p, x["ra"], x["dec"]):
                break
            wait = 30 * 2 ** i
            print(f"gaia {x['field']}: back-off {wait}s", flush=True)
            time.sleep(wait)
        print(f"gaia {x['field']}: {'ok' if os.path.exists(p) else 'FAILED'}", flush=True)
        time.sleep(2)


def fetch():
    import shutil
    import e2_cache
    from concurrent.futures import ThreadPoolExecutor
    e2_cache.ROOT = ROOT
    order = [("calib", "E30 calibration field"), ("test", "E30 test field"), ("ecl", "E30 ecliptic field")]
    allf = [x for g, _ in order for x in group(g)]
    gaia_prefetch(allf)
    df, src = e2_cache.load_index()
    with ThreadPoolExecutor(8) as ex:                     # deviation D2: e30r dropped, 8 download threads
        for g, role in order:
            for x in group(g):
                free = shutil.disk_usage(".").free / 1e9
                if free < 20:
                    sys.exit(f"less than 20 GB free ({free:.1f}); refusing")
                e2_cache.build(x["field"], x["ra"], x["dec"], df, ex,
                               dict(role=role, **{k: x[k] for k in ("b", "band") if k in x}))


# ---------------------------------------------------------------- fields and preprocessing
def sha(p):
    import hashlib
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def band_of(name):
    if name.startswith("e30"):
        fl = draw()
        for g in ("calib", "test", "train_extra"):
            for x in fl[g]["fields"]:
                if x["field"] == name:
                    return x["band"]
    if name.startswith("e27c"):
        return "high"
    if name == "calib":
        return "high"                                   # b = -51.4
    return json.load(open(f"data/efields/{name}/field.json"))["band"]


def root_of(name):
    if name.startswith("e30"):
        return ROOT
    if name.startswith("e27"):
        assert name.startswith("e27c"), "E30 never reads an E27 test or ecliptic field"
        assert os.path.exists(E27_DONE), "E30 reads no E27 field before data/findings/e27.json exists"
        return "data/phase2/e27/efields"
    return "data/efields"


def load(name):
    import run_e1
    run_e1.ROOT = root_of(name)
    assert run_e1.ready(name), f"{name} not READY"
    return run_e1.load_field(name)


def picture_keys(name):
    return {f"{x['obs_id']}_D{x['detector']}" for x in json.load(open(f"{root_of(name)}/{name}/field.json"))["frame_list"]}


def live_v(fd, imgs=None):
    """Live view's preprocessing: unmasked bilinear resample (orrery.e1.resample(...)[1]), float32."""
    return np.array([e1_().resample(imgs[i] if imgs is not None else f["img"], f["flg"], f["gx"], f["gy"])[1]
                     for i, f in enumerate(fd["frames"])]).astype(np.float32)


def e1_():
    from orrery import e1
    return e1


def meta_arrays(fd):
    fr = fd["frames"]
    return (np.array([f["lam"] for f in fr]).astype(np.float32), np.array([f["det"] for f in fr]),
            np.array([f["mjd"] for f in fr]))


def f_floor():
    import run_e1
    return run_e1.f_frozen()


def stage1(v, lam, det, mjd, use=None, jit=None):
    from orrery import detect
    from orrery.change import change_model
    f = f_floor()
    r = change_model(v, lam, det, mjd, f=f)
    sig = r["sigma"] if jit is None else r["sigma"] * jit
    med, z, c = detect.m3b(r["vs"], sig, det, f, use)
    return dict(vs=r["vs"], sig=sig, med=med, z=z, c=c, prev=r["prev_det"], r=r)


def cands(st, det):
    """Every stage-1 candidate of every picture: frame e, pixel x, y, z, cutouts X [n, 5, 9, 9] float32, det."""
    from orrery import detect
    E, X, Y, Z, C = [], [], [], [], []
    for e in range(len(det)):
        if not np.isfinite(st["z"][e]).any():
            continue
        x, y, z = detect.find_candidates(st["z"][e])
        if not len(x):
            continue
        C.append(detect.cutouts(st["vs"][e], st["med"][e], st["z"][e], st["c"][e], st["sig"][e], x, y).astype(np.float32))
        E.append(np.full(len(x), e)); X.append(x); Y.append(y); Z.append(z)
    if not E:
        return dict(e=np.zeros(0, int), x=np.zeros(0, int), y=np.zeros(0, int), z=np.zeros(0),
                    X=np.zeros((0, 5, 9, 9), np.float32), det=np.zeros(0, int))
    e = np.concatenate(E)
    return dict(e=e, x=np.concatenate(X), y=np.concatenate(Y), z=np.concatenate(Z), X=np.concatenate(C), det=det[e])


def n_maps(ms):
    return int(sum(m is not None and np.isfinite(m).any() for m in ms))


def trial_ok(v, prev, e, x, y):
    ix, iy = int(np.clip(round(x), 0, v.shape[1] - 1)), int(np.clip(round(y), 0, v.shape[1] - 1))
    return bool(np.isfinite(v[e, iy, ix]) and prev[e] >= 0)


# ---------------------------------------------------------------- split + training/validation sets
def split():
    if os.path.exists(SPLIT):
        return json.load(open(SPLIT))
    rng = np.random.default_rng(SEED_SPLIT)
    quota = {"high": 4, "mid": 1, "low": 1}                 # D3
    bands = {n: band_of(n) for n in POOL}
    val = []
    for b in ("high", "mid", "low"):
        cand = [n for n in POOL if bands[n] == b]
        val += [cand[i] for i in rng.permutation(len(cand))[:quota[b]]]
    out = dict(rule="validation 4 high / 1 mid / 1 low by default_rng(2026111540).permutation within band, pool order "
                    "calib, e2_00-19 (deviations D2, D3)", seed=SEED_SPLIT, bands=bands,
               val=[n for n in POOL if n in val], train=[n for n in POOL if n not in val])
    json.dump(out, open(SPLIT, "w"), indent=1)
    return out


def build_train(name):
    e1 = e1_()
    p = f"{SETS}/train_{name}.npz"
    if os.path.exists(p):
        return name
    fi = POOL.index(name)
    t0 = time.time()
    fd = load(name)
    N = len(fd["frames"])
    lam, det, mjd = meta_arrays(fd)
    rec = {k: [] for k in ("X", "det", "y", "z", "snr", "pass")}
    npic = 0
    old = e1.SEED0
    e1.SEED0 = SEED_TRPOS
    try:
        for ps in list(range(N_TRAIN_PASSES)) + [12]:
            ra = np.random.default_rng(SEED_AUG + 1000 * ps + fi)
            u = ra.uniform(0.5, 1.0)
            use = np.zeros(N, bool)
            use[ra.choice(N, int(round(u * N)), replace=False)] = True
            jit = ra.uniform(0.9, 1.1, N)
            imgs = None
            if ps < N_TRAIN_PASSES:
                pos = e1.draw_positions(fd, ps, fi)
                snr = np.exp(np.random.default_rng(SEED_TRSNR + 1000 * ps + fi).uniform(np.log(3), np.log(30), N))
                imgs = [e1.plant(fr, x, y, s_)[0] for fr, (x, y), s_ in zip(fd["frames"], pos, snr)]
            st = stage1(live_v(fd, imgs), lam, det, mjd, use, jit)
            cd = cands(st, det)
            if ps < N_TRAIN_PASSES:
                d = np.hypot(cd["x"] - pos[cd["e"], 0], cd["y"] - pos[cd["e"], 1])
                keep = []
                for e in np.unique(cd["e"][d <= R_PLANT]):
                    k = np.flatnonzero((cd["e"] == e) & (d <= R_PLANT))
                    keep.append(k[np.argmin(d[k])])
                keep = np.array(keep, int)
                lab, sn = np.ones(len(keep)), snr[cd["e"][keep]]
            else:
                keep = np.arange(len(cd["e"]))
                lab, sn = np.zeros(len(keep)), np.zeros(len(keep))
                npic = int(sum(np.isfinite(z).any() for z in st["z"]))
            rec["X"].append(cd["X"][keep].astype(np.float16)); rec["det"].append(cd["det"][keep])
            rec["y"].append(lab); rec["z"].append(cd["z"][keep]); rec["snr"].append(sn)
            rec["pass"].append(np.full(len(keep), ps))
    finally:
        e1.SEED0 = old
    out = {k: np.concatenate(v) for k, v in rec.items()}
    out["X"] = out["X"].astype(np.float16)
    np.savez_compressed(p[:-4] + ".tmp.npz", n_pictures_unplanted=npic, **out)
    os.replace(p[:-4] + ".tmp.npz", p)
    print(f"train {name}: {int(out['y'].sum())} pos, {int((out['y'] == 0).sum())} neg, {time.time() - t0:.0f}s", flush=True)
    return name


def build_val(name):
    e1 = e1_()
    p = f"{SETS}/val_{name}.npz"
    if os.path.exists(p):
        return name
    fi = POOL.index(name)
    t0 = time.time()
    fd = load(name)
    lam, det, mjd = meta_arrays(fd)
    st = stage1(live_v(fd), lam, det, mjd)
    cd = cands(st, det)
    out = dict(negX=cd["X"], negD=cd["det"], negE=cd["e"], negZ=cd["z"],
               npic=int(sum(np.isfinite(z).any() for z in st["z"])))
    tX, tD, tT, tZ, tl, te, tdet = [], [], [], [], [], [], []
    old = e1.SEED0
    e1.SEED0 = SEED_VAL
    nt = 0
    try:
        for li, snr in enumerate(e1.LEVELS):
            pos = e1.draw_positions(fd, li, fi)
            imgs = [e1.plant(fr, x, y, snr)[0] for fr, (x, y) in zip(fd["frames"], pos)]
            v = live_v(fd, imgs)
            st = stage1(v, lam, det, mjd)
            cd = cands(st, det)
            for e, (x, y) in enumerate(pos):
                if not trial_ok(v, st["prev"], e, x, y):
                    continue
                sel = np.flatnonzero((cd["e"] == e) & (np.hypot(cd["x"] - x, cd["y"] - y) <= R_PLANT))
                tX.append(cd["X"][sel]); tD.append(cd["det"][sel]); tZ.append(cd["z"][sel]); tT.append(np.full(len(sel), nt))
                tl.append(li); te.append(e); tdet.append(det[e])
                nt += 1
    finally:
        e1.SEED0 = old
    out.update(trX=np.concatenate(tX), trD=np.concatenate(tD), trT=np.concatenate(tT), trZ=np.concatenate(tZ),
               tr_level=np.array(tl), tr_e=np.array(te), tr_det=np.array(tdet), n_trials=nt)
    np.savez_compressed(p[:-4] + ".tmp.npz", **out)
    os.replace(p[:-4] + ".tmp.npz", p)
    print(f"val {name}: {len(out['negZ'])} neg, {nt} trials, {time.time() - t0:.0f}s", flush=True)
    return name


def build(workers=1):
    from multiprocessing import Pool
    os.makedirs(SETS, exist_ok=True)
    sp = split()
    jobs = [("v", n) for n in sp["val"]] + [("t", n) for n in sp["train"]]
    with Pool(workers) as pl:
        list(pl.imap_unordered(_build_job, jobs))


def _build_job(j):
    return build_val(j[1]) if j[0] == "v" else build_train(j[1])


# ---------------------------------------------------------------- training (JAX, hand-written Adam)
def _val_data(names):
    V = [np.load(f"{SETS}/val_{n}.npz") for n in names]
    X = np.concatenate([np.concatenate([v["negX"], v["trX"]]) for v in V]).astype(np.float32)
    D = np.concatenate([np.concatenate([v["negD"], v["trD"]]) for v in V])
    isneg = np.concatenate([np.r_[np.ones(len(v["negD"]), bool), np.zeros(len(v["trD"]), bool)] for v in V])
    off, tr, lev, npic = 0, [], [], 0
    for v in V:
        tr.append(np.r_[np.full(len(v["negD"]), -1), v["trT"] + off]); off += int(v["n_trials"])
        lev.append(v["tr_level"]); npic += int(v["npic"])
    return dict(X=X, D=D, isneg=isneg, trial=np.concatenate(tr), level=np.concatenate(lev), npic=npic, n_trials=off,
                Z=np.concatenate([np.concatenate([v["negZ"], v["trZ"]]) for v in V]))


def val_metric(scores, vd, rate=0.03):
    """Validation recall per level and its mean at `rate` false candidates per picture (threshold from the
    validation unplanted passes)."""
    from orrery import detect
    t = detect.threshold_scores(scores[vd["isneg"]], vd["npic"], rate)
    best = np.full(vd["n_trials"], -np.inf)
    m = ~vd["isneg"]
    np.maximum.at(best, vd["trial"][m], scores[m])
    rec = np.array([np.mean(best[vd["level"] == li] > t) for li in range(5)])
    return float(rec.mean()), rec, t


def _jax():
    os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
    import jax
    import jax.numpy as jnp
    return jax, jnp


def cnn_apply(p, X, D):
    jax, jnp = _jax()
    dn = ("NCHW", "OIHW", "NCHW")
    a = X
    for k in ("c1", "c2", "c3"):
        a = jax.nn.relu(jax.lax.conv_general_dilated(a, p[k + "w"], (1, 1), "VALID", dimension_numbers=dn)
                        + p[k + "b"][None, :, None, None])
    h = jnp.concatenate([a.reshape(a.shape[0], -1), D], 1)
    h = jax.nn.relu(h @ p["d1w"] + p["d1b"])
    return (h @ p["d2w"] + p["d2b"])[:, 0]


def lr_apply(p, F, D=None):
    return F @ p["w"] + p["b"]


def fit(kind, seed, Xtr, Dtr, ytr, vd, vin, log):
    """One training run; returns (best params as numpy, best val metric, history)."""
    jax, jnp = _jax()
    key = jax.random.PRNGKey(SEED_NET + seed)
    if kind == "cnn":
        shapes = dict(c1w=(12, 5, 3, 3), c2w=(16, 12, 3, 3), c3w=(16, 16, 3, 3), d1w=(150, 16), d2w=(16, 1))
        p = {}
        for k, shp in shapes.items():
            key, sk = jax.random.split(key)
            fan = int(np.prod(shp[1:])) if k.startswith("c") else shp[0]
            p[k] = jax.random.normal(sk, shp) * np.sqrt(2.0 / fan)
            p[k[:-1] + "b"] = jnp.zeros(shp[0] if k.startswith("c") else shp[1])
        apply = cnn_apply
    else:
        p = dict(w=jnp.zeros(6), b=jnp.zeros(()))
        apply = lr_apply
    isw = {k: k.endswith("w") for k in p}
    n = len(ytr)
    spe = n // BATCH
    total = NEPOCH * spe

    def loss_fn(p, Xb, Db, yb):
        l = apply(p, Xb, Db)
        return jnp.mean(jnp.maximum(l, 0) - l * yb + jnp.log1p(jnp.exp(-jnp.abs(l))))

    def d4(Xb, k):
        r = [Xb] + [jnp.rot90(Xb, i, axes=(2, 3)) for i in (1, 2, 3)]
        A = jnp.stack(r + [jnp.flip(x, 3) for x in r])
        return A[k, jnp.arange(Xb.shape[0])]

    @jax.jit
    def step(p, m, v, t, Xb, Db, yb, k):
        if kind == "cnn":
            Xb = d4(Xb, jax.random.randint(k, (Xb.shape[0],), 0, 8))
        loss, gr = jax.value_and_grad(loss_fn)(p, Xb, Db, yb)
        lr = LR0 * 0.5 * (1 + jnp.cos(jnp.pi * t / total))
        m = {q: 0.9 * m[q] + 0.1 * gr[q] for q in p}
        v = {q: 0.999 * v[q] + 0.001 * gr[q] ** 2 for q in p}
        p = {q: p[q] - lr * ((m[q] / (1 - 0.9 ** (t + 1))) / (jnp.sqrt(v[q] / (1 - 0.999 ** (t + 1))) + 1e-8)
                             + (WD * p[q] if isw[q] else 0.0)) for q in p}
        return p, m, v, loss

    score = jax.jit(apply)

    def val_scores(p):
        return np.concatenate([np.asarray(score(p, vin[0][i:i + 8192], vin[1][i:i + 8192]))
                               for i in range(0, len(vin[0]), 8192)])
    m = {q: jnp.zeros_like(p[q]) for q in p}
    v = {q: jnp.zeros_like(p[q]) for q in p}
    best, best_p, hist, bad, t = -1.0, None, [], 0, 0
    rng = np.random.default_rng(SEED_NET + 100 * seed)
    for ep in range(NEPOCH):
        perm = rng.permutation(n)
        tl = 0.0
        for b in range(spe):
            i = perm[b * BATCH:(b + 1) * BATCH]
            key, sk = jax.random.split(key)
            p, m, v, loss = step(p, m, v, t, Xtr[i], Dtr[i], ytr[i], sk)
            tl += float(loss)
            t += 1
        met, rec, thr = val_metric(val_scores(p), vd)
        hist.append(dict(epoch=ep + 1, train_loss=round(tl / spe, 5), val_mean_recall=round(met, 4),
                         val_recall=[round(float(x), 4) for x in rec], val_threshold=round(thr, 4)))
        log(f"{kind} seed {seed} epoch {ep + 1}: loss {tl / spe:.4f} val {met:.4f} {np.round(rec, 3)}")
        if met > best:
            best, best_p, bad = met, {q: np.asarray(p[q]) for q in p}, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    return best_p, best, hist


def round6(a):
    return np.vectorize(lambda x: float(f"{x:.6g}"))(np.asarray(a, float))


def to_json_weights(p):
    return {k: round6(v).tolist() for k, v in p.items()}


def train_set(names):
    T = [np.load(f"{SETS}/train_{n}.npz") for n in names]
    X = np.concatenate([t["X"] for t in T]).astype(np.float32)
    from orrery import detect
    D = detect.onehot(np.concatenate([t["det"] for t in T])).astype(np.float32)
    y = np.concatenate([t["y"] for t in T]).astype(np.float32)
    return X, D, y


def train():
    from orrery import detect
    os.makedirs(TRAIN, exist_ok=True)
    logf = open(f"{TRAIN}/train.log", "a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n"); logf.flush()
    sp = split()
    vd = _val_data(sp["val"])
    X, D, y = train_set(sp["train"])
    log(f"training set: {len(y)} samples, {int(y.sum())} positive; validation {vd['n_trials']} trials, "
        f"{int(vd['isneg'].sum())} negatives, {vd['npic']} pictures")
    vD = detect.onehot(vd["D"]).astype(np.float32)
    res = dict(n_train=len(y), n_pos=int(y.sum()), n_neg=int((y == 0).sum()), val_trials=vd["n_trials"],
               val_negatives=int(vd["isneg"].sum()), val_pictures=vd["npic"], cnn=[], lr=[])
    zmet = val_metric(vd["Z"], vd)
    res["z_val"] = dict(mean_recall=zmet[0], recall=zmet[1].tolist(), threshold=zmet[2])
    log(f"Z (stage-1 z) validation: {zmet[0]:.4f} {np.round(zmet[1], 3)}")
    # LR: features of the same cutouts, standardised on the training set
    F = detect.lr_features(X)
    mu, sd = F.mean(0), F.std(0)
    Fs = ((F - mu) / sd).astype(np.float32)
    vF = ((detect.lr_features(vd["X"]) - mu) / sd).astype(np.float32)
    best = {}
    for kind, Xin, Din, vin in (("lr", Fs, D, (vF, vD)), ("cnn", X, D, (vd["X"], vD))):
        for s in range(NSEED):
            p, met, hist = fit(kind, s, Xin, Din, y, vd, vin, log)
            res[kind].append(dict(seed=s, best_val_mean_recall=round(met, 4), epochs=len(hist), history=hist))
            if kind not in best or met > best[kind][1]:
                best[kind] = (p, met, s)
    lr_w = dict(w=best["lr"][0]["w"], b=best["lr"][0]["b"], mean=mu, std=sd)
    models = dict(cnn=dict(kind="cnn", weights=to_json_weights(best["cnn"][0]), seed=best["cnn"][2]),
                  logreg=dict(kind="logreg", weights=to_json_weights(lr_w), seed=best["lr"][2],
                              features=["g(z_c)", "g(sum3x3 z / 3)", "g(z_c - mean8 z)", "log c_c", "g(med_c/sigma)",
                                        "valid fraction 9x9"]))
    for k, mdl in models.items():                        # re-score with the rounded weights through the numpy forward
        sc = detect.forward(mdl, vd["X"], vD)
        met, rec, thr = val_metric(sc, vd)
        mdl["val"] = dict(mean_recall=round(met, 4), recall=[round(float(x), 4) for x in rec], threshold=thr)
    shipped = "cnn" if models["cnn"]["val"]["mean_recall"] > models["logreg"]["val"]["mean_recall"] else "logreg"
    log(f"shipped: {shipped}; cnn {models['cnn']['val']}; logreg {models['logreg']['val']}")
    res["models"], res["shipped"] = models, shipped
    # learning curve (seed 0, nested subsets)
    order = list(np.random.default_rng(SEED_LC).permutation(len(sp["train"])))
    lc = []
    for k in (12,):
        names = [sp["train"][i] for i in order[:k]]
        Xk, Dk, yk = train_set(names)
        p, met, hist = fit("cnn", 0, Xk, Dk, yk, vd, (vd["X"], vD), log)
        lc.append(dict(n_fields=k, fields=names, n_samples=len(yk), best_val_mean_recall=round(met, 4)))
    lc.append(dict(n_fields=len(sp["train"]), n_samples=len(y), best_val_mean_recall=res["cnn"][0]["best_val_mean_recall"]))
    res["learning_curve"] = lc
    log(f"learning curve: {[(x['n_fields'], x['best_val_mean_recall']) for x in lc]}")
    res["generated_local"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    json.dump(res, open(CAND, "w"), indent=1)


def models_for_scoring(src):
    """{'CNN': cnn, 'LR': logreg} from candidate_models.json (pre-freeze) or e30.json (frozen)."""
    if src == "frozen":
        j = json.load(open(MODEL))
        mm = {j["model"]["kind"]: j["model"], j["ablation"]["kind"]: j["ablation"]}
    else:
        mm = json.load(open(CAND))["models"]
    return {"CNN": mm["cnn"], "LR": mm["logreg"]}


def score_all(cd, models):
    from orrery import detect
    D = detect.onehot(cd["det"])
    out = {k: (detect.forward(m, cd["X"], D) if len(cd["e"]) else np.zeros(0)) for k, m in models.items()}
    out["Z"] = np.asarray(cd["z"], float)
    return out


# ---------------------------------------------------------------- sim-to-real check (diagnostic, before Part 2)
def s2r_field(name):
    import build_field as bf
    p = f"{TRAIN}/s2r_{name}.json"
    if os.path.exists(p):
        return name
    models = models_for_scoring("candidate")
    fd = load(name)
    lam, det, mjd = meta_arrays(fd)
    v = live_v(fd)
    st = stage1(v, lam, det, mjd)
    cd = cands(st, det)
    sc = score_all(cd, models)
    ast, nq = bf.known_asteroids(mjd, S, fd["ra0"], fd["dec0"])
    rows = []
    for e in range(len(mjd)):
        for o in ast[e]:
            if not (o["vmag"] < VMAX and EDGE <= o["x"] <= S - 1 - EDGE and EDGE <= o["y"] <= S - 1 - EDGE):
                continue
            if st["prev"][e] < 0:
                continue
            k = np.flatnonzero((cd["e"] == e) & (np.hypot(cd["x"] - o["x"], cd["y"] - o["y"]) <= R_AST))
            r = dict(frame=e, name=o["name"], vmag=o["vmag"], det=int(det[e]), has_candidate=bool(len(k)))
            if len(k):
                j = k[np.argmax(sc["Z"][k])]
                r.update(z=float(sc["Z"][j]), **{s: float(sc[s][j]) for s in ("CNN", "LR")})
            rows.append(r)
    json.dump(dict(field=name, rows=rows, skybot_queries=nq, n_pictures=len(mjd)), open(p, "w"))
    print(name, len(rows), "sightings", sum(r["has_candidate"] for r in rows), "with candidates", flush=True)
    return name


def sim2real(workers=1):
    from multiprocessing import Pool
    from scipy.stats import ks_2samp
    from orrery import detect
    with Pool(workers) as pl:
        list(pl.imap_unordered(s2r_field, OLD_ECL))
    res = json.load(open(CAND))
    sp = split()
    vd = _val_data(sp["val"])
    vD = detect.onehot(vd["D"])
    models = models_for_scoring("candidate")
    bins = [3, 5, 8, 12, 20, np.inf]
    out = {}
    for s in ("CNN", "LR"):
        sc = detect.forward(models[s], vd["X"], vD)
        _, _, thr = val_metric(sc, vd)
        pos = ~vd["isneg"]
        # injected: per trial the candidate with the largest z (as for the real sightings)
        tri, zz, ss = vd["trial"][pos], vd["Z"][pos], sc[pos]
        o = np.lexsort((-zz, tri))
        first = np.r_[True, tri[o][1:] != tri[o][:-1]]
        zi, si = zz[o][first], ss[o][first]
        real = [r for n in OLD_ECL for r in json.load(open(f"{TRAIN}/s2r_{n}.json"))["rows"] if r["has_candidate"]]
        zr, sr = np.array([r["z"] for r in real]), np.array([r[s] for r in real])
        rows = []
        for a, b in zip(bins[:-1], bins[1:]):
            mi, mr = (zi >= a) & (zi < b), (zr >= a) & (zr < b)
            rows.append(dict(z_bin=[a, None if np.isinf(b) else b], n_injected=int(mi.sum()), n_real=int(mr.sum()),
                             frac_above_val_r3_injected=round(float(np.mean(si[mi] > thr)), 4) if mi.any() else None,
                             frac_above_val_r3_real=round(float(np.mean(sr[mr] > thr)), 4) if mr.any() else None,
                             wilson_real=wilson(int(np.sum(sr[mr] > thr)), int(mr.sum())),
                             ks_p=round(float(ks_2samp(si[mi], sr[mr]).pvalue), 4) if mi.sum() > 1 and mr.sum() > 1 else None))
        out[s] = dict(val_threshold_r3=thr, bins=rows)
    allrows = [r for n in OLD_ECL for r in json.load(open(f"{TRAIN}/s2r_{n}.json"))["rows"]]
    res["sim_to_real"] = dict(fields=OLD_ECL, n_sightings=len(allrows), n_with_candidate=sum(r["has_candidate"] for r in allrows),
                              n_objects=len({r["name"] for r in allrows}), per_scorer=out,
                              rule="SkyBoT V < 19, >= 3 px from the edge, pictures with an M2 map; the largest-z stage-1 "
                                   "candidate within 2 px; injected = validation trials' largest-z candidate within 1.5 px")
    json.dump(res, open(CAND, "w"), indent=1)
    print(json.dumps(res["sim_to_real"], indent=1))


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(max(0.0, float(c - h)), 4), round(min(1.0, float(c + h)), 4)]


# ---------------------------------------------------------------- one field, every method
def analyse(fd, imgs, lam, det, mjd, models, crop=None):
    """Live stage 1 + scorers, and the component maps of every comparison method (E27 definitions)."""
    from orrery import e1
    import run_e27
    from orrery.change import change_model
    f = f_floor()
    v = live_v(fd, imgs)
    vm = e1.frame_arrays(fd, imgs)[0]                     # hard-bad + OUTLIER masked (E27 primary)
    if crop is not None:
        v, vm, lam = v[:, crop, crop], vm[:, crop, crop], lam[:, crop, crop]
    st = stage1(v, lam, det, mjd)
    r = st["r"]
    m2 = e1.blink_d1(r["vs"], r["sigma"], r["prev_det"], f)
    rm = change_model(vm, lam, det, mjd, f=f)
    maps = dict(M2=m2, M3=list(e1.m3_maps(r["vs"], r["sigma"], det, f)), M3B=list(st["z"]),
                M2D=[None if m is None else e1.despike(m) for m in m2], M3BD=[e1.despike(m) for m in st["z"]],
                M3BF=[e1.despike(m) for m in run_e27.m3b_maps(rm["vs"], rm["sigma"], det, f)])
    cd = cands(st, det)
    return v, st, maps, cd, score_all(cd, models)


def unplanted_record(st, maps, cd, sc, pre=""):
    from orrery import detect
    out = {f"{pre}npic": int(sum(np.isfinite(z).any() for z in st["z"])),
           f"{pre}npix": int(sum(np.isfinite(z).sum() for z in st["z"])), f"{pre}cand_e": cd["e"]}
    for s in SCORERS:
        out[f"{pre}fa_{s}"] = sc[s]
    for m in COMP:
        V, Dl, Ee = [], [], []
        for e, mp in enumerate(maps[m]):
            if mp is None or not np.isfinite(mp).any():
                continue
            a, d = detect.comp_events(mp, KMIN)
            V.append(a); Dl.append(d); Ee.append(np.full(len(a), e))
        out[f"{pre}ev_{m}"] = np.concatenate(V) if V else np.zeros(0)
        out[f"{pre}evd_{m}"] = np.concatenate(Dl) if Dl else np.zeros(0, np.int32)
        out[f"{pre}eve_{m}"] = np.concatenate(Ee) if Ee else np.zeros(0, int)
        out[f"{pre}maps_{m}"] = np.array([e for e, mp in enumerate(maps[m]) if mp is not None and np.isfinite(mp).any()])
    return out


def trial_record(v, st, maps, cd, sc, pos, li, radius, off=0):
    """Per trial: frame, x, y, level, any stage-1 candidate, per scorer the best score within `radius`, per component
    method the exact kmax within `radius`."""
    from orrery import detect
    rows = []
    Sg = v.shape[1]
    for e, (x, y) in enumerate(pos):
        x, y = x - off, y - off
        if not (-0.5 <= x < Sg - 0.5 and -0.5 <= y < Sg - 0.5) or not trial_ok(v, st["prev"], e, x, y):
            continue
        k = np.flatnonzero((cd["e"] == e) & (np.hypot(cd["x"] - x, cd["y"] - y) <= radius))
        r = [e, x, y, li, float(len(k) > 0)] + [float(sc[s][k].max()) if len(k) else -np.inf for s in SCORERS]
        r += [detect.comp_kmax(maps[m][e], x, y, radius) if maps[m][e] is not None else -np.inf for m in COMP]
        rows.append(r)
    return rows


TRIAL_COLS = ["frame", "x", "y", "level", "stage1"] + list(SCORERS) + list(COMP)


def field_record(name, fd, models, levels_seed, fi, crop_too=True, examples=None):
    from orrery import e1
    lam, det, mjd = meta_arrays(fd)
    rec = dict(keys=np.array([f"{f['obs_id']}_D{f['det']}" for f in fd["frames"]]), det=det)
    v, st, maps, cd, sc = analyse(fd, None, lam, det, mjd, models)
    rec.update(unplanted_record(st, maps, cd, sc))
    if examples is not None:
        examples_neg(rec, st, cd, sc, examples)
    if crop_too:
        vc, stc, mapsc, cdc, scc = analyse(fd, None, lam, det, mjd, models, CROP)
        rec.update(unplanted_record(stc, mapsc, cdc, scc, "crop_"))
    rows, rows_c = [], []
    old = e1.SEED0
    e1.SEED0 = levels_seed
    try:
        for li, snr in enumerate(e1.LEVELS):
            pos = e1.draw_positions(fd, li, fi)
            imgs = [e1.plant(fr, x, y, snr)[0] for fr, (x, y) in zip(fd["frames"], pos)]
            v, st, maps, cd, sc = analyse(fd, imgs, lam, det, mjd, models)
            rows += trial_record(v, st, maps, cd, sc, pos, li, R_PLANT)
            if examples is not None and snr == 8:
                examples_pos(rec, st, cd, sc, pos, examples)
            if crop_too:
                vc, stc, mapsc, cdc, scc = analyse(fd, imgs, lam, det, mjd, models, CROP)
                rows_c += trial_record(vc, stc, mapsc, cdc, scc, pos, li, R_PLANT, off=CROP.start)
    finally:
        e1.SEED0 = old
    rec["trials"] = np.array(rows, float).reshape(-1, len(TRIAL_COLS))
    rec["crop_trials"] = np.array(rows_c, float).reshape(-1, len(TRIAL_COLS))
    return rec


def _stamp(st, e, x, y, h=7):
    """15 x 15 (picture/sigma, reference/sigma, z) around (x, y), NaN off-grid (for the example figure)."""
    Sg = st["z"].shape[1]
    out = np.full((3, 2 * h + 1, 2 * h + 1), np.nan)
    for k, a in enumerate((st["vs"][e] / st["sig"][e], st["med"][e] / st["sig"][e], st["z"][e])):
        P = np.full((Sg + 2 * h, Sg + 2 * h), np.nan)
        P[h:h + Sg, h:h + Sg] = a
        out[k] = P[int(y):int(y) + 2 * h + 1, int(x):int(x) + 2 * h + 1]
    return out


def examples_neg(rec, st, cd, sc, ex):
    s = sc[ex["scorer"]]
    k = np.argsort(-s)[:2]
    rec["ex_neg"] = np.array([_stamp(st, cd["e"][j], cd["x"][j], cd["y"][j]) for j in k]).reshape(-1, 3, 15, 15)
    rec["ex_neg_score"], rec["ex_neg_z"] = s[k], cd["z"][k]


def examples_pos(rec, st, cd, sc, pos, ex):
    s = sc[ex["scorer"]]
    out, scs, zs = [], [], []
    for e, (x, y) in enumerate(pos):
        k = np.flatnonzero((cd["e"] == e) & (np.hypot(cd["x"] - x, cd["y"] - y) <= R_PLANT))
        if len(k) and s[k].max() > ex["t"]:
            j = k[np.argmax(s[k])]
            out.append(_stamp(st, e, cd["x"][j], cd["y"][j])); scs.append(s[j]); zs.append(cd["z"][j])
        if len(out) == 2:
            break
    rec["ex_pos"] = np.array(out).reshape(-1, 3, 15, 15)
    rec["ex_pos_score"], rec["ex_pos_z"] = np.array(scs), np.array(zs)


# ---------------------------------------------------------------- calibration (thresholds) and e30.json
def calib_field(name):
    p = f"{RUNS}/calib_{name}.npz"
    if os.path.exists(p):
        return name
    t0 = time.time()
    fd = load(name)
    lam, det, mjd = meta_arrays(fd)
    v, st, maps, cd, sc = analyse(fd, None, lam, det, mjd, {})       # scorers applied in calib() to the stored cutouts
    sc.update({s: np.zeros(len(cd["e"])) for s in ("CNN", "LR")})
    rec = unplanted_record(st, maps, cd, sc)
    np.savez_compressed(p[:-4] + ".tmp.npz", cand_X=cd["X"], cand_det=cd["det"], **rec)
    os.replace(p[:-4] + ".tmp.npz", p)
    print(f"calib {name}: {rec['npic']} pictures, {len(rec['fa_Z'])} candidates, {time.time() - t0:.0f}s", flush=True)
    return name


def _thr_method(recs, m, rate):
    from orrery import detect
    if m in SCORERS:
        return detect.threshold_scores(np.concatenate([r[f"fa_{m}"] for r in recs]), sum(int(r["npic"]) for r in recs), rate)
    return detect.threshold_events(np.concatenate([r[f"ev_{m}"] for r in recs]), np.concatenate([r[f"evd_{m}"] for r in recs]),
                                   sum(len(r[f"maps_{m}"]) for r in recs), rate, KMIN)


def rate_table(recs, s, n=20):
    from orrery import detect
    sc = np.concatenate([r[f"fa_{s}"] for r in recs])
    npic, npix = sum(int(r["npic"]) for r in recs), sum(int(r["npix"]) for r in recs)
    rates = np.geomspace(3.0, 0.003, n)
    t = [detect.threshold_scores(sc, npic, q) for q in rates]
    return dict(score=[round(float(x), 6) for x in t],
                false_per_pixel=[float(f"{np.sum(sc > x) / npix:.6g}") for x in t],
                false_per_picture_48px=[round(float(np.sum(sc > x) / npic), 6) for x in t])


def calib_fields(workers=1):
    from multiprocessing import Pool
    os.makedirs(RUNS, exist_ok=True)
    with Pool(workers) as pl:
        list(pl.imap_unordered(calib_field, [x["field"] for x in group("calib")]))


def calib(workers=1):
    if os.path.exists(THR):
        sys.exit(f"{THR} exists (frozen)")
    names = [x["field"] for x in group("calib")]
    calib_fields(workers)
    recs = [dict(np.load(f"{RUNS}/calib_{n}.npz")) for n in names]
    from orrery import detect
    models = models_for_scoring("candidate")
    for r in recs:
        for s_, mdl in models.items():
            r[f"fa_{s_}"] = detect.forward(mdl, r["cand_X"], detect.onehot(r["cand_det"])) if len(r["cand_det"]) else np.zeros(0)
    thr = {}
    rng = np.random.default_rng(SEED_CALBOOT)
    boots = rng.integers(0, len(recs), size=(500, len(recs)))
    for m in SCORERS + COMP:
        t = {op: _thr_method(recs, m, r) for op, r in zip(OPS, RATES)}
        tb = {op: [] for op in OPS}
        for b in boots:
            rb = [recs[i] for i in b]
            for op, r in zip(OPS, RATES):
                tb[op].append(_thr_method(rb, m, r))
        thr[m] = dict(t=t, boot95={op: [float(np.percentile(tb[op], 2.5)), float(np.percentile(tb[op], 97.5))] for op in OPS})
        print(m, t, thr[m]["boot95"], flush=True)
    assert all(thr[m]["t"][op] > KMIN for m in COMP for op in OPS)
    cand = json.load(open(CAND))
    shipped = cand["shipped"]
    sname = "CNN" if shipped == "cnn" else "LR"
    tables = {s: rate_table(recs, s) for s in SCORERS}
    out = dict(protocol=PROTOCOL, fields=names, f=f_floor(), rates=list(RATES), thresholds=thr,
               n_pictures=sum(int(r["npic"]) for r in recs), n_valid_pixels=sum(int(r["npix"]) for r in recs),
               n_candidates=int(sum(len(r["fa_Z"]) for r in recs)),
               n_maps={m: int(sum(len(r[f"maps_{m}"]) for r in recs)) for m in COMP},
               rule="smallest t with pooled #(score > t) / pictures <= rate (scorers); smallest exact k >= 2 with pooled "
                    "components / maps <= rate (component methods); calibration fields, nothing planted, all pictures",
               rate_tables=tables, frozen_local=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               code_sha256={x: sha(x) for x in ("src/orrery/detect.py", "scripts/run_e30.py", "src/orrery/e1.py",
                                                "src/orrery/change.py", "scripts/run_e27.py")})
    json.dump(out, open(THR, "w"), indent=1)
    # e30.json: both models, the shipped scorer's thresholds and table, Z as the fallback
    mm = cand["models"]
    other = "logreg" if shipped == "cnn" else "cnn"
    blocks = dict(model=dict(mm[shipped], id=f"e30-{shipped}"), ablation=dict(mm[other], id=f"e30-{other}"))
    wsha = __import__("hashlib").sha256(json.dumps(blocks, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    oname = "LR" if sname == "CNN" else "CNN"
    blocks["model"].update(thresholds=thr[sname]["t"], rate_table=tables[sname])
    blocks["ablation"].update(thresholds=thr[oname]["t"], rate_table=tables[oname])
    j = dict(experiment="E30", protocol=PROTOCOL, protocol_part1_sha256=part1_sha(), id=f"e30-{shipped}-{wsha[:8]}",
             weights_sha256=wsha, weights_sha256_rule="sha256 of json.dumps({'model': ..., 'ablation': ...} before "
             "thresholds and rate tables were added, sort_keys=True, separators=(',', ':'))",
             use="model", f=f_floor(), zmin=3.0, half=4, grid_px=S,
             channels=["g(vs/sigma)", "g(med/sigma)", "g(z)", "log c", "finite(z)"], g="sign(x) log1p|x|",
             thresholds=thr[sname]["t"], rates=dict(zip(OPS, RATES)), rate_table=tables[sname],
             model=blocks["model"], ablation=blocks["ablation"],
             zrank=dict(kind="zrank", id="e30-zrank", thresholds=thr["Z"]["t"], rate_table=tables["Z"]),
             note="thresholds are false candidates per 48 x 48 px picture on the calibration fields; rate_table gives "
                  "false candidates per valid pixel (multiply by a picture's valid pixels)")
    os.makedirs(os.path.dirname(MODEL), exist_ok=True)
    json.dump(j, open(MODEL, "w"), separators=(",", ":"))
    print("weights sha256", wsha, "e30.json sha256", sha(MODEL), flush=True)


def part1_sha():
    """sha256 of docs/protocols/E30.md's Part 1: the text before the first '## Part 2' or '## Deviations' line."""
    import hashlib
    txt = open(PROTOCOL).read()
    i = min(x for x in (txt.find("\n## Part 2"), txt.find("\n## Deviations"), len(txt)) if x >= 0)
    return hashlib.sha256(txt[:i].encode()).hexdigest()


def weights_check():
    j = json.load(open(MODEL))
    blocks = {k: {q: v for q, v in j[k].items() if q not in ("thresholds", "rate_table")} for k in ("model", "ablation")}
    w = __import__("hashlib").sha256(json.dumps(blocks, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert w == j["weights_sha256"], "weights in e30.json do not match their sha256"
    return w


# ---------------------------------------------------------------- frozen test
def test_field(name):
    p = f"{RUNS}/test_{name}.npz"
    if os.path.exists(p):
        return name
    t0 = time.time()
    weights_check()
    j = json.load(open(MODEL))
    sname = "CNN" if j["model"]["kind"] == "cnn" else "LR"
    fi = [x["field"] for x in group("test")].index(name)
    fd = load(name)
    rec = field_record(name, fd, models_for_scoring("frozen"), SEED_PLANT, fi,
                       examples=dict(scorer=sname, t=j["thresholds"]["r3"]))
    np.savez_compressed(p[:-4] + ".tmp.npz", **rec)
    os.replace(p[:-4] + ".tmp.npz", p)
    print(f"test {name}: {len(rec['trials'])} trials, {time.time() - t0:.0f}s", flush=True)
    return name


def ecl_field(name):
    import build_field as bf
    from orrery import detect
    p = f"{RUNS}/ecl_{name}.json"
    if os.path.exists(p):
        return name
    weights_check()
    fd = load(name)
    lam, det, mjd = meta_arrays(fd)
    v, st, maps, cd, sc = analyse(fd, None, lam, det, mjd, models_for_scoring("frozen"))
    ast, nq = bf.known_asteroids(mjd, S, fd["ra0"], fd["dec0"])
    rows = []
    for e in range(len(mjd)):
        for o in ast[e]:
            if not (o["vmag"] < VMAX and EDGE <= o["x"] <= S - 1 - EDGE and EDGE <= o["y"] <= S - 1 - EDGE):
                continue
            if st["prev"][e] < 0:
                continue
            k = np.flatnonzero((cd["e"] == e) & (np.hypot(cd["x"] - o["x"], cd["y"] - o["y"]) <= R_AST))
            r = dict(frame=e, name=o["name"], vmag=o["vmag"], det=int(det[e]), stage1=bool(len(k)),
                     **{s: float(sc[s][k].max()) if len(k) else None for s in SCORERS},
                     **{m: detect.comp_kmax(maps[m][e], o["x"], o["y"], R_AST) if maps[m][e] is not None else None
                        for m in COMP})
            rows.append(r)
    json.dump(dict(field=name, n_pictures=len(mjd), skybot_queries=nq, rows=rows), open(p, "w"))
    print(name, len(rows), "sightings", flush=True)
    return name


def run(grp, workers=1):
    from multiprocessing import Pool
    assert os.path.exists(MODEL) and "## Part 2" in open(PROTOCOL).read(), "Part 2 not frozen"
    import run_e1
    run_e1.ROOT = ROOT
    todo = [x["field"] for x in group(grp) if run_e1.ready(x["field"])]
    with Pool(workers) as pl:
        list(pl.imap_unordered(test_field if grp == "test" else ecl_field, todo))


# ---------------------------------------------------------------- summary
METH = SCORERS + COMP


def tallies(R, th, pre="", excl=None):
    """Detections D[m][op] [field, level], trials N [field, level], false candidates FA[m][op] [field] and pictures
    NM[m] [field]; pictures whose key is in `excl` are removed from trials and false-candidate counts."""
    nl = 5
    N = np.zeros((len(R), nl))
    D = {m: {op: np.zeros((len(R), nl)) for op in OPS} for m in METH}
    FA = {m: {op: np.zeros(len(R)) for op in OPS} for m in METH}
    NM = {m: np.zeros(len(R)) for m in METH}
    for i, r in enumerate(R):
        bad = np.isin(r["keys"], list(excl)) if excl else np.zeros(len(r["keys"]), bool)
        tr = r[pre + "trials"]
        tr = tr[~bad[tr[:, 0].astype(int)]] if len(tr) else tr
        lev = tr[:, 3].astype(int)
        N[i] = np.bincount(lev, minlength=nl)
        for m in METH:
            col = tr[:, TRIAL_COLS.index(m)]
            for op in OPS:
                D[m][op][i] = np.bincount(lev, weights=(col > th[m][op]).astype(float), minlength=nl)
            if m in SCORERS:
                ok = ~bad[r[pre + "cand_e"]]
                sc = r[pre + "fa_" + m][ok]
                mp = r[pre + "maps_M3B"]
                NM[m][i] = int(np.sum(~bad[mp])) if len(mp) else 0
                for op in OPS:
                    FA[m][op][i] = np.sum(sc > th[m][op])
            else:
                ok = ~bad[r[pre + "eve_" + m]]
                ev, evd = r[pre + "ev_" + m][ok], r[pre + "evd_" + m][ok]
                mp = r[pre + "maps_" + m]
                NM[m][i] = int(np.sum(~bad[mp])) if len(mp) else 0
                for op in OPS:
                    FA[m][op][i] = evd[ev > th[m][op]].sum()
    return D, N, FA, NM


def q(a, p):
    return float(np.percentile(np.asarray(a, float), 100 * p))


def stats(D, N, FA, NM, idx, ship, rates=RATES):
    res, ver = {}, {}
    for m in METH:
        res[m] = {}
        for op, rate in zip(OPS, rates):
            rec = D[m][op].sum(0) / N.sum(0)
            rb = D[m][op][idx].sum(1) / N[idx].sum(1)
            mb = rb.mean(1)
            fa = FA[m][op].sum() / NM[m].sum()
            fab = FA[m][op][idx].sum(1) / NM[m][idx].sum(1)
            res[m][op] = dict(target_rate=rate, recall=[round(float(x), 4) for x in rec],
                              recall_ci95=[[round(q(rb[:, l], .025), 4), round(q(rb[:, l], .975), 4)] for l in range(5)],
                              mean_recall=round(float(rec.mean()), 4), mean_recall_ci95=[round(q(mb, .025), 4), round(q(mb, .975), 4)],
                              realised_fa=round(float(fa), 5), realised_fa_ci95=[round(q(fab, .025), 5), round(q(fab, .975), 5)],
                              realised_fa_over_target=round(float(fa / rate), 3), false_candidates=int(FA[m][op].sum()),
                              n_pictures=int(NM[m].sum()), n_detected=D[m][op].sum(0).astype(int).tolist())
    for ch in METH:
        if ch in ("M2", "M3B"):
            continue
        ver[ch] = {}
        for op, rate in zip(OPS, rates):
            v = {}
            for base in ("M3B", "M2"):
                d = (D[ch][op] - D[base][op])
                dp = float((d.sum(0) / N.sum(0)).mean())
                db = (d[idx].sum(1) / N[idx].sum(1)).mean(1)
                v[f"diff_vs_{base}"] = round(dp, 4)
                v[f"diff_vs_{base}_ci95"] = [round(q(db, .025), 4), round(q(db, .975), 4)]
            v["fa_within_1p5x"] = bool(res[ch][op]["realised_fa"] <= 1.5 * rate)
            v["qualifies"] = bool(v["diff_vs_M3B_ci95"][0] > 0 and v["diff_vs_M2_ci95"][0] > 0 and v["fa_within_1p5x"])
            v["below_zero"] = bool(v["diff_vs_M3B_ci95"][1] < 0 or v["diff_vs_M2_ci95"][1] < 0)
            ver[ch][op] = v
        nq = sum(ver[ch][op]["qualifies"] for op in OPS)
        nb = sum(ver[ch][op]["below_zero"] for op in OPS)
        ver[ch]["n_qualifying"] = nq
        ver[ch]["result"] = "beats" if nq >= 2 else ("loss" if nb >= 2 else "tie")
    return res, ver


def summary():
    j = json.load(open(MODEL))
    thr = json.load(open(THR))
    cand = json.load(open(CAND))
    ship = "CNN" if j["model"]["kind"] == "cnn" else "LR"
    other = "LR" if ship == "CNN" else "CNN"
    th = {ship: j["thresholds"], other: j["ablation"]["thresholds"], "Z": j["zrank"]["thresholds"]}
    th.update({m: thr["thresholds"][m]["t"] for m in COMP})
    tests = [x["field"] for x in group("test") if os.path.exists(f"{RUNS}/test_{x['field']}.npz")]
    R = [dict(np.load(f"{RUNS}/test_{n}.npz")) for n in tests]
    nf = len(R)
    idx = np.random.default_rng(SEED_BOOT).integers(0, nf, size=(2000, nf))
    D, N, FA, NM = tallies(R, th)
    res, ver = stats(D, N, FA, NM, idx, ship)
    # shared pictures
    sp = split()
    others = set().union(*[picture_keys(n) for n in sp["train"] + sp["val"] + [x["field"] for x in group("calib")]])
    shared = set().union(*[set(r["keys"].tolist()) & others for r in R])
    Dx, Nx, FAx, NMx = tallies(R, th, excl=shared)
    res_x, ver_x = stats(Dx, Nx, FAx, NMx, idx, ship)
    # matched realised rate on the test unplanted passes (secondary)
    from orrery import detect
    thm = {}
    for m in METH:
        thm[m] = {}
        for op, rate in zip(OPS, RATES):
            if m in SCORERS:
                thm[m][op] = detect.threshold_scores(np.concatenate([r["fa_" + m] for r in R]), sum(len(r["maps_M3B"]) for r in R), rate)
            else:
                thm[m][op] = detect.threshold_events(np.concatenate([r["ev_" + m] for r in R]), np.concatenate([r["evd_" + m] for r in R]),
                                                     sum(len(r["maps_" + m]) for r in R), rate, KMIN)
    res_m, ver_m = stats(*tallies(R, thm), idx, ship)
    # crops
    res_c, ver_c = stats(*tallies(R, th, pre="crop_"), idx, ship)
    # by band and detector at r3; stage-1 ceiling
    bands = {n: band_of(n) for n in tests}
    by_band = {}
    for b in ("high", "mid", "low"):
        sel = [i for i, n in enumerate(tests) if bands[n] == b]
        by_band[b] = {m: dict(mean_recall=round(float((D[m]["r3"][sel].sum(0) / N[sel].sum(0)).mean()), 4),
                              realised_fa=round(float(FA[m]["r3"][sel].sum() / NM[m][sel].sum()), 5)) for m in METH}
        by_band[b]["n_fields"] = len(sel)
    alltr = np.concatenate([r["trials"] for r in R])
    dets = np.concatenate([r["det"][r["trials"][:, 0].astype(int)] for r in R])
    by_det = {}
    for d in range(1, 7):
        t = alltr[dets == d]
        lev = t[:, 3].astype(int)
        n_l = np.bincount(lev, minlength=5)
        by_det[str(d)] = {m: round(float(np.mean(np.bincount(lev, weights=(t[:, TRIAL_COLS.index(m)] > th[m]["r3"]).astype(float),
                                                              minlength=5) / np.maximum(n_l, 1))), 4) for m in METH}
        by_det[str(d)]["n_trials"] = int(len(t))
    lev = alltr[:, 3].astype(int)
    ceiling = (np.bincount(lev, weights=alltr[:, 4], minlength=5) / np.bincount(lev, minlength=5)).round(4).tolist()
    # asteroids
    E = [json.load(open(f"{RUNS}/ecl_{x['field']}.json")) for x in group("ecl") if os.path.exists(f"{RUNS}/ecl_{x['field']}.json")]
    rows = [dict(r, field=e["field"]) for e in E for r in e["rows"]]
    ast = {}
    for m in METH:
        ast[m] = {}
        for op in OPS:
            hit = [r[m] is not None and r[m] > th[m][op] for r in rows]
            objs = {}
            for r, h in zip(rows, hit):
                objs[(r["field"], r["name"])] = objs.get((r["field"], r["name"]), False) or h
            ast[m][op] = dict(hits=int(sum(hit)), n=len(rows), recall=round(sum(hit) / len(rows), 4) if rows else None,
                              wilson95=wilson(int(sum(hit)), len(rows)), objects_detected=int(sum(objs.values())),
                              n_objects=len(objs), objects_wilson95=wilson(int(sum(objs.values())), len(objs)))
    ast_stage1 = dict(n=len(rows), with_candidate=int(sum(r["stage1"] for r in rows)))
    out = dict(experiment="E30", protocol=PROTOCOL, protocol_part1_sha256=j["protocol_part1_sha256"],
               question="does a small CNN scoring M3b candidates beat M3b and M2 at equal false-candidate rate (live preprocessing)?",
               shipped=ship, model_file=MODEL, model_file_sha256=sha(MODEL), weights_sha256=j["weights_sha256"],
               thresholds_file=THR, thresholds_sha256=sha(THR), thresholds=th, f=j["f"],
               methods=dict(CNN="stage-1 candidates scored by the CNN", LR="stage-1 candidates scored by the 6-feature logistic regression",
                            Z="stage-1 candidates scored by M3b z (the fallback)", **COMP_DESC),
               training=dict({k: cand[k] for k in ("n_train", "n_pos", "n_neg", "val_trials", "val_negatives", "val_pictures",
                                                   "z_val", "learning_curve", "shipped")},
                             seeds={k: [dict(seed=x["seed"], best_val_mean_recall=x["best_val_mean_recall"], epochs=x["epochs"])
                                        for x in cand[k]] for k in ("cnn", "lr")},
                             models_val={k: v["val"] for k, v in cand["models"].items()}, sim_to_real=cand.get("sim_to_real")),
               calibration=dict(fields=thr["fields"], n_pictures=thr["n_pictures"], n_candidates=thr["n_candidates"],
                                bootstrap95={m: thr["thresholds"][m]["boot95"] for m in METH}),
               test_fields=tests, n_test_fields=nf, test_pictures=int(sum(len(r["keys"]) for r in R)),
               levels_snr_frame=[3, 5, 8, 12, 20], n_trials=N.sum(0).astype(int).tolist(),
               bootstrap=dict(n=2000, seed=SEED_BOOT, unit="test field"),
               results=res, verdict={ship: ver[ship]}, paired_all=ver, result=ver[ship]["result"],
               decision_rule="shipped scorer qualifies at a rate iff the 95% intervals of (shipped - M3B live) and (shipped - "
                             "M2 live) mean recall both have lower end > 0 and its realised test rate <= 1.5 x target; beats "
                             "iff it qualifies at >= 2 of 3 rates; loss iff at >= 2 rates an interval lies wholly below zero; "
                             "otherwise tie",
               shared_pictures=dict(n_shared_test_pictures=len(shared), results=res_x, verdict={ship: ver_x[ship]}),
               matched_rate=dict(thresholds=thm, results=res_m), crops_20px=dict(results=res_c),
               by_band_r3=by_band, by_detector_r3=by_det, stage1_ceiling=ceiling,
               asteroids=dict(fields=[e["field"] for e in E], n_sightings=len(rows),
                              stage1=ast_stage1, per_method=ast, skybot_queries=int(sum(e["skybot_queries"] for e in E)),
                              rule="SkyBoT V < 19, >= 3 px from the edge, pictures with an M2 map; detection within 2 px; "
                                   "frozen calibration thresholds; live preprocessing (M3BF masked)"),
               generated_local=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    json.dump(out, open(OUT, "w"), indent=1, default=float)
    flat(out)
    figure(out)
    figure_examples(R, ship)
    for m in METH:
        print(m, {op: (res[m][op]["mean_recall"], res[m][op]["mean_recall_ci95"], res[m][op]["realised_fa"]) for op in OPS})
    print(json.dumps(ver[ship], indent=1))
    return out


def flat(out):
    s = dict(source=OUT, shipped=out["shipped"], result=out["result"], n_test_fields=out["n_test_fields"],
             test_pictures=out["test_pictures"], n_calib_fields=len(out["calibration"]["fields"]),
             n_sightings=out["asteroids"]["n_sightings"], n_shared_test_pictures=out["shared_pictures"]["n_shared_test_pictures"],
             n_train_samples=out["training"]["n_train"], val_cnn=out["training"]["models_val"]["cnn"]["mean_recall"],
             val_lr=out["training"]["models_val"]["logreg"]["mean_recall"], val_z=round(out["training"]["z_val"]["mean_recall"], 4))
    for m in METH:
        for op in OPS:
            r = out["results"][m][op]
            s[f"mean_recall_{m}_{op}"] = r["mean_recall"]
            s[f"mean_recall_{m}_{op}_lo"], s[f"mean_recall_{m}_{op}_hi"] = r["mean_recall_ci95"]
            for l, sn in enumerate(out["levels_snr_frame"]):
                s[f"recall{sn}_{m}_{op}"] = r["recall"][l]
            s[f"fa_{m}_{op}"] = r["realised_fa"]
            s[f"fa_{m}_{op}_lo"], s[f"fa_{m}_{op}_hi"] = r["realised_fa_ci95"]
            s[f"thr_{m}_{op}"] = round(float(out["thresholds"][m][op]), 4)
            a = out["asteroids"]["per_method"][m][op]
            s[f"ast_{m}_{op}_hits"], s[f"ast_{m}_{op}_recall"] = a["hits"], a["recall"]
            s[f"ast_{m}_{op}_objects"] = a["objects_detected"]
            s[f"crop_mean_recall_{m}_{op}"] = out["crops_20px"]["results"][m][op]["mean_recall"]
            s[f"crop_fa_{m}_{op}"] = out["crops_20px"]["results"][m][op]["realised_fa"]
            s[f"matched_mean_recall_{m}_{op}"] = out["matched_rate"]["results"][m][op]["mean_recall"]
    v = out["verdict"][out["shipped"]]
    for op in OPS:
        for b in ("M3B", "M2"):
            s[f"diff_{b}_{op}"] = v[op][f"diff_vs_{b}"]
            s[f"diff_{b}_{op}_lo"], s[f"diff_{b}_{op}_hi"] = v[op][f"diff_vs_{b}_ci95"]
        s[f"qualifies_{op}"] = v[op]["qualifies"]
    s["n_objects"] = out["asteroids"]["per_method"]["Z"]["r3"]["n_objects"]
    s = {k.replace("M3b", "M3B"): val for k, val in s.items()}
    assert all(not any(c in k for c in ".- ") for k in s)
    json.dump(s, open(OUTS, "w"), indent=1)


def figure(out, op="r3"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ship = out["shipped"]
    other = "LR" if ship == "CNN" else "CNN"
    show = [ship, other, "Z", "M3B", "M2", "M3BF"]
    col = {ship: "#D55E00", other: "#CC79A7", "Z": "#E69F00", "M3B": "#009E73", "M2": "#56B4E9", "M3BF": "#7f7f7f"}
    sty = {ship: "-", other: "-.", "Z": ":", "M3B": "--", "M2": "--", "M3BF": (0, (1, 1))}
    lab = {"CNN": "CNN score of M3b candidates", "LR": "logistic regression score of M3b candidates",
           "Z": "M3b z of the same candidates", "M3B": "M3b, live", "M2": "M2 one-detector blink, live",
           "M3BF": "M3b with quality flags (reference)"}
    L = out["levels_snr_frame"]
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    for m in show:
        r = out["results"][m][op]
        lo, hi = np.array(r["recall_ci95"]).T
        ax.fill_between(L, lo, hi, color=col[m], alpha=0.13, lw=0)
        ax.plot(L, r["recall"], linestyle=sty[m], marker="o", ms=3.5, color=col[m], lw=1.6, label=lab[m])
    ax.set_xscale("log")
    ax.set_xticks(L)
    ax.set_xticklabels([str(x) for x in L])
    ax.minorticks_off()
    ax.set_xlabel("planted S/N per picture")
    ax.set_ylabel(f"recall at {RATES[OPS.index(op)]:.2f} false candidates per picture")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG + ".png", dpi=200)
    fig.savefig(FIG + ".svg")
    plt.close(fig)


def figure_examples(R, ship):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pos = [(r["ex_pos"][k], r["ex_pos_score"][k], r["ex_pos_z"][k]) for r in R if "ex_pos" in r for k in range(len(r["ex_pos"]))][:2]
    neg = sorted([(r["ex_neg"][k], r["ex_neg_score"][k], r["ex_neg_z"][k]) for r in R if "ex_neg" in r
                  for k in range(len(r["ex_neg"]))], key=lambda t: -t[1])[:2]
    fig, axs = plt.subplots(4, 3, figsize=(5.4, 7.4))
    for row, ((st, sc, z), kind) in enumerate([(p, "injected S/N 8") for p in pos] + [(n, "false (nothing planted)") for n in neg]):
        for k, name in enumerate(("picture / σ", "reference / σ", "z")):
            a = st[k]
            lim = np.nanpercentile(np.abs(st[0]), 99) if k < 2 else max(5, np.nanmax(np.abs(a)))
            axs[row, k].imshow(a, cmap="gray" if k < 2 else "RdBu_r", vmin=-lim if k == 2 else np.nanmin(st[:2]),
                               vmax=lim if k == 2 else np.nanmax(st[:2]), origin="lower")
            axs[row, k].set_xticks([]); axs[row, k].set_yticks([])
            if row == 0:
                axs[row, k].set_title(name, fontsize=8)
        axs[row, 0].set_ylabel(f"{kind}\n{ship} {sc:.1f}, z {z:.1f}", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGEX, dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------- parity fixture
def fixture():
    from orrery import detect
    import run_e27
    j = json.load(open(MODEL))
    rng = np.random.default_rng(2026111554)
    Sg, nd, nper = 20, 3, 8
    det = np.array([1 + (e % nd) for e in range(nd * nper)])
    sigma = rng.uniform(0.8, 1.3, len(det))
    vs = rng.normal(0, 1, (len(det), Sg, Sg)) * sigma[:, None, None]
    yy, xx = np.mgrid[0:Sg, 0:Sg]
    vs += 30 * np.exp(-((xx - 6.3) ** 2 + (yy - 12.6) ** 2) / 1.2)          # a static star
    for e in range(len(det)):                                                 # one moving source per picture
        x0, y0 = 3 + 0.6 * e, 4 + 0.4 * e
        vs[e] += rng.uniform(4, 12) * sigma[e] * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / 1.0)
    vs[5, 2, 17] += 80                                                        # a spike
    vs[7, 15:18, 0:2] = np.nan                                                # off-detector corner
    f = j["f"]
    med, z, c = detect.m3b(vs, sigma, det, f)
    assert np.array_equal(z, run_e27.m3b_maps(vs, sigma, det, f), equal_nan=True)
    cl = []
    for e in range(len(det)):
        x, y, zz = detect.find_candidates(z[e])
        if not len(x):
            continue
        X = detect.cutouts(vs[e], med[e], z[e], c[e], sigma[e], x, y)
        Dh = detect.onehot(np.full(len(x), det[e]))
        lm, la = detect.forward(j["model"], X, Dh), detect.forward(j["ablation"], X, Dh)
        for k in range(len(x)):
            cl.append(dict(e=e, x=int(x[k]), y=int(y[k]), z=float(zz[k]), logit_model=float(lm[k]), logit_ablation=float(la[k]),
                           cutout=[None if not np.isfinite(t) else float(t) for t in X[k].ravel()]))

    def L(a):
        return [[None if not np.isfinite(t) else float(t) for t in fr.ravel()] for fr in a]
    out = dict(S=Sg, f=f, det=det.tolist(), sigma=sigma.tolist(), vs=L(vs), med=L(med), z=L(z), c=L(c), candidates=cl,
               model_id=j["id"], weights_sha256=j["weights_sha256"], seed=2026111554,
               note="generated by scripts/run_e30.py fixture from orrery.detect (== run_e27.m3b_maps)")
    os.makedirs("site/tests/fixtures", exist_ok=True)
    json.dump(out, open("site/tests/fixtures/e30_parity.json", "w"), separators=(",", ":"))
    print(len(cl), "candidates in the fixture")


if __name__ == "__main__":
    a = sys.argv[1]
    w = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 1
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    {"draw": draw, "fetch": fetch, "split": split, "build": lambda: build(w), "train": train,
     "sim2real": lambda: sim2real(w), "calib": lambda: calib(w), "calibfields": lambda: calib_fields(w), "test": lambda: run("test", w),
     "ecl": lambda: run("ecl", w), "summary": lambda: summary(), "fixture": lambda: fixture()}[a]()
