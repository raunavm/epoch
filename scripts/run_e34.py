"""E34 - detector v2 (wider training range + real-star transplants) and tests on real objects (docs/protocols/E34.md).

  build     extra training sets (bright injections, transplants) and validation transplants -> data/phase2/e34/sets/
  train     v2 CNN, 2 seeds -> data/phase2/e34/candidate.json
  calib     v2 thresholds on the E30 calibration candidates, M2-with-flags threshold, new site/data/model/e30.json
  stars     transplant test + v2 false candidates on the 20 E30 test fields -> data/phase2/e34/runs/
  ast       SkyBoT sightings on the e34a fields and (post hoc) the e30e fields -> data/phase2/e34/runs/
  summary   data/findings/e34.json, e34s.json, e34_real.svg
"""
import hashlib, json, os, shutil, sys, time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_e30 as R  # noqa: E402
from orrery import detect, e1  # noqa: E402
from orrery.change import change_model  # noqa: E402

BASE = "data/phase2/e34"
SETS, RUNS, CAND, THR = f"{BASE}/sets", f"{BASE}/runs", f"{BASE}/candidate.json", f"{BASE}/thresholds.json"
V1 = "data/phase2/e30/e30_v1.json"
OUT, OUTS, FIG = "data/findings/e34.json", "data/findings/e34s.json", "data/findings/e34_real.svg"
SEED_BSNR, SEED_BPOS, SEED_TR, SEED_TV, SEED_TT, SEED_NET2, SEED_BAUG, SEED_BOOT = (
    2026111560, 2026111561, 2026111562, 2026111563, 2026111564, 2026111565, 2026111566, 2026111567)
N_B, N_C = 10, 2
R_STAR = 1.0
METH = ("V2", "V1", "Z", "M2", "M3B", "M3BF", "M2F")
SBINS = [3, 5, 8, 12, 20, 100, np.inf]
F = R.f_floor()


# ---------------------------------------------------------------- transplants
def shifted(a, dx, dy):
    """out[y, x] = a[y - dy, x - dx], NaN where that is off the grid."""
    Sg = a.shape[0]
    out = np.full(a.shape, np.nan)
    out[max(0, dy):min(Sg, Sg + dy), max(0, dx):min(Sg, Sg + dx)] = a[max(0, -dy):min(Sg, Sg - dy), max(0, -dx):min(Sg, Sg - dx)]
    return out


def draw_shift(rng):
    while True:
        dx, dy = (int(t) for t in rng.integers(-12, 13, 2))
        if 6 <= np.hypot(dx, dy) <= 12:
            return dx, dy


def zprime(vsp, med, c, sig):
    with np.errstate(all="ignore"):
        return (vsp - med) / (c * np.sqrt(sig ** 2 + (F * med) ** 2))


def transplant_trials(st, e, dx, dy):
    """Trials of picture e: persistent sources (local maxima of med/sigma >= 3) whose shifted position has data and a
    blank reference. Returns vs', z', [(x_new, y_new, snr)]."""
    med, sig = st["med"][e], st["sig"][e]
    Sg = med.shape[0]
    vsp = shifted(st["vs"][e], dx, dy)
    zp = zprime(vsp, med, st["c"][e], sig)
    with np.errstate(all="ignore"):
        a = med / sig
    x0, y0, s0 = detect.find_candidates(a, 3.0)
    ab = np.abs(np.nan_to_num(a, nan=0.0))
    jj, ii = np.mgrid[0:Sg, 0:Sg]
    out = []
    for x, y, s in zip(x0, y0, s0):
        xn, yn = x + dx, y + dy
        if not (0 <= xn < Sg and 0 <= yn < Sg) or not np.isfinite(zp[yn, xn]):
            continue
        if ab[np.hypot(ii - xn, jj - yn) <= 2].max() >= 1.5:
            continue
        out.append((int(xn), int(yn), float(s)))
    return vsp, zp, out


def transplant_pass(st, det, rng, want_all=False):
    """One shift per picture. Per trial the stage-1 candidates of z' within R_STAR. Returns trials [(e, x, y, snr)],
    candidate cutouts X, det, z and trial index per candidate; also per-picture (vs', z', dx, dy) if want_all."""
    T, X, Dd, Z, K, pics = [], [], [], [], [], {}
    for e in range(len(det)):
        dx, dy = draw_shift(rng)
        if st["prev"][e] < 0 or not np.isfinite(st["z"][e]).any():
            continue
        vsp, zp, tr = transplant_trials(st, e, dx, dy)
        if want_all:
            pics[e] = (vsp, zp, dx, dy)
        if not tr:
            continue
        cx, cy, cz = detect.find_candidates(zp)
        for (xn, yn, s) in tr:
            k = np.flatnonzero(np.hypot(cx - xn, cy - yn) <= R_STAR)
            if len(k):
                X.append(detect.cutouts(vsp, st["med"][e], zp, st["c"][e], st["sig"][e], cx[k], cy[k]).astype(np.float32))
                Dd.append(np.full(len(k), det[e])); Z.append(cz[k]); K.append(np.full(len(k), len(T)))
            T.append((e, xn, yn, s))
    cat = lambda L, shp, dt: np.concatenate(L) if L else np.zeros(shp, dt)
    return (np.array(T, float).reshape(-1, 4), cat(X, (0, 5, 9, 9), np.float32), cat(Dd, 0, int), cat(Z, 0, float),
            cat(K, 0, int), pics)


def aug(N, rng):
    u = rng.uniform(0.5, 1.0)
    use = np.zeros(N, bool)
    use[rng.choice(N, int(round(u * N)), replace=False)] = True
    return use, rng.uniform(0.9, 1.1, N)


# ---------------------------------------------------------------- extra sets
def build_train(name):
    p = f"{SETS}/train_{name}.npz"
    if os.path.exists(p):
        return name
    fi = R.POOL.index(name)
    t0 = time.time()
    fd = R.load(name)
    N = len(fd["frames"])
    lam, det, mjd = R.meta_arrays(fd)
    rec = {k: [] for k in ("X", "det", "z", "snr", "kind")}
    old = e1.SEED0
    e1.SEED0 = SEED_BPOS
    try:
        for ps in range(N_B):                                       # (b) bright injections
            use, jit = aug(N, np.random.default_rng(SEED_BAUG + 1000 * ps + fi))
            pos = e1.draw_positions(fd, ps, fi)
            snr = np.exp(np.random.default_rng(SEED_BSNR + 1000 * ps + fi).uniform(np.log(30), np.log(3000), N))
            imgs = [e1.plant(fr, x, y, s_)[0] for fr, (x, y), s_ in zip(fd["frames"], pos, snr)]
            st = R.stage1(R.live_v(fd, imgs), lam, det, mjd, use, jit)
            cd = R.cands(st, det)
            d = np.hypot(cd["x"] - pos[cd["e"], 0], cd["y"] - pos[cd["e"], 1])
            keep = np.array([k[np.argmin(d[k])] for e in np.unique(cd["e"][d <= R.R_PLANT])
                             for k in [np.flatnonzero((cd["e"] == e) & (d <= R.R_PLANT))]], int)
            rec["X"].append(cd["X"][keep]); rec["det"].append(cd["det"][keep]); rec["z"].append(cd["z"][keep])
            rec["snr"].append(snr[cd["e"][keep]]); rec["kind"].append(np.full(len(keep), 1))
    finally:
        e1.SEED0 = old
    v = R.live_v(fd)
    for ps in range(N_C):                                           # (c) real-star transplants
        rng = np.random.default_rng(SEED_TR + 1000 * ps + fi)
        use, jit = aug(N, rng)
        st = R.stage1(v, lam, det, mjd, use, jit)
        T, X, Dd, Z, K, _ = transplant_pass(st, det, rng)
        best = {}
        for j, k in enumerate(K):
            if k not in best or Z[j] > Z[best[k]]:
                best[k] = j
        keep = np.array(sorted(best.values()), int)
        rec["X"].append(X[keep]); rec["det"].append(Dd[keep]); rec["z"].append(Z[keep])
        rec["snr"].append(T[K[keep], 3] if len(keep) else np.zeros(0)); rec["kind"].append(np.full(len(keep), 2))
    out = {k: np.concatenate(x) for k, x in rec.items()}
    out["X"] = out["X"].astype(np.float16)
    np.savez_compressed(p[:-4] + ".tmp.npz", **out)
    os.replace(p[:-4] + ".tmp.npz", p)
    print(f"train {name}: bright {int((out['kind'] == 1).sum())}, transplants {int((out['kind'] == 2).sum())}, {time.time() - t0:.0f}s", flush=True)
    return name


def build_val(name):
    p = f"{SETS}/val_{name}.npz"
    if os.path.exists(p):
        return name
    fi = R.POOL.index(name)
    fd = R.load(name)
    lam, det, mjd = R.meta_arrays(fd)
    st = R.stage1(R.live_v(fd), lam, det, mjd)
    T, X, Dd, Z, K, _ = transplant_pass(st, det, np.random.default_rng(SEED_TV + fi))
    np.savez_compressed(p[:-4] + ".tmp.npz", T=T, X=X, D=Dd, Z=Z, K=K)
    os.replace(p[:-4] + ".tmp.npz", p)
    print(f"val {name}: {len(T)} transplant trials, {len(np.unique(K))} with a candidate", flush=True)
    return name


def _job(j):
    return build_val(j[1]) if j[0] == "v" else build_train(j[1])


def build(w):
    os.makedirs(SETS, exist_ok=True)
    sp = R.split()
    with Pool(w) as pl:
        list(pl.imap_unordered(_job, [("v", n) for n in sp["val"]] + [("t", n) for n in sp["train"]]))


# ---------------------------------------------------------------- training
def val_data():
    sp = R.split()
    vd = R._val_data(sp["val"])
    V = [np.load(f"{SETS}/val_{n}.npz") for n in sp["val"]]
    off, K = 0, []
    for v in V:
        K.append(v["K"] + off); off += len(v["T"])
    vd["n0"] = len(vd["X"])
    vd["X"] = np.concatenate([vd["X"]] + [v["X"] for v in V]).astype(np.float32)
    vd["D"] = np.concatenate([vd["D"]] + [v["D"] for v in V])
    vd["tp_trial"], vd["n_tp"] = np.concatenate(K), off
    vd["tp_snr"] = np.concatenate([v["T"][:, 3] for v in V])
    return vd


_orig_metric = R.val_metric


def metric2(scores, vd, rate=0.03):
    met, rec, t = _orig_metric(scores[:vd["n0"]], {k: vd[k] for k in ("isneg", "npic", "n_trials", "trial", "level")}, rate)
    best = np.full(vd["n_tp"], -np.inf)
    np.maximum.at(best, vd["tp_trial"], scores[vd["n0"]:])
    r2 = float(np.mean(best > t))
    return (met + r2) / 2, np.r_[rec, r2], t


def train(use_c=True):
    os.makedirs(BASE, exist_ok=True)
    logf = open(f"{BASE}/train.log", "a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n"); logf.flush()
    sp = R.split()
    vd = val_data()
    Xa, Da, ya = R.train_set(sp["train"])
    E = [np.load(f"{SETS}/train_{n}.npz") for n in sp["train"]]
    Xe = np.concatenate([t["X"] for t in E]).astype(np.float32)
    De = detect.onehot(np.concatenate([t["det"] for t in E])).astype(np.float32)
    kind = np.concatenate([t["kind"] for t in E])
    na = int(ya.sum())
    rng = np.random.default_rng(SEED_NET2)
    sel = []
    for k in ((1, 2) if use_c else (1,)):
        i = np.flatnonzero(kind == k)
        sel.append(rng.permutation(i)[:na])                          # cap each extra source at the size of (a)
    sel = np.sort(np.concatenate(sel))
    X = np.concatenate([Xa, Xe[sel]]); D = np.concatenate([Da, De[sel]]); y = np.concatenate([ya, np.ones(len(sel), np.float32)])
    counts = dict(a_injected_3_30=na, b_injected_30_3000=int((kind[sel] == 1).sum()), c_transplants=int((kind[sel] == 2).sum()),
                  negatives=int((ya == 0).sum()), available_b=int((kind == 1).sum()), available_c=int((kind == 2).sum()))
    log(f"v2 training set {counts}; validation transplant trials {vd['n_tp']}")
    vD = detect.onehot(vd["D"]).astype(np.float32)
    R.val_metric, R.SEED_NET = metric2, SEED_NET2
    res = dict(counts=counts, use_c=use_c, val_transplant_trials=vd["n_tp"], seeds=[])
    best = None
    for s in range(2):
        p, met, hist = R.fit("cnn", s, X, D, y, vd, (vd["X"], vD), log)
        res["seeds"].append(dict(seed=s, best_val_metric=round(met, 4), epochs=len(hist), history=hist))
        if best is None or met > best[1]:
            best = (p, met, s)
    mdl = dict(kind="cnn", weights=R.to_json_weights(best[0]), seed=best[2])
    v1 = json.load(open(R.MODEL if not os.path.exists(V1) else V1))["model"]
    for name, m in (("v2", mdl), ("v1", v1)):
        met, rec, t = metric2(detect.forward(m, vd["X"], vD), vd)
        res[f"val_{name}"] = dict(metric=round(met, 4), recall_injected=[round(float(x), 4) for x in rec[:5]],
                                  recall_transplants=round(float(rec[5]), 4), threshold=t)
        log(f"{name}: {res[f'val_' + name]}")
    mdl["val"] = res["val_v2"]
    res["model"] = mdl
    json.dump(res, open(CAND, "w"), indent=1)


# ---------------------------------------------------------------- calibration + model file
def masked_stage(fd, lam, det, mjd):
    rm = change_model(e1.frame_arrays(fd)[0], lam, det, mjd, f=F)
    med, z, c = detect.m3b(rm["vs"], rm["sigma"], det, F)
    return rm, med, z, c


def m2f_field(name):
    p = f"{RUNS}/calibm2f_{name}.npz"
    if os.path.exists(p):
        return name
    fd = R.load(name)
    lam, det, mjd = R.meta_arrays(fd)
    rm = change_model(e1.frame_arrays(fd)[0], lam, det, mjd, f=F)
    V, Dl, n = [], [], 0
    for m in e1.blink_d1(rm["vs"], rm["sigma"], rm["prev_det"], F):
        if m is None or not np.isfinite(m).any():
            continue
        a, d = detect.comp_events(e1.despike(m), R.KMIN)
        V.append(a); Dl.append(d); n += 1
    np.savez_compressed(p[:-4] + ".tmp.npz", ev=np.concatenate(V), evd=np.concatenate(Dl), n=n)
    os.replace(p[:-4] + ".tmp.npz", p)
    return name


def calib(w):
    os.makedirs(RUNS, exist_ok=True)
    names = [x["field"] for x in R.group("calib")]
    with Pool(w) as pl:
        list(pl.imap_unordered(m2f_field, names))
    if not os.path.exists(V1):
        shutil.copy(R.MODEL, V1)
    v1 = json.load(open(V1))
    cand = json.load(open(CAND))
    mdl = {k: cand["model"][k] for k in ("kind", "weights", "seed", "val")}
    recs = [dict(np.load(f"{R.RUNS}/calib_{n}.npz")) for n in names]
    for r in recs:
        r["fa_CNN"] = detect.forward(mdl, r["cand_X"], detect.onehot(r["cand_det"]))
    t = {op: R._thr_method(recs, "CNN", q) for op, q in zip(R.OPS, R.RATES)}
    table = R.rate_table(recs, "CNN")
    M = [np.load(f"{RUNS}/calibm2f_{n}.npz") for n in names]
    m2f = {op: detect.threshold_events(np.concatenate([m["ev"] for m in M]), np.concatenate([m["evd"] for m in M]),
                                       sum(int(m["n"]) for m in M), q, R.KMIN) for op, q in zip(R.OPS, R.RATES)}
    blocks = dict(model=dict(mdl, id="e30-cnn-v2"), ablation={k: v for k, v in v1["ablation"].items() if k not in ("thresholds", "rate_table")})
    wsha = hashlib.sha256(json.dumps(blocks, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    j = dict(v1)
    j.update(id=f"e30-cnn-v2-{wsha[:8]}", weights_sha256=wsha, use="model", thresholds=t, rate_table=table,
             protocol_v2="docs/protocols/E34.md", model=dict(blocks["model"], thresholds=t, rate_table=table),
             previous=dict(id=v1["id"], weights_sha256=v1["weights_sha256"], file=V1))
    json.dump(j, open(R.MODEL, "w"), separators=(",", ":"))
    e30thr = json.load(open(R.THR))["thresholds"]
    thr = dict(V2=t, V1=v1["thresholds"], Z=e30thr["Z"]["t"], M2=e30thr["M2"]["t"], M3B=e30thr["M3B"]["t"],
               M3BF=e30thr["M3BF"]["t"], M2F=m2f)
    json.dump(dict(thresholds=thr, model_sha256=R.sha(R.MODEL), weights_sha256=wsha, id=j["id"],
                   frozen_local=time.strftime("%Y-%m-%dT%H:%M:%S%z")), open(THR, "w"), indent=1)
    print(json.dumps(thr, indent=1), "\nweights", wsha, "file", R.sha(R.MODEL), flush=True)


def models():
    return dict(V2=json.load(open(R.MODEL))["model"], V1=json.load(open(V1))["model"])


def thr3():
    return {m: v["r3"] for m, v in json.load(open(THR))["thresholds"].items()}


# ---------------------------------------------------------------- B + C: real stars and false candidates on test fields
def stars_field(name):
    p = f"{RUNS}/stars_{name}.npz"
    if os.path.exists(p):
        return name
    t0 = time.time()
    fi = [x["field"] for x in R.group("test")].index(name)
    mm = models()
    fd = R.load(name)
    lam, det, mjd = R.meta_arrays(fd)
    st = R.stage1(R.live_v(fd), lam, det, mjd)
    cd = R.cands(st, det)
    fa = {k: detect.forward(m, cd["X"], detect.onehot(cd["det"])) for k, m in mm.items()}
    npic = int(sum(np.isfinite(z).any() for z in st["z"]))
    T, X, Dd, Z, K, pics = transplant_pass(st, det, np.random.default_rng(SEED_TT + fi), want_all=True)
    sc = {k: detect.forward(m, X, detect.onehot(Dd)) if len(Dd) else np.zeros(0) for k, m in mm.items()}
    sc["Z"] = Z
    rm, medm, zm, cm = masked_stage(fd, lam, det, mjd)
    r = st["r"]
    rows = np.full((len(T), len(METH)), -np.inf)
    for s in ("V2", "V1", "Z"):
        np.maximum.at(rows[:, METH.index(s)], K, sc[s])
    for i, (e, xn, yn, snr) in enumerate(T):
        e, xn, yn = int(e), int(xn), int(yn)
        vsp, zp, dx, dy = pics[e]
        q, qm = r["prev_det"][e], rm["prev_det"][e]
        with np.errstate(all="ignore"):
            m2 = (vsp - r["vs"][q]) / np.sqrt(r["sigma"][e] ** 2 + r["sigma"][q] ** 2 + (F * r["vs"][q]) ** 2)
            vspm = shifted(rm["vs"][e], dx, dy)
            m3bf = e1.despike(zprime(vspm, medm[e], cm[e], rm["sigma"][e]))
            m2f = e1.despike((vspm - rm["vs"][qm]) / np.sqrt(rm["sigma"][e] ** 2 + rm["sigma"][qm] ** 2 + (F * rm["vs"][qm]) ** 2)) \
                if qm >= 0 else np.full(vsp.shape, np.nan)
        for mname, mp in (("M2", m2), ("M3B", zp), ("M3BF", m3bf), ("M2F", m2f)):
            rows[i, METH.index(mname)] = detect.comp_kmax(mp, xn, yn, R_STAR)
    np.savez_compressed(p[:-4] + ".tmp.npz", T=T, rows=rows, fa_V2=fa["V2"], fa_V1=fa["V1"], npic=npic)
    os.replace(p[:-4] + ".tmp.npz", p)
    print(f"stars {name}: {len(T)} trials, {time.time() - t0:.0f}s", flush=True)
    return name


# ---------------------------------------------------------------- A: real asteroids
def load_any(name):
    import run_e1
    if name.startswith("e34"):
        run_e1.ROOT = f"{BASE}/efields"
        return run_e1.load_field(name)
    return R.load(name)


def ast_field(name):
    import build_field as bf
    import run_e27
    p = f"{RUNS}/ast_{name}.json"
    if os.path.exists(p):
        return name
    mm = models()
    fd = load_any(name)
    lam, det, mjd = R.meta_arrays(fd)
    st = R.stage1(R.live_v(fd), lam, det, mjd)
    cd = R.cands(st, det)
    sc = {k: detect.forward(m, cd["X"], detect.onehot(cd["det"])) if len(cd["e"]) else np.zeros(0) for k, m in mm.items()}
    sc["Z"] = cd["z"]
    r = st["r"]
    rm = change_model(e1.frame_arrays(fd)[0], lam, det, mjd, f=F)
    maps = dict(M2=e1.blink_d1(r["vs"], r["sigma"], r["prev_det"], F), M3B=list(st["z"]),
                M3BF=[e1.despike(m) for m in run_e27.m3b_maps(rm["vs"], rm["sigma"], det, F)],
                M2F=[None if m is None else e1.despike(m) for m in e1.blink_d1(rm["vs"], rm["sigma"], rm["prev_det"], F)])
    with run_e27.outlier_kept():                                # E27's asteroid-secondary mode: OUTLIER not masked, despike
        mk = run_e27.maps_for(e1.frame_arrays(fd)[0], lam, det, mjd, F)
    maps.update(M3BK=mk["M3b"], M2K=mk["M2"])
    ast, nq, source = sightings(name, fd, mjd)
    rows = []
    for e in range(len(mjd)):
        for o in ast[e]:
            if not (o["vmag"] < R.VMAX and R.EDGE <= o["x"] <= R.S - 1 - R.EDGE and R.EDGE <= o["y"] <= R.S - 1 - R.EDGE):
                continue
            if st["prev"][e] < 0:
                continue
            k = np.flatnonzero((cd["e"] == e) & (np.hypot(cd["x"] - o["x"], cd["y"] - o["y"]) <= R.R_AST))
            row = dict(frame=e, name=o["name"], vmag=o["vmag"], det=int(det[e]), stage1=bool(len(k)),
                       **{s: float(sc[s][k].max()) if len(k) else None for s in ("V2", "V1", "Z")},
                       **{m: detect.comp_kmax(maps[m][e], o["x"], o["y"], R.R_AST) if maps[m][e] is not None else None for m in maps})
            rows.append(row)
    json.dump(dict(field=name, n_pictures=len(mjd), skybot_queries=nq, sightings_from=source, rows=rows), open(p, "w"))
    print(name, len(rows), "sightings", source, flush=True)
    return name


def sightings(name, fd, mjd, budget=30):
    """SkyBoT (bf.known_asteroids) as E30; if it fails or takes longer than `budget` s, the selecting asteroid alone
    from Horizons as seen from SPHEREx (scripts/ephem.py), logged per field."""
    import build_field as bf
    from concurrent.futures import ThreadPoolExecutor
    ex = ThreadPoolExecutor(1)
    try:
        ast, nq = ex.submit(bf.known_asteroids, mjd, R.S, fd["ra0"], fd["dec0"]).result(timeout=budget)
        return ast, nq, "skybot"
    except Exception as err:
        print(name, "SkyBoT fallback:", repr(err)[:80], flush=True)
    finally:
        ex.shutdown(wait=False)
    import ephem
    from orrery.change import world_to_grid
    sel = next(x for x in json.load(open(f"{BASE}/fields.json"))["fields"] if x["field"] == name)
    df = ephem.fetch_epochs(str(sel["selected_by"]), [float(m) for m in mjd])
    out = [[] for _ in mjd]
    for e, (_, r) in enumerate(df.iterrows()):
        x, y = world_to_grid(float(r["RA"]), float(r["DEC"]), R.S, fd["ra0"], fd["dec0"])
        out[e].append(dict(name=str(sel["selected_by"]), vmag=float(r["V"]), x=float(x), y=float(y)))
    return out, 0, "horizons_selected_only"


def e34_fields():
    import run_e1
    run_e1.ROOT = f"{BASE}/efields"
    return [x["field"] for x in json.load(open(f"{BASE}/fields.json"))["fields"] if run_e1.ready(x["field"])]


def run(what, w):
    assert "## Part 2" in open("docs/protocols/E34.md").read(), "Part 2 not frozen"
    os.makedirs(RUNS, exist_ok=True)
    if what == "stars":
        jobs, fn = [x["field"] for x in R.group("test")], stars_field
    elif what == "ast":
        jobs, fn = e34_fields(), ast_field
    else:
        jobs, fn = [x["field"] for x in R.group("ecl")], ast_field
    with Pool(w) as pl:
        list(pl.imap_unordered(fn, jobs))


# ---------------------------------------------------------------- summary
def _ast(rows, t):
    vb = dict(V16=lambda v: v <= 16, V16_17p5=lambda v: 16 < v <= 17.5, Vgt17p5=lambda v: v > 17.5, V17p5=lambda v: v <= 17.5)
    out = {}
    for b, fn in vb.items():
        rr = [x for x in rows if fn(x["vmag"])]
        out[b] = dict(n=len(rr), n_objects=len({(x["field"], x["name"]) for x in rr}), stage1=int(sum(x["stage1"] for x in rr)))
        for m in AST_METH:
            h = int(sum(x.get(m) is not None and x[m] > t[m] for x in rr))
            objs = len({(x["field"], x["name"]) for x in rr if x.get(m) is not None and x[m] > t[m]})
            out[b][m] = dict(hits=h, recall=round(h / len(rr), 4) if rr else None, wilson95=R.wilson(h, len(rr)), objects=objs)
    return out


AST_METH = METH + ("M3BK", "M2K")


def summary():
    t = thr3()
    k27 = json.load(open("data/findings/e27.json"))["thresholds"]["kept"]
    t.update(M3BK=k27["M3b"]["k"]["r3"], M2K=k27["M2"]["k"]["r3"])
    tests = [x["field"] for x in R.group("test")]
    Sx = [dict(np.load(f"{RUNS}/stars_{n}.npz")) for n in tests if os.path.exists(f"{RUNS}/stars_{n}.npz")]
    nf = len(Sx)
    idx = np.random.default_rng(SEED_BOOT).integers(0, nf, size=(2000, nf))
    nb = len(SBINS) - 1
    Nn = np.array([[np.sum((s["T"][:, 3] >= a) & (s["T"][:, 3] < b)) for a, b in zip(SBINS[:-1], SBINS[1:])] for s in Sx], float)
    stars = dict(bins=[[a, None if np.isinf(b) else b] for a, b in zip(SBINS[:-1], SBINS[1:])], n=Nn.sum(0).astype(int).tolist())
    for m in METH:
        Dm = np.array([[np.sum((s["T"][:, 3] >= a) & (s["T"][:, 3] < b) & (s["rows"][:, METH.index(m)] > t[m]))
                        for a, b in zip(SBINS[:-1], SBINS[1:])] for s in Sx], float)
        with np.errstate(all="ignore"):
            rb = Dm[idx].sum(1) / Nn[idx].sum(1)
            stars[m] = dict(recall=[round(float(x), 4) for x in Dm.sum(0) / Nn.sum(0)],
                            ci95=[[round(float(np.nanpercentile(rb[:, k], 2.5)), 4), round(float(np.nanpercentile(rb[:, k], 97.5)), 4)] for k in range(nb)],
                            all=round(float(Dm.sum() / Nn.sum()), 4))
    fa = {}
    for m in ("V2", "V1"):
        c = np.array([np.sum(s["fa_" + m] > t[m]) for s in Sx], float)
        n = np.array([int(s["npic"]) for s in Sx], float)
        b = c[idx].sum(1) / n[idx].sum(1)
        fa[m] = dict(rate=round(float(c.sum() / n.sum()), 5), ci95=[round(float(np.percentile(b, 2.5)), 5), round(float(np.percentile(b, 97.5)), 5)],
                     false_candidates=int(c.sum()), n_pictures=int(n.sum()))

    def rows_of(names):
        out = []
        for n in names:
            if os.path.exists(f"{RUNS}/ast_{n}.json"):
                out += [dict(x, field=n) for x in json.load(open(f"{RUNS}/ast_{n}.json"))["rows"]]
        return out
    fresh = [n for n in e34_fields() if os.path.exists(f"{RUNS}/ast_{n}.json")] if os.path.exists(f"{BASE}/fields.json") else []
    old = [x["field"] for x in R.group("ecl")]
    A, Aold = _ast(rows_of(fresh), t), _ast(rows_of(old), t)
    a = A["V17p5"]
    lo = a["V2"]["wilson95"][0]
    ok = bool(lo is not None and lo > 0.5 and fa["V2"]["rate"] <= 0.045)
    cand = json.load(open(CAND))
    thr = json.load(open(THR))
    out = dict(experiment="E34", protocol="docs/protocols/E34.md", model_id=thr["id"], model_sha256=thr["model_sha256"],
               weights_sha256=thr["weights_sha256"], thresholds=thr["thresholds"], operating_point="0.03 false candidates per picture",
               methods=dict(V2="CNN v2 (shipped)", V1="CNN v1 (E30)", Z="M3b z of candidates, live", M2="M2 live", M3B="M3b live",
                            M3BF="M3b with flags + despike", M2F="M2 with flags + despike",
                            M3BK="E27's M3b exactly (OUTLIER not masked, despike, E27's k at 0.03; asteroids only)",
                            M2K="E27's M2 exactly (OUTLIER not masked, despike, E27's k at 0.03; asteroids only)"),
               e27_thresholds_r3=dict(M3BK=t["M3BK"], M2K=t["M2K"]),
               training=dict(counts=cand["counts"], use_transplants=cand["use_c"], val_v2=cand["val_v2"], val_v1=cand["val_v1"],
                             seeds=[dict(seed=s["seed"], best_val_metric=s["best_val_metric"], epochs=s["epochs"]) for s in cand["seeds"]]),
               asteroids_fresh=dict(fields=fresh, by_V=A), asteroids_previously_seen_post_hoc=dict(fields=old, by_V=Aold),
               real_stars=dict(n_test_fields=nf, rule="transplant trials on the 20 E30 test fields, detection within 1 px", **stars),
               false_candidates_test=fa,
               criterion="v2 works on real moving objects iff Wilson lower bound of recall on V <= 17.5 sightings > 0.5 and realised "
                         "false-candidate rate on the 20 test fields <= 0.045",
               verdict="works on real moving objects" if ok else ("not shown" if a["n"] else "no eligible sightings"),
               criterion_met=ok, generated_local=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               asteroids_by_predicted_snr_descriptive=dict(fresh=descriptive(fresh, t), previously_seen=descriptive(old, t)))
    json.dump(out, open(OUT, "w"), indent=1, default=float)
    s = dict(source=OUT, verdict=out["verdict"], criterion_met=ok, model_id=thr["id"], fa_v2_r3=fa["V2"]["rate"], fa_v2_r3_lo=fa["V2"]["ci95"][0],
             fa_v2_r3_hi=fa["V2"]["ci95"][1], fa_v1_r3=fa["V1"]["rate"], n_ast_fields=len(fresh), star_n_trials=int(Nn.sum()))
    nm = dict(V2="v2", V1="v1")
    for b in A:
        s[f"ast_n_{b}"], s[f"ast_objects_{b}"] = A[b]["n"], A[b]["n_objects"]
        s[f"astold_n_{b}"] = Aold[b]["n"]
        for m in AST_METH:
            k = nm.get(m, m)
            s[f"ast_{k}_hits_{b}"] = A[b][m]["hits"]
            s[f"ast_{k}_lo_{b}"], s[f"ast_{k}_hi_{b}"] = A[b][m]["wilson95"]
            s[f"astold_{k}_hits_{b}"] = Aold[b][m]["hits"]
    s["ast_v2_lo"], s["ast_v2_hi"] = a["V2"]["wilson95"]
    lab = ["3_5", "5_8", "8_12", "12_20", "20_100", "gt100"]
    for m in METH:
        for k, l in enumerate(lab):
            s[f"star_recall_{nm.get(m, m)}_s{l}"] = stars[m]["recall"][k]
        s[f"star_recall_{nm.get(m, m)}_all"] = stars[m]["all"]
    for k, l in enumerate(lab):
        s[f"star_n_s{l}"] = stars["n"][k]
    ds = out["asteroids_by_predicted_snr_descriptive"]["fresh"]["bins"]
    for lab, d in ds.items():
        s[f"astsnr_n_{lab}"], s[f"astsnr_any_{lab}"] = d["n"], d["any_stage1"]
        for m in ("V2", "V1", "M3BF", "M2F", "M2", "Z", "M3BK", "M2K"):
            s[f"astsnr_{nm.get(m, m)}_hits_{lab}"] = d[m]["hits"]
            s[f"astsnr_{nm.get(m, m)}_lo_{lab}"], s[f"astsnr_{nm.get(m, m)}_hi_{lab}"] = d[m]["wilson95"]
    assert all(not any(c in k for c in ".- ") for k in s)
    json.dump(s, open(OUTS, "w"), indent=1)
    figure(out)
    print(json.dumps(dict(verdict=out["verdict"], fa=fa, ast={b: {m: (A[b][m]["hits"], A[b]["n"]) for m in AST_METH} for b in A},
                          old={b: {m: (Aold[b][m]["hits"], Aold[b]["n"]) for m in AST_METH} for b in Aold},
                          stars={m: stars[m]["recall"] for m in METH}, n=stars["n"])))


def figure(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = dict(V2="#D55E00", V1="#CC79A7", Z="#E69F00", M3B="#009E73", M2="#56B4E9", M3BF="#7f7f7f", M2F="#000000")
    sty = dict(V2="-", V1="-.", Z=":", M3B="--", M2="--", M3BF=(0, (1, 1)), M2F=(0, (3, 1, 1, 1)))
    lab = dict(V2="CNN v2", V1="CNN v1", Z="M3b z, live", M3B="M3b, live", M2="M2, live", M3BF="M3b with flags", M2F="M2 with flags")
    st = out["real_stars"]
    x = np.arange(len(st["bins"]))
    fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.9), gridspec_kw=dict(width_ratios=[2, 1.2]))
    for m in METH:
        lo, hi = np.array(st[m]["ci95"], float).T
        axs[0].fill_between(x, lo, hi, color=col[m], alpha=0.12, lw=0)
        axs[0].plot(x, st[m]["recall"], linestyle=sty[m], marker="o", ms=3.5, lw=1.5, color=col[m], label=lab[m])
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(["3–5", "5–8", "8–12", "12–20", "20–100", "> 100"])
    axs[0].set_xlabel("real star S/N in one picture")
    axs[0].set_ylabel("recall at 0.03 false candidates per picture")
    axs[0].set_ylim(0, 1.02)
    axs[0].legend(fontsize=7.5, frameon=False, loc="upper left")
    A = out["asteroids_fresh"]["by_V"]
    bl = [("V16", "V ≤ 16"), ("V16_17p5", "16–17.5"), ("Vgt17p5", "> 17.5")]
    for j, m in enumerate(METH):
        for i, (b, _) in enumerate(bl):
            r = A[b][m]
            if not A[b]["n"]:
                continue
            xx = i + (j - 3) * 0.1
            axs[1].errorbar([xx], [r["recall"]], yerr=[[r["recall"] - r["wilson95"][0]], [r["wilson95"][1] - r["recall"]]],
                            fmt="o", ms=4, color=col[m], lw=1.2)
    axs[1].set_xticks(range(3))
    axs[1].set_xticklabels([f"{l}\n(n = {A[b]['n']})" for b, l in bl])
    axs[1].set_xlabel("real asteroid sightings, predicted V")
    axs[1].set_ylim(-0.02, 1.02)
    for ax in axs:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG)
    fig.savefig(FIG[:-4] + ".png", dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------- descriptive (not pre-declared): predicted S/N
def snr_field(name):
    """Predicted second-brightest-pixel S/N per sighting (build_field.predicted_snr2, as E3), wl = the picture's
    wavelength at the source, denom = the live change model's per-pixel noise sqrt(sigma^2 + (f model)^2); PSF FWHM
    7.5 arcsec for every picture (E3's fallback; the stamps carry no PSF_FWHM)."""
    import build_field as bf
    p, q = f"{RUNS}/ast_{name}.json", f"{RUNS}/snr_{name}.json"
    if os.path.exists(q) or not os.path.exists(p):
        return name
    rows = json.load(open(p))["rows"]
    fd = load_any(name)
    lam, det, mjd = R.meta_arrays(fd)
    r = change_model(R.live_v(fd), lam, det, mjd, f=F)
    ast, _, _ = sightings(name, fd, mjd)                         # same list, same order as the ast_ rows
    snr = []
    for e in range(len(mjd)):
        for o in ast[e]:
            if not (o["vmag"] < R.VMAX and R.EDGE <= o["x"] <= R.S - 1 - R.EDGE and R.EDGE <= o["y"] <= R.S - 1 - R.EDGE):
                continue
            if r["prev_det"][e] < 0:
                continue
            ix, iy = int(np.clip(round(o["x"]), 0, R.S - 1)), int(np.clip(round(o["y"]), 0, R.S - 1))
            wl = float(lam[e][iy, ix]) if np.isfinite(lam[e][iy, ix]) else float(np.nanmean(lam[e]))
            denom = np.sqrt(r["sigma"][e] ** 2 + (F * r["model"][e]) ** 2)
            s2, _ = bf.predicted_snr2(o["vmag"], wl, o["x"], o["y"], 7.5, denom)
            snr.append(dict(frame=e, name=o["name"], snr2_pred=float(s2)))
    assert len(snr) == len(rows) and all(a["frame"] == b["frame"] and str(a["name"]) == str(b["name"]) for a, b in zip(snr, rows))
    json.dump(snr, open(q, "w"))
    return name


def snr_all(w):
    names = [x["field"] for x in R.group("ecl")] + (e34_fields() if os.path.exists(f"{BASE}/fields.json") else [])
    with Pool(w) as pl:
        list(pl.imap_unordered(snr_field, names))


def descriptive(names, t):
    rows = []
    for n in names:
        p, q = f"{RUNS}/ast_{n}.json", f"{RUNS}/snr_{n}.json"
        if os.path.exists(p) and os.path.exists(q):
            rows += [dict(a, field=n, snr2_pred=b["snr2_pred"]) for a, b in zip(json.load(open(p))["rows"], json.load(open(q)))]
    bins = [(0, 3, "lt3"), (3, 5, "3_5"), (5, 10, "5_10"), (10, np.inf, "gt10")]
    out = dict(rule="descriptive, not pre-declared: predicted single-picture S/N of the second-brightest pixel "
                    "(build_field.predicted_snr2, FWHM 7.5 arcsec, denom = live change-model noise)", bins={})
    for a, b, lab in bins:
        rr = [x for x in rows if a <= x["snr2_pred"] < b]
        d = dict(n=len(rr), any_stage1=int(sum(x["stage1"] for x in rr)))
        for m in ("V2", "V1", "M3BF", "M2F", "M2", "Z", "M3BK", "M2K"):
            h = int(sum(x.get(m) is not None and x[m] > t[m] for x in rr))
            d[m] = dict(hits=h, wilson95=R.wilson(h, len(rr)))
        out["bins"][lab] = d
    objs = {}
    for x in rows:
        o = objs.setdefault((x["field"], str(x["name"])), dict(field=x["field"], object=str(x["name"]), V=[], n=0, V2=0, V1=0, M3BF=0, snr=[]))
        o["V"].append(x["vmag"]); o["snr"].append(x["snr2_pred"]); o["n"] += 1
        for m in ("V2", "V1", "M3BF"):
            o[m] += int(x.get(m) is not None and x[m] > t[m])
    out["objects"] = [dict(field=o["field"], object=o["object"], V=round(float(np.median(o["V"])), 2), n=o["n"],
                           snr2_pred_median=round(float(np.median(o["snr"])), 1), v2_hits=o["V2"], v1_hits=o["V1"],
                           M3BF_hits=o["M3BF"]) for o in objs.values()]
    return out


if __name__ == "__main__":
    a = sys.argv[1]
    w = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 1
    {"build": lambda: build(w), "train": lambda: train("--no-c" not in sys.argv), "calib": lambda: calib(w),
     "stars": lambda: run("stars", w), "ast": lambda: run("ast", w), "astold": lambda: run("astold", w),
     "summary": summary, "snr": lambda: snr_all(w)}[a]()
