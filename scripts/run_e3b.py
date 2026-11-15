"""E3b - recall with the archive's OUTLIER flag left in (docs/PROTOCOLS.md, E3b; frozen 2026-11-15 10:45 PST).

Switch: `outlier_kept()` sets orrery.e1.MASK_BITS = orrery.data.BAD_BITS for the duration of a block, so every
method M1-M5 is built by the unchanged E1/E3 code path (orrery.e1.resample -> method_maps -> despike -> components)
with OUTLIER (bit 19) NOT masked and hard-bad bits still masked. orrery/e1.py is not edited (its default, used by
E1/E3, stays hard-bad | OUTLIER).

Steps (each resumable; OMP_NUM_THREADS=1):
  calib   thresholds on `calib` under D1 rules, 1 false component / 100 frames -> data/findings/e3b_thresholds.json
  cache_existing  raw cutouts (data/efields format, scripts/e2_cache.build) of the 6 existing fields as e3x_<name>
                  + SkyBoT truth prefetch (build_field._skybot_query cache, <= 3 threads)
  cache_blind     (only after the thresholds commit) draw the 10 new ecliptic fields, default_rng(20261117),
                  exactly as E3 -> data/findings/e3b_fields.json; cache them + SkyBoT prefetch
  test    recall + secondaries -> data/findings/e3b.json, data/findings/e3b_misses.png (refuses before the
          thresholds file is committed in git)
"""
import contextlib, hashlib, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from orrery import e1  # noqa: E402
from orrery.data import BAD_BITS  # noqa: E402
import run_e1  # noqa: E402

THR = "data/findings/e3b_thresholds.json"
FIELDS = "data/findings/e3b_fields.json"
OUT = "data/findings/e3b.json"
S, EDGE, VMAX, R_MATCH = 48, 3, 19.0, 2.0
EXISTING = ["barnard", "wise0855", "epsindib", "rwaur", "rdra", "ecliptic_control"]


@contextlib.contextmanager
def outlier_kept():
    old = e1.MASK_BITS
    e1.MASK_BITS = BAD_BITS
    try:
        yield
    finally:
        e1.MASK_BITS = old


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def maps_for(name, f):
    """All E1 method maps (D1 rules) for a data/efields field with OUTLIER left in."""
    with outlier_kept():
        fd = run_e1.load_field(name)
        v, lam, det, mjd = e1.frame_arrays(fd)
        maps, r = e1.method_maps(v, lam, det, mjd, f)
    return fd, v, lam, det, mjd, maps, r


# ---------------------------------------------------------------- calib
def calib():
    if os.path.exists(THR):
        sys.exit(f"{THR} exists (frozen); refusing to overwrite")
    f = run_e1.f_frozen()
    fd, v, lam, det, mjd, maps, r = maps_for("calib", f)
    out = dict(protocol="docs/PROTOCOLS.md E3b threshold calibration (OUTLIER not masked)", field="calib",
               ra=fd["ra0"], dec=fd["dec0"], f=f, n_frames=len(fd["frames"]),
               frames_dropped_centre_off_detector=len(fd["dropped"]), cache="data/efields/calib",
               index="data/index/frames_dr1.parquet", target_rate_per_frame=e1.TARGET_RATE,
               mask_bits="orrery.data.BAD_BITS only (OUTLIER, bit 19, not masked); switch scripts/run_e3b.py:outlier_kept",
               rule="smallest k on the 0.05 grid with (positive change components, 8-connected, >= 2 px, after the D1 "
                    "despike) / (frames with a map) <= 0.01; nothing planted",
               deviation="D1 rules as in data/findings/e1_thresholds.json", m5_name=getattr(e1._m5(), "M5_NAME", None),
               methods={}, frozen_local=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               code_sha256={p: sha(p) for p in ("src/orrery/e1.py", "src/orrery/change.py", "src/orrery/e1_baselines.py",
                                                "scripts/run_e3b.py", "scripts/run_e1.py")})
    for m, ms in maps.items():
        rate, n = e1.false_rate_curve(ms)
        k = e1.pick_k(rate)
        i = int(np.searchsorted(e1.K_GRID, k - 1e-9)) if k is not None else None
        out["methods"][m] = dict(k=k, n_maps=n, rate_at_k=float(rate[i]) if i is not None else None,
                                 curve=[[float(e1.K_GRID[j]), round(float(rate[j]), 5)] for j in range(len(rate))
                                        if rate[j] > 0 and (j == 0 or rate[j] != rate[j - 1])])
        print(m, k, n, flush=True)
    json.dump(out, open(THR, "w"), indent=1)
    print(f"wrote {THR}")


def committed():
    r = subprocess.run(["git", "log", "--oneline", "-1", "--", THR], capture_output=True, text=True)
    return r.stdout.strip()


# ---------------------------------------------------------------- draw + cache
def draw():
    from astropy.coordinates import SkyCoord, BarycentricMeanEcliptic
    import astropy.units as u
    if os.path.exists(FIELDS):
        return json.load(open(FIELDS))
    rng = np.random.default_rng(20261117)
    out = []
    for i in range(10):
        lon, lat = rng.uniform(0, 360), rng.uniform(-2, 2)
        c = SkyCoord(lon * u.deg, lat * u.deg, frame=BarycentricMeanEcliptic()).icrs
        out.append(dict(field=f"e3b_ecl{i:02d}", ecl_lon=lon, ecl_lat=lat, ra=float(c.ra.deg), dec=float(c.dec.deg)))
    rec = dict(seed=20261117, frame="BarycentricMeanEcliptic (J2000)", rule="per field: lon = U(0, 360) then "
               "lat = U(-2, 2) deg, as E3 (scripts/e3_fields.py)", size_px=48, frame_cap=250, fields=out)
    json.dump(rec, open(FIELDS, "w"), indent=1)
    return rec


def existing_centres():
    return {f"e3x_{n}": (json.load(open(f"data/findings/f1_{n}.json"))["ra"],
                         json.load(open(f"data/findings/f1_{n}.json"))["dec"]) for n in EXISTING}


def cache(which):
    """which = "existing" (the 6 existing fields, not blind; allowed before the thresholds commit) or "blind"."""
    import e2_cache
    import build_field as bf
    df, src = e2_cache.load_index()
    if which == "blind":
        if not committed():
            sys.exit(f"{THR} is not committed; the blind fields are drawn only after the commit")
        jobs = [(x["field"], x["ra"], x["dec"], dict(role="E3b blind ecliptic test field", ecl_lon=x["ecl_lon"],
                                                      ecl_lat=x["ecl_lat"])) for x in draw()["fields"]]
    else:
        jobs = [(n, ra, dec, dict(role="E3b secondary: existing field (not blind)"))
                for n, (ra, dec) in existing_centres().items()]
    with ThreadPoolExecutor(e2_cache.WORKERS) as ex:
        for name, ra, dec, extra in jobs:
            e2_cache.build(name, ra, dec, df, ex, extra)
    # SkyBoT truth prefetch on each field's saved frames (grouping identical to build_field.known_asteroids)
    q = []
    for name, ra, dec, _ in jobs:
        mjd = np.sort([x["mjd"] for x in json.load(open(f"data/efields/{name}/field.json"))["frame_list"]])
        G, start = [], None
        for t in mjd:
            if start is None or t - start > 0.05:
                G.append([]); start = t
            G[-1].append(t)
        q += [(float(np.mean(g)) + 2400000.5, ra, dec) for g in G]
    print(len(q), "SkyBoT queries", flush=True)

    def go(j):
        try:
            return len(bf._skybot_query(*j))
        except Exception as e:
            print("FAIL", j, repr(e), flush=True)
    with ThreadPoolExecutor(3) as ex:
        for i, _ in enumerate(ex.map(go, q)):
            if (i + 1) % 50 == 0:
                print(i + 1, flush=True)
    print("cache done", flush=True)


# ---------------------------------------------------------------- test
def wilson(k, n, z):
    if n == 0:
        return [None, None]
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(max(0.0, float(c - h)), 4), round(min(1.0, float(c + h)), 4)]


def hit(m, k, x, y):
    lab = e1.components(m, k)
    jj, ii = np.mgrid[0:S, 0:S]
    return bool((lab[np.hypot(ii - x, jj - y) <= R_MATCH] > 0).any())


def field_rows(name, k, f, cache_maps=None):
    import build_field as bf
    fd, v, lam, det, mjd, maps, r = maps_for(name, f)
    ast, nq = bf.known_asteroids(mjd, S, fd["ra0"], fd["dec0"])
    methods = list(maps)
    rows, n_no_m2 = [], 0
    for e in range(len(mjd)):
        truth = [o for o in ast[e] if o["vmag"] < VMAX and EDGE <= o["x"] <= S - 1 - EDGE and EDGE <= o["y"] <= S - 1 - EDGE]
        if not truth:
            continue
        if maps["M2"][e] is None:
            n_no_m2 += len(truth)
            continue
        for o in truth:
            ix, iy = int(round(o["x"])), int(round(o["y"]))
            row = dict(field=name, frame=e, obs_id=fd["frames"][e]["obs_id"], detector=int(det[e]),
                       mjd=round(float(mjd[e]), 6), name=o["name"], vmag=o["vmag"], orbit_class=o["cls"],
                       x=round(o["x"], 2), y=round(o["y"], 2),
                       wl_um=round(float(lam[e][iy, ix]), 3) if np.isfinite(lam[e][iy, ix]) else None,
                       centre_has_data=bool(np.isfinite(v[e][iy, ix])))
            for m in methods:
                row[m] = None if maps[m][e] is None else hit(maps[m][e], k[m], o["x"], o["y"])
            rows.append(row)
    rec = dict(field=name, n_frames=len(mjd), frames_dropped_centre_off_detector=len(fd["dropped"]),
               skybot_queries=nq, n_pairs=len(rows), n_pairs_excluded_no_m2_map=n_no_m2,
               n_asteroids=len({q["name"] for q in rows}), **{f"{m}_hits": sum(bool(q[m]) for q in rows) for m in methods})
    if cache_maps is not None:
        cache_maps[name] = (fd, v, r)
    return rec, rows, methods, maps


def summary(rows, methods):
    out = dict(n_pairs=len(rows))
    for m in methods:
        sub = [q for q in rows if q[m] is not None]
        kk = sum(q[m] for q in sub)
        out[m] = dict(hits=kk, n=len(sub), recall=round(kk / len(sub), 4) if sub else None,
                      wilson68=wilson(kk, len(sub), 1.0), wilson95=wilson(kk, len(sub), 1.96))
    ast = {}
    for q in rows:
        a = ast.setdefault((q["field"], q["name"]), {m: False for m in methods})
        for m in methods:
            a[m] |= bool(q[m])
    out["per_asteroid"] = dict(n_asteroids=len(ast), **{
        m: dict(detected_in_ge1_frame=sum(a[m] for a in ast.values()),
                wilson95=wilson(sum(a[m] for a in ast.values()), len(ast), 1.96)) for m in methods})
    out["by_vmag"] = []
    for lo, hi in ((0, 16), (16, 17), (17, 18), (18, 19)):
        sub = [q for q in rows if lo <= q["vmag"] < hi]
        out["by_vmag"].append(dict(vmag=[lo, hi], n=len(sub), **{
            m: dict(hits=sum(bool(q[m]) for q in sub), wilson95=wilson(sum(bool(q[m]) for q in sub), len(sub), 1.96))
            for m in ("M2", "M4")}))
    return out


def gallery(rows, cache_maps, path, n=24, ncol=8):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    miss = sorted([q for q in rows if not q["M4"]], key=lambda q: q["vmag"])[:n]
    if not miss:
        return 0
    nrow = -(-len(miss) // ncol)
    fig, ax = plt.subplots(2 * nrow, ncol, figsize=(1.45 * ncol, 2.9 * nrow), squeeze=False)
    for a in ax.ravel():
        a.set_axis_off()
    cm = plt.get_cmap("PuOr_r").copy(); cm.set_bad("#203050")
    for k_, q in enumerate(miss):
        fd, v, r = cache_maps[q["field"]]
        e, a0, a1 = q["frame"], ax[2 * (k_ // ncol), k_ % ncol], ax[2 * (k_ // ncol) + 1, k_ % ncol]
        a0.imshow(np.arcsinh(r["vs"][e] / r["sigma"][e] / 3), cmap="gray", vmin=0, vmax=np.arcsinh(10 / 3))
        a1.imshow(r["corr"][e], cmap=cm, vmin=-8, vmax=8)
        for a in (a0, a1):
            a.set_axis_on()
            a.plot(q["x"], q["y"], "o", mfc="none", mec="cyan", ms=10, mew=0.7)
            a.set_xlim(q["x"] - 8, q["x"] + 8); a.set_ylim(q["y"] + 8, q["y"] - 8)
            a.set_xticks([]); a.set_yticks([])
        a0.set_title(f"{q['name']} V{q['vmag']}\nD{q['detector']} {q['wl_um']} um", fontsize=6)
        a1.set_title(f"{q['field']} frame {e}", fontsize=5.5)
    fig.suptitle("E3b misses of M4 (brightest first; OUTLIER not masked). Per case: background-subtracted image / sigma "
                 "(top) and rev-2 corrected map +-8 sigma (bottom; navy = hard-bad masked). Cyan = SkyBoT. 16 px boxes",
                 fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return len(miss)


def test():
    c = committed()
    if not c:
        sys.exit(f"{THR} is not committed; refusing to open test fields")
    thr = json.load(open(THR))
    k = {m: v["k"] for m, v in thr["methods"].items()}
    f = thr["f"]
    blind = [x["field"] for x in json.load(open(FIELDS))["fields"]]
    e3 = [f"e3x_{n}" for n in EXISTING] + [x["field"] for x in json.load(open("data/findings/e3_fields.json"))["fields"]] \
        + [x["field"] for x in json.load(open("data/efields/e2_fields.json"))["fields"]]
    cm, recs, rows, rows2, missing = {}, [], [], [], []
    methods = None
    for group, names, sink in (("blind", blind, rows), ("e3_not_blind", e3, rows2)):
        for nm in names:
            if not run_e1.ready(nm):
                missing.append(nm)
                continue
            rec, rr, methods, maps = field_rows(nm, k, f, cm if group == "blind" else None)
            rec["group"] = group
            if nm.startswith("e2_"):        # false components per frame, nothing planted, at the E3b thresholds
                rec["false_components"] = {m: [int(sum(e1.n_components(mm, k[m]) for mm in maps[m] if mm is not None)),
                                               int(sum(mm is not None for mm in maps[m]))] for m in methods}
            recs.append(rec); sink += rr
            print({x: rec[x] for x in rec if x != "false_components"}, flush=True)
    e2r = [r for r in recs if "false_components" in r]
    fc = {m: dict(components=sum(r["false_components"][m][0] for r in e2r), maps=sum(r["false_components"][m][1] for r in e2r))
          for m in methods}
    for m in fc:
        fc[m]["per_frame"] = round(fc[m]["components"] / fc[m]["maps"], 5) if fc[m]["maps"] else None
    n_gal = gallery(rows, cm, "data/findings/e3b_misses.png")
    out = dict(experiment="E3b recall with OUTLIER not masked", protocol="docs/PROTOCOLS.md#e3b",
               thresholds=k, thresholds_file=THR, thresholds_commit=c, noise_floor_f=f,
               switch="orrery.e1.MASK_BITS = BAD_BITS within scripts/run_e3b.py:outlier_kept (e1.py unchanged)",
               rules="D1 (despike, positive components, blink S = reference frame); 8-connected >= 2 px above k; "
                     "detected if any component pixel within 2 px of the SkyBoT position; truth V < 19, >= 3 px from "
                     "the edge; pairs without an M2 map excluded for all methods",
               truth_source="IMCCE SkyBoT conesearch, geocentric (LOC=500); SPHEREx low-Earth-orbit parallax <~1 arcsec "
                            "(0.16 px) for main-belt asteroids",
               frame_cap="250 index rows evenly in MJD (scripts/e2_cache.build), frames whose grid centre is off the "
                         "detector dropped (run_e1.load_field)",
               primary=dict(fields=blind, **summary(rows, methods)),
               secondary_e3_fields_not_blind=dict(note="the 36 E3 fields (existing 6 re-cached as e3x_*, 10 E3 "
                                                       "ecliptic, 20 E2); NOT blind: E3's result motivated E3b",
                                                  **summary(rows2, methods)),
               secondary_false_components_e2=dict(note="20 E2 fields, nothing planted, positive components per map "
                                                       "at the E3b thresholds (real sources incl. asteroids count)",
                                                  **fc),
               per_field=recs, fields_missing=missing,
               gallery=dict(path="data/findings/e3b_misses.png", n_shown=n_gal,
                            n_misses_M4=sum(not q["M4"] for q in rows)),
               rows=rows, rows_e3_fields=rows2)
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps({x: out[x] for x in ("thresholds", "primary", "secondary_false_components_e2")}, indent=1))


if __name__ == "__main__":
    dict(calib=calib, cache_existing=lambda: cache("existing"), cache_blind=lambda: cache("blind"), test=test)[sys.argv[1]]()
