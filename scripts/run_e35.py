"""E35 - simultaneous-pair coincidence (docs/protocols/E35.md).

  fields <grp>  per-field records: grp in {calib, test, e34a, e35a} -> data/phase2/e35/runs/
  summary       thresholds (calibration), false rates (test), sightings -> data/findings/e35.json, e35s.json
"""
import json, os, sys, time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_e30 as R  # noqa: E402
import run_e34 as Q  # noqa: E402
from orrery import detect, e1  # noqa: E402

BASE = "data/phase2/e35"
RUNS = f"{BASE}/runs"
OUT, OUTS = "data/findings/e35.json", "data/findings/e35s.json"
DMJD, RAD, PZ = 1e-4, 1.5, 3.0
F = R.f_floor()


def partners(det, mjd):
    """Partner picture index (det +/- 3, |dMJD| < 1e-4 d) or -1."""
    out = np.full(len(det), -1)
    for e in range(len(det)):
        k = np.flatnonzero((np.abs(mjd - mjd[e]) < DMJD) & (np.abs(det - det[e]) == 3))
        if len(k):
            out[e] = k[np.argmin(np.abs(mjd[k] - mjd[e]))]
    return out


def load(name):
    import run_e1
    if name.startswith("e35") or name.startswith("e34"):
        run_e1.ROOT = f"data/phase2/{name[:3]}/efields"
        return run_e1.load_field(name)
    return R.load(name)


def horizons(name, fd, mjd):
    import ephem
    from orrery.change import world_to_grid
    sel = next(x for x in json.load(open(f"data/phase2/{name[:3]}/fields.json"))["fields"] if x["field"] == name)
    df = ephem.fetch_epochs(str(sel["selected_by"]), [float(m) for m in mjd])
    out = [[] for _ in mjd]
    for e, (_, r) in enumerate(df.iterrows()):
        x, y = world_to_grid(float(r["RA"]), float(r["DEC"]), R.S, fd["ra0"], fd["dec0"])
        out[e].append(dict(name=str(sel["selected_by"]), vmag=float(r["V"]), x=float(x), y=float(y)))
    return out


def field(name):
    p = f"{RUNS}/{name}.json"
    if os.path.exists(p):
        return name
    t0 = time.time()
    v2 = json.load(open(R.MODEL))["model"]
    fd = load(name)
    lam, det, mjd = R.meta_arrays(fd)
    st = R.stage1(R.live_v(fd), lam, det, mjd)
    cd = R.cands(st, det)
    sv2 = detect.forward(v2, cd["X"], detect.onehot(cd["det"])) if len(cd["e"]) else np.zeros(0)
    par = partners(det, mjd)
    has = np.zeros(len(cd["e"]), bool)
    for j in range(len(cd["e"])):
        q = par[cd["e"][j]]
        if q >= 0:
            k = cd["e"] == q
            has[j] = bool(np.any(np.hypot(cd["x"][k] - cd["x"][j], cd["y"][k] - cd["y"][j]) <= RAD))
    rec = dict(field=name, npic=int(sum(np.isfinite(z).any() for z in st["z"])), n_with_partner=int(np.sum(par >= 0)),
               z=cd["z"].tolist(), v2=sv2.tolist(), partner=has.tolist())
    if name.startswith(("e34a", "e35a")):
        import run_e27
        with run_e27.outlier_kept():
            mk = run_e27.maps_for(e1.frame_arrays(fd)[0], lam, det, mjd, F)
        ast = horizons(name, fd, mjd)
        rows = []
        for e in range(len(mjd)):
            for o in ast[e]:
                if not (o["vmag"] < R.VMAX and R.EDGE <= o["x"] <= R.S - 1 - R.EDGE and R.EDGE <= o["y"] <= R.S - 1 - R.EDGE):
                    continue
                if st["prev"][e] < 0:
                    continue
                k = np.flatnonzero((cd["e"] == e) & (np.hypot(cd["x"] - o["x"], cd["y"] - o["y"]) <= R.R_AST))
                kc = k[has[k]]
                rows.append(dict(frame=e, name=o["name"], vmag=o["vmag"], partner_picture=bool(par[e] >= 0),
                                 CZ=float(cd["z"][kc].max()) if len(kc) else None,
                                 Cv2=float(sv2[kc].max()) if len(kc) else None,
                                 V2=float(sv2[k].max()) if len(k) else None,
                                 M3BK=detect.comp_kmax(mk["M3b"][e], o["x"], o["y"], R.R_AST) if mk["M3b"][e] is not None else None,
                                 M2K=detect.comp_kmax(mk["M2"][e], o["x"], o["y"], R.R_AST) if mk["M2"][e] is not None else None))
        rec["rows"] = rows
    json.dump(rec, open(p + ".tmp", "w"))
    os.replace(p + ".tmp", p)
    print(f"{name}: {rec['npic']} pictures, {rec['n_with_partner']} with partner, {len(rec.get('rows', []))} sightings, "
          f"{time.time() - t0:.0f}s", flush=True)
    return name


def names(grp):
    if grp in ("calib", "test"):
        return [x["field"] for x in R.group(grp)]
    import run_e1
    run_e1.ROOT = f"data/phase2/{grp[:3]}/efields"
    return [x["field"] for x in json.load(open(f"data/phase2/{grp[:3]}/fields.json"))["fields"] if run_e1.ready(x["field"])]


def scores(r, m):
    s = np.array(r["z"] if m == "CZ" else r["v2"], float)
    return s[np.array(r["partner"], bool)]


def summary():
    cal = [json.load(open(f"{RUNS}/{n}.json")) for n in names("calib")]
    tst = [json.load(open(f"{RUNS}/{n}.json")) for n in names("test")]
    thr = {m: {op: detect.threshold_scores(np.concatenate([scores(r, m) for r in cal]), sum(r["npic"] for r in cal), q)
               for op, q in zip(R.OPS, R.RATES)} for m in ("CZ", "Cv2")}
    t34 = json.load(open(Q.THR))["thresholds"]
    k27 = json.load(open("data/findings/e27.json"))["thresholds"]["kept"]
    t3 = dict(CZ=thr["CZ"]["r3"], Cv2=thr["Cv2"]["r3"], V2=t34["V2"]["r3"], M3BK=k27["M3b"]["k"]["r3"], M2K=k27["M2"]["k"]["r3"])
    idx = np.random.default_rng(2026111571).integers(0, len(tst), size=(2000, len(tst)))
    fa = {}
    for m in ("CZ", "Cv2"):
        c = np.array([np.sum(scores(r, m) > thr[m]["r3"]) for r in tst], float)
        n = np.array([r["npic"] for r in tst], float)
        b = c[idx].sum(1) / n[idx].sum(1)
        fa[m] = dict(rate=round(float(c.sum() / n.sum()), 5), ci95=[round(float(np.percentile(b, 2.5)), 5), round(float(np.percentile(b, 97.5)), 5)],
                     false_marks=int(c.sum()), n_pictures=int(n.sum()),
                     rate_by_op={op: round(float(sum(np.sum(scores(r, m) > thr[m][op]) for r in tst) / n.sum()), 5) for op in R.OPS})
    meth = ("CZ", "Cv2", "V2", "M3BK", "M2K")

    def ast(grp):
        F_ = [json.load(open(f"{RUNS}/{n}.json")) for n in names(grp) if os.path.exists(f"{RUNS}/{n}.json")]
        rows = [dict(x, field=f["field"]) for f in F_ for x in f["rows"]]
        bins = dict(V17p5=lambda v: v <= 17.5, V16=lambda v: v <= 16, V16_17p5=lambda v: 16 < v <= 17.5)
        out = dict(fields=[f["field"] for f in F_])
        for b, fn in bins.items():
            rr = [x for x in rows if fn(x["vmag"])]
            d = dict(n=len(rr), objects=len({(x["field"], x["name"]) for x in rr}),
                     with_partner_picture=int(sum(x["partner_picture"] for x in rr)))
            for m in meth:
                h = int(sum(x[m] is not None and x[m] > t3[m] for x in rr))
                d[m] = dict(hits=h, recall=round(h / len(rr), 4) if rr else None, wilson95=R.wilson(h, len(rr)),
                            objects=len({(x["field"], x["name"]) for x in rr if x[m] is not None and x[m] > t3[m]}))
            out[b] = d
        return out
    A, B = ast("e35a"), ast("e34a")
    passes = {m: bool(A["V17p5"][m]["wilson95"][0] is not None and A["V17p5"][m]["wilson95"][0] > 0.5 and fa[m]["rate"] <= 0.045)
              for m in ("CZ", "Cv2")}
    if passes["CZ"] and passes["Cv2"]:
        shipped = "Cv2" if A["V17p5"]["Cv2"]["wilson95"][0] > A["V17p5"]["CZ"]["wilson95"][1] else "CZ"
    else:
        shipped = "CZ" if passes["CZ"] else ("Cv2" if passes["Cv2"] else "none")
    out = dict(experiment="E35", protocol="docs/protocols/E35.md", thresholds=thr, thresholds_r3_all=t3, false_rate_test=fa,
               fresh=A, e34a_post_hoc=B, passes=passes, shipped=shipped,
               verdict="works on real moving objects" if shipped != "none" else "not shown",
               rules=dict(pair_dmjd=DMJD, radius_px=RAD, partner_zmin=PZ, n_calib_fields=len(cal), n_test_fields=len(tst),
                          calib_pictures=sum(r["npic"] for r in cal),
                          sightings="selecting asteroid from Horizons (SPHEREx observer), V < 19 then V <= 17.5, >= 3 px from "
                                    "the edge, picture with an M2 map (E34 rule); detection within 2 px"),
               generated_local=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    json.dump(out, open(OUT, "w"), indent=1)
    s = dict(source=OUT, verdict=out["verdict"], shipped=shipped, n_fresh=A["V17p5"]["n"], objects_fresh=A["V17p5"]["objects"],
             n_e34a=B["V17p5"]["n"], objects_e34a=B["V17p5"]["objects"])
    for m in ("CZ", "Cv2"):
        for op in R.OPS:
            s[f"t_{m}_{op}"] = round(float(thr[m][op]), 4)
        s[f"fa_{m}"], s[f"fa_{m}_lo"], s[f"fa_{m}_hi"] = fa[m]["rate"], *fa[m]["ci95"]
        s[f"pass_{m}"] = passes[m]
    for suf, X in (("", A), ("_e34a", B)):
        for b in ("V17p5", "V16", "V16_17p5"):
            bs = "" if b == "V17p5" else f"_{b}"
            s[f"n{bs}{suf}"] = X[b]["n"]
            for m in meth:
                k = "v2" if m == "V2" else m
                s[f"hits_{k}{bs}{suf}"] = X[b][m]["hits"]
                s[f"lo_{k}{bs}{suf}"], s[f"hi_{k}{bs}{suf}"] = X[b][m]["wilson95"]
    assert all(not any(c in k for c in ".- ") for k in s)
    json.dump(s, open(OUTS, "w"), indent=1)
    print(json.dumps(dict(thr=thr, fa=fa, fresh={b: {m: (A[b][m]["hits"], A[b]["n"], A[b][m]["wilson95"]) for m in meth} for b in ("V17p5", "V16", "V16_17p5")},
                          e34a={m: (B["V17p5"][m]["hits"], B["V17p5"]["n"], B["V17p5"][m]["wilson95"]) for m in meth},
                          passes=passes, shipped=shipped, fields=len(A["fields"]))))


if __name__ == "__main__":
    a = sys.argv[1]
    if a == "fields":
        os.makedirs(RUNS, exist_ok=True)
        w = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 1
        with Pool(w) as pl:
            list(pl.imap_unordered(field, [n for g in sys.argv[2].split(",") for n in names(g)]))
    else:
        summary()
