"""E23: census-scale repeat of the frozen E20 Pa-alpha / Br-alpha line-excess rule on a larger, disjoint seeded draw
of Gaia DR3 ESP-ELS candidates. Frozen choices: docs/protocols/E23.md. Statistic, calibration, detection rule,
photometry and picture selection are E20's own functions (scripts/run_e20.py), pointed at data/phase2/e23.

  OMP_NUM_THREADS=1 .venv/bin/python scripts/run_e23.py select    # seeded draw + Gaia fields -> data/phase2/e23/sample.json
  OMP_NUM_THREADS=1 .venv/bin/python scripts/run_e23.py measure   # fetch -> photometry -> delete stamps, 3 processes, in sample order
  OMP_NUM_THREADS=1 .venv/bin/python scripts/run_e23.py analyze   # frozen statistic + rule -> data/findings/e23.json
  OMP_NUM_THREADS=1 .venv/bin/python scripts/run_e23.py simbad    # pre-declared SIMBAD main-type check
  OMP_NUM_THREADS=1 .venv/bin/python scripts/run_e23.py lit       # VSX / SIMBAD all types / VizieR titles / AllWISE for detections
  OMP_NUM_THREADS=1 .venv/bin/python scripts/run_e23.py fig       # site/data/figs/e23_census.png
"""
import json, os, shutil, sys, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_e20 as E  # noqa: E402

from orrery import e7 as _e7  # noqa: E402

_fp = _e7.forced_photometry


def _forced_photometry(image, variance, flags, cols, groups, **kw):
    """E7 fit unchanged; a picture whose sub-cutout has NO usable pixel (the fit raises on an empty SVD) returns
    amp = err = NaN, n_pix = 0 per group, i.e. the picture is excluded by the E20 non-finite / < 18 px rule."""
    try:
        return _fp(image, variance, flags, cols, groups, **kw)
    except ValueError as e:
        if "zero-size array" not in str(e):
            raise
        return [(np.nan, np.nan, 0) for _ in groups]


_e7.forced_photometry = _forced_photometry

P2 = "data/phase2/e23"
E.P2 = P2                                   # E20 functions (field_gaia, phot_one) read/write under data/phase2/e23
OUT = "data/findings/e23.json"
FIG = "data/findings/e23_census.png"
SEED = 2026111523
CLASSES = {"HerbigStar": ("classprob_espels_herbigstar", 0.70, "data/phase2/e20/espels_HerbigStar.ecsv"),
           "beStar": ("classprob_espels_bestar", 0.90, "data/phase2/e20/espels_beStar.ecsv"),
           "TTauri": ("classprob_espels_ttauristar", 0.70, f"{P2}/espels_TTauri.ecsv")}
CAP_PER_CLASS = 10 ** 6                               # no cap: every eligible candidate is drawn
STOP_DISPATCH = os.environ.get("E23_STOP", "")         # optional PST wall clock (HH:MM): no new field started after this
FETCH_THREADS = int(os.environ.get("E23_THREADS", "6"))
EMISSION = {"Em*", "Be*", "Ae*", "Ae?", "Be?", "Or*", "TT*", "TT?", "Y*O", "Y*?", "FU*", "pr*", "pr?"}
SIMBAD_TAP = "https://simbad.harvard.edu/simbad/sim-tap/sync"
VIZ_TAP = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"


# ------------------------------------------------------------------ sample
def tap(q, tries=12):
    """Same query as run_e20.tap, more patient with the AIP rate limit (HTTP 429)."""
    import pyvo
    for i in range(tries):
        try:
            return pyvo.dal.TAPService(E.TAP).run_sync(q, maxrec=200000).to_table().to_pandas()
        except Exception as e:
            print(f"  TAP attempt {i + 1}: {e!r}"[:160], flush=True)
            time.sleep(min(120, 10 * (i + 1)))
    raise RuntimeError("TAP failed")


E.tap = tap


def pick(c, r, labelled):
    """One candidate: E20 isolation rule + E20 control matching; control order from a per-candidate seeded rng."""
    sid = str(int(r.source_id))
    g = E.field_gaia(sid, float(r.ra), float(r.dec)).reset_index(drop=True)
    ic = np.flatnonzero(g.source_id.values == int(sid))
    if len(ic) != 1:
        return None, "not found in field table"
    ic = int(ic[0])
    ok, why = E.isolated(g, ic)
    if not ok:
        return None, why
    rng = np.random.default_rng([SEED, int(sid) % (2 ** 63)])
    s = E.sep_arcsec(g.ra.values, g.dec.values, float(r.ra), float(r.dec))
    ctrl, rule = [], None
    for dG, dC in E.CTRL_MATCH:
        m = np.flatnonzero((s >= E.CTRL_SEP[0]) & (s <= E.CTRL_SEP[1]) & (np.abs(g.phot_g_mean_mag.values - r.phot_g_mean_mag) <= dG)
                           & (np.abs(g.bp_rp.values - r.bp_rp) <= dC) & ~np.isin(g.source_id.values, list(labelled)))
        m = [int(k) for k in m if E.isolated(g, int(k))[0]]
        if len(m) >= E.MIN_CTRL:
            ctrl, rule = [m[k] for k in rng.permutation(len(m))], (dG, dC)
            break
    if len(ctrl) < E.MIN_CTRL:
        return None, "fewer than 2 matched isolated controls"
    ids = ",".join(str(int(g.source_id[k])) for k in ctrl[:8])
    lab = E.tap(f"SELECT source_id, classlabel_espels FROM gaiadr3.astrophysical_parameters WHERE source_id IN ({ids})")
    bad = set(int(x) for x, l in zip(lab.source_id, lab.classlabel_espels) if isinstance(l, str) and l.strip())
    ctrl = [k for k in ctrl[:8] if int(g.source_id[k]) not in bad][:E.N_CTRL]
    if len(ctrl) < E.MIN_CTRL:
        return None, "controls carry ESP-ELS labels"
    return dict(source_id=sid, cls=c, ra=float(r.ra), dec=float(r.dec), l=float(r.l), b=float(r.b),
                G=float(r.phot_g_mean_mag), bp_rp=float(r.bp_rp), prob=float(r[CLASSES[c][0]]),
                flag=int(r.classlabel_espels_flag), ew_halpha_nm=float(r.ew_espels_halpha), ctrl_match=list(rule),
                controls=[dict(source_id=str(int(g.source_id[k])), ra=float(g.ra[k]), dec=float(g.dec[k]),
                               G=float(g.phot_g_mean_mag[k]), bp_rp=float(g.bp_rp[k]), sep_arcsec=float(s[k])) for k in ctrl]), ""


def stage_select():
    from astropy.table import Table
    e20 = json.load(open("data/phase2/e20/sample.json"))
    e20_ids = {x["source_id"] for x in e20["sample"]} | {x["source_id"] for x in e20["rejected"]}
    rng = np.random.default_rng(SEED)
    pools, labelled = {}, set()
    for c, (pcol, pmin, path) in CLASSES.items():
        if not os.path.exists(path):
            print("no table for", c, "- class not drawn", flush=True)
            continue
        t = Table.read(path).to_pandas()
        labelled |= set(int(x) for x in t.source_id)
        flag = t.classlabel_espels_flag.astype(str).astype(int)
        s = t[(t[pcol] >= pmin) & (flag < 10) & t.phot_g_mean_mag.between(*E.G_RANGE) & t.bp_rp.notna()
              & ~t.source_id.astype(str).isin(e20_ids)]
        s = s.sort_values("source_id").reset_index(drop=True)
        pools[c] = s.iloc[rng.permutation(len(s))].reset_index(drop=True)
        print(c, "eligible (E20 draws removed)", len(s), flush=True)
    order, k = [], 0                                   # round robin over classes, each in its seeded order
    while any(k < len(p) for p in pools.values()):
        order += [(c, pools[c].iloc[k]) for c in pools if k < len(pools[c])]
        k += 1
    sample, rejected, n_acc = [], [], {c: 0 for c in pools}
    with ThreadPoolExecutor(int(os.environ.get("E23_SEL_THREADS", "3"))) as ex:
        for i0 in range(0, len(order), 48):
            chunk = [(c, r) for c, r in order[i0:i0 + 48] if n_acc[c] < CAP_PER_CLASS]
            if not chunk:
                if all(n_acc[c] >= CAP_PER_CLASS for c in pools):
                    break
                continue
            res = list(ex.map(lambda cr: pick(cr[0], cr[1], labelled), chunk))
            for (c, r), (acc, why) in zip(chunk, res):
                if acc is None:
                    rejected.append(dict(source_id=str(int(r.source_id)), cls=c, reason=why))
                elif n_acc[c] < CAP_PER_CLASS:
                    acc["draw_index"] = len(sample) + len(rejected)
                    sample.append(acc)
                    n_acc[c] += 1
            out = dict(seed=SEED, classes={c: dict(prob_col=v[0], prob_min=v[1]) for c, v in CLASSES.items()},
                       g_range=E.G_RANGE, cap_per_class=CAP_PER_CLASS, eligible={c: len(p) for c, p in pools.items()},
                       complete=False, n_accepted=n_acc, sample=sample, rejected=rejected)
            json.dump(out, open(f"{P2}/sample.json.tmp", "w"), indent=1)
            os.replace(f"{P2}/sample.json.tmp", f"{P2}/sample.json")
            print(f"drawn {i0 + len(chunk)}: accepted {n_acc} rejected {len(rejected)}", flush=True)
    out["complete"] = True
    json.dump(out, open(f"{P2}/sample.json.tmp", "w"), indent=1)
    os.replace(f"{P2}/sample.json.tmp", f"{P2}/sample.json")
    print("DONE accepted", n_acc, "rejected", len(rejected), "controls", sum(len(x["controls"]) for x in sample), flush=True)


# ------------------------------------------------------------------ measurement (fetch -> photometry -> delete stamps)
_DF = None


def frame_list(s):
    global _DF
    from build_field import candidates
    if _DF is None:
        df = pd.read_parquet("data/index/frames_dr1.parquet")
        _DF = df[(df.dqa == "Pass") & df.detector.isin(E.DETS)].reset_index(drop=True)
    c = candidates(_DF, s["ra"], s["dec"]).sort_values("mjd", kind="stable").reset_index(drop=True)
    if len(c) > E.CAP:
        c = c.iloc[np.linspace(0, len(c) - 1, E.CAP).round().astype(int)]
    return [dict(key=r.key, obs_id=r.obs_id, detector=int(r.detector), mjd=float(r.mjd)) for r in c.itertuples()]


def measure_one(s):
    import e2_cache
    from orrery.data import save_stamp
    sid = s["source_id"]
    st_path = f"{P2}/status/{sid}.json"
    if os.path.exists(f"{P2}/phot/{sid}.json"):
        return sid, "done"
    t0 = time.time()
    fl = frame_list(s)
    d = f"{P2}/fields/{sid}/frames"
    os.makedirs(d, exist_ok=True)

    def one(f):
        p = f"{d}/{f['obs_id']}_D{f['detector']}.npz"
        if os.path.exists(p) or os.path.exists(p + ".off"):
            return "ok"
        for i in range(3):
            try:
                st = e2_cache.fast_stamp(f["key"], s["ra"], s["dec"])
                break
            except Exception as e:
                if i == 2:
                    return f"error {e!r}"[:200]
                time.sleep(5)
        if st is None:
            open(p + ".off", "w").close()
            return "off"
        save_stamp(st, p[:-4] + ".tmp.npz")
        os.replace(p[:-4] + ".tmp.npz", p)
        return "ok"

    with ThreadPoolExecutor(FETCH_THREADS) as ex:
        res = list(ex.map(one, fl))
    err = [r for r in res if r.startswith("error")]
    t1 = time.time()
    if err:
        json.dump(dict(source_id=sid, n_frames=len(fl), n_err=len(err), err=err[:3]), open(st_path, "w"))
        return sid, f"fetch errors {len(err)}"
    open(f"{P2}/fields/{sid}/READY", "w").close()
    E.phot_one(sid)
    shutil.rmtree(d)                                   # stamps not kept (disk cap); photometry JSON is the record
    json.dump(dict(source_id=sid, n_frames=len(fl), n_err=0, t_fetch=round(t1 - t0), t_phot=round(time.time() - t1),
                   finished=time.strftime("%H:%M:%S")), open(st_path, "w"))
    return sid, f"ok {len(fl)} frames fetch {t1 - t0:.0f}s phot {time.time() - t1:.0f}s"


def past_stop():
    if not STOP_DISPATCH:
        return False
    h, m = map(int, STOP_DISPATCH.split(":"))
    lt = time.localtime()
    return lt.tm_hour * 60 + lt.tm_min >= h * 60 + m


def stage_measure(procs=int(os.environ.get("E23_PROCS", "3"))):
    from multiprocessing import Pool
    for sub in ("phot", "status", "fields"):
        os.makedirs(f"{P2}/{sub}", exist_ok=True)
    started, pend, n = set(), [], 0
    with Pool(procs) as pool:
        while True:
            smp = json.load(open(f"{P2}/sample.json"))
            todo = [s for s in smp["sample"] if s["source_id"] not in started and not os.path.exists(f"{P2}/phot/{s['source_id']}.json")]
            while todo and len(pend) < procs and not past_stop():
                s = todo.pop(0)
                started.add(s["source_id"])
                pend.append(pool.apply_async(measure_one, (s,)))
            for a in [a for a in pend if a.ready()]:
                pend.remove(a)
                n += 1
                try:
                    sid, msg = a.get()
                except Exception as e:
                    sid, msg = "?", f"exception {e!r}"[:200]
                print(time.strftime("%H:%M:%S"), n, sid, msg, flush=True)
            if not pend and (past_stop() or (smp["complete"] and not todo)):
                break
            time.sleep(2)
    print("measure stopped", time.strftime("%H:%M:%S"), "fields measured", len(os.listdir(f"{P2}/phot")), flush=True)


# ------------------------------------------------------------------ frozen analysis (E20 statistic, calibration, rule)
def stage_analyze():
    from scipy.stats import beta as B
    from orrery.e7 import windows
    s_d = json.load(open("data/findings/e7_gate0.json"))["s_d"]
    anom = E.anomaly_set()
    smp = json.load(open(f"{P2}/sample.json"))
    sample = smp["sample"]
    rows = []
    for s in sample:
        p = f"{P2}/phot/{s['source_id']}.json"
        if not os.path.exists(p):
            continue
        ph = json.load(open(p))["stars"]
        mj = sorted(set(m for r in ph.values() for m in r["mjd"]))
        win_of = dict(zip(mj, windows(np.array(mj)))) if mj else {}
        for sid, r in ph.items():
            a, q = E.star_arrays(r, s_d, anom)
            cand = r["role"] == "candidate"
            rows.append(dict(source_id=sid, role=r["role"], field=s["source_id"], cls=s["cls"],
                             G=s["G"] if cand else next(c["G"] for c in s["controls"] if c["source_id"] == sid),
                             ra=s["ra"] if cand else next(c["ra"] for c in s["controls"] if c["source_id"] == sid),
                             dec=s["dec"] if cand else next(c["dec"] for c in s["controls"] if c["source_id"] == sid),
                             **q, lines=E.star_lines(a, win_of)))
    cal = {}
    for L in E.LINES:
        zc = [r["lines"][L]["z"] for r in rows if r["role"] == "control" and "z" in r["lines"][L]]
        zw = [v["z"] for r in rows if r["role"] == "control" for v in r["lines"][L]["windows"].values() if "z" in v]
        m, sc = E.robust(zc) if len(zc) >= 10 else (0.0, 1.0)
        mw, sw = E.robust(zw) if len(zw) >= 10 else (0.0, 1.0)
        cal[L] = dict(n_controls=len(zc), median_z=m, scale=sc, n_control_windows=len(zw), median_zw=mw, scale_w=sw)
    for r in rows:
        r["detected"] = {}
        for L in E.LINES:
            o, c = r["lines"][L], cal[L]
            if "z" in o:
                o["Z"] = (o["z"] - c["median_z"]) / c["scale"]
                for v in o["windows"].values():
                    if "z" in v:
                        v["Z"] = (v["z"] - c["median_zw"]) / c["scale_w"]
                o["n_windows_Zw_ge2"] = int(sum(v.get("Z", -99) >= E.Z_WIN for v in o["windows"].values()))
            o["testable"] = o["n_windows"] >= E.N_WIN
            r["detected"][L] = bool(o["testable"] and o["Z"] >= E.Z_DET and o["n_windows_Zw_ge2"] >= E.N_WIN)
        r["testable_main"] = r["lines"]["Paa"]["testable"] or r["lines"]["Bra"]["testable"]
        r["excess_main"] = r["detected"]["Paa"] or r["detected"]["Bra"]
    ctrl = [r for r in rows if r["role"] == "control"]
    nc = sum(r["testable_main"] for r in ctrl)
    kc = sum(r["excess_main"] for r in ctrl)
    p_fp = kc / nc if nc else None
    up95 = float(B.ppf(0.95, kc + 1, nc - kc)) if nc else None
    works = bool(nc > 0 and p_fp <= 0.02)
    out = dict(experiment="E23", protocol="docs/protocols/E23.md", seed=SEED, statistic="scripts/run_e20.py (E20, unchanged)",
               calibration_from_controls=cal, n_fields_selected=len(sample), n_fields_measured=len({r["field"] for r in rows}),
               sample_selection_complete=smp["complete"], n_accepted_per_class=smp["n_accepted"], eligible_per_class=smp["eligible"],
               control_fp=kc, control_testable=nc, control_fp_rate=p_fp, control_fp_rate_upper95=up95,
               control_n=len(ctrl), method_works=works,
               verdict="method works (control false-positive rate <= 0.02)" if works else "FAILED: control false-positive rate > 0.02",
               control_passes=[dict(source_id=r["source_id"], field=r["field"], G=r["G"], cls_field=r["cls"],
                                    Z={L: r["lines"][L].get("Z") for L in E.LINES}) for r in ctrl if r["excess_main"]])
    cand = [r for r in rows if r["role"] == "candidate"]
    totM = totN = 0
    for c in CLASSES:
        cc = [r for r in cand if r["cls"] == c]
        M = sum(r["testable_main"] for r in cc)
        N = sum(r["excess_main"] for r in cc)
        totM, totN = totM + M, totN + N
        out[f"n_measured_{c}"], out[f"M_{c}"], out[f"N_{c}"] = len(cc), M, N
        out[f"expected_fp_{c}"] = p_fp * M if p_fp is not None else None
        out[f"expected_fp_upper95_{c}"] = up95 * M if up95 is not None else None
        out[f"N_brg_{c}"] = sum(r["detected"]["Brg"] for r in cc)
        out[f"sentence_{c}"] = (f"SPHEREx shows Paα/Brα excess consistent with emission-line stars for {N} of {M} Gaia ESP-ELS "
                                f"{CLS_NAME[c]} candidates; low resolution cannot separate stellar from nebular emission."
                                if works else "not reported (method failed the control test)")
    out.update(N_total=totN, M_total=totM, expected_fp=p_fp * totM if p_fp is not None else None,
               expected_fp_upper95=up95 * totM if up95 is not None else None, n_candidates_measured=len(cand),
               n_candidates_not_testable=sum(not r["testable_main"] for r in cand),
               n_not_testable_mostly_saturated=sum(not r["testable_main"] and r["n_sat"] > 0.5 * r["n_pictures"] for r in cand))
    for L in E.LINES:
        e = [r["lines"][L]["E"] for r in cand if r["detected"][L]]
        out[f"median_frac_excess_detected_{L}"] = float(np.median(e)) if e else None
    out["protocol_freeze"] = json.load(open(f"{P2}/protocol_freeze.json"))
    out["stars"] = rows
    json.dump(out, open(f"{P2}/analysis.json.tmp", "w"), indent=1)
    os.replace(f"{P2}/analysis.json.tmp", f"{P2}/analysis.json")
    print(json.dumps({k: v for k, v in out.items() if k not in ("stars",)}, indent=1))


CLS_NAME = {"HerbigStar": "Herbig Ae/Be", "beStar": "Be", "TTauri": "T Tauri"}


# ------------------------------------------------------------------ SIMBAD (pre-declared main-type check)
def simbad_q(q):
    import requests
    for i in range(4):
        try:
            r = requests.post(SIMBAD_TAP, data=dict(request="doQuery", lang="adql", format="json", query=q), timeout=120)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print("simbad retry", repr(e)[:120], flush=True)
            time.sleep(5 * (i + 1))
    raise RuntimeError("SIMBAD TAP failed")


def stage_simbad():
    a = json.load(open(f"{P2}/analysis.json"))
    ids = [r["source_id"] for r in a["stars"] if r["role"] == "candidate"]
    rows = []
    for i in range(0, len(ids), 200):
        lst = ",".join(f"'Gaia DR3 {x}'" for x in ids[i:i + 200])
        js = simbad_q("SELECT i.id, b.oid, b.main_id, b.otype, b.sp_type, b.nbref FROM ident AS i JOIN basic AS b ON b.oid = i.oidref "
                      f"WHERE i.id IN ({lst})")
        rows += js["data"]
    df = pd.DataFrame(rows, columns=["id", "oid", "main_id", "otype", "sp_type", "nbref"])
    df.to_csv(f"{P2}/simbad_ids.csv", index=False)
    print("SIMBAD entries", len(df), "of", len(ids))


# ------------------------------------------------------------------ extended checks for no-emission-type detections
def vsx(ra, dec):
    """VSX (B/vsx/vsx) within 5 arcsec, VizieR CfA mirror (the CDS VizieR TAP did not answer on 2026-11-15)."""
    import requests
    url = ("https://vizier.cfa.harvard.edu/viz-bin/asu-tsv?-source=B/vsx/vsx&-c=%.6f%+.6f&-c.rs=5&-out=Name,Type&-out.add=_r" % (ra, dec))
    for i in range(3):
        try:
            t = requests.get(url, timeout=60).text
            lines = [l for l in t.splitlines() if l and not l.startswith("#")]
            if len(lines) < 3 and "B_vsx" not in t and "vsx" not in t:
                raise RuntimeError("unexpected reply")
            hdr = [h.strip() for h in lines[0].split("\t")] if lines else []
            return [dict(zip(hdr, [x.strip() for x in l.split("\t")])) | {} for l in lines[3:]]
        except Exception as e:
            print("vsx retry", repr(e)[:100], flush=True)
            time.sleep(5)
    return None


LIT_WORDS = ("emission", "be star", "be-star", "herbig", "t tauri", "young stellar", "yso", "h-alpha", "halpha", "h alpha",
             "pre-main", "classical be", "emission-line", "balmer")


def vizier_titles(ra, dec):
    """Tables with a source within 2 arcsec, among the VizieR catalogues returned by a keyword search for emission-line /
    Be / Herbig / T Tauri / YSO / H-alpha / pre-main-sequence catalogues (data/phase2/e23/lit_catalogues.json; Gaia
    I/35x excluded), CfA mirror, queried 100 catalogues per request."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astroquery.vizier import Vizier
    cats = json.load(open(f"{P2}/lit_catalogues.json"))
    keys = sorted(cats)
    hits, n_req = [], 0
    for i in range(0, len(keys), 100):
        v = Vizier(columns=["_RAJ2000"], row_limit=1, timeout=300, vizier_server="vizier.cfa.harvard.edu", catalog=keys[i:i + 100])
        for k in range(3):
            try:
                res = v.query_region(SkyCoord(ra, dec, unit="deg"), radius=2 * u.arcsec)
                break
            except Exception as e:
                print("vizier retry", repr(e)[:100], flush=True)
                time.sleep(5)
        else:
            return None, None
        n_req += 1
        for name in res.keys():
            cat = next((c for c in keys[i:i + 100] if name == c or name.startswith(c + "/")), name)
            hits.append(f"{name}: {cats.get(cat, '')} | {res[name].meta.get('description', '')}")
    return len(keys), hits


def wise(ra, dec):
    import requests
    url = ("https://vizier.cfa.harvard.edu/viz-bin/asu-tsv?-source=II/328/allwise&-c=%.6f%+.6f&-c.rs=3&-sort=_r"
           "&-out=AllWISE,W1mag,e_W1mag,W2mag,e_W2mag,W3mag,e_W3mag,W4mag,e_W4mag,ccf,ex&-out.add=_r" % (ra, dec))
    try:
        t = requests.get(url, timeout=60).text
    except Exception:
        return "error"
    if "#Table\tII_328_allwise" not in t:
        return "error"
    lines = [l for l in t.splitlines() if l and not l.startswith("#")]
    if len(lines) < 4:
        return None
    row = dict(zip(lines[0].split("\t"), lines[3].split("\t")))

    def val(k):
        try:
            return float(row[k])
        except (KeyError, ValueError):
            return None
    d = {k: val(k) for k in ("W1mag", "W3mag", "e_W3mag", "W4mag")}
    d["W1_W3"] = d["W1mag"] - d["W3mag"] if d["W1mag"] is not None and d["W3mag"] is not None else None
    snr3 = 1.0857 / d["e_W3mag"] if d["e_W3mag"] else 0.0
    d["midir_excess_flag"] = bool(d["W1_W3"] is not None and d["W1_W3"] >= 1.0 and snr3 >= 5)
    return d


def stage_lit():
    a = json.load(open(f"{P2}/analysis.json"))
    sim = pd.read_csv(f"{P2}/simbad_ids.csv")
    names = {i.replace("Gaia DR3 ", ""): (m, o, int(n), oid) for i, m, o, n, oid in zip(sim.id, sim.main_id, sim.otype, sim.nbref, sim.oid)}
    p = f"{P2}/lit.json"
    lit = json.load(open(p)) if os.path.exists(p) else {}
    det = [r for r in a["stars"] if r["role"] == "candidate" and r["excess_main"]]

    def complete(rec):
        return rec.get("simbad_emission_main_type") or (rec.get("vsx") is not None and rec.get("vizier_title_hits") is not None)

    def one(r):
        sid = r["source_id"]
        nm = names.get(sid)
        rec = dict(simbad_main_id=nm[0] if nm else None, simbad_otype=nm[1] if nm else None, simbad_nbref=nm[2] if nm else 0)
        rec["simbad_emission_main_type"] = bool(nm and nm[1] in EMISSION)
        rec["wise"] = wise(r["ra"], r["dec"])
        if not rec["simbad_emission_main_type"]:
            if nm:
                js = simbad_q(f"SELECT otype FROM otypes WHERE oidref = {int(nm[3])}")
                rec["simbad_all_otypes"] = sorted({x[0] for x in js["data"]})
            else:
                rec["simbad_all_otypes"] = []
            rec["vsx"] = vsx(r["ra"], r["dec"])
            rec["vizier_n_tables"], rec["vizier_title_hits"] = vizier_titles(r["ra"], r["dec"])
        return sid, rec

    todo = [r for r in det if not complete(lit.get(r["source_id"], {}))]
    print("lit: to check", len(todo), "of", len(det), flush=True)
    with ThreadPoolExecutor(3) as ex:
        for sid, rec in ex.map(one, todo):
            lit[sid] = rec
            json.dump(lit, open(p + ".tmp", "w"), indent=1)
            os.replace(p + ".tmp", p)
            print(sid, rec.get("simbad_otype"), rec.get("vsx"), (rec.get("vizier_title_hits") or [])[:1], flush=True)


CSV = "data/products/epoch_emission_line_candidates.csv"
SEARCH_DATE = "2026-11-15 (PST)"
CHECKED = ("SIMBAD main and all object types via identifier 'Gaia DR3 <id>' (CfA mirror); VSX B/vsx/vsx within 5 arcsec "
           "(VizieR CfA mirror); 1867 VizieR catalogues found by keyword search for emission-line, Be, Herbig, T Tauri, YSO, "
           "H-alpha and pre-main-sequence objects, within 2 arcsec (VizieR CfA mirror; list in data/phase2/e23/lit_catalogues.json)")
VSX_EM = ("BE", "GCAS", "IN", "TTS", "UXOR", "FUOR", "EXOR", "YSO", "CTTS", "WTTS", "ISA", "INA", "INB", "INT", "INS")


def stage_final():
    a = json.load(open(f"{P2}/analysis.json"))
    lit = json.load(open(f"{P2}/lit.json")) if os.path.exists(f"{P2}/lit.json") else {}
    sim = pd.read_csv(f"{P2}/simbad_ids.csv")
    names = {i.replace("Gaia DR3 ", ""): (m, o) for i, m, o in zip(sim.id, sim.main_id, sim.otype)}
    cand = [r for r in a["stars"] if r["role"] == "candidate"]
    det = [r for r in cand if r["excess_main"]]
    no_type = [r for r in det if names.get(r["source_id"], (None, None))[1] not in EMISSION]
    table = []
    for r in no_type:
        L = lit.get(r["source_id"], {})
        sec = [t for t in L.get("simbad_all_otypes", []) if t in EMISSION]
        vx = [v for v in (L.get("vsx") or []) if any(t in str(v.get("Type")).upper().replace("+", " ").replace("|", " ").split() or
                                                     str(v.get("Type")).upper().startswith(t) for t in VSX_EM)]
        hits = [h for h in (L.get("vizier_title_hits") or [])
                if any(w in h.lower().replace("{alpha}", "alpha").replace("{", "").replace("}", "") for w in LIT_WORDS)]
        checked = bool(L) and L.get("vsx") is not None and L.get("vizier_title_hits") is not None
        w = L.get("wise")
        table.append(dict(source_id=r["source_id"], gaia_name=f"Gaia DR3 {r['source_id']}", ra=round(r["ra"], 5), dec=round(r["dec"], 5),
                          G=round(r["G"], 2), cls=r["cls"], simbad_main_id=names.get(r["source_id"], (None, None))[0],
                          simbad_otype=names.get(r["source_id"], (None, None))[1],
                          Z_Paa=r["lines"]["Paa"].get("Z"), Z_Bra=r["lines"]["Bra"].get("Z"), Z_Brg=r["lines"]["Brg"].get("Z"),
                          detected_Paa=r["detected"]["Paa"], detected_Bra=r["detected"]["Bra"],
                          frac_excess_Paa=r["lines"]["Paa"].get("E"), frac_excess_Bra=r["lines"]["Bra"].get("E"),
                          midir_excess_flag=(w.get("midir_excess_flag") if isinstance(w, dict) else None),
                          W1_W3=(w.get("W1_W3") if isinstance(w, dict) else None),
                          simbad_secondary_emission_types=sec, vsx=L.get("vsx"), vsx_emission_like=vx, vizier_emission_title_hits=hits,
                          vizier_all_hits=L.get("vizier_title_hits"),
                          vizier_emission_specific_hits=[h for h in hits if any(w in h.lower().replace("{alpha}", "alpha")
                                                         for w in ("emission", "halpha", "h-alpha", "h alpha", "be star", "herbig", "t tauri", "balmer"))],
                          extended_check_complete=checked,
                          no_published_emission_classification_found=bool(checked and not sec and not vx and not hits)))
    a.update(n_detected=len(det), n_detected_simbad_emission_type=len(det) - len(no_type),
             n_no_simbad_emission_type=len(no_type),
             n_no_published_emission_classification_found=sum(t["no_published_emission_classification_found"] for t in table),
             simbad_emission_types=sorted(EMISSION), candidate_table=table,
             n_only_yso_catalogue_hits=sum(bool(t["extended_check_complete"] and not t["simbad_secondary_emission_types"]
                                           and not t["vsx_emission_like"] and t["vizier_emission_title_hits"]
                                           and not t["vizier_emission_specific_hits"]) for t in table))
    a["simbad_otypes_detected"] = dict(pd.Series([str(names.get(r["source_id"], (None, None))[1]) for r in det]).value_counts())
    a["simbad_otypes_detected"] = {k: int(v) for k, v in a["simbad_otypes_detected"].items()}
    nf = a["n_no_published_emission_classification_found"]
    per = "; ".join(f"{a[f'N_{c}']} of {a[f'M_{c}']} {CLS_NAME[c]}" for c in CLASSES if a.get(f"M_{c}"))
    a["summary_plain"] = (
        f"We measured {a['n_candidates_measured']} stars that Gaia's software flags as likely emission-line stars (none of them in "
        f"the earlier 60-star test) and {a['control_n']} ordinary stars of similar brightness and colour in the same SPHEREx pictures, "
        "with exactly the rule fixed before the earlier test. "
        f"{a['N_total']} of the {a['M_total']} measurable candidates ({per}) show a clear bump at hydrogen Paschen-alpha or "
        f"Brackett-alpha in at least two separate visits; {a['control_fp']} of {a['control_testable']} ordinary stars do, so about "
        f"{a['expected_fp']:.1f} of the candidate results are expected to be flukes. "
        f"{a['n_no_simbad_emission_type']} of the detected stars carry no emission-line or young-star type in SIMBAD; for {nf} of them "
        "we also found no emission-line classification in SIMBAD's secondary types, the VSX variable-star index or the titles of "
        "VizieR catalogues that list them. These are candidate emission-line stars confirmed in the infrared by SPHEREx. SPHEREx's "
        "colour resolution is coarse, so it cannot tell whether the hydrogen glow comes from the star or from gas around it.")
    out = {k: v for k, v in a.items() if k != "stars"}
    out["stars_file"] = f"{P2}/analysis.json"
    json.dump(out, open(OUT + ".tmp", "w"), indent=1)
    os.replace(OUT + ".tmp", OUT)
    flat = dict(e23_interim=not a["sample_selection_complete"], e23_truncated=a["n_fields_measured"] < a["n_fields_selected"],
                e23_fields_not_measured=a["n_fields_selected"] - a["n_fields_measured"],
                e23_fields_selected=a["n_fields_selected"], e23_fields_measured=a["n_fields_measured"],
                e23_candidates_measured=a["n_candidates_measured"], e23_controls_measured=a["control_n"],
                e23_N_total=a["N_total"], e23_M_total=a["M_total"], e23_control_fp=a["control_fp"],
                e23_control_testable=a["control_testable"], e23_control_fp_rate=a["control_fp_rate"],
                e23_control_fp_rate_upper95=a["control_fp_rate_upper95"], e23_expected_fp=a["expected_fp"],
                e23_expected_fp_upper95=a["expected_fp_upper95"], e23_method_works=a["method_works"],
                e23_n_detected=a["n_detected"], e23_n_no_simbad_emission_type=a["n_no_simbad_emission_type"],
                e23_n_no_published_emission_classification_found=nf,
                e23_n_only_yso_catalogue_hits=a["n_only_yso_catalogue_hits"],
                e23_catalogues_checked=CHECKED, e23_search_date=SEARCH_DATE)
    for c in CLASSES:
        for k in ("N", "M", "n_measured", "expected_fp"):
            flat[f"e23_{k}_{c}"] = a.get(f"{k}_{c}")
    json.dump(flat, open("data/findings/e23s.json", "w"), indent=1)
    rows_csv = [dict(gaia_dr3_source_id=t["source_id"], ra=t["ra"], dec=t["dec"], gaia_espels_class=t["cls"], G=t["G"],
                     Paa_Z=t["Z_Paa"], Paa_detected=t["detected_Paa"], Paa_excess=t["frac_excess_Paa"],
                     Bra_Z=t["Z_Bra"], Bra_detected=t["detected_Bra"], Bra_excess=t["frac_excess_Bra"], Brg_Z=t["Z_Brg"],
                     simbad_main_id=t["simbad_main_id"], simbad_type=t["simbad_otype"],
                     simbad_other_emission_types=";".join(t["simbad_secondary_emission_types"]),
                     vsx_entries=";".join(f"{v.get('Name')} ({v.get('Type')})" for v in (t["vsx"] or [])),
                     vizier_emission_catalogues=";".join(h.split(":")[0] for h in t["vizier_emission_title_hits"]),
                     only_yso_catalogue_match=bool(t["vizier_emission_title_hits"] and not t["vizier_emission_specific_hits"]
                                                   and not t["simbad_secondary_emission_types"] and not t["vsx_emission_like"]),
                     no_published_emission_classification_found=t["no_published_emission_classification_found"],
                     checks_complete=t["extended_check_complete"], midir_excess_flag=t["midir_excess_flag"], W1_W3=t["W1_W3"],
                     catalogues_checked=CHECKED, search_date=SEARCH_DATE) for t in table]
    os.makedirs("data/products", exist_ok=True)
    df = pd.DataFrame(rows_csv)
    for c in ("Paa_Z", "Bra_Z", "Brg_Z", "W1_W3"):
        df[c] = df[c].astype(float).round(3)
    for c in ("Paa_excess", "Bra_excess"):
        df[c] = df[c].astype(float).round(4)
    df.to_csv(CSV, index=False)
    print("csv", CSV, len(rows_csv))
    print(json.dumps({k: v for k, v in out.items() if k != "candidate_table"}, indent=1)[:6000])


# ------------------------------------------------------------------ figure
def stage_fig():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    a = json.load(open(f"{P2}/analysis.json"))
    o = json.load(open(OUT))
    rows = a["stars"]
    BLUE, ORANGE, INK, MUTED, RED = E.BLUE, E.ORANGE, E.INK, E.MUTED, E.RED
    plt.rcParams.update({"font.size": 13, "font.family": "DejaVu Sans", "axes.edgecolor": MUTED, "text.color": INK})
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5.8), dpi=140, gridspec_kw=dict(width_ratios=[1.6, 1], wspace=0.28))
    bins = np.linspace(-8, 16, 49)
    for role, col in (("control", BLUE), ("candidate", ORANGE)):
        z = [max(r["lines"][L].get("Z", -99) for L in ("Paa", "Bra")) for r in rows if r["role"] == role and r["testable_main"]]
        z = np.clip(z, bins[0], bins[-1] - 1e-6)
        ax.hist(z, bins=bins, histtype="stepfilled" if role == "control" else "step", color=col, alpha=0.45 if role == "control" else 1,
                lw=2.2, weights=np.full(len(z), 1 / max(1, len(z))), label=f"{role}s (n = {len(z)})")
    ax.axvline(E.Z_DET, color=RED, lw=1.5, ls="--")
    ax.text(E.Z_DET + 0.2, ax.get_ylim()[1] * 0.55, "threshold Z = 5\n(+ ≥ 2 visits)", color=RED, fontsize=11, va="top")
    ax.set_xlabel("line excess Z (larger of Paα, Brα; calibrated on controls; clipped)")
    ax.set_ylabel("fraction of stars")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    cls = [c for c in CLASSES if o.get(f"M_{c}")] + ["controls"]
    frac, lo, hi, lab = [], [], [], []
    from scipy.stats import beta as B
    for c in cls:
        k, n = (o["control_fp"], o["control_testable"]) if c == "controls" else (o[f"N_{c}"], o[f"M_{c}"])
        f = k / n
        l_ = B.ppf(0.16, k, n - k + 1) if k > 0 else 0.0
        h_ = B.ppf(0.84, k + 1, n - k) if k < n else 1.0
        frac.append(f), lo.append(f - l_), hi.append(h_ - f)
        lab.append(f"{'controls' if c == 'controls' else CLS_NAME[c]}\n{k} of {n}")
    y = np.arange(len(cls))
    ax2.barh(y, frac, xerr=[lo, hi], color=[ORANGE] * (len(cls) - 1) + [BLUE], alpha=0.85, capsize=4)
    ax2.set_yticks(y, lab)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("fraction with Paα/Brα excess (68% interval)")
    ax2.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"SPHEREx hydrogen-line census: {o['N_total']} of {o['M_total']} Gaia ESP-ELS candidates; "
                 f"{o['control_fp']} of {o['control_testable']} controls", x=0.06, ha="left", fontsize=15, fontweight="bold")
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, facecolor="white", bbox_inches="tight")
    fig.savefig(FIG[:-4] + ".svg", facecolor="white", bbox_inches="tight")
    print("figure", FIG, FIG[:-4] + ".svg")


if __name__ == "__main__":
    os.makedirs(P2, exist_ok=True)
    {"select": stage_select, "measure": stage_measure, "analyze": stage_analyze, "simbad": stage_simbad, "lit": stage_lit,
     "final": stage_final, "fig": stage_fig}[sys.argv[1]]()
