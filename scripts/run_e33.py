"""E33: vetting the NEP variable candidates (docs/protocols/E33.md). Run from the repo root, one process:
  .venv/bin/python scripts/run_e33.py prep      # sets T, C, F; photometry of those sources -> data/phase2/e33/phot.parquet
  .venv/bin/python scripts/run_e33.py check     # reproduce E26 after statistics of 5 C stars
  .venv/bin/python scripts/run_e33.py halves    # rule (d): split-half statistics of T, C, F, known
  .venv/bin/python scripts/run_e33.py vetoes    # rule (f): E18-B V1-V4 on T and F
  .venv/bin/python scripts/run_e33.py cats      # rule (b) catalogues: 2MASS XSC, HyperLEDA, NED nebulae
  .venv/bin/python scripts/run_e33.py analyze   # apply rules, numbers -> data/phase2/e33/vet.json
"""
import os, sys, time, json, glob, re

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_e26 as E

B = E.B
D = "data/phase2/e33"
CSV = "data/products/epoch_nep_variability.csv"
STATS = B.STATS
SEED_C, N_C = 2026111533, 400
T_SPLIT = 60982.17
SPLITS = {"D1": lambda d: d["det"] <= 3, "D2": lambda d: d["det"] >= 4,
          "T1": lambda d: d["t"] < T_SPLIT, "T2": lambda d: d["t"] >= T_SPLIT}
CM_P, CM_TOL, BAND = (79.36, 95.06), 0.05, (60.0, 120.0)
NGC6543 = (269.6392, 66.6331, 3.5)
MARGIN = 18.0
BLOOM_W1, BLOOM_G, BLOOM_R = 8.0, 9.0, 60.0
PIX, GROUP_R = 6.2, 60.0


def table():
    t = pd.read_csv(CSV)
    t["catalogued_var"] = t.catalogued_var.fillna(False).astype(bool)
    return t


def z99(t):
    cal = t[t.role == "cal"]
    return {s: float(np.nanquantile(cal[f"z_{s}"], 0.99)) for s in STATS}


def sets():
    t = table()
    pop = B.populations()
    idx = dict(zip(pop.id.astype(str), pop.idx))
    t["src"] = t.gaia_or_allwise_id.astype(str).map(idx)
    q = z99(t)
    t["any99"] = np.any([t[f"z_{s}"] > q[s] for s in STATS], axis=0)
    T = t[t.variable_candidate & ~t.catalogued_var & t.milliquas_name.isna()]
    K = t[t.variable_candidate & ~t.src.isin(T.src)]
    F = t[(t.role == "test") & t.any99]
    cal = np.sort(t.src[t.role == "cal"].values.astype(int))
    C = np.random.default_rng(SEED_C).choice(cal, N_C, replace=False)
    return t, dict(T=sorted(T.src.astype(int)), K=sorted(K.src.astype(int)), F=sorted(F.src.astype(int)),
                   C=sorted(int(i) for i in C)), q


# ---------------------------------------------------------------- photometry subset (same rows as run_e18b.prep)
def prep():
    os.makedirs(D, exist_ok=True)
    t, S, q = sets()
    json.dump(dict(S, z99=q, time=time.strftime("%H:%M:%S")), open(f"{D}/sets.json", "w"), indent=1)
    want = set(S["T"]) | set(S["K"]) | set(S["F"]) | set(S["C"])
    print({k: len(v) for k, v in S.items()}, "sources", len(want), flush=True)
    s = pd.read_parquet(f"{B.E18}/sources.parquet")
    sc = s.set_index("idx")
    bad = B.anomaly_set()
    RA0, DEC0 = B.RA0, B.DEC0
    cols = ["src", "x", "y", "n_masked_box", "flags_core", "flux_mJy", "flux_err_mJy", "wl_um", "chi2dof_local", "mjd",
            "detector", "obs_id", "edge_px"]
    keep, npic = [], 0
    parts = sorted(glob.glob(f"{B.E18}/phot/part-*.parquet"))
    t0 = time.time()
    for n, p in enumerate(parts):
        q_ = pd.read_parquet(p, columns=cols)
        key = q_.obs_id.astype(str) + "_" + q_.detector.astype(str)
        uk, pic = np.unique(key.values, return_inverse=True)
        q_["pic"] = (pic + npic).astype(np.int32)
        npic += len(uk)
        ra, de = sc.ra.reindex(q_.src.values).values, sc.dec.reindex(q_.src.values).values
        xi = np.cos(np.radians(de)) * np.sin(np.radians(ra - RA0))
        eta = np.sin(np.radians(de)) * np.cos(np.radians(DEC0)) - np.cos(np.radians(de)) * np.sin(np.radians(DEC0)) * np.cos(np.radians(ra - RA0))
        g = pd.DataFrame(dict(k=q_.pic.values, x=q_.x.values, y=q_.y.values, xi=xi, eta=eta))
        for c in ("x", "y", "xi", "eta"):
            g[c] = g[c] - g.groupby("k")[c].transform("mean")
        g["c"] = g.xi * g.x + g.eta * g.y
        g["s"] = g.xi * g.y - g.eta * g.x
        ang = g.groupby("k")[["c", "s"]].sum()
        q_["pa"] = (np.degrees(np.arctan2(ang.s, ang.c)) % 360).reindex(q_.pic.values).values.astype(np.float32)
        q_ = q_[q_.src.isin(want)]
        ok = np.isfinite(q_.flux_mJy) & np.isfinite(q_.flux_err_mJy) & (q_.flux_err_mJy > 0)
        ok &= q_.n_masked_box <= B.MAX_MASKED
        ok &= q_.edge_px >= B.MIN_EDGE
        ok &= B.flags_ok(q_.flags_core.values)
        ok &= ~np.array([(o, int(d)) in bad for o, d in zip(q_.obs_id.values, q_.detector.values)], bool)
        q_ = q_[ok].drop(columns=["obs_id", "edge_px", "n_masked_box", "flags_core"])
        keep.append(q_.astype(dict(src=np.int32, x=np.float32, y=np.float32, detector=np.int8)))
        if n % 200 == 0:
            print(f"{time.strftime('%H:%M:%S')} part {n}/{len(parts)} {time.time() - t0:.0f}s", flush=True)
    q_ = pd.concat(keep, ignore_index=True)
    med = q_.groupby("src").chi2dof_local.transform("median")
    q_ = q_[q_.chi2dof_local <= np.maximum(B.CHI2LOC_MIN, B.CHI2LOC_FAC * med)]
    q_ = q_.sort_values(["src", "mjd"]).reset_index(drop=True)
    q_.to_parquet(f"{D}/phot.parquet", index=False, row_group_size=200000)
    print("prep done", len(q_), "rows", npic, "pictures", time.strftime("%H:%M:%S"), flush=True)


# ---------------------------------------------------------------- corrected data (exactly as run_e26._work, mode "after")
B.D = D                                            # run_e18b.load_src reads f"{B.D}/phot.parquet"
_raw = B.load_src
_CTX = {}


def ctx():
    if not _CTX:
        pop = B.populations()
        _CTX.update(pop=pop, mi=E.model_index(pop), C=E.load_correction(),
                    g0=json.load(open(f"{E.D}/gate0_after.json")), g00=json.load(open(E.G0_E18B)))
    return _CTX


FOLD_MAP = f"{D}/fold_map.json"


def load_corrected(srcs, fold=None):
    """Corrected data; fold per source from fold_map.json (the fold that reproduces the stored E26 after statistic)
    when it exists, else run_e26.model_index."""
    c = ctx()
    s0, f0 = E.g0_dicts(c["g00"])
    if fold is None and os.path.exists(FOLD_MAP):
        fold = {int(k): v["fold"] for k, v in json.load(open(FOLD_MAP)).items()}
    out = {}
    for i, d in _raw(srcs).items():
        d = B.drop_thin(d)
        if len(d["f"]) >= B.MIN_PICS:
            k = fold.get(int(i), c["mi"][int(i)]) if fold else c["mi"][int(i)]
            out[i] = E.correct(d, *c["C"][k], s0, f0)
    return out


def full_stats(d, s_d, f_d):
    freq = np.arange(B.FMIN, B.FMAX, 1 / (B.OVERS * (d["t"].max() - d["t"].min())))
    allowed, _ = B.alias_mask(freq, d["t"])
    return B.source_stats(d, s_d, f_d, freq, allowed)


def check():
    S = json.load(open(f"{D}/sets.json"))
    s_d, f_d = E.g0_dicts(ctx()["g0"])
    ref = E._read("after", "real").set_index("src")
    out = []
    for i, d in load_corrected(S["C"][:5]).items():
        r = full_stats(d, s_d, f_d)
        out.append({st: (r[st], float(ref.loc[i, st])) for st in STATS + ["P_best"]})
        print(i, out[-1], flush=True)
    dmax = max(abs(a - b) / max(abs(b), 1e-12) for o in out for a, b in o.values())
    json.dump(dict(n=len(out), max_rel_diff=dmax, rows=out), open(f"{D}/check.json", "w"), indent=1)
    print("max relative difference", dmax, flush=True)


def halves():
    S = json.load(open(f"{D}/sets.json"))
    s_d, f_d = E.g0_dicts(ctx()["g0"])
    srcs = sorted(set(S["T"]) | set(S["K"]) | set(S["F"]) | set(S["C"]))
    rows, t0 = [], time.time()
    ref = E._read("after", "real").set_index("src")
    c = ctx()
    s0, f0 = E.g0_dicts(c["g00"])
    fmap = {}
    for n in range(0, len(srcs), 25):
        raw = _raw(srcs[n:n + 25])
        corr = {}
        for i, d0 in raw.items():
            d0 = B.drop_thin(d0)
            k0 = c["mi"][int(i)]
            for k in (k0, 1 - k0):
                d = E.correct(d0, *c["C"][k], s0, f0)
                r0 = full_stats(d, s_d, f_d)
                rel = abs(r0["chi2r"] / ref.loc[i, "chi2r"] - 1)
                if rel < 1e-9:
                    break
            fmap[int(i)] = dict(fold=int(k), model_index=int(k0), match=bool(rel < 1e-9), rel=float(rel))
            if rel >= 1e-9:
                fmap[int(i)]["fold"] = int(k0)
                d = E.correct(d0, *c["C"][k0], s0, f0)
            corr[i] = d
        json.dump(fmap, open(FOLD_MAP, "w"))
        for i, d in corr.items():
            for h, sel in SPLITS.items():
                m = sel(d)
                dh = B.drop_thin({k: v[m] for k, v in d.items()})
                r = dict(src=int(i), half=h, n=int(len(dh["f"])))
                if len(dh["f"]) >= B.MIN_PICS:
                    rr = B.source_stats(dh, s_d, f_d)
                    r.update({k: rr[k] for k in STATS + ["m_ab", "P_best"]})
                rows.append(r)
        pd.DataFrame(rows).to_parquet(f"{D}/halves.parquet", index=False)
        print(f"{time.strftime('%H:%M:%S')} {min(n + 25, len(srcs))}/{len(srcs)} {time.time() - t0:.0f}s", flush=True)


def after_thresholds():
    pop = ctx()["pop"]
    real, shuf = E._read("after", "real"), E._read("after", "shuf")
    real = real.merge(pop[["idx", "half"]].rename(columns={"idx": "src"}), on="src")
    norms, thr, sub = B.calibrate(real, shuf, pop)
    return norms, thr, sub


def vetoes():
    S = json.load(open(f"{D}/sets.json"))
    c = ctx()
    norms, thr, sub = after_thresholds()
    thr99 = {s: dict(Z=v) for s, v in S["z99"].items()}
    bxy = pd.read_parquet(f"{E.D}/bright_xy.parquet")
    B.load_src = load_corrected
    out = {}
    for grp, th in (("T", thr), ("F", thr99)):
        for i in S[grp]:
            res, base, _ = B.vetoes(i, c["pop"], norms, th, sub, c["g0"], bxy)
            out[str(i)] = dict(group=grp, **{k: res[k] for k in ("V1_crosstalk", "V2_persistence", "V3_airglow", "V4_leave_one_out")},
                               z_base=res["z"], P_best=res["P_best"])
            json.dump(out, open(f"{D}/vetoes.json", "w"), indent=1, default=float)
            print(time.strftime("%H:%M:%S"), grp, i, [out[str(i)][k]["passed"] for k in ("V1_crosstalk", "V2_persistence", "V3_airglow", "V4_leave_one_out")], flush=True)
    B.load_src = _raw


# ---------------------------------------------------------------- rule (b) catalogues
def cats():
    from run_e10 import vizier_cone
    import requests
    out = {}
    def norm(v):
        if v is None:
            return None
        v = v.rename(columns={c: ("RAJ2000" if c.startswith("_RA") else "DEJ2000") for c in v.columns if c.startswith(("_RA", "_DE"))})
        return v.to_dict("records")
    out["xsc"] = norm(vizier_cone("VII/233/xsc", B.RA0, B.DEC0, 0.5, "_RAJ2000,_DEJ2000,2MASX,r.ext"))
    out["pgc"] = norm(vizier_cone("VII/237/pgc", B.RA0, B.DEC0, 0.5, "_RAJ2000,_DEJ2000,PGC,logD25"))
    try:
        r = requests.get("https://ned.ipac.caltech.edu/cgi-bin/objsearch",
                         params={"search_type": "Near Position Search", "in_csys": "Equatorial", "in_equinox": "J2000.0",
                                 "lon": f"{B.RA0}d", "lat": f"{B.DEC0}d", "radius": "30", "of": "ascii_tab",
                                 "out_csys": "Equatorial", "out_equinox": "J2000.0", "obj_sort": "Distance to search center",
                                 "list_limit": "0", "img_stamp": "NO"}, timeout=(10, 120))
        r.raise_for_status()
        lines = r.text.splitlines()
        h = next(k for k, ln in enumerate(lines) if ln.startswith("No.") and "Type" in ln)
        cols = [c.strip() for c in lines[h].split("\t")]
        rows = [dict(zip(cols, [c.strip() for c in ln.split("\t")])) for ln in lines[h + 1:] if ln.count("\t") >= len(cols) - 1]
        neb = [r_ for r_ in rows if r_.get("Type", "") in ("PN", "HII", "RfN", "EmN", "SNR", "Neb", "!PN", "!HII")]
        out["ned"] = dict(n_objects=len(rows), nebulae=neb)
    except Exception as e:
        out["ned"] = dict(error=repr(e)[:200])
    out["time"] = time.strftime("%H:%M:%S")
    json.dump(out, open(f"{D}/cats.json", "w"), indent=1)
    print({k: (len(v) if isinstance(v, list) else v) for k, v in out.items() if k != "ned"}, out["ned"].get("n_objects"),
          out["ned"].get("error"), len(out["ned"].get("nebulae", [])), flush=True)


def sep_arcsec(ra1, de1, ra2, de2):
    ra1, de1, ra2, de2 = map(np.radians, (ra1, de1, ra2, de2))
    c = np.sin(de1) * np.sin(de2) + np.cos(de1) * np.cos(de2) * np.cos(ra1 - ra2)
    return np.degrees(np.arccos(np.clip(c, -1, 1))) * 3600


# ---------------------------------------------------------------- apply the rules
def wilson(k, n):
    return B.wilson(k, n)


def analyze():
    t, S, q = sets()
    t = t.set_index("src", drop=False)
    thr = {s: float(v["Z"]) for s, v in after_thresholds()[1].items() if s in STATS}
    cats_ = json.load(open(f"{D}/cats.json"))
    src = pd.read_parquet(f"{B.E18}/sources.parquet")
    H = pd.read_parquet(f"{D}/halves.parquet")
    V = json.load(open(f"{D}/vetoes.json"))
    # rule (d) thresholds on C
    hz = {}
    Hc = H[H.src.isin(S["C"])]
    for h in SPLITS:
        for st in STATS:
            g = Hc[(Hc.half == h)].dropna(subset=[st])
            n_ = B.Norm(g.m_ab.values, g[st].values)
            hz[(h, st)] = (n_, float(np.nanquantile(n_.z(g.m_ab.values, g[st].values), 0.99)))
    Hp = {}
    for (i, h), g in H.groupby(["src", "half"]):
        r = g.iloc[0]
        ok = False
        if np.isfinite(r.get("chi2r", np.nan)):
            ok = any(float(hz[(h, st)][0].z(np.array([r.m_ab]), np.array([r[st]]))[0]) > hz[(h, st)][1] for st in STATS)
        Hp[(int(i), h)] = bool(ok)
    # extended sources
    ext = [("NGC 6543", NGC6543[0], NGC6543[1], NGC6543[2] * 60)]
    for r in cats_.get("xsc") or []:
        ext.append((r["2MASX"], float(r["RAJ2000"]), float(r["DEJ2000"]), float(r["r.ext"] or 0) + MARGIN))
    for r in cats_.get("pgc") or []:
        if r.get("logD25"):
            ext.append((f"PGC {r['PGC']}", float(r["RAJ2000"]), float(r["DEJ2000"]), 0.5 * 6 * 10 ** float(r["logD25"]) + MARGIN))
    for r in cats_.get("ned", {}).get("nebulae", []):
        if "6543" not in r.get("Object Name", ""):
            ext.append((r.get("Object Name"), float(r["RA"]), float(r["DEC"]), 60 + MARGIN))
    bright = src[(src.W1mag < BLOOM_W1) | (src.G < BLOOM_G)]
    above_all = t[t.variable_candidate]

    def rules(i, loose):
        r = t.loc[i]
        zt = q if loose else thr
        hits = {st: bool(r[f"z_{st}"] > zt[st]) for st in STATS}
        a1 = any(abs(r.P_best - p) < CM_TOL for p in CM_P)
        a2 = hits["ls"] and not any(hits[s] for s in STATS if s != "ls") and BAND[0] <= r.P_best <= BAND[1]
        out = dict(a=not (a1 or a2), a1_shared_period=bool(a1), a2_ls_only_band=bool(a2))
        hit_ext = [(nm, round(float(s), 1)) for nm, ra, de, rad in ext
                   for s in [sep_arcsec(r.ra, r.dec, ra, de)] if s < rad]
        bs = sep_arcsec(r.ra, r.dec, bright.ra.values, bright.dec.values)
        bs = bs[bs > 0.5]
        b3 = bool((bs < BLOOM_R).any())
        out.update(b=not (hit_ext or b3), b12_extended=hit_ext, b3_bright_star_arcsec=round(float(bs.min()), 1) if len(bs) else None)
        ss = sep_arcsec(r.ra, r.dec, t.ra.values, t.dec.values)
        nb = t[(ss < PIX) & (ss > 0.5) & (t.spherex_mean_m_ab.values < r.spherex_mean_m_ab)]
        nb_pass = nb.any99 if loose else nb.variable_candidate
        c1 = bool(nb_pass.any())
        sa = sep_arcsec(r.ra, r.dec, above_all.ra.values, above_all.dec.values)
        n_grp = int(((sa < GROUP_R) & (sa > 0.5)).sum())
        out.update(c=not (c1 or n_grp >= 2), c1_brighter_passing_neighbour=c1, c2_n_other_above_within_1arcmin=n_grp)
        dp = {h: Hp.get((int(i), h)) for h in SPLITS}
        out.update(d=bool((dp["D1"] and dp["D2"]) or (dp["T1"] and dp["T2"])), d_halves=dp)
        v = V.get(str(int(i)))
        if v is not None:
            out.update(f=all(v[k]["passed"] for k in ("V1_crosstalk", "V2_persistence", "V3_airglow", "V4_leave_one_out")),
                       f_detail={k: v[k]["passed"] for k in ("V1_crosstalk", "V2_persistence", "V3_airglow", "V4_leave_one_out")})
        out["survives"] = bool(all(out[k] for k in "abcdf" if k in out) and "f" in out)
        return out

    res_T = {int(i): rules(i, False) for i in S["T"]}
    res_F = {int(i): rules(i, True) for i in S["F"]}
    test = t[t.role == "test"]
    unc = {}
    for k in "abc":
        rr = [rules(i, False)[k] for i in test.src.astype(int)] if k != "c" else None
        if rr is not None:
            unc[k] = dict(rejected=int(len(rr) - sum(rr)), n=len(rr))
    rc = [rules(i, False)["c"] for i in test.src.astype(int)]
    unc["c"] = dict(rejected=int(len(rc) - sum(rc)), n=len(rc))
    for v in unc.values():
        v["rate"] = round(v["rejected"] / v["n"], 4)
        v["wilson95"] = wilson(v["rejected"], v["n"])
    nF = len(res_F)
    per_rule_F = {k: dict(passed=int(sum(r[k] for r in res_F.values())), n=nF) for k in "abcdf"}
    kF = int(sum(r["survives"] for r in res_F.values()))
    pv = kF / nF
    k_far = int(test.variable_candidate.sum())
    n_tested = int((~t.catalogued_var & t.milliquas_name.isna()).sum())
    far, pvw = wilson(k_far, len(test)), wilson(kF, nF)
    exp = dict(value=k_far / len(test) * n_tested * pv, approx95=[far[0] * n_tested * pvw[0], far[1] * n_tested * pvw[1]])
    out = dict(sets={k: len(v) for k, v in S.items()}, thresholds=thr, z99=q,
               half_thresholds={f"{h}_{st}": v[1] for (h, st), v in hz.items()},
               n_extended_objects=len(ext), extended_objects=ext, T=res_T, F=res_F, unconditional_test_null=unc,
               F_per_rule=per_rule_F, F_survivors=kF, p_v=pv, p_v_wilson95=pvw,
               test_far=dict(k=k_far, n=len(test), wilson95=far), n_tested=n_tested, expected_false_survivors=exp,
               survivors=[i for i, r in res_T.items() if r["survives"]], time=time.strftime("%H:%M:%S"))
    json.dump(out, open(f"{D}/vet.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({k: out[k] for k in ("sets", "F_per_rule", "F_survivors", "p_v", "test_far", "n_tested",
                                          "expected_false_survivors", "survivors", "unconditional_test_null")}, default=str), flush=True)
    for i, r in res_T.items():
        print(i, {k: r[k] for k in r if len(k) == 1 or k == "survives"}, flush=True)


# ---------------------------------------------------------------- rule (e) literature, figures, outputs
VAR_RE = re.compile(r"variab|periodic|light.?curve|eclips|RR.?Lyr|cephe|flare|Mira|rotation", re.I)
SIMBAD_VAR = {"V*", "Ir*", "Er*", "Ro*", "BY*", "RS*", "EB*", "Pu*", "RR*", "Ce*", "dS*", "LP*", "Mi*", "sv*", "WU*",
              "El*", "Fl*", "Or*", "gD*", "SX*", "bC*", "cC*", "C*", "RI*", "a2*", "Pe*", "CV*", "No*", "XB*"}
EXPLICIT = {"B/vsx/vsx": "VSX", "II/366/catalog": "ASAS-SN variables", "J/ApJS/249/18/table2": "ZTF periodic variables",
            "J/AJ/156/241/table4": "ATLAS variables", "J/ApJS/237/28/table1": "WISE variables",
            "I/358/vclassre": "Gaia DR3 vari classification"}
R_LIT = 5.0


def simbad(ra, dec):
    import requests
    q = (f"SELECT b.main_id, b.otype, o.otype AS ot, DISTANCE(POINT('ICRS', b.ra, b.dec), POINT('ICRS', {ra}, {dec})) * 3600 AS sep "
         f"FROM basic AS b JOIN otypes AS o ON o.oidref = b.oid WHERE CONTAINS(POINT('ICRS', b.ra, b.dec), "
         f"CIRCLE('ICRS', {ra}, {dec}, {R_LIT / 3600})) = 1")
    for host in ("https://simbad.cds.unistra.fr", "https://simbad.cfa.harvard.edu"):
        try:
            r = requests.get(f"{host}/simbad/sim-tap/sync", params=dict(request="doQuery", lang="adql", format="json", query=q),
                             timeout=(10, 60))
            r.raise_for_status()
            j = r.json()
            cols = [c["name"] for c in j["metadata"]]
            return dict(host=host, rows=[dict(zip(cols, row)) for row in j["data"]])
        except Exception as e:
            err = repr(e)[:150]
    return dict(error=err)


def vizier_all(ra, dec):
    """All VizieR catalogues within R_LIT (VOTable); keeps tables whose description matches VAR_RE or is listed."""
    import requests, io
    from astropy.io.votable import parse
    for host in ("https://vizier.cds.unistra.fr", "https://vizier.cfa.harvard.edu"):
        try:
            r = requests.get(f"{host}/viz-bin/votable", params={"-c": f"{ra:.6f} {dec:+.6f}", "-c.rs": f"{R_LIT}",
                                                                 "-out.max": "20", "-out": "_r"}, timeout=(10, 180))
            r.raise_for_status()
            vt = parse(io.BytesIO(r.content), verify="ignore")
            tabs = []
            for res in vt.resources:
                desc = str(res.description or "")
                for tb in res.tables:
                    tabs.append(dict(name=str(tb.name), resource=str(res.name), description=desc[:200],
                                     table_description=str(tb.description or "")[:200], n=int(len(tb.array))))
            pref = [k.rsplit("/", 1)[0] + "/" for k in EXPLICIT]
            hits = [x for x in tabs if VAR_RE.search(x["description"] + " " + x["table_description"])
                    or any(x["name"].startswith(p_) for p_ in pref)]
            return dict(host=host, n_tables=len(tabs), tables=[x["name"] for x in tabs], variability_tables=hits)
        except Exception as e:
            err = repr(e)[:150]
    return dict(error=err)


def explicit(ra, dec):
    from run_e10 import vizier_cone
    out = {}
    for cat, nm in EXPLICIT.items():
        v = vizier_cone(cat, ra, dec, R_LIT / 3600, "_r")
        out[nm] = "unreachable" if v is None else int(len(v))
    return out


def lit():
    vet = json.load(open(f"{D}/vet.json"))
    t = table()
    pop = B.populations()
    t["src"] = t.gaia_or_allwise_id.astype(str).map(dict(zip(pop.id.astype(str), pop.idx)))
    out = {}
    for i in vet["survivors"]:
        r = t[t.src == i].iloc[0]
        sb, vz, ex = simbad(r.ra, r.dec), vizier_all(r.ra, r.dec), explicit(r.ra, r.dec)
        ots = {x["ot"].rstrip("?") for x in sb.get("rows", [])} | {str(x["otype"]).rstrip("?") for x in sb.get("rows", [])}
        known = bool((ots & SIMBAD_VAR) or vz.get("variability_tables") or any(isinstance(v, int) and v > 0 for v in ex.values()))
        out[str(i)] = dict(simbad=sb, vizier=vz, explicit=ex, simbad_types=sorted(ots),
                           status="known" if known else "we found no published variability", time=time.strftime("%H:%M:%S"))
        json.dump(out, open(f"{D}/lit.json", "w"), indent=1, default=str)
        print(i, out[str(i)]["status"], sorted(ots), [x["name"] for x in vz.get("variability_tables", [])], ex, vz.get("error"), sb.get("error"), flush=True)


def figs():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from e2_cache import fast_stamp
    vet = json.load(open(f"{D}/vet.json"))
    t = table()
    pop = B.populations()
    t["src"] = t.gaia_or_allwise_id.astype(str).map(dict(zip(pop.id.astype(str), pop.idx)))
    s_d, f_d = E.g0_dicts(ctx()["g0"])
    fr = pd.read_parquet(f"{B.E18}/frames.parquet")
    fr = fr[(fr.tier == 0) & (fr.detector == 2)]
    cols = {0: "#0072B2", 1: "#D55E00"}
    for i in vet["survivors"]:
        r = t[t.src == i].iloc[0]
        d = load_corrected([i])[i]
        M, sig, _ = B.static_fit(d["f"], d["e"], d["lam"], d["det"], s_d, f_d)
        v = M > 0
        tt, a, w, g = d["t"][v], (d["f"][v] - M[v]) / M[v], (M[v] / sig[v]) ** 2, (d["det"][v] >= 4).astype(int)
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
        for k in (0, 1):
            m = g == k
            ab, sb, ub = B.binned(tt[m], a[m], w[m], d["t"].min())
            ax[0].errorbar(d["t"].min() + ub + 0.5 - 60800, ab, sb, fmt=".", ms=3, lw=0.5, color=cols[k])
            ph = (tt[m] / r.P_best) % 1
            kb = np.floor(ph * 20).astype(int)
            sw = np.bincount(kb, w[m], 20)
            mu = np.bincount(kb, w[m] * a[m], 20) / np.where(sw > 0, sw, np.nan)
            for off in (0, 1):
                ax[1].errorbar((np.arange(20) + 0.5) / 20 + off, mu, 1 / np.sqrt(np.where(sw > 0, sw, np.nan)), fmt="o", ms=3, color=cols[k])
        ax[0].set_xlabel("MJD - 60800 (1-day bins)")
        ax[0].set_ylabel("flux / static spectrum - 1 (blue D1-3, red D4-6)")
        ax[1].set_xlabel(f"phase at P_best = {r.P_best:.4g} d")
        ax[1].set_ylabel("flux / static spectrum - 1")
        fr["dt"] = np.abs(fr.mjd - np.median(d["t"]))
        st = None
        for key in fr.sort_values("dt").key.values[:8]:
            try:
                st = fast_stamp(key, r.ra, r.dec, half=15)
            except Exception:
                st = None
            if st is not None and st.image.shape == (31, 31) and 20 <= st.x0 + 15 <= 2019 and 20 <= st.y0 + 15 <= 2019:
                break
            st = None
        if st is not None:
            im = st.image - np.nanmedian(st.image)
            lim = np.nanpercentile(np.abs(im), 99.5)
            ax[2].imshow(im, origin="lower", cmap="gray", vmin=-0.2 * lim, vmax=lim,
                         extent=(st.x0 - 0.5, st.x0 + 30.5, st.y0 - 0.5, st.y0 + 30.5))
            ax[2].set_xlabel(f"x (pixel, detector 2, MJD {st.mjd:.1f})")
            ax[2].set_ylabel("y (pixel)")
        fig.tight_layout()
        fig.savefig(f"data/findings/e33_{r.gaia_or_allwise_id}.png", dpi=110)
        plt.close(fig)
        print("fig", i, r.gaia_or_allwise_id, st is not None, flush=True)


def _reasons(r):
    out = []
    if r["a1_shared_period"]: out.append("shared 79.36/95.06 d period")
    if r["a2_ls_only_band"]: out.append("LS-only trigger, P_best in 60-120 d")
    if r["b12_extended"]: out.append("extended source: " + "; ".join(f"{n} ({s}\")" for n, s in r["b12_extended"]))
    if r["b3_bright_star_arcsec"] is not None and r["b3_bright_star_arcsec"] < BLOOM_R: out.append(f"bright star {r['b3_bright_star_arcsec']}\"")
    if r["c1_brighter_passing_neighbour"]: out.append("brighter passing neighbour < 6.2\"")
    if r["c2_n_other_above_within_1arcmin"] >= 2: out.append(f"group: {r['c2_n_other_above_within_1arcmin']} other above-threshold sources < 1'")
    if not r["d"]: out.append("fails split halves")
    if not r.get("f", True): out.append("E18-B artefact veto " + ",".join(k for k, v in r["f_detail"].items() if not v))
    return out


def report():
    vet = json.load(open(f"{D}/vet.json"))
    litj = json.load(open(f"{D}/lit.json")) if os.path.exists(f"{D}/lit.json") else {}
    fmap = json.load(open(FOLD_MAP))
    t, S, q = sets()
    t = t.set_index("src", drop=False)
    rows = []
    for i in S["T"]:
        r, v = t.loc[i], vet["T"][str(i)]
        L = litj.get(str(i), {})
        rows.append(dict(gaia_or_allwise_id=r.gaia_or_allwise_id, ra=r.ra, dec=r.dec, spherex_mean_m_ab=r.spherex_mean_m_ab,
                         G=r.G, W1mag=r.W1mag, n_pictures=r.n_pictures, chi2r=r.chi2r, chi2r_det=r.chi2r_det, s2=r.s2, ls=r.ls,
                         z_chi2r=r.z_chi2r, z_chi2r_det=r.z_chi2r_det, z_s2=r.z_s2, z_ls=r.z_ls, amp=r.amp, P_best=r.P_best,
                         pass_a_period=v["a"], pass_b_extended=v["b"], pass_c_neighbour=v["c"], pass_d_halves=v["d"],
                         pass_f_artefact=v.get("f"), vetted_candidate=v["survives"], veto_reasons="; ".join(_reasons(v)),
                         literature=L.get("status", "not checked (vetoed)" if not v["survives"] else "not checked"),
                         simbad_types=",".join(L.get("simbad_types", []))))
    c = pd.DataFrame(rows).sort_values(["vetted_candidate", "ra"], ascending=[False, True])
    for col in c.select_dtypes("float").columns:
        c[col] = c[col].round(6)
    c.to_csv("data/products/epoch_nep_variable_candidates.csv", index=False)
    surv = c[c.vetted_candidate].to_dict("records")
    known = t.loc[S["K"], ["gaia_or_allwise_id", "ra", "dec", "spherex_mean_m_ab", "P_best", "vari_class", "kim2026_name",
                           "milliquas_name"]].to_dict("records")
    nv = {k: int(sum(not vet["T"][str(i)][k] for i in S["T"])) for k in "abcdf"}
    ex = vet["expected_false_survivors"]
    out = dict(experiment="E33", protocol="docs/protocols/E33.md",
               verdict=f"{len(surv)} of {len(S['T'])} uncatalogued above-threshold sources survive the frozen vetting; "
                       f"expected false survivors {ex['value']:.2f} (approx. 95% {ex['approx95'][0]:.2f}-{ex['approx95'][1]:.2f})",
               n_tested=len(S["T"]), n_survivors=len(surv), vetoed_by_rule=nv, survivors=surv, tested=c.to_dict("records"),
               known_above_threshold=known, expected_false_survivors=ex, test_false_alarm=vet["test_far"],
               n_noncatalogued_sources=vet["n_tested"], p_v=vet["p_v"], p_v_wilson95=vet["p_v_wilson95"],
               F_size=len(S["F"]), F_survivors=vet["F_survivors"], F_per_rule=vet["F_per_rule"],
               unconditional_rejection_test_null=vet["unconditional_test_null"], thresholds=vet["thresholds"], z99=vet["z99"],
               half_thresholds=vet["half_thresholds"], extended_objects=vet["extended_objects"],
               fold_map=dict(n=len(fmap), n_matched=int(sum(v["match"] for v in fmap.values())),
                             n_differs_from_model_index=int(sum(v["fold"] != v["model_index"] for v in fmap.values()))),
               literature=litj, figures=[f"data/findings/e33_{r['gaia_or_allwise_id']}.png" for r in surv], time=time.strftime("%H:%M:%S"))
    json.dump(out, open("data/findings/e33.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    f = dict(n_tested=len(S["T"]), n_survivors=len(surv), n_known_above=len(S["K"]),
             expected_false_survivors=round(ex["value"], 3), expected_false_lo=round(ex["approx95"][0], 3),
             expected_false_hi=round(ex["approx95"][1], 3), test_far_k=vet["test_far"]["k"], test_far_n=vet["test_far"]["n"],
             test_far_lo=vet["test_far"]["wilson95"][0], test_far_hi=vet["test_far"]["wilson95"][1],
             n_noncatalogued_sources=vet["n_tested"], p_v=round(vet["p_v"], 4), p_v_lo=vet["p_v_wilson95"][0],
             p_v_hi=vet["p_v_wilson95"][1], n_F=len(S["F"]), F_survivors=vet["F_survivors"],
             **{f"vetoed_{k}": v for k, v in nv.items()},
             **{f"F_pass_{k}": v["passed"] for k, v in vet["F_per_rule"].items()},
             **{f"null_reject_rate_{k}": v["rate"] for k, v in vet["unconditional_test_null"].items()})
    for n, r in enumerate(surv, 1):
        f.update({f"survivor{n}_id": str(r["gaia_or_allwise_id"]), f"survivor{n}_ra": r["ra"], f"survivor{n}_dec": r["dec"],
                  f"survivor{n}_mab": r["spherex_mean_m_ab"], f"survivor{n}_pbest": r["P_best"], f"survivor{n}_lit": r["literature"]})
    assert all(re.fullmatch(r"[A-Za-z0-9_]+", k) for k in f)
    json.dump(f, open("data/findings/e33s.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps(f, indent=1, default=str), flush=True)


if __name__ == "__main__":
    {"prep": prep, "check": check, "halves": halves, "vetoes": vetoes, "cats": cats, "analyze": analyze,
     "lit": lit, "figs": figs, "report": report}[sys.argv[1]]()
