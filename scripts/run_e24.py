"""E24: predefined replication of the E18-B AGN population test in the south ecliptic pole deep field
(docs/protocols/E24.md). Reuses scripts/run_e18.py (extraction) and scripts/run_e18b.py (Gate-0, statistics) with the
field constants re-pointed. Run from the repo root:
  .venv/bin/python scripts/run_e24.py catalogs   # Gaia / AllWISE / Milliquas / VSX pulls -> data/phase2/e24/*_raw.parquet
  .venv/bin/python scripts/run_e24.py catalogue  # sources.parquet (E18 rule) + Kim / Milliquas / star columns
  .venv/bin/python scripts/run_e24.py frames     # candidate pictures, tiers (E18 rule)
  .venv/bin/python scripts/run_e24.py phot       # photometry (E18 machinery), no new picture after CUTOFF
  .venv/bin/python scripts/run_e24.py prep|gate0|stats|test
"""
import os, sys, time, json, glob

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_e18 as e18          # noqa: E402
import run_e18b as e18b        # noqa: E402

OUT = "data/phase2/e24"
D = "data/phase2/e24b"
RA0, DEC0, R = 90.0, -66.56, 0.40
CUTOFF = "20:15"
WORKERS = 4

# re-point the E18 / E18-B module constants at the SEP field (no code of theirs is changed)
e18.OUT, e18.RA0, e18.DEC0, e18.R, e18.CUTOFF = OUT, RA0, DEC0, R, CUTOFF
e18.HALF = int(np.ceil((R + e18.MARGIN) / e18.PIX)) + 4
e18b.E18, e18b.D, e18b.RA0, e18b.DEC0, e18b.RCAT = OUT, D, RA0, DEC0, R + 0.07
e18b.WORKERS = WORKERS


def catalogs():
    from run_e10 import tap, vizier_cone
    os.makedirs(OUT, exist_ok=True)
    q = (f"SELECT g.source_id, g.ra, g.dec, g.phot_g_mean_mag, g.bp_rp, g.parallax, g.parallax_error, g.pmra, g.pmdec, "
         f"g.pmra_error, g.pmdec_error, g.ruwe, g.phot_variable_flag, g.classprob_dsc_combmod_quasar, "
         f"g.classprob_dsc_combmod_galaxy, g.classprob_dsc_combmod_star, v.best_class_name, v.best_class_score "
         f"FROM gaiadr3.gaia_source AS g LEFT OUTER JOIN gaiadr3.vari_classifier_result AS v ON g.source_id = v.source_id "
         f"WHERE 1 = CONTAINS(POINT('ICRS', g.ra, g.dec), CIRCLE('ICRS', {RA0}, {DEC0}, {R + 0.16}))")
    if os.path.exists(f"{OUT}/gaia_all_raw.parquet"):
        g = pd.read_parquet(f"{OUT}/gaia_all_raw.parquet")
    else:
        g = tap(q)
    g["source_id"] = g.source_id.astype("int64").astype(str)
    g.to_parquet(f"{OUT}/gaia_all_raw.parquet", index=False)
    g[g.phot_g_mean_mag < 19.5].reset_index(drop=True).to_parquet(f"{OUT}/gaia_raw.parquet", index=False)
    print("gaia", len(g), (g.phot_g_mean_mag < 19.5).sum(), flush=True)
    cols = "AllWISE,RAJ2000,DEJ2000,W1mag,e_W1mag,W2mag,e_W2mag,W3mag,W4mag,Jmag,Hmag,Kmag,ccf,ex,qph"
    w = vizier_cone("II/328/allwise", RA0, DEC0, R + 0.06, cols)
    for c in cols.split(",")[1:12]:
        w[c] = pd.to_numeric(w[c], errors="coerce")
    w = w[w.W1mag < 16].reset_index(drop=True)
    w.to_parquet(f"{OUT}/allwise_raw.parquet", index=False)
    print("allwise W1<16", len(w), flush=True)
    m = vizier_cone("VII/294", RA0, DEC0, R + 0.06, "RAJ2000,DEJ2000,Name,Type,Rmag,z")
    for c in ("RAJ2000", "DEJ2000", "Rmag", "z"):
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m.to_parquet(f"{OUT}/milliquas_raw.parquet", index=False)
    print("milliquas", len(m), flush=True)
    a = tap(f"SELECT g.source_id, g.ra, g.dec, g.phot_g_mean_mag, a.catalogue_name FROM gaiadr3.agn_cross_id AS a "
            f"JOIN gaiadr3.gaia_source AS g ON g.source_id = a.source_id "
            f"WHERE 1 = CONTAINS(POINT('ICRS', g.ra, g.dec), CIRCLE('ICRS', {RA0}, {DEC0}, {R + 0.06}))")
    a["source_id"] = a.source_id.astype("int64").astype(str)
    a.to_parquet(f"{OUT}/gaia_crf3_raw.parquet", index=False)
    print("gaia-crf3 agn_cross_id", len(a), flush=True)
    v = vizier_cone("B/vsx/vsx", RA0, DEC0, R + 0.07, "OID,Name,Type,max,min,Period,RAJ2000,DEJ2000")
    v.to_parquet(f"{OUT}/vsx_raw.parquet", index=False)
    print("vsx", len(v), flush=True)


def catalogue():
    """E18 catalogue rule (run_e18.catalogue) + Kim+2026 SEP rows, Milliquas (2") and a Gaia-any-G astrometric-star flag."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    e18.catalogue()
    s = pd.read_parquet(f"{OUT}/sources.parquet")
    cs = SkyCoord(s.ra.values * u.deg, s.dec.values * u.deg)
    k = pd.read_parquet("data/phase2/e18/kim2026_neowise_var.parquet")
    k = k[k.dec < 0].reset_index(drop=True)
    i, d2, _ = cs.match_to_catalog_sky(SkyCoord(k.ra.values * u.deg, k.dec.values * u.deg))
    s["neowise_var_kim2026"] = d2.arcsec < 2.0
    s["kim2026_name"] = np.where(s.neowise_var_kim2026, k.name.values[i], None)
    q = pd.read_parquet(f"{OUT}/milliquas_raw.parquet")
    if len(q):
        i, d2, _ = cs.match_to_catalog_sky(SkyCoord(q.RAJ2000.values * u.deg, q.DEJ2000.values * u.deg))
        ok = d2.arcsec < 2.0
        s["milliquas_name"] = np.where(ok, q.Name.values[i], None)
        s["milliquas_type"] = np.where(ok, q.Type.values[i], None)
        s["milliquas_z"] = np.where(ok, q.z.values[i], np.nan)
    else:
        s["milliquas_name"] = s["milliquas_type"] = None
        s["milliquas_z"] = np.nan
    # known-star flag for AllWISE-only sources: any Gaia DR3 source (any G) within 2" (propagated to J2010.5) with
    # parallax/error > 5 or total proper motion / error > 5
    ga = pd.read_parquet(f"{OUT}/gaia_all_raw.parquet")
    pm = np.hypot(ga.pmra.fillna(0), ga.pmdec.fillna(0))
    pme = np.hypot(ga.pmra_error.fillna(np.inf), ga.pmdec_error.fillna(np.inf))
    star = (ga.parallax / ga.parallax_error > 5).fillna(False) | (pm / pme > 5).fillna(False)
    ga = ga[star].reset_index(drop=True)
    dt = 2010.5 - 2016.0
    gd = ga.dec + ga.pmdec.fillna(0) * dt / 3.6e6
    gr = ga.ra + ga.pmra.fillna(0) * dt / 3.6e6 / np.cos(np.radians(ga.dec))
    i, d2, _ = cs.match_to_catalog_sky(SkyCoord(gr.values * u.deg, gd.values * u.deg))
    s["gaia_astrometric_star_2as"] = d2.arcsec < 2.0
    s.to_parquet(f"{OUT}/sources.tmp", index=False)
    os.replace(f"{OUT}/sources.tmp", f"{OUT}/sources.parquet")
    r = s[s.in_region & s.is_primary]
    print(len(s), "fit list;", len(r), "reported groups;", r.kind.value_counts().to_dict(), "kim", int(r.neowise_var_kim2026.sum()),
          "mq", int(r.milliquas_name.notna().sum()), flush=True)


def frames():
    e18.frames()
    os.makedirs(D, exist_ok=True)
    v = pd.read_parquet(f"{OUT}/vsx_raw.parquet")
    v.to_parquet(f"{D}/vsx.parquet", index=False)        # where run_e18b.populations looks for VSX


def _du(path):
    return sum(os.path.getsize(os.path.join(a, f)) for a, _, fs in os.walk(path) for f in fs)


def phot():
    """run_e18.phot with a disk guard: no new chunk is started once data/phase2/e24 exceeds E24_MAX_GB
    (storage hold); resumable exactly like run_e18.phot (pictures already in meta/ are skipped)."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    lim = float(os.environ.get("E24_MAX_GB", "0.8")) * 1e9
    os.makedirs(f"{OUT}/phot", exist_ok=True)
    os.makedirs(f"{OUT}/meta", exist_ok=True)
    fr = pd.read_parquet(f"{OUT}/frames.parquet")
    done = set()
    for p in glob.glob(f"{OUT}/meta/meta-*.parquet"):
        mt = pd.read_parquet(p)
        done |= set(mt.key[~mt.status.str.startswith("error")])
    todo = [k for k in fr.key if k not in done]
    nmax = max([int(os.path.basename(p)[5:9]) for p in glob.glob(f"{OUT}/meta/meta-*.parquet")] + [-1])
    chunks = [todo[i:i + 12] for i in range(0, len(todo), 12)]
    t0 = time.time()
    print(f"{len(done)} done, {len(todo)} to do, {len(chunks)} chunks, limit {lim / 1e9:.2f} GB", flush=True)
    with ProcessPoolExecutor(WORKERS) as ex:
        it = iter(chunks)
        futs = set()
        for _ in range(WORKERS):
            c = next(it, None)
            if c is not None:
                futs.add(ex.submit(e18._work, c))
        n, nd, stop = nmax, 0, False
        while futs:
            fu = next(as_completed(futs))
            futs.remove(fu)
            out, metas = fu.result()
            n += 1
            if out:
                pd.concat(out, ignore_index=True).to_parquet(f"{OUT}/phot/part-{n:04d}.tmp", index=False)
                os.replace(f"{OUT}/phot/part-{n:04d}.tmp", f"{OUT}/phot/part-{n:04d}.parquet")
            pd.DataFrame(metas).to_parquet(f"{OUT}/meta/meta-{n:04d}.tmp", index=False)
            os.replace(f"{OUT}/meta/meta-{n:04d}.tmp", f"{OUT}/meta/meta-{n:04d}.parquet")
            nd += len(metas)
            size = _du(OUT)
            if n % 5 == 0:
                print(f"{time.strftime('%H:%M:%S')} part {n}: {nd} pictures this run, "
                      f"{nd / (time.time() - t0):.3f}/s, {size / 1e9:.3f} GB", flush=True)
            if size > lim and not stop:
                stop = True
                print(f"{time.strftime('%H:%M:%S')} disk guard: {size / 1e9:.3f} GB > limit, no new chunks", flush=True)
            if not stop:
                c = next(it, None)
                if c is not None:
                    futs.add(ex.submit(e18._work, c))
    print("done", time.strftime("%H:%M:%S"), f"{_du(OUT) / 1e9:.3f} GB", flush=True)


def needed():
    """Sources this test uses: AllWISE-only, Kim / Milliquas / Gaia-CRF3 matches, calibration-half Gaia null, Gate-0 stars."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    s = e18b.populations()
    a = pd.read_parquet(f"{OUT}/gaia_crf3_raw.parquet")
    _, d2, _ = SkyCoord(s.ra.values * u.deg, s.dec.values * u.deg).match_to_catalog_sky(
        SkyCoord(a.ra.values * u.deg, a.dec.values * u.deg))
    g0 = s.null & (s.ruwe < 1.4) & (s.d_nearest_arcsec > 15) & (s.blend_n == 1) & (s.G < 17)
    keep = (s.kind == "allwise_only") | s.kim | s.mq | (d2.arcsec < 2.0) | (s.half == "cal") | g0
    return set(s.idx[keep].astype(int))


def prep():
    """run_e18b.prep's usable-picture rule, part by part, keeping only the rows of needed() sources (storage);
    picture PA from all sources in the picture as in run_e18b.prep; no bright_xy (vetoes not used here)."""
    os.makedirs(D, exist_ok=True)
    need = needed()
    s = pd.read_parquet(f"{OUT}/sources.parquet")
    sc = s.set_index("idx")
    bad = e18b.anomaly_set()
    cols = ["src", "x", "y", "n_masked_box", "flags_core", "flux_mJy", "flux_err_mJy", "wl_um", "chi2dof_local", "mjd",
            "detector", "obs_id", "edge_px"]
    keep, cnt = [], dict(rows=0, rows_needed=0, nonfinite=0, masked=0, edge=0, flags=0, anomaly=0)
    parts = sorted(glob.glob(f"{OUT}/phot/part-*.parquet"))
    npic, pics_det = 0, {}
    for p in parts:
        q = pd.read_parquet(p, columns=cols)
        cnt["rows"] += len(q)
        key = q.obs_id.astype(str) + "_" + q.detector.astype(str)
        uk, pic = np.unique(key.values, return_inverse=True)
        q["pic"] = (pic + npic).astype(np.int32)
        npic += len(uk)
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
        q = q[q.src.isin(need)]
        cnt["rows_needed"] += len(q)
        ok = np.isfinite(q.flux_mJy) & np.isfinite(q.flux_err_mJy) & (q.flux_err_mJy > 0)
        cnt["nonfinite"] += int((~ok).sum())
        m = q.n_masked_box <= e18b.MAX_MASKED
        cnt["masked"] += int((ok & ~m).sum()); ok &= m
        m = q.edge_px >= e18b.MIN_EDGE
        cnt["edge"] += int((ok & ~m).sum()); ok &= m
        m = e18b.flags_ok(q.flags_core.values)
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
    m = q.chi2dof_local <= np.maximum(e18b.CHI2LOC_MIN, e18b.CHI2LOC_FAC * med)
    cnt["chi2_local"] = int((~m).sum())
    q = q[m].sort_values(["src", "mjd"]).reset_index(drop=True)
    q.to_parquet(f"{D}/phot.parquet", index=False, row_group_size=200000, compression="zstd")
    cnt.update(n_parts=len(parts), n_pictures=npic, n_sources_needed=len(need), kept=len(q),
               n_pictures_with_usable_rows_per_detector=pics_det,
               mjd_range=[float(q.mjd.min()), float(q.mjd.max())], time=time.strftime("%H:%M:%S"))
    json.dump(cnt, open(f"{D}/prep.json", "w"), indent=1)
    print(cnt, flush=True)


def gate0():
    """E18-B Gate-0 rule on SEP constant stars; a detector without stars in a Gate-0 sample takes the NEP value."""
    e18b.gate0()
    g = json.load(open(f"{D}/gate0.json"))
    nep = json.load(open("data/findings/e18b.json"))["gate0"]
    s = e18b.populations()
    c = s[s.null & (s.ruwe < 1.4) & (s.d_nearest_arcsec > 15) & (s.blend_n == 1) & (s.G < 17)]
    data = e18b.load_src(c.idx.values)
    nb = {d: 0 for d in range(1, 7)}
    for i in c.idx[c.G < 13]:
        if i in data:
            dd = e18b.drop_thin(data[i])
            if len(dd["f"]) >= e18b.MIN_PICS:
                for d in np.unique(dd["det"]):
                    nb[int(d)] += 1
    g["n_bright_per_detector"] = nb
    g["fallback_to_nep"] = []
    for d in range(1, 7):
        k = str(d) if str(d) in g["s_d"] else d
        if k not in g["s_d"]:
            g["s_d"][str(d)] = nep["s_d"][str(d)]
            g["fallback_to_nep"].append(f"s_d{d}")
        if nb[d] == 0:
            g["f_d"][str(d)] = nep["f_d"][str(d)]
            g["fallback_to_nep"].append(f"f_d{d}")
    json.dump(g, open(f"{D}/gate0.json", "w"), indent=1)
    print(g["n_bright_per_detector"], g["fallback_to_nep"], flush=True)


def _work(task):
    chunk, g0 = task
    s_d = {int(k): v for k, v in g0["s_d"].items()}
    f_d = {int(k): v for k, v in g0["f_d"].items()}
    data = e18b.load_src(chunk)
    real = []
    for i in chunk:
        if i not in data:
            continue
        d = e18b.drop_thin(data[i])
        if len(d["f"]) < e18b.MIN_PICS:
            continue
        freq = np.arange(e18b.FMIN, e18b.FMAX, 1 / (e18b.OVERS * (d["t"].max() - d["t"].min())))
        allowed, pk = e18b.alias_mask(freq, d["t"])
        try:
            r = e18b.source_stats(d, s_d, f_d, freq, allowed)
        except Exception as e:
            print("fail", i, repr(e)[:100], flush=True)
            continue
        real.append(dict(src=int(i), **e18b._flat(r)))
    return real


def stats():
    """Real statistics only (run_e18b.source_stats) for every reported source."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    g0 = json.load(open(f"{D}/gate0.json"))
    s = e18b.populations()
    idx = s.idx.values
    chunks = [idx[i:i + 25] for i in range(0, len(idx), 25)]
    rows = []
    t0 = time.time()
    with ProcessPoolExecutor(WORKERS) as ex:
        futs = [ex.submit(_work, (c, g0)) for c in chunks]
        for k, fu in enumerate(as_completed(futs)):
            rows += fu.result()
            if k % 10 == 0:
                print(f"{time.strftime('%H:%M:%S')} {k + 1}/{len(chunks)} chunks, {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_parquet(f"{D}/real.parquet", index=False)
    print("stats done", len(rows), time.strftime("%H:%M:%S"), flush=True)


def _mwu(a, b):
    from scipy.stats import mannwhitneyu
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return None
    return float(mannwhitneyu(a, b, alternative="greater").pvalue)


def _matched(g, c, k=10):
    """Union of the k nearest controls to each test source in (m_AB / 0.5, (W1 - W2) / 0.2)."""
    g = g[np.isfinite(g.w12)]
    c = c[np.isfinite(c.w12)]
    if len(g) == 0 or len(c) == 0:
        return c.iloc[:0]
    G = np.stack([g.m_ab / 0.5, g.w12 / 0.2], 1)
    C = np.stack([c.m_ab / 0.5, c.w12 / 0.2], 1)
    nn = np.argsort(((G[:, None, :] - C[None, :, :]) ** 2).sum(-1), 1)[:, :k]
    return c.iloc[np.unique(nn.ravel())]


def test():
    """The frozen test (docs/protocols/E24.md)."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    pop = e18b.populations()
    real = pd.read_parquet(f"{D}/real.parquet")
    r = pop.merge(real, left_on="idx", right_on="src", how="inner")
    cal = r[r.half == "cal"]
    norms = {st: e18b.Norm(cal.m_ab.values, cal[st].values) for st in ("chi2r", "s2", "ls", "amp")}
    for st, n in norms.items():
        r[f"z_{st}"] = n.z(r.m_ab.values, r[st].values)
    r["w12"] = r.W1mag - r.W2mag
    a = pd.read_parquet(f"{OUT}/gaia_crf3_raw.parquet")
    _, d2, _ = SkyCoord(r.ra.values * u.deg, r.dec.values * u.deg).match_to_catalog_sky(
        SkyCoord(a.ra.values * u.deg, a.dec.values * u.deg))
    r["crf3"] = d2.arcsec < 2.0
    star = r.gaia_astrometric_star_2as.fillna(False).astype(bool)
    agn = r[r.mq | r.crf3 | (r.kim & (r.w12 >= 0.8))]
    aw = r[(r.kind == "allwise_only") & ~r.kim & ~r.mq & ~r.crf3 & r.vsx.isna() & ~star]

    def ctrl(g):
        return aw[(aw.m_ab > g.m_ab.min() - 0.3) & (aw.m_ab < g.m_ab.max() + 0.3)] if len(g) else aw.iloc[:0]

    def ok(g):
        return int(np.isfinite(g.z_amp).sum()) >= 5
    c = ctrl(agn)
    prep_ = json.load(open(f"{D}/prep.json"))
    g0 = json.load(open(f"{D}/gate0.json"))
    out = dict(experiment="E24", protocol="docs/protocols/E24.md", time=time.strftime("%Y-%m-%d %H:%M:%S PST"),
               field=f"SEP deep field, RA {RA0} Dec {DEC0}, R {R} deg",
               n_pictures=int(prep_["n_pictures"]), n_sources=int(len(r)), n_sources_reported=int(len(pop)),
               n_null_cal=int(len(cal)), n_agn=int(len(agn)), n_agn_milliquas=int(agn.mq.sum()),
               n_agn_crf3=int(agn.crf3.sum()), n_agn_kim_w12=int((agn.kim & (agn.w12 >= 0.8)).sum()),
               n_control=int(len(c)),
               agn_median=float(agn.z_amp.median()) if len(agn) else None,
               control_median=float(c.z_amp.median()) if len(c) else None,
               p_primary=_mwu(agn.z_amp, c.z_amp) if ok(agn) else None)
    out["testable"] = ok(agn)
    out["replicated"] = bool(out["p_primary"] is not None and out["p_primary"] < 0.01)
    for key, g in (("s0_kim_or_milliquas", r[r.kim | r.mq]), ("s1_crf3_or_milliquas", r[r.crf3 | r.mq])):
        cc = ctrl(g)
        out[f"{key}_n"] = int(len(g))
        out[f"{key}_n_control"] = int(len(cc))
        out[f"{key}_median"] = float(g.z_amp.median()) if len(g) else None
        out[f"{key}_control_median"] = float(cc.z_amp.median()) if len(cc) else None
        out[f"{key}_p"] = _mwu(g.z_amp, cc.z_amp) if ok(g) else None
    m = _matched(agn, c)
    gg = agn[np.isfinite(agn.w12)]
    out.update(s3_n=int(len(gg)), s3_n_matched_control=int(len(m)),
               s3_median=float(gg.z_amp.median()) if len(gg) else None,
               s3_matched_control_median=float(m.z_amp.median()) if len(m) else None,
               s3_w12_median=float(gg.w12.median()) if len(gg) else None,
               s3_matched_w12_median=float(m.w12.median()) if len(m) else None,
               s3_m_ab_median=float(gg.m_ab.median()) if len(gg) else None,
               s3_matched_m_ab_median=float(m.m_ab.median()) if len(m) else None,
               s3_p=_mwu(gg.z_amp, m.z_amp) if ok(gg) else None)
    out["s3_not_artefact"] = bool(out["s3_p"] is not None and out["s3_p"] < 0.01)
    for st in ("chi2r", "s2", "ls"):
        out[f"s4_{st}_agn_median"] = float(agn[f"z_{st}"].median()) if len(agn) else None
        out[f"s4_{st}_control_median"] = float(c[f"z_{st}"].median()) if len(c) else None
        out[f"s4_{st}_p"] = _mwu(agn[f"z_{st}"], c[f"z_{st}"]) if ok(agn) else None
    out["gate0_s_d"] = g0["s_d"]
    out["gate0_f_d"] = g0["f_d"]
    out["gate0_fallback_to_nep"] = g0.get("fallback_to_nep")
    out["mjd_range"] = prep_["mjd_range"]
    out["agn_sources"] = agn[["id", "kind", "m_ab", "w12", "mq", "crf3", "kim", "z_amp", "n"]].round(3).to_dict("records")
    r["group"] = np.where(r.idx.isin(agn.idx), "test", np.where(r.idx.isin(c.idx), "control", ""))
    keep = ["id", "kind", "ra", "dec", "m_ab", "w12", "kim", "mq", "crf3", "z_amp", "z_s2", "z_ls", "z_chi2r", "amp", "n", "group"]
    r[keep].to_parquet(f"{D}/e24_sources.parquet", index=False)
    json.dump(out, open("data/findings/e24.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps(out, indent=1, default=str), flush=True)


def figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = json.load(open("data/findings/e24.json"))
    sep = pd.read_parquet(f"{D}/e24_sources.parquet")
    nep = pd.read_parquet("data/phase2/e18b/e18b_sources.parquet")
    ngc = np.hypot((nep.ra - e18b.NGC6543[0]) * np.cos(np.radians(nep.dec)), nep.dec - e18b.NGC6543[1]) * 60 > e18b.NGC6543[2]
    nep = nep[ngc]
    ng = nep[nep.kim | nep.mq]
    aw = nep[(nep.kind == "allwise_only") & ~nep.catalogued_var & ~nep.mq]
    nc = aw[(aw.m_ab > ng.m_ab.min() - 0.3) & (aw.m_ab < ng.m_ab.max() + 0.3)]
    nep_p = json.load(open("data/findings/e18b.json"))["post_hoc"]["population_test_agn_vs_allwise_only"]["results"]["Kim or Milliquas"]["amp"]["mann_whitney_p_greater"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True)
    bins = np.linspace(-4, 8, 37)
    for ax, (name, g, c, p, lab) in zip(axs, (
            ("NEP (E18-B, post hoc)", ng, nc, nep_p, "Milliquas AGN or NEOWISE variables"),
            ("SEP (E24, predefined)", sep[sep.group == "test"], sep[sep.group == "control"], out["p_primary"], "known AGN"))):
        ax.hist(c.z_amp.dropna().clip(-4, 8), bins, density=True, histtype="stepfilled", color="0.75",
                label=f"controls (n={len(c)})")
        ax.hist(g.z_amp.dropna().clip(-4, 8), bins, density=True, histtype="step", lw=2, color="C3",
                label=f"{lab} (n={len(g)})")
        ax.axvline(c.z_amp.median(), color="0.4", ls="--", lw=1)
        ax.axvline(g.z_amp.median(), color="C3", ls="--", lw=1)
        ax.set_title(f"{name}: Mann-Whitney p = {p:.2g}" if p is not None else name, fontsize=10)
        ax.set_xlabel("z of time-binned amplitude 'amp'")
        ax.legend(fontsize=8, frameon=False)
    axs[0].set_ylabel("density")
    fig.tight_layout()
    fig.savefig("site/data/figs/e24_agn.png", dpi=150)
    print("figure written")


def var(step):
    """Addendum A: E18-B machinery unchanged on all SEP sources (outputs in data/phase2/e24b/var,
    data/findings/e24_var.json; figures kept in data/phase2/e24b/var/figs). step: catalogs|prep|gate0|stats|analyze."""
    vd = f"{D}/var"
    e18b.D, e18b.OUT, e18b.FIGS = vd, "data/findings/e24_var.json", f"{vd}/figs"
    os.makedirs(e18b.FIGS, exist_ok=True)
    if step == "gate0":
        gate0()                      # E18-B rule + recorded NEP fallback (same wrapper as the AGN test)
    else:
        getattr(e18b, step)()


def extcat():
    """Addendum A: ASAS-SN / OGLE variable catalogues in the region (labels for candidates only)."""
    from run_e10 import vizier_cone
    tabs = {"asassn": ("II/366/catalog", "ID,RAJ2000,DEJ2000,Type,Period"),
            "ogle4_ecl": ("J/AcA/66/421/ecl", "ID,RAJ2000,DEJ2000,Type,Per"),
            "ogle4_rrl": ("J/AcA/66/131/rrlyr", "ID,RAJ2000,DEJ2000,Type,Per"),
            "ogle4_cep": ("J/AcA/65/297/cepheids", "ID,RAJ2000,DEJ2000,Type,Per"),
            "ogle3_lpv": ("J/AcA/59/239/lpv", "ID,RAJ2000,DEJ2000,Type,P1")}
    rep = {}
    for k, (t, cols) in tabs.items():
        v = vizier_cone(t, RA0, DEC0, R + 0.07, cols)
        rep[k] = dict(table=t, rows=None if v is None else len(v))
        if v is not None and len(v):
            v.to_parquet(f"{OUT}/extcat_{k}.parquet", index=False)
    json.dump(rep, open(f"{D}/extcat.json", "w"), indent=1)
    print(rep, flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "var":
        var(sys.argv[2])
        sys.exit()
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(k, "1")
    globals()[sys.argv[1]]()
