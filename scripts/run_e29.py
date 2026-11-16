"""E29 — is E12's parallax of Epsilon Indi B biased by colour? Red Gaia comparison stars fitted with the E12 pipeline.

Choices: docs/protocols/E29.md (frozen before any red-star or split fit). Stages:
  select   red comparison list from Gaia DR3 (rule in E29.md)            -> data/phase2/e29/red_stars.json
  check    batched reader reproduces stored E12 blue-star positions      -> data/phase2/e29/adapter_check.json
  measure  red-star positions in the 151 E12 pictures, priority order    -> data/phase2/e29/red_<source_id>.parquet
  fit      E12 fit5 for blue + red + target, decision, detector split    -> data/findings/e29.json, e29s.json, e29_colour.png/.svg

Position measurement and fit are imported unchanged from scripts/run_e12.py (measure_one, fit5, GRID_FLOOR).
The only change is speed: one worker call measures all stars of a batch in one picture, and the FITS headers of that
picture are read once (memoised); pixel content is the same bytes. Per-star Gaia neighbour tables (G < 19, 0.085 deg)
are cut from one Gaia DR3 cone (1.6 deg, G < 19.5) instead of one query per star; same table, same columns.
  OMP_NUM_THREADS=1 .venv/bin/python scripts/run_e29.py <stage>
"""
import json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_e12 as e12  # noqa: E402

OUT = "data/phase2/e29"
CONE = f"{OUT}/gaia_cone_1p6_G19p5.parquet"
FIELD, EPS_INDI_A = e12.FIELD, e12.EPS_INDI_A_2026
R_MAX, G_MIN, G_MAX, BPRP_MIN, N_RED, MIN_COVER, BATCH = 1.3, 10.5, 17.0, 2.5, 30, 80, 2
CUTOFF = "21:25"   # stopping rule (E29.md): only batches finished by 21:25 PST count
WORKERS = 2


# ------------------------------------------------------------------ selection
def _cover(ra, dec, idx):
    from matplotlib.path import Path
    n = 0
    for cr, cdd in zip(idx.c_ra, idx.c_dec):
        c = np.c_[(np.asarray(cr) - ra) * np.cos(np.radians(dec)), np.asarray(cdd) - dec]
        c = c - 0.03 * np.sign(c - c.mean(0))          # corners pulled in by 0.03 deg (~1.8 arcmin, > the 15 px stamp)
        n += Path(c).contains_point((0, 0))
    return int(n)


def select():
    t = pd.read_parquet(CONE)
    tp = pd.read_parquet(e12.TARGET_PARQUET)
    idx = pd.read_parquet("data/index/frames_dr1.parquet").set_index("key").loc[tp.key]
    cd = np.cos(np.radians(FIELD[1]))

    def sep(ra, dec, r2, d2):
        return 3600 * np.hypot((ra - r2) * cd, dec - d2)
    ok = ((t.bp_rp > BPRP_MIN) & (t.phot_g_mean_mag >= G_MIN) & (t.phot_g_mean_mag <= G_MAX) & (t.ruwe < 1.4)
          & t.parallax.notna() & t.pmra.notna() & (sep(t.ra, t.dec, *FIELD) <= 3600 * R_MAX) & (sep(t.ra, t.dec, *EPS_INDI_A) > 300))
    for i in t.index[ok]:
        s = sep(t.ra, t.dec, t.ra[i], t.dec[i])
        if ((s > 0) & (s < 25) & (t.phot_g_mean_mag < t.phot_g_mean_mag[i] + 3)).any() or ((s > 0) & (s < 120) & (t.phot_g_mean_mag < 9)).any():
            ok[i] = False
    el = t[ok].copy()
    el["n_cover"] = [_cover(a, b, idx) for a, b in zip(el.ra, el.dec)]
    el = el[el.n_cover >= MIN_COVER].sort_values(["phot_g_mean_mag", "source_id"]).reset_index(drop=True)
    stars = el.head(N_RED)
    rec = dict(rule=dict(r_max_deg=R_MAX, G=[G_MIN, G_MAX], bp_rp_min=BPRP_MIN, ruwe_max=1.4, min_cover=MIN_COVER, n_red=N_RED,
                         order="G ascending (brightest first), ties by source_id; measured in this order in batches of 2"),
               n_eligible=int(len(el)),
               stars=[dict(source_id=str(r.source_id), ra=r.ra, dec=r.dec, pmra=r.pmra, pmdec=r.pmdec, parallax=r.parallax,
                           parallax_error=r.parallax_error, G=r.phot_g_mean_mag, bp_rp=r.bp_rp, ruwe=r.ruwe, n_cover=int(r.n_cover),
                           sep_arcmin=float(sep(r.ra, r.dec, *FIELD) / 60)) for r in stars.itertuples()])
    json.dump(rec, open(f"{OUT}/red_stars.json", "w"), indent=1)
    print(len(el), "eligible")
    for s in rec["stars"]:
        print(s["source_id"], round(s["G"], 2), round(s["bp_rp"], 2), round(s["parallax"], 2), round(s["sep_arcmin"], 1), s["n_cover"])


def _write_tables(stars):
    t = pd.read_parquet(CONE)
    os.makedirs(f"{OUT}/gaia", exist_ok=True)
    for s in stars:
        p = f"{OUT}/gaia/{s['source_id']}.parquet"
        if os.path.exists(p):
            continue
        k = np.hypot((t.ra - s["ra"]) * np.cos(np.radians(s["dec"])), t.dec - s["dec"]) < 0.085   # CIRCLE(0.085) as in run_e12
        t.loc[k & (t.phot_g_mean_mag < 19.0), ["ra", "dec", "pmra", "pmdec", "phot_g_mean_mag"]].reset_index(drop=True).to_parquet(p)


# ------------------------------------------------------------------ measurement (run_e12.measure_one, batched per picture)
_patched = False


def _batch(a):
    global _patched
    key, mjd, stars, out = a
    import e2_cache
    if not _patched:
        e2_cache._header = lru_cache(maxsize=16)(e2_cache._header)
        _patched = True
    e12.OUT = out
    return [e12.measure_one((key, s["source_id"], dict(s, own_table=True), mjd)) for s in stars]


def _run(stars, keys, mjds, out):
    with ProcessPoolExecutor(WORKERS) as ex:
        res = list(ex.map(_batch, [(k, m, stars, out) for k, m in zip(keys, mjds)]))
    return {s["source_id"]: [r[i] for r in res if r[i]] for i, s in enumerate(stars)}


def check():
    """Re-measure two E12 blue stars in 6 pictures with the batched reader; positions must equal the stored ones."""
    blue = json.load(open(f"{e12.OUT}/null_stars.json"))["stars"][:2]
    tp = pd.read_parquet(e12.TARGET_PARQUET)
    sub = tp.iloc[np.linspace(0, len(tp) - 1, 6).round().astype(int)]
    t = time.time()
    got = _run(blue, sub.key.values, sub.mjd.values, e12.OUT)
    dt = time.time() - t
    rows = []
    for s in blue:
        old = pd.read_parquet(f"{e12.OUT}/null_{s['source_id']}.parquet").set_index("key")
        for r in got[s["source_id"]]:
            if "ra_meas" in r and r["key"] in old.index:
                o = old.loc[r["key"]]
                rows.append(dict(source_id=s["source_id"], key=r["key"], d_ra_mas=(r["ra_meas"] - o.ra_meas) * 3.6e6,
                                 d_dec_mas=(r["dec_meas"] - o.dec_meas) * 3.6e6, d_snr=r["snr"] - o.snr))
    rec = dict(n_compared=len(rows), max_abs_mas=float(max(max(abs(v["d_ra_mas"]), abs(v["d_dec_mas"])) for v in rows)),
               seconds=dt, seconds_per_star_picture=dt / (len(sub) * len(blue)) * WORKERS, rows=rows)
    json.dump(rec, open(f"{OUT}/adapter_check.json", "w"), indent=1)
    print({k: v for k, v in rec.items() if k != "rows"})


def measure():
    stars = json.load(open(f"{OUT}/red_stars.json"))["stars"]
    _write_tables(stars)
    tp = pd.read_parquet(e12.TARGET_PARQUET)
    for b in range(0, len(stars), BATCH):
        todo = [s for s in stars[b:b + BATCH] if not os.path.exists(f"{OUT}/red_{s['source_id']}.parquet")]
        if not todo:
            continue
        if time.strftime("%H:%M") >= CUTOFF:
            print("cutoff reached; stop", flush=True)
            break
        t = time.time()
        got = _run(todo, tp.key.values, tp.mjd.values, OUT)
        for s in todo:
            pd.DataFrame(got[s["source_id"]]).to_parquet(f"{OUT}/red_{s['source_id']}.parquet")
        log = f"{OUT}/batches.json"
        bl = json.load(open(log)) if os.path.exists(log) else []
        bl.append(dict(batch=b // BATCH, stars=[s["source_id"] for s in todo], finished=time.strftime("%Y-%m-%d %H:%M:%S")))
        json.dump(bl, open(log, "w"), indent=1)
        print(time.strftime("%H:%M:%S"), "batch", b // BATCH, [len(got[s["source_id"]]) for s in todo], f"{time.time() - t:.0f}s", flush=True)


# ------------------------------------------------------------------ fits, decision, outputs
TARGET_BPRP, TARGET_G = 5.081964, 18.030659    # Gaia DR3 6412596012146801152 (Epsilon Indi B, blend)


def _target_windows():
    tp = pd.read_parquet(e12.TARGET_PARQUET).dropna(subset=["ra_meas"])
    tp = tp[(tp.snr > 20) & np.isfinite(tp.err_arcsec)].sort_values("mjd")
    tw = e12.windows(tp.mjd.values)
    return [(tp.mjd.values[tw == k].min() - 5, tp.mjd.values[tw == k].max() + 5) for k in np.unique(tw)]


def star_fit(df, s, wins, half=None):
    """E12 per-star rule (run_e12.analyse): eligibility, fit5 with GRID_FLOOR, offset from Gaia. half: None, 'D1-3', 'D4-6'."""
    rec = dict(source_id=s["source_id"], G=s["G"], bp_rp=s["bp_rp"], sep_arcmin=s["sep_arcmin"], gaia_parallax_mas=s["parallax"])
    if "ra_meas" not in df:
        return dict(rec, used=False, reason="no measurements")
    if half:
        df = df[df.det <= 3] if half == "D1-3" else df[df.det >= 4]
    sel = df.dropna(subset=["ra_meas"])
    sel = sel[(sel.snr > 20) & np.isfinite(sel.err_arcsec)]
    per = [int(((sel.mjd > a) & (sel.mjd < b)).sum()) for a, b in wins]
    rec.update(n_selected=int(len(sel)), n_per_target_window=per, median_snr=float(sel.snr.median()) if len(sel) else None)
    nmin, wmin = (30, 3) if half else (60, 5)
    if len(sel) < nmin or min(per) < wmin:
        return dict(rec, used=False, reason="too few selected pictures")
    r, _ = e12.fit5(df, sig_floor=e12.GRID_FLOOR)
    plx = 1000 * r["params"]["parallax_arcsec"]
    return dict(rec, used=True, parallax_mas=plx, parallax_err_mas=1000 * r["errors"]["parallax_arcsec"], n_used=r["n_used"],
                chi2_red=r["chi2_red"], d_parallax_mas=plx - s["parallax"])


def _stats(v):
    v = np.asarray(v, float)
    n = len(v)
    return dict(n=n, mean=float(v.mean()) if n else None, sd=float(v.std(ddof=1)) if n > 1 else None,
                se=float(v.std(ddof=1) / np.sqrt(n)) if n > 1 else None, median=float(np.median(v)) if n else None)


def fit():
    from scipy import stats as sst
    assert os.path.exists("docs/protocols/E29.md")
    wins = _target_windows()
    blue = json.load(open(f"{e12.OUT}/null_stars.json"))["stars"]
    red_all = json.load(open(f"{OUT}/red_stars.json"))["stars"]
    done = {sid for b in json.load(open(f"{OUT}/batches.json")) if b["finished"][11:16] < CUTOFF for sid in b["stars"]}
    red = [s for s in red_all if s["source_id"] in done]
    ld = lambda p: pd.read_parquet(p) if os.path.exists(p) else pd.DataFrame()  # noqa: E731
    bdf = {s["source_id"]: ld(f"{e12.OUT}/null_{s['source_id']}.parquet") for s in blue}
    rdf = {s["source_id"]: ld(f"{OUT}/red_{s['source_id']}.parquet") for s in red}
    B = [star_fit(bdf[s["source_id"]], s, wins) for s in blue]
    R = [star_fit(rdf[s["source_id"]], s, wins) for s in red]
    e12_null = {v["source_id"]: v["d_parallax_mas"] for v in json.load(open("data/findings/e12.json"))["null"]["stars"]}
    repro = max(abs(v["d_parallax_mas"] - e12_null[v["source_id"]]) for v in B)
    tgt = json.load(open("data/findings/e12.json"))
    plx_t = tgt["parallax_mas"]
    sfit = max(tgt["parallax_err_mas"]["fit_scaled"], tgt["parallax_err_mas"]["bootstrap"])
    dB = [v["d_parallax_mas"] for v in B if v["used"]]
    dR = [v["d_parallax_mas"] for v in R if v["used"]]
    sB, sR = _stats(dB), _stats(dR)
    prim = dict(blue=sB, red=sR)
    if sR["n"] < 8:
        verdict = "INCONCLUSIVE"
    else:
        D, SE = sR["mean"] - sB["mean"], float(np.hypot(sR["se"], sB["se"]))
        det = abs(D) > 2 * SE
        verdict = "COLOUR-DEPENDENT BIAS REPORTED" if det else "NO COLOUR DEPENDENCE DETECTED"
        prim.update(D_red_minus_blue_mas=D, SE_D_mas=SE, D_over_SE=D / SE, ci95_D_mas=[D - 1.96 * SE, D + 1.96 * SE],
                    upper_limit95_abs_D_mas=abs(D) + 1.96 * SE, welch_p=float(sst.ttest_ind(dR, dB, equal_var=False).pvalue),
                    median_red_minus_blue_mas=sR["median"] - sB["median"])
        if det:
            pc = plx_t - sR["mean"]
            sc = float(np.sqrt(sR["sd"] ** 2 + sR["se"] ** 2 + sfit ** 2))
            gp, sp = tgt["gaia_parallax_mas"], tgt["secondary_reference"]["parallax_mas"]
            prim["corrected"] = dict(parallax_mas=pc, err_mas=sc, distance_pc=1000 / pc,
                                     distance_pc_68=[1000 / (pc + sc), 1000 / (pc - sc) if pc > sc else None],
                                     distance_ly=1000 / pc * 3.2616, sigma_from_gaia=(pc - gp) / np.hypot(sc, tgt["gaia_parallax_err_mas"]),
                                     sigma_from_system=(pc - sp) / sc, sigma_from_zero=pc / sc)
    allc = [(v["bp_rp"], v["d_parallax_mas"]) for v in B + R if v["used"]]
    x, y = np.array(allc).T
    lr = sst.linregress(x, y)
    s_res = np.sqrt(np.sum((y - lr.intercept - lr.slope * x) ** 2) / (len(x) - 2))
    se_pred = float(s_res * np.sqrt(1 / len(x) + (TARGET_BPRP - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2)))
    rr = [(v["sep_arcmin"], v["d_parallax_mas"]) for v in R if v["used"]]
    lr_sep = sst.linregress(*np.array(rr).T) if len(rr) > 2 else None
    desc = dict(line_vs_bprp=dict(intercept=lr.intercept, slope_mas_per_mag=lr.slope, slope_err=lr.stderr, p=lr.pvalue,
                                  value_at_target_bprp=lr.intercept + lr.slope * TARGET_BPRP, value_at_target_err_approx=se_pred,
                                  note="extrapolation from BP-RP <= %.2f to 5.08; illustrative only" % x.max()),
                red_offset_vs_distance_from_target=dict(slope_mas_per_arcmin=lr_sep.slope, slope_err=lr_sep.stderr, p=lr_sep.pvalue) if lr_sep else None)
    # secondary: detector halves
    def halves(df, s):
        a, b = star_fit(df, s, wins, "D1-3"), star_fit(df, s, wins, "D4-6")
        o = dict(source_id=s["source_id"], bp_rp=s["bp_rp"], d13=a, d46=b)
        if a["used"] and b["used"]:
            o["delta_mas"] = a["parallax_mas"] - b["parallax_mas"]
        return o
    HB = [halves(bdf[s["source_id"]], s) for s in blue]
    HR = [halves(rdf[s["source_id"]], s) for s in red]
    tp = pd.read_parquet(e12.TARGET_PARQUET)
    th = {}
    for h, m in (("D1-3", tp.det <= 3), ("D4-6", tp.det >= 4)):
        r, _ = e12.fit5(tp[m], sig_floor=e12.GRID_FLOOR)
        th[h] = dict(parallax_mas=1000 * r["params"]["parallax_arcsec"], err_mas=1000 * r["errors"]["parallax_arcsec"], n_used=r["n_used"])
    dT = th["D1-3"]["parallax_mas"] - th["D4-6"]["parallax_mas"]
    dTf = float(np.hypot(th["D1-3"]["err_mas"], th["D4-6"]["err_mas"]))
    dl = [o["delta_mas"] for o in HB + HR if "delta_mas" in o]
    sd_ = _stats(dl)
    sec = dict(comparison_pooled=sd_, blue=_stats([o["delta_mas"] for o in HB if "delta_mas" in o]),
               red=_stats([o["delta_mas"] for o in HR if "delta_mas" in o]))
    sec["comparison_wavelength_dependence"] = bool(sd_["n"] > 1 and abs(sd_["mean"]) > 2 * sd_["se"])
    semp = float(np.hypot(sd_["sd"], sd_["se"]))
    sec["target"] = dict(halves=th, delta_mas=dT, delta_formal_err_mas=dTf, minus_comparison_mean_mas=dT - sd_["mean"],
                         empirical_err_mas=semp, z=(dT - sd_["mean"]) / semp, departs=bool(abs(dT - sd_["mean"]) > 2 * semp))
    for nm, H in (("blue", HB), ("red", HR)):
        for h, k in (("D1-3", "d13"), ("D4-6", "d46")):
            sec[f"{nm}_offset_{k}"] = _stats([o[k]["d_parallax_mas"] for o in H if o[k]["used"]])
    sec["target_minus_gaia_per_half_mas"] = {h: th[h]["parallax_mas"] - tgt["gaia_parallax_mas"] for h in th}
    res = dict(experiment="E29", protocol="docs/protocols/E29.md (frozen 19:36:01 PST, sha256 in data/phase2/e29/freeze.txt)",
               question="does a colour-dependent SPHEREx position error bias E12's parallax of Epsilon Indi B?",
               verdict=verdict, criterion="|mean_red - mean_blue| > 2 SE_D  (SE_D = sqrt(SE_red^2 + SE_blue^2))",
               target=dict(parallax_mas=plx_t, gaia_parallax_mas=tgt["gaia_parallax_mas"], d_parallax_mas=plx_t - tgt["gaia_parallax_mas"],
                           bp_rp=TARGET_BPRP, G=TARGET_G, sigma_fit_mas=sfit, e12_adopted_err_mas=tgt["parallax_err_mas"]["adopted"]),
               primary=prim, descriptive=desc, secondary_detector_split=sec,
               blue_reproduces_e12_max_abs_diff_mas=repro,
               red_set=dict(listed=len(red_all), measured_by_cutoff=len(red), usable=sR["n"], cutoff=CUTOFF + " PST"),
               blue_stars=B, red_stars=R, split_blue=HB, split_red=HR,
               limitations=["reddest comparison star BP-RP %.2f vs target 5.08; no M dwarf reproduces a T-dwarf spectrum across 0.75-5 um" % x.max(),
                            "red stars lie 26-77 arcmin from the target, blue < 18 arcmin: different detector positions",
                            "Ba and Bb unresolved", "brightness matched in Gaia G only (2MASS service unreachable at 19:31 PST)"],
               deviations=[], written=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    json.dump(res, open("data/findings/e29.json", "w"), indent=1, default=float)
    flat = {}

    def put(k, v):
        k = k.replace("-", "").replace(".", "p").replace(" ", "_")
        if isinstance(v, dict):
            for a, b in v.items():
                put(f"{k}_{a}", b)
        elif isinstance(v, list):
            for i, b in enumerate(v):
                put(f"{k}_{i}", b)
        else:
            flat[k] = v if not isinstance(v, (np.floating, np.integer, np.bool_)) else v.item()
    for k in ("verdict", "target", "primary", "descriptive", "red_set", "blue_reproduces_e12_max_abs_diff_mas"):
        put(k, res[k])
    put("secondary", {k: v for k, v in sec.items()})
    json.dump(flat, open("data/findings/e29s.json", "w"), indent=1, default=float)
    print(json.dumps(dict(verdict=verdict, primary=prim, descriptive=desc, repro=repro), indent=1, default=float))
    print(json.dumps({k: v for k, v in sec.items()}, indent=1, default=float))
    figure(B, R, res)


def figure(B, R, res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cb, cr, ck, cg = "#2a78d6", "#eb6834", "#16181d", "#8a8f98"
    fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=200)
    ax.axhline(0, color=cg, lw=1, ls="--", zorder=0)
    for S, c, lab in ((B, cb, "comparison stars, blue (E12)"), (R, cr, "comparison stars, red (E29)")):
        u = [v for v in S if v["used"]]
        ax.errorbar([v["bp_rp"] for v in u], [v["d_parallax_mas"] for v in u], yerr=[v["parallax_err_mas"] for v in u],
                    fmt="o", ms=5, color=c, ecolor=c, elinewidth=0.8, alpha=0.9, capsize=0, label=lab)
        st = _stats([v["d_parallax_mas"] for v in u])
        xm = np.mean([v["bp_rp"] for v in u])
        ax.errorbar([xm], [st["mean"]], yerr=[st["se"]], fmt="s", ms=8, mfc="white", mec=c, ecolor=c, elinewidth=2, capsize=4, zorder=5)
    t = res["target"]
    ax.errorbar([t["bp_rp"]], [t["d_parallax_mas"]], yerr=[t["sigma_fit_mas"]], fmt="*", ms=14, color=ck, capsize=3, label="Epsilon Indi B")
    ax.plot([], [], "s", ms=8, mfc="white", mec=cg, label="set mean ± standard error")
    ax.set_xlabel("Gaia BP − RP colour (mag)")
    ax.set_ylabel("SPHEREx parallax − Gaia parallax (mas)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.2, lw=0.5)
    ax.legend(fontsize=8, frameon=False, loc="best")
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(f"data/findings/e29_colour.{ext}")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    dict(select=select, check=check, measure=measure, fit=fit)[sys.argv[1]]()
