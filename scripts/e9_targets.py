"""E9 stages `freeze` (comparison stars only) and `targets` (after docs/protocols/E9.md is frozen). See run_e9.py."""
import hashlib, json, os, time

import numpy as np
import pandas as pd

import e9_lib as L
import run_e9 as R

OUT = R.OUT
FINDINGS, FIGS = "data/findings", "site/data/figs"
VSX = dict(rdra=dict(name="R Dra", type="M", period_d=246.12, epoch_jd=2457370.0, max_V=6.7, min_V=13.2,
                     source="VSX (VizieR B/vsx/vsx, CfA mirror; AAVSO VSX API), retrieved 2026-11-15"),
           rwaur=dict(name="RW Aur", type="CTTS/ROT+CTTS", period_d=2.64, max_V=9.2, min_V=13.6,
                      source="VSX (VizieR B/vsx/vsx, CfA mirror), retrieved 2026-11-15"))
SEED = 2026111509
INJ = -0.20


def variant_key(v):
    return f"{v['est']}_{v['r_in']}_{v['r_out']}", v["method"], v["use_x"]


def choose(variants):
    """Frozen rule (evaluated on the selection half of the comparison stars): among variants measurable in >= 95% of
    pictures, prefer those with < 1% of comparison windows over the verdict cut, then maximise
    |returned contrast for an injected -20% change| / typical error of the all-detector factor."""
    ok = [v for v in variants if v["frac_unmeasured"] <= 0.05 and v["median_recovered"] is not None]
    for v in ok:
        v["fom"] = abs(v["median_recovered"]) / v["sigma_all"]
    return max(ok, key=lambda v: (v["frac_exceed"] < 0.01, v["fom"]))


def within_scatter(df, col, mode, method, use_x):
    """Per detector: MAD of per-picture ln-flux residuals about the star's own window medians."""
    tab, resid, kept = R.star_a(df, col, mode, method, 2, use_x)
    win = L.window_index(df.mjd.values, mode)
    out = {}
    for d in range(1, 7):
        m = (df.det.values == d) & kept & np.isfinite(resid)
        if m.sum() < R.MIN_STAR_PIC:
            continue
        r = resid[m].copy()
        w = win[m]
        for u in np.unique(w):
            r[w == u] -= np.median(r[w == u])
        out[d] = R.mad(r)
    return out, tab, resid, kept


def contrast_rows(tab):
    """Two-window stars: per detector (1 + a_2) / (1 + a_1) - 1 (second visit relative to first)."""
    rows = []
    for (star, det), g in tab[tab.n >= R.MIN_PIC].groupby(["star", "det"]):
        g = g.sort_values("win")
        if len(g) == 2:
            rows.append(dict(star=star, det=det, c=(1 + g.a.values[1]) / (1 + g.a.values[0]) - 1))
    return pd.DataFrame(rows)


def all_contrast(ad):
    """Two-window stars: all-detector (1 + a_2) / (1 + a_1) - 1 per star."""
    out = []
    for star, g in ad.groupby("star"):
        g = g.sort_values("win")
        if len(g) == 2:
            out.append((1 + g.a.values[1]) / (1 + g.a.values[0]) - 1)
    return np.array(out)


def slope(c_by_det, lam_by_det, sig):
    """Weighted straight line c = g + h ln(lam / 2 um) over detectors -> (g, h)."""
    d = [k for k in c_by_det if k in sig]
    if len(d) < 3:
        return None
    x = np.log(np.array([lam_by_det[k] for k in d]) / 2.0)
    y = np.array([c_by_det[k] for k in d])
    w = 1.0 / np.array([sig[k] for k in d])
    A = np.stack([np.ones(len(d)), x], 1)
    co, *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
    return float(co[0]), float(co[1])


DET_LAM = {d: float(np.mean(e)) for d, e in L.WAVE_EDGES.items()}


def wing_fraction(col):
    """Empirical wing fraction W_d(lam) = wing sum / core sum (r < 3 px), from comparison-star pictures with no
    OVERFLOW/NONLINEAR pixel within 12 px. Per detector: median in 6 wavelength bins + star-to-star scatter."""
    import glob
    rows = []
    for f in sorted(glob.glob(f"{OUT}/phot/*_c*.parquet")):
        df = pd.read_parquet(f)
        if not len(df):
            continue
        m = (df.n_sat == 0) & (df.core_sum > 0) & np.isfinite(df[col]) & (df[col] > 0)
        t = df[m]
        if len(t):
            rows.append(pd.DataFrame(dict(star=os.path.basename(f)[:-8], det=t.det, lam=t.lam, w=t[col] / t.core_sum)))
    out = {}
    if not rows:
        return out
    t = pd.concat(rows, ignore_index=True)
    for d in range(1, 7):
        s = t[t.det == d]
        if len(s) < 30 or s.star.nunique() < 5:
            continue
        lo, hi = L.WAVE_EDGES[d]
        edges = np.linspace(lo - 0.02, hi + 0.02, 7)
        cen, med = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            q = s[(s.lam >= a) & (s.lam < b)]
            if len(q) >= 5:
                cen.append((a + b) / 2), med.append(float(q.w.median()))
        if len(cen) < 2:
            continue
        wi = np.interp(s.lam.values, cen, med)
        per_star = (s.w / wi).groupby(s.star).median()
        out[d] = dict(lam=cen, W=med, n_pictures=int(len(s)), n_stars=int(s.star.nunique()),
                      star_to_star_scatter=R.mad(per_star.values), picture_scatter=R.mad((s.w / wi).values))
    return out


def stage_freeze():
    scan = json.load(open(f"{OUT}/null_scan.json"))
    fro = dict(made=time.strftime("%Y-%m-%dT%H:%M:%S%z"), seed=SEED, min_pictures_per_cell=R.MIN_PIC,
               min_pictures_per_detector=R.MIN_STAR_PIC, classes={})
    for cname, c in R.CLASS.items():
        v = choose(scan[cname]["variants"])
        col, method, use_x = variant_key(v)
        stars, info = R.load_class(cname)
        tab = R.class_table(stars, col, c["mode"], method, 2, use_x)
        ns = R.null_summary(tab)
        sig_every, sig, model, k_all = ns["sig_every"], ns["sig"], ns["model"], ns["k_all"]
        ad, sall, fr, nh, nt = ns["table"], ns["sigma_all_typical"], ns["frac_exceed"], ns["n_exceed"], ns["n_windows"]
        tab, swmed = ns["tab"], ns["swmed"]
        held = set(scan[cname]["held_out_half"])
        fr_h, nh_h, nt_h, _ = R.exceed(tab[tab.star.isin(held)], model, k_all)
        infl = tab.drop_duplicates(["star", "det"])
        n_inflated = int((infl.f > 2).sum())
        tab = tab[tab.det.isin(list(sig))]
        # saturation range covered by the comparison stars (per detector: star-level median number of masked pixels)
        nm = pd.DataFrame([dict(star=n, det=d, nm=float(q.n_masked.median())) for n, df in stars.items()
                           for d, q in df.groupby("det") if len(q) >= R.MIN_STAR_PIC])
        nm_max = {int(d): float(q.nm.max()) for d, q in nm.groupby("det")}
        t2 = tab[tab.n >= R.MIN_PIC].merge(nm, on=["star", "det"])
        sat_dep = {}
        for d, q in t2.groupby("det"):
            if len(q) >= 30:
                lo = q.nm <= q.nm.median()
                sat_dep[int(d)] = dict(nm_median=float(q.nm.median()), sigma_low=R.mad(q.a[lo]), sigma_high=R.mad(q.a[~lo]))
        sw, corr = {d: [] for d in range(1, 7)}, []
        for name, df in stars.items():
            s, _, resid, kept = within_scatter(df, col, c["mode"], method, use_x)
            for d, x in s.items():
                sw[d].append(x)
            win = L.window_index(df.mjd.values, c["mode"])
            r = resid.copy()
            for d in range(1, 7):
                for u in np.unique(win):
                    m = (df.det.values == d) & (win == u) & kept
                    if m.sum() >= 3:
                        r[m] -= np.median(r[m])
                    else:
                        r[m] = np.nan
            mfc = ("mfb" if v["est"].endswith("b") else "mf") + f"_{v['r_in']}_{v['r_out']}"
            q = df.assign(r=r, star=name)[["star", "det", "r", "x", "y", "lam", "bg", "bg_gx", "bg_gy", "r_sat", "n_sat", mfc]]
            q = q.rename(columns={mfc: "masked_frac"})
            q = q.assign(grad=np.hypot(q.bg_gx, q.bg_gy), xfrac=(df.x.values % 1.0), yfrac=(df.y.values % 1.0))
            corr.append(q[np.isfinite(q.r)])
        cc = pd.concat(corr, ignore_index=True)
        from scipy.stats import spearmanr
        dep = {}
        for k in ("x", "y", "grad", "bg", "r_sat", "n_sat", "masked_frac", "xfrac", "yfrac"):
            dep[k] = {}
            for d in range(1, 7):
                s = cc[cc.det == d]
                if len(s) > 50 and s[k].nunique() > 3:
                    rho, p = spearmanr(s[k], s.r)
                    dep[k][d] = [round(float(rho), 3), float(p), int(len(s))]
        # detector-position artefacts (e.g. the band-1 'shell' reported by Karambelkar et al. 2026): picture residuals
        # in a 6 x 6 grid of detector cells, cells with >= 8 pictures from >= 3 stars
        posmap = {}
        for d in range(1, 7):
            q = cc[cc.det == d]
            if len(q) < 100:
                continue
            s0 = R.mad(q.r)
            g = q.assign(cx=(q.x // 340).astype(int), cy=(q.y // 340).astype(int)).groupby(["cx", "cy"])
            cells = [(float(v.r.median()), len(v), float(abs(v.r.median()) / (1.2533 * s0 / np.sqrt(len(v)))), k)
                     for k, v in g if len(v) >= 8 and v.star.nunique() >= 3]
            if cells:
                worst = max(cells, key=lambda t: t[2])
                posmap[d] = dict(n_cells=len(cells), picture_scatter=s0, max_abs_cell_median=float(max(abs(t[0]) for t in cells)),
                                 n_cells_over_3sigma=int(sum(t[2] > 3 for t in cells)),
                                 worst_cell=dict(cell_xy=[int(worst[3][0]), int(worst[3][1])], median=worst[0], n=worst[1], z=worst[2]))
        rec = dict(estimator="B (wing photometry)", column=col, est=v["est"], r_in=v["r_in"], r_out=v["r_out"],
                   background_annulus_px=list(L.BG), method=method, use_x=use_x, window_mode=c["mode"],
                   sigma_null={str(d): s for d, s in sig.items()}, sigma_null_all=sall,
                   sigma_model={str(d): list(m) for d, m in model.items()}, k_all=k_all,
                   held_out=dict(frac_windows_exceeding=fr_h, n_windows_exceeding=nh_h, n_windows=nt_h),
                   sigma_null_every_detector={str(d): s for d, s in sig_every.items()},
                   detectors_not_used_sigma_over_0p15=[int(d) for d in sig_every if d not in sig],
                   n_masked_max_comparison={str(d): x for d, x in nm_max.items()}, sigma_vs_saturation=sat_dep,
                   frac_windows_exceeding=fr, n_windows_exceeding=nh, n_windows=nt,
                   returned_contrast_for_injected=dict(injected=INJ, per_detector=v["recovered"], median=v["median_recovered"]),
                   fom=v["fom"], within_window_scatter_median={str(d): x for d, x in swmed.items()},
                   n_star_detectors_with_error_inflated_over_2=n_inflated, n_star_detectors=int(len(infl)),
                   picture_residual_dependence_spearman=dep, detector_position_map=posmap,
                   n_stars_downloaded=len(info), n_stars_used=len(stars),
                   stars_used=[i for i in info if i["used"]], stars_not_used=[i for i in info if not i["used"]],
                   masked_frac_median=v["masked_frac"], frac_unmeasured=v["frac_unmeasured"])
        # wing sum versus catalogue brightness across the comparison stars (expected slope -0.4 dex / mag if linear)
        import glob
        kmag = {q["name"]: q["K"] for f in glob.glob(f"{OUT}/comparison_draw*.json")
                for cl in json.load(open(f))["classes"].values() for q in cl["accepted"]}
        lin = {}
        for d in range(1, 7):
            xy = [(kmag[n], np.log10(np.nanmedian(df[col].values[df.det.values == d]))) for n, df in stars.items()
                  if n in kmag and (df.det.values == d).sum() >= R.MIN_STAR_PIC]
            xy = np.array([q for q in xy if np.isfinite(q[1])])
            if len(xy) >= 8 and np.ptp(xy[:, 0]) > 0.3:
                pf = np.polyfit(xy[:, 0], xy[:, 1], 1)
                lin[d] = dict(slope_dex_per_mag=float(pf[0]), scatter_dex=R.mad(xy[:, 1] - np.polyval(pf, xy[:, 0])), n_stars=int(len(xy)))
        rec["wing_vs_K"] = lin
        if c["mode"] == "gap":
            ct = contrast_rows(tab)
            csig = {int(d): R.mad(ct.c[ct.det == d]) for d in sorted(ct.det.unique()) if (ct.det == d).sum() >= 10}
            hs = []
            for star, gdf in ct.groupby("star"):
                s = slope(dict(zip(gdf.det, gdf.c)), DET_LAM, csig)
                if s:
                    hs.append(s)
            hs = np.array(hs)
            ac = all_contrast(ad)
            rec.update(contrast_sigma={str(d): s for d, s in csig.items()}, slope_sigma=R.mad(hs[:, 1]),
                       grey_sigma=R.mad(hs[:, 0]), n_slope_stars=int(len(hs)),
                       all_contrast_null=[float(x) for x in ac], all_contrast_sigma=R.mad(ac))
        fro["classes"][cname] = rec
        print(cname, json.dumps({k: rec[k] for k in ("column", "method", "use_x", "sigma_null", "sigma_null_every_detector", "sigma_model", "k_all", "held_out", "sigma_null_all",
                                                     "frac_windows_exceeding", "n_windows", "n_stars_used", "fom",
                                                     "within_window_scatter_median")}, indent=1))
        print(" wing vs K:", {d: (round(x["slope_dex_per_mag"], 2), round(x["scatter_dex"], 2)) for d, x in lin.items()})
        print(" sigma vs saturation:", sat_dep)
        print(" position map:", {d: (x["n_cells"], round(x["max_abs_cell_median"], 3), x["n_cells_over_3sigma"], x["worst_cell"]) for d, x in posmap.items()})
        print(" dependence:", {k: {d: x[0] for d, x in vv.items()} for k, vv in dep.items()})
    cols = sorted({fro["classes"][c]["column"] for c in fro["classes"]})
    fro["wing_fraction"] = {col: {str(d): w for d, w in wing_fraction(col).items()} for col in cols}
    for col in cols:
        print("wing fraction", col, {d: (np.round(w["W"], 4).tolist(), round(w["star_to_star_scatter"], 3), w["n_stars"])
                                     for d, w in fro["wing_fraction"][col].items()})
    json.dump(fro, open(f"{OUT}/frozen.json", "w"), indent=1)


# ---------------------------------------------------------------- targets (only after docs/protocols/E9.md is frozen)
def target_gaia(field):
    fj = json.load(open(f"data/efields/e7_{field}/field.json"))
    g = pd.read_parquet(f"data/efields/e7_{field}/gaia.parquet")
    sep = np.hypot((g.ra - fj["ra"]) * np.cos(np.radians(fj["dec"])), g.dec - fj["dec"]) * 3600
    g = g[sep < 5].sort_values("phot_g_mean_mag")
    r = g.iloc[0]
    return dict(source_id=str(int(r.source_id)), ra=float(r.ra), dec=float(r.dec), pmra=float(np.nan_to_num(r.pmra)),
                pmdec=float(np.nan_to_num(r.pmdec)), G=float(r.phot_g_mean_mag), sep_arcsec=float(sep[g.index[0]]))


def sine_max(t, y, sig, period, rng, n_mc=2000):
    """ln brightness = c + A cos(2 pi (t - t_max) / P), P fixed. Returns t_max (nearest to the data's weighted mean
    date), its Monte Carlo error (points perturbed by sig), semi-amplitude in mag."""
    def fit(yy):
        ph = 2 * np.pi * t / period
        A = np.stack([np.ones(len(t)), np.cos(ph), np.sin(ph)], 1) / sig[:, None]
        c, *_ = np.linalg.lstsq(A, yy / sig, rcond=None)
        return c
    def tmax(c):
        t0 = np.arctan2(c[2], c[1]) / (2 * np.pi) * period
        tm = np.mean(t)
        return t0 + np.round((tm - t0) / period) * period
    c = fit(y)
    t0 = tmax(c)
    mc = np.array([tmax(fit(y + rng.normal(0, sig))) for _ in range(n_mc)])
    d = (mc - t0 + period / 2) % period - period / 2
    return float(t0), float(np.std(d)), float(1.0857 * np.hypot(c[1], c[2]))


def binned_spectrum(df, col, wf, win):
    """Per window: median flux in wavelength bins (bin width ~ lam / 25). Flux [mJy] = wing sum / W_d(lam) x pixel
    solid angle, W from wing_fraction (comparison stars). Error = 1.4826 MAD / sqrt(n) of the pictures in the bin."""
    edges = [0.74]
    while edges[-1] < 5.0:
        edges.append(edges[-1] * 1.04)
    edges = np.array(edges)
    out = []
    f = np.full(len(df), np.nan)
    for d in range(1, 7):
        m = df.det.values == d
        if str(d) in wf:
            f[m] = df[col].values[m] / np.interp(df.lam.values[m], wf[str(d)]["lam"], wf[str(d)]["W"]) \
                   * df.omega.values[m] / 206264.806 ** 2 * 1e9
    for w in np.unique(win):
        rec = dict(window=int(w), mjd_min=float(df.mjd[win == w].min()), mjd_max=float(df.mjd[win == w].max()),
                   n_pictures=int((win == w).sum()), lam_um=[], flux_mJy=[], err_mJy=[], n=[], detector=[],
                   wing_sum_MJy_sr_px=[])
        for d in range(1, 7):
            for a, b in zip(edges[:-1], edges[1:]):
                m = (win == w) & (df.det.values == d) & (df.lam.values >= a) & (df.lam.values < b) & np.isfinite(df[col].values)
                if m.sum() < 1:
                    continue
                v = f[m]
                rec["lam_um"].append(float(np.median(df.lam.values[m]))), rec["detector"].append(d), rec["n"].append(int(m.sum()))
                rec["wing_sum_MJy_sr_px"].append(float(np.median(df[col].values[m])))
                rec["flux_mJy"].append(None if not np.isfinite(np.median(v)) else float(np.median(v)))
                rec["err_mJy"].append(None if (m.sum() < 3 or not np.isfinite(np.median(v))) else float(R.mad(v) / np.sqrt(m.sum())))
        out.append(rec)
    return out


def stage_targets():
    fro = json.load(open(f"{OUT}/frozen.json"))
    proto = "docs/protocols/E9.md"
    assert os.path.exists(proto), "freeze docs/protocols/E9.md first"
    rng = np.random.default_rng(SEED)
    res = dict(experiment="E9", protocol="docs/PROTOCOLS.md E9 + docs/protocols/E9.md",
               protocol_sha256=hashlib.sha256(open(proto, "rb").read()).hexdigest(),
               generated_local=time.strftime("%Y-%m-%dT%H:%M:%S%z"), seed=SEED,
               verdict_rule="changed between visits iff |a_w| > 5 sigma_null (all-detector factor) OR |a_{d,w}| > 5 "
                            "sigma_null(d) in >= 2 detectors (same window), AND < 1% of comparison-star windows exceed "
                            "the same cut; INCONCLUSIVE if any sigma_null used > 0.15",
               targets={})
    spectra = dict(experiment="E9 spectra (for E11)", generated_local=res["generated_local"], targets={})
    for field, c in R.CLASS.items():
        F = fro["classes"][field]
        col, method, use_x, mode = F["column"], F["method"], F["use_x"], F["window_mode"]
        sig_cal = {int(d): s for d, s in F["sigma_null"].items()}
        sig_every = {int(d): s for d, s in F["sigma_null_every_detector"].items()}
        g = target_gaia(field)
        pp = f"{OUT}/phot/target_{field}.parquet"
        if os.path.exists(pp):
            df = pd.read_parquet(pp)
        else:
            df = L.measure(L.load_target(field, g), R.S_D)
            df.to_parquet(pp)
        sw, tab, resid, kept = within_scatter(df, col, mode, method, use_x)
        win = L.window_index(df.mjd.values, mode)
        tab["star"] = field
        # saturation guard: a detector is used only if the target's median number of masked pixels is inside the
        # range covered by the comparison stars in that detector
        t_nm = {int(d): float(q.n_masked.median()) for d, q in df.groupby("det")}
        out_of_range = [d for d in sig_cal if t_nm.get(d, 0.0) > F["n_masked_max_comparison"][str(d)]]
        sig = {d: x for d, x in sig_cal.items() if d not in out_of_range}
        model = {int(d): tuple(m) for d, m in F["sigma_model"].items() if int(d) in sig}
        k_all = F["k_all"]
        swmed = {int(d): x for d, x in F["within_window_scatter_median"].items()}
        tab, _ = R.add_inflation(tab, swmed)
        if out_of_range and len(model) >= 1:            # recalibrate the all-detector factor on the detectors actually used
            stars, _ = R.load_class(field)
            ctab, _ = R.add_inflation(R.class_table(stars, col, mode, method, 2, use_x), swmed)
            adc = R.all_det(ctab, model)
            k_all = R.mad(adc.a / adc.err)
            fr, nh, nt, _ = R.exceed(ctab, model, k_all)
            F = dict(F, sigma_null_all=float(np.median(adc.err) * k_all), frac_windows_exceeding=fr,
                     n_windows_exceeding=nh, n_windows=nt, held_out=None)
        allc = tab[(tab.n >= R.MIN_PIC) & tab.det.isin(list(sig_every))].copy()
        allc["sigma_null"] = allc.det.map(sig_every)
        allc["used"] = allc.det.isin(list(sig))
        allc["mjd"] = [float(df.mjd.values[(df.det.values == r.det) & (win == r.win)].mean()) for r in allc.itertuples()]
        tab = tab[tab.det.isin(list(sig))]
        ok = allc[allc.used].copy()
        ok["sigma_plain"] = R.cell_sigma(model, ok.det.values, ok.n.values) if len(ok) else []
        ok["sigma_null"] = ok.sigma_plain * ok.f
        ok["z"] = ok.a / ok.sigma_null
        ok["z_plain"] = ok.a / ok.sigma_plain
        ok["mjd"] = [float(df.mjd.values[(df.det.values == r.det) & (win == r.win)].mean()) for r in ok.itertuples()]
        ad = R.all_det(tab, model)
        ad["sigma"] = ad.err * k_all
        ad["z"] = ad.a / ad.sigma
        nbig = ok.assign(big=np.abs(ok.z) > 5).groupby("win").big.sum()
        ad["n_det_over_5sigma"] = ad.win.map(nbig).fillna(0).astype(int)
        ad["mjd"] = [float(df.mjd.values[win == w].mean()) for w in ad.win]
        ad["hit"] = (np.abs(ad.z) > 5) | (ad.n_det_over_5sigma >= 2)
        sig_ok = len(sig) >= 2
        null_ok = F["frac_windows_exceeding"] < 0.01
        if not sig_ok:
            verdict = "INCONCLUSIVE (fewer than 2 detectors with comparison-star sigma_null <= 0.15 inside the calibrated saturation range)"
        elif ad.hit.any() and null_ok:
            verdict = "changed between visits"
        elif ad.hit.any():
            verdict = "INCONCLUSIVE (target exceeds the cut, but so do >= 1% of comparison-star windows)"
        else:
            verdict = "no change detected at the frozen threshold"
        # robustness: the same rule with errors inflated by the target's own within-window scatter
        nbig_i = ok.assign(big=np.abs(ok.z_plain) > 5).groupby("win").big.sum()
        rec = dict(name=c["name"], gaia=g, vsx=VSX[field], estimator=F["estimator"], column=col, method=method,
                   n_pictures=int(len(df)), n_pictures_measured=int(np.isfinite(df[col]).sum()),
                   core_flag_fraction=float(df.core_flag.mean()), window_mode=mode,
                   sigma_null={str(d): s for d, s in sig.items()}, sigma_null_all_typical=F["sigma_null_all"],
                   sigma_model=dict(formula="sigma(d, n)^2 = S_d^2 + P_d^2 / n pictures in the cell; all-detector error = k_all / sqrt(sum 1 / sigma^2)",
                                    S_P={str(d): list(m) for d, m in model.items()}, k_all=k_all),
                   detectors_used=sorted(sig), detectors_not_used=dict(
                       sigma_null_over_0p15={str(d): sig_every[d] for d in sig_every if d not in sig_cal},
                       outside_calibrated_saturation_range={str(d): dict(target_median_n_masked=t_nm.get(d), comparison_max=F["n_masked_max_comparison"][str(d)]) for d in out_of_range}),
                   n_masked_median={str(d): x for d, x in t_nm.items()},
                   a_dw_not_used=[dict(detector=int(r.det), window=int(r.win), mjd=float(r.mjd), n_pictures=int(r.n), a=float(r.a),
                                       sigma_null=float(r.sigma_null)) for r in allc[~allc.used].sort_values(["det", "win"]).itertuples()],
                   comparison=dict(n_stars=F["n_stars_used"], n_windows=F["n_windows"],
                                   frac_windows_exceeding=F["frac_windows_exceeding"],
                                   n_windows_exceeding=F["n_windows_exceeding"], held_out_half=F.get("held_out")),
                   a_dw=[dict(detector=int(r.det), window=int(r.win), mjd=r.mjd, n_pictures=int(r.n), a=float(r.a),
                              sigma=float(r.sigma_null), z=float(r.z), error_inflation=float(r.f),
                              z_without_inflation=float(r.z_plain)) for r in ok.sort_values(["det", "win"]).itertuples()],
                   a_w=[dict(window=int(r.win), mjd=r.mjd, a=float(r.a), sigma_null=float(r.sigma), z=float(r.z),
                             n_detectors=int(r.ndet), n_det_over_5sigma=int(r.n_det_over_5sigma), exceeds=bool(r.hit))
                        for r in ad.sort_values("win").itertuples()],
                   within_window_scatter=dict(target={str(d): s for d, s in sw.items()}, comparison_median=F["within_window_scatter_median"]),
                   verdict=verdict,
                   without_error_inflation=dict(note="for information: the cut with sigma(d, n) not multiplied by the target's own picture-scatter factor",
                                                windows_with_2_detectors_over_5sigma=int((nbig_i >= 2).sum())))
        if mode == "gap":
            ct = contrast_rows(tab)
            csig = {int(d): s for d, s in F["contrast_sigma"].items()}
            cd = {int(r.det): float(r.c) for r in ct.itertuples() if int(r.det) in csig}
            sl = slope(cd, DET_LAM, csig)
            w8 = np.array([1 / csig[d] ** 2 for d in cd])
            mean_c = float((np.array(list(cd.values())) * w8).sum() / w8.sum())
            chi2_grey = float(sum(((cd[d] - mean_c) / csig[d]) ** 2 for d in cd))
            rec["colour"] = dict(definition="second visit relative to first, (1 + a_2) / (1 + a_1) - 1, per detector",
                                 contrast=[dict(detector=d, lam_um=DET_LAM[d], c=cd[d], sigma=csig[d]) for d in sorted(cd)],
                                 weighted_mean=mean_c, chi2_about_grey=chi2_grey, dof=len(cd) - 1,
                                 slope_per_ln_lambda=sl[1] if sl else None, slope_sigma_null=F["slope_sigma"],
                                 value_at_2um=sl[0] if sl else None, value_sigma_null=F["grey_sigma"],
                                 n_comparison_stars=F["n_slope_stars"],
                                 all_detector=float(all_contrast(ad)[0]) if len(ad) == 2 else None,
                                 all_detector_sigma_null=F["all_contrast_sigma"])
            # estimator A on the pictures with an unflagged core (cross-check only)
            a = df[(~df.core_flag) & np.isfinite(df.A_mJy)]
            rec["estimator_A_unflagged_core"] = dict(n_pictures=int(len(a)), by_detector_window=[
                dict(detector=int(d), window=int(w), n=int(len(q)), median_mJy=float(q.A_mJy.median()))
                for (d, w), q in a.assign(win=win[a.index]).groupby(["det", "win"])])
        else:
            P = VSX[field]["period_d"]
            e0 = VSX[field]["epoch_jd"] - 2400000.5
            lc, tm = [], []
            for d in sorted(ok.det.unique()):
                q = ok[ok.det == d].sort_values("mjd")
                mag = -2.5 * np.log10(1 + q.a.values)
                merr = 1.0857 * q.sigma_null.values / (1 + q.a.values)
                lc.append(dict(detector=int(d), mjd=q.mjd.tolist(), rel_mag=mag.tolist(), rel_mag_err=merr.tolist(),
                               n_pictures=q.n.tolist(), range_mag=float(mag.max() - mag.min())))
                if len(q) >= 5:
                    t0, te, amp = sine_max(q.mjd.values, np.log1p(q.a.values), q.sigma_null.values / (1 + q.a.values), P, rng)
                    k = np.round((t0 - e0) / P)
                    tm.append(dict(detector=int(d), lam_um=DET_LAM[int(d)], n_bins=int(len(q)), mjd_max=t0, mjd_max_err=te,
                                   semi_amplitude_mag=amp, brightest_bin_mjd=float(q.mjd.values[np.argmin(mag)]),
                                   catalogue_visual_max_mjd=float(e0 + k * P), lag_days=float(t0 - (e0 + k * P)),
                                   lag_phase=float((t0 - (e0 + k * P)) / P)))
            rec["light_curve"] = lc
            rec["date_of_maximum"] = dict(model="ln brightness = c + A cos(2 pi (t - t_max) / P), P fixed to the VSX period; "
                                                "error = Monte Carlo with sigma_null(d)", period_d=P,
                                          catalogue_epoch_mjd=e0, per_detector=tm)
        res["targets"][field] = rec
        wf = fro["wing_fraction"].get(col, {})
        gw = L.window_index(df.mjd.values, "gap")
        spectra["targets"][field] = dict(
            name=c["name"], estimator=f"B wing photometry ({col}); flux = wing sum / W_d(lambda)",
            windows=binned_spectrum(df, col, wf, gw), wing_fraction_calibration=wf,
            detectors_with_calibrated_relative_photometry=sorted(sig),
            detectors_unreliable={str(d): ("comparison-star scatter > 0.15" if d not in sig_cal else "target more saturated than any comparison star")
                                  for d in range(1, 7) if d not in sig},
            window_definition="visit windows split at gaps > 30 d (orrery.e7.windows)",
            absolute_scale_note="Saturated star: the core is flagged, so the flux scale rests on the wing fraction W measured on "
                                "fainter comparison stars with unflagged cores (core sum within 3 px, no aperture correction). "
                                "Treat the absolute level as unreliable (see star_to_star_scatter per detector; detectors with no "
                                "W have flux null and only the uncalibrated wing sum). Ratios between windows at the same "
                                "wavelength do not depend on W.")
        json.dump(res, open(f"{FINDINGS}/e9.json", "w"), indent=1)
        json.dump(spectra, open(f"{FINDINGS}/e9_spectra.json", "w"), indent=1)
        print(field, verdict)
        print(ok[["det", "win", "n", "a", "f", "sigma_null", "z", "z_plain"]].round(3).to_string())
        print(ad.round(3).to_string())


# ---------------------------------------------------------------- figures
INK, MUTED, BLUE, BAND = "#0b0b0b", "#52514e", "#2a78d6", "#c9c8c2"
BAND_NAME = {d: f"{a:.2f}\u2013{b:.2f} \u00b5m" for d, (a, b) in L.WAVE_EDGES.items()}


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 15, "axes.titlesize": 16, "axes.labelsize": 15, "xtick.labelsize": 13,
                         "ytick.labelsize": 13, "figure.facecolor": "white", "axes.facecolor": "white",
                         "savefig.facecolor": "white", "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False,
                         "axes.spines.right": False, "font.family": "DejaVu Sans"})
    return plt


def stage_figs():
    plt = _style()
    from astropy.time import Time
    res = json.load(open(f"{FINDINGS}/e9.json"))["targets"]
    fro = json.load(open(f"{OUT}/frozen.json"))["classes"]
    os.makedirs(FIGS, exist_ok=True)
    if "rdra" in res:
        _fig_rdra(plt, Time, res["rdra"])
    if "rwaur" not in res:
        return
    # ---- RW Aurigae
    t = res["rwaur"]
    col = t["colour"]
    infl = {q["detector"]: q["sigma"] for q in col.get("post_hoc_inflated", {}).get("contrast", [])}
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(14, 6.4), gridspec_kw=dict(width_ratios=[1.35, 1]))
    for cdet in col["contrast"]:
        lo, hi = L.WAVE_EDGES[cdet["detector"]]
        s = 100 * cdet["sigma"]
        e = 100 * infl.get(cdet["detector"], cdet["sigma"])
        ax.fill_between([lo, hi], [-s, -s], [s, s], color=BAND, lw=0, zorder=1)
        ax.plot([lo, hi], [5 * e, 5 * e], color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=2)
        ax.plot([lo, hi], [-5 * e, -5 * e], color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=2)
        ax.errorbar([cdet["lam_um"]], [100 * cdet["c"]], yerr=[e], xerr=[[cdet["lam_um"] - lo], [hi - cdet["lam_um"]]],
                    fmt="o", ms=9, color=BLUE, ecolor=BLUE, elinewidth=2, capsize=0, mec="white", mew=1.5, zorder=4)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xlabel("Wavelength of the colour band (micrometres)")
    ax.set_ylabel("Change in brightness, second visit vs first (%)")
    ax.set_title("RW Aurigae: change between two visits, by colour", loc="left", fontweight="bold")
    ax.plot([], [], "o", ms=9, color=BLUE, label="RW Aurigae")
    ax.fill_between([], [], [], color=BAND, label="typical scatter of ordinary stars")
    ax.plot([], [], color=MUTED, lw=1.2, ls=(0, (4, 3)), label="5 \u00d7 RW Aurigae's error bar (our threshold)")
    ax.set_ylim(-95, 95)
    ax.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=0.9, loc="upper right", fontsize=12.5)
    null = 100 * np.array(fro["rwaur"]["all_contrast_null"])
    lim = max(np.abs(null).max(), abs(100 * col["all_detector"])) * 1.25
    bx.hist(null, bins=np.linspace(-lim, lim, 25), color=BAND, edgecolor="white", linewidth=1.5)
    bx.axvline(100 * col["all_detector"], color=BLUE, lw=3)
    bx.text(100 * col["all_detector"], bx.get_ylim()[1] * 0.97, " RW Aurigae", color=INK, ha="left", va="top", fontsize=14)
    bx.set_xlabel("Change in brightness, all colours together (%)")
    bx.set_ylabel(f"Number of ordinary stars (of {len(null)})")
    bx.set_title("The same test on ordinary stars", loc="left", fontweight="bold")
    d1, d2 = [Time(a["mjd"], format="mjd").strftime("%b %Y") for a in t["a_w"]]
    fig.text(0.01, 0.01, f"SPHEREx visits: {d1} and {d2}. Brightness measured from the star's outer halo of light (its centre is saturated). "
             "\n4.4\u20135.0 \u00b5m not shown: RW Aur is more saturated there than any comparison star.\n"
             f"Grey = {t['comparison']['n_stars']} ordinary stars measured the same way. RW Aurigae's error bars are larger because its own pictures "
             "scatter 1.1\u20133 times more than theirs.\nNo colour band passes 5 times its error.", fontsize=11, color=MUTED)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(f"{FIGS}/e9_rwaur.png", dpi=150)
    plt.close(fig)


def _fig_rdra(plt, Time, t):
    lcs = {q["detector"]: q for q in t["light_curve"]}
    tm = {q["detector"]: q for q in t["date_of_maximum"]["per_detector"]}
    ph = t.get("post_hoc_guard_off", {})
    lph = {q["detector"]: q for q in ph.get("light_curve", []) if q["detector"] not in lcs}
    tph = {q["detector"]: q for q in ph.get("date_of_maximum", []) if q["detector"] in lph}
    P, e0 = t["date_of_maximum"]["period_d"], t["date_of_maximum"]["catalogue_epoch_mjd"]
    allm = np.concatenate([q["mjd"] for q in list(lcs.values()) + list(lph.values())])
    ally = np.concatenate([q["rel_mag"] for q in list(lcs.values()) + list(lph.values())])
    fig, axs = plt.subplots(2, 3, figsize=(15, 8.4), sharex=True, sharey=True)
    ticks = Time([f"{y}-{m:02d}-01" for y, m in ((2025, 5), (2025, 8), (2025, 11), (2026, 2), (2026, 5))]).mjd
    for d, ax in zip(range(1, 7), axs.ravel()):
        ax.set_title(BAND_NAME[d], loc="left")
        s = float(np.median(lcs[d]["rel_mag_err"])) / 1.0857 if d in lcs else np.nan
        if np.isfinite(s):
            ax.axhspan(-1.0857 * s, 1.0857 * s, color=BAND, lw=0, zorder=1)
        ax.axhline(0, color=MUTED, lw=1, zorder=2)
        for k in range(int(np.ceil((allm.min() - 20 - e0) / P)), int(np.floor((allm.max() + 20 - e0) / P)) + 1):
            ax.axvline(e0 + k * P, color=MUTED, lw=1.3, ls=(0, (2, 3)), zorder=2)
        if d in lcs or d in lph:
            q = lcs.get(d, lph.get(d))
            hollow = d not in lcs
            if d in tm or d in tph:
                tt = np.linspace(allm.min() - 10, allm.max() + 10, 300)
                c = tm.get(d, tph.get(d))
                mean = np.mean(q["rel_mag"])
                # sinusoid in magnitudes about the fitted mean level (for display only)
                ph = 2 * np.pi * (np.array(q["mjd"]) - c["mjd_max"]) / P
                off = np.mean(np.array(q["rel_mag"]) + c["semi_amplitude_mag"] * np.cos(ph))
                ax.plot(tt, off - c["semi_amplitude_mag"] * np.cos(2 * np.pi * (tt - c["mjd_max"]) / P), color=BLUE, lw=1.5, alpha=0.55, zorder=3)
            ax.errorbar(q["mjd"], q["rel_mag"], yerr=q["rel_mag_err"], fmt="o", ms=8, color=BLUE, ecolor=BLUE, elinewidth=1.8,
                        mfc="white" if hollow else BLUE, mec=BLUE if hollow else "white", mew=1.6 if hollow else 1.3, zorder=4)
            if hollow:
                ax.text(0.03, 0.97, "for information only: R Dra is more saturated\nhere than any comparison star", transform=ax.transAxes,
                        ha="left", va="top", color=MUTED, fontsize=11.5, bbox=dict(fc="white", ec="none", alpha=0.85, pad=2))
        else:
            nu = [q for q in t["a_dw_not_used"] if q["detector"] == d and q["a"] > -0.99]
            if nu:
                ax.plot([q["mjd"] for q in nu], [-2.5 * np.log10(1 + q["a"]) for q in nu], "o", ms=6, mfc="none", mec=MUTED, zorder=3)
            ax.text(0.03, 0.97, "not reliable in this band: the halo\nmeasurement fails for very bright stars", transform=ax.transAxes,
                    ha="left", va="top", color=MUTED, fontsize=12, bbox=dict(fc="white", ec="none", alpha=0.85, pad=2))
        ax.set_xticks(ticks)
        ax.set_xticklabels(["May\n2025", "Aug", "Nov", "Feb\n2026", "May"])
        ax.grid(axis="y", color="#e6e5e0", lw=0.8, zorder=0)
    pad = 0.15 * (ally.max() - ally.min() + 0.1)
    axs[0, 0].set_ylim(ally.max() + pad, ally.min() - pad)
    axs[0, 0].set_xlim(allm.min() - 15, allm.max() + 15)
    fig.supylabel("Brightness compared with its own average (magnitudes; up = brighter)", fontsize=15, x=0.012)
    h = [plt.Line2D([], [], marker="o", ls="", ms=8, color=BLUE, label="R Draconis, 30-day averages"),
         plt.Line2D([], [], marker="o", ls="", ms=8, mfc="white", mec=BLUE, mew=1.6, label="same, outside the pre-registered range"),
         plt.Rectangle((0, 0), 1, 1, color=BAND, label="typical scatter of ordinary bright stars"),
         plt.Line2D([], [], color=MUTED, lw=1.3, ls=(0, (2, 3)), label="catalogue date of maximum in visible light"),
         plt.Line2D([], [], color=BLUE, lw=1.5, alpha=0.55, label=f"smooth curve with the catalogue period ({P:.0f} days)")]
    fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, fontsize=12.5)
    fig.suptitle("R Draconis: brightness through the year in six infrared colour bands", x=0.01, ha="left", fontweight="bold", fontsize=18)
    fig.tight_layout(rect=(0, 0.09, 1, 0.96))
    fig.savefig(f"{FIGS}/e9_rdra.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- post-hoc, descriptive (deviations D-E9-7/8; never the verdict)
def stage_posthoc():
    """Run after `targets`. (1) R Dra with the saturation guard switched off (detectors 5 and 6 back in), same frozen
    sigma model; (2) RW Aur colour contrast with errors multiplied by the target's own picture-scatter factor f."""
    fro = json.load(open(f"{OUT}/frozen.json"))["classes"]
    res = json.load(open(f"{FINDINGS}/e9.json"))
    rng = np.random.default_rng(SEED + 7)
    # (1) R Dra
    F = fro["rdra"]
    col, method, use_x = F["column"], F["method"], F["use_x"]
    df = pd.read_parquet(f"{OUT}/phot/target_rdra.parquet")
    _, tab, _, _ = within_scatter(df, col, "bin", method, use_x)
    win = L.window_index(df.mjd.values, "bin")
    tab["star"] = "rdra"
    model = {int(d): tuple(m) for d, m in F["sigma_model"].items()}
    swmed = {int(d): x for d, x in F["within_window_scatter_median"].items()}
    tab, _ = R.add_inflation(tab, swmed)
    ok = tab[(tab.n >= R.MIN_PIC) & tab.det.isin(list(model))].copy()
    ok["sigma"] = R.cell_sigma(model, ok.det.values, ok.n.values) * ok.f
    ok["z"] = ok.a / ok.sigma
    ok["mjd"] = [float(df.mjd.values[(df.det.values == r.det) & (win == r.win)].mean()) for r in ok.itertuples()]
    ad = R.all_det(tab, model)
    ad["sigma"] = ad.err * F["k_all"]
    ad["z"] = ad.a / ad.sigma
    nbig = ok.assign(big=np.abs(ok.z) > 5).groupby("win").big.sum()
    ad["n_det_over_5sigma"] = ad.win.map(nbig).fillna(0).astype(int)
    ad["mjd"] = [float(df.mjd.values[win == w].mean()) for w in ad.win]
    P, e0 = VSX["rdra"]["period_d"], VSX["rdra"]["epoch_jd"] - 2400000.5
    lc, tm = [], []
    for d in sorted(ok.det.unique()):
        q = ok[ok.det == d].sort_values("mjd")
        mag = -2.5 * np.log10(1 + q.a.values)
        merr = 1.0857 * q.sigma.values / (1 + q.a.values)
        lc.append(dict(detector=int(d), mjd=q.mjd.tolist(), rel_mag=mag.tolist(), rel_mag_err=merr.tolist(), n_pictures=q.n.tolist(),
                       error_inflation_f=float(q.f.iloc[0]), range_mag=float(mag.max() - mag.min())))
        if len(q) >= 5:
            t0, te, amp = sine_max(q.mjd.values, np.log1p(q.a.values), q.sigma.values / (1 + q.a.values), P, rng)
            k = np.round((t0 - e0) / P)
            chi_flat = float((((np.log1p(q.a.values) - np.average(np.log1p(q.a.values), weights=(1 + q.a.values) ** 2 / q.sigma.values ** 2))
                               / (q.sigma.values / (1 + q.a.values))) ** 2).sum())
            tm.append(dict(detector=int(d), lam_um=DET_LAM[int(d)], n_bins=int(len(q)), mjd_max=t0, mjd_max_err=te,
                           semi_amplitude_mag=amp, chi2_constant=chi_flat, dof_constant=int(len(q) - 1),
                           catalogue_visual_max_mjd=float(e0 + k * P), lag_days=float(t0 - (e0 + k * P))))
    hit = (np.abs(ad.z) > 5) | (ad.n_det_over_5sigma >= 2)
    res["targets"]["rdra"]["post_hoc_guard_off"] = dict(
        label="POST-HOC, descriptive only (deviation D-E9-7): saturation guard switched off so detectors 5 and 6 (target median "
              "masked pixels 137 and 95 vs comparison maxima 133 and 82) are used with the frozen sigma model. Not the verdict.",
        detectors=sorted(int(d) for d in ok.det.unique()),
        a_w=[dict(window=int(r.win), mjd=r.mjd, a=float(r.a), sigma=float(r.sigma), z=float(r.z), n_detectors=int(r.ndet),
                  n_det_over_5sigma=int(r.n_det_over_5sigma)) for r in ad.sort_values("win").itertuples()],
        n_windows_meeting_cut=int(hit.sum()), comparison_frac_windows_meeting_cut=F["frac_windows_exceeding"],
        light_curve=lc, date_of_maximum=tm)
    print("R Dra post hoc:", ad.round(3).to_string(), "\n", [(x["detector"], round(x["mjd_max"]), round(x["mjd_max_err"]), round(x["semi_amplitude_mag"], 2),
                                                        round(x["lag_days"]), round(x["chi2_constant"], 1), x["dof_constant"]) for x in tm])
    # (2) RW Aur colour with target-specific error inflation
    t = res["targets"]["rwaur"]
    fmap = {}
    for q in t["a_dw"]:
        fmap[q["detector"]] = q["error_inflation"]
    csig = {int(d): s * fmap.get(int(d), 1.0) for d, s in fro["rwaur"]["contrast_sigma"].items() if int(d) in fmap}
    cd = {q["detector"]: q["c"] for q in t["colour"]["contrast"] if q["detector"] in csig}
    sl = slope(cd, DET_LAM, csig)
    # slope null with the same inflation: comparison slopes scaled by the median f of the target's detectors
    fmed = float(np.median(list(fmap.values())))
    t["colour"]["post_hoc_inflated"] = dict(
        label="POST-HOC (deviation D-E9-8): contrast errors multiplied by the target's own picture-scatter factor f (as in the verdict rule)",
        f=fmap, contrast=[dict(detector=d, c=cd[d], sigma=csig[d], z=cd[d] / csig[d]) for d in sorted(cd)],
        slope_per_ln_lambda=sl[1] if sl else None, slope_sigma_scaled=fro["rwaur"]["slope_sigma"] * fmed)
    print("RW Aur colour inflated:", t["colour"]["post_hoc_inflated"])
    json.dump(res, open(f"{FINDINGS}/e9.json", "w"), indent=1)
