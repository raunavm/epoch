"""E28 — confirmatory test of the post-hoc R Draconis D6 (and D5) light curve from E9 (deviation D-E9-7) on SPHEREx
pictures NOT used by E9, and a secondary RW Aurigae visit-to-visit test. Protocol: docs/protocols/E28.md.

Stages (in order; `fetch`, `test`, `rwaur`, `fig` must not run before docs/protocols/E28.md is frozen):
  OMP_NUM_THREADS=1 .venv/bin/python scripts/run_e28.py null    # comparison stars only -> data/phase2/e28/null.json
  OMP_NUM_THREADS=1 .venv/bin/python scripts/run_e28.py rwnull  # RW-class comparison stars only -> data/phase2/e28/rw_null.json
  OMP_NUM_THREADS=1 .venv/bin/python scripts/run_e28.py list    # confirmation picture list (metadata only)
  OMP_NUM_THREADS=1 .venv/bin/python scripts/run_e28.py fetch   # R Dra D5/D6 stamps of the confirmation pictures
  OMP_NUM_THREADS=1 .venv/bin/python scripts/run_e28.py test    # R Dra verdict -> data/findings/e28.json
  OMP_NUM_THREADS=1 .venv/bin/python scripts/run_e28.py rwaur   # RW Aur secondary -> data/findings/e28.json
  OMP_NUM_THREADS=1 .venv/bin/python scripts/run_e28.py posthoc # information only (deviation D-E28-2)
  OMP_NUM_THREADS=1 .venv/bin/python scripts/run_e28.py fig     # data/findings/e28_rdra.png/.svg, e28s.json
"""
import hashlib, json, os, sys, time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e9_lib as L  # noqa: E402
import run_e9 as R  # noqa: E402
import e9_targets as T  # noqa: E402

OUT = "data/phase2/e28"
FIND = "data/findings"
PROTO = "docs/protocols/E28.md"
SEED = 2026111528
DETS = (5, 6)
P = T.VSX["rdra"]["period_d"]            # 246.12 d, VSX as retrieved by E9
T_PRED, HALF_WIN = 61073.0, 15.0         # predicted date of maximum and half-width of the window (MJD)
Z_PASS = 5.0
F = json.load(open(f"{R.OUT}/frozen.json"))["classes"]


def dphase(t):
    """Signed distance in days from T_PRED modulo the period."""
    return (t - T_PRED + P / 2) % P - P / 2


def sine_fit(t, y, s):
    """Weighted LSQ y = c0 + c1 cos(2 pi t / P) + c2 sin(2 pi t / P). Returns coefficients, covariance, amplitude A
    (ln units), formal sigma_A (along the fitted direction), t_max (nearest T_PRED), chi2 vs constant and vs sine."""
    ph = 2 * np.pi * t / P
    X = np.stack([np.ones(len(t)), np.cos(ph), np.sin(ph)], 1)
    Xw, yw = X / s[:, None], y / s
    c, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    cov = np.linalg.inv(Xw.T @ Xw)
    A = float(np.hypot(c[1], c[2]))
    g = np.array([0.0, c[1], c[2]]) / A
    sA = float(np.sqrt(g @ cov @ g))
    t0 = np.arctan2(c[2], c[1]) / (2 * np.pi) * P
    t0 = float(t0 + np.round((T_PRED - t0) / P) * P)
    w = 1 / s ** 2
    chi_c = float((((y - (w * y).sum() / w.sum()) / s) ** 2).sum())
    chi_s = float((((y - X @ c) / s) ** 2).sum())
    return dict(c=c.tolist(), cov=cov.tolist(), A=A, sA=sA, t_max=t0, chi2_constant=chi_c, chi2_sine=chi_s, n_bins=int(len(t)))


def light_curve(df, n_cap=None):
    """E9 R-class pipeline unchanged (fillb_7_12, static model with detector-x term, 30-day bins from MJD 60780, E9 swmed
    and sigma(d, n) model, per-star picture-scatter factor f). Returns cells of D5/D6 with n >= 2 and their error."""
    c = F["rdra"]
    tab, _, _ = R.star_a(df, c["column"], "bin", c["method"], 2, c["use_x"])
    if not len(tab):
        return tab
    tab["star"] = "x"
    tab, _ = R.add_inflation(tab, {int(d): v for d, v in c["within_window_scatter_median"].items()})
    tab = tab[(tab.n >= R.MIN_PIC) & tab.det.isin(DETS)].copy()
    model = {int(d): tuple(m) for d, m in c["sigma_model"].items()}
    n_eff = tab.n.values if n_cap is None else np.minimum(tab.n.values, n_cap)
    tab["sigma"] = R.cell_sigma(model, tab.det.values, n_eff) * tab.f.values
    win = L.window_index(df.mjd.values, "bin")
    tab["mjd"] = [float(df.mjd.values[(df.det.values == r.det) & (win == r.win)].mean()) for r in tab.itertuples()]
    tab["y"] = np.log1p(tab.a)
    tab["sy"] = tab.sigma / (1 + tab.a)
    return tab


def fit_star(tab, d):
    q = tab[tab.det == d].sort_values("mjd")
    if len(q) < 5:
        return None
    return sine_fit(q.mjd.values, q.y.values, q.sy.values)


# ---------------------------------------------------------------- calibration (comparison stars only)
def stage_null():
    stars, _ = R.load_class("rdra")
    # n_cap: largest calibrated cell size = 90th percentile of n over the comparison cells of D5/D6
    tabs = {k: light_curve(v) for k, v in stars.items()}
    ns = np.concatenate([t.n.values for t in tabs.values() if len(t)])
    n_cap = int(np.percentile(ns, 90))
    rows = []
    for name, df in stars.items():
        tab = light_curve(df, n_cap)
        for d in DETS:
            r = fit_star(tab, d) if len(tab) else None
            if r is None:
                continue
            sd = np.sqrt(np.diag(r["cov"]))
            rows.append(dict(star=name, det=d, n_bins=r["n_bins"], n_pictures=int(tab[tab.det == d].n.sum()),
                             z1=r["c"][1] / sd[1], z2=r["c"][2] / sd[2], A=r["A"], sA=r["sA"], t_max=r["t_max"],
                             chi2_constant=r["chi2_constant"]))
    t = pd.DataFrame(rows)
    # (i) cell level, all comparison stars: robust spread of (a - star's median a) / sigma per detector
    cells = pd.concat([tb.assign(star=k) for k, tb in ((k, light_curve(v, n_cap)) for k, v in stars.items()) if len(tb)])
    cells["z"] = (cells.y - cells.groupby(["star", "det"]).y.transform("median")) / cells.sy
    k_cell = {d: R.mad(cells.z[cells.det == d]) for d in DETS}
    # (ii) light-curve level, stars with >= 5 bins, D5 and D6 pooled: spread of the cos / sin coefficients in sigma units
    k_coef = R.mad(np.concatenate([t.z1.values, t.z2.values]))
    out = dict(made=time.strftime("%Y-%m-%dT%H:%M:%S%z"), n_cap=n_cap, n_cells_quantiles=np.percentile(ns, [10, 50, 90, 100]).tolist(),
               k_cell={str(d): k_cell[d] for d in DETS}, n_cell_stars={str(d): int(cells[cells.det == d].star.nunique()) for d in DETS},
               n_cells={str(d): int((cells.det == d).sum()) for d in DETS},
               k_coef_pooled=k_coef, n_coef_light_curves=int(len(t)), n_coef_stars=int(t.star.nunique()),
               k={str(d): max(k_cell[d], k_coef) for d in DETS}, n_star_detectors={}, stars={})
    for d in DETS:
        q = t[t.det == d]
        k = out["k"][str(d)]
        zA = q.A / (k * q.sA)
        inwin = np.abs(dphase(q.t_max)) <= HALF_WIN
        out["n_star_detectors"][str(d)] = int(len(q))
        out[f"D{d}"] = dict(n_stars=int(len(q)), median_bins=float(q.n_bins.median()), median_pictures=float(q.n_pictures.median()),
                            zA_max=float(zA.max()), zA_median=float(zA.median()), n_zA_ge5=int((zA >= Z_PASS).sum()),
                            n_full_rule=int(((zA >= Z_PASS) & inwin).sum()), frac_phase_in_window=float(inwin.mean()),
                            A_mag_median=float(1.0857 * q.A.median()), A_mag_max=float(1.0857 * q.A.max()))
    out["stars"] = t.to_dict("records")
    json.dump(out, open(f"{OUT}/null.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "stars"}, indent=1))


# ---------------------------------------------------------------- confirmation picture list (metadata only)
def stage_list():
    import e2_cache
    from e9_select import n_frames
    fj = json.load(open("data/efields/e7_rdra/field.json"))
    df = e2_cache.load_index()[0]
    fr = n_frames(df, fj["ra"], fj["dec"]).sort_values("mjd", kind="stable").reset_index(drop=True)
    used = {f["key"] for f in fj["frame_list"]}
    e9_measured = set(pd.read_parquet(f"{R.OUT}/phot/target_rdra.parquet").key)
    assert e9_measured <= used
    rest = fr[~fr.key.isin(used)]
    det = rest.key.str.extract(r"/(\d)/level2_")[0].astype(int)
    conf = rest[det.isin(DETS).values]
    tm = json.load(open("site/data/timemachine/rdra.json"))["frames"]
    out = dict(n_indexed=int(len(fr)), n_e9_selected=len(used), n_e9_measured=len(e9_measured),
               n_not_in_e9=int(len(rest)), n_not_in_e9_by_detector={str(k): int(v) for k, v in det.value_counts().sort_index().items()},
               n_timemachine=len(tm), confirmation_keys=conf.key.tolist(), confirmation_mjd=conf.mjd.tolist(),
               e9_keys=sorted(used))
    json.dump(out, open(f"{OUT}/confirm_list.json", "w"), indent=1)
    print({k: v for k, v in out.items() if not k.endswith(("keys", "mjd"))})


# ---------------------------------------------------------------- after the freeze
def stage_fetch():
    from concurrent.futures import ThreadPoolExecutor
    import e9_cache as C
    assert os.path.exists(PROTO)
    lst = json.load(open(f"{OUT}/confirm_list.json"))
    g = T.target_gaia("rdra")
    yr = 9.9
    ra = g["ra"] + g["pmra"] * yr / 3.6e6 / np.cos(np.radians(g["dec"]))
    dec = g["dec"] + g["pmdec"] * yr / 3.6e6

    def get(key):
        for i in range(3):
            try:
                return key, C.lean_stamp(key, ra, dec, half=C.HALF)
            except Exception:
                time.sleep(3 * (i + 1))
        return key, "error"

    t = time.time()
    with ThreadPoolExecutor(8) as ex:
        got = list(ex.map(get, lst["confirmation_keys"]))
    S = 2 * C.HALF + 1
    ok = [(k, s) for k, s in got if s is not None and not isinstance(s, str) and s.image.shape == (S, S)]
    d = C.pack(ok, g)
    d["n_selected"], d["n_error"] = len(got), sum(isinstance(s, str) for _, s in got)
    np.savez_compressed(f"{OUT}/rdra_confirm.npz", **d)
    print(f"{len(ok)} of {len(got)} stamps, {d['n_error']} errors, {time.time() - t:.0f}s")


def stage_test():
    assert os.path.exists(PROTO)
    null = json.load(open(f"{OUT}/null.json"))
    pp = f"{OUT}/phot_rdra_confirm.parquet"
    if not os.path.exists(pp):
        L.measure(L.load_star(f"{OUT}/rdra_confirm.npz"), do_a=False).to_parquet(pp)
    df = pd.read_parquet(pp)
    rng = np.random.default_rng(SEED)
    tab = light_curve(df, null["n_cap"])
    nm = {str(int(d)): float(q.n_masked.median()) for d, q in df.groupby("det")}
    res = dict(experiment="E28", protocol=PROTO, protocol_sha256=hashlib.sha256(open(PROTO, "rb").read()).hexdigest(),
               generated_local=time.strftime("%Y-%m-%dT%H:%M:%S%z"), seed=SEED,
               rule=f"PASS in a detector iff A / (k_d sigma_A) >= {Z_PASS} AND fitted date of maximum within {T_PRED} +/- {HALF_WIN} d "
                    f"(modulo P = {P} d); primary = D6, D5 reported with the same rule",
               rdra=dict(period_d=P, predicted_max_mjd=T_PRED, window_half_days=HALF_WIN,
                         n_pictures_confirmation=int(len(df)), n_pictures_by_detector={str(int(d)): int(n) for d, n in df.det.value_counts().sort_index().items()},
                         n_cap=null["n_cap"], k=null["k"], target_median_n_masked=nm,
                         comparison_max_median_n_masked={str(d): F["rdra"]["n_masked_max_comparison"][str(d)] for d in DETS},
                         detectors={}))
    for d in DETS:
        r = fit_star(tab, d)
        q = tab[tab.det == d].sort_values("mjd")
        rec = dict(cells=[dict(mjd=float(x.mjd), n=int(x.n), a=float(x.a), sigma=float(x.sigma), f=float(x.f)) for x in q.itertuples()])
        if r is None:
            rec["verdict"] = "INCONCLUSIVE (fewer than 5 bins)"
        else:
            k = null["k"][str(d)]
            zA = r["A"] / (k * r["sA"])
            mc = []
            for _ in range(2000):
                mc.append(sine_fit(q.mjd.values, q.y.values + rng.normal(0, k * q.sy.values), q.sy.values)["t_max"])
            te = float(np.std(dphase(np.array(mc)) - dphase(r["t_max"])))
            inwin = abs(dphase(r["t_max"])) <= HALF_WIN
            rec.update(fit=r, semi_amplitude_mag=1.0857 * r["A"], semi_amplitude_mag_err=1.0857 * k * r["sA"], z_amplitude=zA,
                       t_max_mjd=r["t_max"], t_max_err_d=te, offset_from_prediction_d=float(dphase(r["t_max"])),
                       phase_in_window=bool(inwin), chi2_constant=r["chi2_constant"], chi2_sine=r["chi2_sine"], dof_sine=r["n_bins"] - 3,
                       verdict="PASS" if (zA >= Z_PASS and inwin) else "FAIL")
        res["rdra"]["detectors"][str(d)] = rec
        print(d, {k: v for k, v in rec.items() if k not in ("cells", "fit")})
    res["rdra"]["verdict"] = "CONFIRMED (D6 passes the frozen rule)" if res["rdra"]["detectors"]["6"].get("verdict") == "PASS" \
        else "NOT CONFIRMED (D6 fails the frozen rule)"
    old = json.load(open(f"{FIND}/e28.json")) if os.path.exists(f"{FIND}/e28.json") else {}
    old.update({k: v for k, v in res.items()})
    json.dump(old, open(f"{FIND}/e28.json", "w"), indent=1)
    print(res["rdra"]["verdict"])


# ---------------------------------------------------------------- RW Aurigae (secondary)
RW_DETS = (1, 2, 3, 4, 5)                # E9 frozen saturation guard left D1-D5 for RW Aur; D6 descriptive only


def rw_stat(df, dets=RW_DETS, use_f=True):
    """E9 RW-class pipeline unchanged (med_4_7, joint model, visits split at gaps > 30 d, E9 per-star picture-scatter
    factor f). Per detector: contrast c_d = (1 + a_2) / (1 + a_1) - 1, error E9 contrast_sigma(d) x f_d.
    Q = sum_d (c_d / error)^2; all-detector contrast = inverse-variance mean with its formal error."""
    c = F["rwaur"]
    tab, _, _ = R.star_a(df, c["column"], "gap", c["method"], 2, c["use_x"])
    if not len(tab):
        return None
    tab["star"] = "x"
    tab, _ = R.add_inflation(tab, {int(d): v for d, v in c["within_window_scatter_median"].items()})
    rows = []
    for d, g in tab[(tab.n >= R.MIN_PIC) & tab.det.isin(dets)].groupby("det"):
        g = g.sort_values("win")
        if len(g) != 2:
            continue
        cd = (1 + g.a.values[1]) / (1 + g.a.values[0]) - 1
        sd = c["contrast_sigma"][str(int(d))] * (float(g.f.max()) if use_f else 1.0)
        rows.append(dict(det=int(d), c=float(cd), sigma=sd, z=float(cd / sd), f=float(g.f.max()),
                         a1=float(g.a.values[0]), a2=float(g.a.values[1]), n1=int(g.n.values[0]), n2=int(g.n.values[1])))
    if not rows:
        return None
    r = pd.DataFrame(rows)
    w = 1 / r.sigma ** 2
    return dict(Q=float((r.z ** 2).sum()), n_det=int(len(r)), c_all=float((r.c * w).sum() / w.sum()),
                c_all_err_formal=float(1 / np.sqrt(w.sum())), per_detector=r.to_dict("records"))


def stage_rwnull():
    from scipy.stats import chi2
    stars, _ = R.load_class("rwaur")
    rows = []
    for name, df in stars.items():
        s = rw_stat(df)
        if s is not None:
            rows.append(dict(star=name, **{k: v for k, v in s.items() if k != "per_detector"}))
    t = pd.DataFrame(rows)
    full = t[t.n_det == len(RW_DETS)]
    k_Q = float(np.median(full.Q) / chi2.median(len(RW_DETS)))
    k_c = R.mad(t.c_all / t.c_all_err_formal)
    out = dict(made=time.strftime("%Y-%m-%dT%H:%M:%S%z"), n_stars=int(len(t)), n_stars_all_detectors=int(len(full)),
               Q_max=float(t.Q.max()), Q_median=float(t.Q.median()), k_Q=k_Q,
               Q_threshold_5sigma=float(k_Q * chi2.isf(2.866515719e-7, len(RW_DETS))), k_c_all=k_c,
               n_Q_over_threshold=int((t.Q > k_Q * chi2.isf(2.866515719e-7, t.n_det)).sum()),
               stars=t.to_dict("records"))
    json.dump(out, open(f"{OUT}/rw_null.json", "w"), indent=1)
    print({k: v for k, v in out.items() if k != "stars"})


def stage_rwaur():
    assert os.path.exists(PROTO)
    null = json.load(open(f"{OUT}/rw_null.json"))
    df = pd.read_parquet(f"{R.OUT}/phot/target_rwaur.parquet")         # E9's measurement of the same 220 pictures
    s = rw_stat(df)
    s6 = rw_stat(df, dets=(6,))
    s_nof = rw_stat(df, use_f=False)
    thr = max(null["Q_threshold_5sigma"], null["Q_max"])
    z_all = s["c_all"] / (null["k_c_all"] * s["c_all_err_formal"])
    rank = int((np.array([x["Q"] for x in null["stars"]]) >= s["Q"]).sum())
    rec = dict(note="Re-test of E9's pictures (no fresh RW Aur data exist); not blind: E9 window values were known when the rule was set.",
               n_pictures=int(len(df)), mjd_visits=[float(df.mjd[L.window_index(df.mjd.values, "gap") == w].mean()) for w in (0, 1)],
               detectors=list(RW_DETS), per_detector=s["per_detector"], Q=s["Q"], Q_threshold=thr,
               n_comparison_stars=null["n_stars"], n_comparison_with_Q_ge_target=rank, comparison_Q_max=null["Q_max"],
               c_all=s["c_all"], c_all_sigma=null["k_c_all"] * s["c_all_err_formal"], z_all=z_all,
               d6_descriptive=s6["per_detector"] if s6 else None,
               without_f_information_only=dict(Q=s_nof["Q"], per_detector=s_nof["per_detector"]),
               verdict="changed between visits more than the comparison stars" if s["Q"] > thr
               else "not more than the comparison stars (Q below the frozen threshold)")
    res = json.load(open(f"{FIND}/e28.json")) if os.path.exists(f"{FIND}/e28.json") else {}
    res["rwaur"] = rec
    json.dump(res, open(f"{FIND}/e28.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in rec.items() if k not in ("per_detector", "without_f_information_only")}, indent=1))
    print(pd.DataFrame(s["per_detector"]).round(3).to_string())
    print(pd.DataFrame(s_nof["per_detector"]).round(3).to_string())


def stage_fig():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from astropy.time import Time
    res = json.load(open(f"{FIND}/e28.json"))
    r = res["rdra"]
    ph = json.load(open(f"{FIND}/e9.json"))["targets"]["rdra"]["post_hoc_guard_off"]
    A_pred = ph["date_of_maximum"][0]["semi_amplitude_mag"] / 1.0857
    k = r["k"]
    model = {int(d): tuple(m) for d, m in F["rdra"]["sigma_model"].items()}
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    tt = np.linspace(60780, 61170, 600)
    dates = lambda m: Time(np.asarray(m), format="mjd").datetime
    for ax, d in zip(axes, (6, 5)):
        rec = r["detectors"][str(d)]
        cells = pd.DataFrame(rec["cells"])
        band = R.cell_sigma(model, [d], [r["n_cap"]])[0] * k[str(d)]
        ax.axhspan(1 - band, 1 + band, color="#c9c8c2", alpha=0.6, lw=0, label="comparison stars, \u00b11\u03c3 per bin")
        if "fit" in rec:
            c = rec["fit"]["c"]
            ax.plot(dates(tt), np.exp(c[0] + c[1] * np.cos(2 * np.pi * tt / P) + c[2] * np.sin(2 * np.pi * tt / P)),
                    color="#2a78d6", lw=2, label="fitted sinusoid (P fixed)")
            c0 = c[0]
        else:
            c0 = 0.0
        ax.plot(dates(tt), np.exp(c0 + A_pred * np.cos(2 * np.pi * (tt - T_PRED) / P)), color="#52514e", lw=1.5,
                ls=(0, (4, 3)), label="predicted (maximum MJD 61073)")
        ax.errorbar(dates(cells.mjd), 1 + cells.a, yerr=cells.sigma * k[str(d)], fmt="o", ms=6, color="#0b0b0b",
                    ecolor="#0b0b0b", elinewidth=1, capsize=0, label="E28 pictures (new)")
        e9 = [x for x in ph["light_curve"] if x["detector"] == d]
        if e9:
            m = np.array(e9[0]["mjd"])
            f = 10 ** (-0.4 * np.array(e9[0]["rel_mag"]))
            ax.plot(dates(m), f, "o", ms=6, mfc="none", mec="#52514e", label="E9 pictures (post hoc)")
        ax.set_ylabel(f"relative flux, D{d} ({'4.4\u20135.0' if d == 6 else '3.8\u20134.4'} \u00b5m)")
        ax.grid(axis="y", color="#e6e5e0", lw=0.8)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, fontsize=9, loc="lower center", ncol=2)
    axes[1].set_xlabel("date")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(f"{FIND}/e28_rdra.png", dpi=200)
    fig.savefig(f"{FIND}/e28_rdra.svg")
    # flat summary
    flat = {}
    def put(prefix, obj):
        if isinstance(obj, dict):
            for kk, v in obj.items():
                put(f"{prefix}_{kk}" if prefix else str(kk), v)
        elif isinstance(obj, (int, float, str, bool)) or obj is None:
            import re
            flat[re.sub(r"[^A-Za-z0-9_]", "_", prefix)] = obj
    for d in ("5", "6"):
        x = {kk: v for kk, v in r["detectors"][d].items() if kk not in ("cells", "fit")}
        put(f"rdra_D{d}", x)
    put("rdra", {kk: v for kk, v in r.items() if kk in ("verdict", "period_d", "predicted_max_mjd", "window_half_days",
                                                    "n_pictures_confirmation", "n_cap", "k", "target_median_n_masked")})
    w = res.get("rwaur", {})
    put("rwaur", {kk: v for kk, v in w.items() if kk in ("verdict", "Q", "Q_threshold", "comparison_Q_max", "c_all", "c_all_sigma",
                                                      "z_all", "n_comparison_stars", "n_comparison_with_Q_ge_target", "n_pictures")})
    ph2 = r.get("post_hoc_information_only", {})
    if ph2:
        flat["rdra_posthoc_D6_z_errors_scaled_by_fit"] = ph2["d6_z_errors_scaled_by_fit_reduced_chi2"]["z"]
        flat["rdra_posthoc_D6_Ftest_p"] = ph2["d6_F_test_sine_vs_constant_data_only_errors"]["p"]
        flat["rdra_posthoc_D6_loo_min_z"] = min(x["z_amplitude"] for x in ph2["d6_leave_one_bin_out"])
        flat["rdra_posthoc_D6_loo_max_abs_offset_d"] = max(abs(x["offset_from_prediction_d"]) for x in ph2["d6_leave_one_bin_out"])
        flat["rdra_posthoc_D5_fixed_phase_scale"] = ph2["d5_fixed_phase_scale_of_predicted_curve"]["scale"]
        flat["rdra_posthoc_D5_fixed_phase_scale_sigma"] = ph2["d5_fixed_phase_scale_of_predicted_curve"]["sigma"]
    for x in w.get("per_detector", []) + (w.get("d6_descriptive") or []):
        flat[f"rwaur_D{x['det']}_contrast"] = x["c"]
        flat[f"rwaur_D{x['det']}_z"] = x["z"]
    flat["protocol_sha256"] = res["protocol_sha256"]
    json.dump(flat, open(f"{FIND}/e28s.json", "w"), indent=1)


def stage_posthoc():
    """Information only, after the verdict (deviation D-E28-2): D6 significance under other error assumptions,
    leave-one-bin-out, saturation footprint per bin; D5 amplitude of the predicted curve with its phase held fixed."""
    from scipy.stats import f as fdist, norm, spearmanr
    null = json.load(open(f"{OUT}/null.json"))
    df = pd.read_parquet(f"{OUT}/phot_rdra_confirm.parquet")
    tab = light_curve(df, null["n_cap"])
    k = null["k"]["6"]
    q = tab[tab.det == 6].sort_values("mjd")
    r = sine_fit(q.mjd.values, q.y.values, q.sy.values)
    dof = r["n_bins"] - 3
    s = np.sqrt(r["chi2_sine"] / dof)
    Fv = ((r["chi2_constant"] - r["chi2_sine"]) / 2) / (r["chi2_sine"] / dof)
    p = float(fdist.sf(Fv, 2, dof))
    loo = []
    for i in range(len(q)):
        m = np.arange(len(q)) != i
        rr = sine_fit(q.mjd.values[m], q.y.values[m], q.sy.values[m])
        loo.append(dict(dropped_bin_mjd=float(q.mjd.values[i]), semi_amplitude_mag=1.0857 * rr["A"], z_amplitude=rr["A"] / (k * rr["sA"]),
                        offset_from_prediction_d=float(dphase(rr["t_max"]))))
    win = L.window_index(df.mjd.values, "bin")
    d6 = df[df.det == 6].assign(w=win[df.det.values == 6])
    nm = d6.groupby("w").agg(mjd=("mjd", "mean"), median_n_masked=("n_masked", "median"), n=("mjd", "size")).reset_index(drop=True)
    rho = spearmanr(np.log(d6.fillb_7_12), d6.n_masked)
    A_pred = json.load(open(f"{FIND}/e9.json"))["targets"]["rdra"]["post_hoc_guard_off"]["date_of_maximum"][0]["semi_amplitude_mag"] / 1.0857
    q5 = tab[tab.det == 5].sort_values("mjd")
    pred = A_pred * np.cos(2 * np.pi * (q5.mjd.values - T_PRED) / P)
    X = np.stack([np.ones(len(q5)), pred], 1) / q5.sy.values[:, None]
    c, *_ = np.linalg.lstsq(X, q5.y.values / q5.sy.values, rcond=None)
    cov = np.linalg.inv(X.T @ X)
    res = json.load(open(f"{FIND}/e28.json"))
    res["rdra"]["post_hoc_information_only"] = dict(
        label="POST-HOC, information only (deviation D-E28-2); does not change the frozen verdict",
        d6_z_errors_scaled_by_fit_reduced_chi2=dict(scale=float(s), z=float(r["A"] / (max(k, s) * r["sA"]))),
        d6_F_test_sine_vs_constant_data_only_errors=dict(F=float(Fv), dof=[2, dof], p=p, one_sided_gaussian_equivalent=float(norm.isf(p))),
        d6_leave_one_bin_out=loo, d6_median_n_masked_per_bin=nm.to_dict("records"),
        d6_spearman_ln_wing_vs_n_masked=dict(rho=float(rho.statistic), p=float(rho.pvalue), n=int(len(d6))),
        d5_fixed_phase_scale_of_predicted_curve=dict(scale=float(c[1]), sigma=float(null["k"]["5"] * np.sqrt(cov[1, 1])), n_bins=int(len(q5)),
                                                     note="1 = the E9 post-hoc D6 amplitude at the predicted phase"))
    json.dump(res, open(f"{FIND}/e28.json", "w"), indent=1)
    print(json.dumps(res["rdra"]["post_hoc_information_only"], indent=1)[:1500])


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    globals()[f"stage_{sys.argv[1]}"]()
