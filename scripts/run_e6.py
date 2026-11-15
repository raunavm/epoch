"""E6 runner — M6 native-pixel scene reference (docs/PROTOCOLS.md E6, frozen). Resumable. Reuses the E1 harness
(scripts/run_e1.py load_field, orrery.e1 planting / detection / thresholds) unchanged; M6 code is src/orrery/e6.py.

  export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  .venv/bin/python scripts/run_e6.py floor      # f6 on `calib`            -> data/findings/e6_floor.json (+ e6_nogo.md)
  .venv/bin/python scripts/run_e6.py calib      # k6 on `calib`, D1 rules  -> data/findings/e6_threshold.json
  .venv/bin/python scripts/run_e6.py test       # only after e6_threshold.json is committed; 20 E2 fields
  .venv/bin/python scripts/run_e6.py summary    # data/findings/e6.json, site/data/figs/fig_e1_m6.png
"""
import glob, json, os, subprocess, sys, time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_e1  # noqa: E402
from astropy.io import fits  # noqa: E402
from astropy.wcs import WCS  # noqa: E402
from orrery import e1, e6  # noqa: E402
from orrery.change import world_to_grid  # noqa: E402
from orrery.data import load_stamp  # noqa: E402

FLOOR = "data/findings/e6_floor.json"
NOGO = "data/findings/e6_nogo.md"
THR6 = "data/findings/e6_threshold.json"
RUNS6 = "data/findings/e6_runs"
OUT6 = "data/findings/e6.json"
FIG6 = "site/data/figs/fig_e1_m6.png"
F6_GO = 0.05
CODE = ("src/orrery/e6.py", "src/orrery/e1.py", "src/orrery/fit.py", "src/orrery/psf.py", "scripts/run_e6.py",
        "scripts/run_e1.py", "tests/test_e6.py")


def code_sha():
    return {p: run_e1.sha(p) for p in CODE}


def load_field6(name):
    """run_e1.load_field (same frames, same order) + per frame: pipeline variance, WCS, star columns (G < 19)."""
    fd = run_e1.load_field(name)
    files = {fl["key"]: fl["file"] for fl in fd["meta"]["frame_list"]}
    for fr in fd["frames"]:
        st = load_stamp(f"{run_e1.ROOT}/{name}/{files[fr['key']]}")
        fr["var"] = st.variance
        fr["wcs"] = WCS(fits.Header.fromstring(st.wcs_header))
        xy, G = e6.star_xy_native(fd["gaia"], fr["mjd"], fr["wcs"], fr["x0"], fr["y0"])
        fr["star_xy"], fr["star_G"] = xy, G
        fr["cols"] = e6.star_columns(xy, fr["det"], fr["x0"], fr["y0"], fr["img"].shape)
    return fd


def fit_all(fd, extra=None):
    """Per frame scene_fit of [img] (+ extra[e] images). Returns list of (resid, stars, mask)."""
    out = []
    for e, fr in enumerate(fd["frames"]):
        ims = [fr["img"]] + (list(extra[e]) if extra is not None else [])
        out.append(e6.scene_fit(ims, fr["var"], fr["flg"], fr["cols"]))
    return out


# ---------------------------------------------------------------- f6 (noise floor)
def floor_pixels(fd, fr, gmax=16.0):
    """Native pixels within 1 px of G < gmax stars (frame epoch) whose grid position lies on the 48 px grid."""
    S = fd["S"]
    ra, dec, _ = e6.star_radec(fd["gaia"], fr["mjd"], gmax)
    m = np.zeros(fr["img"].shape, bool)
    if len(ra) == 0:
        return m
    i, j = world_to_grid(ra, dec, S, fd["ra0"], fd["dec0"])
    on = (i >= -0.5) & (i <= S - 0.5) & (j >= -0.5) & (j <= S - 0.5)
    x, y = fr["wcs"].all_world2pix(ra[on], dec[on], 0)
    yy, xx = np.indices(m.shape)
    for a, b in zip(x - fr["x0"], y - fr["y0"]):
        m |= np.hypot(xx - a, yy - b) <= 1.0
    return m


def grid_floor_pixels(fd, gmax=16.0):
    """rev 2 (scripts/build_field.py star_pixels): grid pixels within 1 px of G < gmax stars at the mean MJD."""
    S = fd["S"]
    mjd = np.mean([fr["mjd"] for fr in fd["frames"]])
    ra, dec, _ = e6.star_radec(fd["gaia"], mjd, gmax)
    i, j = world_to_grid(ra, dec, S, fd["ra0"], fd["dec0"])
    jj, ii = np.mgrid[0:S, 0:S]
    m = np.zeros((S, S), bool)
    for a, b in zip(i, j):
        m |= np.hypot(ii - a, jj - b) <= 1.0
    return m, len(ra)


def bisect_f(med, target=1.0):
    """rev 2's search (scripts/build_field.py calibrate): geometric bisection on [1e-4, 1], 40 steps."""
    if med(0.0) <= target:
        return 0.0
    lo, hi = 1e-4, 1.0
    for _ in range(40):
        mid = np.sqrt(lo * hi)
        lo, hi = (mid, hi) if med(mid) > target else (lo, mid)
    return float(hi)


def floor():
    if os.path.exists(FLOOR):
        sys.exit(f"{FLOOR} exists; refusing to overwrite")
    t0 = time.time()
    fd = load_field6("calib")
    fits_ = fit_all(fd)
    # native (primary): pooled over frames, pixels used in the fit, not hard-bad/OUTLIER, near G < 16 stars
    R, Sm, sig = [], [], []
    for fr, (r, s, m) in zip(fd["frames"], fits_):
        sel = floor_pixels(fd, fr) & m & ((fr["flg"] & e1.MASK_BITS) == 0)
        R.append(r[0][sel]), Sm.append(s[0][sel]), sig.append(np.full(sel.sum(), fr["sigma_native"]))
    R, Sm, sig = np.concatenate(R), np.concatenate(Sm), np.concatenate(sig)

    def med_native(f):
        return float(np.median(R ** 2 / (sig ** 2 + (f * Sm) ** 2)))

    f6 = bisect_f(med_native)
    # grid (secondary): rev-2 pixel set on the resampled M6 maps (no despike, as rev 2)
    gm, ng16 = grid_floor_pixels(fd)

    def med_grid(f):
        Z = [e1.resample(e6.native_z(r[0], s[0], m, fr["sigma_native"], f), fr["flg"], fr["gx"], fr["gy"])[0][gm]
             for fr, (r, s, m) in zip(fd["frames"], fits_)]
        return float(np.nanmedian(np.concatenate(Z) ** 2))

    f6_grid = bisect_f(med_grid)
    go = bool(f6 < F6_GO)
    nst = [len(fr["star_xy"]) for fr in fd["frames"]]
    out = dict(protocol="docs/PROTOCOLS.md E6 (frozen 2026-11-15 10:10 PST)", field="calib", ra=fd["ra0"],
               dec=fd["dec0"], index=run_e1.INDEX, cache=f"{run_e1.ROOT}/calib", n_frames=len(fd["frames"]),
               frames_dropped_centre_off_detector=len(fd["dropped"]),
               rule="f6 such that the median corrected chi^2 = 1 on pixels within 1 px of Gaia DR3 G < 16 stars, "
                    "M6 map z = resid / sqrt(sigma_e^2 + (f6 |S|)^2), S = star part of the scene model; same "
                    "geometric bisection as rev 2 (scripts/build_field.py calibrate)",
               primary="native pixels (where M6's residual/sigma is defined): per frame, native pixels within 1 px "
                       "of G < 16 stars propagated to the frame epoch and lying on the 48 px grid, used in the fit "
                       "and not hard-bad/OUTLIER; pooled over frames",
               f6=f6, median_chi2_at_f6=med_native(f6), median_chi2_at_0=med_native(0.0), n_pixels=int(R.size),
               secondary_grid=dict(what="rev-2 pixel set: grid pixels within 1 px of G < 16 stars at the mean MJD, "
                                        "on the resampled (e1.resample) M6 maps, no despike",
                                   f6=f6_grid, median_chi2_at_f6=med_grid(f6_grid), median_chi2_at_0=med_grid(0.0),
                                   n_grid_pixels=int(gm.sum()), n_g16_stars=ng16),
               rev2_f=run_e1.f_frozen(), go_threshold=F6_GO, go=go, decision="GO" if go else "NO-GO",
               fit=dict(stars="Gaia DR3 G < 19, proper motion to frame epoch, zone ePSF nearest each star, psf_model "
                              "box 3", weights="pipeline VARIANCE (orrery.fit.fit_stamp)", masked="orrery.data.BAD_BITS",
                        plane="c0 + c1 (x - xm) + c2 (y - ym)", stars_per_frame=[int(min(nst)), int(max(nst))]),
               runtime_s=round(time.time() - t0, 1), generated_local=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               code_sha256=code_sha())
    json.dump(out, open(FLOOR, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "code_sha256"}, indent=1))
    if not go:
        open(NOGO, "w").write(
            f"# E6 — NO-GO (docs/PROTOCOLS.md E6)\n\n"
            f"Measured on `calib` only ({out['n_frames']} frames, {out['n_pixels']} native star pixels): the M6 "
            f"noise floor f6 = {f6:.4f} (median corrected chi^2 = 1 within 1 px of Gaia G < 16 stars; median chi^2 "
            f"at f = 0 is {out['median_chi2_at_0']:.2f}). Secondary, rev-2 grid pixel set: f6 = {f6_grid:.4f}. "
            f"For comparison rev 2's f = {out['rev2_f']:.4f}.\n\n"
            f"The frozen go criterion is f6 < {F6_GO}. It is not met, so per protocol E6 stops here: no threshold, no "
            f"test field opened. Details: data/findings/e6_floor.json.\n")
        print(f"wrote {NOGO}")


# ---------------------------------------------------------------- k6 (threshold)
def calib():
    if os.path.exists(THR6):
        sys.exit(f"{THR6} exists (frozen); refusing to overwrite")
    fl = json.load(open(FLOOR))
    if not fl["go"]:
        sys.exit("E6 is NO-GO")
    f6 = fl["f6"]
    fd = load_field6("calib")
    maps = [e6.m6_grid(e6.native_z(r[0], s[0], m, fr["sigma_native"], f6), fr)
            for fr, (r, s, m) in zip(fd["frames"], fit_all(fd))]
    rate, n = e1.false_rate_curve(maps)
    k = e1.pick_k(rate)
    i = int(np.searchsorted(e1.K_GRID, k - 1e-9))
    out = dict(protocol="docs/PROTOCOLS.md E6 / E1 threshold calibration, D1 rules", field="calib", ra=fd["ra0"],
               dec=fd["dec0"], index=run_e1.INDEX, cache=f"{run_e1.ROOT}/calib", n_frames=len(fd["frames"]),
               frames_dropped_centre_off_detector=len(fd["dropped"]), f6=f6, floor_file=FLOOR,
               floor_sha256=run_e1.sha(FLOOR), target_rate_per_frame=e1.TARGET_RATE,
               k_grid=dict(start=float(e1.K_GRID[0]), stop=float(e1.K_GRID[-1]), step=0.05),
               rule="smallest k on the 0.05 grid with (change components with map > k, 8-connected, >= 2 px, after "
                    "the D1 despike) / (frames with a map) <= 0.01; nothing planted (orrery.e1.false_rate_curve, "
                    "pick_k: same code path as data/findings/e1_thresholds.json)",
               method="M6 native-pixel scene reference (src/orrery/e6.py)",
               M6=dict(k=k, n_maps=n, rate_at_k=float(rate[i]),
                       rate_at_5=float(rate[int(np.searchsorted(e1.K_GRID, 5 - 1e-9))]),
                       curve=[[float(e1.K_GRID[j]), round(float(rate[j]), 5)] for j in range(len(rate))
                              if rate[j] > 0 and (j == 0 or rate[j] != rate[j - 1])]),
               frozen_local=time.strftime("%Y-%m-%dT%H:%M:%S%z"), code_sha256=code_sha())
    json.dump(out, open(THR6, "w"), indent=1)
    print("M6", k, n, out["M6"]["rate_at_k"], out["M6"]["rate_at_5"])
    print(f"wrote {THR6}")


# ---------------------------------------------------------------- test fields
def committed(path):
    r = subprocess.run(["git", "log", "--oneline", "-1", "--", path], capture_output=True, text=True)
    return bool(r.stdout.strip())


def run_field(name):
    p = f"{RUNS6}/{name}.json"
    if os.path.exists(p):
        return name
    fi = int(name.split("_")[1])
    thr = json.load(open(THR6))
    k6, f6 = thr["M6"]["k"], thr["f6"]
    t = time.time()
    fd = load_field6(name)
    S, nl = fd["S"], len(e1.LEVELS)
    pos = [e1.draw_positions(fd, li, fi) for li in range(nl)]
    planted, info = [], []
    for e, fr in enumerate(fd["frames"]):
        pi = [e1.plant(fr, *pos[li][e], e1.LEVELS[li]) for li in range(nl)]
        planted.append([a for a, _ in pi])
        info.append([b for _, b in pi])
    fits_ = fit_all(fd, planted)
    fp, levels = [], [[] for _ in range(nl)]
    for e, (fr, (r, s, m)) in enumerate(zip(fd["frames"], fits_)):
        m0 = e6.m6_grid(e6.native_z(r[0], s[0], m, fr["sigma_native"], f6), fr)
        fp.append(e1.n_components(m0, k6))
        for li in range(nl):
            x, y = pos[li][e]
            v = e1.resample(planted[e][li], fr["flg"], fr["gx"], fr["gy"])[0].astype(np.float32)   # as frame_arrays
            ix, iy = int(np.clip(round(x), 0, S - 1)), int(np.clip(round(y), 0, S - 1))
            rec = dict(x=round(float(x), 3), y=round(float(y), 3), has_data=bool(np.isfinite(v[iy, ix])),
                       flux=info[e][li]["flux"], sumP2=info[e][li]["sumP2"], sigma_native=fr["sigma_native"])
            if rec["has_data"]:
                mm = e6.m6_grid(e6.native_z(r[li + 1], s[li + 1], m, fr["sigma_native"], f6), fr)
                rec["M6"] = e1.detected(mm, k6, x, y)
                rec["M6_kmax"] = e1.kmax_detected(mm, x, y)
            else:
                rec["M6"] = None
            levels[li].append(rec)
    res = [dict(level_index=li, snr=e1.LEVELS[li], seed=e1.SEED0 + 1000 * li + fi,
                M6=dict(n_trials=sum(q["M6"] is not None for q in levels[li]),
                        n_detected=int(sum(bool(q["M6"]) for q in levels[li])), n_no_map=0),
                per_frame=levels[li]) for li in range(nl)]
    rec = dict(field=name, field_index=fi, k6=k6, f6=f6, n_frames=len(fd["frames"]),
               frames_dropped_centre_off_detector=len(fd["dropped"]), obs=[fr["key"] for fr in fd["frames"]],
               stars_per_frame=[len(fr["star_xy"]) for fr in fd["frames"]],
               false_alarm=dict(n_components=int(sum(fp)), n_maps=len(fp), n_frames_with_component=int(sum(c > 0 for c in fp))),
               levels=res, runtime_s=round(time.time() - t, 1))
    json.dump(rec, open(p + ".tmp", "w"), separators=(",", ":"))
    os.replace(p + ".tmp", p)
    print(f"{name}: " + " ".join(f"L{li} {x['M6']['n_detected']}/{x['M6']['n_trials']}" for li, x in enumerate(res))
          + f"  fp {sum(fp)}/{len(fp)}  ({rec['runtime_s']:.0f}s)", flush=True)
    return name


def test():
    if not committed(THR6):
        sys.exit(f"{THR6} is not committed; refusing to open test fields")
    os.makedirs(RUNS6, exist_ok=True)
    names = [f"e2_{i:02d}" for i in range(run_e1.N_TEST)]
    assert all(run_e1.ready(n) for n in names)
    todo = [n for n in names if not os.path.exists(f"{RUNS6}/{n}.json")]
    # biggest star lists first so the long fields do not end up last
    todo.sort(key=lambda n: -len(__import__("pandas").read_parquet(f"{run_e1.ROOT}/{n}/gaia.parquet")))
    if todo:
        with Pool(min(run_e1.WORKERS, len(todo))) as p:
            for n in p.imap_unordered(run_field, todo):
                print(f"done {n}", flush=True)


# ---------------------------------------------------------------- summary
def summary():
    thr = json.load(open(THR6))
    e1thr = json.load(open(run_e1.THR))
    names = [f"e2_{i:02d}" for i in range(run_e1.N_TEST)]
    R6 = {n: json.load(open(f"{RUNS6}/{n}.json")) for n in names if os.path.exists(f"{RUNS6}/{n}.json")}
    fields = sorted(R6)
    others = [m for m in ("M1", "M2", "M3", "M4", "M5") if m in e1thr["methods"]]
    counts = {}
    for n in fields:
        counts[n] = {}
        for li in range(len(e1.LEVELS)):
            r1 = json.load(open(f"{run_e1.RUNS}/{n}_L{li}.json"))["results"]
            c = {m: (r1[m]["n_detected"], r1[m]["n_trials"]) for m in others}
            c["M6"] = (R6[n]["levels"][li]["M6"]["n_detected"], R6[n]["levels"][li]["M6"]["n_trials"])
            counts[n][li] = c
    methods = others + ["M6"]
    eff, eff_b, s50_b = e1.bootstrap(counts, methods)
    i5 = e1.LEVELS.index(5)
    per = {}
    for m in methods:
        sv, cens = e1.s50(eff[m])
        per[m] = dict(
            k=thr["M6"]["k"] if m == "M6" else e1thr["methods"][m]["k"],
            efficiency=[round(float(x), 4) for x in eff[m]],
            efficiency_ci95=[[round(e1.q(eff_b[m][:, l], .025), 4), round(e1.q(eff_b[m][:, l], .975), 4)] for l in range(len(e1.LEVELS))],
            n_detected=[int(sum(counts[n][l][m][0] for n in fields)) for l in range(len(e1.LEVELS))],
            n_trials=[int(sum(counts[n][l][m][1] for n in fields)) for l in range(len(e1.LEVELS))],
            eff_at_snr5=round(float(eff[m][i5]), 4),
            eff_at_snr5_ci95=[round(e1.q(eff_b[m][:, i5], .025), 4), round(e1.q(eff_b[m][:, i5], .975), 4)],
            snr50=None if not np.isfinite(sv) else round(sv, 3), snr50_censored=cens,
            snr50_ci68=[e1.q(s50_b[m], .16), e1.q(s50_b[m], .84)],
            snr50_ci95=[e1.q(s50_b[m], .025), e1.q(s50_b[m], .975)])

    def diff(a, b):
        with np.errstate(invalid="ignore"):
            d = np.where(np.isinf(a) & np.isinf(b), 0.0, a - b)
        return dict(ci68=[e1.q(d, .16), e1.q(d, .84)], ci95=[e1.q(d, .025), e1.q(d, .975)])

    pt = {m: e1.s50(eff[m])[0] for m in methods}
    comp = [m for m in others if m != "M4"]
    best_b = np.min([s50_b[m] for m in comp], 0)
    comparisons = dict(
        M6_minus_M4=dict(point=pt["M6"] - pt["M4"], **diff(s50_b["M6"], s50_b["M4"])),
        M6_minus_best_of_M1_M2_M3_M5=dict(point=pt["M6"] - min(pt[m] for m in comp), **diff(s50_b["M6"], best_b),
                                          note="per replicate min over M1, M2, M3, M5 (conservative), as e1.json"),
        M6_minus_best_of_M1_to_M5=dict(point=pt["M6"] - min(pt[m] for m in others),
                                       **diff(s50_b["M6"], np.min([s50_b[m] for m in others], 0))),
        eff5_M6_minus_M4=dict(point=float(eff["M6"][i5] - eff["M4"][i5]),
                              ci95=[e1.q(eff_b["M6"][:, i5] - eff_b["M4"][:, i5], .025),
                                    e1.q(eff_b["M6"][:, i5] - eff_b["M4"][:, i5], .975)]))
    fa_c = np.array([R6[n]["false_alarm"]["n_components"] for n in fields], float)
    fa_n = np.array([R6[n]["false_alarm"]["n_maps"] for n in fields], float)
    idx = np.random.default_rng(e1.SEED0).integers(0, len(fields), size=(2000, len(fields)))
    fb = fa_c[idx].sum(1) / fa_n[idx].sum(1)
    fp_e1 = json.load(open("data/findings/e1_test_false_rates.json"))["per_method"]
    out = dict(experiment="E6", protocol="docs/PROTOCOLS.md E6 (frozen 2026-11-15 10:10 PST); E1 rules with D1",
               label="stretch; M6 reported under its own name; does not replace rev 2 (M4)",
               index=run_e1.INDEX, raw_cache=run_e1.ROOT, threshold_file=THR6, threshold_sha256=run_e1.sha(THR6),
               floor_file=FLOOR, f6=thr["f6"], k6=thr["M6"]["k"], levels_snr_frame=list(e1.LEVELS),
               fields_used=fields, n_fields=len(fields),
               fields_missing=[n for n in names if n not in fields],
               n_frames_total=int(sum(R6[n]["n_frames"] for n in fields)),
               frames_used={n: R6[n]["n_frames"] for n in fields},
               bootstrap=dict(n=2000, seed=e1.SEED0, unit="field", efficiency="pooled detections / pooled trials",
                              note="same field-resample indices for all methods (paired)"),
               per_method=per, comparisons=comparisons,
               M6_test_false_alarm=dict(rate_per_frame=round(float(fa_c.sum() / fa_n.sum()), 5),
                                        ci95=[round(e1.q(fb, .025), 5), round(e1.q(fb, .975), 5)],
                                        n_components=int(fa_c.sum()), n_maps=int(fa_n.sum()),
                                        calib_rate_at_k=thr["M6"]["rate_at_k"],
                                        per_field={n: [R6[n]["false_alarm"]["n_components"], R6[n]["false_alarm"]["n_maps"]] for n in fields},
                                        others_from_e1_test_false_rates={m: v["rate_per_frame"] for m, v in fp_e1.items()}),
               code_sha256=code_sha(), generated_local=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    json.dump(out, open(OUT6, "w"), indent=1)
    figure(out)
    print(json.dumps({m: dict(k=v["k"], eff=v["efficiency"], s50=v["snr50"], ci95=v["snr50_ci95"]) for m, v in per.items()}, indent=1))
    print(json.dumps(comparisons, indent=1))
    print(json.dumps({k: v for k, v in out["M6_test_false_alarm"].items() if k != "per_field"}, indent=1))


def figure(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"M1": "#E69F00", "M2": "#56B4E9", "M3": "#009E73", "M4": "#0072B2", "M5": "#CC79A7", "M6": "#D55E00"}
    sty = {"M1": "--", "M2": "--", "M3": "-.", "M4": "-", "M5": ":", "M6": "-"}
    mk = {"M6": "s"}
    lab = {"M1": "M1 blink, all frames", "M2": "M2 blink, same detector", "M3": "M3 median, no wavelength term",
           "M4": "M4 change model rev 2", "M5": "M5 wavelength-neighbour median",
           "M6": "M6 native-pixel scene reference (E6)"}
    L = out["levels_snr_frame"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for m, v in out["per_method"].items():
        lo, hi = np.array(v["efficiency_ci95"]).T
        ax.fill_between(L, lo, hi, color=col[m], alpha=0.15, lw=0)
        ax.plot(L, v["efficiency"], sty[m], marker=mk.get(m, "o"), ms=4, color=col[m],
                lw=1.8 if m in ("M4", "M6") else 1.4, label=f"{lab[m]} (k = {v['k']:.2f})")
    ax.axhline(0.5, color="0.6", lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks(L)
    ax.set_xticklabels([str(x) for x in L])
    ax.minorticks_off()
    ax.set_xlabel("planted S/N per frame")
    ax.set_ylabel("detection efficiency")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"E1 + E6: {out['n_fields']} fields, {out['n_frames_total']} frames; equal false-alarm rate "
                 f"(0.01 per frame on calib)", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.99, 0.01, "bands: 95% field-bootstrap intervals (2,000 resamples)", ha="right", fontsize=6, color="0.4")
    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG6), exist_ok=True)
    fig.savefig(FIG6, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    {"floor": floor, "calib": calib, "test": test, "summary": summary}[sys.argv[1]]()
