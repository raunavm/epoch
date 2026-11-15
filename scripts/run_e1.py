"""E1 runner (docs/PROTOCOLS.md, E1). Resumable; reads the raw native-pixel cache data/efields/<field>/ (READY only).

  export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  .venv/bin/python scripts/run_e1.py calib      # thresholds on `calib`, nothing planted -> data/findings/e1_thresholds.json
  .venv/bin/python scripts/run_e1.py test       # every READY e2 field x 5 levels -> data/findings/e1_runs/ (skips done)
  .venv/bin/python scripts/run_e1.py summary    # bootstrap, data/findings/e1.json, site/data/figs/fig_e1.png
  .venv/bin/python scripts/run_e1.py watch      # test + summary on newly READY fields until all 20 are done
"""
import glob, hashlib, json, os, sys, time
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from astropy.io import fits  # noqa: E402
from astropy.wcs import WCS, FITSFixedWarning  # noqa: E402
import warnings  # noqa: E402
from orrery import e1  # noqa: E402
from orrery.change import grid_world  # noqa: E402
from orrery.data import load_stamp  # noqa: E402

warnings.simplefilter("ignore", FITSFixedWarning)
ROOT = "data/efields"
THR = "data/findings/e1_thresholds.json"
RUNS = "data/findings/e1_runs"
OUT = "data/findings/e1.json"
FIG = "site/data/figs/fig_e1.png"
INDEX = "data/index/frames_dr1.parquet"
WORKERS = 3
N_TEST = 20


def f_frozen():
    return json.load(open("data/findings/f1_calibration.json"))["f"]


def load_field(name, S=48):
    """Raw cache -> e1 field dict. Rev-2 frame rule: kept if the grid's centre pixel maps onto the detector."""
    d = f"{ROOT}/{name}"
    meta = json.load(open(f"{d}/field.json"))
    ra0, dec0 = meta["ra"], meta["dec"]
    gra, gdec = grid_world(S, ra0, dec0)
    frames, dropped = [], []
    for fl in meta["frame_list"]:
        st = load_stamp(f"{d}/{fl['file']}")
        w = WCS(fits.Header.fromstring(st.wcs_header))
        x, y = w.all_world2pix(gra.ravel(), gdec.ravel(), 0)
        x, y = x.reshape(S, S), y.reshape(S, S)
        c = S // 2
        if not (0 <= x[c, c] <= 2039 and 0 <= y[c, c] <= 2039):
            dropped.append(fl["key"])
            continue
        frames.append(dict(key=fl["key"], obs_id=st.obs_id, det=st.detector, mjd=st.mjd, x0=st.x0, y0=st.y0,
                           img=st.image, flg=st.flags, gx=x - st.x0, gy=y - st.y0))
    frames.sort(key=lambda r: (r["mjd"], r["det"]))
    gaia = pd.read_parquet(f"{d}/gaia.parquet") if meta.get("gaia") else None
    fd = dict(name=name, ra0=ra0, dec0=dec0, S=S, frames=frames, gaia=gaia, target_xy=None, meta=meta,
              dropped=dropped)
    return e1.prepare(fd)


def ready(name):
    return os.path.exists(f"{ROOT}/{name}/READY")


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


# ---------------------------------------------------------------- calibration
def calib():
    if os.path.exists(THR):
        sys.exit(f"{THR} exists (frozen); refusing to overwrite")
    if not ready("calib"):
        sys.exit("calib not READY")
    f = f_frozen()
    fd = load_field("calib")
    v, lam, det, mjd = e1.frame_arrays(fd)
    maps, r = e1.method_maps(v, lam, det, mjd, f)
    out = dict(protocol="docs/PROTOCOLS.md E1 threshold calibration", field="calib", ra=fd["ra0"], dec=fd["dec0"],
               f=f, n_frames=len(fd["frames"]), frames_dropped_centre_off_detector=len(fd["dropped"]),
               index=INDEX, cache=f"{ROOT}/calib", target_rate_per_frame=e1.TARGET_RATE,
               k_grid=dict(start=float(e1.K_GRID[0]), stop=float(e1.K_GRID[-1]), step=0.05),
               rule="smallest k on the 0.05 grid with (change components with map > k, 8-connected, >= 2 px, after "
                    "the D1 despike) / (frames with a map) <= 0.01; nothing planted",
               deviation="D1 (docs/PROTOCOLS.md, recorded 2026-11-15 ~09:45 PST before any E1 test field): blink noise "
                         "S = |reference (earlier) frame value|; every map despiked (pixel with |m| > 5 zeroed when the "
                         "median |m| over its finite 3x3 neighbourhood is < 2.5; site/js/live/change.js despike); only "
                         "positive components count. Applied identically to M1-M5.",
               m5_name=getattr(e1._m5(), "M5_NAME", None), methods={}, frozen_local=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               code_sha256={"src/orrery/e1.py": sha("src/orrery/e1.py"), "src/orrery/change.py": sha("src/orrery/change.py"),
                            "src/orrery/e1_baselines.py": sha("src/orrery/e1_baselines.py")
                            if os.path.exists("src/orrery/e1_baselines.py") else None})
    for m, ms in maps.items():
        rate, n = e1.false_rate_curve(ms)
        k = e1.pick_k(rate)
        i = int(np.searchsorted(e1.K_GRID, k - 1e-9)) if k is not None else None
        out["methods"][m] = dict(k=k, n_maps=n, rate_at_k=float(rate[i]) if i is not None else None,
                                 rate_at_5=float(rate[int(np.searchsorted(e1.K_GRID, 5 - 1e-9))]),
                                 curve=[[float(e1.K_GRID[j]), round(float(rate[j]), 5)] for j in range(len(rate))
                                        if rate[j] > 0 and (j == 0 or rate[j] != rate[j - 1])])  # change points
        print(m, k, n, flush=True)
    json.dump(out, open(THR, "w"), indent=1)
    print(f"wrote {THR}")


# ---------------------------------------------------------------- test fields
def run_field(name):
    fi = int(name.split("_")[1])
    todo = [li for li in range(len(e1.LEVELS)) if not os.path.exists(f"{RUNS}/{name}_L{li}.json")]
    if not todo:
        return name
    thr = json.load(open(THR))
    k = {m: v["k"] for m, v in thr["methods"].items()}
    f = thr["f"]
    t = time.time()
    fd = load_field(name)
    for li in todo:
        res, pf = e1.run_level(fd, fi, li, f, k)
        rec = dict(field=name, field_index=fi, level_index=li, snr=e1.LEVELS[li],
                   seed=e1.SEED0 + 1000 * li + fi, k=k, f=f, n_frames=len(fd["frames"]),
                   frames_dropped_centre_off_detector=len(fd["dropped"]),
                   obs=[fr["key"] for fr in fd["frames"]], results=res, per_frame=pf)
        p = f"{RUNS}/{name}_L{li}.json"
        json.dump(rec, open(p + ".tmp", "w"), separators=(",", ":"))
        os.replace(p + ".tmp", p)
        print(f"{name} L{li} S/N {e1.LEVELS[li]}: " +
              " ".join(f"{m} {v['n_detected']}/{v['n_trials']}" for m, v in res.items()) +
              f"  ({time.time() - t:.0f}s)", flush=True)
    return name


def test():
    if not os.path.exists(THR):
        sys.exit("thresholds not frozen; run `calib` first")
    os.makedirs(RUNS, exist_ok=True)
    names = [f"e2_{i:02d}" for i in range(N_TEST) if ready(f"e2_{i:02d}")]
    todo = [n for n in names if any(not os.path.exists(f"{RUNS}/{n}_L{li}.json") for li in range(len(e1.LEVELS)))]
    if todo:
        with Pool(min(WORKERS, len(todo))) as p:
            for n in p.imap_unordered(run_field, todo):
                print(f"done {n}", flush=True)
    return names


# ---------------------------------------------------------------- summary
def summary():
    thr = json.load(open(THR))
    runs = {}
    for p in sorted(glob.glob(f"{RUNS}/e2_*_L*.json")):
        r = json.load(open(p))
        runs.setdefault(r["field"], {})[r["level_index"]] = r
    fields = sorted(n for n, v in runs.items() if len(v) == len(e1.LEVELS))
    if not fields:
        sys.exit("no complete fields")
    methods = [m for m in thr["methods"] if all(m in runs[n][li]["results"] for n in fields for li in runs[n])]
    counts = {n: {li: {m: (runs[n][li]["results"][m]["n_detected"], runs[n][li]["results"][m]["n_trials"])
                       for m in methods} for li in range(len(e1.LEVELS))} for n in fields}
    eff, eff_b, s50_b = e1.bootstrap(counts, methods)
    i5 = e1.LEVELS.index(5)
    per_method = {}
    for m in methods:
        sv, cens = e1.s50(eff[m])
        per_method[m] = dict(
            k=thr["methods"][m]["k"],
            efficiency=[round(float(x), 4) for x in eff[m]],
            efficiency_ci68=[[round(e1.q(eff_b[m][:, l], .16), 4), round(e1.q(eff_b[m][:, l], .84), 4)] for l in range(len(e1.LEVELS))],
            efficiency_ci95=[[round(e1.q(eff_b[m][:, l], .025), 4), round(e1.q(eff_b[m][:, l], .975), 4)] for l in range(len(e1.LEVELS))],
            n_detected=[int(sum(counts[n][l][m][0] for n in fields)) for l in range(len(e1.LEVELS))],
            n_trials=[int(sum(counts[n][l][m][1] for n in fields)) for l in range(len(e1.LEVELS))],
            eff_at_snr5=round(float(eff[m][i5]), 4),
            eff_at_snr5_ci95=[round(e1.q(eff_b[m][:, i5], .025), 4), round(e1.q(eff_b[m][:, i5], .975), 4)],
            snr50=None if not np.isfinite(sv) else round(sv, 3), snr50_censored=cens,
            snr50_ci68=[e1.q(s50_b[m], .16), e1.q(s50_b[m], .84)],
            snr50_ci95=[e1.q(s50_b[m], .025), e1.q(s50_b[m], .975)],
            snr50_boot_frac_censored_high=float(np.mean(~np.isfinite(s50_b[m]))))
    comp = [m for m in methods if m != "M4"]
    s4 = s50_b["M4"]
    best_b = np.min([s50_b[m] for m in comp], 0)
    with np.errstate(invalid="ignore"):
        diff = np.where(np.isinf(s4) & np.isinf(best_b), 0.0, s4 - best_b)
    pt = {m: e1.s50(eff[m])[0] for m in methods}
    best_pt = min(comp, key=lambda m: pt[m])
    with np.errstate(invalid="ignore"):
        diff_fixed = np.where(np.isinf(s4) & np.isinf(s50_b[best_pt]), 0.0, s4 - s50_b[best_pt])
    d_pt = pt["M4"] - pt[best_pt] if np.isfinite(pt["M4"]) or np.isfinite(pt[best_pt]) else 0.0
    ci = [e1.q(diff, .025), e1.q(diff, .975)]
    met = bool(pt["M4"] < pt[best_pt] and ci[1] < 0)
    crit = dict(
        rule="docs/PROTOCOLS.md: gain iff M4's S/N50 is lower than the best of M1-M3 and M5 and the 95% field-"
             "bootstrap interval of the difference excludes zero. Difference per bootstrap replicate = S/N50(M4) - "
             "min over M1, M2, M3, M5 of S/N50 in that replicate (conservative); the variant against the fixed "
             "point-estimate-best competitor is also given.",
        best_competitor_point=best_pt, snr50_M4=pt["M4"], snr50_best=pt[best_pt], difference_point=d_pt,
        difference_ci95=ci, difference_ci68=[e1.q(diff, .16), e1.q(diff, .84)],
        difference_vs_fixed_best_ci95=[e1.q(diff_fixed, .025), e1.q(diff_fixed, .975)],
        met=met, statement="MET" if met else "NOT MET")
    frames_used = {n: runs[n][0]["n_frames"] for n in fields}
    out = dict(experiment="E1", protocol="docs/PROTOCOLS.md E1 (frozen)", index=INDEX,
               index_frames=int(json.load(open(f"{ROOT}/{fields[0]}/field.json"))["n_index_frames"]),
               raw_cache=ROOT, calibration_field="calib", thresholds_file=THR, thresholds_sha256=sha(THR),
               f=thr["f"], levels_snr_frame=list(e1.LEVELS), detection_radius_px=e1.DET_RADIUS,
               fields_used=fields, n_fields=len(fields), fields_missing=[f"e2_{i:02d}" for i in range(N_TEST)
                                                                          if f"e2_{i:02d}" not in fields],
               frames_used=frames_used, n_frames_total=int(sum(frames_used.values())),
               frames_used_obs={n: runs[n][0]["obs"] for n in fields},
               methods={"M1": "consecutive blink, all frames in time order", "M2": "consecutive blink, same detector",
                        "M3": "per-pixel per-detector median of the other frames, rev-2 rejection, no wavelength term",
                        "M4": "change model revision 2 (orrery.change.change_model, corr)",
                        "M5": thr.get("m5_name")},
               trial_rule="a trial = a planted frame whose nearest grid pixel to the planted position has data; "
                          "blink methods also need a previous frame (first frame / first of a detector excluded)",
               bootstrap=dict(n=2000, seed=e1.SEED0, unit="field", efficiency="pooled detections / pooled trials"),
               per_method=per_method, gain_criterion=crit,
               code_sha256={p: sha(p) for p in ("src/orrery/e1.py", "src/orrery/change.py", "scripts/run_e1.py",
                                                 "src/orrery/e1_baselines.py") if os.path.exists(p)},
               generated_local=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    json.dump(out, open(OUT, "w"), indent=1)
    figure(out)
    print(json.dumps({m: dict(k=v["k"], eff=v["efficiency"], s50=v["snr50"], ci95=v["snr50_ci95"])
                      for m, v in per_method.items()}, indent=1))
    print(json.dumps(crit, indent=1))


def figure(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"M1": "#E69F00", "M2": "#56B4E9", "M3": "#009E73", "M4": "#0072B2", "M5": "#CC79A7"}   # Okabe-Ito; line style = second encoding
    sty = {"M1": "--", "M2": "--", "M3": "-.", "M4": "-", "M5": ":"}
    lab = {"M1": "M1 blink, all frames", "M2": "M2 blink, same detector", "M3": "M3 median, no wavelength term",
           "M4": "M4 change model rev 2", "M5": "M5 wavelength-neighbour median"}
    L = out["levels_snr_frame"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for m, v in out["per_method"].items():
        lo, hi = np.array(v["efficiency_ci95"]).T
        ax.fill_between(L, lo, hi, color=col[m], alpha=0.15, lw=0)
        ax.plot(L, v["efficiency"], sty[m], marker="o", ms=4, color=col[m], lw=1.8 if m == "M4" else 1.4,
                label=f"{lab[m]} (k = {v['k']:.2f})")
    ax.axhline(0.5, color="0.6", lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks(L)
    ax.set_xticklabels([str(x) for x in L])
    ax.minorticks_off()
    ax.set_xlabel("planted S/N per frame")
    ax.set_ylabel("detection efficiency")
    ax.set_ylim(0, 1.02)
    c = out["gain_criterion"]
    ax.set_title(f"E1: {out['n_fields']} fields, {out['n_frames_total']} frames; equal false-alarm rate "
                 f"(0.01 per frame)\nsensitivity-gain criterion {c['statement']}", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(0.99, 0.01, "bands: 95% field-bootstrap intervals (2,000 resamples)", ha="right", fontsize=6, color="0.4")
    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, dpi=150)
    plt.close(fig)


def watch(deadline_hours=3.0):
    t0 = time.time()
    while True:
        names = test()
        if any(os.path.exists(p) for p in glob.glob(f"{RUNS}/e2_*_L4.json")):
            summary()
        if len(names) == N_TEST or time.time() - t0 > deadline_hours * 3600:
            break
        time.sleep(60)


if __name__ == "__main__":
    {"calib": calib, "test": test, "summary": summary, "watch": watch}[sys.argv[1]]()
