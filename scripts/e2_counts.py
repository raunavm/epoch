"""E2 counts (docs/PROTOCOLS.md E2): naive-all (M1), naive-same-detector (M2), corrected (M4) at rev 2's frozen
k = 5, f from data/findings/f1_calibration.json, on the 20 E2 fields of data/efields/e2_fields.json.

  OMP_NUM_THREADS=1 .venv/bin/python scripts/e2_counts.py [--field calib]   (one field only, no summary)

Frames come from the shared native-cutout cache (scripts/e2_cache.py) and are resampled exactly as
scripts/build_field.py:cut() does (bilinear on the native image via the full TAN-SIP WCS, FLAGS hard-bad|OUTLIER
-> NaN through bilinear of a NaN mask, central wavelength from WCS-WAVE at the fractional source pixel).
Per-field results are written as they finish (data/efields/<field>/e2_counts.json), so the run is resumable.
"""
import argparse, json, os, sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from astropy.io import fits  # noqa: E402
from astropy.time import Time  # noqa: E402
from astropy.wcs import WCS  # noqa: E402
from build_field import MASK_BITS, passes  # noqa: E402
from orrery.change import THRESH, bilinear, change_model, grid_world  # noqa: E402
from orrery.data import Frame, _calib, load_stamp  # noqa: E402

ROOT = "data/efields"
CALIB = "data/findings/f1_calibration.json"
CONTROL = "data/findings/f1_ecliptic_control.json"
NBOOT, BOOT_SEED = 2000, 20261115


def resample(st, gra, gdec):
    """build_field.cut() on a cached native cutout. None if the grid centre is off the frame."""
    S = gra.shape[0]
    w = WCS(fits.Header.fromstring(st.wcs_header))
    x, y = w.all_world2pix(gra.ravel(), gdec.ravel(), 0)
    x, y = x.reshape(S, S), y.reshape(S, S)
    c = S // 2
    if not (0 <= x[c, c] <= 2039 and 0 <= y[c, c] <= 2039):
        return None
    xs, ys = x - st.x0, y - st.y0
    v_raw = bilinear(st.image, xs, ys)
    v = v_raw + bilinear(np.where(st.flags & MASK_BITS, np.nan, 0.0), xs, ys)
    fr = Frame.__new__(Frame)
    k = _calib(st.release, st.detector)
    fr.wx, fr.wy, fr.wv = k["wx"], k["wy"], k["wv"]
    lam, _ = fr.wave_at(x, y)
    lam = np.where(np.isfinite(v_raw), lam, np.nan)
    return v.astype(np.float32), lam.astype(np.float32)


def load_field(name):
    fj = json.load(open(f"{ROOT}/{name}/field.json"))
    gra, gdec = grid_world(fj["size_px"], fj["ra"], fj["dec"])
    rows = []
    for fl in fj["frame_list"]:
        st = load_stamp(f"{ROOT}/{name}/{fl['file']}")
        r = resample(st, gra, gdec)
        if r is not None:
            rows.append((st.mjd, st.detector, st.obs_id, r[0], r[1]))
    rows.sort(key=lambda t: (t[0], t[1]))                    # build_field.gather order
    d = dict(mjd=np.array([t[0] for t in rows]), detector=np.array([t[1] for t in rows]),
             obs_id=np.array([t[2] for t in rows]), v=np.array([t[3] for t in rows]),
             lam=np.array([t[4] for t in rows]))
    return fj, d


def count_field(name, f):
    out_p = f"{ROOT}/{name}/e2_counts.json"
    if os.path.exists(out_p):
        return json.load(open(out_p))
    fj, d = load_field(name)
    r = change_model(d["v"], d["lam"], d["detector"], d["mjd"], f=f)
    P = passes(d["mjd"])
    by_det = {}
    for dd in sorted(set(d["detector"].tolist())):
        m = d["detector"] == dd
        by_det[str(dd)] = dict(n_frames=int(m.sum()), naive_all=int(r["n_naive_all"][m].sum()),
                               naive_det=int(r["n_naive_det"][m].sum()), corrected=int(r["n_corr"][m].sum()))
    na, nd, nc = int(r["n_naive_all"].sum()), int(r["n_naive_det"].sum()), int(r["n_corr"].sum())
    rec = dict(field=name, ra=fj["ra"], dec=fj["dec"], b=fj.get("b"), band=fj.get("band"), size_px=fj["size_px"],
               n_indexed=fj["n_indexed"], n_selected=fj["n_selected"], n_cached=fj["n_frames"],
               n_frames=len(d["mjd"]), n_not_saved=len(fj["not_saved"]),
               naive_all=na, naive_det=nd, corrected=nc,
               ratio_corr_over_naive_det=nc / nd if nd else (np.inf if nc else None),
               ratio_corr_over_naive_all=nc / na if na else (np.inf if nc else None),
               frames_per_detector={k: v["n_frames"] for k, v in by_det.items()}, by_detector=by_det,
               n_passes=len(P), passes=[dict(n=len(p), start=Time(d["mjd"][p[0]], format="mjd").iso[:10],
                                             end=Time(d["mjd"][p[-1]], format="mjd").iso[:10]) for p in P])
    json.dump(rec, open(out_p + ".tmp", "w"), indent=1)
    os.replace(out_p + ".tmp", out_p)
    print(f"{name}: N={rec['n_frames']} naive_all={na} naive_det={nd} corrected={nc}", flush=True)
    return rec


def boot(vals, rng):
    vals = np.asarray(vals, float)
    idx = rng.integers(0, len(vals), (NBOOT, len(vals)))
    med = np.median(vals[idx], 1)
    q = lambda a, b: [float(np.percentile(med, a)), float(np.percentile(med, b))]
    return dict(median=float(np.median(vals)), ci68=q(16, 84), ci95=q(2.5, 97.5), n_fields=len(vals))


def figure(recs, ctrl, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    recs = sorted(recs, key=lambda r: abs(r["b"]))
    x = np.arange(len(recs))
    cols = {"naive-all (M1)": "#E69F00", "naive same-detector (M2)": "#56B4E9", "corrected rev 2 (M4)": "#009E73"}
    keys = ["naive_all", "naive_det", "corrected"]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for k, (lab, c) in enumerate(cols.items()):
        ax.bar(x + (k - 1) * 0.27, [max(r[keys[k]], 0.5) for r in recs], 0.27, color=c, label=lab, edgecolor="white", linewidth=0.6)
    ax.set_yscale("log")
    ax.set_xticks(x, [f"{r['field'][3:]}\n{abs(r['b']):.0f}°" for r in recs], fontsize=6.5)
    ax.set_xlabel("E2 field (number, |b|), sorted by Galactic |b|", fontsize=8)
    ax.set_ylabel("change components (sum over frames)", fontsize=8)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_title(f"E2: change counts at k = 5, rev 2 (zero plotted as 0.5). Control field (separate): "
                 f"{ctrl['naive_all']} / {ctrl['naive_det']} / {ctrl['corrected']}", fontsize=8)
    ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(a):
    f = json.load(open(CALIB))["f"]
    if a.field:
        print(json.dumps(count_field(a.field, f), indent=1))
        return
    draw = json.load(open(f"{ROOT}/e2_fields.json"))
    names = [x["field"] for x in draw["fields"]]
    missing = [n for n in names if not os.path.exists(f"{ROOT}/{n}/READY")]
    if missing:
        sys.exit(f"not READY: {missing}")
    recs = [count_field(n, f) for n in names]
    rng = np.random.default_rng(BOOT_SEED)
    summ = {}
    for key in ("ratio_corr_over_naive_det", "ratio_corr_over_naive_all"):
        vals = [r[key] for r in recs]
        undefined = [r["field"] for r in recs if r[key] is None]
        summ[key] = dict(boot([v for v in vals if v is not None], rng), undefined_0_over_0=undefined)
    tot = {k: sum(r[k] for r in recs) for k in ("naive_all", "naive_det", "corrected")}
    c = json.load(open(CONTROL))
    ctrl = dict(field=c["field"], name=c["name"], ra=c["ra"], dec=c["dec"], n_frames=c["n_frames"],
                naive_all=c["n_naive_all"], naive_det=c["n_naive_det"], corrected=c["n_corrected"],
                ratio_corr_over_naive_det=c["n_corrected"] / c["n_naive_det"],
                ratio_corr_over_naive_all=c["n_corrected"] / c["n_naive_all"],
                source=CONTROL, note="computed earlier with scripts/build_field.py on all 221 frames (no cap needed)")
    out = dict(experiment="E2 generality of false-change removal", protocol="docs/PROTOCOLS.md E2",
               method="docs/INTERFACES.md change model revision 2 (orrery.change.change_model, unchanged)",
               k=THRESH, min_pix=2, f=f, f_source=CALIB, index=draw["index"], n_index_frames=draw["n_index_frames"],
               total_frames=sum(r["n_frames"] for r in recs), field_draw="data/efields/e2_fields.json",
               n_draws=draw["n_draws"], n_rejected_lt40_frames=draw["n_rejected_low_frames"],
               n_rejected_band_full=draw["n_rejected_quota_full"],
               bootstrap=dict(n=NBOOT, seed=BOOT_SEED, unit="field", statistic="median of per-field ratios",
                              intervals="percentile 16-84 (68%) and 2.5-97.5 (95%)"),
               summary=summ, pooled=dict(tot, ratio_corr_over_naive_det=tot["corrected"] / tot["naive_det"],
                                         ratio_corr_over_naive_all=tot["corrected"] / tot["naive_all"],
                                         note="secondary: sums over all 20 fields"),
               n_fields_corr_lt_naive_det=sum(r["corrected"] < r["naive_det"] for r in recs),
               n_fields_corr_gt_naive_det=sum(r["corrected"] > r["naive_det"] for r in recs),
               n_fields_corr_lt_naive_all=sum(r["corrected"] < r["naive_all"] for r in recs),
               control_field=ctrl, fields=recs, cache="data/efields (README.md)")
    json.dump(out, open("data/findings/e2.json", "w"), indent=1)
    figure(recs, ctrl, "site/data/figs/fig_e2.png")
    print(json.dumps({k: v for k, v in out.items() if k != "fields"}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--field")
    main(ap.parse_args())
