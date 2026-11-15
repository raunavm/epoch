"""E14 (docs/protocols/E14.md): blind test of stacking along the orbit on a fresh sample.

Stages (run in this order):
  --draw     SBDB query + seeded draw + screening -> data/phase2/e14/sample.json
  --predict  P_nom = sqrt(sum p^2/sigma^2) and the empty-path noise scale (no target amplitude) -> predictions.json
  (none)     the test: calibrated stacks, singles, split/peak, amplitude ratio, false alarms, verdict, figure
Usage: .venv/bin/python scripts/run_e14.py [--draw | --predict]
"""
import argparse, glob, hashlib, json, os, sys
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from build_reveal import items_for, null_calibration, HALF  # noqa: E402

OUT = "data/phase2/e14"
SEED = 14
N_TARGET, N_RESERVE = 12, 6
E13_IDS = ["11874", "14052", "80253", "68894", "56374", "98397", "92301", "80455"]


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def draw():
    from orrery.truth import sbdb
    from orrery.forced import candidate_frames
    from run_forced import load_index
    from ephem import fetch
    os.makedirs(OUT, exist_ok=True)
    q = f"{OUT}/sbdb_query.json"
    if not os.path.exists(q):
        r = requests.get("https://ssd-api.jpl.nasa.gov/sbdb_query.api", timeout=(10, 120), params={
            "fields": "spkid,pdes,H,diameter,albedo", "sb-kind": "a", "sb-ns": "n", "sb-class": "MBA",
            "sb-cdata": json.dumps({"AND": ["H|RG|15.0|17.0", "diameter|DF", "albedo|DF"]})})
        r.raise_for_status()
        d = r.json()
        d["_retrieved_utc"] = now()
        json.dump(d, open(q, "w"))
    d = json.load(open(q))
    tab = pd.DataFrame(d["data"], columns=d["fields"])
    tab = tab.iloc[np.argsort(tab["pdes"].astype(int).values, kind="stable")].reset_index(drop=True)
    excl = {os.path.basename(f)[:-8] for f in glob.glob("data/forced/*.parquet")}
    for l in open("data/faint_ids.txt"):
        l = l.strip()
        excl |= {"10373", "85284"} if l == "1037385284" else ({l} if l else set())
    open(f"{OUT}/excluded.txt", "w").write("\n".join(sorted(excl)) + "\n")
    order = np.random.default_rng(SEED).permutation(len(tab))
    index = load_index()
    acc, log = [], []
    for k in order:
        p = str(tab.pdes[k])
        if p in excl:
            log.append(dict(pdes=p, reason="excluded"))
            continue
        t = sbdb(p)
        if "neowise" not in str(t.get("diameter_ref") or "").lower():
            log.append(dict(pdes=p, reason="diameter not NEOWISE"))
            continue
        f = fetch(p)
        if f is None:
            log.append(dict(pdes=p, reason="no ephemeris"))
            continue
        eph = pd.read_parquet(f)
        cand = candidate_frames(index, eph)
        v = np.interp(cand["mjd"].values, eph["mjd"].values, eph["V"].values)
        v = v[v < 22]
        if len(v) < 20 or not (19.5 <= np.median(v) <= 21.5):
            log.append(dict(pdes=p, reason=f"n_cand={len(v)} medV={np.median(v) if len(v) else None}"))
            continue
        acc.append(dict(pdes=p, H=t["H"], D=t["diameter"], pV=t["albedo"], n_candidate=int(len(v)),
                        median_V=round(float(np.median(v)), 2)))
        log.append(dict(pdes=p, reason="accepted"))
        print(acc[-1], flush=True)
        if len(acc) >= N_TARGET + N_RESERVE:
            break
    json.dump(dict(seed=SEED, drawn_utc=now(), n_query=len(tab), sample=acc[:N_TARGET], reserves=acc[N_TARGET:],
                   screening_log=log), open(f"{OUT}/sample.json", "w"), indent=1)


def read_forced(f):
    """Forced table without run_forced's error rows (frames missing from the bucket). Error rows turn the detector
    column into float, so items_for would look for stamp files named '_D4.0' and find none; cast back to int."""
    df = pd.read_parquet(f)
    if "error" in df:
        df = df[df["error"].isna()].copy()
    if "detector" in df:
        df["detector"] = df["detector"].astype(int)
    return df


def usable(pdes):
    f = f"data/forced/{pdes}.parquet"
    if not os.path.exists(f):
        return None, None
    df = read_forced(f)
    if len(df) == 0 or "snr" not in df:
        return df, []
    items, _ = items_for(pdes, df)
    return df, items


def predict_one(pdes):
    df, items = usable(pdes)
    if items is None or len(items) < 20:
        return dict(pdes=pdes, n=0 if items is None else len(items), stackable=False)
    p = np.array([i["p"] for i in items])
    s = np.array([i["A_err"] for i in items])
    ok = np.isfinite(p) & np.isfinite(s) & (s > 0)
    P_nom = float(np.sqrt(np.sum(p[ok] ** 2 / s[ok] ** 2)))
    m0, s0, s1, nulls, chk, _ = null_calibration(pdes, df)  # empty paths only
    return dict(pdes=pdes, n=len(items), stackable=True, P_nom=round(P_nom, 3), null_mean=round(m0, 4),
                null_scale=round(s0, 4), single_std=round(s1, 4), P_cal=round(P_nom / s0, 3), e5_check=chk,
                null_snrs_nominal=[round(float(x), 4) for x in nulls])


def predict():
    smp = json.load(open(f"{OUT}/sample.json"))
    ids = [o["pdes"] for o in smp["sample"] + smp["reserves"]]
    pred = {}
    for p in ids + E13_IDS:
        if os.path.exists(f"data/forced/{p}.parquet"):
            pred[p] = predict_one(p)
            print({k: v for k, v in pred[p].items() if k != "null_snrs_nominal"}, flush=True)
    body = json.dumps(pred, sort_keys=True)
    json.dump(dict(written_utc=now(), sha256=hashlib.sha256(body.encode()).hexdigest(), predictions=pred),
              open(f"{OUT}/predictions.json", "w"), indent=1)


def measure(pdes, pr):
    """The test for one object (reads the fitted amplitudes; run only after predictions.json exists)."""
    import types
    import run_e13
    from orrery.stack import combine
    run_e13.pd = types.SimpleNamespace(read_parquet=read_forced)  # run_e13.run reads the table itself
    df, items = usable(pdes)
    snr = np.array([i["snr"] for i in items])
    _, _, a, e = combine(items, HALF)
    m0, s0 = pr["null_mean"], pr["null_scale"]
    nulls = (np.array(pr["null_snrs_nominal"]) - m0) / s0
    out = dict(pdes=pdes, n=len(items), max_single=round(float(np.nanmax(snr) / pr["single_std"]), 2),
               P_nom=pr["P_nom"], P_cal=pr["P_cal"], null_scale=round(s0, 3), e5_passed=pr["e5_check"]["passed"],
               stack_nominal=round(float(a / e), 2), stack=round(float((a / e - m0) / s0), 2),
               amp_ratio=round(float(a), 3), amp_ratio_err=round(float(e * s0), 3), amp_ratio_err_nominal=round(float(e), 3),
               null_calibrated=[round(float(x), 2) for x in nulls], median_V=round(float(df.V.median()), 2))
    if out["stack"] >= 4:
        r = run_e13.run(pdes)
        out["split"] = {k: r[k] for k in ("odd", "even", "first_half_time", "second_half_time")}
        out["peak_offset_px"], out["split_verdict"] = r["peak_offset_px"], r["verdict"]
    return out


def figure(res, e13, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 15, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#0b0b0b",
                         "xtick.color": "#333333", "ytick.color": "#333333"})
    fig, axs = plt.subplots(1, 2, figsize=(15, 7), facecolor="white")
    for ax, key, title in [(axs[0], "P_nom", "Prediction from the brightness model\nand the noise in each picture"),
                           (axs[1], "P_cal", "Same prediction, allowing for the extra noise\nmeasured on empty paths next to the asteroid")]:
        allv = [r[key] for r in res + e13] + [r["stack"] for r in res + e13]
        hi = max(allv) * 1.08 + 1
        lo = min(0, min(r["stack"] for r in res + e13)) - 1
        ax.set_facecolor("white")
        ax.grid(color="#e1e0d9", lw=0.8, zorder=0)
        ax.plot([0, hi], [0, hi], color="#888888", lw=2, ls="-", zorder=1)
        ax.text(hi * 0.6, hi * 0.53, "measured = expected", ha="left", va="top", fontsize=13, color="#555555")
        ax.axhline(5, color="#e34948", lw=2, ls="--", zorder=2)
        if key == "P_nom":
            ax.axvline(7, color="#888888", lw=1.5, ls=":", zorder=1)
            ax.text(7.3, hi * 0.72, "test set:\nexpected ≥ 7 →", fontsize=12.5, color="#555555", va="bottom")
        ax.text(hi * 0.02, 5.25, "5σ: the usual threshold for a detection", fontsize=13, color="#b33636", va="bottom")
        ax.scatter([r[key] for r in res], [r["stack"] for r in res], s=110, color="#2a78d6", edgecolor="white", lw=2,
                   zorder=3, label=f"New asteroids, prediction written first (E14, {len(res)})")
        ax.scatter([r[key] for r in e13], [r["stack"] for r in e13], s=110, facecolor="none", edgecolor="#eb6834", lw=2.2,
                   marker="s", zorder=3, label=f"Earlier asteroids, not blind (E13, {len(e13)})")
        ax.set_xlim(0, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("Expected signal-to-noise of the combined picture")
        ax.set_ylabel("Measured signal-to-noise of the combined picture\n(calibrated on 24 empty paths)")
        ax.set_title(title, fontsize=15)
    axs[0].legend(loc="upper left", fontsize=12.5, frameon=True, framealpha=1, edgecolor="#e1e0d9")
    fig.suptitle("Combining many pictures along each asteroid's orbit: expected vs measured", fontsize=18)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor="white")
    plt.close(fig)


def test():
    from scipy.stats import binomtest, norm
    smp = json.load(open(f"{OUT}/sample.json"))
    P = json.load(open(f"{OUT}/predictions.json"))
    pred = P["predictions"]
    assert hashlib.sha256(json.dumps(pred, sort_keys=True).encode()).hexdigest() == P["sha256"]
    order = [o["pdes"] for o in smp["sample"] + smp["reserves"]]
    # first 12 stackable objects in draw order among those whose photometry completed (protocol (a))
    done = [p for p in order if p in pred]
    analysed = [p for p in done if pred[p]["stackable"]][:N_TARGET]
    not_stackable = [dict(pdes=p, n=pred[p]["n"]) for p in done if not pred[p]["stackable"]]
    res = []
    for p in analysed:
        res.append(measure(p, pred[p]))
        print({k: v for k, v in res[-1].items() if k not in ("null_calibrated", "split")}, flush=True)
        json.dump(dict(partial=True, results=res), open("data/findings/e14.json", "w"), indent=1)
    e13 = [measure(p, pred[p]) for p in E13_IDS if p in pred and pred[p]["stackable"]]
    elig = [r for r in res if r["max_single"] < 5 and r["P_nom"] >= 7]
    n, k = len(elig), sum(r["stack"] >= 5 for r in elig)
    out = dict(protocol="docs/protocols/E14.md", seed=SEED, predictions_written_utc=P["written_utc"],
               predictions_sha256=P["sha256"], n_query=smp["n_query"], sample_drawn=order,
               n_photometry_completed=len(done), n_analysed=len(res), not_stackable=not_stackable,
               not_reached=[p for p in order if p not in pred], n_eligible=n, n_eligible_ge5=k)
    if n:
        ci = binomtest(k, n).proportion_ci(0.95, method="exact")
        ratios = np.array([r["amp_ratio"] for r in elig])
        boot = np.median(np.random.default_rng(SEED).choice(ratios, (10000, n)), axis=1)
        out.update(fraction=round(k / n, 3), fraction_ci95=[round(ci.low, 3), round(ci.high, 3)],
                   median_amp_ratio=round(float(np.median(ratios)), 3),
                   median_amp_ratio_ci95=[round(float(x), 3) for x in np.percentile(boot, [2.5, 97.5])])
        crit_i, crit_ii = k / n >= 0.7, 0.7 <= np.median(ratios) <= 1.3
        out.update(criterion_fraction_ge_0p7=bool(crit_i), criterion_median_ratio_0p7_1p3=bool(crit_ii))
    out["verdict"] = ("INCONCLUSIVE (fewer than 3 eligible)" if n < 3 else
                      "works as predicted" if (out["criterion_fraction_ge_0p7"] and out["criterion_median_ratio_0p7_1p3"])
                      else "does not work as predicted")
    allnull = np.concatenate([r["null_calibrated"] for r in res]) if res else np.array([])
    outer = np.concatenate([r["null_calibrated"][12:] for r in res]) if res else np.array([])
    out["false_alarm"] = dict(n_tracks=int(allnull.size), n_ge5=int((allnull >= 5).sum()),
                              expected_ge5=float(allnull.size * norm.sf(5)), n_ge3=int((allnull >= 3).sum()),
                              expected_ge3=round(float(allnull.size * norm.sf(3)), 2), n_outer=int(outer.size),
                              n_outer_ge5=int((outer >= 5).sum()), n_outer_ge3=int((outer >= 3).sum()),
                              expected_outer_ge3=round(float(outer.size * norm.sf(3)), 2),
                              passed=bool((allnull >= 5).sum() == 0))
    z = np.array([r["stack"] - r["P_cal"] for r in res])
    out["secondary_measured_minus_P_cal"] = dict(mean=round(float(z.mean()), 2), std=round(float(z.std(ddof=1)), 2),
                                                 sem=round(float(z.std(ddof=1) / np.sqrt(len(z))), 2), n=int(len(z)))
    out["all_ge5_split"] = [dict(pdes=r["pdes"], verdict=r.get("split_verdict")) for r in res if r["stack"] >= 5]
    out["results"], out["e13_same_method"] = res, e13
    out["figure"] = "site/data/figs/e14_pred_vs_meas.png"
    out["deviations"] = "see docs/protocols/E14.md, section Deviations"
    json.dump(out, open("data/findings/e14.json", "w"), indent=1)
    figure(res, e13, out["figure"])
    print(json.dumps({k_: v for k_, v in out.items() if k_ not in ("results", "e13_same_method", "sample_drawn")}, indent=1))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    ap = argparse.ArgumentParser()
    ap.add_argument("--draw", action="store_true")
    ap.add_argument("--predict", action="store_true")
    a = ap.parse_args()
    if a.draw:
        draw()
    elif a.predict:
        predict()
    else:
        test()
