"""E13 (docs/protocols/E13.md): split-half, peak-position and brightness checks for every faint asteroid whose
null-calibrated stack is >= 4σ, plus the population statement over all stacked candidates.
Usage: .venv/bin/python scripts/run_e13.py
"""
import json, os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from build_reveal import items_for, e5_scale, detection_image, HALF  # noqa: E402
from orrery.stack import combine, centre  # noqa: E402

OFFS = [(rad, a) for rad in (6.5, 8.5) for a in np.arange(12) * np.pi / 6 + (0 if rad == 6.5 else np.pi / 12)]


def subsets(n):
    idx = np.arange(n)
    return dict(odd=idx[0::2], even=idx[1::2], first_half_time=idx[: n // 2], second_half_time=idx[n // 2:], all=idx)


def run(pdes):
    df = pd.read_parquet(f"data/forced/{pdes}.parquet")
    items, meta = items_for(pdes, df)
    items = sorted(items, key=lambda i: i["mjd"])
    keys = [(i["obs_id"], i["det"]) for i in items]
    sub = subsets(len(items))
    nominal = {k: combine([items[j] for j in s], HALF) for k, s in sub.items()}
    nulls = {k: ([], []) for k in sub}
    for rad, a in OFFS:
        it, _ = items_for(pdes, df, offset=(rad * np.cos(a), rad * np.sin(a)))
        by = {(i["obs_id"], i["det"]): i for i in it}
        j = 0 if rad == 6.5 else 1
        for k, s in sub.items():
            sel = [by[keys[q]] for q in s if keys[q] in by]
            _, _, A, E = combine(sel, HALF)
            nulls[k][j].append(A / E)
    out = dict(pdes=str(pdes), name=meta["name"], n=len(items))
    for k in sub:
        img, _, A, E = nominal[k]
        m, sc, chk = e5_scale(*nulls[k])
        out[k] = dict(n=int(len(sub[k])), nominal=round(float(A / E), 2), calibrated=round(float((A / E - m) / sc), 2),
                      null_mean=round(m, 3), null_scale=round(sc, 3), e5_passed=chk["passed"])
    img = nominal["all"][0]
    det = detection_image(img)
    yy, xx = np.unravel_index(np.nanargmax(det[HALF - 4:HALF + 5, HALF - 4:HALF + 5]), (9, 9))
    out["peak_offset_px"] = round(float(np.hypot(xx - 4, yy - 4)), 2)
    # combine() returns a_hat already as measured / predicted (fraction of the H-based expectation)
    A, E = nominal["all"][2], nominal["all"][3]
    out["amp_ratio_measured_over_predicted"] = round(float(A), 2)
    out["amp_ratio_err"] = round(float(E), 2)
    h1, h2 = out["odd"]["calibrated"], out["even"]["calibrated"]
    out["verdict"] = ("consistent with a real signal" if min(h1, h2) >= 2.5 and abs(h1 - h2) / np.sqrt(2) < 2.5
                      and out["peak_offset_px"] <= 1.5 else "not confirmed by the split test")
    return out


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    cand = pd.read_csv("data/findings/f2_candidates.csv")
    tested = cand[cand["stack"] >= 4].sort_values("stack", ascending=False)
    res = []
    for p in tested["pdes"]:
        r = run(p)
        print(json.dumps(r), flush=True)
        res.append(r)
        json.dump(dict(partial=True, results=res), open("data/findings/e13.json", "w"), indent=1)
    single_ok = cand["max_single"] < 5
    k = int((single_ok & (cand["stack"] >= 5)).sum())
    m = sum(1 for r in res if r["verdict"] == "consistent with a real signal"
            and float(cand.loc[cand.pdes.astype(str) == r["pdes"], "stack"].iloc[0]) >= 5)
    summary = dict(protocol="docs/protocols/E13.md", n_stacked=int(len(cand)), n_no_single_5sigma=int(single_ok.sum()),
                   n_combined_ge5=k, n_ge5_pass_split=m,
                   candidates=cand[["pdes", "n", "max_single", "stack", "V"]].round(2).to_dict("records"),
                   results=res, null_tracks=24, null_note="scale from 12 inner tracks, checked on 12 outer (E5 rule)")
    json.dump(summary, open("data/findings/e13.json", "w"), indent=1)
    print(json.dumps({k_: v for k_, v in summary.items() if k_ not in ("candidates", "results")}))
