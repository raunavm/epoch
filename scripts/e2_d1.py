"""Secondary, post-hoc: E2 counts and E1 recall at a common k = 5 under deviation D1 rules (referee consistency check).
Nothing planted for the counts (20 E2 fields, data/efields); recall from the existing E1 per-trial kmax records
(data/findings/e1_runs). Resumable: per-field counts in data/findings/e2_d1_runs/<field>.json.

  OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/e2_d1.py
"""
import glob, json, os, sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_e1  # noqa: E402
from orrery import e1  # noqa: E402

K = 5.0
RUNS = "data/findings/e2_d1_runs"
OUT = "data/findings/e2_d1.json"


def one(name):
    p = f"{RUNS}/{name}.json"
    if os.path.exists(p):
        return name
    f = json.load(open(run_e1.THR))["f"]
    fd = run_e1.load_field(name)
    v, lam, det, mjd = e1.frame_arrays(fd)
    maps, _ = e1.method_maps(v, lam, det, mjd, f)
    res = {m: dict(n_components=int(sum(e1.n_components(x, K) for x in ms if x is not None)),
                   n_maps=sum(x is not None for x in ms)) for m, ms in maps.items()}
    json.dump(dict(field=name, n_frames=len(fd["frames"]), results=res), open(p + ".tmp", "w"))
    os.replace(p + ".tmp", p)
    print(name, {m: r["n_components"] for m, r in res.items()}, flush=True)
    return name


def main():
    os.makedirs(RUNS, exist_ok=True)
    names = [f"e2_{i:02d}" for i in range(run_e1.N_TEST)]
    with Pool(run_e1.WORKERS) as p:
        list(p.imap_unordered(one, names))
    R = {n: json.load(open(f"{RUNS}/{n}.json")) for n in names}
    methods = list(R[names[0]]["results"])
    C = {m: np.array([R[n]["results"][m]["n_components"] for n in names], float) for m in methods}
    rng = np.random.default_rng(e1.SEED0)
    idx = rng.integers(0, len(names), size=(2000, len(names)))
    ratios = {}
    for m in ("M2", "M3", "M4", "M5"):
        with np.errstate(divide="ignore", invalid="ignore"):
            r = C[m] / C["M1"]
        ok = np.isfinite(r)
        b = np.array([np.median(r[i][ok[i]]) for i in idx])
        ratios[f"{m}/M1"] = dict(per_field={n: (round(float(x), 4) if np.isfinite(x) else None) for n, x in zip(names, r)},
                                 median=round(float(np.median(r[ok])), 4),
                                 ci95=[round(e1.q(b, .025), 4), round(e1.q(b, .975), 4)],
                                 n_fields_defined=int(ok.sum()))
    # recall at k = 5 from the per-trial kmax records (detected at k=5 <=> kmax >= 5)
    det = {m: np.zeros((len(names), len(e1.LEVELS))) for m in methods}
    tri = {m: np.zeros((len(names), len(e1.LEVELS))) for m in methods}
    for pth in glob.glob("data/findings/e1_runs/e2_*_L*.json"):
        r = json.load(open(pth))
        i, li = names.index(r["field"]), r["level_index"]
        for fr in r["per_frame"]:
            for m in methods:
                if fr.get(m) is not None:
                    tri[m][i, li] += 1
                    det[m][i, li] += fr[m + "_kmax"] >= K
    recall = {}
    for m in methods:
        e = det[m].sum(0) / tri[m].sum(0)
        b = det[m][idx].sum(1) / tri[m][idx].sum(1)
        recall[m] = dict(efficiency=[round(float(x), 4) for x in e],
                         ci95=[[round(e1.q(b[:, l], .025), 4), round(e1.q(b[:, l], .975), 4)] for l in range(len(e1.LEVELS))],
                         n_detected=det[m].sum(0).astype(int).tolist(), n_trials=tri[m].sum(0).astype(int).tolist())
    out = dict(label="secondary: E2 counts and E1 recall at a common k = 5 under D1 rules (post-hoc consistency check "
                     "requested by the referee)",
               rules="deviation D1 (docs/PROTOCOLS.md): blink noise S = |reference (earlier) frame value|; every map "
                     "despiked; positive components only; 8-connected, >= 2 px, map > 5",
               k=K, fields=names, index=run_e1.INDEX, raw_cache=run_e1.ROOT, f=json.load(open(run_e1.THR))["f"],
               n_frames_total=int(sum(R[n]["n_frames"] for n in names)),
               bootstrap=dict(n=2000, seed=e1.SEED0, unit="field",
                              ratio="median over fields of per-field count ratio (fields with M1 = 0 excluded)",
                              recall="pooled detections / pooled trials"),
               counts={m: dict(per_field={n: R[n]["results"][m]["n_components"] for n in names},
                               total=int(C[m].sum()), n_maps=int(sum(R[n]["results"][m]["n_maps"] for n in names)))
                       for m in methods},
               ratios=ratios, levels_snr_frame=list(e1.LEVELS), recall_at_k5=recall,
               recall_source="data/findings/e1_runs per-trial kmax (largest grid k still detected); same trials as E1",
               code_sha256={p: run_e1.sha(p) for p in ("src/orrery/e1.py", "src/orrery/change.py",
                                                        "src/orrery/e1_baselines.py", "scripts/e2_d1.py")})
    json.dump(out, open(OUT, "w"), indent=1)
    for m in methods:
        print(m, "count", out["counts"][m]["total"], "recall", recall[m]["efficiency"], "S/N20 ci95", recall[m]["ci95"][-1])
    for k_, v in ratios.items():
        print(k_, v["median"], v["ci95"], v["n_fields_defined"])


if __name__ == "__main__":
    main()
