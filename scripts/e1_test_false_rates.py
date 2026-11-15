"""E1 secondary, post-hoc (no thresholds changed): false positive change components per frame on the 20 E2 test
fields with NOTHING planted, each method at its frozen k (data/findings/e1_thresholds.json), D1 rules exactly.
Resumable: per-field results in data/findings/e1_fp_runs/<field>.json.

  OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/e1_test_false_rates.py
"""
import json, os, sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_e1  # noqa: E402
from orrery import e1  # noqa: E402

RUNS = "data/findings/e1_fp_runs"
OUT = "data/findings/e1_test_false_rates.json"


def one(name):
    p = f"{RUNS}/{name}.json"
    if os.path.exists(p):
        return name
    thr = json.load(open(run_e1.THR))
    fd = run_e1.load_field(name)
    v, lam, det, mjd = e1.frame_arrays(fd)
    maps, _ = e1.method_maps(v, lam, det, mjd, thr["f"])
    res = {}
    for m, ms in maps.items():
        k = thr["methods"][m]["k"]
        n = [e1.n_components(x, k) for x in ms if x is not None]
        res[m] = dict(k=k, n_components=int(sum(n)), n_maps=len(n), n_frames_with_component=int(sum(c > 0 for c in n)))
    json.dump(dict(field=name, n_frames=len(fd["frames"]), results=res), open(p + ".tmp", "w"))
    os.replace(p + ".tmp", p)
    print(name, {m: (r["n_components"], r["n_maps"]) for m, r in res.items()}, flush=True)
    return name


def main():
    os.makedirs(RUNS, exist_ok=True)
    names = [f"e2_{i:02d}" for i in range(run_e1.N_TEST)]
    assert all(run_e1.ready(n) for n in names)
    with Pool(run_e1.WORKERS) as p:
        list(p.imap_unordered(one, names))
    R = [json.load(open(f"{RUNS}/{n}.json")) for n in names]
    thr = json.load(open(run_e1.THR))
    methods = list(R[0]["results"])
    rng = np.random.default_rng(e1.SEED0)
    idx = rng.integers(0, len(R), size=(2000, len(R)))
    per = {}
    for m in methods:
        c = np.array([r["results"][m]["n_components"] for r in R], float)
        n = np.array([r["results"][m]["n_maps"] for r in R], float)
        b = c[idx].sum(1) / n[idx].sum(1)
        per[m] = dict(k=thr["methods"][m]["k"], rate_per_frame=round(float(c.sum() / n.sum()), 5),
                      ci95=[round(e1.q(b, .025), 5), round(e1.q(b, .975), 5)],
                      ci68=[round(e1.q(b, .16), 5), round(e1.q(b, .84), 5)],
                      n_components=int(c.sum()), n_maps=int(n.sum()),
                      calib_rate_at_k=thr["methods"][m]["rate_at_k"],
                      per_field={r["field"]: [r["results"][m]["n_components"], r["results"][m]["n_maps"]] for r in R})
    out = dict(label="secondary, post-hoc, no thresholds changed",
               what="false positive change components per frame (map) with NOTHING planted on the 20 E2 test fields, "
                    "each method at its frozen k; deviation D1 rules (blink S = |reference frame|, despike, positive "
                    "components, 8-connected, >= 2 px)",
               thresholds_file=run_e1.THR, thresholds_sha256=run_e1.sha(run_e1.THR), f=thr["f"],
               fields=names, n_frames_total=int(sum(r["n_frames"] for r in R)), index=run_e1.INDEX,
               bootstrap=dict(n=2000, seed=e1.SEED0, unit="field", rate="pooled components / pooled maps"),
               target_rate_per_frame=e1.TARGET_RATE, per_method=per,
               code_sha256={p: run_e1.sha(p) for p in ("src/orrery/e1.py", "src/orrery/change.py",
                                                        "src/orrery/e1_baselines.py", "scripts/e1_test_false_rates.py")})
    json.dump(out, open(OUT, "w"), indent=1)
    for m, v in per.items():
        print(m, v["k"], v["rate_per_frame"], v["ci95"], v["n_components"], v["n_maps"])


if __name__ == "__main__":
    main()
