"""Finding 1 aggregate: data/findings/f1_<field>.json -> data/findings/f1.json.

fraction_removed = 1 - sum(corrected changes) / sum(naive changes), summed over the test fields; the headline
naive baseline is (i) all frames in time order, (ii) same-detector consecutive frames is quoted alongside
(docs/INTERFACES.md, "Change model — revision 2"). err: bootstrap over fields (fields resampled with replacement).
Usage: .venv/bin/python scripts/aggregate_f1.py barnard epsindib wise0855 rwaur rdra ecliptic_control
"""
import json, sys

import numpy as np

fields = sys.argv[1:]
recs = [json.load(open(f"data/findings/f1_{f}.json")) for f in fields]
cal = json.load(open("data/findings/f1_calibration.json"))
na = np.array([r["n_naive_all"] for r in recs])
nd = np.array([r["n_naive_det"] for r in recs])
nc = np.array([r["n_corrected"] for r in recs])
rng = np.random.default_rng(0)
idx = rng.integers(0, len(recs), (20000, len(recs)))


def frac(n):
    boot = 1 - nc[idx].sum(1) / n[idx].sum(1)
    lo, hi = np.percentile(boot, [16, 84])
    return float(1 - nc.sum() / n.sum()), float((hi - lo) / 2), [float(lo), float(hi)]


fa, ea, ca = frac(na)
fd, ed, cd = frac(nd)
out = dict(
    fraction_removed=fa, err=ea, ci68=ca, n_naive=int(na.sum()), n_corrected=int(nc.sum()),
    baseline="naive (i): consecutive frames in time order, any detector (what a plain blink shows)",
    same_detector=dict(fraction_removed=fd, err=ed, ci68=cd, n_naive=int(nd.sum()),
                       baseline="naive (ii): consecutive frames of the same detector"),
    corrected_matched_known=int(sum(r["corrected_matched_known"] for r in recs)),
    corrected_unexplained=int(sum(r["corrected_unexplained"] for r in recs)),
    err_method="half-width of the 68% bootstrap interval over fields (20000 resamples)",
    noise_floor_f=cal["f"], calibration="data/findings/f1_calibration.json",
    per_field=[dict(field=r["field"], name=r["name"], n_frames=r["n_frames"], n_passes=r["n_passes"],
                    n_naive_all=r["n_naive_all"], n_naive_det=r["n_naive_det"], n_corrected=r["n_corrected"],
                    fraction_removed_all=1 - r["n_corrected"] / r["n_naive_all"] if r["n_naive_all"] else None,
                    fraction_removed_det=1 - r["n_corrected"] / r["n_naive_det"] if r["n_naive_det"] else None)
               for r in recs],
    method="docs/INTERFACES.md#wavelength-aware-change-model", method_revision=2, fields=fields, release="dr1",
)
json.dump(out, open("data/findings/f1.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "per_field"}, indent=1))
for p in out["per_field"]:
    print(p)
