"""Recalibrate the WISE 0855 slider data (site/data/e4/reveal.json) with E4b's 200 empty paths (docs/protocols/E4b.md).

Keeps every nominal S/N and image; replaces the 24-path calibration (snr, null mean/spread, null list, notes) with E4b's
200-path values for each stack step and single picture, using the same site rule (S/N - mean) / max(spread, 1).
Usage: .venv/bin/python scripts/e4b_reveal.py
"""
import json

P = "site/data/e4/reveal.json"
r = json.load(open(P))
b = json.load(open("data/findings/e4b.json"))
p2 = next(x for x in b["results"] if x["n_pictures"] == r["n_frames"])
steps = {s["n"]: s for s in b["pass2_steps"]}
singles = {s["obs_id"] + f"_{s['detector']}": s for s in b["pass2_singles"]}

for st in r["stack_pngs"]:
    s = steps[st["n"]]
    assert abs(s["snr_nominal"] - st["snr_nominal"]) < 0.01, (st["n"], s["snr_nominal"], st["snr_nominal"])
    st.update(snr=round(s["sig_200_site_rule"], 2), null_mean=round(s["null200_mean"], 3), null_std=round(s["null200_std"], 3))
for f in r["frames"]:
    s = singles[f["obs_id"] + f"_{f['detector']}"]
    assert abs(s["snr_nominal"] - f["snr_nominal"]) < 0.01
    f["snr"] = round(s["sig_200_site_rule"], 2)
last = steps[max(steps)]
m, sd = last["null200_mean"], last["null200_std"]
r["stack_snr"] = round(last["sig_200_site_rule"], 2)
r["null_snrs_nominal"] = [round(x, 3) for x in p2["null_snrs"]]
r["null_snrs"] = [round((x - m) / max(sd, 1), 2) for x in p2["null_snrs"]]
nc = r["noise_calibration"]
nc.update(stack_null_mean=round(m, 3), stack_null_std=round(sd, 3), single_std=round(b["pass2_single_std200"], 3), n_null_tracks=len(p2["null_snrs"]),
          per_step=f"each step: the same n pictures combined along {len(p2['null_snrs'])} empty paths (E4b); (S/N - null mean)/max(null spread, 1)",
          singles=f"single pictures: S/N divided by the spread of single-picture S/N on the {len(p2['null_snrs'])} empty paths (floored at 1); no mean correction")
n = len(p2["null_snrs"])
r["hedge"] = f"Nov–Dec 2025 visits only; each step against the same pictures combined along {n} empty paths"
r["hedge_es"] = f"Solo las visitas de nov–dic 2025; cada paso, frente a las mismas imágenes combinadas a lo largo de {n} caminos vacíos"
r["rerun"] += "; recalibrated with E4b: .venv/bin/python scripts/run_e4b.py && .venv/bin/python scripts/run_e4b.py steps && .venv/bin/python scripts/e4b_reveal.py"
json.dump(r, open(P, "w"), indent=1, ensure_ascii=False)
print("stack_snr", r["stack_snr"], "steps", [s["snr"] for s in r["stack_pngs"]], "nulls", n)
