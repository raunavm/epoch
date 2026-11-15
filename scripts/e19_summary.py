"""Flatten data/findings/e19.json into data/findings/e19s.json (token-friendly keys, no spaces) for the site text."""
import json

d = json.load(open("data/findings/e19.json"))
key = {"VY CMa": "vycma", "eps Oct": "epsoct", "GY UMa": "gyuma"}
out = dict(source="data/findings/e19.json", verdict_frozen=d["verdict_frozen"], verdict_post_hoc=d["verdict_post_hoc_pattern"],
           threshold_T=d["frozen"]["threshold"]["T"], n_comparison=d["frozen"]["threshold"]["n_c_else"])
for name, k in key.items():
    p, f = d["post_hoc_pattern"]["named"][name], d["frozen"]["named"][name]
    out[k] = dict(c_spot=p["c_spot"], c_else_median=p["c_else_median"], c_else_max=p["c_else_max"], n_elsewhere=p["n_elsewhere"],
                  S_spot_max=max(s["value"] for s in f["spot"]))
ph = d["post_hoc_pattern"]
out.update(blank_n=ph["blank_spot_control"]["n"], blank_median=ph["blank_spot_control"]["median"], blank_max=ph["blank_spot_control"]["max"],
           pairs_spot_n=ph["spot_spot"]["n"], pairs_spot_median=ph["spot_spot"]["median"], pairs_else_n=ph["spot_vs_c_else"]["n"],
           pairs_else_median=ph["spot_vs_c_else"]["median"])
json.dump(out, open("data/findings/e19s.json", "w"), indent=1)
print(json.dumps(out, indent=1))
