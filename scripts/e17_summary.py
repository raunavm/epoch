"""Derived E17 depth numbers -> data/findings/e17s.json: the fitted completeness curve is cmax / (1 + exp((V - V50) / w)),
so V50 is where completeness is cmax/2; this file adds where it reaches an absolute 50%, and the found fraction."""
import json, math

d = json.load(open("data/findings/e17.json"))
f, inj = d["injections"]["fit_all"], d["injections"]
cmax, v50, w = f["cmax"], f["V50"], f["width"]
v_abs50 = v50 + w * math.log(cmax / 0.5 - 1) if cmax > 0.5 else None
out = dict(source="data/findings/e17.json", cmax=cmax, V_half_of_max=v50, V_abs50=v_abs50, n_found=inj["recovered"], n_planted=inj["n"],
           found_frac=inj["recovered"] / inj["n"])
json.dump(out, open("data/findings/e17s.json", "w"), indent=1)
print(out)
