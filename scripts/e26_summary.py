"""Flat token file for E26 (data/findings/e26.json -> e26s.json): sinusoid 50%-recall floors before/after the learned
correction, false alarms and common-mode counts. Keys contain no dots, hyphens or spaces (scripts/fill_tokens.py)."""
import json

d = json.load(open("data/findings/e26.json"))
out = {k.replace("-", "_"): v for k, v in d.items() if not isinstance(v, (dict, list)) and k.startswith(("floor_sin_", "fa_", "recall_", "common_mode_"))}
after = {b: d[f"floor_sin_after_{b}"] for b in ("12-13", "13-14", "14-15", "15-16")}
assert len(set(after.values())) == 1, after          # one floor for 12-16 lets the text say "12-16"
out.update(floor_after_12_16=after["12-13"], floor_after_16_17=d["floor_sin_after_16-17"],
           floor_after_17_19=max(d["floor_sin_after_17-18"], d["floor_sin_after_18-19"]), floor_after_19_20=d["floor_sin_after_19-20"],
           n_components=d["n_components"], verdict=d["verdict"].split(":")[0])
out.update({f"{k}_pct": round(100 * out[k]) for k in ("floor_after_12_16", "floor_after_16_17", "floor_after_17_19", "floor_after_19_20")})
json.dump(out, open("data/findings/e26s.json", "w"), indent=1)
print({k: out[k] for k in ("floor_after_12_16", "floor_after_16_17", "floor_after_17_19", "floor_after_19_20", "fa_after_rate", "verdict")})
