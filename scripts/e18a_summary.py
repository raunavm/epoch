"""Flatten data/findings/e18a.json into data/findings/e18as.json (token-friendly keys) for the site text."""
import json

d = json.load(open("data/findings/e18a.json"))
vc, ir, kc = d["verdict_counts"], d["injection_recall"], d["known_class_recall"]
hi = lambda k: dict(zip(ir[k]["snr_bins"], ir[k]["recall"]))  # recall per host-S/N bin
held = [v for k, v in kc.items() if not k.startswith("(")]
out = dict(source="data/findings/e18a.json", n_artefact=vc.get("artefact veto", 0), n_explained=vc.get("explained / known class", 0),
           n_survivor=vc.get("survivor", 0), n_held_out=sum(v["n"] for v in held), n_held_out_flagged=sum(v["n_flagged"] for v in held),
           tdwarf_medium_100_300=hi("tdwarf/medium")["100-300"] if "tdwarf/medium" in ir else None,
           null_flag_rate=d["calibration_checks"]["null_injection_flag_rate"], null_n=d["calibration_checks"]["null_injection_n"])
json.dump(out, open("data/findings/e18as.json", "w"), indent=1)
print(out)
