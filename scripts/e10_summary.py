"""Flatten data/findings/e10.json amplitude bins into data/findings/e10s.json (token-friendly keys) for the site text."""
import json

r = json.load(open("data/findings/e10.json"))["recall"]["by_amplitude"]
key = {"0.3-0.5": "amp03_05", "0.5-1.0": "amp05_10", "1.0-99.0": "amp_gt1"}
out = dict(source="data/findings/e10.json", **{key[b]: r[b]["testable"] for b in r})
json.dump(out, open("data/findings/e10s.json", "w"), indent=1)
print(out)
