"""Flatten data/findings/e16.json into data/findings/e16s.json (token-friendly, milliarcseconds) for the site text."""
import json

d = json.load(open("data/findings/e16.json"))
v = d["validation_on_B"]
out = dict(source="data/findings/e16.json", rms_before_mas=v["rms_before_arcsec"] * 1000, rms_after_mas=v["rms_after_arcsec"] * 1000,
           reduction=v["reduction"], threshold=v["threshold"], n_stars_B=v["n_stars"], n_windows_B=v["n_star_windows"],
           w2w_before_mas=v["window_to_window_before_arcsec"] * 1000, w2w_after_mas=v["window_to_window_after_arcsec"] * 1000,
           verdict=d["calibration_verdict"])
json.dump(out, open("data/findings/e16s.json", "w"), indent=1)
print(out)
