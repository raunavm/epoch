"""Flat file of derived public numbers (data/findings/deriveds.json) for [[deriveds.key]] tokens.

wise0855_flux_err_mJy: E4's formal amplitude error scaled by the spread of the 200 empty paths (E4b), so the flux
error matches the 12.1 sigma significance; _with_control adds E4's short-wavelength control offset in quadrature (as the
target card does). rdra_n_pictures_D6: pictures in the 4.4-5.0 um fit (E28; the 512 total includes 51 at 3.8-4.4 um).
epsindib_null_*: E12 comparison-star parallaxes, root-mean-square about zero (the quoted error) and standard deviation.
"""
import json
import numpy as np

F = lambda n: json.load(open(f"data/findings/{n}.json"))
e4, e4b, e28, e12 = F("e4"), F("e4bs"), F("e28"), F("e12")
p = e4["passes"][1]
ctrl = e4["control_short_wavelength"][1]["neighbour_restored"]["a_hat_mJy"]
err = p["sigma_a_mJy"] * e4b["pass2_null200_std"]
out = dict(wise0855_flux_mJy=p["a_hat_mJy"], wise0855_flux_err_mJy=err, wise0855_flux_err_with_control_mJy=float(np.hypot(err, ctrl)),
           rdra_n_pictures_D6=e28["rdra"]["n_pictures_by_detector"]["6"])
s = e12["null"]["summary"]
for k in ("rms_about_zero_mas", "std_mas", "mean_mas", "n_used"):
    if k in s:
        out[f"epsindib_null_{k}"] = s[k]
out["e23_n_vizier_catalogues"] = len(json.load(open("data/phase2/e23/lit_catalogues.json")))
json.dump(out, open("data/findings/deriveds.json", "w"), indent=1)
print(out)
