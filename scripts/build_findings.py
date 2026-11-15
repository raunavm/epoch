"""Assemble site/data/findings.json from data/findings/f*.json. A finding appears only if its measured
file exists and passes its own check (owner rule: if a finding fails its check, remove its feature and sentence).
Usage: .venv/bin/python scripts/build_findings.py
"""
import json, os


def load(name):
    p = f"data/findings/{name}.json"
    return json.load(open(p)) if os.path.exists(p) else None


out, log = {}, []
f1 = load("f1")
if f1 and f1.get("fraction_removed") is not None and f1["fraction_removed"] > 0.5:
    pct = round(100 * f1["fraction_removed"])
    err = round(100 * f1.get("err", 0))
    out["f1"] = dict(title="Which change is real?", interaction="timemachine", label="derived", unit="%",
                     number=pct, err=err,
                     sentence=f"In a patch of sky chosen in advance as a fair test, {pct}% (±{err}%) of the changes a plain blink shows are only colour: "
                              f"they disappear once each picture's wavelength is accounted for, while moving objects remain.",
                     detail=f1)
else:
    log.append(f"f1 omitted: {f1 and f1.get('fraction_removed')}")
f2 = load("f2")
if f2 and f2.get("single_frame_snr_max", 99) < 5 and f2.get("stack_snr", 0) >= 5 and abs(f2.get("null_snr_mean", 9)) < 2:
    out["f2"] = dict(title="Invisible in every frame, visible in all of them", interaction="reveal", label="data",
                     unit="mag", number=f2["depth_gain_mag"], err=None,
                     sentence=f"Asteroid {f2['object']} is too faint to detect in any of its {f2["n_frames"]} single pictures "
                              f"(best single-picture signal {f2['single_frame_snr_max']}σ), but combining them along its orbit "
                              f"detects it at {f2['stack_snr']:.0f}σ — about {f2['depth_gain_mag']} magnitudes "
                              f"({10 ** (0.4 * f2['depth_gain_mag']):.0f} times) fainter than one picture can reach.",
                     detail=f2)
else:
    log.append(f"f2 omitted: {f2}")
f3 = load("f3")
if f3 and f3.get("n", 0) >= 10:
    pct = round(100 * f3["scatter_frac_robust"])
    out["f3"] = dict(title="How big, and what is it made of?", interaction="object", label="derived", unit="%",
                     number=pct, err=None,
                     sentence=f"For {f3['n']} asteroids, sizes measured from SPHEREx's heat glow agree with independent "
                              f"infrared measurements (NEOWISE, AKARI, IRAS) with a typical difference of {pct}% "
                              f"(median ratio {f3['median_ratio']}); colours agree with ESA Gaia's spectra "
                              f"(median χ²/dof {f3.get('gaia_chi2dof_median')} for {f3.get('gaia_n')} asteroids).",
                     detail=f3)
else:
    log.append(f"f3 omitted: {f3}")
f4 = load("f4")
if f4 and f4.get("V50_stack") and abs(f4.get("null_snr_mean", 9)) < 2:
    out["f4"] = dict(title="Could we have seen it?", interaction="explorer", label="derived", unit="mag",
                     number=f4["V50_stack"], err=None,
                     sentence=f"Planted test objects in real SPHEREx pictures are recovered half the time down to visual "
                              f"magnitude {f4['V50_stack']:.1f} when {f4['n_frames']} pictures are combined.",
                     detail=f4)
else:
    log.append(f"f4 omitted: {f4}")
json.dump(out, open("site/data/findings.json", "w"), indent=1)
cal = load("f1_calibration")
if cal:
    json.dump({"f": cal["f"], "source": "data/findings/f1_calibration.json", "frozen_utc": cal.get("frozen_utc")},
              open("site/data/f1_calibration.json", "w"), indent=1)
print(json.dumps({k: v["sentence"] for k, v in out.items()}, indent=1))
print("\n".join(log))
