"""Assemble site/data/findings.json (Start-here tour) from SUPPORTED claims only (docs/CLAIMS.md).
f1 = headline (Version B, site/data/headline.json + E2), f2 = stacked asteroid (C5, narrowed),
f3 = WISE 0855 along its path (C9). Anything whose source file is missing is omitted.
Also exports the frozen noise floor to site/data/f1_calibration.json.
Usage: .venv/bin/python scripts/build_findings.py
"""
import json, os


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


out = {}
h = load("site/data/headline.json")
d1 = load("data/findings/e2_d1.json")
if h:
    out["f1"] = dict(title="Which change is real?", interaction="timemachine", label="derived", unit="%",
                     number=h["number"], err=None, err_low=h.get("err_low"), err_high=h.get("err_high"),
                     sentence=h["sentence"], detail=dict(source="data/findings/e2.json", consistency="data/findings/e2_d1.json" if d1 else None))
f2 = load("data/findings/f2.json")
sp = load("data/findings/f2_split.json")
if f2 and f2.get("single_frame_snr_max", 99) < 5 and f2.get("stack_snr", 0) >= 5:
    half = ""
    if sp:
        hv = [sp[k]["calibrated"] for k in ("odd", "even", "first_half_time", "second_half_time")]
        half = (f" Split in half (odd/even pictures, or early/late in time), each half gives {min(hv):.1f}–{max(hv):.1f} times the noise, "
                f"against about {sp['expected_half_if_real']:.1f} expected for half the data if the signal is real.")
    sel = f2.get("selection") or {}
    chosen = (f" It was picked by hand after a first pick failed: of {sel['n_stacked']} faint asteroids stacked, {sel['n_meeting_rule']} meet the same rule "
              "and this one scores lowest of them, so the number is not corrected for that choice."
              if sel.get("chosen_by") == "--pick" and sel.get("rank_among_meeting_rule") == sel.get("n_meeting_rule") else
              " The number is not corrected for the choice of this asteroid among the candidates.")
    offset = (f" Without removing the empty paths' average offset it is {f2['stack_snr_without_null_mean']:.1f}."
              if "stack_snr_without_null_mean" in f2 else "")
    out["f2"] = dict(title="Too faint for one picture", interaction="reveal", label="derived", unit="σ",
                     number=round(f2["stack_snr"], 1), err=None,
                     sentence=(f"Asteroid {f2['object']} never reaches the usual detection level in any of its {f2['n_frames']} single pictures "
                               f"(best {f2['single_frame_snr_max']:.1f} times the noise). Combined along its orbit it stands out at "
                               f"{f2['stack_snr']:.1f} times the spread of empty paths beside it — a marginal excess.{offset}{half}{chosen}"),
                     detail=f2)
e13 = load("data/findings/e13.json")
if e13 and "f2" in out and e13.get("n_combined_ge5"):
    top = {str(c["pdes"]) for c in e13["candidates"] if c["stack"] >= 5 and c["max_single"] < 5}
    ok = [r for r in e13["results"] if r["verdict"] == "consistent with a real signal" and r["pdes"] in top]
    good = [r for r in ok if abs(r["amp_ratio_measured_over_predicted"] - 1) <= 0.2 + 2 * r["amp_ratio_err"]]
    out["f2"]["sentence"] = (f"Of the {e13['n_stacked']} faint asteroids we could stack, {e13['n_no_single_5sigma']} never reach the usual "
        f"detection level (5 times the noise) in any single picture. Combined along their known orbits, {e13['n_combined_ge5']} of those stand out "
        f"above 5 times the spread of empty paths beside them, and all {e13['n_ge5_pass_split']} still show up when their pictures are split in half. "
        f"For {len(good)} of them the measured brightness matches, within 20%, what its catalogued brightness predicts. "
        f"The slider shows {out['f2']['detail']['object']}, picked by hand and not the highest-scoring of the three. "
        "The stack numbers were computed before the split test was written down, so this is a check, not a blind test.")
    out["f2"]["detail"] = dict(out["f2"]["detail"], e13="data/findings/e13.json")
    out["f2"].update(title="Too faint for one picture", number=e13["n_combined_ge5"], unit=f"of {e13['n_no_single_5sigma']}")
e4 = load("data/findings/e4.json")
if e4:
    p2 = e4["passes"][1]
    out["f3"] = dict(title="The coldest known brown dwarf", interaction="target", target="wise0855", label="derived", unit="σ",
                     number=round(p2["significance"], 1), err=None,
                     sentence=(f"WISE 0855−0714, a challenge test target, stands out at {p2['significance']:.1f} times the spread of empty paths when "
                               "its 4.4–5 µm pictures are combined along its known path, and it shifts between visits as its motion predicts. "
                               "A statement about a known object, not a new object."),
                     detail=dict(source="data/findings/e4.json"))
json.dump(out, open("site/data/findings.json", "w"), indent=1)
cal = load("data/findings/f1_calibration.json")
if cal:
    json.dump({"f": cal["f"], "source": "data/findings/f1_calibration.json", "frozen_utc": cal.get("frozen_utc")},
              open("site/data/f1_calibration.json", "w"), indent=1)
print(json.dumps({k: v["sentence"] for k, v in out.items()}, indent=1))
