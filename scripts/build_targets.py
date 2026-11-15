"""Result cards for the five challenge targets -> site/data/targets/<id>.json, from findings files only.
Usage: .venv/bin/python scripts/build_targets.py
"""
import json, os

OUT = "site/data/targets"
os.makedirs(OUT, exist_ok=True)
S3 = "https://nasa-irsa-spherex.s3.amazonaws.com/"


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def num(label, value, err=None, kind="derived", unit="", **kw):
    d = dict(label=label, value=value, err=err, kind=kind, unit=unit)
    d.update(kw)
    return d


def write(tid, card):
    json.dump(card, open(f"{OUT}/{tid}.json", "w"), indent=1)
    print(tid, "->", card["plain_result"][:120])


# WISE 0855 — E4 (wording narrowed per docs/REFEREE.md items 4–5)
e4 = load("data/findings/e4.json")
if e4:
    p = e4["passes"]
    p2 = p[1]
    p3 = p[2].get("neighbour_restored") or p[2]
    sh = e4["centroid_shift_vs_prediction"][0]
    ctrl = e4["control_short_wavelength"][1]["neighbour_restored"]
    exp = e4["expected_flux_mJy"]
    write("wise0855", dict(
        title="WISE 0855−0714 — the coldest known brown dwarf",
        plain_result=(f"At 4.4–5 µm, combining SPHEREx pictures along its known path gives a clear signal where it should be: "
                      f"{p2['a_hat_mJy']:.2f} ± {(p2['sigma_a_mJy']**2 + ctrl['a_hat_mJy']**2)**0.5:.2f} mJy in the Nov–Dec 2025 visits, between the brightness Spitzer "
                      f"({exp['IRAC_4p5']:.2f} mJy) and WISE ({exp['WISE_W2']:.2f} mJy) measured in similar bands. Between visits it shifts the way "
                      "its known motion predicts, and a source standing still fits the positions far worse. A statement about a known object, not a discovery."),
        numbers=[num("Signal, Nov–Dec 2025 visits, against 24 empty paths", round(p2["significance"], 1), unit="σ",
                     nulls=p2.get("null_snrs"), nulls_note="same pictures, path shifted 6–9 px", method="science.html#e4"),
                 num("Signal, Apr–May 2026 visits (a neighbouring star restored to the model)", round(p3.get("significance", float("nan")), 1), unit="σ",
                     method="science.html#e4"),
                 num("Brightness at 4.4–5 µm, Nov–Dec 2025 (error includes the short-wavelength control offset)", round(p2["a_hat_mJy"], 3),
                     round((p2["sigma_a_mJy"] ** 2 + ctrl["a_hat_mJy"] ** 2) ** 0.5, 3), unit="mJy"),
                 num("Shift between visit windows: measured", f"{sh['measured_shift_arcsec'][0]:+.1f} ± {sh['err_arcsec'][0]:.1f}″, {sh['measured_shift_arcsec'][1]:+.1f} ± {sh['err_arcsec'][1]:.1f}″", None),
                 num("Shift between visit windows: predicted", f"{sh['predicted_shift_arcsec'][0]:+.1f}″, {sh['predicted_shift_arcsec'][1]:+.1f}″", None, kind="model"),
                 num("Match to prediction (p-value)", round(sh["p_vs_prediction"], 2), None),
                 num("Same test for a source standing still (p-value)", f"{sh['p_vs_stationary']:.0e}", None)],
        evidence_links=[dict(text="All 46 pictures used (4.4–5 µm)", href="science.html#e4"),
                        dict(text="Astrometry: Luhman & Esplin 2016, AJ 152, 78", href="https://arxiv.org/abs/1605.06655")],
        method_note=("Position from proper motion and parallax (Luhman & Esplin 2016); each picture fitted with Gaia stars; pictures at 4.4–5 µm "
                     "combined per visit window; significance against the same path shifted to 24 empty positions. Caveats: a G = 16.3 star sits "
                     "0.7–1.4 pixels away (our code first dropped it; restored — see What failed); at short wavelengths, where the brown dwarf is "
                     f"invisible, the Nov–Dec 2025 control is {ctrl['a_hat_mJy']:+.3f} ± {ctrl['sigma_a_mJy']:.3f} mJy, not zero, likely the neighbour's "
                     "model; the May 2025 window has only 3 usable pictures and is not informative."),
        method_anchor="science.html#e4", slider="e4/reveal.json"))

# Motion measurements (Epsilon Indi B, Barnard's Star)
for tid, name in (("epsindib", "Epsilon Indi B — a pair of nearby brown dwarfs"), ("barnard", "Barnard's Star — the fastest-moving star in our sky")):
    m = load(f"data/findings/motion_{tid}.json")
    if not m:
        continue
    pa, pd_ = m["proper_motion"]["pmra"], m["proper_motion"]["pmdec"]
    chi2 = sum(((x["spherex_mas_yr"] - x["gaia_mas_yr"]) / x["err_mas_yr"]) ** 2 for x in (pa, pd_))
    from math import exp
    p_agree = exp(-chi2 / 2)  # chi-square survival function for 2 dof
    if p_agree < 0.01:
        write(tid, dict(
            title=name,
            plain_result=("Our position measurement fails for this star (likely cause, untested: it saturates SPHEREx's detector, so the fit rests on its outer light): "
                          f"from SPHEREx we get {pa['spherex_mas_yr']/1000:+.1f} ± {pa['err_mas_yr']/1000:.1f}″ and {pd_['spherex_mas_yr']/1000:+.1f} ± {pd_['err_mas_yr']/1000:.1f}″ per year, "
                          f"which disagrees with Gaia's {pa['gaia_mas_yr']/1000:+.2f}″ and {pd_['gaia_mas_yr']/1000:+.2f}″. The picture of survey 1 minus survey 2 does show the star's light "
                          "moving north, as Gaia predicts, but we do not turn that picture into a number."),
            numbers=[num("Motion east–west (SPHEREx, fails)", round(pa["spherex_mas_yr"] / 1000, 2), round(pa["err_mas_yr"] / 1000, 2), unit="″/yr"),
                     num("Motion east–west (Gaia DR3)", round(pa["gaia_mas_yr"] / 1000, 3), None, kind="data", unit="″/yr"),
                     num("Motion north–south (SPHEREx, fails)", round(pd_["spherex_mas_yr"] / 1000, 2), round(pd_["err_mas_yr"] / 1000, 2), unit="″/yr"),
                     num("Motion north–south (Gaia DR3)", round(pd_["gaia_mas_yr"] / 1000, 3), None, kind="data", unit="″/yr"),
                     num("Chance the two agree", f"{p_agree:.0e}", None)],
            evidence_links=[dict(text="Why it fails: What failed, item 13", href="failed.html"),
                            dict(text="See the survey difference live", href="timemachine.html?ra=269.4463&dec=4.7677&fov=2&mode=subtract")],
            method_note="Likely cause (untested): saturation — saturated pixels are masked, so the position rests on the star's outer light.",
            method_anchor="failed.html"))
        continue
    write(tid, dict(
        title=name,
        plain_result=(f"From SPHEREx positions (with Gaia's distance held fixed) we measured how fast it moves across the sky: "
                      f"{pa['spherex_mas_yr']/1000:+.2f} ± {pa['err_mas_yr']/1000:.2f}″ and {pd_['spherex_mas_yr']/1000:+.2f} ± {pd_['err_mas_yr']/1000:.2f}″ per year "
                      f"(east–west, north–south). ESA's Gaia measured {pa['gaia_mas_yr']/1000:+.2f}″ and {pd_['gaia_mas_yr']/1000:+.2f}″."),
        numbers=[num("Motion east–west (SPHEREx)", round(pa["spherex_mas_yr"] / 1000, 3), round(pa["err_mas_yr"] / 1000, 3), unit="″/yr"),
                 num("Motion east–west (Gaia DR3)", round(pa["gaia_mas_yr"] / 1000, 3), None, kind="data", unit="″/yr"),
                 num("Motion north–south (SPHEREx)", round(pd_["spherex_mas_yr"] / 1000, 3), round(pd_["err_mas_yr"] / 1000, 3), unit="″/yr"),
                 num("Motion north–south (Gaia DR3)", round(pd_["gaia_mas_yr"] / 1000, 3), None, kind="data", unit="″/yr"),
                 num("Pictures used", m["n_frames_fit"], None, kind="data")],
        evidence_links=[dict(text="Per-picture positions (data/findings/motion_%s.json)" % tid, href="science.html#motion"),
                        dict(text="Gaia DR3 (Vallenari et al. 2023)", href="https://www.cosmos.esa.int/web/gaia/dr3")],
        method_note=("Each picture is fitted with the star's position free, saturated pixels masked and Gaia neighbours modelled; "
                     "the motion is a straight-line fit to SPHEREx positions with Gaia's parallax held fixed. "
                     + ("Positions sit a steady ~0.5″ south of Gaia's prediction in every survey; the pair is unresolved, so a shifted centre of light is possible — not explained here." if tid == "epsindib" else "")),
        method_anchor="science.html#motion"))

# Variables: neutral counts only (referee items on R Dra / RW Aur cards)
for tid, name in (("rwaur", "RW Aurigae — a young variable star"), ("rdra", "R Draconis — a pulsating giant star")):
    f = load(f"data/findings/f1_{tid}.json")
    if not f:
        continue
    write(tid, dict(
        title=name,
        status="not measured yet",
        plain_result=(f"Not measured yet. {f['n_frames']} SPHEREx pictures of this variable star are available in the Time Machine. "
                      + ("We tried a per-star test (one brightness per picture, colour removed, compared with ordinary stars), but this star "
                         "saturates SPHEREx's detector in most pictures: our fit of its outer light is poor, so a change cannot be told apart from "
                         "a bad fit (What failed #16). " if tid == "rwaur" else
                         "We tried a per-star test, but this star saturates SPHEREx's detector in almost every picture (3 usable of 250), so it "
                         "cannot be measured with our method (What failed #16). ")
                      + "The counts below are for the whole patch of sky, not for the star itself."),
        numbers=[num("Pictures", f["n_frames"], None, kind="data"),
                 num("Flagged changes, plain blink", f["n_naive_all"], None),
                 num("Flagged changes, one detector at a time", f["n_naive_det"], None),
                 num("Flagged changes, colour-aware view (experimental)", f["n_corrected"], None)],
        evidence_links=[dict(text="Open the prepared field", href=f"timemachine.html?field={tid}")],
        method_note="Counts use the same rule as the 20-patch test (groups of ≥ 2 pixels beyond 5σ).",
        method_anchor="science.html#which-change-is-real"))
