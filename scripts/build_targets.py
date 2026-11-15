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


SUBTRACT_URL = "timemachine.html?ra=269.4463&dec=4.7677&fov=2&mode=subtract"  # Barnard's Star, live Subtract view
MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()


def windows(parquet, gap=30.0):
    """Month labels of the visit windows in a per-picture table (split at gaps > 30 days), e.g. ['Aug–Oct 2025', ...]."""
    import pandas as pd
    from datetime import datetime, timedelta
    m = sorted(pd.read_parquet(parquet, columns=["mjd"]).mjd.dropna())
    cuts = [0] + [i + 1 for i in range(len(m) - 1) if m[i + 1] - m[i] > gap] + [len(m)]
    out = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        d0, d1 = (datetime(1858, 11, 17) + timedelta(days=float(x)) for x in (m[a], m[b - 1]))
        out.append(f"{MONTHS[d0.month - 1]}–{MONTHS[d1.month - 1]} {d1.year}" if d0.month != d1.month else f"{MONTHS[d0.month - 1]} {d1.year}")
    return out


def sci(x):
    """5e-04 -> '5 × 10⁻⁴'."""
    m, e = f"{x:.0e}".split("e")
    return f"{m} × 10{'⁻' if int(e) < 0 else ''}{''.join('⁰¹²³⁴⁵⁶⁷⁸⁹'[int(c)] for c in str(abs(int(e))))}"


E11 = (load("data/findings/e11.json") or {}).get("targets", {})


def add_e11(tid, card):
    """Spectrum figure + one summary number from E11 (docs/protocols/E11.md), if present."""
    t = E11.get(tid)
    if not t:
        return card
    figs = card.setdefault("figures", [])
    nums = card.setdefault("numbers", [])
    if tid == "wise0855":
        c = t["jwst"]["comparison"]["PRISM"]
        nums.append(num(f"Spectrum 3.8–5 µm vs JWST: average ratio over the {c['n_bins_detected']} detected colour steps",
                        round(c["mean_ratio_spherex_over_jwst"], 2), round(c["mean_ratio_err"], 2), method="science.html#e11"))
        figs.append(dict(src="data/figs/e11_wise0855.png", href="science.html#e11",
                         alt="SPHEREx spectrum of WISE 0855 from 3.8 to 5 micrometres, following the JWST spectrum",
                         caption=(f"SPHEREx spectrum ({t['n_pictures_used']} pictures; blue) next to JWST (Luhman et al. 2024; orange). "
                                  "The two reddest points are lower than JWST; we do not know why.")))
    elif tid == "epsindib":
        nums.append(num("Absorption bands found in its SPHEREx spectrum (water, methane, carbon monoxide)", len(t["bands_named"]), None,
                        method="science.html#e11"))
        figs.append(dict(src="data/figs/e11_epsindib.png", href="science.html#e11",
                         alt="SPHEREx spectrum of Epsilon Indi B from 0.75 to 5 micrometres with water, methane and carbon monoxide bands marked",
                         caption=(f"SPHEREx spectrum from {t['n_pictures_used']} pictures. Grey strips: absorption bands that passed our "
                                  "pre-set test. Green: published brightness through standard filters.")))
    elif tid == "barnard":
        card["method_note"] = card.get("method_note", "") + (f" A spectrum also failed: {t['n_pictures'] - t['n_pictures_used']} of {t['n_pictures']} pictures "
                                                             "have a saturated pixel next to the star (Science page, spectra).")
    return card


def write(tid, card):
    add_e11(tid, card)
    card["generated_with"] = "scripts in this repository"
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
                      f"({exp['IRAC_4p5']:.2f} mJy) and WISE ({exp['WISE_W2']:.2f} mJy) measured in broader bands. Between visits it shifts the way "
                      "its known motion predicts, and a source standing still fits the positions far worse. A statement about a known object, not a new object."),
        numbers=[num("Signal, Nov–Dec 2025 visits, against 24 empty paths", round(p2["significance"], 1), unit="σ",
                     nulls=p2.get("null_snrs"), nulls_note="same pictures, path shifted 6–9 px", method="science.html#e4"),
                 num("Signal, Apr–May 2026 visits (a neighbouring star restored to the model)", round(p3.get("significance", float("nan")), 1), unit="σ",
                     method="science.html#e4"),
                 num("Brightness at 4.4–5 µm, Nov–Dec 2025 (error includes the short-wavelength control offset)", round(p2["a_hat_mJy"], 3),
                     round((p2["sigma_a_mJy"] ** 2 + ctrl["a_hat_mJy"] ** 2) ** 0.5, 3), unit="mJy"),
                 num("Shift between visit windows: measured", f"{sh['measured_shift_arcsec'][0]:+.1f} ± {sh['err_arcsec'][0]:.1f}″, {sh['measured_shift_arcsec'][1]:+.1f} ± {sh['err_arcsec'][1]:.1f}″", None),
                 num("Shift between visit windows: predicted", f"{sh['predicted_shift_arcsec'][0]:+.1f}″, {sh['predicted_shift_arcsec'][1]:+.1f}″", None, kind="model"),
                 num("Match to prediction (p-value)", round(sh["p_vs_prediction"], 2), None),
                 num("Same test for a source standing still (p-value)", sci(sh["p_vs_stationary"]), None)],
        evidence_links=[dict(text=f"All {e4['counts']['n_frames_primary_4p4_5p0um']} pictures at 4.4–5 µm used for the numbers above", href="science.html#e4"),
                        dict(text="Astrometry: Luhman & Esplin 2016, AJ 152, 78", href="https://arxiv.org/abs/1605.06655")],
        method_note=("Position from proper motion and parallax (Luhman & Esplin 2016); each picture fitted with Gaia stars; pictures at 4.4–5 µm "
                     "combined per visit window; significance against the same path shifted to 24 empty positions. Caveats: a G = 16.3 star sits "
                     "0.7–1.4 pixels away (our code first dropped it; restored — see What failed); at short wavelengths, where the brown dwarf is "
                     f"invisible, the Nov–Dec 2025 control is {ctrl['a_hat_mJy']:+.3f} ± {ctrl['sigma_a_mJy']:.3f} mJy, not zero, likely the neighbour's "
                     "model; the May 2025 window has only 3 usable pictures and is not informative."),
        method_anchor="science.html#e4", slider="e4/reveal.json",
        slider_caption=(f"Nov–Dec 2025 visits only: slide from one picture to all {e4['slider_stack']['n_frames']} combined along its path. "
                        "Each step is measured against the same pictures combined along 24 empty paths.")))

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
        w = windows(f"data/findings/motion_{tid}.parquet")
        write(tid, dict(
            title=name,
            status="picture only",
            plain_result=(f"Subtract the later pictures ({w[-1]}) from the earlier ones ({w[0]}) and this star shows as an orange–blue pair. By eye, its light has moved "
                          "north, as Gaia predicts; we do not turn this picture into a number, and it is made from the experimental colour-aware pictures "
                          "(open the live view below). Measuring its speed from SPHEREx positions did not work: we get "
                          f"{pa['spherex_mas_yr']/1000:+.1f} ± {pa['err_mas_yr']/1000:.1f}″ and {pd_['spherex_mas_yr']/1000:+.1f} ± {pd_['err_mas_yr']/1000:.1f}″ per year, "
                          f"against Gaia's {pa['gaia_mas_yr']/1000:+.2f}″ and {pd_['gaia_mas_yr']/1000:+.2f}″ (likely cause, untested: the star saturates SPHEREx's detector, "
                          "so the fit rests on its outer light; only two time windows are available; What failed #13)."),
            numbers=[num("Motion east–west (SPHEREx fit, failed)", round(pa["spherex_mas_yr"] / 1000, 1), round(pa["err_mas_yr"] / 1000, 1), unit="″/yr"),
                     num("Motion east–west (Gaia DR3)", round(pa["gaia_mas_yr"] / 1000, 3), None, kind="data", unit="″/yr"),
                     num("Motion north–south (SPHEREx fit, failed)", round(pd_["spherex_mas_yr"] / 1000, 1), round(pd_["err_mas_yr"] / 1000, 1), unit="″/yr"),
                     num("Motion north–south (Gaia DR3)", round(pd_["gaia_mas_yr"] / 1000, 3), None, kind="data", unit="″/yr"),
                     num("p-value for agreement with Gaia", f"p = {sci(p_agree)}", None)],
            evidence_links=[dict(text="See the earlier-minus-later picture live", href=SUBTRACT_URL),
                            dict(text="Why the motion fit fails: What failed, item 13", href="failed.html")],
            method_note=("The picture: the tool splits the pictures at the largest gap in time and subtracts the later set from the earlier one, using the "
                         "experimental colour-aware pictures. The motion fit: likely cause of the failure (untested) is saturation — saturated pixels are "
                         "masked, so the position rests on the star's outer light; Gaia's parallax is held fixed."),
            method_anchor="failed.html"))
        continue
    write(tid, dict(
        title=name,
        plain_result=(f"From SPHEREx positions (with Gaia's distance held fixed) we measured how fast it moves across the sky: "
                      f"{pa['spherex_mas_yr']/1000:+.2f} ± {pa['err_mas_yr']/1000:.2f}″ and {pd_['spherex_mas_yr']/1000:+.2f} ± {pd_['err_mas_yr']/1000:.2f}″ per year "
                      f"(east–west, north–south). ESA's Gaia measured {pa['gaia_mas_yr']/1000:+.2f}″ and {pd_['gaia_mas_yr']/1000:+.2f}″."),
        numbers=[num("Motion east–west (SPHEREx)", round(pa["spherex_mas_yr"] / 1000, 2), round(pa["err_mas_yr"] / 1000, 2), unit="″/yr"),
                 num("Motion east–west (Gaia DR3)", round(pa["gaia_mas_yr"] / 1000, 3), None, kind="data", unit="″/yr"),
                 num("Motion north–south (SPHEREx)", round(pd_["spherex_mas_yr"] / 1000, 2), round(pd_["err_mas_yr"] / 1000, 2), unit="″/yr"),
                 num("Motion north–south (Gaia DR3)", round(pd_["gaia_mas_yr"] / 1000, 3), None, kind="data", unit="″/yr"),
                 num("Pictures used", m["n_frames_fit"], None, kind="data")],
        evidence_links=[dict(text="Per-picture positions (data/findings/motion_%s.json)" % tid, href="science.html#motion"),
                        dict(text="Gaia DR3 (Vallenari et al. 2023)", href="https://www.cosmos.esa.int/web/gaia/dr3")],
        method_note=("Each picture is fitted with the star's position free, saturated pixels masked and Gaia neighbours modelled; "
                     "the motion is a straight-line fit to SPHEREx positions with Gaia's parallax held fixed. This fit was not written down in advance. "
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
                      + "The counts below are for the whole patch of sky, not for the star itself."
                      + (f" On this patch the experimental colour-aware view flags more changes than the plain blink ({f['n_corrected']} against "
                         f"{f['n_naive_all']}); we have not established why, so these counts are not a measurement of the star."
                         if f["n_corrected"] > f["n_naive_all"] else "")),
        numbers=[num("Pictures", f["n_frames"], None, kind="data"),
                 num("Flagged changes, plain blink", f["n_naive_all"], None),
                 num("Flagged changes, one detector at a time", f["n_naive_det"], None),
                 num("Flagged changes, colour-aware view (experimental)", f["n_corrected"], None)],
        evidence_links=[dict(text="Open the prepared field", href=f"timemachine.html?field={tid}")],
        method_note="Counts use the same rule as the 20-patch test (groups of ≥ 2 pixels beyond 5σ).",
        method_anchor="science.html#which-change-is-real"))
