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
    """Spectrum figure + one summary number from E11 (docs/protocols/E11.md), if present (once per card)."""
    t = E11.get(tid)
    if not t or card.get("_e11"):
        return card
    card["_e11"] = True
    figs = card.setdefault("figures", [])
    nums = card.setdefault("numbers", [])
    if tid == "wise0855":
        c = t["jwst"]["comparison"]["PRISM"]
        nums.append(num(f"Spectrum 3.8–5 µm vs JWST: average ratio over the {c['n_bins_detected']} detected colour steps",
                        round(c["mean_ratio_spherex_over_jwst"], 2), round(c["mean_ratio_err"], 2), method="science.html#e11"))
        figs.append(dict(src="data/figs/e11_wise0855.png", href="science.html#e11",
                         alt="SPHEREx spectrum of WISE 0855 from 3.8 to 5 micrometres, following the JWST spectrum",
                         caption=(f"SPHEREx spectrum ({sum(b['n'] for b in t['bins'])} pictures at 3.8–5 µm; blue) next to JWST (Luhman et al. 2024; orange). "
                                  "The two reddest points are lower than JWST; we do not know why.")))
    elif tid == "epsindib":
        nums.append(num("Absorption bands found in its SPHEREx spectrum (water, methane, carbon monoxide)", len(t["bands_named"]), None,
                        method="science.html#e11"))
        figs.append(dict(src="data/figs/e11_epsindib.png", href="science.html#e11",
                         alt="SPHEREx spectrum of Epsilon Indi B from 0.75 to 5 micrometres with water, methane and carbon monoxide bands marked",
                         caption=(f"SPHEREx spectrum from {t['n_pictures_used']} pictures. Grey strips: absorption bands that passed our "
                                  "predefined test. Green: published brightness through standard filters.")))
    elif tid == "barnard":
        card["method_note"] = card.get("method_note", "") + (f" A spectrum also failed: {t['n_pictures'] - t['n_pictures_used']} of {t['n_pictures']} pictures "
                                                             "have a saturated pixel next to the star (Science page, spectra).")
    return card


def write(tid, card):
    add_e11(tid, card)
    json.dump(card, open(f"{OUT}/{tid}.json", "w"), indent=1)
    print(tid, "->", card["plain_result"][:120])


# WISE 0855 — E4 (wording narrowed per docs/REFEREE.md items 4–5)
e4 = load("data/findings/e4.json")
if e4:
    p = e4["passes"]
    e4b = load("data/findings/e4bs.json")
    p2 = p[1]
    p3 = p[2].get("neighbour_restored") or p[2]
    sh = e4["centroid_shift_vs_prediction"][0]
    ctrl = e4["control_short_wavelength"][1]["neighbour_restored"]
    exp = e4["expected_flux_mJy"]
    write("wise0855", dict(
        title="WISE 0855−0714",
        plain_result=(f"At 4.4–5 µm, combining SPHEREx pictures along its known path gives a clear signal where it should be: "
                      f"{p2['a_hat_mJy']:.2f} ± {(p2['sigma_a_mJy']**2 + ctrl['a_hat_mJy']**2)**0.5:.2f} mJy in the Nov–Dec 2025 visits, consistent with Spitzer (0.48–0.54 mJy) and WISE (0.41–0.50 mJy) in broader bands."
                      " Between visits it shifts the way "
                      "its known motion predicts, and a source standing still fits the positions far worse. This is a measurement of a known object."),
        # Significance against E4b's 200 empty paths (docs/protocols/E4b.md); E4's 24-path figures (16.5σ, 10.6σ) do not stand.
        numbers=[num(f"Detection against {e4b['pass2_null200_n']} empty paths, Nov–Dec 2025 visits", round(e4b["pass2_sig_200"], 1), unit="σ",
                     method="results.html#e4"),
                 num("Brightness at 4.4–5 µm, Nov–Dec 2025 (error includes the short-wavelength control offset)", round(p2["a_hat_mJy"], 3),
                     round((p2["sigma_a_mJy"] ** 2 + ctrl["a_hat_mJy"] ** 2) ** 0.5, 3), unit="mJy"),
                 num("Shift between visit windows: measured", f"{sh['measured_shift_arcsec'][0]:+.1f} ± {sh['err_arcsec'][0]:.1f}″, {sh['measured_shift_arcsec'][1]:+.1f} ± {sh['err_arcsec'][1]:.1f}″", None),
                 num("Shift between visit windows: predicted", f"{sh['predicted_shift_arcsec'][0]:+.1f}″, {sh['predicted_shift_arcsec'][1]:+.1f}″", None, kind="model"),
                 num("Match to prediction (p-value)", round(sh["p_vs_prediction"], 2), None),
                 num("Same test for a source standing still (p-value)", sci(sh["p_vs_stationary"]), None)],
        evidence_links=[dict(text=f"All {e4['counts']['n_frames_primary_4p4_5p0um']} pictures at 4.4–5 µm used for the numbers above", href="science.html#e4"),
                        dict(text="Astrometry: Luhman & Esplin 2016, AJ 152, 78", href="https://arxiv.org/abs/1605.06655")],
        method_note=("Position from proper motion and parallax (Luhman & Esplin 2016); each picture fitted with Gaia stars; pictures at 4.4–5 µm "
                     "combined per visit window; significance against the same pictures combined along 200 empty paths (E4b). Caveats: a G = 16.3 star sits "
                     "0.7–1.4 pixels away (our code first dropped it; restored — see What failed); at short wavelengths, where the brown dwarf is "
                     f"invisible, the Nov–Dec 2025 control is {ctrl['a_hat_mJy']:+.3f} ± {ctrl['sigma_a_mJy']:.3f} mJy, not zero, likely the neighbour's "
                     "model; the May 2025 window has only 3 usable pictures and is not informative."),
        method_anchor="science.html#e4", slider="e4/reveal.json",
        slider_caption=(f"Nov–Dec 2025 visits only: slide from one picture to all {e4['slider_stack']['n_frames']} combined along its path. "
                        "Each step is measured against the same pictures combined along 24 empty paths.")))

# Motion measurements (Epsilon Indi B, Barnard's Star)
for tid, name in (("epsindib", "Epsilon Indi B"), ("barnard", "Barnard's Star")):
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
                     "the motion is a straight-line fit to SPHEREx positions with Gaia's parallax held fixed. This fit was not predefined. "
                     + ("Positions sit a steady ~0.5″ south and ~0.2″ east of Gaia's prediction in every window. Gaia's entry is Ba alone and its motion includes its orbit; from the pair's published orbit (Chen et al. 2022) the centre of Ba+Bb's infrared light should sit 0.50–0.56″ south and 0.13–0.19″ east — a check we made after the measurement (Science page)." if tid == "epsindib" else "")),
        method_anchor="science.html#motion"))

# Variables: neutral counts only (referee items on R Dra / RW Aur cards)
for tid, name in (("rwaur", "RW Aurigae"), ("rdra", "R Draconis")):
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

# Phase 2: E12 (distance to Epsilon Indi B) and E8 (Barnard's Star motion) — docs/PROTOCOLS.md PHASE 2
e12 = load("data/findings/e12.json")
if e12 and os.path.exists(f"{OUT}/epsindib.json"):
    card = json.load(open(f"{OUT}/epsindib.json"))
    lo, hi = e12["distance_interval_ly_68"]
    card["plain_result"] += (f" Fitting its yearly wobble as well, with no Gaia measurement of this object (the archive astrometry is tied to Gaia DR3 reference stars), SPHEREx gives a parallax of "
                             f"{e12['parallax_mas']:.0f} ± {e12['parallax_err_mas']['adopted']:.0f} thousandths of an arcsecond: a distance of about "
                             f"{e12['distance_ly']:.0f} light-years ({lo:.0f}–{hi:.0f}); Gaia gives {3.26156 * e12['gaia_distance_pc']:.1f}.")
    card["numbers"] = [n for n in card["numbers"] if not n["label"].startswith("Pictures used")] + [
        num("Parallax from SPHEREx positions (predefined test, passed)", round(e12["parallax_mas"]), round(e12["parallax_err_mas"]["adopted"]),
            unit="mas", method="science.html#e12"),
        num("Parallax (Gaia DR3: Ba 270.7; system, from Epsilon Indi A, 274.8)", round(e12["gaia_parallax_mas"], 1), None, kind="data", unit="mas"),
        num("Distance (68% interval)", f"{e12['distance_ly']:.1f} light-years ({lo:.1f}–{hi:.1f})", None, method="science.html#e12")]
    card.setdefault("figures", []).insert(0, dict(src="data/figs/e12_parallax.png", href="science.html#e12",
        alt="Epsilon Indi B's path across the sky over a year, with the yearly wobble that gives its distance",
        caption=(f"Left: its path over a year; the straight dashed line is motion alone, the blue curve adds the yearly wobble. Right: the wobble "
                 f"with the steady motion removed. Error bar set by {e12['null']['summary']['n_used']} ordinary stars measured the same way, which "
                 "all show a small false wobble of their own (Science page).")))
    card["method_note"] += (" Distance test (E12): five quantities fitted to the per-picture positions (offset, motion, parallax); the error is the "
                            f"largest of the fit, a bootstrap and the scatter of {e12['null']['summary']['n_used']} ordinary stars fitted the same way; "
                            f"those stars all come out with a parallax below zero (mean {e12['null']['summary']['mean_mas']:.0f} mas), a shared bias we "
                            "did not correct.")
    card.pop("generated_with", None)
    write("epsindib", card)  # add_e11 skipped: card already carries "_e11"

e8 = load("data/findings/e8.json")
if e8:
    g = e8["gaia"]
    c = e8["controls"]["summary"]
    write("barnard", dict(
        title="Barnard's Star",
        status="motion matches Gaia; predefined check failed",
        plain_result=(f"Measured from the centre of its outer glow (its core overloads SPHEREx's detector), Barnard's Star moves "
                      f"{e8['pmdec_arcsec_yr']:+.1f} ± {e8['pmdec_err']:.1f}″ per year north and {e8['pmra_arcsec_yr']:+.1f} ± {e8['pmra_err']:.1f}″ east, "
                      f"against Gaia's {g['pmdec_mas_yr']/1000:+.1f}″ and {g['pmra_mas_yr']/1000:+.1f}″. By the rule we wrote down beforehand the "
                      f"result still counts as failed: a check on {e8['controls']['n_used']} faint stars nearby, measured the same way, showed a small "
                      f"shared drift ({c['mean_d_pmra_arcsec_yr']:+.2f}″ per year east–west), larger than their own error bars allow. "
                      "Our first attempt, from the overloaded core, failed outright (What failed #13)."),
        numbers=[num("Motion north–south (SPHEREx)", round(e8["pmdec_arcsec_yr"], 2), round(e8["pmdec_err"], 2), unit="″/yr", method="science.html#e8"),
                 num("Motion north–south (Gaia DR3)", round(g["pmdec_mas_yr"] / 1000, 2), None, kind="data", unit="″/yr"),
                 num("Motion east–west (SPHEREx)", round(e8["pmra_arcsec_yr"], 2), round(e8["pmra_err"], 2), unit="″/yr", method="science.html#e8"),
                 num("Motion east–west (Gaia DR3)", round(g["pmra_mas_yr"] / 1000, 2), None, kind="data", unit="″/yr"),
                 num("Total motion against zero", round(e8["total_sigma"], 1), None, unit="σ", method="science.html#e8"),
                 num("Predefined check on faint neighbours", "failed (shared drift)", None)],
        figures=[dict(src="data/figs/e8_barnard.png", href="science.html#e8",
                      alt="Barnard's Star's position in two SPHEREx visits about seven months apart, next to Gaia's prediction",
                      caption=(f"Left: SPHEREx positions in two visits ({e8['pictures']['used']} pictures) and Gaia's prediction. Right: how far each "
                               f"visit lands from Gaia, next to {e8['comparison']['n_stars']} comparison stars that do not move."))],
        evidence_links=[dict(text="NASA WISE (2010) next to SPHEREx (2025–26): the star has moved", href="timemachine.html?target=barnard&fov=8&wise=1#live"),
                        dict(text="Earlier minus later SPHEREx pictures, live", href=SUBTRACT_URL),
                        dict(text="Method and the failed check (Science page)", href="science.html#e8")],
        method_note=(f"Method chosen on {e8['comparison']['n_stars']} bright comparison stars before Barnard's Star was measured (docs/protocols/E8.md): "
                     "the centre of symmetry of the unflagged outer light, with a per-detector correction; error per visit from the comparison stars "
                     f"({e8['per_window_error_arcsec']:.2f}″). Only two visit windows, so the fit has no spare freedom. Gaia's parallax held fixed. "
                     "The faint-neighbour drift flips sign with the detector's orientation between visits, which points to a small offset in "
                     "the detector frame (a reading made after the result)."),
        method_anchor="science.html#e8"))

# Phase 2: E9 (RW Aurigae, R Draconis) — halo photometry with comparison-star errors
e9 = load("data/findings/e9.json")
if e9:
    from datetime import datetime, timedelta
    day = lambda mjd: (datetime(1858, 11, 17) + timedelta(days=float(mjd))).strftime("%-d %b %Y")
    w = e9["targets"]["rwaur"]
    col = {c["detector"]: c for c in w["colour"]["post_hoc_inflated"]["contrast"]}
    lam = {c["detector"]: c["lam_um"] for c in w["colour"]["contrast"]}
    zbest = max((d for d in w["a_dw"] if d.get("z") is not None), key=lambda d: abs(d["z"]))
    write("rwaur", dict(
        title="RW Aurigae",
        status="measured; no change meets the predefined criterion",
        plain_result=(f"Its centre overloads SPHEREx's detector, so we measured it from the halo of light around it, a method tested first on "
                      f"{w['comparison']['n_stars']} ordinary stars of similar brightness. Between the two visits (Sep 2025 and Mar 2026) it looks "
                      f"{100*col[1]['c']:.0f} ± {100*col[1]['sigma']:.0f}% brighter at {lam[1]:.1f} µm and {100*col[2]['c']:.0f} ± {100*col[2]['sigma']:.0f}% at {lam[2]:.1f} µm, "
                      f"but {abs(100*col[5]['c']):.0f} ± {100*col[5]['sigma']:.0f}% fainter at {lam[5]:.1f} µm: not a grey change, bigger in the bluest colours, "
                      "the pattern our literature notes predicted, before this measurement, if the fading reported in early 2025 had ended between our "
                      "visits. One band passes 5 times its error in one visit, but the rule we wrote down beforehand needs two, and its own pictures "
                      "scatter 1.1–3 times more than the ordinary stars' (which widens its errors), so no change is confirmed."),
        numbers=[num(f"Change at {lam[1]:.2f} µm, Mar 2026 vs Sep 2025", f"{100*col[1]['c']:+.0f} ± {100*col[1]['sigma']:.0f}%", None, method="science.html#e9"),
                 num(f"Change at {lam[2]:.2f} µm", f"{100*col[2]['c']:+.0f} ± {100*col[2]['sigma']:.0f}%", None, method="science.html#e9"),
                 num(f"Change at {lam[4]:.2f} µm", f"{100*col[4]['c']:+.0f} ± {100*col[4]['sigma']:.0f}%", None, method="science.html#e9"),
                 num(f"Change at {lam[5]:.2f} µm", f"{100*col[5]['c']:+.0f} ± {100*col[5]['sigma']:.0f}%", None, method="science.html#e9"),
                 num("Largest single-band change against the predefined threshold", f"{abs(zbest['z']):.1f} times the error at {lam[zbest['detector']]:.1f} µm (rule: 5×, in 2 bands)", None),
                 num("Without the star's own scatter factor", f"{w['without_error_inflation']['windows_with_2_detectors_over_5sigma']} visits with 2 bands over 5×", None),
                 num("Ordinary stars over the threshold", f"{w['comparison']['n_windows_exceeding']} of {w['comparison']['n_windows']} visits", None)],
        figures=[dict(src="data/figs/e9_rwaur.png", href="science.html#e9",
                      alt="RW Aurigae's brightness change between two SPHEREx visits in five colour bands, next to ordinary stars",
                      caption="Left: change by colour band, with the scatter of ordinary stars in grey and our 5× threshold dashed. Right: the same test on 41 ordinary stars.")],
        evidence_links=[dict(text="Open the prepared field", href="timemachine.html?field=rwaur"), dict(text="Method (Science page)", href="science.html#e9")],
        method_note=("Halo photometry (annulus 4–7 pixels) chosen and its errors set on comparison stars before RW Aurigae was measured "
                     "(docs/protocols/E9.md); 4.4–5 µm not used (more saturated than any comparison star). The colour-by-colour errors use the "
                     "star's own picture scatter, applied after the result (labelled in the results file). RW Aurigae is an unresolved pair 1.5″ apart."),
        method_anchor="science.html#e9"))
    r = e9["targets"]["rdra"]
    dm = r["post_hoc_guard_off"]["date_of_maximum"][0]
    write("rdra", dict(
        title="R Draconis",
        status="measured; no change meets the predefined criterion",
        plain_result=(f"R Draconis is so bright that only the halo around its overloaded centre can be measured, and the predefined test could use only "
                      f"three noisy colour bands: there no change passes 5 times the error ({r['comparison']['n_stars']} ordinary bright stars measured "
                      f"the same way). Shown for information only, outside the predefined range: at {dm['lam_um']:.1f} µm its brightness rises and falls by "
                      f"about {dm['semi_amplitude_mag']:.1f} magnitudes either side of average and peaks around {day(dm["mjd_max"])} "
                      f"(± 3 days, formal). The catalogue predicts a visible-light maximum near {day(dm['catalogue_visual_max_mjd'])}, uncertain by 2–3 weeks, so the "
                      "infrared–visible lag is not measured."),
        numbers=[num("Pictures", r["n_pictures"], None, kind="data"),
                 num("Colour bands usable in the predefined test", len(r["detectors_used"]), None),
                 num(f"Peak at {dm['lam_um']:.1f} µm (for information)", day(dm["mjd_max"]), None),
                 num("Catalogue visible-light maximum (VSX period 246 days)", day(dm["catalogue_visual_max_mjd"]), None, kind="model")],
        figures=[dict(src="data/figs/e9_rdra.png", href="science.html#e9",
                      alt="R Draconis brightness through the year in six SPHEREx colour bands",
                      caption="Brightness through the year in six colour bands; open circles are outside the predefined range and for information only.")],
        evidence_links=[dict(text="Open the prepared field", href="timemachine.html?field=rdra"), dict(text="Method (Science page)", href="science.html#e9")],
        method_note=("Halo photometry (annulus 7–12 pixels) chosen on comparison stars before R Draconis was measured (docs/protocols/E9.md). "
                     "For a star this bright, much of the 'halo' is detector bloom rather than starlight (Fazar et al. 2026: bloom reaches 100–270 px and depends on detector position and timing); ordinary stars of this brightness swing by factors of 2–4 in detectors 1–4, so the predefined 5× threshold corresponds to a 35–65% change per band — a weak test. "
                     "Detectors 5–6 are excluded by a predefined saturation guard; the 4.4–5 µm light curve was made after the result with that guard "
                     "switched off, so it is descriptive. Published infrared maxima of such stars lag the visible ones by about 0.05–0.13 of a period, "
                     "12–32 days for R Draconis (Smith et al. 2002)."),
        method_anchor="science.html#e9"))
