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


# WISE 0855 — E4
e4 = load("data/findings/e4.json")
if e4:
    p = e4["passes"]
    p2 = p[1]
    p3 = p[2].get("neighbour_restored") or p[2]  # pass 3 with the wrongly dropped Gaia neighbour put back (see deviations)
    sh = e4["centroid_shift_vs_prediction"][0]
    sl = e4["slider_stack"]
    write("wise0855", dict(
        title="WISE 0855−0714 — the coldest known brown dwarf",
        plain_result=(f"Too faint for most single pictures, but where it should be at 4.4–5 µm SPHEREx measures "
                      f"{p2['a_hat_mJy']:.2f} ± {p2['sigma_a_mJy']:.2f} mJy in its second sky survey — the brightness Spitzer measured "
                      f"({e4['expected_flux_mJy']['IRAC_4p5']:.2f} mJy) — and it moved between surveys as predicted; standing still is ruled out."),
        numbers=[num("Signal in survey 2, against 24 empty paths", round(p2["significance"], 1), unit="σ",
                     nulls=p2.get("null_snrs"), nulls_note="same pictures, path shifted 6–9 px", method="science.html#e4"),
                 num("Signal in survey 3 (a neighbour star restored to the model)", round(p3.get("significance", float("nan")), 1), unit="σ",
                     method="science.html#e4"),
                 num("Brightness at 4.4–5 µm, survey 2", round(p2["a_hat_mJy"], 3), round(p2["sigma_a_mJy"], 3), unit="mJy", kind="derived"),
                 num("Shift between surveys vs prediction", f"{sh['measured_shift_arcsec'][0]:+.1f}″, {sh['measured_shift_arcsec'][1]:+.1f}″ "
                     f"(predicted {sh['predicted_shift_arcsec'][0]:+.1f}″, {sh['predicted_shift_arcsec'][1]:+.1f}″)", None, unit=""),
                 num("Chance it is a source that does not move", f"{sh['p_vs_stationary']:.0e}", None, unit="")],
        evidence_links=[dict(text="All 46 pictures used (4.4–5 µm)", href="science.html#e4"),
                        dict(text="Astrometry: Luhman & Esplin 2016, AJ 152, 78", href="https://arxiv.org/abs/1605.06655")],
        method_note=("Position predicted from proper motion and parallax (Luhman & Esplin 2016); each picture fitted with Gaia stars; "
                     "pictures at 4.4–5 µm combined per survey; significance quoted against the same path shifted to 24 empty positions. "
                     "A statement about a known object, not a discovery. Survey 1 has 3 usable pictures and is not informative."),
        method_anchor="science.html#e4", slider="e4/reveal.json"))

# Motion measurements (Epsilon Indi B, Barnard's Star)
for tid, name in (("epsindib", "Epsilon Indi B — a pair of nearby brown dwarfs"), ("barnard", "Barnard's Star — the fastest-moving star in our sky")):
    m = load(f"data/findings/motion_{tid}.json")
    if not m:
        continue
    pa, pd_ = m["proper_motion"]["pmra"], m["proper_motion"]["pmdec"]
    write(tid, dict(
        title=name,
        plain_result=(f"From SPHEREx pictures alone we measured how fast it moves across the sky: "
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
                     "the motion is a straight-line fit to SPHEREx positions alone with Gaia's parallax held fixed. "
                     + ("Positions sit a steady ~0.5″ south of Gaia's prediction in every survey; the pair is unresolved, so a shifted centre of light is possible — not explained here." if tid == "epsindib" else "")),
        method_anchor="science.html#motion"))

# Variables: report what the change view shows, with honest scope
for tid, name, kind in (("rwaur", "RW Aurigae — a young variable star", "variable"), ("rdra", "R Draconis — a pulsating giant star", "variable")):
    f = load(f"data/findings/f1_{tid}.json")
    if not f:
        continue
    write(tid, dict(
        title=name,
        plain_result=(f"{f['n_frames']} SPHEREx pictures over {f['n_passes']} visits. "
                      + (f"The star really changes brightness, and the colour-aware view keeps flagging it: {f.get('corrected_matched_known', 0)} of the "
                         f"{f['n_corrected']} remaining changes are at the star itself. " if f.get('corrected_matched_known', 0) > 0 else
                         f"None of the {f['n_corrected']} remaining changes were matched to the star itself — likely because a star this bright "
                         "saturates SPHEREx's detector at its centre, where pixels are excluded (not yet checked for this field). ")
                      + "A light curve (one brightness per picture after removing the star's colour) is not measured yet."),
        numbers=[num("Pictures", f["n_frames"], None, kind="data"),
                 num("Changes in a time-ordered blink", f["n_naive_all"], None),
                 num("Changes in a same-detector blink", f["n_naive_det"], None),
                 num("Changes after the colour-aware view", f["n_corrected"], None),
                 num("…of which at the star itself", f.get("corrected_matched_known", 0), None)],
        evidence_links=[dict(text="Open the prepared field", href=f"timemachine.html?field={tid}")],
        method_note="Counts use the frozen change model (revision 2). A variable star should keep showing up after correction; a still star should not.",
        method_anchor="science.html#which-change-is-real"))
