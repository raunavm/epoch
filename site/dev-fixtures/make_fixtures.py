"""Write SYNTHETIC dev fixtures matching the site data contract (stdlib only).

Every value here is made up for layout testing. Nothing in this directory is data.
Run:  python3 site/dev-fixtures/make_fixtures.py
"""
import json, math, random, struct, zlib
from pathlib import Path

OUT = Path(__file__).resolve().parent
rng = random.Random(42)


def png(path, img):
    """Write a list-of-rows float image (any range) as an 8-bit grayscale PNG."""
    flat = [v for row in img for v in row]
    lo, hi = min(flat), max(flat)
    raw = b"".join(b"\x00" + bytes(int(255 * (v - lo) / (hi - lo or 1)) for v in row) for row in img)
    h, w = len(img), len(img[0])
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    p = OUT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
                  + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))
    return path


def field(n, sources, noise=1.0):
    img = [[rng.gauss(0, noise) for _ in range(n)] for _ in range(n)]
    for x, y, a in sources:
        for j in range(n):
            for i in range(n):
                img[j][i] += a * math.exp(-((i - x) ** 2 + (j - y) ** 2) / (2 * 1.3 ** 2))
    return img


def dump(path, obj):
    p = OUT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1))


# --- reveal.json -----------------------------------------------------------
N = 24
wls = sorted(0.75 + 4.2 * rng.random() for _ in range(N))
frames = [{"obs_id": f"FIXTURE_{k:03d}", "mjd": 60800 + 0.4 * k, "wl_um": round(w, 3),
           "single_png": png(f"png/reveal/single_{k:02d}.png", field(32, [(16, 16, 0.8)]))} for k, w in enumerate(wls)]
stacks = [{"n": n, "png": png(f"png/reveal/stack_{n:02d}.png", field(32, [(16, 16, 0.8)], noise=1 / math.sqrt(n))),
           "snr": round(0.8 * math.sqrt(n), 2)} for n in (1, 2, 4, 8, 12, 16, 20, 24)]
flux = lambda w: 0.3 + 0.08 * w
spec = [{"n_frames_used": n, "wl_um": [f["wl_um"] for f in frames[:n]],
         "flux_mJy": [round(flux(f["wl_um"]) + rng.gauss(0, 0.04), 3) for f in frames[:n]],
         "err": [0.05] * n} for n in (4, 8, 12, 16, 20, 24)]
dump("reveal.json", {"object": "Synthetic asteroid FX-1 (fixture)", "frames": frames, "stack_pngs": stacks,
                     "spectrum_points": spec, "depth_gain_mag": 1.7, "depth_gain_err": 0.2})

# --- findings.json ---------------------------------------------------------
dump("findings.json", {
    "f1": {"sentence": "Fixture: share of raw-blink changes that are only colour.", "number": 0.0, "err": 0.0, "unit": "%",
           "detail": {"all_fields": {"n_naive": 10, "n_corrected": 5, "fraction_removed": 0.5, "note": "fixture note"},
                      "per_field": [{"field": "a"}, {"field": "b"}]}},
    "f2": {"sentence": "Fixture: magnitudes gained by combining frames.", "number": 1.7, "err": 0.2, "unit": "mag"},
    "f3": {"sentence": "Fixture: median ratio of our size to the independent size.", "number": 1.0, "err": 0.1},
    "f4": {"sentence": "Fixture: size found half the time at 2.5 AU.", "number": 1.0, "err": 0.5, "unit": "km"},
})

# --- time machine ----------------------------------------------------------
stars = [(rng.uniform(4, 60), rng.uniform(4, 60), rng.uniform(4, 12)) for _ in range(14)]
tm_frames = []
for k in range(8):
    wl = round(0.8 + 0.5 * k, 2)
    ast = (10 + 6 * k, 40 - 3 * k, 6)
    raw = field(64, [(x, y, a * (0.6 + 0.8 * ((hash((x, k)) % 100) / 100))) for x, y, a in stars] + [ast])
    corr_src = [ast] + ([(50, 12, 5)] if k == 3 else [])
    tm_frames.append({
        "obs_id": f"FIXTURE_TM_{k}", "mjd": 60800 + 3.1 * k, "date": f"2026-03-{3 + 3 * k:02d}", "wl_um": wl,
        "img_png": png(f"png/tm/raw_{k}.png", raw), "corr_png": png(f"png/tm/corr_{k}.png", field(64, corr_src)),
        "detector": 1 + k % 6, "s3_url": "https://example.invalid/fixture.fits",
        "flags": {"SOURCE": 300 + k, "OUTLIER": k}, "meta": {"PSF_FWHM": 7.5, "L2DQAFLG": "Pass", "L2 N_OUTLIER": 1000 + k},
        "labels": [{"x": 32, "y": 32, "name": "Fixture target", "kind": "star", "label": "target", "confidence": None, "change": False},
                   {"x": ast[0], "y": ast[1], "name": "FX-1 (synthetic)", "kind": "mover", "label": "known asteroid", "confidence": 0.9, "change": True}]
        + ([{"x": 50, "y": 12, "name": "", "kind": "glint", "label": "unexplained", "confidence": 0.6, "change": True}] if k == 3 else []),
    })
dump("timemachine/index.json", [{"field": "fx-field", "name": "Synthetic field (fixture)", "ra": 270.0, "dec": 66.5}])
dump("timemachine/fx-field.json", {"field": "fx-field", "name": "Synthetic field (fixture)", "ra": 270.0, "dec": 66.5,
                                   "size_px": 64, "frames": tm_frames, "gaia": [{"x": x, "y": y, "g": 15.0} for x, y, _ in stars[:6]]})

# --- objects ---------------------------------------------------------------
dump("objects.json", [
    {"id": "fx-1", "name": "FX-1 (synthetic)", "number": 0, "kind": "fixture", "n_frames": 6, "D_km": 2.1, "D_err_km": 0.3,
     "D_truth_km": 2.0, "D_truth_source": "fixture", "pV": 0.2, "class_plain": "fixture rocky", "page": "object.html?id=fx-1"},
    {"id": "fx-missing", "name": "FX-2 (no detail file)", "number": 1, "kind": "fixture", "n_frames": 0, "D_km": None,
     "D_err_km": None, "D_truth_km": None, "D_truth_source": None, "pV": None, "class_plain": None, "page": "object.html?id=fx-missing"},
])
ofr = []
for k in range(6):
    wl = round(0.9 + 0.7 * k, 2)
    st = field(24, [(12, 12, 8)])
    ofr.append({"obs_id": f"FIXTURE_OBJ_{k}", "mjd": 60810 + 0.5 * k, "detector": 1 + k % 6, "wl_um": wl, "bw_um": round(wl / 40, 3),
                "flux_mJy": round(1 + 0.2 * k, 3), "flux_err_mJy": 0.1, "chi2_dof_local": 1.0,
                "stamp_png": png(f"png/obj/stamp_{k}.png", st), "model_png": png(f"png/obj/model_{k}.png", field(24, [(12, 12, 8)], noise=0)),
                "resid_png": png(f"png/obj/resid_{k}.png", field(24, [])), "s3_url": "https://example.invalid/fixture.fits"})
wl_s = [0.9 + 0.2 * i for i in range(12)]
dump("objects/fx-1.json", {
    "id": "fx-1", "name": "FX-1 (synthetic)", "kind": "fixture", "release": "fixture", "code_version": "fixture",
    "rerun_command": "echo fixture", "track": [{"mjd": 60810 + 0.5 * k, "ra": 359.8 + 0.05 * k, "dec": -1 + 0.03 * k} for k in range(8)],
    "frames": ofr,
    "spectrum": {"wl_um": wl_s, "refl": [round(1 + 0.1 * (w - 0.9) + rng.gauss(0, 0.03), 3) for w in wl_s], "refl_err": [0.04] * 12,
                 "gaia": {"wl_um": [0.4 + 0.05 * i for i in range(12)], "refl": [0.9 + 0.02 * i for i in range(12)], "err": [0.02] * 12}},
    "size": {"D_km": 2.1, "D_err_km": 0.3, "pV": 0.2, "eta": 1.0, "truth": {"D_km": 2.0, "pV": 0.22, "source": "fixture"},
             "familiar": {"name": "a fixture landmark", "km": 1.0}},
    "composition_plain": "Fixture text: rocky, like many inner-belt asteroids.",
    "labels": {"D_km": "derived", "eta": "model"},
})

# --- injections / limits ---------------------------------------------------
Ds, Rs, Ps = [0.5, 1.0, 2.0, 5.0], [1.5, 2.5, 3.5], [0.06, 0.25]
ex1 = png("png/inj/single.png", field(32, [(16, 16, 1.0)]))
ex2 = png("png/inj/stack.png", field(32, [(16, 16, 1.0)], noise=0.25))
cells = []
for d in Ds:
    for r in Rs:
        for p in Ps:
            V = 5 * math.log10(1329 / (d * math.sqrt(p))) + 5 * math.log10(r * (r - 1))
            f = 1 / (1 + math.exp(V - 19))
            cells.append({"D_km": d, "r_au": r, "pV": p, "V": round(V, 2), "n_frames": 20, "recovered_frac": round(f, 2),
                          "snr_median": round(10 * f, 1), "example_single_png": ex1, "example_stack_png": ex2})
dump("injections.json", {"grid": {"D_km": Ds, "r_au": Rs, "pV": Ps}, "cells": cells})
dump("limits.json", {"sentences": ["Fixture sentence: objects fainter than the faintest grid cell are not tested.",
                                   "Fixture sentence: limits text comes from the pipeline."]})

# --- science.md / about.json -----------------------------------------------
(OUT / "science.md").write_text("""# Science (fixture)

## In plain language
Fixture paragraph with **bold**, *italic*, `code` and a [link](https://example.invalid).

- first point
- second point

## Technical account
1. Step one
2. Step two

| quantity | value |
|---|---|
| fixture | 1.0 |

![Fixture figure: a synthetic stack](png/reveal/stack_24.png "Fixture figure caption")

> Fixture quote.

```
python scripts/fixture.py --all
```
<script>alert('this must render as text')</script>
""")
dump("about.json", {
    "note_statement": "Made by the Epoch team.",
    "citations": ["Fixture citation A", {"text": "Fixture citation B", "url": "https://example.invalid"}],
    "acknowledgement": "Fixture acknowledgement text.",
    "data_sources": [{"text": "Fixture data source", "url": "https://example.invalid"}],
    "repo_url": "https://example.invalid/repo",
})
# --- one-question landing, target cards, what failed ---------------------------
dump("headline.json", {"version": "B", "field": "fx-field", "question": "Fixture: at this spot in the sky, what really changed?",
                       "number": 0, "unit": "%", "err_low": 0, "err_high": 0, "kind": "derived",
                       "number_label": "fixture: share of plain-blink changes that are only colour",
                       "sentence": "Fixture sentence: compare a plain blink with one detector at a time.",
                       "modes": [{"id": "all", "label": "Plain blink (every picture, in time order)"}, {"id": "det", "label": "One detector at a time"},
                                 {"id": "corr", "label": "Colour-aware view (experimental)"}], "default_mode": "all",
                       "method": "science.html#which-change-is-real", "evidence_note": "Fixture evidence note.",
                       "evidence": [{"text": "fixture patch: 10 -> 1", "href": "science.html#which-change-is-real"}]})
for tid, name in [("barnard", "Barnard's Star"), ("wise0855", "WISE 0855")]:
    dump(f"targets/{tid}.json", {"title": f"{name} (fixture)", "plain_result": "Fixture result sentence.",
                                 "numbers": [{"label": "fixture motion", "value": 1.0, "err": 0.1, "unit": "″/yr", "kind": "derived"},
                                             {"label": "fixture count", "value": 3, "err": None, "kind": "data"}],
                                 "evidence_links": [{"text": "Science: which change is real", "href": "science.html#which-change-is-real"}],
                                 "method_note": "Fixture method note."})
(OUT / "failed.md").write_text("# What failed (fixture)\n\n## Fixture failure\nFixture paragraph.\n")
print("fixtures written to", OUT)
