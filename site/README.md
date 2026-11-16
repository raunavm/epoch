# Epoch site

Static site: plain HTML, CSS and JavaScript modules, no framework and no build step for visitors. Fonts are Inter, Public Sans and DM Mono from Google Fonts. Pixels come from NASA's public SPHEREx bucket on AWS. Two optional overlays, only when switched on, also call outside services: the Gaia DR3 overlay (VizieR `I/355/gaiadr3` on the CfA mirror, falling back to GAVO's `gaia.dr3lite` TAP) and the WISE comparison (AllWISE W2 HiPS tiles from NASA/IPAC IRSA's mirror of the CDS HiPS).

## Run

```bash
cd site
python3 -m http.server 8000
# real data:     http://localhost:8000/
# dev fixtures:  http://localhost:8000/?fixtures=1
```

Opening the files with `file://` does not work, because browsers block module and `fetch` loads from `file://`. Use any static server.

## Pages

| Page | What it shows | Reads |
|---|---|---|
| `index.html` | Time machine: opening picture (Barnard's Star, WISE 2010 and SPHEREx 2025), any position live, prepared fields, challenge-target result cards | `hero/*.png`, `index/dr1/*` + S3 pixels, `timemachine/index.json`, `timemachine/<field>.json`, `targets/<id>.json` |
| `results.html` | Results, one short section per result | `science.md` |
| `methods.html` | Methods, written like a paper, with equations | `methods.md`, `methods.math.json` |
| `about.html` | Data sources, data products, credits, references | `about.json`, `products/` |

`timemachine.html`, `science.html`, `objects.html` and `object.html` only redirect, keeping the query and hash, so old links still work.

Paths in data files (figures, products) are relative to `site/data/`. If a file is missing or malformed, the part of the page that needs it shows "Not yet available".

## Text and numbers

- `data/i18n/en.json` and `es.json` hold every interface string. To add a language, add `data/i18n/<code>.json` and an entry in `LANGS` in `js/core.js`. The chosen language is kept in `localStorage`; `?lang=es` also works. `data/i18n/data_es.json` translates text that comes from data files.
- `science.md` and `about.json` are made from `science.md.tmpl` and `about.json.tmpl` by `scripts/fill_tokens.py`, which replaces each `[[file.key|format]]` with the value in `data/findings/<file>.json`. Edit the `.tmpl`, then refill; never type a number into the filled file.
- Target cards (`data/targets/<id>.json`) are edited by hand. `scripts/build_targets.py` would overwrite them.

## Equations (Methods page)

`js/markdown.js` turns `$…$` and `$$…$$` into MathML taken from a map built in advance, so no maths code runs in the visitor's browser. Display equations are numbered in order. To rebuild the map after editing `methods.md`:

```bash
cd tools/math && npm install && cd ../..        # once; Temml (MIT) is a build tool and is not deployed
node tools/math/build.mjs site/data/methods.md site/data/methods.math.json
```

The build stops on any TeX error. TeX with no entry in the map is shown as code.

## Figures

`data/figs/v2/*.svg` (with PDF and PNG) are drawn by `figures/plot_<name>.py` with the shared style in `figures/style.mplstyle` and `figures/common.py`: Okabe–Ito colours in fixed roles (SPHEREx, reference, model, control, highlight), no text inside the axes, legends outside. Run from the repo root, for example `.venv/bin/python figures/plot_e28_rdra.py`.

## Live mode (any coordinates)

`index.html?ra=<deg|hms>&dec=<deg|dms>&fov=<1|2|4|8>` (or `?target=barnard|epsindib|wise0855|rwaur|rdra|nep`). Positions covered by more than 400 pictures load 300 spread evenly in time by default; `&all=1` (or the "Load all" button) loads every picture. The page reads SPHEREx DR1 pixels **directly from the public S3 bucket in the browser**. The bucket allows cross-origin requests and the `Range` header. IRSA services are never called for pixels.

1. The page loads `data/index/dr1/meta.json` and the 3×3 neighbourhood of sky tiles (`js/live/index.js`, the same rule as `scripts/build_tiles.py`). It keeps pictures whose footprint contains the target with a margin of FOV/2.
2. For each picture, at most 6 at a time, it makes two range requests: the header (bytes 0–28799), then the needed full-width image rows (`js/live/fits.js`).
3. It applies TAN-SIP world→pixel (`js/live/wcs.js`), resamples bilinearly onto a north-up 6.15″ grid centred on the target, and takes the wavelength from `wave_tables.json` (`js/live/grid.js`).
4. It counts changes for the plain blink and for one detector at a time (`js/live/change.js`), with the noise floor `f` from `data/f1_calibration.json`. Live mode has no flag plane, so single-pixel spikes are removed with a 3×3 median test.
5. Clicking a pixel gives quick aperture photometry for every picture, with a CSV download (a quick look; IRSA's SPHEREx Spectrophotometry Tool is the reference).
6. `&gaia=1`: Gaia DR3 stars moved from J2016.0 to each picture's date (`js/live/gaia.js`).
7. `&wise=1`: AllWISE W2 (2010) on the same grid, beside and blinking with the SPHEREx detector-6 median (`js/live/wise.js`).
8. `&mode=track&rate=<″/h>&pa=<deg east of north>&t0=<mjd>&span=<h>`: pictures shifted along a straight-line motion and added (`js/live/track.js`).
9. `&exact=1` keeps a typed position even when it lies on the path of a fast-moving challenge target.

Both viewers have a contrast control and "Save film", which saves the pictures as an animated PNG encoded in the browser (`js/components/film.js`).

A new data release needs new index files in `data/index/<release>/`, written by `scripts/build_tiles.py --release <name>`, and one change: set `RELEASE` in `js/live/live.js`.

## Tests and checks

```bash
node --test site/tests/live.test.mjs        # pure functions; the WCS test reads real headers from S3 (OFFLINE=1 skips it)
for f in site/js/*.js site/js/components/*.js site/js/live/*.js; do node --check "$f"; done
```

## Dev fixtures

`dev-fixtures/` holds small **synthetic** files for development. Every page loaded with `?fixtures=1` shows a "DEV FIXTURE — NOT DATA" banner. Never copy anything from `dev-fixtures/` into `data/`.

## Layout

- `js/core.js`: data loading, i18n, navigation, DOM builder, number formatting, colour scales
- `js/timemachine.js`: the home page; `js/science.js`: Markdown pages (Results, Methods); `js/about.js`
- `js/components/`: `intro` (opening picture), `blink` (prepared-field viewer), `film` (contrast and Save film), `targets` (result cards), `reveal`, `evidence`, `injgrid`
- `js/markdown.js`: small Markdown renderer that escapes all text; equations from the prebuilt map
- `js/live/`: live mode (`index`, `fits`, `wcs`, `grid`, `change`, `targets`, `gaia`, `wise`, `track`, and the interface `live.js`)
