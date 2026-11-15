# Orrery site

Static site: plain HTML, CSS and JavaScript modules. No build step, no framework, and no requests to outside services at runtime. It uses system fonts only.

## Run

```bash
cd site
python3 -m http.server 8000
# real data:     http://localhost:8000/
# dev fixtures:  http://localhost:8000/?fixtures=1
```

Opening the files with `file://` does not work, because browsers block module and `fetch` loads from `file://`. Use any static server.

## Data

Pages read only the JSON files in `site/data/` and the PNGs those files reference. `scripts/build_site_data.py` writes them. PNG and markdown figure paths are **relative to `site/data/`**. Absolute `http(s)` URLs are also accepted.

If a file is missing or malformed, the part of the page that needs it shows "Not yet available". A finding missing from `findings.json` is left out of the Start-here tour.

| Page | Reads |
|---|---|
| `index.html` | `headline.json` `{field, question, number, err?, unit?, number_label, sentence, kind?, version, nulls?, method?}`, `timemachine/<field>.json`, `reveal.json` (below the fold) |
| `start.html` | `findings.json` (f1–f3; f4 is no longer on the main path), plus each step's demo data |
| `timemachine.html` | live: `index/dr1/*` + S3 pixels; `?target=<id>`: `targets/<id>.json` `{title, plain_result, numbers:[{label, value, err, unit?, kind, nulls?, method?}], evidence_links:[{text, href}], method_note}` and `timemachine/<id>.json`; prepared fields: `timemachine/index.json`, `timemachine/<field>.json` (optional `gaia: [{x, y, g}]`, per-frame `flags`, `meta`, `s3_url`); `findings.json` (f1) |
| `objects.html` ("Evidence") | `objects.json` |
| `object.html?id=<id>` | `objects/<id>.json` (`id` is restricted to `[A-Za-z0-9_.-]`) |
| `science.html`, `failed.html` | `science.md`, `failed.md` (rendered by `js/markdown.js`, which escapes all text) |
| `about.html` | `about.json` |
| `explorer.html` (not linked) | `injections.json` |

Every number on the landing page, the target cards and the findings is a button that opens its evidence (`js/components/evidence.js`): the pictures behind it with links to the original S3 files, any null tests, and the method section on `science.html`.

`data/i18n/en.json` and `data/i18n/es.json` hold every UI string. These two files belong to the site, not the pipeline: **`build_site_data.py` must not delete `data/i18n/`.** To add a language, add `data/i18n/<code>.json` and an entry in `LANGS` in `js/core.js`. The chosen language is kept in `localStorage`, and `?lang=es` also works.

## Live mode (any coordinates)

`timemachine.html?ra=<deg|hms>&dec=<deg|dms>&fov=<1|2|4|8>&mode=<raw|corrected|add|subtract>` (or `?target=barnard|epsindib|wise0855|rwaur|rdra`). Positions covered by more than 400 frames load 300 frames spread evenly in time by default; `&all=1` (or the "Load all" button) loads every frame. The page then reads SPHEREx DR1 pixels **directly from the public S3 bucket in the browser**. The bucket allows cross-origin requests and the `Range` header. IRSA services are never called.

1. The page loads `data/index/dr1/meta.json` and the 3×3 neighbourhood of sky tiles (`js/live/index.js`, the same rule as `scripts/build_tiles.py`). It keeps frames whose footprint contains the target with a margin of FOV/2.
2. For each frame, at most 6 at a time, it makes two range requests: the header (bytes 0–28799), then the needed full-width image rows. Each frame costs about `rows × 8160` bytes (`js/live/fits.js`).
3. It applies TAN-SIP world→pixel (`js/live/wcs.js`: inverse SIP, then Newton iterations on the forward SIP). It resamples bilinearly onto a north-up 6.15″ grid centred on the target and takes the wavelength from `wave_tables.json` (`js/live/grid.js`).
4. When all frames are in, it runs the change model, revision 2 in `docs/INTERFACES.md` (`js/live/change.js`). The noise floor `f` is read from `data/f1_calibration.json` `{f}`; until that file exists the page uses 0.1 and marks it provisional. Live mode has no FLAGS plane, so single-pixel spikes are removed with a 3×3 median test. This step enables Corrected, Add (per-detector co-adds) and Subtract (earlier pictures minus later ones, split at the largest gap in time; made from the colour-aware pictures). The finding-1 headline comes only from `findings.json` f1; the live counts are labelled "this view (live)".
5. Clicking a pixel gives quick aperture photometry for every frame, with a CSV download.

A new data release needs new index files in `data/index/<release>/`, written by `scripts/build_tiles.py --release <name>`, and a one-line change: point `INDEX` in `js/live/live.js` at `data/index/<name>/`. Live mode shows one release at a time.

Tests (pure functions; the WCS test reads real headers from S3, set `OFFLINE=1` to skip it):

```bash
node --test site/tests/live.test.mjs
```

Object pages accept `&check=all` to open every "Check it" panel on load (for screenshots).

## Dev fixtures

`dev-fixtures/` holds small **synthetic** files that match the contract. Every page loaded with `?fixtures=1` shows a "DEV FIXTURE — NOT DATA" banner. To regenerate them (standard library only):

```bash
python3 site/dev-fixtures/make_fixtures.py
```

Never copy anything from `dev-fixtures/` into `data/`.

## Checks

```bash
for f in site/js/*.js site/js/components/*.js; do node --check "$f"; done
```

## Layout

- `js/core.js`: data loading (`?fixtures=1` switch), i18n, DOM builder, DATA/MODEL/DERIVED/RESIDUAL badges, colour scales (viridis for wavelength, Okabe-Ito for categories)
- `js/plot.js`: small SVG plotting
- `js/components/`: the four interactions (`reveal`, `blink`, `injgrid`), shared by the full pages and the Start-here tour
- `js/markdown.js`: minimal Markdown renderer for `science.md`
- `js/live/`: live mode (`index`, `fits`, `wcs`, `grid`, `change`, `targets`, and the UI `live.js`)
