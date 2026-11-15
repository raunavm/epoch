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
| `index.html` | `reveal.json` |
| `start.html` | `findings.json`, plus the files of each finding's demo (`reveal.json`, `timemachine/index.json` and its first field, `objects.json`, `injections.json`) |
| `timemachine.html` | live: `index/dr1/meta.json`, `index/dr1/<b>_<r>.json`, `index/dr1/wave_tables.json` + S3 pixels; showcase: `timemachine/index.json`, `timemachine/<field>.json`, `findings.json` (f1) |
| `objects.html` | `objects.json` |
| `object.html?id=<id>` | `objects/<id>.json` (`id` is restricted to `[A-Za-z0-9_.-]`) |
| `explorer.html` | `injections.json`, `limits.json` |
| `science.html` | `science.md` (rendered by `js/markdown.js`, which escapes all text) |
| `about.html` | `about.json` |

`data/i18n/en.json` and `data/i18n/es.json` hold every UI string. These two files belong to the site, not the pipeline: **`build_site_data.py` must not delete `data/i18n/`.** To add a language, add `data/i18n/<code>.json` and an entry in `LANGS` in `js/core.js`. The chosen language is kept in `localStorage`, and `?lang=es` also works.

## Live mode (any coordinates)

`timemachine.html?ra=<deg|hms>&dec=<deg|dms>&fov=<1|2|4|8>&mode=<raw|corrected|add|subtract>` (or `?target=barnard|epsindib|wise0855|rwaur|rdra`) reads SPHEREx DR1 pixels **directly from the public S3 bucket in the browser**. The bucket allows cross-origin requests and the `Range` header. IRSA services are never called.

1. The page loads `data/index/dr1/meta.json` and the 3×3 neighbourhood of sky tiles (`js/live/index.js`, the same rule as `scripts/build_tiles.py`). It keeps frames whose footprint contains the target with a margin of FOV/2.
2. For each frame, at most 6 at a time, it makes two range requests: the header (bytes 0–28799), then the needed full-width image rows. Each frame costs about `rows × 8160` bytes (`js/live/fits.js`).
3. It applies TAN-SIP world→pixel (`js/live/wcs.js`: inverse SIP, then Newton iterations on the forward SIP). It resamples bilinearly onto a north-up 6.15″ grid centred on the target and takes the wavelength from `wave_tables.json` (`js/live/grid.js`).
4. When all frames are in, it runs the change model, revision 2 in `docs/INTERFACES.md` (`js/live/change.js`). The noise floor `f` is read from `data/f1_calibration.json` `{f}`; until that file exists the page uses 0.1 and marks it provisional. Live mode has no FLAGS plane, so single-pixel spikes are removed with a 3×3 median test. This step enables Corrected, Add (per-detector co-adds) and Subtract (pass 1 − pass 2). The finding-1 headline comes only from `findings.json` f1; the live counts are labelled "this view (live)".
5. Clicking a pixel gives quick aperture photometry for every frame, with a CSV download.

New data releases need only new index files in `data/index/<release>/`, written by `scripts/build_tiles.py`.

Tests (pure functions; the WCS test reads real headers from S3, set `OFFLINE=1` to skip it):

```bash
node --test site/tests/live.test.mjs
```

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
