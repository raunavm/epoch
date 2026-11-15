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
| `timemachine.html` | `timemachine/index.json`, `timemachine/<field>.json`, `findings.json` (f1) |
| `objects.html` | `objects.json` |
| `object.html?id=<id>` | `objects/<id>.json` (`id` is restricted to `[A-Za-z0-9_.-]`) |
| `explorer.html` | `injections.json`, `limits.json` |
| `science.html` | `science.md` (rendered by `js/markdown.js`, which escapes all text) |
| `about.html` | `about.json` |

`data/i18n/en.json` and `data/i18n/es.json` hold every UI string. These two files belong to the site, not the pipeline: **`build_site_data.py` must not delete `data/i18n/`.** To add a language, add `data/i18n/<code>.json` and an entry in `LANGS` in `js/core.js`. The chosen language is kept in `localStorage`, and `?lang=es` also works.

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
