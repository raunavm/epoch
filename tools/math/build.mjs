// Build-time LaTeX -> MathML for a Markdown page of the site (Temml, MIT; https://temml.org).
// Usage (from the repo root, after `npm install` in tools/math):
//   node tools/math/build.mjs site/data/methods.md site/data/methods.math.json
// The equations are collected by running the site's own renderer (site/js/markdown.js) with a recording map, so the keys
// ('i:<tex>' inline, 'd:<tex>' display) are exactly the ones the page will look up. Any TeX error stops the build.
import { readFileSync, writeFileSync } from 'node:fs';
import temml from 'temml';
import { renderMarkdown } from '../../site/js/markdown.js';

const [src, out] = process.argv.slice(2);
if (!src || !out) { console.error('usage: node tools/math/build.mjs <page.md> <page.math.json>'); process.exit(2); }

const keys = new Set();
renderMarkdown(readFileSync(src, 'utf8'), u => u, new Proxy({}, { get: (_, k) => { if (typeof k === 'string') keys.add(k); } }));

const map = {}, errors = [];
for (const k of [...keys].sort()) {
  const tex = k.slice(2);
  try { map[k] = temml.renderToString(tex, { displayMode: k.startsWith('d:'), throwOnError: true, annotate: true }); }
  catch (e) { errors.push(`${k}\n  ${e.message}`); }
}
if (errors.length) { console.error(`${errors.length} equation(s) failed:\n${errors.join('\n')}`); process.exit(1); }
writeFileSync(out, JSON.stringify(map) + '\n');
console.log(`${Object.keys(map).length} equations -> ${out}`);
