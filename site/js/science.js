// Renders a Markdown page from the data directory: science.md (default) or the file named in <body data-doc>.
// <body data-math> names the LaTeX-to-MathML map built for that page (see markdown.js).
import { init, getJSON, getText, h, t, asset, unavailable } from './core.js';
import { renderMarkdown } from './markdown.js';

await init();
const doc = document.getElementById('doc');
const file = document.body.dataset.doc || 'science.md';
const md = await getText(file);
const math = document.body.dataset.math ? (await getJSON(document.body.dataset.math)) || {} : null;
if (!md) doc.replaceChildren(h('h1', {}, t(document.body.dataset.title)), unavailable(t('what.science')));
else {
  doc.innerHTML = renderMarkdown(md, asset, math); // renderer escapes all text; see markdown.js
  const target = location.hash ? document.getElementById(decodeURIComponent(location.hash.slice(1))) : null;
  target?.scrollIntoView();
}
