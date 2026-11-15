// Renders a Markdown page from the data directory: science.md (default) or the file named in <body data-doc>.
import { init, getText, h, t, asset, unavailable } from './core.js';
import { renderMarkdown } from './markdown.js';

await init();
const doc = document.getElementById('doc');
const file = document.body.dataset.doc || 'science.md';
const md = await getText(file);
if (!md) doc.replaceChildren(h('h1', {}, t(document.body.dataset.title)), unavailable(t(file === 'failed.md' ? 'what.failed' : 'what.science')));
else {
  doc.innerHTML = renderMarkdown(md, asset); // renderer escapes all text; see markdown.js
  const target = location.hash && document.getElementById(decodeURIComponent(location.hash.slice(1)));
  target?.scrollIntoView();
}
