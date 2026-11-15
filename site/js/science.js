import { init, getText, h, t, asset, unavailable } from './core.js';
import { renderMarkdown } from './markdown.js';

await init();
const doc = document.getElementById('doc');
const md = await getText('science.md');
if (!md) doc.replaceChildren(h('h1', {}, t('title.science')), unavailable(t('what.science')));
else doc.innerHTML = renderMarkdown(md, asset); // renderer escapes all text; see markdown.js
