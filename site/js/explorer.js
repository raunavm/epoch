import { init, getJSON, h, t, unavailable } from './core.js';
import { mountInjections } from './components/injgrid.js';

await init();
const [inj, limits] = await Promise.all([getJSON('injections.json'), getJSON('limits.json')]);
const box = document.getElementById('inj');
if (!inj?.grid || !inj.cells?.length) box.replaceChildren(unavailable(t('what.injections')));
else mountInjections(box, inj);
const lim = document.getElementById('limits');
if (!limits?.sentences?.length) lim.replaceChildren(unavailable(t('what.limits')));
else lim.replaceChildren(h('ul', { class: 'plain-list' }, limits.sentences.map(s => h('li', {}, s))));
