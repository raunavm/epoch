import { init, getJSON, t, unavailable } from './core.js';
import { mountInjections } from './components/injgrid.js';

await init();
const inj = await getJSON('injections.json');
const box = document.getElementById('inj');
if (!inj?.grid || !inj.cells?.length) box.replaceChildren(unavailable(t('what.injections')));
else mountInjections(box, inj);
