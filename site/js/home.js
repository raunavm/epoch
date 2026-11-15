import { init, getJSON, h, t, link, unavailable } from './core.js';
import { mountReveal } from './components/reveal.js';

await init();
document.getElementById('explore').href = link('timemachine.html');
const el = document.getElementById('reveal');
const data = await getJSON('reveal.json');
if (!data || !(data.frames?.length || data.stack_pngs?.length)) el.replaceChildren(unavailable(t('what.reveal')));
else {
  el.prepend(h('h2', { class: 'object-name' }, data.object));
  const body = h('div');
  el.append(body);
  mountReveal(body, data, { autoplay: true });
}
