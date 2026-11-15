// Landing: the question, one interaction (raw <-> corrected on a real field), one measured number.
import { init, getJSON, h, put, t, link, badge, unavailable, REDUCED_MOTION } from './core.js';
import { mountBlink } from './components/blink.js';
import { mountReveal } from './components/reveal.js';
import { headlineNumber } from './components/evidence.js';

await init();
const $ = id => document.getElementById(id);
$('explore').href = link('timemachine.html');

const [headline, index] = await Promise.all([getJSON('headline.json'), getJSON('timemachine/index.json')]);
if (headline?.version) document.body.dataset.version = headline.version;
if (headline?.question) { $('h-hero').textContent = headline.question; document.title = `${headline.question} — ${t('site.name')}`; }
$('headline-sentence').textContent = headline?.sentence || t('home.sentence_default');

const fieldId = String(headline?.field || index?.[0]?.field || '').replace(/[^\w.-]/g, '');
const field = fieldId ? await getJSON(`timemachine/${fieldId}.json`) : null;
const box = $('headline-field');
if (field?.frames?.length) {
  const body = h('div');
  put(box, h('p', { class: 'field-name' }, field.name), body);
  mountBlink(body, field, { minimal: true });
} else put(box, unavailable(t('what.field')));

if (headline?.number != null) {
  const [btn, panel, range] = headlineNumber(headline, true);
  put($('headline-number'), h('p', { class: 'figure-line' }, btn, ' ',
    h('span', { class: 'figure-label' }, headline.number_label || '', range ? h('span', { class: 'muted' }, ` (${range})`) : null), ' ', badge(headline.kind || 'derived')), panel);
}

// Below the fold: two objects too faint for one picture, each played once when it scrolls into view.
const examples = [['reveal.json', 'home.faint_asteroid'], ['e4/reveal.json', 'home.faint_bd']];
const rv = $('reveal');
const datas = await Promise.all(examples.map(([f]) => getJSON(f)));
if (!datas.some(d => d?.frames?.length)) rv.replaceChildren(unavailable(t('what.reveal')));
else put(rv, ...datas.map((data, k) => {
  if (!data?.frames?.length) return null;
  const body = h('div'), wrap = h('div', { class: 'example' }, h('h3', { class: 'object-name' }, data.object || data.designation),
    h('p', { class: 'hint' }, t(examples[k][1])), body);
  mountReveal(body, data, { autoplay: false });
  if (!REDUCED_MOTION && 'IntersectionObserver' in window) {
    const io = new IntersectionObserver(es => { if (es.some(e => e.isIntersecting)) { io.disconnect(); mountReveal(body, data, { autoplay: true }); } }, { threshold: 0.4 });
    io.observe(wrap);
  }
  return wrap;
}));
