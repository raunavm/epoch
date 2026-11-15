// Landing: the question, one interaction (raw <-> corrected on a real field), one measured number.
import { init, getJSON, h, put, t, fmt, loc, locLang, link, badge, unavailable, REDUCED_MOTION } from './core.js';
import { mountBlink } from './components/blink.js';
import { mountReveal } from './components/reveal.js';
import { headlineNumber } from './components/evidence.js';

await init();
const $ = id => document.getElementById(id);
$('explore').href = link('timemachine.html');

const [headline, index] = await Promise.all([getJSON('headline.json'), getJSON('timemachine/index.json')]);
if (headline?.version) document.body.dataset.version = headline.version;
// The question is also an i18n string (home.question): the data file's wording replaces it only in its own language.
const question = locLang(headline, 'question') ? null : loc(headline, 'question');
if (question) { $('h-hero').textContent = question; document.title = `${question} — ${t('site.name')}`; }
$('headline-sentence').textContent = loc(headline, 'sentence') || t('home.sentence_default');
if (headline?.sentence) $('headline-sentence').lang = locLang(headline, 'sentence') || '';

const fieldId = String(headline?.field || index?.[0]?.field || '').replace(/[^\w.-]/g, '');
const field = fieldId ? await getJSON(`timemachine/${fieldId}.json`) : null;
const box = $('headline-field');
if (field?.frames?.length) {
  const body = h('div');
  // This patch's own counts (summed from its field file), so the demo and the headline median are not confused.
  const sum = k => field.frames.reduce((a, f) => a + (f[k] || 0), 0), nAll = sum('n_changes_naive_all'), nDet = sum('n_changes_naive_det');
  const own = nAll > 0 && field.frames.some(f => f.n_changes_naive_det != null) ? h('p', { class: 'muted small' },
    t('home.patch_line', { all: nAll, det: nDet, pct: fmt(100 * (1 - nDet / nAll), 2) }),
    headline?.number != null && headline.evidence?.length ? ` ${t('home.patch_median', { hl: `${fmt(headline.number)}${headline.unit || ''}`, n: headline.evidence.length })}` : '',
    ' ', badge('derived')) : null;
  put(box, h('p', { class: 'field-name', lang: locLang(field, 'name') }, loc(field, 'name')), own, body);
  mountBlink(body, field, { minimal: true });
} else put(box, unavailable(t('what.field')));

if (headline?.number != null) {
  const [btn, panel, interval] = headlineNumber(headline, true);
  put($('headline-number'), h('p', { class: 'figure-line' }, btn, ' ', h('span', { class: 'figure-label', lang: locLang(headline, 'number_label') }, loc(headline, 'number_label') || ''), ' ', badge(headline.kind || 'derived')),
    interval ? h('p', { class: 'figure-ci' }, interval) : null, panel);
}

// Below the fold: two objects too faint for most single pictures, each played once when it scrolls into view.
const examples = [['reveal.json', 'home.faint_asteroid'], ['e4/reveal.json', 'home.faint_bd']];
const rv = $('reveal');
const datas = await Promise.all(examples.map(([f]) => getJSON(f)));
if (!datas.some(d => d?.frames?.length)) rv.replaceChildren(unavailable(t('what.reveal')));
else put(rv, ...datas.map((data, k) => {
  if (!data?.frames?.length) return null;
  const body = h('div'), wrap = h('div', { class: 'example' }, h('h3', { class: 'object-name' }, data.object || data.designation),
    h('p', { class: 'hint' }, t(examples[k][1])), body, data.hedge ? h('p', { class: 'hedge small', lang: locLang(data, 'hedge') }, loc(data, 'hedge')) : null);
  mountReveal(body, data, { autoplay: false });
  if (!REDUCED_MOTION && 'IntersectionObserver' in window) {
    const io = new IntersectionObserver(es => { if (es.some(e => e.isIntersecting)) { io.disconnect(); mountReveal(body, data, { autoplay: true }); } }, { threshold: 0.4 });
    io.observe(wrap);
  }
  return wrap;
}));
