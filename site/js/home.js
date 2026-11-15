// Landing: the question, one interaction (raw <-> corrected on a real field), one measured number.
import { init, getJSON, h, put, t, link, badge, fmtPM, unavailable, REDUCED_MOTION } from './core.js';
import { mountBlink } from './components/blink.js';
import { mountReveal } from './components/reveal.js';
import { evidenceNum, fieldFrames } from './components/evidence.js';

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
  const [btn, panel] = evidenceNum(h('span', { class: 'big-figure' }, fmtPM(headline.number, headline.err), headline.unit ? ` ${headline.unit}` : ''), {
    loadFrames: () => fieldFrames(getJSON)(fieldId),
    nulls: headline.nulls, nullsNote: headline.nulls_note,
    method: headline.method || 'science.html#which-change-is-real',
    note: headline.evidence_note,
  });
  put($('headline-number'), h('p', { class: 'figure-line' }, btn, ' ', h('span', { class: 'figure-label' }, headline.number_label || ''), ' ', badge(headline.kind || 'derived')), panel);
}

// Below the fold: the stacking reveal, played once when it scrolls into view.
const rv = $('reveal'), data = await getJSON('reveal.json');
if (!data?.frames?.length) rv.replaceChildren(unavailable(t('what.reveal')));
else {
  const body = h('div');
  put(rv, h('h3', { class: 'object-name' }, data.designation || data.object), body);
  const play = () => mountReveal(body, data, { autoplay: !REDUCED_MOTION });
  if (!('IntersectionObserver' in window)) play();
  else {
    let done = false;
    const io = new IntersectionObserver(es => { if (!done && es.some(e => e.isIntersecting)) { done = true; io.disconnect(); play(); } }, { threshold: 0.3 });
    mountReveal(body, data, { autoplay: false });
    io.observe(rv);
  }
}
