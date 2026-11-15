// Guided path through the findings. A finding missing from findings.json is omitted.
import { init, getJSON, h, t, link, badge, fmtPM, unavailable } from './core.js';
import { findingLine, headlineLine } from './components/evidence.js';
import { renderTargetCard } from './components/targets.js';
import { mountReveal } from './components/reveal.js';
import { mountBlink } from './components/blink.js';
import { sizeText } from './components/injgrid.js';

await init();
const box = document.getElementById('tour');
const [findings, headline] = await Promise.all([getJSON('findings.json'), getJSON('headline.json')]);
const DEFAULT = { f1: 'timemachine', f2: 'reveal', f3: 'object' };
const KINDS = ['timemachine', 'reveal', 'object', 'target'];
// Steps: every finding present in findings.json (f1, f2, …), in order, whose interaction this page can show.
const steps = Object.keys(findings || {}).filter(k => /^f\d+$/.test(k)).sort((a, b) => a.slice(1) - b.slice(1))
  .map(k => ({ key: k, f: findings[k], kind: findings[k].interaction || DEFAULT[k] })).filter(s => KINDS.includes(s.kind));
const PAGE = { timemachine: 'timemachine.html', reveal: 'index.html#faint', object: 'objects.html' };
const pageOf = s => (s.kind === 'target' ? `timemachine.html?target=${encodeURIComponent(s.f.target || '')}` : PAGE[s.kind] || 'index.html');

const MINI = {
  async reveal(el) { const d = await getJSON('reveal.json'); return d?.frames?.length ? mountReveal(el, d, { compact: true, autoplay: true }) : el.replaceChildren(unavailable(t('what.reveal'))); },
  async target(el, f) { await renderTargetCard(el, String(f.target || '').replace(/[^\w.-]/g, ''), getJSON); },
  async timemachine(el) {
    const idx = await getJSON('timemachine/index.json');
    const id = headline?.field || idx?.[0]?.field;
    const f = id ? await getJSON(`timemachine/${String(id).replace(/[^\w.-]/g, '')}.json`) : null;
    return f?.frames?.length ? mountBlink(el, f, { compact: true }) : el.replaceChildren(unavailable(t('what.field')));
  },
  async object(el) {
    const rows = await getJSON('objects.json');
    const r = rows?.find(o => o.D_km != null && o.D_truth_km != null) || rows?.[0];
    if (!r) return el.replaceChildren(unavailable(t('what.objects')));
    const region = h('p', { id: 'start-check', hidden: true }, t('obj.independent'), ' ', h('strong', {}, r.D_truth_km != null ? sizeText(r.D_truth_km) : '—'),
      r.D_truth_source ? ` (${r.D_truth_source}) ` : ' ', badge('data'));
    const btn = h('button', { type: 'button', class: 'check', 'aria-expanded': 'false', 'aria-controls': 'start-check', onclick: () => {
      region.hidden = !region.hidden; btn.setAttribute('aria-expanded', String(!region.hidden));
    } }, t('obj.check'));
    el.replaceChildren(h('p', { class: 'mini-object' }, h('a', { href: link(r.page || `object.html?id=${encodeURIComponent(r.id)}`) }, r.name), ': ',
      h('span', { class: 'big-num' }, r.D_km != null ? `${fmtPM(r.D_km, r.D_err_km)} km` : '—'), ' ', badge('derived'), ' ', btn), region);
  },
};

let cur = 0, active = null;
if (!steps.length) box.replaceChildren(unavailable(t('what.findings')));
else {
  document.addEventListener('keydown', e => {
    if (e.defaultPrevented || e.target.closest?.('input, select, textarea')) return;
    if (e.key === 'ArrowRight') go(cur + 1); else if (e.key === 'ArrowLeft') go(cur - 1);
  });
  go(0, false);
}

async function go(k, focus = true) {
  if (k < 0 || k >= steps.length) return;
  active?.stop?.();
  cur = k;
  const s = steps[k], mini = h('div', { class: 'mini panel' });
  const title = h('h2', { tabindex: '-1' }, s.f.title || t(`start.title_${s.kind}`));
  box.replaceChildren(
    h('p', { class: 'muted' }, t('start.progress', { k: k + 1, n: steps.length })),
    title, s.key === 'f1' && headline?.sentence ? headlineLine(headline) : findingLine(s.f, s.key),
    h('div', { class: 'tour-nav' },
      h('button', { type: 'button', class: 'btn', disabled: k === 0, onclick: () => go(cur - 1) }, '← ', t('start.back')),
      h('ol', { class: 'dots', 'aria-label': t('start.dots') }, steps.map((_, j) => h('li', {},
        h('button', { type: 'button', class: 'dot', 'aria-label': t('start.progress', { k: j + 1, n: steps.length }), 'aria-current': j === k ? 'step' : null, onclick: () => go(j) })))),
      k < steps.length - 1
        ? h('button', { type: 'button', class: 'btn primary', onclick: () => go(cur + 1) }, t('start.next'), ' →')
        : h('a', { class: 'btn primary', href: link('timemachine.html') }, t('home.cta2'))),
    mini,
    h('p', {}, h('a', { href: link(pageOf(s)) }, t('start.open_full'))),
  );
  if (focus) title.focus();
  const ctl = await MINI[s.kind]?.(mini, s.f);
  if (cur === k) active = ctl; else ctl?.stop?.();
}
