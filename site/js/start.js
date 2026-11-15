// Guided path through the findings. A finding missing from findings.json is omitted.
import { init, getJSON, h, t, link, badge, fmtPM, unavailable, findingLine } from './core.js';
import { mountReveal } from './components/reveal.js';
import { mountBlink } from './components/blink.js';
import { mountInjections, sizeText } from './components/injgrid.js';

await init();
const box = document.getElementById('tour');
const findings = await getJSON('findings.json');
const DEFAULT = { f1: 'timemachine', f2: 'reveal', f3: 'object', f4: 'explorer' };
const steps = Object.keys(DEFAULT).filter(k => findings?.[k]).map(k => ({ key: k, f: findings[k], kind: findings[k].interaction || DEFAULT[k] }));
const PAGE = { timemachine: 'timemachine.html', reveal: 'index.html', object: 'objects.html', explorer: 'explorer.html' };

const MINI = {
  async reveal(el) { const d = await getJSON('reveal.json'); return d?.frames?.length ? mountReveal(el, d, { compact: true, autoplay: true }) : el.replaceChildren(unavailable(t('what.reveal'))); },
  async timemachine(el) {
    const idx = await getJSON('timemachine/index.json');
    const f = idx?.[0] ? await getJSON(`timemachine/${String(idx[0].field).replace(/[^\w.-]/g, '')}.json`) : null;
    return f?.frames?.length ? mountBlink(el, f, { compact: true }) : el.replaceChildren(unavailable(t('what.field')));
  },
  async explorer(el) { const d = await getJSON('injections.json'); return d?.cells?.length ? mountInjections(el, d, { compact: true }) : el.replaceChildren(unavailable(t('what.injections'))); },
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
    title, findingLine(s.f),
    h('div', { class: 'tour-nav' },
      h('button', { type: 'button', class: 'btn', disabled: k === 0, onclick: () => go(cur - 1) }, '← ', t('start.back')),
      h('ol', { class: 'dots', 'aria-label': t('start.dots') }, steps.map((_, j) => h('li', {},
        h('button', { type: 'button', class: 'dot', 'aria-label': t('start.progress', { k: j + 1, n: steps.length }), 'aria-current': j === k ? 'step' : null, onclick: () => go(j) })))),
      k < steps.length - 1
        ? h('button', { type: 'button', class: 'btn primary', onclick: () => go(cur + 1) }, t('start.next'), ' →')
        : h('a', { class: 'btn primary', href: link('timemachine.html') }, t('home.cta'))),
    mini,
    h('p', {}, h('a', { href: link(PAGE[s.kind] || 'index.html') }, t('start.open_full'))),
  );
  if (focus) title.focus();
  const ctl = await MINI[s.kind]?.(mini);
  if (cur === k) active = ctl; else ctl?.stop?.();
}
