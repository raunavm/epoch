// A short guided tour of the home page for first-time visitors: each step outlines one part of the page, scrolls it into
// view and explains it in plain words in a card fixed to the bottom of the screen. Steps whose part is not on the page
// are skipped. Esc or Close ends it.
import { h, t, REDUCED_MOTION } from '../core.js';

// [selector inside the visible viewer (or the page), string key]
const STEPS = [
  ['.intro-fig', 'tour.s_idea'],
  ['@ .viewer-img', 'tour.s_picture'],
  ['@ .viewer-side .controls', 'tour.s_play'],
  ['@ .scrub', 'tour.s_timeline'],
  ['@ .segmented.views', 'tour.s_views'],
  ['@ .flag-counter', 'tour.s_counts'],
  ['@ .legend-row', 'tour.s_markers'],
  ['#coord-form', 'tour.s_any'],
  ['.site-header nav', 'tour.s_more'],
];

export function startTour() {
  const scope = [document.getElementById('live'), document.getElementById('showcase')].find(e => e && !e.hidden) || document;
  const steps = STEPS.map(([sel, key]) => ({ el: sel.startsWith('@ ') ? scope.querySelector(sel.slice(2)) : document.querySelector(sel), key }))
    .filter(s => s.el && s.el.offsetParent !== null);
  if (!steps.length) return;
  let k = 0, lit = null;
  const title = h('h2', { id: 'tour-title', tabindex: '-1' });
  const body = h('p', {});
  const count = h('span', { class: 'muted small' });
  const back = h('button', { type: 'button', class: 'btn small', onclick: () => go(k - 1) }, t('tour.back'));
  const next = h('button', { type: 'button', class: 'btn small primary', onclick: () => (k === steps.length - 1 ? end() : go(k + 1)) });
  const card = h('div', { class: 'tour-card', role: 'dialog', 'aria-labelledby': 'tour-title', 'aria-modal': 'false' },
    h('div', { class: 'tour-top' }, count, h('button', { type: 'button', class: 'btn small', onclick: () => end() }, t('tour.close'))),
    title, body, h('div', { class: 'tour-nav' }, back, next));
  const onKey = e => { if (e.key === 'Escape') end(); };
  function go(i) {
    k = Math.max(0, Math.min(i, steps.length - 1));
    lit?.classList.remove('tour-lit');
    lit = steps[k].el; lit.classList.add('tour-lit');
    lit.scrollIntoView({ block: 'start', behavior: REDUCED_MOTION ? 'auto' : 'smooth' });
    count.textContent = t('tour.count', { i: k + 1, n: steps.length });
    title.textContent = t(`${steps[k].key}_h`); body.textContent = t(steps[k].key);
    back.disabled = k === 0; next.textContent = t(k === steps.length - 1 ? 'tour.done' : 'tour.next');
    title.focus({ preventScroll: true });
  }
  function end() { lit?.classList.remove('tour-lit'); card.remove(); document.removeEventListener('keydown', onKey); }
  document.body.append(card); document.addEventListener('keydown', onKey);
  go(0);
}
