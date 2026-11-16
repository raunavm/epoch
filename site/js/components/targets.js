// Result cards for the five challenge targets, read from data/targets/<id>.json:
// {title, plain_result, numbers:[{label, value, err, unit?, kind, nulls?, method?}], evidence_links:[{text, href}], method_note,
//  figures?: [{src (relative to site/), alt, caption, href?}]}
import { h, put, t, link, badge, fmt, fmtPM, loc, locLang } from '../core.js';
import { evidenceNum, fieldFrames } from './evidence.js';
import { mountReveal } from './reveal.js';
import { TARGETS } from '../live/targets.js';

// Strings are shown exactly as written in the file; numbers are formatted with their uncertainty.
export const value = n => (typeof n.value !== 'number' ? String(loc(n, 'value') ?? '—') : n.err != null ? fmtPM(n.value, n.err) : fmt(n.value, 6)) + (n.unit ? ` ${n.unit}` : '');

export async function renderTargetCard(el, id, getJSON) {
  const tg = TARGETS.find(x => x.id === id);
  const r = await getJSON(`targets/${id}.json`);
  if (!r) { put(el, h('h2', { id: 'h-result' }, t('tr.h', { name: tg?.name || id })), h('p', { class: 'muted' }, t('tr.pending'))); return; }
  const frames = fieldFrames(getJSON);
  const slider = h('div', { class: 'card-slider' });
  if (r.slider) getJSON(String(r.slider).replace(/[^\w./-]|\.\./g, '')).then(d => {
    if (!d?.frames?.length) return slider.replaceChildren();
    const body = h('div');
    put(slider, h('h3', {}, loc(r, 'slider_caption') || t('tr.slider_caption')), body);
    mountReveal(body, d, { compact: true, autoplay: false, snrNote: loc(d, 'hedge') || null });
  });
  put(el,
    h('h2', { id: 'h-result' }, t('tr.h', { name: tg?.name || id })),
    r.status ? h('p', {}, t('tr.status_short', { status: loc(r, 'status') })) : null,
    h('p', { class: 'plain-result', lang: locLang(r, 'plain_result') }, loc(r, 'plain_result') || ''),
    r.numbers?.length ? h('dl', { class: 'facts result-numbers' }, r.numbers.flatMap(n => {
      const [btn, panel] = evidenceNum(value(n), {
        loadFrames: () => frames(id), nulls: n.nulls, nullsNote: loc(n, 'nulls_note'),
        method: n.method || r.method_anchor || 'results.html#which-change-is-real', links: r.evidence_links,
      });
      return [h('dt', { lang: locLang(n, 'label') }, loc(n, 'label')), h('dd', {}, btn, ' ', badge(n.kind || 'derived'), panel)];
    })) : null,
    (r.figures || []).map(f => { // optional figures, after the numbers
      const img = h('img', { src: f.src, alt: loc(f, 'alt') || '', loading: 'lazy' });
      return h('figure', { class: 'card-fig' }, f.href ? h('a', { href: /^https?:/.test(f.href) ? f.href : link(f.href) }, img) : img,
        f.caption ? h('figcaption', { class: 'small', lang: locLang(f, 'caption') }, loc(f, 'caption')) : null);
    }),
    r.slider ? slider : null,
    r.method_note ? h('details', { class: 'method-note' }, h('summary', {}, t('tr.method_h')), h('p', { class: 'small', lang: locLang(r, 'method_note') }, loc(r, 'method_note'))) : null,
    r.evidence_links?.length ? h('p', { class: 'small' }, t('tr.evidence'), ' ',
      r.evidence_links.flatMap((l, i) => [i ? ' · ' : '', h('a', { href: /^https?:/.test(l.href) ? l.href : link(l.href), lang: locLang(l, 'text') }, loc(l, 'text'))])) : null,
  );
}
