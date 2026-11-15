// Result cards for the five challenge targets, read from data/targets/<id>.json:
// {title, plain_result, numbers:[{label, value, err, unit?, kind, nulls?, method?}], evidence_links:[{text, href}], method_note,
//  figures?: [{src (relative to site/), alt, caption, href?}]}
import { h, put, t, link, badge, fmt, fmtPM, loc, locLang } from '../core.js';
import { evidenceNum, fieldFrames } from './evidence.js';
import { mountReveal } from './reveal.js';
import { TARGETS } from '../live/targets.js';

// Strings are shown exactly as written in the file; numbers are formatted with their uncertainty.
const value = n => (typeof n.value !== 'number' ? String(n.value ?? '—') : n.err != null ? fmtPM(n.value, n.err) : fmt(n.value, 6)) + (n.unit ? ` ${n.unit}` : '');

export async function renderTargetCard(el, id, getJSON) {
  const tg = TARGETS.find(x => x.id === id);
  const r = await getJSON(`targets/${id}.json`);
  if (!r) { put(el, h('p', { class: 'eyebrow' }, t('tr.eyebrow')), h('h2', { id: 'h-result' }, tg?.name || id), h('p', { class: 'muted' }, t('tr.pending'))); return; }
  const frames = fieldFrames(getJSON);
  const slider = h('div', { class: 'card-slider' });
  if (r.slider) getJSON(String(r.slider).replace(/[^\w./-]|\.\./g, '')).then(d => {
    if (!d?.frames?.length) return slider.replaceChildren();
    const body = h('div');
    put(slider, h('h3', {}, loc(r, 'slider_caption') || t('tr.slider_caption')), body);
    mountReveal(body, d, { compact: true, autoplay: false, snrNote: loc(d, 'hedge') || null });
  });
  put(el,
    r.status ? h('p', { class: 'eyebrow status' }, t('tr.eyebrow_status', { status: String(loc(r, 'status')).toUpperCase() })) : h('p', { class: 'eyebrow' }, t('tr.eyebrow')),
    h('h2', { id: 'h-result', lang: locLang(r, 'title') }, loc(r, 'title') || tg?.name || id),
    h('p', { class: 'plain-result', lang: locLang(r, 'plain_result') }, loc(r, 'plain_result') || ''),
    r.numbers?.length ? h('dl', { class: 'facts result-numbers' }, r.numbers.flatMap(n => {
      const [btn, panel] = evidenceNum(value(n), {
        loadFrames: () => frames(id), nulls: n.nulls, nullsNote: loc(n, 'nulls_note'),
        method: n.method || r.method_anchor || 'science.html#which-change-is-real', links: r.evidence_links,
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

export async function renderStrip(el, getJSON, current) {
  const results = await Promise.all(TARGETS.map(tg => getJSON(`targets/${tg.id}.json`)));
  put(el, h('h2', { id: 'h-strip' }, t('tr.strip_h')),
    h('ul', { class: 'strip' }, TARGETS.map((tg, i) => h('li', {},
      h('a', { class: 'strip-card', href: link(`timemachine.html?target=${tg.id}`), 'aria-current': tg.id === current ? 'true' : null },
        h('strong', {}, loc(results[i], 'title') || tg.name),
        h('span', { class: 'small' }, results[i]?.status ? t('tr.status_short', { status: loc(results[i], 'status') })
          : firstSentence(loc(results[i], 'plain_result')) || t('tr.pending')))))));
}

const firstSentence = txt => (txt ? (txt.match(/^.*?[.!?](\s|$)/) || [txt])[0].trim() : '');
