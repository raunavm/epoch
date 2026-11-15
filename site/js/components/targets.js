// Result cards for the five challenge targets, read from data/targets/<id>.json:
// {title, plain_result, numbers:[{label, value, err, unit?, kind, nulls?, method?}], evidence_links:[{text, href}], method_note}
import { h, put, t, link, badge, fmtPM } from '../core.js';
import { evidenceNum, fieldFrames } from './evidence.js';
import { TARGETS } from '../live/targets.js';

const value = n => (Number.isFinite(+n.value) ? fmtPM(n.value, n.err) : String(n.value ?? '—')) + (n.unit ? ` ${n.unit}` : '');

export async function renderTargetCard(el, id, getJSON) {
  const tg = TARGETS.find(x => x.id === id);
  const r = await getJSON(`targets/${id}.json`);
  if (!r) { put(el, h('p', { class: 'eyebrow' }, t('tr.eyebrow')), h('h2', { id: 'h-result' }, tg?.name || id), h('p', { class: 'muted' }, t('tr.pending'))); return; }
  const frames = fieldFrames(getJSON);
  put(el,
    h('p', { class: 'eyebrow' }, t('tr.eyebrow')),
    h('h2', { id: 'h-result' }, r.title || tg?.name || id),
    h('p', { class: 'plain-result' }, r.plain_result || ''),
    r.numbers?.length ? h('dl', { class: 'facts result-numbers' }, r.numbers.flatMap(n => {
      const [btn, panel] = evidenceNum(value(n), {
        loadFrames: () => frames(id), nulls: n.nulls, nullsNote: n.nulls_note,
        method: n.method || r.method_anchor || 'science.html#which-change-is-real', links: r.evidence_links,
      });
      return [h('dt', {}, n.label), h('dd', {}, btn, ' ', badge(n.kind || 'derived'), panel)];
    })) : null,
    r.method_note ? h('p', { class: 'muted small' }, r.method_note) : null,
    r.evidence_links?.length ? h('p', { class: 'small' }, t('tr.evidence'), ' ',
      r.evidence_links.flatMap((l, i) => [i ? ' · ' : '', h('a', { href: /^https?:/.test(l.href) ? l.href : link(l.href) }, l.text)])) : null,
  );
}

export async function renderStrip(el, getJSON, current) {
  const results = await Promise.all(TARGETS.map(tg => getJSON(`targets/${tg.id}.json`)));
  put(el, h('h2', { id: 'h-strip' }, t('tr.strip_h')),
    h('ul', { class: 'strip' }, TARGETS.map((tg, i) => h('li', {},
      h('a', { class: 'strip-card', href: link(`timemachine.html?target=${tg.id}`), 'aria-current': tg.id === current ? 'true' : null },
        h('strong', {}, results[i]?.title || tg.name),
        h('span', { class: 'small' }, results[i]?.plain_result || t('tr.pending')))))));
}
