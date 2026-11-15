// Evidence list: asteroids measured with the pixel model, compared with independent sizes.
import { init, getJSON, h, put, t, link, fmt, badge, safeUrl, unavailable } from './core.js';

await init();
const box = document.getElementById('table'), q = document.getElementById('q'), count = document.getElementById('count');
const rows = await getJSON('objects.json');
const COLS = [
  ['name', 'objects.col_name', null], ['n_frames', 'objects.col_frames', 'data'],
  ['D_km', 'objects.col_D', 'derived'], ['D_truth_km', 'objects.col_Dtruth', 'data'],
  ['ratio_to_independent', 'objects.col_ratio', 'derived'], ['pV', 'objects.col_pV', 'derived'],
  ['class_label', 'objects.col_class', 'data'],
];
let sortKey = 'name', dir = 1;
const srcUrl = r => safeUrl((String(r.D_truth_source || '').match(/https?:\/\/[^\s)]+/) || [])[0]);

if (!rows?.length) { q.disabled = true; box.replaceChildren(unavailable(t('what.objects'))); }
else {
  // The banner is the same on every object page; show it here too.
  const first = await getJSON(`objects/${String(rows[0].id).replace(/[^\w.-]/g, '')}.json`);
  if (first?.banner) box.before(h('p', { class: 'banner', role: 'note' }, first.banner));
  q.addEventListener('input', render); render();
}

function render() {
  const needle = q.value.trim().toLowerCase();
  const list = rows.filter(r => !needle || [r.name, r.number, r.id, r.class_label, r.class_plain].some(v => String(v ?? '').toLowerCase().includes(needle)))
    .sort((a, b) => {
      const x = a[sortKey], y = b[sortKey];
      return (typeof x === 'number' && typeof y === 'number' ? x - y : String(x ?? '').localeCompare(String(y ?? ''))) * dir;
    });
  count.textContent = t('objects.count', { n: list.length, N: rows.length });
  put(box, h('p', { class: 'scroll-cue small muted', 'aria-hidden': 'true' }, t('objects.scroll')), h('table', { class: 'data-table objects-table' },
    h('caption', { class: 'sr-only' }, t('objects.h1')),
    h('thead', {}, h('tr', {}, COLS.map(([k, key, kind]) => h('th', { scope: 'col', 'aria-sort': k === sortKey ? (dir > 0 ? 'ascending' : 'descending') : null },
      h('button', { type: 'button', class: 'sort', onclick: () => { dir = k === sortKey ? -dir : 1; sortKey = k; render(); } }, t(key)),
      kind ? [' ', badge(kind)] : null)))),
    h('tbody', {}, list.map(r => h('tr', {},
      h('th', { scope: 'row' }, h('a', { href: link(r.page || `object.html?id=${encodeURIComponent(r.id)}`) }, r.name)),
      h('td', { class: 'num' }, r.n_frames ?? '—'),
      h('td', { class: 'num' }, r.D_km != null ? `${fmt(r.D_km)} km` : '—'),
      h('td', { class: 'num' }, r.D_truth_km != null ? `${fmt(r.D_truth_km)} km` : '—',
        r.D_truth_source_short ? [h('br'), srcUrl(r) ? h('a', { class: 'small', href: srcUrl(r), rel: 'noopener' }, r.D_truth_source_short) : h('span', { class: 'small muted' }, r.D_truth_source_short)] : null),
      h('td', { class: 'num' }, r.ratio_to_independent != null ? fmt(r.ratio_to_independent, 3) : '—'),
      h('td', { class: 'num' }, fmt(r.pV, 2)),
      h('td', {}, r.class_label ?? r.class_plain ?? '—')))),
  ));
}
