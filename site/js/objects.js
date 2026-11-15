import { init, getJSON, h, t, link, fmt, fmtPM, badge, unavailable } from './core.js';

await init();
const box = document.getElementById('table'), q = document.getElementById('q'), count = document.getElementById('count');
const rows = await getJSON('objects.json');
const COLS = [
  ['name', 'objects.col_name', null], ['kind', 'objects.col_kind', null], ['n_frames', 'objects.col_frames', 'data'],
  ['D_km', 'objects.col_D', 'derived'], ['D_truth_km', 'objects.col_Dtruth', 'data'], ['pV', 'objects.col_pV', 'derived'],
  ['class_plain', 'objects.col_class', 'derived'],
];
let sortKey = 'name', dir = 1;

if (!rows?.length) { q.disabled = true; box.replaceChildren(unavailable(t('what.objects'))); }
else { q.addEventListener('input', render); render(); }

function render() {
  const needle = q.value.trim().toLowerCase();
  const list = rows.filter(r => !needle || [r.name, r.number, r.id, r.kind, r.class_plain].some(v => String(v ?? '').toLowerCase().includes(needle)))
    .sort((a, b) => {
      const x = a[sortKey], y = b[sortKey];
      return (typeof x === 'number' && typeof y === 'number' ? x - y : String(x ?? '').localeCompare(String(y ?? ''))) * dir;
    });
  count.textContent = t('objects.count', { n: list.length, N: rows.length });
  box.replaceChildren(h('table', { class: 'data-table' },
    h('caption', { class: 'sr-only' }, t('objects.h1')),
    h('thead', {}, h('tr', {}, COLS.map(([k, key, kind]) => h('th', { scope: 'col', 'aria-sort': k === sortKey ? (dir > 0 ? 'ascending' : 'descending') : null },
      h('button', { type: 'button', class: 'sort', onclick: () => { dir = k === sortKey ? -dir : 1; sortKey = k; render(); } }, t(key)),
      kind ? badge(kind) : null)))),
    h('tbody', {}, list.map(r => h('tr', {},
      h('th', { scope: 'row' }, h('a', { href: link(r.page || `object.html?id=${encodeURIComponent(r.id)}`) }, r.name), r.number != null ? h('span', { class: 'muted' }, ` (${r.number})`) : null),
      h('td', {}, r.kind ?? '—'),
      h('td', { class: 'num' }, r.n_frames ?? '—'),
      h('td', { class: 'num' }, r.D_km != null ? `${fmtPM(r.D_km, r.D_err_km)} km` : '—'),
      h('td', { class: 'num' }, r.D_truth_km != null ? `${fmt(r.D_truth_km)} km` : '—', r.D_truth_source ? h('span', { class: 'muted small' }, ` ${r.D_truth_source}`) : null),
      h('td', { class: 'num' }, fmt(r.pV, 2)),
      h('td', {}, r.class_plain ?? '—')))),
  ));
}
