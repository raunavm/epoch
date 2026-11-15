// Results at a glance: one row per challenge target and per main result. Rows, verdicts and the numbers that are on
// no card come from results.json (scripts/build_results.py); titles, numbers and figures from targets/<id>.json and
// findings.json. No number is written in this file.
import { init, getJSON, h, put, t, fmt, loc, locLang, link, badge, unavailable } from './core.js';
import { value } from './components/targets.js';

await init();
const box = document.getElementById('results');
const res = await getJSON('results.json');
if (!res?.rows?.length) put(box, unavailable(t('results.what')));
else {
  const findings = (await getJSON('findings.json')) || {};
  const targets = Object.fromEntries(await Promise.all(res.rows.filter(r => r.target).map(async r => [r.target, await getJSON(`targets/${r.target}.json`)])));
  const COLS = ['results.col_what', 'results.col_measured', 'results.col_verdict', 'results.col_links'];
  const groups = [['target', 'results.group_targets'], ['other', 'results.group_other']];
  put(box,
    h('p', { class: 'small muted' }, t('results.legend')),
    h('table', { class: 'results-table', role: 'table' },
      h('caption', { class: 'sr-only' }, t('results.caption')),
      h('thead', { role: 'rowgroup' }, h('tr', { role: 'row' }, COLS.map(k => h('th', { scope: 'col', role: 'columnheader' }, t(k))))),
      groups.map(([g, key]) => h('tbody', { role: 'rowgroup' },
        h('tr', { class: 'group', role: 'row' }, h('th', { colspan: 4, scope: 'colgroup', role: 'columnheader' }, t(key))),
        res.rows.filter(r => (g === 'target' ? r.group === 'target' : r.group !== 'target')).map(r => row(r, targets[r.target], findings[r.finding]))))));

  function row(r, tg, fd) {
    const title = tg ? loc(tg, 'title') : fd ? loc(fd, 'title') : t(`results.t.${r.id}`);
    const fig = tg?.figures?.[0] ? { src: tg.figures[0].src, alt: loc(tg.figures[0], 'alt') } : r.figure ? { src: r.figure, alt: title } : null;
    const nums = [
      ...(r.numbers || []).map(i => tg?.numbers?.[i]).filter(Boolean).map(n => h('li', {}, h('span', { lang: locLang(n, 'label') }, loc(n, 'label')), ': ', h('strong', {}, value(n)), ' ', badge(n.kind || 'derived'))),
      fd ? h('li', {}, t(`results.m.${r.id}`), ': ', h('strong', {}, findingValue(fd)), ' ', badge(fd.label || 'derived')) : null,
      ...(r.extra || []).map(x => h('li', {}, t(`results.m.${x.key}`, { v: fmt(x.value, 3), rule: fmt(x.rule, 3), text: x.text, hits: x.hits, n: x.n, lo: fmt(x.lo, 2), hi: fmt(x.hi, 2) }), ' ', badge('derived'))),
    ].filter(Boolean);
    const td = (k, ...kids) => h('td', { role: 'cell', 'data-label': t(k) }, ...kids);
    return h('tr', { role: 'row' },
      h('th', { scope: 'row', role: 'rowheader' }, h('span', { lang: tg ? locLang(tg, 'title') : fd ? locLang(fd, 'title') : null }, title),
        fig ? h('a', { class: 'res-thumb', href: fig.src }, h('img', { src: fig.src, alt: fig.alt || '', loading: 'lazy', width: 240 })) : null),
      td('results.col_measured', h('ul', { class: 'res-nums' }, nums)),
      td('results.col_verdict', r.claims.map(c => h('p', { class: 'res-verdict' },
        h('span', { class: `verdict v-${c.verdict.replace(/\s+/g, '-').toLowerCase()}`, title: c.status }, t(`results.v.${c.verdict.replace(/\s+/g, '_')}`)),
        c.status.length > c.verdict.length ? h('span', { class: 'small', lang: 'en' }, ' ' + c.status.slice(c.verdict.length).trim()) : null,
        h('span', { class: 'small muted' }, ` · ${t('results.claim', { id: c.id })}`),
        c.late ? h('span', { class: 'late small' }, t('results.late')) : null))),
      td('results.col_links', h('ul', { class: 'res-links' },
        r.card ? h('li', {}, h('a', { href: link(r.card) }, t(r.target ? 'results.link_card' : 'results.link_tour'))) : null,
        fig ? h('li', {}, h('a', { href: fig.src }, t('results.link_figure'))) : null,
        h('li', {}, h('a', { href: link(r.science) }, t('results.link_science'))),
        r.failed.map(f => h('li', {}, h('a', { href: link(f.href) }, t('results.link_failed', { n: f.n })))))));
  }
}

// A finding's headline number: value, its error or interval, and unit, as written in findings.json.
function findingValue(f) {
  const unit = loc(f, 'unit') || '';
  const v = `${fmt(f.number, 4)}${unit === '%' ? '%' : unit ? ` ${unit}` : ''}`;
  if (f.err != null) return `${v} ± ${fmt(f.err, 3)}`;
  return f.err_low != null && f.err_high != null ? `${v} (${t('results.interval', { lo: fmt(f.err_low, 4), hi: fmt(f.err_high, 4) })})` : v;
}
