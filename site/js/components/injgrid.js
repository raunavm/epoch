// "Could we have seen it?": planted test objects on a size x distance grid.
import { h, put, t, asset, badge, fmt, viridis } from '../core.js';

// Reflected light only, at opposition: H from size and albedo, V from distances (r from Sun, Δ ≈ r − 1 AU).
export const absMag = (D_km, pV) => 5 * Math.log10(1329 / (D_km * Math.sqrt(pV)));
export const appMag = (D_km, pV, r) => absMag(D_km, pV) + 5 * Math.log10(r * (r - 1));

const near = (a, b) => Math.abs(a - b) <= 1e-6 * Math.max(1, Math.abs(b));

export function mountInjections(el, inj, { compact = false } = {}) {
  const D = [...inj.grid.D_km].sort((a, b) => b - a), R = [...inj.grid.r_au].sort((a, b) => a - b), P = [...inj.grid.pV].sort((a, b) => a - b);
  const cellAt = (d, r, p) => inj.cells.find(c => near(c.D_km, d) && near(c.r_au, r) && near(c.pV, p));
  const st = { p: P[0], d: D[Math.floor(D.length / 2)], r: R[Math.floor(R.length / 2)] };
  const surfName = p => P.length === 1 ? t('inj.surface') : p === P[0] ? t('inj.dark') : p === P[P.length - 1] ? t('inj.bright') : t('inj.medium');

  const surf = h('div', { class: 'segmented', role: 'group', 'aria-label': t('inj.surface_label') },
    P.map(p => h('button', { type: 'button', class: 'seg', 'data-p': p, onclick: () => { st.p = p; render(); } },
      `${surfName(p)} (${t('inj.reflects', { pct: fmt(100 * p, 2) })})`)));
  const capId = `injcap-${Math.random().toString(36).slice(2, 7)}`;
  const table = h('table', { class: 'grid-table', 'aria-describedby': capId });
  const detail = h('div', { class: 'inj-detail', 'aria-live': 'polite' });

  function render() {
    for (const b of surf.children) b.setAttribute('aria-pressed', String(near(+b.dataset.p, st.p)));
    table.replaceChildren(
      h('thead', {}, h('tr', {}, h('th', { scope: 'col' }, t('inj.size_dist')), R.map(r => h('th', { scope: 'col' }, `${fmt(r, 3)} ${t('unit.au')}`)))),
      h('tbody', {}, D.map(d => h('tr', {}, h('th', { scope: 'row' }, sizeText(d)), R.map(r => {
        const c = cellAt(d, r, st.p), sel = near(d, st.d) && near(r, st.r);
        if (!c) return h('td', { class: 'empty' }, '—');
        const f = c.recovered_frac;
        return h('td', {}, h('button', {
          type: 'button', class: 'cell' + (sel ? ' sel' : ''), 'aria-pressed': String(sel),
          style: `--c:${viridis(f)};--fg:${f > 0.55 ? '#000' : '#fff'}`,
          'aria-label': t('inj.cell_aria', { size: sizeText(d), r: fmt(r, 3), pct: fmt(100 * f, 2) }),
          onclick: () => { st.d = d; st.r = r; render(); },
        }, `${fmt(100 * f, 2)}%`));
      })))));
    const c = cellAt(st.d, st.r, st.p);
    if (!c) { put(detail, h('p', { class: 'muted' }, t('inj.pick'))); return; }
    const yes = c.recovered_frac >= 0.5;
    const before = c.example_single_png, after = c.example_stack_png || c.example_png;
    put(detail,
      h('p', { class: 'verdict ' + (yes ? 'yes' : 'no') },
        t(yes ? 'inj.yes' : 'inj.no', { size: sizeText(c.D_km), r: fmt(c.r_au, 3) })),
      h('div', { class: 'pair' },
        before ? h('figure', {}, h('img', { class: 'stamp', loading: 'lazy', src: asset(before), alt: t('inj.alt_before'), width: 160, height: 160 }), h('figcaption', {}, t('inj.before'), ' ', badge('model'))) : null,
        after ? h('figure', {}, h('img', { class: 'stamp', loading: 'lazy', src: asset(after), alt: t('inj.alt_after', { n: c.n_frames }), width: 160, height: 160 }), h('figcaption', {}, t('inj.after', { n: c.n_frames }), ' ', badge('derived'))) : null),
      h('dl', { class: 'facts' },
        h('dt', {}, t('inj.recovered')), h('dd', {}, `${fmt(100 * c.recovered_frac, 2)}% `, badge('derived')),
        h('dt', {}, t('inj.snr')), h('dd', {}, fmt(c.snr_median, 2), ' ', badge('derived')),
        h('dt', {}, t('inj.frames')), h('dd', {}, String(c.n_frames), ' ', badge('data')),
        h('dt', {}, t('inj.v')), h('dd', {}, fmt(c.V, 3), ' ', badge('model'))),
      compact ? null : h('details', { class: 'model' },
        h('summary', {}, t('inj.model_summary'), ' ', badge('model')),
        h('p', {}, t('inj.model_text', { V: fmt(appMag(c.D_km, c.pV, c.r_au), 3) })),
        h('p', { class: 'formula' }, 'H = 5 log₁₀(1329 / (D √pV)),  V = H + 5 log₁₀(r Δ),  Δ ≈ r − 1 AU'),
        h('p', { class: 'muted' }, t('inj.model_caveat'))),
    );
  }
  el.replaceChildren(surf, h('p', { id: capId, class: 'caption' }, t('inj.grid_caption')), h('div', { class: 'table-scroll' }, table), detail);
  render();
}

export function sizeText(km) {
  return km < 1 ? `${fmt(km * 1000, 3)} ${t('unit.m')}` : `${fmt(km, 3)} ${t('unit.km')}`;
}
