import { init, getJSON, h, put, t, link, asset, safeUrl, badge, fmt, fmtPM, wlColor, unavailable } from './core.js';
import { plot } from './plot.js';
import { sizeText } from './components/injgrid.js';

await init();
const root = document.getElementById('obj');
const id = (new URLSearchParams(location.search).get('id') || '').replace(/[^\w.-]/g, '');
const o = id ? await getJSON(`objects/${id}.json`) : null;
if (!o) root.replaceChildren(h('h1', {}, t('title.object')), unavailable(t('what.object')));
else render(o);

function render(o) {
  const L = (k, d) => (o.labels?.[k] || d).toLowerCase();
  const frames = o.frames || [], sz = o.size || {}, tr = sz.truth || {};
  document.title = `${o.name} — ${t('site.name')}`;
  let uid = 0;

  // "Check it": a button that reveals the independent value or the pixels behind a number.
  function checkIt(...content) {
    const rid = `chk-${uid++}`;
    const region = h('span', { id: rid, class: 'check-region', hidden: true }, ...content);
    const btn = h('button', { type: 'button', class: 'check', 'aria-expanded': 'false', 'aria-controls': rid, onclick: () => {
      const open = region.hidden; region.hidden = !open; btn.setAttribute('aria-expanded', String(open));
    } }, t('obj.check'));
    return [btn, region];
  }
  const Dtxt = sz.D_lo_km != null && sz.D_hi_km != null ? `${fmt(sz.D_km)} km (${fmt(sz.D_lo_km)}–${fmt(sz.D_hi_km)}${sz.interval_label ? `; ${sz.interval_label}` : ''})`
    : sz.D_err_km != null ? `${fmtPM(sz.D_km, sz.D_err_km)} km` : `${fmt(sz.D_km)} km`;
  const Ttxt = tr.D_km != null ? `${fmtPM(tr.D_km, tr.D_sigma_km)} km` : null;
  const shortSrc = src => /neowise/i.test(src || '') ? 'NEOWISE' : (src || '—');
  const toFrames = () => h('a', { href: '#frames', onclick: () => { document.getElementById('frames').open = true; } }, t('obj.see_pixels', { n: frames.length }));
  const numRow = (label, value, kind, ...check) => h('div', { class: 'num-row' }, h('dt', {}, label), h('dd', {}, h('span', { class: 'big-num' }, value), ' ', badge(kind), ' ', ...(check.length ? checkIt(...check) : [])));

  // Size beside a familiar object
  const sizeSec = h('section', { 'aria-labelledby': 'h-size' }, h('h2', { id: 'h-size' }, t('obj.size_h')));
  if (sz.D_km != null) {
    const dl = h('dl', { class: 'facts' },
      numRow(t('obj.diameter'), Dtxt, L('D_km', 'derived'),
        tr.D_km != null ? [t('obj.independent'), ' ', h('strong', {}, Ttxt), ` (${tr.source || '—'}) `, badge('data'),
          ` · ${t('obj.ratio', { r: fmt(sz.D_km / tr.D_km, 3) })} `, badge('derived'), ' · '] : [t('obj.no_independent'), ' · '], toFrames()),
      sz.pV != null ? numRow(t('obj.albedo'), fmt(sz.pV, 2), L('pV', 'derived'),
        tr.pV != null ? [t('obj.independent'), ' ', h('strong', {}, fmt(tr.pV, 2)), ` (${tr.source || '—'}) `, badge('data')] : t('obj.no_independent')) : null,
      sz.eta != null ? numRow(t('obj.eta'), fmt(sz.eta, 2), L('eta', 'model'), t('obj.eta_check')) : null);
    sizeSec.append(dl);
  } else sizeSec.append(unavailable(t('what.size')));

  // First screen: our size beside the independent one, and a familiar object
  const sizeTop = sz.D_km != null ? h('div', { class: 'size-top' },
    h('div', { class: 'tiles' },
      h('div', { class: 'tile' }, h('span', { class: 'tile-label' }, t('obj.ours_size')), h('span', { class: 'big-num' }, Dtxt), badge(L('D_km', 'derived'))),
      h('div', { class: 'tile' }, h('span', { class: 'tile-label' }, t('obj.indep_size', { src: shortSrc(tr.source) })),
        h('span', { class: 'big-num', title: tr.source || '' }, Ttxt ?? '—'), Ttxt ? badge('data') : null)),
    sz.familiar?.km > 0 ? compare(sz.D_km, sz.familiar) : null) : null;

  // Spectrum with Gaia overlay behind "Check it"
  const sp = o.spectrum || {}, specSec = h('section', { 'aria-labelledby': 'h-spec' }, h('h2', { id: 'h-spec' }, t('obj.spec_h')));
  if (sp.wl_um?.length) {
    const g = sp.gaia;
    const draw = withGaia => plot([
      { x: sp.wl_um, y: sp.refl, err: sp.refl_err, kind: 'points', color: 'var(--accent)', colors: sp.wl_um.map(wlColor), label: t('obj.ours') },
      ...(withGaia && g?.wl_um?.length ? [{ x: g.wl_um, y: g.refl.map(r => r * (g.scale_to_ours ?? 1)), err: g.err?.map(e => e * (g.scale_to_ours ?? 1)), kind: 'line', color: '#E69F00', label: g.scale_to_ours ? `${t('obj.gaia')} × ${fmt(g.scale_to_ours, 3)}` : t('obj.gaia') }] : []),
    ], { xlabel: t('plot.wl'), ylabel: t('plot.refl'), title: t('obj.spec_title', { name: o.name }) });
    const holder = h('div', {}, draw(!!g?.wl_um?.length));
    const gbtn = h('button', { type: 'button', class: 'check', 'aria-pressed': String(!!g?.wl_um?.length), disabled: !g?.wl_um?.length, onclick: () => {
      const on = gbtn.getAttribute('aria-pressed') !== 'true'; gbtn.setAttribute('aria-pressed', String(on)); holder.replaceChildren(draw(on));
    } }, g?.wl_um?.length ? t('obj.check_gaia') : t('obj.no_gaia'));
    specSec.append(h('p', {}, t('obj.spec_lede'), ' ', badge(L('refl', 'derived')), ' ', gbtn, ' ', ...soundButton(sp)), holder,
      h('p', { class: 'muted' }, t('obj.spec_note', { n: sp.wl_um.length }), ' ', toFrames()));
  } else specSec.append(unavailable(t('what.spectrum')));

  put(root,
    o.banner ? h('p', { class: 'banner', role: 'note' }, o.banner) : null,
    h('div', { class: 'obj-hero' },
      h('div', {}, h('h1', {}, o.name), h('p', { class: 'lede' }, o.class_plain || o.kind || ''), sizeTop),
      specSec),
    sizeSec,
    o.composition_plain ? h('section', { 'aria-labelledby': 'h-comp' }, h('h2', { id: 'h-comp' }, t('obj.comp_h')),
      h('p', {}, o.composition_plain, ' ', badge(L('composition', 'derived')))) : null,
    h('section', { 'aria-labelledby': 'h-path' }, h('h2', { id: 'h-path' }, t('obj.path_h')), skyChart(o.track || [], frames)),
    framesSection(frames, L),
    h('section', { 'aria-labelledby': 'h-prov' }, h('h2', { id: 'h-prov' }, t('obj.prov_h')),
      h('dl', { class: 'facts' },
        h('dt', {}, t('obj.release')), h('dd', {}, o.release ?? '—'),
        h('dt', {}, t('obj.code_version')), h('dd', {}, h('code', {}, o.code_version ?? '—')),
        h('dt', {}, t('obj.rerun')), h('dd', {}, h('a', { href: link('science.html#reproduce-every-figure') }, t('obj.reproduce_link'))))),
  );
  if (new URLSearchParams(location.search).get('check') === 'all') { // open every "Check it" panel (for screenshots)
    for (const b of root.querySelectorAll('button.check[aria-expanded="false"]')) b.click();
    document.getElementById('frames').open = true;
  }
}

function compare(D, fam) {
  const W = 600, maxKm = Math.max(D, fam.km), bar = v => Math.max(2, v / maxKm * (W - 20));
  const r = D / fam.km;
  const words = r > 0.8 && r < 1.25 ? t('obj.fam_same', { name: fam.name }) : t('obj.fam_times', { x: fmt(r, 2), name: fam.name });
  const NS = 'http://www.w3.org/2000/svg', svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('viewBox', `0 0 ${W} 70`); svg.setAttribute('class', 'compare'); svg.setAttribute('role', 'img'); svg.setAttribute('aria-label', words);
  svg.innerHTML = `<rect x="10" y="8" height="20" rx="3" fill="var(--accent)"/><rect x="10" y="42" height="20" rx="3" fill="#E69F00"/>`;
  svg.children[0].setAttribute('width', bar(D)); svg.children[1].setAttribute('width', bar(fam.km));
  return h('figure', { class: 'compare-fig' }, svg,
    h('figcaption', {}, h('strong', {}, words), ' ', badge('derived'), h('br'),
      h('span', { class: 'key-bar', style: '--c:var(--accent)' }), ` ${t('obj.this_object')} · `,
      h('span', { class: 'key-bar', style: '--c:#E69F00' }), ` ${/\d/.test(fam.name) ? fam.name : `${fam.name} (${sizeText(fam.km)})`}`));
}

function skyChart(track, frames) {
  if (track.length < 2) return unavailable(t('what.track'));
  const ra0 = track[0].ra, unwrap = ra => ra0 + ((ra - ra0 + 540) % 360) - 180;
  const tr = track.map(p => ({ mjd: p.mjd, ra: unwrap(p.ra), dec: p.dec })).sort((a, b) => a.mjd - b.mjd);
  const at = mjd => { // linear interpolation along the track
    const j = tr.findIndex(p => p.mjd >= mjd);
    if (j <= 0) return tr[j < 0 ? tr.length - 1 : 0];
    const a = tr[j - 1], b = tr[j], u = (mjd - a.mjd) / (b.mjd - a.mjd || 1);
    return { ra: a.ra + u * (b.ra - a.ra), dec: a.dec + u * (b.dec - a.dec) };
  };
  const fp = frames.map(f => ({ ...at(f.mjd), wl: f.wl_um }));
  const svg = plot([
    { x: tr.map(p => p.ra), y: tr.map(p => p.dec), kind: 'line', color: 'var(--muted)', label: t('obj.orbit_path') },
    { x: fp.map(p => p.ra), y: fp.map(p => p.dec), kind: 'points', color: 'var(--accent)', colors: fp.map(p => wlColor(p.wl)), label: t('obj.frame_points') },
  ], { invertX: true, xlabel: t('plot.ra'), ylabel: t('plot.dec'), title: t('obj.path_title'), height: 320 });
  return h('figure', {}, svg, h('figcaption', { class: 'muted' }, t('obj.path_caption'), ' ', badge('model')));
}

function framesSection(frames, L) {
  const det = h('details', { id: 'frames', class: 'frames' }, h('summary', {}, h('h2', { id: 'h-frames' }, t('obj.frames_h', { n: frames.length }))));
  if (!frames.length) { det.append(unavailable(t('what.frames'))); return det; }
  const th = (key, kind) => h('th', { scope: 'col' }, t(key), kind ? [' ', badge(kind)] : null);
  const thumb = (src, alt) => src ? h('img', { src: asset(src), alt, loading: 'lazy', width: 48, height: 48, class: 'thumb' }) : '—';
  det.append(h('div', { class: 'table-scroll' }, h('table', { class: 'data-table' },
    h('thead', {}, h('tr', {}, th('obj.col_obs'), th('obj.col_mjd', 'data'), th('obj.col_det', 'data'), th('obj.col_wl', 'data'),
      th('obj.col_flux', L('flux_mJy', 'derived')), th('obj.col_chi2', 'derived'),
      th('obj.col_stamp', 'data'), th('obj.col_model', 'model'), th('obj.col_resid', 'resid'), th('obj.col_s3'))),
    h('tbody', {}, frames.map(f => h('tr', {},
      h('th', { scope: 'row' }, h('code', {}, f.obs_id)),
      h('td', { class: 'num' }, fmt(f.mjd, 8)), h('td', { class: 'num' }, f.detector ?? '—'),
      h('td', { class: 'num' }, h('span', { class: 'swatch', style: `background:${wlColor(f.wl_um)}`, 'aria-hidden': 'true' }), `${fmt(f.wl_um, 4)} ± ${fmt(f.bw_um / 2, 2)}`),
      h('td', { class: 'num' }, fmtPM(f.flux_mJy, f.flux_err_mJy)), h('td', { class: 'num' }, fmt(f.chi2_dof_local, 3)),
      h('td', {}, thumb(f.stamp_png, t('obj.alt_stamp', { id: f.obs_id }))),
      h('td', {}, thumb(f.model_png, t('obj.alt_model', { id: f.obs_id }))),
      h('td', {}, thumb(f.resid_png, t('obj.alt_resid', { id: f.obs_id }))),
      h('td', {}, safeUrl(f.s3_url) ? h('a', { href: f.s3_url, rel: 'noopener' }, t('obj.s3')) : '—')))))));
  return det;
}

// Spectrum as sound: wavelength → time (3 s), reflectance → pitch (two octaves from 220 Hz).
function soundButton(sp) {
  const pts = sp.wl_um.map((w, i) => [w, sp.refl[i]]).filter(p => p.every(Number.isFinite)).sort((a, b) => a[0] - b[0]);
  const b = h('button', { type: 'button', class: 'btn small', 'aria-describedby': 'sound-help' }, t('obj.sound'));
  const help = h('span', { id: 'sound-help', class: 'sr-only' }, t('obj.sound_help'));
  let ctx = null;
  b.addEventListener('click', () => {
    if (pts.length < 2 || !(window.AudioContext || window.webkitAudioContext)) { b.textContent = t('obj.sound_na'); return; }
    ctx ??= new (window.AudioContext || window.webkitAudioContext)();
    const w0 = pts[0][0], w1 = pts[pts.length - 1][0], ys = pts.map(p => p[1]), lo = Math.min(...ys), hi = Math.max(...ys);
    const T = 3, now = ctx.currentTime + 0.05, osc = ctx.createOscillator(), gain = ctx.createGain();
    osc.type = 'sine';
    pts.forEach(([w, y]) => osc.frequency.linearRampToValueAtTime(220 * 2 ** (2 * (y - lo) / (hi - lo || 1)), now + (w - w0) / (w1 - w0) * T));
    osc.frequency.setValueAtTime(220 * 2 ** (2 * (ys[0] - lo) / (hi - lo || 1)), now);
    gain.gain.setValueAtTime(0, now); gain.gain.linearRampToValueAtTime(0.2, now + 0.1);
    gain.gain.setValueAtTime(0.2, now + T - 0.15); gain.gain.linearRampToValueAtTime(0, now + T);
    osc.connect(gain).connect(ctx.destination); osc.start(now); osc.stop(now + T + 0.05);
  });
  return [b, help];
}
