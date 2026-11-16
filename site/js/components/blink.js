// A prepared field in three views: plain blink (every picture, in time order), one detector at a time,
// and the colour-aware view (experimental); with labelled objects, pixel-flag counts and an optional Gaia overlay.
import { h, put, t, link, loc, asset, badge, fmt, wlColor, CATS, REDUCED_MOTION } from '../core.js';
import { contrastControl, filmButton, fromURL } from './film.js';
import { s } from '../plot.js';

const SPEEDS = [0.5, 1, 2, 4];
const META_KEYS = ['PSF_FWHM', 'L2DQAFLG', 'L2 N_OUTLIER', 'L2 N_PERSIST', 'L2 SATSTRKS', 'L2 N_GHOST'];
export const VIEWS = ['all', 'det', 'corr'];

// Per-detector picture count and wavelength range.
export function detectorInfo(frames, key = 'detector', wl = 'wl_um') {
  const info = {};
  for (const f of frames) {
    const d = f[key]; if (d == null) continue;
    const r = info[d] ??= { n: 0, lo: Infinity, hi: -Infinity };
    r.n++; r.lo = Math.min(r.lo, f[wl]); r.hi = Math.max(r.hi, f[wl]);
  }
  return info;
}
export const busiestDetector = info => +Object.entries(info).sort((a, b) => b[1].n - a[1].n || a[0] - b[0])[0]?.[0];

// Segmented control of the three views plus the detector picker; onChange(view, det).
// update(view, det, corrUnavailable?, newInfo?) refreshes it (live mode passes new info as pictures arrive).
export function viewControls(info, st, onChange) {
  const views = VIEWS.map(k => h('button', { type: 'button', class: 'seg', onclick: () => onChange(k, st.det) }, t(`view.${k}`)));
  const dets = [1, 2, 3, 4, 5, 6].map(d => h('button', { type: 'button', class: 'seg', onclick: () => onChange('det', d) }, String(d)));
  const range = h('span', { class: 'small det-range' });
  const detRow = h('div', { class: 'det-row' }, h('p', { class: 'small det-what' }, t('view.det_what')), h('span', { class: 'small muted' }, t('view.det_pick')), ' ',
    h('div', { class: 'segmented small-seg', role: 'group', 'aria-label': t('view.det_pick') }, dets), ' ', range);
  const explain = h('p', { class: 'hint' });
  const root = h('div', {}, h('div', { class: 'segmented big views', role: 'group', 'aria-label': t('view.label') }, views), detRow, explain);
  root.update = (view, det, corrUnavailable = false, newInfo = null) => {
    if (newInfo) info = newInfo;
    const hasDet = Object.keys(info).length > 0;
    views.forEach((b, k) => {
      b.setAttribute('aria-pressed', String(VIEWS[k] === view));
      b.disabled = (VIEWS[k] === 'det' && !hasDet) || (VIEWS[k] === 'corr' && corrUnavailable);
    });
    detRow.hidden = view !== 'det';
    dets.forEach((b, k) => {
      const d = k + 1, r = info[d];
      b.disabled = !r?.n;
      b.setAttribute('aria-pressed', String(d === det));
      b.setAttribute('aria-label', r?.n ? t('view.det_aria', { d, lo: fmt(r.lo, 2), hi: fmt(r.hi, 2), n: r.n }) : `${d}`);
      b.title = r?.n ? `${fmt(r.lo, 2)}–${fmt(r.hi, 2)} µm` : '';
    });
    const r = info[det];
    range.textContent = r ? t('view.det_range', { lo: fmt(r.lo, 2), hi: fmt(r.hi, 2), n: r.n }) : '';
    explain.textContent = corrUnavailable && view === 'corr' ? t('live.corrected_wait') : t(`view.${view}_explain`, { d: det ?? '' });
  };
  return root;
}

// compact: small version (tour). minimal: views + picture only (landing).
export function mountBlink(el, field, { compact = false, minimal = false, view = 'det', det = null } = {}) {
  const frames = [...(field.frames || [])].sort((a, b) => a.mjd - b.mjd);
  if (!frames.length) return null;
  const size = field.size_px || 64;
  const info = detectorInfo(frames);
  const st = { view: VIEWS.includes(view) ? view : 'all', det: info[det]?.n ? det : busiestDetector(info), i: 0, playing: !REDUCED_MOTION, fps: 2, layout: 'blink', a: 0, b: Math.min(1, frames.length - 1), gaia: false };
  let timer = null;
  const uid = Math.random().toString(36).slice(2, 7);
  const small = compact || minimal;
  const list = () => (st.view === 'det' ? frames.filter(f => f.detector === st.det) : frames);
  const corr = () => st.view === 'corr';

  const controls = viewControls(info, st, (v, d) => {
    const cur = list()[st.i];
    st.view = v; st.det = d;
    const L = list(); // keep the same moment in time when switching
    st.i = cur ? L.reduce((best, f, k) => (Math.abs(f.mjd - cur.mjd) < Math.abs(L[best].mjd - cur.mjd) ? k : best), 0) : 0;
    render();
  });

  const viewA = view_(), viewB = view_();
  const playBtn = h('button', { type: 'button', class: 'btn', onclick: () => { st.playing = !st.playing; schedule(); render(); } });
  const prev = h('button', { type: 'button', class: 'btn', 'aria-label': t('tm.prev'), onclick: () => step(-1) }, '←');
  const next = h('button', { type: 'button', class: 'btn', 'aria-label': t('tm.next'), onclick: () => step(1) }, '→');
  const speed = h('select', { id: `sp-${uid}`, onchange: e => { st.fps = +e.target.value; schedule(); } },
    SPEEDS.map(v => h('option', { value: v, selected: v === st.fps }, t('tm.fps', { v }))));
  const scrub = h('input', { type: 'range', class: 'scrub', min: 0, max: frames.length - 1, value: 0, id: `sc-${uid}`, oninput: e => { st.i = +e.target.value; st.playing = false; schedule(); render(); } });
  const ends = h('div', { class: 'scrub-ends muted' });
  const labelList = h('ul', { class: 'labels' });
  const counter = h('p', { class: 'flag-counter', 'aria-live': 'off' });
  const COUNT_KEY = { all: 'n_changes_naive_all', det: 'n_changes_naive_det', corr: 'n_changes_corr' };
  // The two comparable totals side by side (the colour-aware count uses another noise model and is not compared).
  const sum = k => frames.reduce((a, f) => a + (f[k] || 0), 0);
  const compare = frames.some(f => f.n_changes_naive_all != null) && frames.some(f => f.n_changes_naive_det != null)
    ? h('p', { class: 'small muted' }, t('view.compare', { N: frames.length, all: sum(COUNT_KEY.all), det: sum(COUNT_KEY.det) })) : null;
  const astList = h('p', { class: 'small asteroid-list' });
  const flagBox = h('dl', { class: 'facts small' });
  const hasGaia = Array.isArray(field.gaia) && field.gaia.length > 0;
  const gaiaBtn = h('button', { type: 'button', class: 'check', 'aria-pressed': 'false', disabled: !hasGaia, onclick: () => { st.gaia = !st.gaia; render(); } },
    t('tm.gaia', { n: field.gaia?.length || 0 }));
  // No prepared star list: the Gaia overlay is drawn in the live view of the same patch.
  const gaiaLive = !hasGaia && field.ra != null && field.dec != null
    ? h('a', { href: link(`index.html?ra=${field.ra}&dec=${field.dec}&fov=2&gaia=1#live`) }, t('tm.gaia_live_link')) : null;

  const layoutBtns = ['blink', 'side'].map(k => h('button', { type: 'button', class: 'seg', onclick: () => { st.layout = k; st.playing = k === 'blink' && st.playing; schedule(); render(); } }, t(`tm.mode_${k}`)));
  const selA = frameSelect('a'), selB = frameSelect('b');
  let lastList = null;

  function frameSelect(which) {
    return h('select', { 'aria-label': t(which === 'a' ? 'tm.frame_a' : 'tm.frame_b'), onchange: e => { st[which] = +e.target.value; render(); } },
      frames.map((f, k) => h('option', { value: k }, `${f.date} · ${fmt(f.wl_um, 3)} µm`)));
  }

  function view_() {
    const img = h('img', { class: 'stamp', width: 320, height: 320, decoding: 'async', alt: '' });
    const ov = s('svg', { viewBox: `0 0 ${size} ${size}`, class: 'overlay', 'aria-hidden': 'true' });
    const meta = h('p', { class: 'frame-meta' });
    const tip = h('p', { class: 'marker-tip small', 'aria-live': 'polite' });
    ov.addEventListener('click', e => { // tap a marker: pause and name it
      const g = e.target.closest('.mk'); tip.textContent = g ? g.dataset.tip : '';
      if (g && st.playing) { st.playing = false; schedule(); render(); }
    });
    return { img, ov, meta, tip, root: h('figure', { class: 'tm-view' }, h('div', { class: 'imgwrap' }, img, ov), meta, tip) };
  }

  function fill(v, f, k, n) {
    const src = corr() ? f.corr_png : f.img_png;
    v.img.src = src ? asset(src) : '';
    v.img.alt = t(corr() ? 'tm.alt_corr' : 'tm.alt_raw', { name: loc(field, 'name'), date: f.date, wl: fmt(f.wl_um, 3) });
    v.meta.replaceChildren(
      h('span', { class: 'swatch', style: `background:${wlColor(f.wl_um)}`, 'aria-hidden': 'true' }),
      h('strong', {}, f.date), `\u00a0· ${fmt(f.wl_um, 3)}\u00a0µm`, f.detector != null ? `\u00a0· ${t('live.detector')}\u00a0${f.detector}` : '',
      `\u00a0· ${k + 1}/${n} `, badge(corr() ? 'resid' : 'data'));
    v.ov.replaceChildren();
    if (st.gaia && hasGaia) for (const g of field.gaia) v.ov.append(s('circle', { cx: g.x + 0.5, cy: g.y + 0.5, r: size / 32, fill: 'none', stroke: '#ffffff', 'stroke-opacity': 0.55, 'stroke-width': size / 160 }));
    for (const L of f.labels || []) {
      const isChange = L.change ?? (L.label === 'unexplained' || L.label === 'artifact');
      if (L.label === 'unexplained' && !corr()) continue; // these changes are defined by the colour-aware view
      v.ov.append(marker(L, corr() && isChange));
    }
  }

  // Marker with a hover title and a tap target; the tip text is the label, name and (for asteroids) V magnitude.
  function marker(L, emphasise) {
    const c = CATS[L.label] || CATS.unexplained, x = L.x + 0.5, y = L.y + 0.5, r = size / 18;
    const attrs = { fill: 'none', stroke: c.color, 'stroke-width': size / (emphasise ? 60 : 110) };
    let m;
    if (c.shape === 'circle') m = s('circle', { cx: x, cy: y, r, ...attrs });
    else if (c.shape === 'ring') m = s('circle', { cx: x, cy: y, r, ...attrs, 'stroke-dasharray': `${r / 2} ${r / 3}` });
    else if (c.shape === 'square') m = s('rect', { x: x - r, y: y - r, width: 2 * r, height: 2 * r, ...attrs });
    else if (c.shape === 'diamond') m = s('path', { d: `M${x},${y - r}L${x + r},${y}L${x},${y + r}L${x - r},${y}Z`, ...attrs });
    else if (c.shape === 'cross') m = s('path', { d: `M${x},${y - 2 * r}V${y - r}M${x},${y + r}V${y + 2 * r}M${x - 2 * r},${y}H${x - r}M${x + r},${y}H${x + 2 * r}`, ...attrs });
    else m = s('path', { d: `M${x},${y - r}L${x + r},${y + r * 0.8}L${x - r},${y + r * 0.8}Z`, ...attrs });
    const tip = markerText(L);
    const g = s('g', { class: 'mk', 'data-tip': tip }, s('title', {}, tip), m, s('circle', { cx: x, cy: y, r: r * 1.6, fill: 'transparent', class: 'hit' }));
    return g;
  }
  function markerText(L) {
    const c = CATS[L.label] || CATS.unexplained;
    return [t(c.key), L.name && L.name !== 'unknown' ? L.name : null, L.vmag != null ? `V ${fmt(L.vmag, 3)}` : null,
      L.confidence != null && L.label === 'unexplained' ? t('tm.confidence', { p: fmt(100 * L.confidence, 2) }) : null].filter(Boolean).join(' · ');
  }

  function step(d) { const n = list().length; st.i = (st.i + d + n) % n; st.playing = false; schedule(); render(); }
  function schedule() {
    clearInterval(timer);
    if (st.playing && st.layout === 'blink') timer = setInterval(() => { st.i = (st.i + 1) % list().length; render(); }, 1000 / st.fps);
  }

  function render() {
    const L = list(), side = st.layout === 'side';
    st.i = Math.min(st.i, L.length - 1);
    controls.update(st.view, st.det);
    layoutBtns.forEach((b, k) => b.setAttribute('aria-pressed', String((k === 1) === side)));
    gaiaBtn.setAttribute('aria-pressed', String(st.gaia));
    playBtn.textContent = t(st.playing ? 'tm.pause' : 'tm.play');
    viewB.root.hidden = !side; blinkCtl.hidden = side; sideCtl.hidden = !side;
    const fa = side ? frames[st.a] : L[st.i];
    fill(viewA, fa, side ? st.a : st.i, side ? frames.length : L.length);
    if (side) fill(viewB, frames[st.b], st.b, frames.length);
    if (lastList !== L.length + st.view + st.det) { // timeline of the active list, coloured by wavelength
      lastList = L.length + st.view + st.det;
      scrub.max = L.length - 1;
      scrub.style.setProperty('--track', `linear-gradient(90deg, ${L.map((f, k) => `${wlColor(f.wl_um)} ${k / L.length * 100}% ${(k + 1) / L.length * 100}%`).join(', ')})`);
      ends.replaceChildren(h('span', {}, L[0].date), h('span', {}, t('tm.timeline_note')), h('span', {}, L[L.length - 1].date));
    }
    scrub.value = st.i;
    scrub.setAttribute('aria-valuetext', `${L[st.i].date}, ${fmt(L[st.i].wl_um, 3)} µm`);
    selA.value = st.a; selB.value = st.b;
    // Running count of flagged changes in the current view (from the field file's per-picture counts).
    const key = COUNT_KEY[st.view];
    if (L.some(f => f[key] != null)) {
      const upto = L.slice(0, st.i + 1).reduce((a, f) => a + (f[key] || 0), 0), total = L.reduce((a, f) => a + (f[key] || 0), 0);
      counter.replaceChildren(...(total === 0 ? [t('view.flagged_none', { N: L.length })]
        : [h('strong', {}, t('view.flagged', { n: upto })), ' ', h('span', { class: 'muted' }, t('view.flagged_total', { total, N: L.length }))]));
    } else counter.replaceChildren();
    const nxt = L[(st.i + 1) % L.length];
    new Image().src = asset(corr() ? nxt.corr_png : nxt.img_png); // preload one ahead only
    if (small) return;

    const shown = side ? [frames[st.a], frames[st.b]] : [fa];
    const asts = shown.flatMap(f => (f.labels || []).filter(x => x.label === 'known asteroid'));
    astList.replaceChildren(...(asts.length ? [h('span', { class: 'key key-circle', style: `--c:${CATS['known asteroid'].color}`, 'aria-hidden': 'true' }),
      t('tm.asteroids_here', { list: asts.map(markerText).map(x => x.replace(`${t(CATS['known asteroid'].key)} · `, '')).join('; ') })] : []));
    labelList.replaceChildren();
    if (corr()) {
      const Ls = shown.flatMap(f => (f.labels || []).filter(x => x.change ?? x.label === 'unexplained'));
      if (!Ls.length) labelList.append(h('li', { class: 'muted' }, t('tm.no_labels')));
      for (const x of Ls) {
        const c = CATS[x.label] || CATS.unexplained;
        labelList.append(h('li', {}, h('span', { class: `key key-${c.shape}`, style: `--c:${c.color}`, 'aria-hidden': 'true' }),
          h('strong', {}, t(c.key)), ` — ${x.name && x.name !== 'unknown' ? x.name : t('tm.unnamed')}`,
          x.confidence != null ? h('span', { class: 'muted' }, ` · ${t('tm.confidence', { p: fmt(100 * x.confidence, 2) })} `) : ' ',
          badge('derived')));
      }
    }
    const fl = Object.entries(fa.flags || {}), meta = META_KEYS.filter(k => fa.meta?.[k] != null);
    put(flagBox,
      fl.length ? [h('dt', { class: 'flag-h' }, t('tm.flags_pixels', { n: size })), h('dd', {}, fl.map(([k, n]) => `${k} ${n}`).join(' · '), ' ', badge('data'))] : null,
      meta.length ? [h('dt', { class: 'flag-h' }, t('tm.flags_header')), h('dd', {}, meta.map(k => `${k} ${typeof fa.meta[k] === 'number' ? fmt(fa.meta[k], 4) : fa.meta[k]}`).join(' · '), ' ', badge('data'))] : null,
      fa.s3_url ? [h('dt', {}, t('live.meta_file')), h('dd', {}, h('a', { href: fa.s3_url, rel: 'noopener' }, fa.obs_id))] : null,
      !fl.length && !meta.length && !fa.s3_url ? h('dd', { class: 'muted' }, t('tm.flags_none')) : null);
  }

  const blinkCtl = h('div', { class: 'controls' }, prev, playBtn, next,
    h('label', { for: speed.id }, t('tm.speed')), speed);
  const sideCtl = h('div', { class: 'controls' }, selA, selB);
  const legend = h('ul', { class: 'legend-row', 'aria-label': t('tm.legend') },
    Object.values(CATS).map(c => h('li', {}, h('span', { class: `key key-${c.shape}`, style: `--c:${c.color}`, 'aria-hidden': 'true' }), t(c.key))));

  const imgs = h('div', { class: 'tm-views viewer-img' }, viewA.root, viewB.root);
  const contrast = contrastControl(imgs);
  const film = filmButton(() => ({ sources: list().map(f => fromURL(asset(corr() ? f.corr_png : f.img_png))), fps: st.fps, k: contrast.getK(),
    name: `epoch_${String(field.field || 'field').replace(/[^\w-]/g, '')}` }));
  put(el, h('div', { class: small ? '' : 'viewer' }, // picture left, its controls right (stacked on narrow screens)
    imgs,
    h('div', { class: 'viewer-side' },
      minimal ? null : [blinkCtl, sideCtl,
        h('div', { class: 'scrub-wrap' }, h('label', { for: scrub.id, class: 'sr-only' }, t('tm.timeline')), scrub, ends),
        h('div', { class: 'controls film-row' }, contrast.el, film)],
      counter, small ? null : compare, controls,
      small ? null : h('div', { class: 'segmented', role: 'group', 'aria-label': t('tm.mode_label') }, layoutBtns),
      small ? null : [astList, legend, hasGaia ? h('p', {}, gaiaBtn) : gaiaLive ? h('p', { class: 'small' }, gaiaLive) : null, labelList,
        h('details', { class: 'flags', id: 'tm-flags' }, h('summary', {}, t('tm.flags_h')), flagBox)])));

  const onKey = e => {
    if (e.target.closest?.('input, select, textarea')) return;
    if (e.key === 'ArrowRight') { step(1); e.preventDefault(); }
    else if (e.key === 'ArrowLeft') { step(-1); e.preventDefault(); }
  };
  const keyTarget = small ? el : document;
  keyTarget.addEventListener('keydown', onKey);
  schedule(); render();
  return { stop() { clearInterval(timer); st.playing = false; keyTarget.removeEventListener('keydown', onKey); } };
}
