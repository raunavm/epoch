// Live mode: any RA/Dec -> SPHEREx DR1 frames read straight from the public S3 bucket in the browser.
import { h, put, t, fmt, badge, link, wlColor, REDUCED_MOTION, getJSON } from '../core.js';
import { headlineLine, evidenceNum } from '../components/evidence.js';
import { viewControls, detectorInfo, busiestDetector, VIEWS } from '../components/blink.js';
import { plot, s as svg } from '../plot.js';
import { findFrames, spreadInTime } from './index.js';
import { readHeader, readRows } from './fits.js';
import { makeWCS, waveAt } from './wcs.js';
import { gridSize, rowsNeeded, resample, aperture, toMJy, asinhRGBA, divergingRGBA, SCALE_ARCSEC } from './grid.js';
import { changeModel, median } from './change.js';
import { fetchGaia, starXY, starRadius, starPM, MAX_STARS } from './gaia.js';
import { wiseStamp, hipsDates, HIPS } from './wise.js';

const RELEASE = 'dr1'; // the one line to change for a new data release (after building data/index/<release>/)
const INDEX = `data/index/${RELEASE}/`;
const MAX_CONCURRENT = 6, CONFIRM_MB = 120, F_PROVISIONAL = 0.1, SUBSET_ABOVE = 400, SUBSET_N = 300;
const PASS_COLORS = ['#56B4E9', '#E69F00'];
// Gaia overlay: cone margin for stars that drift into view, arrow length, and the speed above which a star gets an arrow.
const GAIA_COLOR = '#009E73', GAIA_MARGIN_DEG = 0.05, ARROW_YEARS = 10, FAST_PM = 0.3;
const WISE_K = 3; // WISE picture drawn at 3 x 3 sub-pixels per live-grid pixel
const WISE_DET = 6; // the SPHEREx detector (4.4-5.0 µm) whose band contains WISE's 4.6 µm; enough for the WISE comparison
const META_KEYS = [['PSF_FWHM', 'live.meta_psf'], ['L2DQAFLG', 'live.meta_dqa'], ['L2 N_OUTLIER', 'live.meta_outlier'],
  ['L2 N_PERSIST', 'live.meta_persist'], ['L2 SATSTRKS', 'live.meta_streaks'], ['L2 N_GHOST', 'live.meta_ghost']];

export const bytesPerFrame = S => (Math.ceil(S * Math.SQRT2) + 6) * 2040 * 4 + 28800;

let cached = null;
async function getJSONAbs(path, signal) {
  try { const r = await fetch(path, { signal }); return r.ok ? await r.json() : null; } catch { return null; }
}
export async function loadIndex() {
  cached ??= Promise.all([getJSONAbs(INDEX + 'meta.json'), getJSONAbs(INDEX + 'wave_tables.json'), getJSONAbs('data/f1_calibration.json')]);
  const [meta, wave, cal] = await cached;
  const f = Number.isFinite(cal?.f) ? cal.f : null;
  return meta ? { meta, wave, f: f ?? F_PROVISIONAL, fProvisional: f == null } : null;
}

/**
 * Mount a live view into `el` for target {ra, dec, fov (arcmin), mode, gaia?, wise?, pm?: {pmra, pmdec, epochMjd}}.
 * Returns {stop()}. onState({mode, gaia, wise}) is called when the view changes (for the URL).
 */
export function mountLive(el, target, onState) {
  const ctl = new AbortController(), signal = ctl.signal;
  const fetchFn = (u, o) => fetch(u, { ...o, signal });
  const S = gridSize(target.fov), c = (S - 1) / 2;
  const LEGACY = { raw: 'all', corrected: 'corr' }, m0 = LEGACY[target.mode] || target.mode;
  // st.mode: which pictures/images the blink shows (all | det | corr); st.view: blink | add | subtract | wise.
  const st = { frames: [], i: 0, playing: !REDUCED_MOTION, fps: 2, mode: VIEWS.includes(m0) ? m0 : 'all', det: null, detChosen: false,
    view: target.wise ? 'wise' : ['add', 'subtract'].includes(target.mode) ? target.mode : 'blink', cm: null, sel: null, bytes: 0,
    gaia: !!target.gaia, stars: null, gaiaFrom: '', gaiaFailed: false, wisePlay: !REDUCED_MOTION };
  const list = () => (st.mode === 'det' ? st.frames.filter(f => f.det === st.det) : st.frames);
  let timer = null, wiseTimer = null, wiseFlip = null, wisePanels = [], gaiaAsked = false;

  const status = h('p', { class: 'live-status', role: 'status', 'aria-live': 'polite' }, t('live.searching'));
  const prog = h('progress', { max: 1, value: 0, 'aria-label': t('live.progress') });
  const canvas = h('canvas', { width: S, height: S, class: 'live-canvas', role: 'img', tabindex: '0', 'aria-label': t('live.canvas_aria') });
  const ov = svg('svg', { viewBox: `0 0 ${S} ${S}`, class: 'overlay', 'aria-hidden': 'true' });
  const caption = h('p', { class: 'frame-meta' });
  const metaBox = h('dl', { class: 'facts small live-meta' });
  const scrub = h('input', { type: 'range', class: 'scrub', min: 0, max: 0, value: 0, id: 'live-scrub', 'aria-label': t('tm.timeline') });
  const playBtn = h('button', { type: 'button', class: 'btn', onclick: () => { st.playing = !st.playing; schedule(); draw(); } });
  let info = {};
  const controls = viewControls({}, st, (m, d) => setMode(m, d));
  const VIEW_KEYS = ['blink', 'add', 'subtract', 'wise'];
  const views = VIEW_KEYS.map(k => h('button', { type: 'button', class: 'seg', onclick: () => { st.view = k; emit(); draw(); } }, t(`live.view_${k}`)));
  const addBox = h('div', { class: 'coadds', hidden: true });
  const subBox = h('div', { class: 'subtract', hidden: true });
  const wiseBox = h('div', { class: 'wise', hidden: true }, h('p', { class: 'hint' }, t('live.wise_wait')));
  const gaiaBtn = h('button', { type: 'button', class: 'check', onclick: () => { st.gaia = !st.gaia; emit(); loadGaia(); draw(); } }, t('live.gaia_show'));
  const gaiaNote = h('span', { class: 'muted small', role: 'status' });
  const gaiaTip = h('p', { class: 'marker-tip small', 'aria-live': 'polite' });
  const blinkBox = h('div', { class: 'live-blink' });
  const specBox = h('section', { class: 'live-spec', 'aria-labelledby': 'h-live-spec' },
    h('h3', { id: 'h-live-spec' }, t('live.spec_h')), h('p', { class: 'hint' }, t('live.spec_hint')));
  const findingBox = h('div', { class: 'finding-box' });
  const subsetNote = h('p', { class: 'hint' });

  function setMode(m, d) {
    if (m === 'corr' && !st.cm) return;
    const cur = list()[st.i];
    st.mode = m; if (d != null && d !== st.det) { st.det = d; st.detChosen = true; }
    const L = list();
    st.i = cur ? Math.max(0, L.reduce((best, f, k) => (Math.abs(f.mjd - cur.mjd) < Math.abs(L[best].mjd - cur.mjd) ? k : best), 0)) : 0;
    emit(); draw();
  }
  function emit() { onState?.({ mode: ['add', 'subtract'].includes(st.view) ? st.view : st.mode, gaia: st.gaia, wise: st.view === 'wise' }); }

  canvas.addEventListener('click', e => {
    const r = canvas.getBoundingClientRect();
    st.sel = [Math.floor((e.clientX - r.left) / r.width * S), Math.floor((e.clientY - r.top) / r.height * S)];
    starTip(e, canvas, list()[st.i]?.mjd);
    spectrum(); draw();
  });
  canvas.addEventListener('keydown', e => { // keyboard pixel picking: Enter selects the centre
    if (e.key === 'Enter') { st.sel = [Math.round(c), Math.round(c)]; spectrum(); draw(); }
  });
  scrub.addEventListener('input', () => { st.i = +scrub.value; st.playing = false; schedule(); draw(); });
  const onKey = e => {
    if (e.target.closest?.('input, select, textarea, button') && e.key === ' ') return;
    if (e.target.closest?.('input, select, textarea')) return;
    if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') { step(e.key === 'ArrowRight' ? 1 : -1); e.preventDefault(); }
    else if (e.key === ' ') { st.playing = !st.playing; schedule(); draw(); e.preventDefault(); }
    else if (e.key === 'c' || e.key === 'C') setMode(st.mode === 'corr' ? 'all' : 'corr', st.det);
  };
  document.addEventListener('keydown', onKey);

  function step(d) { const n = list().length; if (!n) return; st.i = (st.i + d + n) % n; st.playing = false; schedule(); draw(); }
  function schedule() {
    clearInterval(timer);
    if (st.playing && st.view === 'blink') timer = setInterval(() => { const n = list().length; if (n) { st.i = (st.i + 1) % n; draw(); } }, 1000 / st.fps);
  }

  function paint(cv, rgba, n = S) { cv.getContext('2d').putImageData(new ImageData(rgba, n, n), 0, 0); }

  function draw() {
    info = detectorInfo(st.frames, 'det', 'wl');
    if (!st.detChosen) st.det = busiestDetector(info) || null;
    controls.update(st.mode, st.det, !st.cm, info);
    controls.hidden = st.view !== 'blink'; // Add, Subtract and the WISE comparison do not use the three blink views
    views.forEach((b, k) => { b.setAttribute('aria-pressed', String(VIEW_KEYS[k] === st.view)); b.disabled = k > 0 && !st.cm; });
    blinkBox.hidden = st.view !== 'blink'; addBox.hidden = st.view !== 'add'; subBox.hidden = st.view !== 'subtract'; wiseBox.hidden = st.view !== 'wise';
    playBtn.textContent = t(st.playing ? 'tm.pause' : 'tm.play');
    gaiaBtn.setAttribute('aria-pressed', String(st.gaia));
    gaiaNote.textContent = !st.gaia ? '' : st.gaiaFailed ? t('live.gaia_none') : !st.stars ? t('live.gaia_loading')
      : t('live.gaia_legend', { max: MAX_STARS, service: st.gaiaFrom, yr: ARROW_YEARS, pm: fmt(FAST_PM, 2) });
    if (!st.gaia) gaiaTip.textContent = '';
    if (st.view === 'add') drawAdd();
    if (st.view === 'subtract') drawSubtract();
    if (st.view === 'wise') drawWise();
    if (st.view === 'wise') wisePanels.forEach(p => markers(p.o, p.mjd));
    clearInterval(wiseTimer);
    wiseTimer = st.view === 'wise' && st.wisePlay && wiseFlip ? setInterval(wiseFlip, 1000) : null;
    const L = list();
    st.i = Math.max(0, Math.min(st.i, L.length - 1));
    const f = L[st.i];
    if (!f) return;
    scrub.max = L.length - 1; scrub.value = st.i;
    scrub.setAttribute('aria-valuetext', `${f.date}, ${fmt(f.wl, 3)} µm, ${t('live.detector')} ${f.det}`);
    scrub.style.setProperty('--track', `linear-gradient(90deg, ${L.map((g, k) => `${wlColor(g.wl)} ${k / L.length * 100}% ${(k + 1) / L.length * 100}%`).join(', ')})`);
    const useCorr = st.mode === 'corr' && st.cm;
    paint(canvas, useCorr ? divergingRGBA(st.cm.corr[f.idx]) : f.rgba);
    caption.replaceChildren(h('span', { class: 'swatch', style: `background:${wlColor(f.wl)}`, 'aria-hidden': 'true' }),
      h('strong', {}, f.date), `\u00a0· ${fmt(f.wl, 3)}\u00a0µm\u00a0· ${t('live.detector')}\u00a0${f.det}\u00a0· ${st.i + 1}/${L.length} `, badge(useCorr ? 'resid' : 'data'));
    put(metaBox, ...META_KEYS.flatMap(([k, key]) => [h('dt', {}, t(key)), h('dd', {}, typeof f.hdr[k] === 'number' && !Number.isInteger(f.hdr[k]) ? fmt(f.hdr[k], 4) : f.hdr[k] ?? '—')]),
      h('dt', {}, t('live.meta_file')), h('dd', {}, h('a', { href: f.url, rel: 'noopener' }, f.obs_id + ` D${f.det}`)));
    markers(ov, f.mjd);
  }

  function markers(o, mjd) {
    o.replaceChildren(
      svg('path', { d: `M${c + 0.5},${c - 3}V${c - 1.5}M${c + 0.5},${c + 2.5}V${c + 4}M${c - 3},${c + 0.5}H${c - 1.5}M${c + 2.5},${c + 0.5}H${c + 4}`, stroke: '#F0E442', 'stroke-width': Math.max(0.15, S / 120), fill: 'none' }));
    if (target.pm && mjd) { // predicted position of a fast-moving catalogue star at this frame's time
      const yr = (mjd - target.pm.epochMjd) / 365.25, dx = target.pm.pmra * yr / 1000 / SCALE_ARCSEC, dy = target.pm.pmdec * yr / 1000 / SCALE_ARCSEC;
      o.append(svg('circle', { cx: c + 0.5 - dx, cy: c + 0.5 - dy, r: 2, fill: 'none', stroke: '#CC79A7', 'stroke-width': Math.max(0.15, S / 120) }));
    }
    if (st.sel) o.append(svg('rect', { x: st.sel[0] - 1, y: st.sel[1] - 1, width: 3, height: 3, fill: 'none', stroke: '#56B4E9', 'stroke-width': Math.max(0.12, S / 150) }));
    if (st.gaia && st.stars && mjd) for (const s of st.stars) { // Gaia DR3 stars, moved to this picture's date
      const p = starXY(s, mjd, S, target.ra, target.dec);
      if (!p || p[0] < -0.5 || p[1] < -0.5 || p[0] > S - 0.5 || p[1] > S - 0.5) continue;
      const x = p[0] + 0.5, y = p[1] + 0.5, sw = Math.max(0.12, S / 150);
      o.append(svg('circle', { cx: x, cy: y, r: starRadius(s.g), fill: 'none', stroke: GAIA_COLOR, 'stroke-width': sw }));
      if (starPM(s) > FAST_PM) { // arrow: where it will be ARROW_YEARS later
        const q = starXY(s, mjd + ARROW_YEARS * 365.25, S, target.ra, target.dec), x2 = q[0] + 0.5, y2 = q[1] + 0.5, a = Math.atan2(y2 - y, x2 - x);
        const tip = da => `${x2 - 1.2 * Math.cos(a + da)},${y2 - 1.2 * Math.sin(a + da)}`;
        o.append(svg('path', { d: `M${x},${y}L${x2},${y2}M${tip(0.5)}L${x2},${y2}L${tip(-0.5)}`, fill: 'none', stroke: GAIA_COLOR, 'stroke-width': sw }));
      }
    }
  }

  async function loadGaia() {
    if (!st.gaia || gaiaAsked) return;
    gaiaAsked = true;
    try {
      const r = await fetchGaia(target.ra, target.dec, target.fov / 60 * Math.SQRT1_2 + GAIA_MARGIN_DEG, fetchFn);
      st.stars = r.stars; st.gaiaFrom = r.service;
    } catch { st.gaiaFailed = true; }
    if (!signal.aborted) draw();
  }
  // Name the Gaia star under a click on canvas `cv`, which shows the sky at time `mjd`.
  function starTip(e, cv, mjd) {
    if (!st.gaia || !st.stars || !mjd) return;
    const r = cv.getBoundingClientRect(), x = (e.clientX - r.left) / r.width * S - 0.5, y = (e.clientY - r.top) / r.height * S - 0.5;
    let best = null, bd = Infinity;
    for (const s of st.stars) {
      const p = starXY(s, mjd, S, target.ra, target.dec), d = p ? Math.hypot(p[0] - x, p[1] - y) : Infinity;
      if (d < bd && d <= starRadius(s.g) + 0.5) { bd = d; best = s; }
    }
    gaiaTip.textContent = !best ? t('live.gaia_tip_none') : t(best.pmra == null ? 'live.gaia_tip_nopm' : 'live.gaia_tip',
      { id: best.id, g: fmt(best.g, 4), pm: fmt(starPM(best), 3), pmra: fmt(best.pmra / 1000, 3), pmdec: fmt(best.pmdec / 1000, 3) });
  }

  // n: pixels along a side (the WISE picture is finer than the live grid); mjd: date of the picture, for the moving markers.
  function smallCanvas(rgba, label, kind, n = S, mjd = null) {
    const cv = h('canvas', { width: n, height: n, class: 'live-canvas', role: 'img', 'aria-label': label });
    paint(cv, rgba, n);
    const o = svg('svg', { viewBox: `0 0 ${S} ${S}`, class: 'overlay', 'aria-hidden': 'true' }); markers(o, mjd);
    if (mjd) { wisePanels.push({ o, mjd }); cv.addEventListener('click', e => starTip(e, cv, mjd)); }
    return h('figure', {}, h('div', { class: 'imgwrap' }, cv, o), h('figcaption', {}, label, ' ', badge(kind)));
  }
  // Mean of maps[f.idx] over the pictures g, ignoring empty pixels.
  function meanOf(g, maps) {
    const m = new Float32Array(S * S);
    for (let p = 0; p < S * S; p++) { let s = 0, n = 0; for (const f of g) { const x = maps[f.idx][p]; if (Number.isFinite(x)) { s += x; n++; } } m[p] = n ? s / n : NaN; }
    return m;
  }

  function drawAdd() {
    if (!st.cm || addBox.dataset.done) return;
    const groups = [1, 2, 3, 4, 5, 6].map(d => st.frames.filter(f => f.det === d)).concat([st.frames]);
    put(addBox, h('p', { class: 'hint' }, t('live.add_hint')), h('div', { class: 'coadd-grid' }, groups.map((g, k) => {
      if (!g.length) return h('figure', {}, h('p', { class: 'unavailable small' }, t('live.add_none', { d: k + 1 })));
      const mean = meanOf(g, st.cm.vs);
      const wl = g.map(f => f.wl);
      const label = k < 6 ? t('live.add_det', { d: k + 1, n: g.length, lo: fmt(Math.min(...wl), 2), hi: fmt(Math.max(...wl), 2) }) : t('live.add_all', { n: g.length });
      return smallCanvas(asinhRGBA(mean), label, 'derived');
    })));
    addBox.dataset.done = '1';
  }

  function passes() {
    const fr = st.frames; let gap = -1, k = 0;
    for (let i = 1; i < fr.length; i++) if (fr[i].mjd - fr[i - 1].mjd > gap) { gap = fr[i].mjd - fr[i - 1].mjd; k = i; }
    return { gap, split: k, p1: fr.slice(0, k), p2: fr.slice(k) };
  }

  function drawSubtract() {
    if (!st.cm || subBox.dataset.done) return;
    const { gap, p1, p2 } = passes();
    if (p1.length < 1 || p2.length < 1) { put(subBox, h('p', { class: 'unavailable' }, t('live.sub_none'))); subBox.dataset.done = '1'; return; }
    const a = meanOf(p1, st.cm.corr), b = meanOf(p2, st.cm.corr), d = a.map((x, p) => x - b[p]);
    put(subBox,
      h('p', { class: 'hint' }, t('live.sub_hint', { gap: fmt(gap, 3), a: p1[0].date, b: p1[p1.length - 1].date, c: p2[0].date, d: p2[p2.length - 1].date, n1: p1.length, n2: p2.length })),
      gap < 60 ? h('p', { class: 'unavailable small' }, t('live.sub_short')) : null,
      h('div', { class: 'coadd-grid wide' }, smallCanvas(divergingRGBA(d, 4), t('live.sub_label'), 'resid')));
    subBox.dataset.done = '1';
  }

  // WISE look-back: the same patch in the AllWISE 4.6 µm picture beside SPHEREx's own pictures at the closest colours.
  async function drawWise() {
    if (!st.cm || wiseBox.dataset.done) return;
    wiseBox.dataset.done = '1';
    put(wiseBox, h('p', { class: 'hint', role: 'status' }, t('live.wise_loading')));
    try {
      const [dates, w] = await Promise.all([hipsDates(fetchFn), wiseStamp(target.ra, target.dec, S, WISE_K, fetchFn)]);
      const wMjd = (dates.mjdMin + dates.mjdMax) / 2, y1 = mjdDate(wMjd).slice(0, 4);
      const band = Object.keys(info).find(d => info[d].lo <= HIPS.um && HIPS.um <= info[d].hi); // the detector that covers WISE's colour
      const g = band ? st.frames.filter(f => f.det === +band) : st.frames, wl = g.map(f => f.wl);
      const sMjd = g.reduce((a, f) => a + f.mjd, 0) / g.length, a = g[0].date.slice(0, 10), b = g[g.length - 1].date.slice(0, 10);
      const y2 = [...new Set([a.slice(0, 4), b.slice(0, 4)])].join('–');
      // Median, not mean: live mode reads no pixel flags, and a median ignores a spike that is in one picture only.
      const sx = Float32Array.from({ length: S * S }, (_, p) => median(g.map(f => st.cm.vs[f.idx][p]).filter(Number.isFinite)));
      const pair = () => [smallCanvas(asinhRGBA(w.v), t('live.wise_label', { um: HIPS.um, year: y1 }), 'data', w.n, wMjd),
        smallCanvas(asinhRGBA(sx), t('live.wise_sx', { n: g.length, lo: fmt(Math.min(...wl), 2), hi: fmt(Math.max(...wl), 2), a, b }), 'derived', S, sMjd)];
      const flip = pair(); flip[1].hidden = true;
      wiseFlip = () => { flip[0].hidden = !flip[0].hidden; flip[1].hidden = !flip[0].hidden; };
      const playWise = h('button', { type: 'button', class: 'btn small', onclick: () => { st.wisePlay = !st.wisePlay; playWise.textContent = t(st.wisePlay ? 'tm.pause' : 'tm.play'); draw(); } },
        t(st.wisePlay ? 'tm.pause' : 'tm.play'));
      const swap = h('button', { type: 'button', class: 'btn small', onclick: () => wiseFlip() }, t('live.wise_swap'));
      const yr = target.pm ? Math.abs(wMjd - target.pm.epochMjd) / 365.25 : 0;
      const outside = target.pm && Math.max(Math.abs(target.pm.pmra), Math.abs(target.pm.pmdec)) * yr / 1000 / SCALE_ARCSEC > S / 2;
      put(wiseBox,
        h('p', { class: 'hint' }, t('live.wise_hint', { y1, y2 })),
        outside ? h('p', { class: 'unavailable small' }, t('live.wise_outside', { year: y1, d: fmt(Math.hypot(target.pm.pmra, target.pm.pmdec) * yr / 1000 / 60, 2) })) : null,
        h('div', { class: 'coadd-grid wise-pair' }, pair(), h('div', { class: 'wise-blink' }, flip, h('p', {}, h('strong', {}, t('live.wise_blink')), ' ', playWise, ' ', swap))),
        h('p', { class: 'muted small' }, t('live.wise_credit', { mb: fmt(w.nTiles * HIPS.width ** 2 * 4 / 1e6, 2) })));
      draw();
    } catch { if (!signal.aborted) put(wiseBox, h('p', { class: 'unavailable' }, t('live.wise_none'))); }
  }

  function spectrum() {
    if (!st.sel || !st.frames.length) return;
    const [i, j] = st.sel, { split } = passes();
    const rows = st.frames.map((f, k) => {
      const a = aperture(f.v, S, i, j);
      return { obs_id: f.obs_id, mjd: f.mjd, det: f.det, wl: f.lam[j * S + i], flux: toMJy(a.flux, f.omega), err: toMJy(a.err, f.omega), pass: k < split || !split ? 1 : 2 };
    }).filter(r => Number.isFinite(r.flux) && Number.isFinite(r.wl));
    const series = [1, 2].map(pn => { const r = rows.filter(x => x.pass === pn); return { x: r.map(x => x.wl), y: r.map(x => x.flux), err: r.map(x => x.err), kind: 'points', color: PASS_COLORS[pn - 1], label: t(`live.pass_${pn}`) }; }).filter(s => s.x.length);
    const csv = ['obs_id,mjd,detector,wavelength_um,flux_mJy,flux_err_mJy,group_1_earlier_2_later', ...rows.map(r => [r.obs_id, r.mjd, r.det, r.wl.toFixed(4), r.flux.toPrecision(5), r.err.toPrecision(3), r.pass].join(','))].join('\n');
    const dl = h('a', { class: 'btn small', download: `orrery_spectrum_${target.ra.toFixed(5)}_${target.dec.toFixed(5)}_px${i}_${j}.csv`, href: URL.createObjectURL(new Blob([csv], { type: 'text/csv' })) }, t('live.csv'));
    const [sra, sdec] = [target.ra, target.dec];
    put(specBox, h('h3', { id: 'h-live-spec' }, t('live.spec_h')),
      h('p', {}, h('strong', {}, t('live.spec_label')), ' ', badge('data'), ' ', t('live.spec_where', { i, j, dx: fmt((c - i) * SCALE_ARCSEC, 3), dy: fmt((c - j) * SCALE_ARCSEC, 3) })),
      rows.length ? plot(series, { xlabel: t('plot.wl'), ylabel: t('plot.flux'), title: t('live.spec_title', { n: rows.length, ra: fmt(sra, 7), dec: fmt(sdec, 7) }) }) : h('p', { class: 'unavailable' }, t('live.spec_none')),
      h('p', { class: 'muted small' }, t('live.spec_caveat')), rows.length ? dl : null);
  }

  // ---- loading ----
  put(el,
    h('div', { class: 'live-head' }, status, prog, subsetNote),
    controls,
    h('div', { class: 'segmented', role: 'group', 'aria-label': t('live.view_label') }, views),
    h('p', {}, gaiaBtn, ' ', gaiaNote), gaiaTip,
    blinkBox, addBox, subBox, wiseBox, findingBox, specBox);
  blinkBox.append(
    h('div', { class: 'live-grid' },
      h('figure', { class: 'tm-view' }, h('div', { class: 'imgwrap' }, canvas, ov), caption),
      h('details', { class: 'live-meta-wrap' }, h('summary', {}, t('live.meta_h')), metaBox)),
    h('div', { class: 'controls' },
      h('button', { type: 'button', class: 'btn', 'aria-label': t('tm.prev'), onclick: () => step(-1) }, '←'), playBtn,
      h('button', { type: 'button', class: 'btn', 'aria-label': t('tm.next'), onclick: () => step(1) }, '→'),
      h('span', { class: 'muted small' }, t('live.keys'))),
    h('div', { class: 'scrub-wrap' }, scrub, h('div', { class: 'scrub-ends muted' }, h('span', {}, t('tm.timeline_note')))),
    h('p', { class: 'muted small' }, t('live.markers')));
  draw(); loadGaia();
  run().catch(err => { if (!signal.aborted) status.textContent = t('live.error', { msg: err.message }); });

  async function run() {
    const t0 = performance.now();
    const idx = await loadIndex();
    if (!idx) { status.textContent = t('live.no_index'); return; }
    const { meta, wave, f: fFloor, fProvisional } = idx;
    const tileCache = {};
    const found = (await findFrames(meta, target.ra, target.dec, target.fov / 60,
      n => (tileCache[n] ??= getJSONAbs(`${INDEX}${n}.json`, signal)))).frames;
    if (signal.aborted) return;
    // Opened on the WISE comparison: load only the detector that matches WISE's colour, to keep the download small.
    const wiseOnly = !!target.wise && !target.all && found.some(f => f.det === WISE_DET);
    const pool = wiseOnly ? found.filter(f => f.det === WISE_DET) : found;
    const frames = pool.length > SUBSET_ABOVE && !target.all ? spreadInTime(pool, SUBSET_N) : pool;
    if (frames.length < found.length) {
      put(subsetNote, t(wiseOnly ? 'live.subset_wise' : 'live.subset', { n: frames.length, N: found.length, d: WISE_DET }), ' ',
        h('button', { type: 'button', class: 'btn small', onclick: () => target.onLoadAll?.() },
          t('live.load_all', { N: found.length, mb: fmt(found.length * bytesPerFrame(S) / 1e6, 2) })));
    }
    if (!frames.length) {
      status.textContent = t('live.none', { n: meta.n_frames.toLocaleString(), a: mjdDate(meta.mjd_min), b: mjdDate(meta.mjd_max) });
      return;
    }
    const estMB = frames.length * bytesPerFrame(S) / 1e6;
    if (estMB > CONFIRM_MB) {
      status.textContent = t('live.confirm', { n: frames.length, mb: fmt(estMB, 2) });
      await new Promise(res => status.append(' ', h('button', { type: 'button', class: 'btn small', onclick: res }, t('live.confirm_go'))));
    }
    prog.max = frames.length;
    let done = 0, failed = 0, first = null;
    const queue = frames.map((f, k) => ({ ...f, n: k }));
    const work = async () => {
      while (queue.length && !signal.aborted) {
        const f = queue.shift();
        try {
          const url = `${meta.bucket}/${f.key}`;
          const head = await readHeader(url, fetchFn);
          const wcs = makeWCS(head.hdr);
          const [y0, y1] = rowsNeeded(wcs, S, target.ra, target.dec);
          const rows = await readRows(url, head, y0, y1, fetchFn);
          st.bytes += head.bytes + rows.bytes;
          const tab = wave?.[String(f.det)];
          const { v, lam } = resample(wcs, rows, tab, S, target.ra, target.dec);
          const p0 = wcs.world2pix(target.ra, target.dec);
          const fr = { ...f, url, hdr: head.hdr, v, lam, wl: tab ? waveAt(tab, p0[0], p0[1])[0] : NaN, omega: head.hdr.OMEGA_MEDIAN,
            date: (head.hdr['DATE-AVG'] || mjdDate(f.mjd)).slice(0, 16).replace('T', ' '), rgba: asinhRGBA(v), det: f.det, obs_id: f.obs_id };
          const pos = st.frames.findIndex(g => g.mjd > fr.mjd);
          const cur = list()[st.i];
          st.frames.splice(pos < 0 ? st.frames.length : pos, 0, fr);
          if (cur) st.i = Math.max(0, list().indexOf(cur));
          if (!first) { first = (performance.now() - t0) / 1000; schedule(); }
        } catch (err) { if (signal.aborted) return; failed++; }
        done++; prog.value = done;
        status.textContent = t('live.loading', { done, n: frames.length, mb: fmt(st.bytes / 1e6, 3), failed });
        draw();
      }
    };
    await Promise.all(Array.from({ length: MAX_CONCURRENT }, work));
    if (signal.aborted) return;
    const total = (performance.now() - t0) / 1000;
    if (!st.frames.length) { status.textContent = t('live.all_failed'); return; }
    st.frames.forEach((f, k) => { f.idx = k; });
    st.cm = changeModel(st.frames.map(f => ({ v: f.v, lam: f.lam, det: f.det, mjd: f.mjd })), S, fFloor);
    status.textContent = t('live.done', { n: st.frames.length, failed, mb: fmt(st.bytes / 1e6, 3), first: fmt(first, 2), total: fmt(total, 2) });
    const headline = await getJSON('headline.json'), fTxt = fmt(fFloor, 2);
    const liveEv = evidenceNum(t('live.changes_ev', { n: st.frames.length }), {
      frames: st.frames.map(f => ({ obs_id: f.obs_id, date: f.date, wl_um: f.wl, detector: f.det, s3_url: f.url })),
      method: 'science.html#which-change-is-real',
    });
    if (!wiseOnly) put(findingBox, // comparing views needs pictures from every detector
      h('h3', {}, t('live.changes_h')),
      headline?.number != null ? headlineLine(headline, t('tm.f1_line')) : null,
      h('p', {}, h('strong', {}, t('live.changes_view')), ' ', t('live.changes', { all: st.cm.nNaiveAll, det: st.cm.nNaiveDet, corr: st.cm.nCorr }), ' ', badge('derived'), ' ', liveEv[0]),
      liveEv[1],
      h('p', { class: 'small' }, t('live.changes_note'), ' ', h('a', { href: link('failed.html') }, t('nav.failed')), '.'),
      h('p', { class: 'muted small' }, t('live.changes_method', { f: fTxt }), fProvisional ? [' ', h('strong', {}, t('live.provisional', { f: fTxt }))] : null));
    if (st.sel) spectrum();
    draw();
  }

  return {
    stop() { ctl.abort(); clearInterval(timer); clearInterval(wiseTimer); document.removeEventListener('keydown', onKey); },
    get state() { return st; },
  };
}

export function mjdDate(mjd) {
  return new Date((mjd - 40587) * 86400000).toISOString().slice(0, 10);
}
