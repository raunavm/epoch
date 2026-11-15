// Live mode: any RA/Dec -> SPHEREx DR1 frames read straight from the public S3 bucket in the browser.
import { h, put, t, fmt, badge, wlColor, REDUCED_MOTION, getJSON, findingLine } from '../core.js';
import { plot, s as svg } from '../plot.js';
import { findFrames, spreadInTime } from './index.js';
import { readHeader, readRows } from './fits.js';
import { makeWCS, waveAt } from './wcs.js';
import { gridSize, rowsNeeded, resample, aperture, toMJy, asinhRGBA, divergingRGBA, SCALE_ARCSEC } from './grid.js';
import { changeModel } from './change.js';

const INDEX = 'data/index/dr1/';
const MAX_CONCURRENT = 6, CONFIRM_MB = 120, F_PROVISIONAL = 0.1, SUBSET_ABOVE = 400, SUBSET_N = 300;
const PASS_COLORS = ['#56B4E9', '#E69F00'];
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
 * Mount a live view into `el` for target {ra, dec, fov (arcmin), mode, pm?: {ra0, dec0, pmra, pmdec, epochMjd}}.
 * Returns {stop()}. onState(state) is called when ra/dec/fov/mode change (for the URL).
 */
export function mountLive(el, target, onState) {
  const ctl = new AbortController(), signal = ctl.signal;
  const fetchFn = (u, o) => fetch(u, { ...o, signal });
  const S = gridSize(target.fov), c = (S - 1) / 2;
  const st = { frames: [], i: 0, playing: !REDUCED_MOTION, fps: 2, corrected: target.mode === 'corrected', view: ['add', 'subtract'].includes(target.mode) ? target.mode : 'blink', cm: null, sel: null, bytes: 0 };
  let timer = null;

  const status = h('p', { class: 'live-status', role: 'status', 'aria-live': 'polite' }, t('live.searching'));
  const prog = h('progress', { max: 1, value: 0, 'aria-label': t('live.progress') });
  const canvas = h('canvas', { width: S, height: S, class: 'live-canvas', role: 'img', tabindex: '0', 'aria-label': t('live.canvas_aria') });
  const ov = svg('svg', { viewBox: `0 0 ${S} ${S}`, class: 'overlay', 'aria-hidden': 'true' });
  const caption = h('p', { class: 'frame-meta' });
  const metaBox = h('dl', { class: 'facts small live-meta' });
  const scrub = h('input', { type: 'range', class: 'scrub', min: 0, max: 0, value: 0, id: 'live-scrub', 'aria-label': t('tm.timeline') });
  const playBtn = h('button', { type: 'button', class: 'btn', onclick: () => { st.playing = !st.playing; schedule(); draw(); } });
  const sw = ['raw', 'corrected'].map(k => h('button', { type: 'button', class: 'seg', onclick: () => setCorrected(k === 'corrected') }, t(`live.${k}`)));
  const swExplain = h('p', { class: 'hint' });
  const views = ['blink', 'add', 'subtract'].map(k => h('button', { type: 'button', class: 'seg', onclick: () => { st.view = k; emit(); draw(); } }, t(`live.view_${k}`)));
  const addBox = h('div', { class: 'coadds', hidden: true });
  const subBox = h('div', { class: 'subtract', hidden: true });
  const blinkBox = h('div', { class: 'live-blink' });
  const specBox = h('section', { class: 'live-spec', 'aria-labelledby': 'h-live-spec' },
    h('h3', { id: 'h-live-spec' }, t('live.spec_h')), h('p', { class: 'hint' }, t('live.spec_hint')));
  const findingBox = h('div', { class: 'finding-box' });
  const subsetNote = h('p', { class: 'hint' });

  function setCorrected(on) {
    if (on && !st.cm) return;
    st.corrected = on; emit(); draw();
  }
  function emit() { onState?.({ mode: st.view === 'blink' ? (st.corrected ? 'corrected' : 'raw') : st.view }); }

  canvas.addEventListener('click', e => {
    const r = canvas.getBoundingClientRect();
    st.sel = [Math.floor((e.clientX - r.left) / r.width * S), Math.floor((e.clientY - r.top) / r.height * S)];
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
    else if (e.key === 'c' || e.key === 'C') setCorrected(!st.corrected);
  };
  document.addEventListener('keydown', onKey);

  function step(d) { if (!st.frames.length) return; st.i = (st.i + d + st.frames.length) % st.frames.length; st.playing = false; schedule(); draw(); }
  function schedule() {
    clearInterval(timer);
    if (st.playing && st.view === 'blink') timer = setInterval(() => { if (st.frames.length) { st.i = (st.i + 1) % st.frames.length; draw(); } }, 1000 / st.fps);
  }

  function paint(cv, rgba) { cv.getContext('2d').putImageData(new ImageData(rgba, S, S), 0, 0); }

  function draw() {
    sw.forEach((b, k) => { b.setAttribute('aria-pressed', String((k === 1) === st.corrected)); b.disabled = k === 1 && !st.cm; });
    swExplain.textContent = !st.cm ? t('live.corrected_wait') : t(st.corrected ? 'tm.corrected_explain' : 'tm.raw_explain');
    views.forEach((b, k) => { b.setAttribute('aria-pressed', String(['blink', 'add', 'subtract'][k] === st.view)); b.disabled = k > 0 && !st.cm; });
    blinkBox.hidden = st.view !== 'blink'; addBox.hidden = st.view !== 'add'; subBox.hidden = st.view !== 'subtract';
    playBtn.textContent = t(st.playing ? 'tm.pause' : 'tm.play');
    if (st.view === 'add') drawAdd();
    if (st.view === 'subtract') drawSubtract();
    const f = st.frames[st.i];
    if (!f) return;
    scrub.max = st.frames.length - 1; scrub.value = st.i;
    scrub.setAttribute('aria-valuetext', `${f.date}, ${fmt(f.wl, 3)} µm, ${t('live.detector')} ${f.det}`);
    scrub.style.setProperty('--track', `linear-gradient(90deg, ${st.frames.map((g, k) => `${wlColor(g.wl)} ${k / st.frames.length * 100}% ${(k + 1) / st.frames.length * 100}%`).join(', ')})`);
    const useCorr = st.corrected && st.cm;
    paint(canvas, useCorr ? divergingRGBA(st.cm.corr[f.idx]) : f.rgba);
    caption.replaceChildren(h('span', { class: 'swatch', style: `background:${wlColor(f.wl)}`, 'aria-hidden': 'true' }),
      h('strong', {}, f.date), ` · ${fmt(f.wl, 3)} µm · ${t('live.detector')} ${f.det} · ${st.i + 1}/${st.frames.length} · `, badge(useCorr ? 'resid' : 'data'));
    put(metaBox, ...META_KEYS.flatMap(([k, key]) => [h('dt', {}, t(key)), h('dd', {}, f.hdr[k] ?? '—')]),
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
  }

  function smallCanvas(rgba, label, kind) {
    const cv = h('canvas', { width: S, height: S, class: 'live-canvas', role: 'img', 'aria-label': label });
    paint(cv, rgba);
    const o = svg('svg', { viewBox: `0 0 ${S} ${S}`, class: 'overlay', 'aria-hidden': 'true' }); markers(o, null);
    return h('figure', {}, h('div', { class: 'imgwrap' }, cv, o), h('figcaption', {}, label, ' ', badge(kind)));
  }

  function drawAdd() {
    if (!st.cm || addBox.dataset.done) return;
    const groups = [1, 2, 3, 4, 5, 6].map(d => st.frames.filter(f => f.det === d)).concat([st.frames]);
    put(addBox, h('p', { class: 'hint' }, t('live.add_hint')), h('div', { class: 'coadd-grid' }, groups.map((g, k) => {
      if (!g.length) return h('figure', {}, h('p', { class: 'unavailable small' }, t('live.add_none', { d: k + 1 })));
      const mean = new Float32Array(S * S);
      for (let p = 0; p < S * S; p++) { let s = 0, n = 0; for (const f of g) { const x = st.cm.vs[f.idx][p]; if (Number.isFinite(x)) { s += x; n++; } } mean[p] = n ? s / n : NaN; }
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
    const mean = g => { const m = new Float32Array(S * S); for (let p = 0; p < S * S; p++) { let s = 0, n = 0; for (const f of g) { const x = st.cm.corr[f.idx][p]; if (Number.isFinite(x)) { s += x; n++; } } m[p] = n ? s / n : NaN; } return m; };
    const a = mean(p1), b = mean(p2), d = a.map((x, p) => x - b[p]);
    put(subBox,
      h('p', { class: 'hint' }, t('live.sub_hint', { gap: fmt(gap, 3), a: p1[0].date, b: p1[p1.length - 1].date, c: p2[0].date, d: p2[p2.length - 1].date, n1: p1.length, n2: p2.length })),
      gap < 60 ? h('p', { class: 'unavailable small' }, t('live.sub_short')) : null,
      h('div', { class: 'coadd-grid wide' }, smallCanvas(divergingRGBA(d, 4), t('live.sub_label'), 'resid')));
    subBox.dataset.done = '1';
  }

  function spectrum() {
    if (!st.sel || !st.frames.length) return;
    const [i, j] = st.sel, { split } = passes();
    const rows = st.frames.map((f, k) => {
      const a = aperture(f.v, S, i, j);
      return { obs_id: f.obs_id, mjd: f.mjd, det: f.det, wl: f.lam[j * S + i], flux: toMJy(a.flux, f.omega), err: toMJy(a.err, f.omega), pass: k < split || !split ? 1 : 2 };
    }).filter(r => Number.isFinite(r.flux) && Number.isFinite(r.wl));
    const series = [1, 2].map(pn => { const r = rows.filter(x => x.pass === pn); return { x: r.map(x => x.wl), y: r.map(x => x.flux), err: r.map(x => x.err), kind: 'points', color: PASS_COLORS[pn - 1], label: t('live.pass', { n: pn }) }; }).filter(s => s.x.length);
    const csv = ['obs_id,mjd,detector,wavelength_um,flux_mJy,flux_err_mJy,pass', ...rows.map(r => [r.obs_id, r.mjd, r.det, r.wl.toFixed(4), r.flux.toPrecision(5), r.err.toPrecision(3), r.pass].join(','))].join('\n');
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
    h('div', { class: 'segmented big', role: 'group', 'aria-label': t('tm.switch_label') }, sw), swExplain,
    h('div', { class: 'segmented', role: 'group', 'aria-label': t('live.view_label') }, views),
    blinkBox, addBox, subBox, findingBox, specBox);
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
  draw();
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
    const frames = found.length > SUBSET_ABOVE && !target.all ? spreadInTime(found, SUBSET_N) : found;
    if (frames.length < found.length) {
      put(subsetNote, t('live.subset', { n: frames.length, N: found.length }), ' ',
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
          const cur = st.frames[st.i];
          st.frames.splice(pos < 0 ? st.frames.length : pos, 0, fr);
          if (cur) st.i = st.frames.indexOf(cur);
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
    const headline = (await getJSON('findings.json'))?.f1;
    put(findingBox,
      h('h3', {}, t('live.changes_h')),
      headline ? findingLine(headline) : null,
      h('p', {}, h('strong', {}, t('live.changes_view')), ' ', t('live.changes', { all: st.cm.nNaiveAll, det: st.cm.nNaiveDet, corr: st.cm.nCorr }), ' ', badge('derived')),
      h('p', { class: 'muted small' }, t('live.changes_method', { f: fFloor }), fProvisional ? [' ', h('strong', {}, t('live.provisional', { f: fFloor }))] : null));
    if (st.sel) spectrum();
    draw();
  }

  return {
    stop() { ctl.abort(); clearInterval(timer); document.removeEventListener('keydown', onKey); },
    get state() { return st; },
  };
}

export function mjdDate(mjd) {
  return new Date((mjd - 40587) * 86400000).toISOString().slice(0, 10);
}
