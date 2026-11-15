// Finding 2: combining frames along the orbit. Slider from 1 frame to all frames.
import { h, put, t, asset, badge, fmt, wlColor, REDUCED_MOTION } from '../core.js';
import { plot } from '../plot.js';
import { evidenceNum } from './evidence.js';

export function mountReveal(el, data, { compact = false, autoplay = false, snrNote = null } = {}) {
  const frames = data.frames || [], stacks = [...(data.stack_pngs || [])].sort((a, b) => a.n - b.n);
  const specs = [...(data.spectrum_points || [])].sort((a, b) => a.n_frames_used - b.n_frames_used);
  const N = Math.max(frames.length, ...stacks.map(s => s.n), 1);
  const full = specs[specs.length - 1];
  const xlim = full ? pad(full.wl_um) : null;
  const ylim = full ? pad(full.flux_mJy.flatMap((f, i) => [f - (full.err?.[i] || 0), f + (full.err?.[i] || 0)])) : null;

  const img = h('img', { class: 'stamp', alt: '', width: 256, height: 256, decoding: 'async' });
  const imgBadge = h('span', { class: 'img-badge' });
  const cap = h('p', { class: 'caption', 'aria-live': 'polite' });
  const snr = h('p', { class: 'snr' });
  const specBox = h('div', { class: 'spec-box' });
  const slider = h('input', { type: 'range', min: 1, max: N, value: 1, step: 1, id: `rv-${Math.random().toString(36).slice(2, 7)}` });
  const play = h('button', { type: 'button', class: 'btn' }, t('reveal.play'));
  let timer = null;

  function stop() { clearInterval(timer); timer = null; play.textContent = t('reveal.play'); }
  play.addEventListener('click', () => {
    if (timer) return stop();
    if (+slider.value >= N) slider.value = 1;
    play.textContent = t('reveal.pause');
    timer = setInterval(() => {
      if (+slider.value >= N) return stop();
      slider.value = Math.min(N, +slider.value + Math.max(1, Math.round(N / 40)));
      render();
    }, 160);
    render();
  });
  slider.addEventListener('input', () => { stop(); render(); });

  function render() {
    const n = +slider.value;
    const st = stacks.filter(s => s.n <= n).pop();
    let src, kind, label;
    if (st && (st.n > 1 || !frames[0])) { src = st.png; kind = st.n > 1 ? 'derived' : 'data'; label = st.n; }
    else { src = frames[0]?.single_png; kind = 'data'; label = 1; }
    if (src) img.src = asset(src);
    img.alt = t('reveal.alt', { object: data.object, n: label });
    imgBadge.replaceChildren(badge(kind));
    slider.setAttribute('aria-valuetext', t('reveal.valuetext', { n, N }));
    cap.replaceChildren(t('reveal.caption', { n: label, N }));
    if (label === 1 && frames[0]) cap.append('\u00a0· ', t('reveal.frame_meta', { wl: fmt(frames[0].wl_um, 3) }));
    snr.replaceChildren();
    if (st && st.snr != null) snr.append(t('reveal.snr_times', { snr: fmt(st.snr, 3), note: snrNote || (data.noise_calibration ? t('reveal.against_empty') : t('reveal.own_noise')) }), ' ', badge('derived'));

    const sp = specs.filter(s => s.n_frames_used <= n).pop();
    specBox.replaceChildren();
    if (!sp) { specBox.append(h('p', { class: 'muted' }, t('reveal.no_spectrum'))); return; }
    const svg = plot([{ x: sp.wl_um, y: sp.flux_mJy, err: sp.err, kind: 'points', colors: sp.wl_um.map(wlColor), color: 'var(--accent)' }],
      { xlim, ylim, xlabel: t('plot.wl'), ylabel: t('plot.flux'), height: compact ? 240 : 300,
        title: t('reveal.spec_title', { object: data.object, n: sp.n_frames_used }) });
    specBox.append(svg, h('p', { class: 'caption' }, t('reveal.spec_caption', { n: sp.n_frames_used, k: sp.wl_um.length }), ' ', badge('derived')));
  }

  // The pictures and empty-path tests behind every number on this card.
  const [evBtn, evPanel] = evidenceNum(t('reveal.evidence', { N }), evidenceFor(data));
  put(el,
    h('div', { class: 'reveal' + (compact ? ' compact' : '') + (specs.length ? '' : ' no-spec') },
      h('figure', { class: 'imgpanel' }, h('div', { class: 'imgwrap' }, img, imgBadge), cap, snr),
      specs.length ? h('figure', { class: 'specpanel' }, specBox) : null),
    h('div', { class: 'reveal-controls' },
      h('label', { for: slider.id }, t('reveal.slider')), slider, play),
    h('p', { class: 'small' }, evBtn), evPanel,
  );
  render();
  if (autoplay && !REDUCED_MOTION && N > 1) play.click();
  return { stop };
}

function evidenceFor(data) {
  return {
    frames: (data.frames || []).map(f => ({ ...f, date: f.date || (f.mjd ? new Date((f.mjd - 40587) * 864e5).toISOString().slice(0, 10) : null) })),
    nulls: data.null_snrs, nullsNote: t('ev.reveal_nulls'), method: 'science.html#stacking-along-an-orbit',
  };
}

function pad(v) {
  const f = v.filter(Number.isFinite), lo = Math.min(...f), hi = Math.max(...f), p = (hi - lo) * 0.06 || 0.5;
  return [lo - p, hi + p];
}
