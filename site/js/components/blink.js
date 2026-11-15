// Finding 1: raw blink vs wavelength-corrected change images, with labelled changes.
import { h, put, t, asset, badge, fmt, wlColor, CATS, REDUCED_MOTION } from '../core.js';
import { s } from '../plot.js';

const SPEEDS = [0.5, 1, 2, 4];

export function mountBlink(el, field, { compact = false } = {}) {
  const frames = field.frames || [];
  if (!frames.length) return null;
  const size = field.size_px || 64;
  const st = { i: 0, corrected: false, playing: !REDUCED_MOTION, fps: 2, mode: 'blink', a: 0, b: Math.min(1, frames.length - 1) };
  let timer = null;
  const uid = Math.random().toString(36).slice(2, 7);

  // Raw / corrected switch (the main control).
  const sw = ['raw', 'corrected'].map(k => h('button', { type: 'button', class: 'seg', 'aria-pressed': String(k === 'raw'), onclick: () => { st.corrected = k === 'corrected'; render(); } }, t(`tm.${k}`)));
  const swExplain = h('p', { class: 'hint' });

  const viewA = view(), viewB = view();
  const playBtn = h('button', { type: 'button', class: 'btn', onclick: () => { st.playing = !st.playing; schedule(); render(); } });
  const prev = h('button', { type: 'button', class: 'btn', 'aria-label': t('tm.prev'), onclick: () => step(-1) }, '←');
  const next = h('button', { type: 'button', class: 'btn', 'aria-label': t('tm.next'), onclick: () => step(1) }, '→');
  const speed = h('select', { id: `sp-${uid}`, onchange: e => { st.fps = +e.target.value; schedule(); } },
    SPEEDS.map(v => h('option', { value: v, selected: v === st.fps }, t('tm.fps', { v }))));
  const scrub = h('input', { type: 'range', class: 'scrub', min: 0, max: frames.length - 1, value: 0, id: `sc-${uid}`, oninput: e => { st.i = +e.target.value; st.playing = false; schedule(); render(); } });
  scrub.style.setProperty('--track', `linear-gradient(90deg, ${frames.map((f, k) => `${wlColor(f.wl_um)} ${k / frames.length * 100}% ${(k + 1) / frames.length * 100}%`).join(', ')})`);
  const labelList = h('ul', { class: 'labels' });

  const modeBtns = ['blink', 'side'].map(k => h('button', { type: 'button', class: 'seg', 'aria-pressed': String(k === 'blink'), onclick: () => { st.mode = k; st.playing = k === 'blink' && st.playing; schedule(); render(); } }, t(`tm.mode_${k}`)));
  const selA = frameSelect('a'), selB = frameSelect('b');

  function frameSelect(which) {
    return h('select', { 'aria-label': t(which === 'a' ? 'tm.frame_a' : 'tm.frame_b'), onchange: e => { st[which] = +e.target.value; render(); } },
      frames.map((f, k) => h('option', { value: k }, `${f.date} · ${fmt(f.wl_um, 3)} µm`)));
  }

  function view() {
    const img = h('img', { class: 'stamp', width: 320, height: 320, decoding: 'async', alt: '' });
    const ov = s('svg', { viewBox: `0 0 ${size} ${size}`, class: 'overlay', 'aria-hidden': 'true' });
    const meta = h('p', { class: 'frame-meta' });
    return { img, ov, meta, root: h('figure', { class: 'tm-view' }, h('div', { class: 'imgwrap' }, img, ov), meta) };
  }

  function fill(v, f) {
    const src = st.corrected ? f.corr_png : f.img_png;
    v.img.src = src ? asset(src) : '';
    v.img.alt = t(st.corrected ? 'tm.alt_corr' : 'tm.alt_raw', { name: field.name, date: f.date, wl: fmt(f.wl_um, 3) });
    v.meta.replaceChildren(
      h('span', { class: 'swatch', style: `background:${wlColor(f.wl_um)}`, 'aria-hidden': 'true' }),
      h('strong', {}, f.date), ` · ${t('tm.wl')} ${fmt(f.wl_um, 3)} µm · `, badge(st.corrected ? 'resid' : 'data'));
    v.ov.replaceChildren();
    if (st.corrected) for (const L of f.labels || []) v.ov.append(marker(L));
  }

  function marker(L) {
    const c = CATS[L.label] || CATS.unexplained, x = L.x + 0.5, y = L.y + 0.5, r = size / 18;
    const attrs = { fill: 'none', stroke: c.color, 'stroke-width': size / 110 };
    let m;
    if (c.shape === 'circle') m = s('circle', { cx: x, cy: y, r, ...attrs });
    else if (c.shape === 'square') m = s('rect', { x: x - r, y: y - r, width: 2 * r, height: 2 * r, ...attrs });
    else if (c.shape === 'diamond') m = s('path', { d: `M${x},${y - r}L${x + r},${y}L${x},${y + r}L${x - r},${y}Z`, ...attrs });
    else m = s('path', { d: `M${x},${y - r}L${x + r},${y + r * 0.8}L${x - r},${y + r * 0.8}Z`, ...attrs });
    return m;
  }

  function step(d) { st.i = (st.i + d + frames.length) % frames.length; st.playing = false; schedule(); render(); }
  function schedule() {
    clearInterval(timer);
    if (st.playing && st.mode === 'blink') timer = setInterval(() => { st.i = (st.i + 1) % frames.length; render(); }, 1000 / st.fps);
  }

  function render() {
    sw.forEach((b, k) => b.setAttribute('aria-pressed', String((k === 1) === st.corrected)));
    swExplain.textContent = t(st.corrected ? 'tm.corrected_explain' : 'tm.raw_explain');
    modeBtns.forEach((b, k) => b.setAttribute('aria-pressed', String((k === 1) === (st.mode === 'side'))));
    playBtn.textContent = t(st.playing ? 'tm.pause' : 'tm.play');
    const side = st.mode === 'side';
    viewB.root.hidden = !side; blinkCtl.hidden = side; sideCtl.hidden = !side;
    const fa = side ? frames[st.a] : frames[st.i];
    fill(viewA, fa);
    if (side) fill(viewB, frames[st.b]);
    scrub.value = st.i;
    scrub.setAttribute('aria-valuetext', `${frames[st.i].date}, ${fmt(frames[st.i].wl_um, 3)} µm`);
    selA.value = st.a; selB.value = st.b;
    const nxt = frames[(st.i + 1) % frames.length];
    new Image().src = asset(st.corrected ? nxt.corr_png : nxt.img_png); // preload one ahead only

    labelList.replaceChildren();
    const shown = side ? [frames[st.a], frames[st.b]] : [fa];
    if (st.corrected) {
      const Ls = shown.flatMap(f => f.labels || []);
      if (!Ls.length) labelList.append(h('li', { class: 'muted' }, t('tm.no_labels')));
      for (const L of Ls) {
        const c = CATS[L.label] || CATS.unexplained;
        labelList.append(h('li', {}, h('span', { class: `key key-${c.shape}`, style: `--c:${c.color}`, 'aria-hidden': 'true' }),
          h('strong', {}, t(c.key)), ` — ${L.name || t('tm.unnamed')}`,
          L.confidence != null ? h('span', { class: 'muted' }, ` · ${t('tm.confidence', { p: fmt(100 * L.confidence, 2) })} `) : null,
          badge('derived')));
      }
    }
  }

  const blinkCtl = h('div', { class: 'controls' }, prev, playBtn, next,
    h('label', { for: speed.id }, t('tm.speed')), speed);
  const sideCtl = h('div', { class: 'controls' }, selA, selB);
  const legend = h('ul', { class: 'legend-row', 'aria-label': t('tm.legend') },
    Object.values(CATS).map(c => h('li', {}, h('span', { class: `key key-${c.shape}`, style: `--c:${c.color}`, 'aria-hidden': 'true' }), t(c.key))));

  put(el,
    h('div', { class: 'segmented big', role: 'group', 'aria-label': t('tm.switch_label') }, sw), swExplain,
    compact ? null : h('div', { class: 'segmented', role: 'group', 'aria-label': t('tm.mode_label') }, modeBtns),
    h('div', { class: 'tm-views' }, viewA.root, viewB.root),
    blinkCtl, sideCtl,
    h('div', { class: 'scrub-wrap' }, h('label', { for: scrub.id, class: 'sr-only' }, t('tm.timeline')), scrub,
      h('div', { class: 'scrub-ends muted' }, h('span', {}, frames[0].date), h('span', {}, t('tm.timeline_note')), h('span', {}, frames[frames.length - 1].date))),
    compact ? null : legend, labelList,
  );

  const onKey = e => {
    if (e.target.closest?.('input, select, textarea')) return;
    if (e.key === 'ArrowRight') { step(1); e.preventDefault(); }
    else if (e.key === 'ArrowLeft') { step(-1); e.preventDefault(); }
  };
  const keyTarget = compact ? el : document;
  keyTarget.addEventListener('keydown', onKey);
  schedule(); render();
  return { stop() { clearInterval(timer); st.playing = false; keyTarget.removeEventListener('keydown', onKey); } };
}
