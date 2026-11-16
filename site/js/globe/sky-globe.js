// SPHEREx all-sky globe with the five challenge targets and the kept brown dwarfs as clickable markers.
// Rendering and projection ported from Tracklet (skygl.js, sphere.js). The globe computes no numbers about any object.
import { t, getText } from '../core.js';
import { TARGETS } from '../live/targets.js';
import { cameraBasis, projectUnit, unitFromRaDec, unprojectUnit } from './sphere.js';
import { createSkyGL } from './skygl.js';

const SKY = 'data/sky';
const DEG = Math.PI / 180;
const SPIN_DEG_S = 2.4, SPIN_DELAY_MS = 2500, HIT_PX = 10, KEY_STEP = 5;

const liveHref = (ra, dec) => `index.html?ra=${ra}&dec=${dec}&fov=2`;

// Minimal CSV reader (quoted fields, doubled quotes).
function parseCSV(text) {
  const rows = []; let row = [], f = '', q = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (q) {
      if (c === '"' && text[i + 1] === '"') { f += '"'; i++; } else if (c === '"') q = false; else f += c;
    } else if (c === '"') q = true;
    else if (c === ',') { row.push(f); f = ''; }
    else if (c === '\n' || c === '\r') {
      if (c === '\r' && text[i + 1] === '\n') i++;
      row.push(f); f = ''; if (row.some(x => x !== '')) rows.push(row); row = [];
    } else f += c;
  }
  row.push(f); if (row.some(x => x !== '')) rows.push(row);
  const [head, ...body] = rows;
  return body.map(r => Object.fromEntries(head.map((k, i) => [k, r[i] ?? ''])));
}

export async function loadEpochMarkers() {
  const markers = TARGETS.map(tg => ({ ra: tg.ra, dec: tg.dec, label: tg.name, kind: 'target', href: liveHref(tg.ra, tg.dec) }));
  const csv = await getText('products/epoch_methane_candidates.csv');
  if (csv) {
    for (const r of parseCSV(csv)) {
      if (r.vet_verdict !== 'KEEP') continue;
      const ra = Number(r.ra_deg), dec = Number(r.dec_deg);
      if (r.ra_deg === '' || r.dec_deg === '' || !Number.isFinite(ra) || !Number.isFinite(dec)) continue;
      markers.push({ ra, dec, label: r.name, kind: 'browndwarf', href: liveHref(ra, dec) });
    }
  }
  return markers;
}

// Decode the all-sky sheet into grey and observed planes, checked against the manifest size.
async function loadSheet() {
  const meta = await (await fetch(`${SKY}/index.json`)).json();
  const { file, geometry: geom } = meta;
  const resp = await fetch(`${SKY}/${file}`);
  if (!resp.ok) throw new Error(`${file}: HTTP ${resp.status}`);
  const bmp = await createImageBitmap(await resp.blob());
  const iw = bmp.width, ih = bmp.height;
  if (iw !== geom.image_width || ih !== geom.image_height) throw new Error(`${file}: unexpected size ${iw}x${ih}`);
  const c = document.createElement('canvas'); c.width = iw; c.height = ih;
  const cx = c.getContext('2d', { willReadFrequently: true });
  cx.drawImage(bmp, 0, 0); bmp.close?.();
  const rgba = cx.getImageData(0, 0, iw, ih).data;
  const n = iw * ih, grey = new Uint8Array(n), alpha = new Uint8Array(n);
  for (let i = 0; i < n; i++) { grey[i] = rgba[i * 4]; alpha[i] = rgba[i * 4 + 3]; }
  return { geom, grey, alpha };
}

const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
const fmtDeg = (v, sign) => (sign && v >= 0 ? '+' : '') + v.toFixed(3);

export async function mountSkyGlobe(container, { markers = [], onPick = null, labels = {} } = {}) {
  const L = {
    aria: t('globe.aria'), hint: t('globe.hint'), loading: t('globe.loading'),
    legendTargets: t('globe.legend_targets'), legendBd: t('globe.legend_bd'), list: t('globe.list'), ...labels,
  };
  const pick = onPick || ((ra, dec) => { location.href = liveHref(+ra.toFixed(5), +dec.toFixed(5)); });

  // DOM
  const root = el('div', 'sky-globe');
  const stage = el('div', 'sky-globe-stage');
  const canvas = el('canvas', 'sky-globe-canvas');
  canvas.tabIndex = 0; canvas.setAttribute('role', 'application'); canvas.setAttribute('aria-label', L.aria);
  const tip = el('div', 'sky-globe-tip'); tip.hidden = true; tip.setAttribute('aria-hidden', 'true');
  stage.append(canvas, tip);
  const readout = el('p', 'sky-globe-readout', L.loading); readout.setAttribute('aria-live', 'polite');
  const legend = el('p', 'sky-globe-legend');
  const nT = markers.filter(m => m.kind === 'target').length, nB = markers.filter(m => m.kind === 'browndwarf').length;
  if (nT) legend.append(el('span', 'sky-globe-key key-target'), el('span', null, L.legendTargets));
  if (nB) legend.append(el('span', 'sky-globe-key key-bd'), el('span', null, L.legendBd));
  const list = el('ul', 'sr-only'); list.setAttribute('aria-label', L.list);
  for (const m of markers) { const li = el('li'); const a = el('a', null, m.label); a.href = m.href; li.append(a); list.append(li); }
  root.append(stage, readout, legend, list);
  container.append(root);

  let skyGL = null;
  try {
    const sheet = await loadSheet();
    skyGL = await createSkyGL(sheet);
    if (!skyGL) throw new Error('WebGL2 unavailable');
  } catch (err) {
    root.remove();
    throw err;
  }
  const glc = skyGL.canvas; glc.className = 'sky-globe-gl'; glc.setAttribute('aria-hidden', 'true');
  stage.prepend(glc);
  readout.textContent = L.hint;

  const ctx = canvas.getContext('2d');
  const css = getComputedStyle(root);
  const color = { target: css.getPropertyValue('--accent').trim() || '#1c67e3', browndwarf: css.getPropertyValue('--hds-red').trim() || '#f64137' };
  const pts = markers.map(m => ({ ...m, v: unitFromRaDec(m.ra, m.dec) }));
  const grid = [];
  for (let ra = 0; ra < 360; ra += 30) { const l = []; for (let d = -90; d <= 90; d += 3) l.push(unitFromRaDec(ra, d)); grid.push(l); }
  for (let d = -60; d <= 60; d += 30) { const l = []; for (let ra = 0; ra <= 360; ra += 3) l.push(unitFromRaDec(ra, d)); grid.push(l); }

  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  let w = 1, h = 1, dpr = 1, radius = 1, cx = 0, cy = 0;
  let lon0 = 270, lat0 = 20, hovered = null, pointerOver = false, lastInput = performance.now();
  let dragging = false, moved = 0, sx = 0, sy = 0, raf = 0, prevT = 0, dead = false;

  function size() {
    dpr = Math.min(2, devicePixelRatio || 1);
    w = Math.max(1, Math.round(stage.clientWidth)); h = Math.max(1, Math.round(stage.clientHeight));
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
    radius = Math.min(w, h) * 0.47; cx = w / 2; cy = h / 2;
    draw();
  }
  const project = (v, b) => { const p = projectUnit(b, v); return { x: cx + radius * p.u, y: cy - radius * p.w, front: p.depth > 0 }; };

  function draw() {
    const b = cameraBasis(lon0, lat0);
    const W = canvas.width, H = canvas.height, side = Math.round(2 * radius * dpr);
    const x = Math.round(cx * dpr - side / 2), y = Math.round(cy * dpr - side / 2);
    if (!skyGL.render(b, { x, y, side, width: W, height: H, cx: cx * dpr, cy: cy * dpr, r: radius * dpr })) {
      readout.textContent = t('common.unavailable');
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    ctx.lineWidth = 0.6; ctx.strokeStyle = 'rgba(255,255,255,0.18)';
    for (const line of grid) {
      ctx.beginPath(); let pen = false;
      for (const v of line) { const p = project(v, b); if (!p.front) { pen = false; continue; } pen ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y); pen = true; }
      ctx.stroke();
    }
    for (const m of pts) {
      const p = project(m.v, b); m.sx = p.x; m.sy = p.y; m.front = p.front;
      if (!p.front) continue;
      const big = m === hovered, r = (m.kind === 'target' ? 6 : 4.5) + (big ? 2 : 0);
      ctx.beginPath();
      if (m.kind === 'target') ctx.arc(p.x, p.y, r, 0, 2 * Math.PI);
      else { ctx.moveTo(p.x, p.y - r - 1); ctx.lineTo(p.x + r + 1, p.y); ctx.lineTo(p.x, p.y + r + 1); ctx.lineTo(p.x - r - 1, p.y); ctx.closePath(); }
      ctx.fillStyle = color[m.kind]; ctx.fill();
      ctx.lineWidth = 1.5; ctx.strokeStyle = '#fff'; ctx.stroke();
    }
  }

  function hit(px, py) {
    let best = null, bd = HIT_PX;
    for (const m of pts) { if (!m.front) continue; const d = Math.hypot(m.sx - px, m.sy - py); if (d <= bd) { bd = d; best = m; } }
    return best;
  }
  const local = e => { const r = canvas.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };

  function hover(px, py) {
    const m = hit(px, py);
    if (m !== hovered) { hovered = m; draw(); }
    canvas.style.cursor = m ? 'pointer' : (dragging ? 'grabbing' : 'grab');
    if (m) {
      tip.textContent = m.label; tip.hidden = false;
      tip.style.left = `${Math.min(m.sx + 10, w - tip.offsetWidth - 2)}px`; tip.style.top = `${Math.max(0, m.sy - 28)}px`;
    } else tip.hidden = true;
    const s = unprojectUnit(cameraBasis(lon0, lat0), (px - cx) / radius, -(py - cy) / radius);
    readout.textContent = s ? t('globe.readout', { ra: fmtDeg(s[0]), dec: fmtDeg(s[1], true) }) : L.hint;
  }

  function rotate(dLon, dLat) {
    lon0 = ((lon0 + dLon) % 360 + 360) % 360;
    lat0 = Math.max(-89, Math.min(89, lat0 + dLat));
    draw();
  }

  const onDown = e => { dragging = true; moved = 0; [sx, sy] = local(e); canvas.setPointerCapture(e.pointerId); lastInput = performance.now(); };
  const onMove = e => {
    const [px, py] = local(e); lastInput = performance.now();
    if (dragging) {
      const dx = px - sx, dy = py - sy; moved += Math.abs(dx) + Math.abs(dy); sx = px; sy = py;
      if (moved > 4) { tip.hidden = true; rotate(-dx / radius / DEG, dy / radius / DEG); return; }
    }
    hover(px, py);
  };
  const onUp = e => {
    if (!dragging) return; dragging = false; lastInput = performance.now();
    if (moved > 4) return;
    const [px, py] = local(e), m = hit(px, py);
    if (m) { location.href = m.href; return; }
    const s = unprojectUnit(cameraBasis(lon0, lat0), (px - cx) / radius, -(py - cy) / radius);
    if (s) pick(s[0], s[1]);
  };
  const onEnter = () => { pointerOver = true; };
  const onLeave = () => { pointerOver = false; hovered = null; tip.hidden = true; readout.textContent = L.hint; draw(); };
  const onKey = e => {
    const k = { ArrowLeft: [-KEY_STEP, 0], ArrowRight: [KEY_STEP, 0], ArrowUp: [0, KEY_STEP], ArrowDown: [0, -KEY_STEP] }[e.key];
    if (!k) return;
    e.preventDefault(); lastInput = performance.now(); rotate(...k);
  };
  canvas.addEventListener('pointerdown', onDown);
  canvas.addEventListener('pointermove', onMove);
  canvas.addEventListener('pointerup', onUp);
  canvas.addEventListener('pointercancel', () => { dragging = false; });
  canvas.addEventListener('pointerenter', onEnter);
  canvas.addEventListener('pointerleave', onLeave);
  canvas.addEventListener('keydown', onKey);

  function tick(now) {
    if (dead) return;
    const dt = prevT ? (now - prevT) / 1000 : 0; prevT = now;
    if (!reduced.matches && !dragging && !pointerOver && !document.hidden && document.activeElement !== canvas
        && now - lastInput > SPIN_DELAY_MS && dt > 0 && dt < 0.5) rotate(-SPIN_DEG_S * dt, 0);
    raf = requestAnimationFrame(tick);
  }
  raf = requestAnimationFrame(tick);

  const ro = new ResizeObserver(size);
  ro.observe(stage);
  size();

  return {
    resize: size,
    destroy() {
      dead = true; cancelAnimationFrame(raf); ro.disconnect(); skyGL.destroy(); root.remove();
    },
  };
}
