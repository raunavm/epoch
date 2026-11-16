// Contrast control and "Save film" for a picture viewer. The film is the pictures only (no markers), saved as an
// animated PNG at the viewer's playback speed, encoded in the browser.
import { h, t } from '../core.js';

const FILM_PX = 320;

// Contrast slider: applies a CSS filter to the pictures inside `target` and reports the value through getK().
export function contrastControl(target) {
  let k = 1;
  const id = `ct-${Math.random().toString(36).slice(2, 8)}`;
  const input = h('input', { type: 'range', id, min: 0.5, max: 3, step: 0.1, value: 1, class: 'contrast' });
  input.addEventListener('input', () => { k = +input.value; target.style.setProperty('--contrast', k); });
  const el = h('label', { for: id, class: 'contrast-label' }, t('film.contrast'), ' ', input);
  return { el, getK: () => k };
}

// Animated PNG (APNG) encoder: RGBA frames, zlib-compressed with the browser's CompressionStream.
const CRC = Array.from({ length: 256 }, (_, n) => { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; return c >>> 0; });
const crc32 = b => { let c = 0xffffffff; for (const x of b) c = CRC[(c ^ x) & 255] ^ (c >>> 8); return (c ^ 0xffffffff) >>> 0; };
const u32 = v => [(v >>> 24) & 255, (v >>> 16) & 255, (v >>> 8) & 255, v & 255];
const u16 = v => [(v >>> 8) & 255, v & 255];
function chunk(type, data) {
  const td = new Uint8Array(4 + data.length);
  td.set([...type].map(c => c.charCodeAt(0))); td.set(data, 4);
  return [new Uint8Array(u32(data.length)), td, new Uint8Array(u32(crc32(td)))];
}
async function zlib(bytes) {
  return new Uint8Array(await new Response(new Blob([bytes]).stream().pipeThrough(new CompressionStream('deflate'))).arrayBuffer());
}
async function apng(frames, n, delayMs) {
  const parts = [new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10])];
  parts.push(...chunk('IHDR', new Uint8Array([...u32(n), ...u32(n), 8, 6, 0, 0, 0])));
  parts.push(...chunk('acTL', new Uint8Array([...u32(frames.length), ...u32(0)])));
  let seq = 0;
  for (let f = 0; f < frames.length; f++) {
    const raw = new Uint8Array(n * (4 * n + 1)); // one filter byte (0) per row
    for (let y = 0; y < n; y++) raw.set(frames[f].subarray(y * n * 4, (y + 1) * n * 4), y * (4 * n + 1) + 1);
    const z = await zlib(raw);
    parts.push(...chunk('fcTL', new Uint8Array([...u32(seq++), ...u32(n), ...u32(n), ...u32(0), ...u32(0), ...u16(delayMs), ...u16(1000), 0, 0])));
    if (f === 0) parts.push(...chunk('IDAT', z));
    else { const d = new Uint8Array(4 + z.length); d.set(u32(seq++)); d.set(z, 4); parts.push(...chunk('fdAT', d)); }
  }
  parts.push(...chunk('IEND', new Uint8Array(0)));
  return new Blob(parts, { type: 'image/png' });
}

// sources: array of async () => CanvasImageSource, one per picture, in order.
async function record(sources, fps, k, onProgress) {
  if (!('CompressionStream' in window)) throw new Error('unsupported');
  const cv = h('canvas', { width: FILM_PX, height: FILM_PX });
  const ctx = cv.getContext('2d', { willReadFrequently: true });
  ctx.imageSmoothingEnabled = false;
  const frames = [];
  for (let i = 0; i < sources.length; i++) {
    const img = await sources[i]();
    ctx.filter = 'none'; ctx.fillStyle = '#000'; ctx.fillRect(0, 0, FILM_PX, FILM_PX);
    ctx.filter = `contrast(${k})`; ctx.drawImage(img, 0, 0, FILM_PX, FILM_PX);
    frames.push(ctx.getImageData(0, 0, FILM_PX, FILM_PX).data);
    onProgress(i + 1, sources.length);
  }
  return { blob: await apng(frames, FILM_PX, Math.round(1000 / fps)), ext: 'png' };
}

// Button that records the current list of pictures. getJob() returns {sources, fps, k, name}.
export function filmButton(getJob) {
  const status = h('span', { class: 'muted small', role: 'status' });
  const btn = h('button', { type: 'button', class: 'btn small' }, t('film.save'));
  btn.addEventListener('click', async () => {
    const { sources, fps, k, name } = getJob();
    if (!sources.length) return;
    btn.disabled = true;
    try {
      const { blob, ext } = await record(sources, fps, k, (i, n) => { status.textContent = t('film.saving', { i, n }); });
      const a = h('a', { href: URL.createObjectURL(blob), download: `${name}.${ext}` });
      document.body.append(a); a.click(); a.remove();
      status.textContent = '';
    } catch { status.textContent = t('film.unsupported'); }
    btn.disabled = false;
  });
  return [btn, ' ', status];
}

// Helpers to turn a picture into a drawable source.
export const fromURL = url => () => new Promise((res, rej) => { const im = new Image(); im.onload = () => res(im); im.onerror = rej; im.src = url; });
export const fromRGBA = (rgba, n) => async () => { const c = h('canvas', { width: n, height: n }); c.getContext('2d').putImageData(new ImageData(rgba, n, n), 0, 0); return c; };
