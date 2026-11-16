// Common north-up grid centred on the target, resampling, aperture photometry, colour maps. Pure functions.
import { deproject, waveAt } from './wcs.js';

export const SCALE_ARCSEC = 6.15;
export const gridSize = fovArcmin => Math.max(5, Math.round(fovArcmin * 60 / SCALE_ARCSEC));

// Sky position of output pixel (i = column, j = row); north up, east left.
export function gridToWorld(i, j, S, ra, dec) {
  const c = (S - 1) / 2, s = SCALE_ARCSEC / 3600;
  return deproject((c - i) * s, (c - j) * s, ra, dec);
}

// Source rows [y0, y1] needed to cover the grid in a frame (grid corners + 2 px padding).
export function rowsNeeded(wcs, S, ra, dec) {
  const ys = [];
  for (const [i, j] of [[0, 0], [S - 1, 0], [0, S - 1], [S - 1, S - 1]]) {
    const p = wcs.world2pix(...gridToWorld(i, j, S, ra, dec));
    if (p) ys.push(p[1]);
  }
  return [Math.floor(Math.min(...ys)) - 2, Math.ceil(Math.max(...ys)) + 2];
}

// Bilinear resample of a row block {y0, y1, nx, data} onto the grid. Also returns the per-pixel wavelength map.
export function resample(wcs, rows, waveTab, S, ra, dec) {
  const v = new Float32Array(S * S).fill(NaN), lam = new Float32Array(S * S).fill(NaN);
  const { y0, y1, nx, data } = rows;
  for (let j = 0; j < S; j++) for (let i = 0; i < S; i++) {
    const p = wcs.world2pix(...gridToWorld(i, j, S, ra, dec));
    if (!p) continue;
    const [x, y] = p, xi = Math.floor(x), yi = Math.floor(y), fx = x - xi, fy = y - yi;
    if (xi < 0 || xi + 1 >= nx || yi < y0 || yi + 1 > y1) continue;
    const r0 = (yi - y0) * nx + xi, r1 = r0 + nx;
    v[j * S + i] = (1 - fy) * ((1 - fx) * data[r0] + fx * data[r0 + 1]) + fy * ((1 - fx) * data[r1] + fx * data[r1 + 1]);
    if (waveTab) lam[j * S + i] = waveAt(waveTab, x, y)[0];
  }
  return { v, lam };
}

// 3x3 aperture sum minus 9 x median of the annulus 3 <= r <= 5 px (grid pixels), in units of v.
// err: annulus robust scatter x sqrt(9 + 81 (pi/2) / n_annulus); ignores the pixel-to-pixel correlation from resampling.
// Aperture corrections for the 3 x 3 quick look, per detector D1-D6: IRSA's published r = 5 aperture recipe divided by
// our 3 x 3 sum on 5 isolated stars (data/findings/e31s.json, ql_fov2_D<n>_irsa_r5_correction for a star on a pixel corner,
// ql_fov4_D<n>_irsa_r5_correction for a star centred on a pixel; per-picture scatter 3-4%).
export const APCORR = { corner: [1.106, 1.118, 1.142, 1.226, 1.209, 1.254], centred: [1.051, 1.066, 1.079, 1.129, 1.126, 1.144] };
// The target sits on a pixel corner when the grid has an even number of pixels, so a click on one of the four central
// pixels gets the corner value; every other click gets the centred value (a star between pixels then reads up to ~10% low).
export function apCorr(S, i, j, det) {
  const c = (S - 1) / 2, corner = S % 2 === 0 && Math.abs(i - c) === 0.5 && Math.abs(j - c) === 0.5;
  return APCORR[corner ? 'corner' : 'centred'][det - 1] ?? 1;
}

export function aperture(v, S, i, j) {
  let sum = 0, n = 0; const ann = [];
  for (let y = j - 5; y <= j + 5; y++) for (let x = i - 5; x <= i + 5; x++) {
    if (x < 0 || y < 0 || x >= S || y >= S) continue;
    const val = v[y * S + x], r = Math.hypot(x - i, y - j);
    if (!Number.isFinite(val)) continue;
    if (Math.abs(x - i) <= 1 && Math.abs(y - j) <= 1) { sum += val; n++; } else if (r >= 3 && r <= 5) ann.push(val);
  }
  if (n < 9 || ann.length < 5) return { flux: NaN, err: NaN };
  const med = a => { const s = [...a].sort((p, q) => p - q), k = s.length; return k % 2 ? s[k >> 1] : (s[k / 2 - 1] + s[k / 2]) / 2; };
  const m = med(ann), sd = 1.4826 * med(ann.map(x => Math.abs(x - m)));
  return { flux: sum - 9 * m, err: sd * Math.sqrt(9 + 81 * (Math.PI / 2) / ann.length) };
}
// MJy/sr summed over pixels -> mJy, with OMEGA in arcsec^2.
export const toMJy = (sumMJySr, omegaArcsec2) => sumMJySr * omegaArcsec2 / 206264.806 ** 2 * 1e9;

// Per-image asinh stretch -> RGBA bytes (grey). NaN -> dark blue-grey.
export function asinhRGBA(v) {
  const f = []; for (const x of v) if (Number.isFinite(x)) f.push(x);
  f.sort((a, b) => a - b);
  const q = t => f[Math.min(f.length - 1, Math.floor(t * (f.length - 1)))] ?? 0;
  const lo = q(0.5), sd = Math.max(1e-12, (q(0.84) - q(0.16)) / 2), hi = Math.max(q(0.995) - lo, 3 * sd);
  const k = Math.asinh(hi / sd);
  const out = new Uint8ClampedArray(v.length * 4);
  v.forEach((x, p) => {
    if (!Number.isFinite(x)) { out.set([30, 34, 48, 255], 4 * p); return; }
    const g = 255 * Math.min(1, Math.max(0, Math.asinh((x - lo) / sd) / k));
    out.set([g, g, g, 255], 4 * p);
  });
  return out;
}
// Diverging map for significance maps, clipped at +-clip: blue (#56B4E9) <- black -> orange (#E69F00).
export function divergingRGBA(v, clip = 8) {
  const out = new Uint8ClampedArray(v.length * 4);
  v.forEach((x, p) => {
    if (!Number.isFinite(x)) { out.set([30, 34, 48, 255], 4 * p); return; }
    const t = Math.min(1, Math.abs(x) / clip), c = x < 0 ? [86, 180, 233] : [230, 159, 0];
    out.set([c[0] * t, c[1] * t, c[2] * t, 255], 4 * p);
  });
  return out;
}

// Sexagesimal or decimal parsing. RA: "08h55m10.831s", "08 55 10.831", "133.795" (deg). Dec: "-07d14m42.53s", "-7 14 42.53", "-7.245".
export function parseRA(s) {
  s = String(s).trim();
  const parts = s.replace(/[hHmMsS:]/g, ' ').trim().split(/\s+/).map(Number);
  if (parts.some(Number.isNaN) || !parts.length) return NaN;
  if (parts.length === 1 && !/[hH:]/.test(s)) return ((parts[0] % 360) + 360) % 360;
  const [h, m = 0, sec = 0] = parts;
  return (h + m / 60 + sec / 3600) * 15;
}
export function parseDec(s) {
  s = String(s).trim();
  const neg = /^-|^−/.test(s);
  const parts = s.replace(/^[+\-−]/, '').replace(/[dD°'′"″:mMsS]/g, ' ').trim().split(/\s+/).map(Number);
  if (parts.some(Number.isNaN) || !parts.length) return NaN;
  const [d, m = 0, sec = 0] = parts, v = d + m / 60 + sec / 3600;
  return v > 90 ? NaN : neg ? -v : v;
}
