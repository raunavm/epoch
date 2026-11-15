// "Follow a moving object": shift each picture back along a straight-line motion to a reference time, then add.
// Pure functions on the live grid (north up, east left, SCALE_ARCSEC per pixel; x = column, y = row).
import { SCALE_ARCSEC } from './grid.js';
import { madSigma } from './change.js';

const RAD = Math.PI / 180;

// Where a mover with `rate` (″ per hour) towards position angle `pa` (degrees east of north) is at time `mjd`,
// relative to where it was at `mjd0`, in grid pixels [dx, dy]. East is left (−x), north is up (−y).
export function trackShift(rate, pa, mjd, mjd0) {
  const d = rate * (mjd - mjd0) * 24 / SCALE_ARCSEC;
  return [-d * Math.sin(pa * RAD), -d * Math.cos(pa * RAD)];
}

// Rate (″/h) and position angle (degrees east of north, 0–360) from the object marked at two grid
// positions a = {x, y, mjd} and b = {x, y, mjd} in two pictures.
export function motionFromMarks(a, b) {
  const hrs = (b.mjd - a.mjd) * 24;
  if (!(Math.abs(hrs) > 0)) return null;
  const east = -(b.x - a.x) * SCALE_ARCSEC / hrs, north = -(b.y - a.y) * SCALE_ARCSEC / hrs;
  return { rate: Math.hypot(east, north), pa: ((Math.atan2(east, north) / RAD) % 360 + 360) % 360 };
}

// out(x, y) = src(x + dx, y + dy): moves the content by (−dx, −dy). Whole-pixel copy when the shift is within
// 0.05 px of whole pixels, bilinear otherwise; pixels with no data (or sampled from outside) are NaN.
export function shiftImage(src, S, dx, dy) {
  const out = new Float32Array(S * S).fill(NaN);
  const rx = Math.round(dx), ry = Math.round(dy);
  if (Math.abs(dx - rx) < 0.05 && Math.abs(dy - ry) < 0.05) {
    for (let y = Math.max(0, -ry); y < Math.min(S, S - ry); y++) {
      const a = (y + ry) * S;
      for (let x = Math.max(0, -rx); x < Math.min(S, S - rx); x++) out[y * S + x] = src[a + x + rx];
    }
    return out;
  }
  const ix = Math.floor(dx), iy = Math.floor(dy), fx = dx - ix, fy = dy - iy;
  for (let y = Math.max(0, -iy); y < Math.min(S, S - iy - 1); y++) {
    const r0 = (y + iy) * S, r1 = r0 + S;
    for (let x = Math.max(0, -ix); x < Math.min(S, S - ix - 1); x++) {
      const X = x + ix;
      out[y * S + x] = (1 - fy) * ((1 - fx) * src[r0 + X] + fx * src[r0 + X + 1]) + fy * ((1 - fx) * src[r1 + X] + fx * src[r1 + X + 1]);
    }
  }
  return out;
}

// Add S/N maps: each value clipped to ±clip (so one spike cannot dominate), sum / sqrt(number of pictures with data),
// optionally smoothed with a 3x3 [1 2 1] kernel (about the blur size, which favours point sources), then divided by
// the map's own robust scatter (so the colour scale is in units of the stack's measured noise). Returns {z, scale}.
export function stackSN(maps, S, { clip = Infinity, smooth = false } = {}) {
  const sum = new Float32Array(S * S), n = new Uint16Array(S * S);
  for (const m of maps) for (let p = 0; p < S * S; p++) if (Number.isFinite(m[p])) { sum[p] += Math.max(-clip, Math.min(clip, m[p])); n[p]++; }
  let z = new Float32Array(S * S);
  for (let p = 0; p < S * S; p++) z[p] = n[p] ? sum[p] / Math.sqrt(n[p]) : NaN;
  if (smooth) z = smooth3(z, S);
  const scale = madSigma(z);
  if (scale > 0) for (let p = 0; p < S * S; p++) z[p] /= scale;
  return { z, scale };
}

// NaN-aware 3x3 smoothing with weights [1 2 1] x [1 2 1].
export function smooth3(z, S) {
  const out = new Float32Array(S * S).fill(NaN), w = [1, 2, 1];
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
    if (!Number.isFinite(z[y * S + x])) continue;
    let s = 0, sw = 0;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      const X = x + dx, Y = y + dy, v = X >= 0 && Y >= 0 && X < S && Y < S ? z[Y * S + X] : NaN;
      if (Number.isFinite(v)) { s += w[dx + 1] * w[dy + 1] * v; sw += w[dx + 1] * w[dy + 1]; }
    }
    out[y * S + x] = s / sw;
  }
  return out;
}

// The `spanH`-hour stretch of time holding the most pictures: {t0 (middle), lo, hi, n}. mjds need not be sorted.
export function busiestSpan(mjds, spanH) {
  const t = [...mjds].sort((a, b) => a - b), w = spanH / 24;
  let best = { n: 0 };
  for (let i = 0, j = 0; i < t.length; i++) {
    while (j < t.length && t[j] - t[i] <= w) j++;
    if (j - i > best.n) best = { n: j - i, lo: t[i], hi: t[j - 1] };
  }
  return best.n ? { ...best, t0: (best.lo + best.hi) / 2 } : null;
}

// Brightest pixel of a map within `r` pixels of the centre: {x, y, v}.
export function peakNear(z, S, r) {
  const c = (S - 1) / 2; let best = { v: -Infinity };
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
    const v = z[y * S + x];
    if (Number.isFinite(v) && v > best.v && Math.hypot(x - c, y - c) <= r) best = { x, y, v };
  }
  return best.v > -Infinity ? best : null;
}
