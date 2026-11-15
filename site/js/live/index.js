// Sky-tile index of DR1 frames (same rules as scripts/build_tiles.py) and footprint search.
import { project } from './wcs.js';

export const band = dec => Math.min(44, Math.max(0, Math.floor((dec + 90) / 4)));
export const nra = b => Math.max(1, Math.floor(360 * Math.cos((-90 + 4 * b + 2) * Math.PI / 180) / 4));
export function tileOf(ra, dec) {
  const b = band(dec), n = nra(b);
  return [b, ((Math.floor((((ra % 360) + 360) % 360) / 360 * n)) % n + n) % n];
}

// Tile names ("b_r") of the 3x3 neighbourhood around (ra, dec).
export function neighbourTiles(ra, dec) {
  const out = new Set(), b0 = band(dec);
  const delta = 4 / Math.max(Math.cos(dec * Math.PI / 180), 0.05);
  for (let b = Math.max(0, b0 - 1); b <= Math.min(44, b0 + 1); b++) {
    const n = nra(b);
    if (2 * delta >= 360 || b === 0 || b === 44) { for (let r = 0; r < n; r++) out.add(`${b}_${r}`); continue; }
    const r1 = Math.floor((ra - delta) / 360 * n), r2 = Math.floor((ra + delta) / 360 * n);
    if (r2 - r1 + 1 >= n) { for (let r = 0; r < n; r++) out.add(`${b}_${r}`); continue; }
    for (let r = r1; r <= r2; r++) out.add(`${b}_${((r % n) + n) % n}`);
  }
  return [...out];
}

// Is (0,0) inside the polygon pts=[[x,y],...] with at least `margin` distance to every edge?
export function containsWithMargin(pts, margin) {
  let inside = false;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const [xi, yi] = pts[i], [xj, yj] = pts[j];
    if ((yi > 0) !== (yj > 0) && 0 < (xj - xi) * (0 - yi) / (yj - yi) + xi) inside = !inside;
  }
  if (!inside) return false;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const [x1, y1] = pts[j], [x2, y2] = pts[i], dx = x2 - x1, dy = y2 - y1;
    const t = Math.max(0, Math.min(1, -(x1 * dx + y1 * dy) / (dx * dx + dy * dy)));
    if (Math.hypot(x1 + t * dx, y1 + t * dy) < margin) return false;
  }
  return true;
}

// Footprint corners [ra0,dec0,...,ra3,dec3] contain the target with margin (deg)?
export function footprintContains(c, ra, dec, marginDeg) {
  const pts = [];
  for (let k = 0; k < 4; k++) { const p = project(c[2 * k], c[2 * k + 1], ra, dec); if (!p) return false; pts.push(p); }
  return containsWithMargin(pts, marginDeg);
}

export const s3Key = (release, obsId, proc, d) =>
  `${release}/level2/${obsId.slice(0, 10)}/${proc}/${d}/level2_${obsId}D${d}_spx_${proc}.fits`;

// Frames whose footprint contains (ra, dec) with a margin of fovDeg/2, sorted by time.
// getTile(name) -> tile JSON or null. meta = meta.json.
export async function findFrames(meta, ra, dec, fovDeg, getTile) {
  const names = neighbourTiles(ra, dec).filter(n => meta.tiles[n]);
  const tiles = await Promise.all(names.map(getTile));
  const byKey = new Map();
  for (const t of tiles) {
    if (!t) continue;
    for (let i = 0; i < t.o.length; i++) {
      if (!footprintContains(t.c[i], ra, dec, fovDeg / 2)) continue;
      const proc = t.proc[t.p[i]], key = `${t.o[i]}D${t.d[i]}`;
      const prev = byKey.get(key);
      if (!prev || proc > prev.proc) byKey.set(key, { obs_id: t.o[i], det: t.d[i], mjd: t.t[i], proc, key: s3Key(meta.release, t.o[i], proc, t.d[i]) });
    }
  }
  return { tiles: names, frames: [...byKey.values()].sort((a, b) => a.mjd - b.mjd) };
}

// Pick n frames (sorted by mjd) closest to n evenly spaced times, without repeats.
export function spreadInTime(frames, n) {
  const t0 = frames[0].mjd, t1 = frames[frames.length - 1].mjd, taken = new Set();
  for (let k = 0; k < n; k++) {
    const target = t0 + (t1 - t0) * k / (n - 1);
    let best = -1, bd = Infinity;
    for (let i = 0; i < frames.length; i++) if (!taken.has(i) && Math.abs(frames[i].mjd - target) < bd) { bd = Math.abs(frames[i].mjd - target); best = i; }
    taken.add(best);
  }
  return [...taken].sort((a, b) => a - b).map(i => frames[i]);
}
