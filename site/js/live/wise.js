// WISE look-back: the AllWISE W2 (4.6 µm) picture of any field, sampled onto the live grid from HiPS sky tiles.
// The tiles (HEALPix cells stored as plain FITS files, made by CDS) are read from the NASA/IPAC IRSA mirror,
// which allows cross-origin requests. Pure functions apart from the two fetches.
import { deproject } from './wcs.js';
import { parseCards } from './fits.js';
import { SCALE_ARCSEC } from './grid.js';

export const HIPS = { base: 'https://irsa.ipac.caltech.edu/data/hips/CDS/AllWISE/W2/', order: 8, width: 512, um: 4.6 };

// HEALPix: base face (0-11) and integer (x, y) inside it for a sky position, with 2^order pixels along a face side.
export function healpix(ra, dec, order) {
  const ns = 2 ** order, z = Math.sin(dec * Math.PI / 180), za = Math.abs(z), tt = (((ra % 360) + 360) % 360) / 90;
  if (za <= 2 / 3) { // equatorial belt
    const t1 = ns * (0.5 + tt), t2 = ns * z * 0.75, jp = Math.floor(t1 - t2), jm = Math.floor(t1 + t2);
    const ifp = Math.floor(jp / ns), ifm = Math.floor(jm / ns);
    return [ifp === ifm ? (ifp & 3) + 4 : ifp < ifm ? ifp & 3 : (ifm & 3) + 8, jm % ns, ns - (jp % ns) - 1];
  }
  const ntt = Math.min(3, Math.floor(tt)), tp = tt - ntt, tmp = ns * Math.sqrt(3 * (1 - za)); // polar caps
  const jp = Math.min(ns - 1, Math.floor(tp * tmp)), jm = Math.min(ns - 1, Math.floor((1 - tp) * tmp));
  return z >= 0 ? [ntt, ns - jm - 1, ns - jp - 1] : [ntt + 8, jp, jm];
}

// HiPS tile number (nested scheme) at `order`, and the FITS row and column of the position inside that tile.
// Tile layout checked against a real tile: row = width - 1 - x, column = y (see site/tests/live.test.mjs).
export function tilePixel(ra, dec, order = HIPS.order, width = HIPS.width) {
  const bits = Math.log2(width), [face, x, y] = healpix(ra, dec, order + bits);
  const spread = v => { let r = 0; for (let b = 0; b < order; b++) r += ((v >> b) & 1) * 4 ** b; return r; };
  return { tile: face * 4 ** order + spread(x >> bits) + 2 * spread(y >> bits), row: width - 1 - (x & (width - 1)), col: y & (width - 1) };
}
export const tileUrl = (tile, hips = HIPS) => `${hips.base}Norder${hips.order}/Dir${Math.floor(tile / 10000) * 10000}/Npix${tile}.fits`;

// A HiPS FITS tile (one HDU, big-endian float32) -> Float32Array in file order.
export function parseTile(buf, width = HIPS.width) {
  const u8 = new Uint8Array(buf);
  let head = null;
  for (let n = 2880; !head && n <= Math.min(u8.length, 28800); n += 2880) head = parseCards(new TextDecoder('ascii').decode(u8.subarray(0, n)));
  if (!head || head.hdr.BITPIX !== -32 || head.hdr.NAXIS1 !== width || head.hdr.NAXIS2 !== width) throw new Error('unexpected HiPS tile');
  const dv = new DataView(u8.buffer, u8.byteOffset + Math.ceil(head.nCards * 80 / 2880) * 2880), out = new Float32Array(width * width);
  for (let i = 0; i < out.length; i++) out[i] = dv.getFloat32(4 * i, false);
  return out;
}

// Survey dates from the HiPS "properties" file: {mjdMin, mjdMax} (its t_min and t_max keys).
export async function hipsDates(fetchFn = fetch, hips = HIPS) {
  const r = await fetchFn(hips.base + 'properties');
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  const text = await r.text(), key = k => +(text.match(new RegExp(`^${k}\\s*=\\s*([\\d.]+)`, 'm')) || [])[1];
  return { mjdMin: key('t_min'), mjdMax: key('t_max') };
}

// WISE pixels on an n x n north-up grid (east left) covering the same sky as the S x S live grid; n = S * k.
// Nearest-pixel sampling: the tiles' 1.6″ pixels are much finer than WISE's 6″ blur.
export async function wiseStamp(ra, dec, S, k, fetchFn = fetch) {
  const n = S * k, c = (n - 1) / 2, s = SCALE_ARCSEC / 3600 / k, px = [], tiles = new Map();
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    const p = tilePixel(...deproject((c - i) * s, (c - j) * s, ra, dec));
    px.push(p); tiles.set(p.tile, null);
  }
  await Promise.all([...tiles.keys()].map(async tile => {
    const r = await fetchFn(tileUrl(tile));
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    tiles.set(tile, parseTile(await r.arrayBuffer()));
  }));
  return { n, v: Float32Array.from(px, p => tiles.get(p.tile)[p.row * HIPS.width + p.col]), nTiles: tiles.size };
}
