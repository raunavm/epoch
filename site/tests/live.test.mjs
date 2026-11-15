// Node tests for the live-mode pure functions. Run from the repo root:
//   node --test site/tests/            (the WCS test reads real headers from the public SPHEREx S3 bucket;
//                                       set OFFLINE=1 to skip it)
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { tileOf, neighbourTiles, containsWithMargin, footprintContains, s3Key, spreadInTime } from '../js/live/index.js';
import { makeWCS, project, deproject, waveAt } from '../js/live/wcs.js';
import { readHeader, parseCards } from '../js/live/fits.js';
import { changeModel, countChanges, despike } from '../js/live/change.js';
import { parseRA, parseDec, aperture, gridSize } from '../js/live/grid.js';
import { TARGETS, propagate, movedTarget } from '../js/live/targets.js';
import { parseStars, starXY, starRadius, starPM, fetchGaia, GAIA_EPOCH_MJD } from '../js/live/gaia.js';
import { healpix, tilePixel, tileUrl, parseTile, wiseStamp, hipsDates, HIPS } from '../js/live/wise.js';

const DATA = fileURLToPath(new URL('../data/index/dr1/', import.meta.url));
const BUCKET = 'https://nasa-irsa-spherex.s3.amazonaws.com';

test('tileOf matches scripts/build_tiles.py', () => {
  // [ra, dec, b, r] from build_tiles.tile_of
  const ref = [[0, 0, 22, 0], [359.999, -89.9, 0, 2], [133.795, -7.245, 20, 33], [269.452, 4.693, 23, 66], [331.04, -56.78, 8, 45],
    [248.17, 66.75, 39, 22], [76.96, 30.4, 30, 16], [7.79, -65.37, 6, 0], [180, 89.99, 44, 1], [12.3, 45.0, 33, 2], [300.1, -30.2, 14, 63]];
  for (const [ra, dec, b, r] of ref) assert.deepEqual(tileOf(ra, dec), [b, r], `${ra},${dec}`);
});

test('neighbourhood contains own tile; poles load whole band', () => {
  for (const [ra, dec] of [[0, 0], [359.9, 10], [133.8, -7.2], [10, 88.5], [200, -89]]) {
    const n = neighbourTiles(ra, dec);
    assert.ok(n.includes(tileOf(ra, dec).join('_')));
  }
  assert.ok(neighbourTiles(0.1, 0).includes('22_89')); // RA wrap: last bin of the band (nra(22) = 90)
});

test('every indexed frame lies in or next to its tile', { skip: !existsSync(DATA + 'meta.json') }, () => {
  const meta = JSON.parse(readFileSync(DATA + 'meta.json'));
  for (const name of Object.keys(meta.tiles).slice(0, 40)) {
    const t = JSON.parse(readFileSync(DATA + name + '.json'));
    for (const c of t.c) {
      const ra = Math.atan2(c.filter((_, k) => k % 2 === 0).reduce((s, a) => s + Math.sin(a * Math.PI / 180), 0),
        c.filter((_, k) => k % 2 === 0).reduce((s, a) => s + Math.cos(a * Math.PI / 180), 0)) * 180 / Math.PI;
      const dec = (c[1] + c[3] + c[5] + c[7]) / 4;
      assert.ok(neighbourTiles((ra + 360) % 360, dec).includes(name), `${name} ${ra} ${dec}`);
    }
  }
});

test('polygon containment with margin', () => {
  const sq = [[-1, -1], [1, -1], [1, 1], [-1, 1]];
  assert.equal(containsWithMargin(sq, 0.5), true);
  assert.equal(containsWithMargin(sq, 1.5), false);
  assert.equal(containsWithMargin(sq.map(([x, y]) => [x + 3, y]), 0), false);
  // footprint around a target, on the sky, across RA = 0
  const c = [359, -1, 1, -1, 1, 1, 359, 1];
  assert.equal(footprintContains(c, 0, 0, 0.5), true);
  assert.equal(footprintContains(c, 0, 0.8, 0.5), false);
});

test('gnomonic round trip', () => {
  for (const [ra0, dec0, ra, dec] of [[10, 20, 10.5, 20.3], [359.9, -89, 120, -88.5], [133.8, -7.2, 133.7, -7.25]]) {
    const [r, d] = deproject(...project(ra, dec, ra0, dec0), ra0, dec0);
    assert.ok(Math.abs(((r - ra + 540) % 360) - 180) * Math.cos(dec * Math.PI / 180) < 1e-9 && Math.abs(d - dec) < 1e-9);
  }
});

test('S3 key formula', () => {
  assert.equal(s3Key('dr1', '2025W18_1B_0001_1', 'l2b-v27-2026-196', 1),
    'dr1/level2/2025W18_1B/l2b-v27-2026-196/1/level2_2025W18_1B_0001_1D1_spx_l2b-v27-2026-196.fits');
});

test('FITS card parsing incl. HIERARCH and quoted strings', () => {
  const card = s => s.padEnd(80, ' ');
  const text = [card("SIMPLE  =                    T"), card("OBSID   = '2025W18_1B_0001_1'  / Observation ID"),
    card('HIERARCH L2 N_OUTLIER = 67708 / number of flagged outlier pixels'), card("NAME    = 'it''s'"), card('CRPIX1  =               1020.5'), card('END')].join('');
  const { hdr, nCards } = parseCards(text);
  assert.equal(nCards, 6);
  assert.deepEqual([hdr.SIMPLE, hdr.OBSID, hdr['L2 N_OUTLIER'], hdr.NAME, hdr.CRPIX1], [true, '2025W18_1B_0001_1', 67708, "it's", 1020.5]);
});

test('coordinate parsing', () => {
  assert.ok(Math.abs(parseRA('08h55m10.831s') - 133.79512917) < 1e-7);
  assert.ok(Math.abs(parseDec('-07d14m42.53s') - -7.24514722) < 1e-7);
  assert.ok(Math.abs(parseDec('-7 14 42.53') - -7.24514722) < 1e-7);
  assert.equal(parseRA('133.5'), 133.5);
  assert.equal(parseDec('+4.5'), 4.5);
  assert.ok(Number.isNaN(parseDec('95')));
  assert.equal(gridSize(2), 20);
});

test('TAN-SIP reproduces astropy all_pix2world to < 0.05 px (real DR1 headers from S3)', { skip: process.env.OFFLINE === '1' }, async () => {
  const vecs = JSON.parse(readFileSync(DATA + 'wcs_test_vectors.json'));
  const cache = {};
  for (const t of vecs) {
    const head = cache[t.key] ??= await readHeader(`${BUCKET}/${t.key}`);
    const w = makeWCS(head.hdr);
    const [x, y] = w.world2pix(t.ra, t.dec);
    const err = Math.hypot(x - t.x, y - t.y);
    const [ra, dec] = w.pix2world(t.x, t.y), [x2, y2] = w.world2pix(ra, dec);
    const fwdErr = Math.hypot(...project(ra, dec, t.ra, t.dec)) * 3600 / 6.15; // forward error in px
    assert.ok(err < 0.05, `world2pix error ${err.toFixed(4)} px for ${t.key} (${t.x},${t.y})`);
    assert.ok(fwdErr < 0.05, `pix2world error ${fwdErr.toFixed(4)} px`);
    assert.ok(Math.hypot(x2 - t.x, y2 - t.y) < 1e-3);
    console.log(`  ${t.key.split('/').pop()} (${t.x},${t.y}): inverse ${err.toExponential(2)} px, forward ${fwdErr.toExponential(2)} px`);
  }
});

test('wave table bilinear hits node values', { skip: !existsSync(DATA + 'wave_tables.json') }, () => {
  const tab = JSON.parse(readFileSync(DATA + 'wave_tables.json'))['1'];
  assert.ok(Math.abs(waveAt(tab, tab.x[2] - 1, tab.y[3] - 1)[0] - tab.values[3][2][0]) < 1e-9);
});

// Synthetic sky: a static star with a steep spectrum, observed at many wavelengths, plus a mover in one frame.
function synth({ withMover }) {
  const S = 24, frames = [];
  let seed = 7; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
  const gauss = () => Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd());
  const blob = (x, y, x0, y0) => Math.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * 1.2 ** 2));
  for (let e = 0; e < 24; e++) {
    const det = 1 + (e % 3), lam0 = [0.9, 1.6, 2.6][det - 1] + 0.05 * Math.floor(e / 3); // different wavelength every visit
    const v = new Float32Array(S * S), lam = new Float32Array(S * S);
    for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
      const p = y * S + x, l = lam0 + 0.002 * y;
      lam[p] = l;
      v[p] = 3 + 0.4 * l + gauss() + 400 * (l - 0.8) * blob(x, y, 8, 12) // steep red static star
        + (withMover && e === 13 ? 60 * blob(x, y, 17, 6) : 0);
    }
    frames.push({ v, lam, det, mjd: 60800 + e });
  }
  return { frames, S };
}

test('change model rev. 2 (f = 0.1): steep-spectrum static star gives many naive changes and zero corrected; a mover survives', () => {
  const { frames, S } = synth({ withMover: false });
  const r = changeModel(frames, S, 0.1);
  assert.ok(r.nNaiveAll >= 10, `naive (all frames) changes ${r.nNaiveAll}`);
  assert.equal(r.nCorr, 0);
  const m = changeModel(synth({ withMover: true }).frames, S, 0.1);
  assert.ok(countChanges(despike(m.corr[13], S), S) >= 1, 'mover lost in corrected map');
  assert.ok(Math.abs(m.corr[13][6 * S + 17]) > 5);
  assert.ok(m.nCorr >= 1);
  for (const e of [0, 5, 20]) assert.equal(countChanges(despike(m.corr[e], S), S), 0);
});

test('single-pixel spike rejection removes a resampled hot pixel but keeps a PSF-sized source', () => {
  const S = 15, m = new Float32Array(S * S);
  m[3 * S + 3] = 40; m[3 * S + 4] = 20; m[4 * S + 3] = 15; m[4 * S + 4] = 8; // bilinear-spread spike (2x2)
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) m[y * S + x] += 30 * Math.exp(-((x - 10) ** 2 + (y - 10) ** 2) / (2 * 1.2 ** 2));
  assert.equal(countChanges(m, S), 2);
  assert.equal(countChanges(despike(m, S), S), 1);
});

test('aperture photometry subtracts the annulus', () => {
  const S = 15, v = new Float32Array(S * S).fill(2);
  v[7 * S + 7] += 10;
  const a = aperture(v, S, 7, 7);
  assert.ok(Math.abs(a.flux - 10) < 1e-6);
  assert.equal(a.err, 0);
});

test('spreadInTime keeps n distinct frames spanning the whole time range', () => {
  const fr = [...Array(1000)].map((_, i) => ({ mjd: i < 900 ? i * 0.01 : 100 + i })); // dense cluster + sparse tail
  const sub = spreadInTime(fr, 300);
  assert.equal(new Set(sub).size, 300);
  assert.equal(sub[0], fr[0]); assert.equal(sub[299], fr[999]);
  assert.ok(sub.filter(f => f.mjd > 100).length >= 90); // the sparse tail is fully represented
  assert.ok(sub.every((f, k) => !k || f.mjd > sub[k - 1].mjd));
});

test('typed old-epoch positions of fast movers are recognised; slow or unrelated positions are not', () => {
  const now = 60982; // about the middle of DR1
  assert.equal(movedTarget(parseRA('08h55m10.831s'), parseDec('-7d14m42.53s'), now)?.id, 'wise0855'); // the challenge text (2010)
  assert.equal(movedTarget(parseRA('17 57 48.50'), parseDec('+04 41 36.2'), now)?.id, 'barnard');       // SIMBAD, epoch 2000
  assert.equal(movedTarget(269.44850252544, 4.73942005111, now)?.id, 'barnard');                          // Gaia DR3, epoch 2016
  const eps = TARGETS.find(x => x.id === 'epsindib');
  assert.equal(movedTarget(eps.ra, eps.dec, now)?.id, 'epsindib');
  assert.equal(movedTarget(...propagate(eps, now), now)?.id, 'epsindib'); // already where it is now: same target, no jump
  const rw = TARGETS.find(x => x.id === 'rwaur');
  assert.equal(movedTarget(rw.ra, rw.dec, now), null);              // a variable star has not moved
  assert.equal(movedTarget(269.452, 4.693 + 0.2, now), null);       // 12 arcmin off Barnard's track
  assert.equal(movedTarget(180, 0, now), null);
});

test('Gaia rows: CSV and TSV parse alike, ids keep all 19 digits, missing motion is null', () => {
  const csv = 'source_id,ra,dec,pmra,pmdec,phot_g_mean_mag\n4472832130942575872,269.44850252543836,4.739420051112412,-801.55096,10362.395,8.1939745\n4472835914798807040,269.4156046589069,4.757268219215219,,,21.216444\n';
  const tsv = '#comment\nSource\tRA_ICRS\tDE_ICRS\tpmRA\tpmDE\tGmag\n \tdeg\tdeg\tmas/yr\tmas/yr\tmag\n---\t---\t---\t---\t---\t---\n4472832130942575872\t269.44850252544\t+04.73942005111\t -801.551\t10362.394\t 8.193974\n';
  const a = parseStars(csv), b = parseStars(tsv);
  assert.equal(a.length, 2); assert.equal(b.length, 1);
  assert.equal(a[0].id, '4472832130942575872'); assert.equal(b[0].id, a[0].id);
  assert.ok(Math.abs(a[0].pmdec - b[0].pmdec) < 0.01 && Math.abs(b[0].dec - 4.73942005111) < 1e-12);
  assert.deepEqual([a[1].pmra, a[1].pmdec], [null, null]);
  assert.ok(Math.abs(starPM(a[0]) - 10.393) < 0.001); assert.equal(starPM(a[1]), 0);
  assert.ok(starRadius(8) > starRadius(16) && starRadius(16) > starRadius(21) && starRadius(null) === starRadius(21));
});

test('Gaia stars are moved from J2016.0 to the picture date (pmRA includes cos dec)', () => {
  const S = 21, c = (S - 1) / 2, st = { ra: 100, dec: 60, pmra: 6150, pmdec: -12300 }; // 1 and 2 grid pixels per year
  assert.deepEqual(starXY(st, GAIA_EPOCH_MJD, S, 100, 60).map(v => +v.toFixed(6)), [c, c]);
  const [x, y] = starXY(st, GAIA_EPOCH_MJD + 365.25, S, 100, 60);
  assert.ok(Math.abs(x - (c - 1)) < 1e-3 && Math.abs(y - (c + 2)) < 1e-3, `${x}, ${y}`); // east is left, south is down
  assert.deepEqual(starXY({ ra: 100, dec: 60, pmra: null, pmdec: null }, 70000, S, 100, 60).map(v => +v.toFixed(6)), [c, c]);
});

test('HEALPix faces and HiPS tile pixel', () => {
  assert.equal(healpix(10, 80, 0)[0], 0); assert.equal(healpix(100, 80, 0)[0], 1);
  assert.equal(healpix(359.9, 0, 0)[0], 4); assert.equal(healpix(90, 0, 0)[0], 5);
  assert.equal(healpix(100, -80, 0)[0], 9); assert.equal(healpix(300, -60, 0)[0], 11);
  // Checked against the real AllWISE W2 tile: Barnard's Star (2010 position) is the bright source at this row and column.
  assert.deepEqual(tilePixel(269.4497, 4.7236), { tile: 508502, row: 386, col: 192 });
  assert.equal(tileUrl(508502), HIPS.base + 'Norder8/Dir500000/Npix508502.fits');
  for (const [ra, dec] of [[0, 0], [359.999, 89.99], [123.4, -89.99], [200, 41.81], [45, -41.82]]) {
    const p = tilePixel(ra, dec);
    assert.ok(Number.isInteger(p.tile) && p.tile >= 0 && p.tile < 12 * 4 ** 8 && p.row >= 0 && p.row < 512 && p.col >= 0 && p.col < 512, `${ra},${dec}`);
  }
});

test('HiPS FITS tile parsing', () => {
  const card = (k, v) => `${k.padEnd(8)}= ${String(v).padStart(20)}`.padEnd(80);
  const head = (card('SIMPLE', 'T') + card('BITPIX', -32) + card('NAXIS', 2) + card('NAXIS1', 4) + card('NAXIS2', 4) + 'END'.padEnd(80)).padEnd(2880);
  const buf = new Uint8Array(2880 + 64); buf.set(new TextEncoder().encode(head));
  const dv = new DataView(buf.buffer); for (let i = 0; i < 16; i++) dv.setFloat32(2880 + 4 * i, i * 1.5, false);
  assert.deepEqual([...parseTile(buf.buffer, 4)], Array.from({ length: 16 }, (_, i) => i * 1.5));
  assert.throws(() => parseTile(buf.buffer, 512)); // wrong size
  assert.throws(() => parseTile(new TextEncoder().encode('<html>not a tile</html>').buffer));
});

test('WISE tiles and Gaia agree on where Barnard\'s Star was in 2010 (IRSA HiPS mirror + catalogue service)', { skip: process.env.OFFLINE === '1' }, async () => {
  const [ra, dec] = [269.4463, 4.7677], S = gridSize(8), k = 2;
  const [w, dates, gaia] = await Promise.all([wiseStamp(ra, dec, S, k), hipsDates(), fetchGaia(ra, dec, 0.1)]);
  assert.equal(new Date(((dates.mjdMin + dates.mjdMax) / 2 - 40587) * 864e5).getUTCFullYear(), 2010);
  const star = gaia.stars.find(s => s.id === '4472832130942575872');
  assert.ok(star, 'Barnard\'s Star is in the Gaia answer with its full id');
  const [x, y] = starXY(star, (dates.mjdMin + dates.mjdMax) / 2, S, ra, dec); // live-grid pixels -> WISE stamp pixels
  const at = w.v[Math.round((y + 0.5) * k - 0.5) * w.n + Math.round((x + 0.5) * k - 0.5)];
  const sorted = Float32Array.from(w.v).sort(), med = sorted[sorted.length >> 1];
  assert.ok(at > 100 * Math.max(med, 0.1), `WISE value ${at} at the predicted 2010 position, median ${med}`);
  assert.ok(y - (S - 1) / 2 > 20, 'in 2010 the star was well south of where it is in 2025-26');
});

test('overlay fetches fail soft: an unreachable service rejects instead of hanging or returning junk', async () => {
  const down = async () => { throw new Error('network down'); }, notFound = async () => ({ ok: false, status: 404, text: async () => '<html>404</html>' });
  await assert.rejects(fetchGaia(10, 10, 0.05, down));
  await assert.rejects(fetchGaia(10, 10, 0.05, notFound));
  await assert.rejects(wiseStamp(10, 10, 5, 1, notFound));
  await assert.rejects(hipsDates(notFound));
});
