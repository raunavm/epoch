// Node tests for the live-mode pure functions. Run from the repo root:
//   node --test site/tests/            (the WCS test reads real headers from the public SPHEREx S3 bucket;
//                                       set OFFLINE=1 to skip it)
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { tileOf, neighbourTiles, containsWithMargin, footprintContains, s3Key } from '../js/live/index.js';
import { makeWCS, project, deproject, waveAt } from '../js/live/wcs.js';
import { readHeader, parseCards } from '../js/live/fits.js';
import { changeModel, countChanges, despike } from '../js/live/change.js';
import { parseRA, parseDec, aperture, gridSize } from '../js/live/grid.js';

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
