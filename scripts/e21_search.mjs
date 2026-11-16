// E21: Tracklet's default search (detect.js + link.js, tolerances from node_search.mjs) on the
// DR1 NEP stack baked by run_e21_bake.py; plus its time-permutation nulls (scramble_nulls.mjs)
// and its injection-recovery (app.js depthFaithful, without the browser). Adapted from Tracklet
// (MIT, Team Tracklet): scripts/node_search.mjs, scripts/scramble_nulls.mjs, web/js/app.js.
//
//   node scripts/e21_search.mjs real
//   node scripts/e21_search.mjs nulls --n 400 [--start 0 --count 100 --tag a]
//   node scripts/e21_search.mjs inject --cells 0,1,2 --tag w0
//   node scripts/e21_search.mjs check500
//   node scripts/e21_search.mjs summary
//   any mode: --field nep|sep (default nep), --grid main|supp (default main, the frozen grid)

import { existsSync, mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const JS = join(ROOT, 'src', 'tracklet', 'js');
const OUT = join(ROOT, 'data', 'phase2', 'e21');
const { Wcs } = await import(join(JS, 'wcs.js'));
const { detect, plantSource } = await import(join(JS, 'detect.js'));
const { findTracklets, requiredEpochs } = await import(join(JS, 'link.js'));
const { robustSigma } = await import(join(JS, 'stretch.js'));
const D = await import(join(JS, 'depth.js'));
const E = await import(join(JS, 'ephem.js'));
const { angularSeparation } = await import(join(JS, 'wcs.js'));

const { values: args, positionals } = parseArgs({ allowPositionals: true, options: {
  n: { type: 'string', default: '400' }, start: { type: 'string', default: '0' },
  count: { type: 'string', default: '0' }, seed: { type: 'string', default: '11' },
  cells: { type: 'string', default: '' }, tag: { type: 'string', default: 'x' },
  percell: { type: 'string', default: '48' }, field: { type: 'string', default: 'nep' },
  grid: { type: 'string', default: 'main' },
} });
const mode = positionals[0];

// ---- frozen settings (docs/protocols/E21.md) --------------------------------------------------
const THRESHOLD = 6.5;
// 'main' is the frozen grid; 'supp' the supplementary grid added later (E21.md, addendum A).
const SUPP = args.grid === 'supp';
const SNRS = SUPP ? [5, 6, 7, 8, 10, 13, 17, 22, 30] : [5, 6, 7, 8, 10, 13, 17];
const AUS = SUPP ? [200, 350, 500, 700, 1000, 1200, 1500, 2000] : [200, 350, 500, 700, 1000, 1200];
const PER_CELL = Number(args.percell);
const INJ_SEED = SUPP ? 8 : 7;
const PREFIX = SUPP ? 'supp_cell' : 'inj_cell';
const linkOptions = (scale, extra = 0) => ({
  matchArcsec: 2.0 * scale, staticTolArcsec: 1.0 * scale, sigmaArcsec: 0.5 * scale, maxPerEpoch: 40 + extra,
});

// ---- the stack ---------------------------------------------------------------------------------
const FIELD = args.field;
const RES = FIELD === 'nep' ? OUT : join(OUT, `${FIELD}_res`);
mkdirSync(RES, { recursive: true });
const m = JSON.parse(readFileSync(join(OUT, `${FIELD}_dr1.json`), 'utf8'));
const frames = m.epochs.map((e) => {
  const buf = readFileSync(join(OUT, FIELD, e.stamp.file));
  const f32 = new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
  const data = Float64Array.from(f32);
  const shrink = e.stats.resample_var_shrink ?? 1;
  return { i: e.i, mjd: e.mjd, grid: new Wcs(e.stamp.wcs), width: e.stamp.w, height: e.stamp.h,
           data, sigma: robustSigma(Array.from(data).filter(Number.isFinite)), resampled: shrink > 0 && shrink < 1,
           shrink };
});
const scale = frames[0].grid.scaleArcsec;
const nEp = frames.length;
const REQUIRED = requiredEpochs(nEp);

function catalogue(list) {
  const dets = [];
  for (const f of list) {
    const found = detect(f.data, f.width, f.height, { thresholdSigma: THRESHOLD, pixelNoiseShrink: f.shrink });
    for (const s of found.sources) {
      const [ra, dec] = f.grid.pixelToWorld(s.x + 1, s.y + 1);
      dets.push({ epoch: f.i, mjd: f.mjd, raDeg: ra, decDeg: dec, snr: s.snr, x: s.x, y: s.y });
    }
  }
  return dets;
}

const candRow = (t) => {
  const s = t.solution || {};
  return { members: t.members.length, epochs: t.members.map((d) => d.epoch),
           distanceAu: s.distanceAu, distanceErrAu: s.distanceErrAu, reducedChi2: s.reducedChi2,
           significance: s.significance, ra0Deg: s.ra0Deg, dec0Deg: s.dec0Deg,
           members_xy: t.members.map((d) => [d.epoch, d.x, d.y, d.raDeg, d.decDeg, d.snr]) };
};
const save = (name, obj) => writeFileSync(join(RES, name), JSON.stringify(obj, null, 1));

function plantAndSearch(bodies, amp) {
  const list = frames.map((f) => {
    const data = Float64Array.from(f.data);
    for (const b of bodies) {
      const at = b.places.get(f.i);
      if (at && at[0] >= 0 && at[1] >= 0 && at[0] <= f.width - 1 && at[1] <= f.height - 1) {
        plantSource(data, f.width, f.height, at[0], at[1], amp);
      }
    }
    return { ...f, data };
  });
  const dets = catalogue(list);
  const report = findTracklets(dets, linkOptions(scale, bodies.length));
  return { dets, report };
}

/** Per body: recovered (scoreRecovery's test), and the best-matching candidate's fit. */
function perBody(bodies, report) {
  const rec = D.scoreRecovery(bodies, report, { required: REQUIRED, matchArcsec: 1.5 * scale });
  return bodies.map((b, k) => {
    let best = null, bestK = 0;
    for (const t of report.tracklets) {
      const n = t.members.filter((d) => { const at = b.sky.get(d.epoch);
        return at && angularSeparation(d.raDeg, d.decDeg, at[0], at[1]) * 3600 <= 1.5 * scale; }).length;
      if (n > bestK) { bestK = n; best = t; }
    }
    const s = best && best.solution;
    return { recovered: rec[k], matched: bestK, trueAu: b.dAu, fitAu: s ? s.distanceAu : null,
             fitErrAu: s ? s.distanceErrAu : null, chi2: s ? s.reducedChi2 : null };
  });
}

const t0 = performance.now();
if (mode === 'real') {
  const dets = catalogue(frames);
  const opts = linkOptions(scale);
  const rep = findTracklets(dets, opts);
  const looser = [REQUIRED - 1, REQUIRED - 2].map((k) => ({ minEpochs: k,
    candidates: findTracklets(dets, { ...opts, minEpochs: k }).tracklets.map(candRow) }));
  save('real.json', { n_epochs: nEp, required: REQUIRED, scale_arcsec: scale,
    baseline_days: Math.max(...frames.map((f) => f.mjd)) - Math.min(...frames.map((f) => f.mjd)),
    n_detections: rep.nDetections, n_static: rep.nStatic, n_moving: rep.nMoving, n_linked: rep.nLinked,
    max_distance_au: rep.maxDistanceAu, blocked: rep.blocked,
    candidates: rep.tracklets.map(candRow), looser, ms: performance.now() - t0,
    per_epoch: frames.map((f) => ({ i: f.i, mjd: f.mjd, sigma: f.sigma, n: dets.filter((d) => d.epoch === f.i).length })) });
  save('real_dets.json', dets);
  console.log(`real: ${rep.nDetections} det, ${rep.nStatic} static, ${rep.nMoving} moving, ${rep.nLinked} linked, `
    + `${rep.tracklets.length} candidates (needs ${REQUIRED}/${nEp}); looser ${looser.map((l) => `${l.minEpochs}:${l.candidates.length}`).join(' ')}; `
    + `${((performance.now() - t0) / 1000).toFixed(1)} s`);
} else if (mode === 'nulls') {
  const N = Number(args.n), start = Number(args.start), count = Number(args.count) || N;
  const dets = catalogue(frames);
  const opts = linkOptions(scale);
  const rand = D.rng(Number(args.seed) * 7919 + nEp);
  const perms = [];
  for (let k = 0; k < N; k++) {   // same sequence for every worker; each takes its slice
    for (;;) {
      const p = [...Array(nEp).keys()];
      for (let i = nEp - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [p[i], p[j]] = [p[j], p[i]]; }
      if (p.some((v, i) => v !== i)) { perms.push(p); break; }
    }
  }
  const order = frames.map((f) => f.i);
  const mjdOf = new Map(frames.map((f) => [f.i, f.mjd]));
  const levels = [REQUIRED, REQUIRED - 1, REQUIRED - 2];
  const rows = [];
  for (let k = start; k < Math.min(N, start + count); k++) {
    const perm = perms[k];
    const remap = new Map(order.map((ep, i) => [ep, mjdOf.get(order[perm[i]])]));
    const sc = dets.map((d) => ({ ...d, mjd: remap.get(d.epoch) }));
    const counts = levels.map((lv) => findTracklets(sc, { ...opts, minEpochs: lv }).tracklets.length);
    rows.push({ k, counts });
    if (rows.length % 10 === 0) {
      save(`nulls_${args.tag}.json`, { levels, rows });
      console.log(`nulls ${args.tag}: ${rows.length} done, ${((performance.now() - t0) / 1000).toFixed(0)} s`);
    }
  }
  save(`nulls_${args.tag}.json`, { levels, rows });
} else if (mode === 'inject') {
  const cells = args.cells.split(',').filter(Boolean).map(Number);
  for (const c of cells) {
    const si = Math.floor(c / AUS.length), di = c % AUS.length;
    const fn = `${PREFIX}${String(c).padStart(2, '0')}.json`;
    if (existsSync(join(RES, fn))) continue;
    const tc = performance.now();
    const bodies = D.placeBodies(frames, m.ra, m.dec, AUS[di], PER_CELL, D.cellSeed(INJ_SEED, si, di), { scale });
    const amp = D.cellAmplitude(SNRS[si], frames);
    const { report } = plantAndSearch(bodies, amp);
    const pb = perBody(bodies, report);
    const k = pb.filter((b) => b.recovered).length;
    save(fn, { cell: c, snr: SNRS[si], au: AUS[di], n: bodies.length, k, bodies: pb,
               n_candidates: report.tracklets.length, ms: performance.now() - tc });
    console.log(`cell ${c} snr ${SNRS[si]} au ${AUS[di]}: ${k}/${bodies.length} (${((performance.now() - tc) / 1000).toFixed(1)} s)`);
  }
} else if (mode === 'check500') {
  const bodies = D.placeBodies(frames, m.ra, m.dec, 500, 1, 500, { scale });
  const amp = D.cellAmplitude(12, frames);
  const { report } = plantAndSearch(bodies, amp);
  const b = bodies[0];
  const pb = perBody(bodies, report)[0];
  save('check500.json', { snr: 12, au: 500, seed: 500, ...pb, n_candidates: report.tracklets.length,
    n_detections: report.nDetections, n_static: report.nStatic, n_moving: report.nMoving,
    candidates: report.tracklets.map(candRow),
    planted: frames.map((f) => ({ i: f.i, xy: b.places.get(f.i), sky: b.sky.get(f.i) })),
    // the planted body's true apparent track, densely, for the figure
    track: Array.from({ length: 400 }, (_, j) => {
      const t = frames[0].mjd + j * (frames[nEp - 1].mjd - frames[0].mjd) / 399;
      const ap = E.apparentPlace(...E.driftDirection(b.ra0, b.dec0, b.pmRa, b.pmDec, (t - frames[0].mjd) / E.YEAR_DAYS), 500, t);
      const [x, y] = frames[0].grid.worldToPixel(ap.raDeg, ap.decDeg);
      return [t, x - 1, y - 1];
    }) });
  console.log(`check500: recovered=${pb.recovered} matched=${pb.matched} fit=${pb.fitAu?.toFixed(2)} ± ${pb.fitErrAu?.toFixed(2)} chi2=${pb.chi2?.toFixed(2)}`);
} else if (mode === 'summary') {
  // completeness map and Tracklet's exclusion, from whatever cells are on disk
  const frac = SNRS.map(() => AUS.map(() => NaN)), n = SNRS.map(() => AUS.map(() => 0));
  const k = SNRS.map(() => AUS.map(() => 0));
  for (const f of readdirSync(RES).filter((x) => x.startsWith(PREFIX))) {
    const r = JSON.parse(readFileSync(join(RES, f), 'utf8'));
    const si = SNRS.indexOf(r.snr), di = AUS.indexOf(r.au);
    frac[si][di] = r.k / r.n; n[si][di] = r.n; k[si][di] = r.k;
  }
  const area = frames[0].grid.pixelAreaSr;
  const sigma = D.referenceSigma(frames);
  const resampled = frames.some((f) => f.resampled);
  const abOfSnr = (s) => D.snrToAb(s, sigma, area, resampled);
  const map = { snr: SNRS, au: AUS, frac, n, ab: SNRS.map(abOfSnr) };
  const best = D.bestCell(map);
  const chance = D.chanceMap(map).frac;
  const ex = {};
  for (const cl of [0.5, 0.9, 0.95]) {
    ex[cl] = { reached: chance.some((row, si) => row.some((c, di) => n[si][di] > 0 && c >= cl)),
               bodies: D.exclusion(map, { radii: D.BODY_RADII.map((b) => b.radius), albedo: 0.30, band: 'z', level: cl, abOfSnr }),
               smallest: D.smallestReached(map, { albedo: 0.30, band: 'z', level: cl, abOfSnr, max: 400 }) };
  }
  save(SUPP ? 'summary_supp.json' : 'summary_inj.json', { snr: SNRS, au: AUS, k, n, frac, chance, ab: map.ab, best_cell: best,
    best_chance: best ? D.recoveryChance(best.k, best.n) : null, needed90: D.neededFor(PER_CELL, 0.9),
    needed95: D.neededFor(PER_CELL, 0.95), exclusion: ex, ref_sigma: sigma,
    area_deg2: Math.PI * (0.36 * frames[0].width * scale / 3600) ** 2 });
  console.log(JSON.stringify({ best, ex50: ex[0.5].smallest, reached90: ex[0.9].reached, reached95: ex[0.95].reached }));
}
