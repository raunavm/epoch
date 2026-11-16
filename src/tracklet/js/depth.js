// How deep the search goes, and what that rules out.
//
// A search that finds nothing has said something only if you know what it
// would have found. This module plants synthetic bodies across a grid of
// brightness and distance and asks, cell by cell, what fraction the search
// gets back -- the completeness map -- and then turns that map into physical
// terms through reflected light: a body of this radius and albedo, in this
// patch of sky, is excluded out to that distance.
//
// Two ways to fill the map, and they are kept deliberately apart:
//
//   quick     predicts each visit's detection from the matched filter's
//             linear response -- the sky's own significance at the planted
//             position plus exactly what the planted body adds to it -- and
//             applies the search's cuts (threshold, static rejection, visits
//             needed) to the result. Milliseconds, and a model of the search.
//   faithful  plants the bodies into copies of the pixels and runs the real
//             search on them, end to end (app.js drives it; this module only
//             places the bodies and scores the result). Minutes, and a
//             measurement of the search.
//
// The baked fields ship a faithful map; anywhere else the quick one appears at
// once and the faithful run is there to confirm it. The audit checks that the
// two agree where both exist, which is what makes the quick one usable.

import { apparentPlace, driftDirection, orbitalRateArcsecPerYear, solveAstrometry, YEAR_DAYS } from './ephem.js';
import { tanDeproject, angularSeparation } from './wcs.js';
import { backgroundMap, gaussianKernel, matchedFilterGain, psfFluxIntegral, resampledFilterNorm,
         PSF_SIGMA_PX } from './detect.js';
import { apparentMag, fluxToAb } from './reflected.js';

//: Planted brightness per visit, in the search's own sigma, and distance, AU.
//: The distances span the linker's 30-2,000 AU grid; the brightness starts
//: below its 6.5-sigma threshold so the map shows where it gives out.
export const DEPTH_SNR = [4, 5, 6, 7, 8, 10, 13, 17];
export const DEPTH_AU = [40, 70, 120, 200, 350, 600, 1000, 1700];

/** The seed for one cell's bodies. Shared by the quick and the faithful maps,
 *  so the two can be compared on the very same bodies, cell by cell, and a
 *  difference between them is the model's and not the dice's. */
export const cellSeed = (seed, si, di) => seed * 10007 + si * 101 + di;

/** Seeded generator (mulberry32), so a map can be reproduced exactly. */
export function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Place `n` bodies at `dAu`: random starting points inside the middle of the
 *  stamp, at least `sepPx` apart, drifting at the circular-orbit rate for that
 *  distance in a random direction. Returns each body's array position in every
 *  frame (`places`, a Map from epoch to [x, y]) and its sky position there.
 *
 *  `frames` are `{ i, mjd, grid, width, height }`, where `grid` is the Wcs that
 *  indexes that frame's pixels 1-based from its corner. */
export function placeBodies(frames, raDeg, decDeg, dAu, n, seed, { sepPx = 12, scale = 6.15 } = {}) {
  const r = rng(seed);
  const t0 = frames[0].mjd;
  const w0 = frames[0].width, h0 = frames[0].height;
  const reach = 0.36 * Math.min(w0, h0);        // start within the middle of the stamp
  const rate = orbitalRateArcsecPerYear(dAu);
  const bodies = [];
  for (let tries = 0; bodies.length < n && tries < 40 * n; tries++) {
    const rr = reach * Math.sqrt(r()), th = 2 * Math.PI * r();
    const ox = rr * Math.cos(th), oy = rr * Math.sin(th);
    if (bodies.some((b) => Math.hypot(b.ox - ox, b.oy - oy) < sepPx)) continue;
    const dir = 2 * Math.PI * r();
    const [ra0, dec0] = tanDeproject(ox * scale / 3600, oy * scale / 3600, raDeg, decDeg);
    const pmRa = rate * Math.cos(dir), pmDec = rate * Math.sin(dir);
    const places = new Map(), sky = new Map();
    for (const f of frames) {
      const yr = (f.mjd - t0) / YEAR_DAYS;
      const ap = apparentPlace(...driftDirection(ra0, dec0, pmRa, pmDec, yr), dAu, f.mjd);
      let xy = null;
      try { const [x, y] = f.grid.worldToPixel(ap.raDeg, ap.decDeg); xy = [x - 1, y - 1]; } catch { /* off */ }
      places.set(f.i, xy);
      sky.set(f.i, [ap.raDeg, ap.decDeg]);
    }
    bodies.push({ ox, oy, ra0, dec0, pmRa, pmDec, dAu, places, sky });
  }
  return bodies;
}

/** Peak amplitude to plant for `snr` sigma in a frame of scatter `sigma` --
 *  the same calibration the console's own test object uses, including the
 *  widening a resampled stamp's correlated noise needs. */
export function plantAmplitude(snr, sigma, resampled) {
  return snr * sigma * (resampled ? resampledFilterNorm() : 1) / matchedFilterGain();
}

/** The one amplitude a cell plants into every visit, from the field's median
 *  visit. The map's brightness axis is a *flux*, not a significance.
 *
 *  It was a significance, planted afresh against each visit's own scatter, and
 *  that is not a body: it made the planted object brighter exactly where the
 *  sky was noisier, so it cleared the threshold in visits a real body of the
 *  same magnitude would have been lost in. On NEP one of the twelve visits is
 *  2.34x noisier than the median, and scoring the same bodies both ways over
 *  120 plants put the old convention 5.8 points of completeness too high there,
 *  3.3 on SEP and 2.5 on the ecliptic field -- always optimistic, which for a
 *  page whose whole output is an exclusion is the wrong direction to be wrong
 *  in. A real body has one flux and lets its significance fall where it may.
 *
 *  So the row labels are the significance that flux reaches in the *median*
 *  visit, and `snrToAb` -- which already converted through the median -- now
 *  states the magnitude actually planted rather than an average of twelve. */
export function referenceSigma(frames) {
  const s = frames.map((f) => f.sigma).sort((a, b) => a - b);
  return s.length ? s[s.length >> 1] : 0;
}

/** The amplitude for a whole cell: one flux, every visit. */
export function cellAmplitude(snr, frames) {
  return plantAmplitude(snr, referenceSigma(frames), frames.some((f) => f.resampled));
}

/** The matched filter's significance map of a frame, as `detect` computes it,
 *  plus what it needs to add a planted body's response on top. */
export function skyResponse(data, width, height, { pixelNoiseShrink = 1 } = {}) {
  let { bg, sigma } = backgroundMap(data, width, height);
  if (pixelNoiseShrink > 0 && pixelNoiseShrink < 1) sigma *= resampledFilterNorm();
  const { kernel, r, n } = gaussianKernel(PSF_SIGMA_PX);
  const snr = new Float64Array(width * height);
  const q = new Float64Array(width * height);
  for (let j = 0; j < height; j++) {
    for (let i = 0; i < width; i++) {
      let acc = 0, qq = 0;
      for (let dj = -r; dj <= r; dj++) {
        const jj = j + dj;
        if (jj < 0 || jj >= height) continue;
        for (let di = -r; di <= r; di++) {
          const ii = i + di;
          if (ii < 0 || ii >= width) continue;
          const v = data[jj * width + ii];
          if (!Number.isFinite(v)) continue;
          const w = kernel[(dj + r) * n + (di + r)];
          acc += w * (v - bg[jj * width + ii]);
          qq += w * w;
        }
      }
      snr[j * width + i] = qq > 0 ? acc / (sigma * Math.sqrt(qq)) : 0;
      q[j * width + i] = qq;
    }
  }
  return { snr, q, sigma, kernel, r, n, width, height, data };
}

/** The significance a body of peak `amp` at (x, y) reaches at its best pixel,
 *  sky included: the filter is linear, so it is the sky's map plus exactly
 *  what the body's light adds there. Also whether that best pixel is a static
 *  source already -- the sky alone above threshold -- which the search's static
 *  cut would take the detection for. */
export function predictDetection(resp, x, y, amp, threshold) {
  const { snr, q, sigma, kernel, r, n, width, height, data } = resp;
  const ci = Math.round(x), cj = Math.round(y);
  let best = -Infinity, bestSky = 0;
  const s2 = 2 * PSF_SIGMA_PX * PSF_SIGMA_PX;
  for (let pj = cj - 1; pj <= cj + 1; pj++) {
    for (let pi = ci - 1; pi <= ci + 1; pi++) {
      if (pi < 0 || pj < 0 || pi >= width || pj >= height) continue;
      const p = pj * width + pi;
      if (!(q[p] > 0)) continue;
      let add = 0;
      for (let dj = -r; dj <= r; dj++) {
        const jj = pj + dj;
        if (jj < 0 || jj >= height) continue;
        for (let di = -r; di <= r; di++) {
          const ii = pi + di;
          if (ii < 0 || ii >= width || !Number.isFinite(data[jj * width + ii])) continue;
          add += kernel[(dj + r) * n + (di + r)] * amp * Math.exp(-((ii - x) ** 2 + (jj - y) ** 2) / s2);
        }
      }
      const total = snr[p] + add / (sigma * Math.sqrt(q[p]));
      if (total > best) { best = total; bestSky = snr[p]; }
    }
  }
  // No allowance for blending with a neighbour. It was tried -- a source above
  // threshold within 2 px counts the visit as lost -- and scored against the
  // real search body by body (60 bodies, five cells) it made the model worse:
  // 83% of calls right against 90% without it, turning six optimistic calls
  // into nine pessimistic ones. The detector keeps a body beside a star far
  // more often than that rule assumed.
  return { snr: best, onStatic: bestSky >= threshold, seen: best >= threshold && bestSky < threshold };
}

/** Whether a body would survive the search's static cut on its own: a
 *  detection is taken for a fixed source when it sits within `tolPx` of the
 *  body's own detections in `minEpochs` visits, counting itself. */
export function selfStatic(points, tolPx, minEpochs = 3) {
  return points.map((a) => points.filter((b) => Math.hypot(a[0] - b[0], a[1] - b[1]) <= tolPx).length >= minEpochs);
}

/** The quick map: every cell predicted from the sky's response, no search run.
 *
 *  `frames` carry `{ i, mjd, grid, width, height, data, sigma, resampled,
 *  shrink }`. Returns `{ snr, au, frac, n, bodies }` with `frac[s][d]` the
 *  fraction of bodies predicted recovered at brightness `snr[s]` and distance
 *  `au[d]`, and `bodies` how many were simulated a cell.
 *
 *  `n` is null: nothing was planted and no search ran, so no cell has a count
 *  of recoveries to put a confidence on. It used to carry the 24 simulated
 *  bodies as if they were, and a cell predicted 24 of 24 then read as a 96%
 *  recovery chance -- on the North Ecliptic Pole's own frames, a 95% exclusion
 *  of a 53 R⊕ body at 1,000 AU, where the 3,072 bodies actually planted there
 *  exclude nothing at 95%. A prediction can say what is found half the time;
 *  an exclusion is a claim about misses, and needs them measured. */
export function quickMap(frames, raDeg, decDeg, {
  snrs = DEPTH_SNR, dists = DEPTH_AU, perCell = 24, threshold = 6.5, required = 6,
  staticTolPx = 1, scale = 6.15, seed = 7, centroidPx = 0.5, minSignificance = 5,
  minBaselineFraction = 0.25,
} = {}) {
  const mjds = frames.map((f) => f.mjd);
  const baseline = Math.max(...mjds) - Math.min(...mjds);
  // What the linker demands of a candidate once it has the detections: a fit
  // that separates parallax from drift and puts the parallax 5 sigma clear,
  // over a stretch of the baseline. A body seen only on the near half of its
  // loop -- the rest off the stamp -- has the visits and fails this, which the
  // first version of this model ignored and so promised 54% at 350 AU where
  // the real search managed 25%.
  const credible = (seenIdx, b) => {
    if (seenIdx.length < 3) return false;
    const ts = seenIdx.map((i) => frames[i].mjd);
    if (Math.max(...ts) - Math.min(...ts) < minBaselineFraction * baseline) return false;
    try {
      const sol = solveAstrometry(seenIdx.map((i) => {
        const [ra, dec] = b.sky.get(frames[i].i);
        return { mjd: frames[i].mjd, ra, dec, sigma: centroidPx * scale };
      }));
      return sol.separable && sol.significance >= minSignificance;
    } catch { return false; }
  };
  const resp = new Map(frames.map((f) => [f.i, skyResponse(f.data, f.width, f.height,
    { pixelNoiseShrink: f.shrink })]));
  const frac = snrs.map(() => dists.map(() => 0));
  dists.forEach((d, di) => {
    snrs.forEach((s, si) => {
      const bodies = placeBodies(frames, raDeg, decDeg, d, perCell, cellSeed(seed, si, di), { scale });
      const amp = cellAmplitude(s, frames);
      let got = 0;
      for (const b of bodies) {
        const seen = [], idx = [];
        frames.forEach((f, fi) => {
          const at = b.places.get(f.i);
          if (!at || at[0] < 0 || at[1] < 0 || at[0] > f.width - 1 || at[1] > f.height - 1) return;
          const pr = predictDetection(resp.get(f.i), at[0], at[1], amp, threshold);
          if (pr.seen) { seen.push(at); idx.push(fi); }
        });
        const stat = selfStatic(seen, staticTolPx);
        const movingIdx = idx.filter((_, k) => !stat[k]);
        if (movingIdx.length >= required && credible(movingIdx, b)) got++;
      }
      frac[si][di] = bodies.length ? got / bodies.length : NaN;
    });
  });
  return { snr: snrs, au: dists, frac, n: null, bodies: perCell, kind: 'quick' };
}

/** Score one faithful scan: which planted bodies the search got back.
 *
 *  A body counts as recovered when some candidate has at least `required`
 *  members within `matchArcsec` of where the body was planted in their visits
 *  -- the test the console's own injection check applies. */
export function scoreRecovery(bodies, report, { required, matchArcsec }) {
  return bodies.map((b) => {
    let best = 0;
    for (const t of report.tracklets || []) {
      let k = 0;
      for (const m of t.members) {
        const at = b.sky.get(m.epoch);
        if (at && angularSeparation(m.raDeg, m.decDeg, at[0], at[1]) * 3600 <= matchArcsec) k++;
      }
      best = Math.max(best, k);
    }
    return best >= required;
  });
}

/** Per-visit AB magnitude of a body planted at `snr` sigma: its total planted
 *  flux, through the frame's pixel solid angle. `sigma` is the frame's pixel
 *  scatter in MJy/sr. The planted profile is summed, not assumed, so this is
 *  the magnitude of exactly what was put into the pixels. */
export function snrToAb(snr, sigma, pixelAreaSr, resampled) {
  return fluxToAb(plantAmplitude(snr, sigma, resampled) * psfFluxIntegral() * 1e6 * pixelAreaSr);
}

/** The magnitude a whole cell plants, from the frames themselves. The same
 *  number `snrToAb` gives for the median visit, and the axis the exclusions
 *  are read against. */
export function cellAb(snr, frames, pixelAreaSr) {
  return fluxToAb(cellAmplitude(snr, frames) * psfFluxIntegral() * 1e6 * pixelAreaSr);
}

/** Completeness at an arbitrary brightness and one of the map's distances,
 *  interpolated in log(S/N). Below the faintest row it is the faintest row's
 *  value (a floor, since fainter cannot be easier), above the brightest the
 *  brightest's. */
export function completenessAt(map, snr, di) {
  const s = map.snr;
  if (snr <= s[0]) return map.frac[0][di];
  if (snr >= s[s.length - 1]) return map.frac[s.length - 1][di];
  let k = 0;
  while (snr > s[k + 1]) k++;
  const t = (Math.log(snr) - Math.log(s[k])) / (Math.log(s[k + 1]) - Math.log(s[k]));
  return map.frac[k][di] + t * (map.frac[k + 1][di] - map.frac[k][di]);
}

//: The bodies the exclusion table names, in Earth radii. Physical anchors
//: rather than a round ladder: a reader knows how big Neptune is, and "4 R-Earth"
//: means nothing until you say it is Neptune. Equatorial radii over Earth's.
export const BODY_RADII = [
  { radius: 1, label: '1 R⊕', name: 'Earth' },
  { radius: 1.6, label: '1.6 R⊕', name: 'a super-Earth' },
  { radius: 3.86, label: '3.9 R⊕', name: 'Neptune' },
  { radius: 11.21, label: '11.2 R⊕', name: 'Jupiter' },
];

// ── confidence ─────────────────────────────────────────────────────────── //

//: The levels the page states exclusions at. 95% is the headline, as in the
//: literature it would be compared with; 90% beside it.
export const CONFIDENCE = [0.9, 0.95];

/** The chance a body in a cell is recovered, from `k` of `n` planted there:
 *  (k + 1) / (n + 2), not k / n. A cell where all 48 came back has not shown
 *  that the search never misses -- 48 trials cannot tell 100% from 98% -- and
 *  an exclusion is exactly a claim about the misses. This is the chance the
 *  next body planted there is recovered (Laplace's rule of succession), which
 *  is what a real body would be. A body is excluded at confidence CL where it
 *  is at least CL and the search found nothing. Mirrors
 *  `tracklet.stats.limits`; the golden vectors hold the two together. */
export function recoveryChance(k, n) {
  return (k + 1) / (n + 2);
}

/** The fewest of `n` planted bodies a cell must recover to exclude at `cl`, or
 *  null if all `n` are not enough: 47 of 48 for 95%, 44 of 48 for 90%. */
export function neededFor(n, cl) {
  // The tolerance absorbs the product's last bit: 0.9 * 50 need not come out
  // as exactly 45. No level in use puts the true value a hair above an integer.
  const k = Math.max(0, Math.ceil(cl * (n + 2) - 1 - 1e-9));
  return k <= n ? k : null;
}

/** The map with each cell's recovered fraction replaced by its recovery
 *  chance. A cell without a count -- a live run's cell not measured yet, still
 *  showing the estimate, or any cell of the quick map -- keeps its fraction.
 *  "Recovered half the time" reads the same off either, since the correction
 *  is affine in k / n and fixes one half. */
export function chanceMap(map) {
  if (map.chance) return map;
  return {
    ...map,
    chance: true,
    frac: map.frac.map((row, si) => row.map((f, di) => {
      const n = map.n ? map.n[si][di] : 0;
      return n > 0 && Number.isFinite(f) ? recoveryChance(Math.round(f * n), n) : f;
    })),
  };
}

/** The map an exclusion at `level` is read off: the recovery chance where a
 *  cell was measured, and zero where it was not. Above one half a cell has to
 *  show its misses to rule anything out, and a predicted cell has none to show,
 *  so it rules nothing out however high its estimate. At one half and below
 *  the estimate stands, as "found half the time" always has. */
export function levelMap(map, level) {
  const cm = chanceMap(map);
  if (!(level > 0.5)) return cm;
  return {
    ...cm,
    frac: cm.frac.map((row, si) => row.map((f, di) => ((map.n ? map.n[si][di] : 0) > 0 ? f : 0))),
  };
}

/** Whether any cell of the map was measured, rather than predicted. */
export function anyMeasured(map) {
  return Boolean(map.n && map.n.some((row) => row.some((v) => v > 0)));
}

/** The most bodies any measured cell recovers, as `{ k, n }`, or null for a
 *  map without counts: what a statement that nothing is excluded points to. */
export function bestCell(map) {
  let best = null;
  map.frac.forEach((row, si) => row.forEach((f, di) => {
    const n = map.n ? map.n[si][di] : 0;
    if (!(n > 0) || !Number.isFinite(f)) return;
    const k = Math.round(f * n);
    if (!best || recoveryChance(k, n) > recoveryChance(best.k, best.n)) best = { k, n };
  }));
  return best;
}

/** The smallest body this map reaches at `level`: the first radius on a fine
 *  geometric ladder whose exclusion range is not empty, or null if none is.
 *
 *  The table names Earth, a super-Earth, Neptune and Jupiter because those are
 *  the bodies the question is about, and in a single-visit search of one stamp
 *  the honest answer for all four is often "none". That is worth printing, but
 *  four rows of "none" look like a broken page rather than a result, so the
 *  map's own reach goes in beside them and the two together say what happened:
 *  not these, and here is what it would take. */
export function smallestReached(map, { albedo = 0.30, band = 'z', level = 0.5, abOfSnr, max = 60 } = {}) {
  const radii = [];
  for (let R = 0.5; R <= max; R *= 1.04) radii.push(Number(R.toFixed(3)));
  return exclusion(map, { radii, albedo, band, level, abOfSnr }).find((r) => r.range) || null;
}

// ── more than one field ────────────────────────────────────────────────── //

/** Completeness of a baked map at an arbitrary planted magnitude, at one of
 *  its distances, interpolated between the rows either side.
 *
 *  In magnitude, not in sigma, because a sigma is a different brightness in
 *  every field: these four differ by about half a magnitude in what their
 *  median visit's sigma is worth. `map.ab` is the axis the file carries, and it
 *  descends as the rows brighten. Outside the rows it is flat -- the faintest
 *  row's value below, the brightest row's above -- since a map cannot say what
 *  it did not plant. */
export function completenessAtAb(map, ab, di) {
  const a = map.ab;
  if (!a || !a.length) return NaN;
  if (ab >= a[0]) return map.frac[0][di];
  if (ab <= a[a.length - 1]) return map.frac[a.length - 1][di];
  let k = 0;
  while (k + 2 < a.length && ab < a[k + 1]) k++;
  const t = (ab - a[k]) / (a[k + 1] - a[k]);
  return map.frac[k][di] + t * (map.frac[k + 1][di] - map.frac[k][di]);
}

/** What several fields say together, distance by distance: the chance the
 *  search would have found a body of this radius had it been somewhere in the
 *  sky these maps cover, which is each field's recovery chance weighted by the
 *  area it speaks for.
 *
 *  The fields are disjoint patches, so this is the honest aggregate of a survey
 *  made of them -- not a claim about the sky between them, and the total area
 *  is quoted beside it because 0.15 square degrees of 41,253 is the whole
 *  result's real size. */
export function combined(maps, { radius, albedo = 0.30, band = 'z' }) {
  maps = maps.map(chanceMap);
  const au = maps[0].au;
  const area = maps.reduce((s, m) => s + (m.areaDeg2 || 0), 0);
  const frac = au.map((d, di) => {
    const ab = apparentMag(radius, d, albedo, { band });
    return area > 0
      ? maps.reduce((s, m) => s + (m.areaDeg2 || 0) * completenessAtAb(m, ab, di), 0) / area
      : NaN;
  });
  return { au, frac, areaDeg2: area, fields: maps.length };
}

/** The same, as an exclusion: for each radius, the run of distances at which
 *  the fields together would have recovered the body at least `level` of the
 *  time. Shaped like `exclusion` so the page can draw one table either way. */
export function combinedExclusion(maps, { radii = BODY_RADII.map((b) => b.radius),
                                          albedo = 0.30, band = 'z', level = 0.5 } = {}) {
  return radii.map((R) => {
    const c = combined(maps, { radius: R, albedo, band });
    const ok = c.frac.map((f) => f >= level);
    const first = ok.indexOf(true);
    if (first < 0) return { radius: R, range: null, frac: c.frac };
    let last = first;
    while (last + 1 < ok.length && ok[last + 1]) last++;
    return { radius: R, range: [c.au[first], c.au[last]], open: last === ok.length - 1,
             frac: c.frac };
  });
}

/** The smallest body the fields together reach at `level`, as `smallestReached`
 *  does for one. */
export function combinedReach(maps, { albedo = 0.30, band = 'z', level = 0.5, max = 60 } = {}) {
  const radii = [];
  for (let R = 0.5; R <= max; R *= 1.04) radii.push(Number(R.toFixed(3)));
  return combinedExclusion(maps, { radii, albedo, band, level }).find((r) => r.range) || null;
}

//: Each abOfSnr's inversions, by magnitude. A draw of the Depth page asks for
//: the same body at the same distance once per confidence level, and for the
//: smallest body reached at each level twice over -- six identical 80-step
//: bisections per body and distance, 470,000 calls of abOfSnr, on every step
//: of the albedo slider. The same function and magnitude give the same
//: answer, so each is worked out once per function.
const inversions = new WeakMap();
//: How many inversions one function keeps: about ten albedos' worth of a
//: draw. Its callers keep a function for as long as its scale holds, and the
//: albedo slider alone has 79 positions.
const KEEP_INVERSIONS = 10000;

/** S/N for a magnitude, by bisection on the monotonic abOfSnr. */
function inverse(abOfSnr) {
  let memo = inversions.get(abOfSnr);
  if (!memo) {
    memo = new Map();
    // Only a function can be a key; anything else fails as it did, when called.
    if (typeof abOfSnr === 'function') inversions.set(abOfSnr, memo);
  }
  return (m) => {
    let s = memo.get(m);
    if (s === undefined) {
      let lo = 0.01, hi = 1e6;
      for (let k = 0; k < 80; k++) {
        const mid = Math.sqrt(lo * hi);
        if (abOfSnr(mid) > m) lo = mid; else hi = mid;
      }
      s = Math.sqrt(lo * hi);
      if (memo.size >= KEEP_INVERSIONS) memo.clear();
      memo.set(m, s);
    }
    return s;
  };
}

/** Turn the map into exclusions: for each radius, the distances at which a
 *  body of that radius and albedo would be recovered at least `level` of the
 *  time -- its recovery chance, so an exclusion at confidence CL is `level`
 *  CL. Returned as the contiguous run of map distances satisfying it that
 *  contains the nearest such distance, `[lo, hi]` in AU, or null for none.
 *
 *  The body's per-visit brightness at each distance is its reflected-light
 *  magnitude converted to the search's sigma through the field's own noise
 *  (`abOfSnr`, which is monotonic, is inverted numerically). */
export function exclusion(map, { radii = [1, 2, 4, 10], albedo = 0.30, band = 'z', level = 0.5, abOfSnr }) {
  const snrOfAb = inverse(abOfSnr);
  map = levelMap(map, level);
  return radii.map((R) => {
    const ok = map.au.map((d, di) => completenessAt(map, snrOfAb(apparentMag(R, d, albedo, { band })), di) >= level);
    const first = ok.indexOf(true);
    if (first < 0) return { radius: R, range: null };
    let last = first;
    while (last + 1 < ok.length && ok[last + 1]) last++;
    return { radius: R, range: [map.au[first], map.au[last]], open: last === ok.length - 1 };
  });
}
