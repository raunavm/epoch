// Turning per-epoch detections into candidate moving objects, in the browser.
//
// The mirror of `src/tracklet/search/link.py`; that module's docstring is the
// explanation. In one line: reject everything that stood still, then link what
// is left against the geometry, then let the five-parameter solver decide what
// the distance actually is.

import {
  YEAR_DAYS, apparentPlace, apparentRaDec, earthState, mjdToJdTt,
  orbitalRateArcsecPerYear, solveAstrometry, solutionPlace,
} from './ephem.js';
import { passesGate } from './candscore.js';

const DEG = Math.PI / 180;

/** Great-circle separation in arcsec. Haversine, so it holds at the pole --
 *  and SPHEREx's deepest field is 23 degrees from one. */
export function sepArcsec(ra1, dec1, ra2, dec2) {
  const d1 = dec1 * DEG, d2 = dec2 * DEG;
  const dd = d2 - d1, da = (ra2 - ra1) * DEG;
  const h = Math.sin(dd / 2) ** 2 + Math.cos(d1) * Math.cos(d2) * Math.sin(da / 2) ** 2;
  return 2 * Math.asin(Math.min(1, Math.sqrt(h))) / DEG * 3600;
}

/** Exact inverse of apparentPlace: the heliocentric direction of something that
 *  appears at (ra, dec) if it is dAu from the Sun.
 *
 *  With u the apparent unit vector, E Earth's position and rho the unknown
 *  geocentric range, the body is at E + rho*u and that has length d:
 *      rho^2 + 2 rho (E.u) + |E|^2 - d^2 = 0
 *  Solved, not iterated, and not approximated by subtracting a first-order
 *  parallax offset -- at 40 AU that shortcut is 2.5% out. Returns null where no
 *  such body exists along that line of sight. */
export function helioDirection(raApp, decApp, dAu, mjdUtc, earth = null) {
  const a = raApp * DEG, d = decApp * DEG, cd = Math.cos(d);
  const u = [cd * Math.cos(a), cd * Math.sin(a), Math.sin(d)];
  const e = earth || earthState(mjdToJdTt(mjdUtc));
  const b = e.x * u[0] + e.y * u[1] + e.z * u[2];
  const disc = b * b + dAu * dAu - (e.x * e.x + e.y * e.y + e.z * e.z);
  if (disc < 0) return null;
  const rho = -b + Math.sqrt(disc);
  if (rho <= 0) return null;
  const v = [e.x + rho * u[0], e.y + rho * u[1], e.z + rho * u[2]];
  const n = Math.hypot(v[0], v[1], v[2]);
  return [((Math.atan2(v[1], v[0]) / DEG) % 360 + 360) % 360,
          Math.asin(Math.max(-1, Math.min(1, v[2] / n))) / DEG];
}

/** Fewest visits a candidate may be built from, whatever the field.
 *
 *  Measured, not chosen, and the measurement is the floor's own case: the
 *  four-visit ecliptic-plane field, where `requiredEpochs` returns exactly
 *  this. It holds no known mover, so every candidate is wrong -- three visits
 *  gives 27 of them and four gives none.
 *
 *  This note used to add "four returns none on either", with the pole field as
 *  the other one, and the floor does not decide that field: twelve visits ask
 *  for six, and it takes six. At four the pole field still returns 53, and the
 *  fall is 773 / 53 / 6 / 0 across three, four, five and six. Re-measure with
 *  `scripts/measure_search.py --sweep`. */
export const MIN_EPOCHS_FLOOR = 4;

/** Fewest visits a candidate may be built from on a field of six or more.
 *  The mirror of ROOMY_FLOOR in link.py, whose note has the measurement: on
 *  every field re-baked with all its separate days, four or five required
 *  still let false candidates through on fields of six to twenty-five
 *  visits, and six let none. Fields of five or fewer keep four, which is what
 *  the anticentre's blind search ran with. */
export const ROOMY_FLOOR = 6;

/** How many visits a candidate must appear in, for a field with this many.
 *
 *  Half the visits, never fewer than the floor: four, or six once there are
 *  six visits. The *fraction* is what carries the trials factor -- more visits
 *  means more detections means more combinations to be fooled by. */
export function requiredEpochs(nEpochs) {
  const floor = nEpochs >= ROOMY_FLOOR ? ROOMY_FLOOR : MIN_EPOCHS_FLOOR;
  return Math.max(floor, Math.ceil(nEpochs / 2));
}

/** Log-spaced, because parallax goes as 1/d and a linear grid would spend its
 *  resolution where the observable barely changes. */
export function distanceGrid(loAu = 30, hiAu = 2000, n = 40) {
  if (n < 2) return [loAu];
  const r = Math.log(hiAu / loAu) / (n - 1);
  return Array.from({ length: n }, (_, k) => loAu * Math.exp(r * k));
}

/** A grid on the sky in which two points within `tolArcsec` of each other
 *  always share a cell or sit in neighbouring ones -- anywhere on the sky.
 *
 *  Rows are tol high in declination, which is exact at every declination.
 *  Columns cannot be tol wide everywhere: a degree of right ascension is
 *  cos(dec) degrees on the sky, so near a pole a separation of tol spans far
 *  more of it than at the equator. Each column is therefore as wide as tol can
 *  span at the most polar of `decs` pushed a further tol poleward (so a query
 *  that does match one of them is covered too), and the columns divide 360°
 *  exactly, so the one after 359.99° is the one at 0°. Where tol reaches the
 *  pole it spans every right ascension, and there is one column.
 *
 *  This replaced one cos(dec) taken at the field's median declination, which
 *  held for a baked stamp a third of a degree across and nowhere else: a field
 *  straddling right ascension 0 split its stars into columns a whole sky
 *  apart, so none of them was cut as fixed, and cells were narrower than tol
 *  on the polar side of the median. */
export function skyCells(decs, tolArcsec) {
  const tol = tolArcsec / 3600;
  let far = 0;
  for (const d of decs) if (Math.abs(d) > far) far = Math.abs(d);
  const cosd = Math.cos(Math.min(90, far + tol) * DEG);
  // sin^2(sep/2) >= cos(d1) cos(d2) sin^2(dRA/2), so sep <= tol bounds dRA.
  const s = Math.sin(tol * DEG / 2) / cosd;
  const span = s < 1 ? 2 * Math.asin(s) / DEG * (1 + 1e-9) : 360;
  let ncol = Math.floor(360 / span);
  if (!(ncol >= 3)) ncol = 1;
  const width = 360 / ncol, rowH = tol * (1 + 1e-9);
  const col = (ra) => ((Math.floor(ra / width) % ncol) + ncol) % ncol;
  return {
    ncol,
    row: (dec) => Math.floor(dec / rowH),
    col,
    around: (c) => (ncol === 1 ? [0] : [(c + ncol - 1) % ncol, c, (c + 1) % ncol]),
    // The k-th of the columns `around` lists (k = 0, 1, 2), or -1 past the
    // last: for the lookup the linker makes millions of times, which should
    // not build an array each time it asks.
    near: (c, k) => (ncol === 1 ? (k === 0 ? 0 : -1) : (c + ncol - 1 + k) % ncol),
  };
}

/** `items` bucketed in `cells`, as row -> column -> items in their order.
 *  Keyed by the cell's two integers rather than a string made of them: the
 *  linker looks cells up millions of times, and building and hashing a string
 *  for each was most of the cost of the search. */
function bucketCells(items, cells, decOf, raOf) {
  const grid = new Map();
  for (const it of items) {
    const r = cells.row(decOf(it)), c = cells.col(raOf(it));
    let line = grid.get(r);
    if (!line) grid.set(r, (line = new Map()));
    let bucket = line.get(c);
    if (!bucket) line.set(c, (bucket = []));
    bucket.push(it);
  }
  return grid;
}

/** True for detections that keep turning up in the same place.
 *
 *  `minEpochs` distinct epochs must agree. Two is too few: two detections
 *  agreeing once is what a slow body near the turning point of its parallax
 *  ellipse looks like, and the third epoch is what makes it a pattern. */
export function markStatic(dets, { tolArcsec = 6.2, minEpochs = 3 } = {}) {
  const n = dets.length;
  if (!n || !(tolArcsec > 0)) return new Array(n).fill(false);

  // Bucketed on the sky rather than compared pairwise. The honest version is
  // O(n^2), and a real twelve-epoch scan carries a few thousand detections --
  // several million great-circle separations, and seconds of someone's
  // attention. Two detections can only match if they share a cell or touch
  // one, so a grid of tol-sized cells makes the sweep linear without changing
  // any answer: see `skyCells` for why that holds anywhere on the sky.
  const cells = skyCells(dets.map((d) => d.decDeg), tolArcsec);
  const grid = bucketCells(dets.keys(), cells, (i) => dets[i].decDeg, (i) => dets[i].raDeg);

  const out = new Array(n);
  for (let i = 0; i < n; i++) {
    const a = dets[i];
    const hits = new Set([a.epoch]);
    const ky = cells.row(a.decDeg), kx = cells.col(a.raDeg);
    for (let dy = -1; dy <= 1; dy++) {
      const line = grid.get(ky + dy);
      if (!line) continue;
      for (const cx of cells.around(kx)) {
        const bucket = line.get(cx);
        if (!bucket) continue;
        for (const j of bucket) {
          const b = dets[j];
          if (j === i || hits.has(b.epoch)) continue;
          if (sepArcsec(a.raDeg, a.decDeg, b.raDeg, b.decDeg) <= tolArcsec) hits.add(b.epoch);
        }
      }
    }
    out[i] = hits.size >= minEpochs;
  }
  return out;
}

/** Distance beyond which the static cut removes a body as a star, for these
 *  visits. The mirror of `blind_beyond` in link.py, whose docstring is the
 *  explanation: the count of a body's sightings that survive the cut falls as
 *  its parallax shrinks, and past the distance where fewer than `required`
 *  are left the search is blind. That distance depends on when the visits were
 *  made -- twelve visits forty days apart reach about 22,600 AU at the pole, six
 *  made on three nights reach nowhere -- and it used to be quoted as
 *  206265/tol whatever the visits. For a body at rest, at the detections' mean
 *  position; 0 when not even `nearAu` survives, Infinity when the cut can never
 *  gather `staticMinEpochs` sightings. */
export function blindBeyond(dets, { tolArcsec = 6.2, staticMinEpochs = 3, required = 4, nearAu = 30 } = {}) {
  if (!(tolArcsec > 0)) return Infinity;
  const mjdOf = new Map();
  let sx = 0, sy = 0, sz = 0;
  for (const d of dets) {
    if (!mjdOf.has(d.epoch)) mjdOf.set(d.epoch, d.mjd);
    const a = d.raDeg * DEG, b = d.decDeg * DEG;
    sx += Math.cos(b) * Math.cos(a); sy += Math.cos(b) * Math.sin(a); sz += Math.sin(b);
  }
  const mjds = [...mjdOf.keys()].sort((a, b) => a - b).map((e) => mjdOf.get(e));
  if (mjds.length < staticMinEpochs || Math.hypot(sx, sy, sz) === 0) return Infinity;
  const ra0 = (((Math.atan2(sy, sx) / DEG) % 360) + 360) % 360;
  const dec0 = Math.atan2(sz, Math.hypot(sx, sy)) / DEG;
  const kept = (dAu) => {
    const places = mjds.map((m) => apparentPlace(ra0, dec0, dAu, m));
    let n = 0;
    for (const a of places) {
      let near = 0;
      for (const b of places) if (sepArcsec(a.raDeg, a.decDeg, b.raDeg, b.decDeg) <= tolArcsec) near++;
      if (near < staticMinEpochs) n++;
    }
    return n;
  };
  let lo = nearAu;
  // Where the whole ellipse fits inside tol, nothing survives past this.
  let hi = 2 * 206264.806247 / tolArcsec;
  if (kept(lo) < required) return 0;
  if (kept(hi) >= required) return Infinity;
  while (hi / lo > 1 + 1e-10) {
    const mid = Math.sqrt(lo * hi);
    if (kept(mid) >= required) lo = mid; else hi = mid;
  }
  return lo;
}

/** How many separate nights `mjds` fall on: runs split by a day's gap. */
function nights(mjds) {
  const ts = [...mjds].sort((a, b) => a - b);
  let n = ts.length ? 1 : 0;
  for (let k = 1; k < ts.length; k++) if (ts[k] - ts[k - 1] > 1) n++;
  return n;
}

/** Where a fitted solution says the body is at `mjd`: `solutionPlace`, the
 *  model the fit minimised and the one the overlay draws as the white curve, so
 *  the linker and the picture cannot disagree about what the fit means. */
export function predict(sol, mjd) {
  const p = solutionPlace(sol, mjd);
  return [p.ra, p.dec];
}

/**
 * Reject the fixed sky, then link the rest into candidate trajectories.
 *
 * `dets` are `{epoch, mjd, raDeg, decDeg, snr, x, y}`. Returns a report whose
 * counts are worth showing even when it finds nothing -- a null result with
 * "480 detections, 435 static, 45 moving" beside it is a statement; a blank
 * list is not.
 *
 * `minEpochs` is derived from the field unless you pass one -- see
 * `requiredEpochs`. Whatever it works out to, it is more than the three the
 * geometry needs, and the extra is the trials factor rather than timidity. Five parameters fitted to
 * three positions leaves one degree of freedom, so nearly any three detections
 * fit something -- and the search tries every triple, which on this field is
 * over half a million hypotheses. At three epochs the real NEP data returns 773
 * candidates, at a median parallax significance of 164 sigma, and all of them
 * are noise. Someone clicking three times has a trials factor of one and
 * three epochs is genuinely enough for them; an exhaustive search does not get
 * to quote the same threshold.
 *
 * Six is measured, not chosen. Against the real NEP field -- which holds no
 * known moving object, so every candidate is by definition false -- at 6.5
 * sigma: three epochs gives 773 false candidates, four 53, five 6, six 0
 * (`scripts/measure_search.py --sweep`). A source planted at 12 sigma is
 * recovered in ten of the twelve, so it clears the cut with room to spare.
 */
export function findTracklets(dets, {
  matchArcsec = 12,
  staticTolArcsec = 6.2,
  minEpochs = null,
  staticMinEpochs = 3,
  distances = null,
  maxPerEpoch = 40,
  minBaselineFraction = 0.25,
  driftTolerance = 3,
  sigmaArcsec = 2,
  seedSlack = 4,
  maxReducedChi2 = 4,
  minSignificance = 5,
  gate = null,
  rbCut = null,
  onProgress = null,
} = {}) {
  const grid = distances || distanceGrid();
  const nEpochs = new Set(dets.map((d) => d.epoch)).size;
  const required = minEpochs == null ? requiredEpochs(nEpochs) : minEpochs;

  // Order matters, and it used to be wrong. Trimming to the brightest
  // `maxPerEpoch` *before* the static cut spends the whole budget on stars --
  // the fixed sky is brighter than anything moving, so on the real NEP stamp
  // the top 40 of 288 detections per epoch were 40 stars and a planted source
  // never reached the linker. Reject the fixed sky first, then spend the
  // budget on what is left.
  //
  // Deliberately not `minEpochs`: these count different things. Three
  // sightings in one place is already a pattern and calling it a star is
  // cheap; four detections on one trajectory is a claim and costs more.
  const isStatic = markStatic(dets,
    { tolArcsec: staticTolArcsec, minEpochs: staticMinEpochs });
  const survivors = dets.filter((_, i) => !isStatic[i]);

  const byEpoch = new Map();
  for (const d of survivors) {
    if (!byEpoch.has(d.epoch)) byEpoch.set(d.epoch, []);
    byEpoch.get(d.epoch).push(d);
  }
  // Brightest first, unless the real/bogus network has scored them: then the
  // detections it calls real come first and brightness orders within each
  // group. On the crowded South Ecliptic Pole two visits carry 98 and 105
  // moving detections, most of them the residue of stars, and brightest-first
  // spent the budget on those and dropped 13 sigma planted bodies.
  const real = (d) => (rbCut != null && Number.isFinite(d.rb) && d.rb < rbCut ? 0 : 1);
  const moving = [];
  for (const ep of [...byEpoch.keys()].sort((a, b) => a - b)) {
    moving.push(...byEpoch.get(ep).sort((a, b) => (real(b) - real(a)) || (b.snr - a.snr))
      .slice(0, maxPerEpoch));
  }
  const mjds = [...new Set(dets.map((d) => d.mjd))].sort((a, b) => a - b);
  const baseline = mjds.length > 1 ? mjds[mjds.length - 1] - mjds[0] : 0;

  const report = {
    nDetections: dets.length,
    nStatic: isStatic.filter(Boolean).length,
    nMoving: survivors.length,
    // Reported because a silent truncation reads as "searched everything".
    nLinked: moving.length,
    nEpochs,
    requiredEpochs: required,
    // Set when the search could not run at all, which is a different claim
    // from "searched and found nothing". The panel must not conflate them.
    blocked: '',
    // Beyond this a body at rest sits still in enough of these visits to be
    // cut as a star: see `blindBeyond`. A search with a ceiling says where.
    maxDistanceAu: blindBeyond(dets, { tolArcsec: staticTolArcsec, staticMinEpochs,
                                       required, nearAu: Math.min(...grid) }),
    tracklets: [],
  };
  if (nEpochs < required) {
    report.blocked = `${nEpochs} visits is too few to search: a candidate needs `
      + `${required}, and below that the trials factor swamps any result`;
    return report;
  }
  // Enough visits, made too close together: every body the grid could propose
  // sits still in enough of them to be cut as a star. Same distinction.
  if (report.maxDistanceAu === 0) {
    report.blocked = `${nEpochs} visits on ${nights(mjds)} nights are too close in time to `
      + `search: from ${Math.min(...grid).toFixed(0)} AU out, a body sits still in enough of them `
      + `to be cut as a star, leaving fewer than the ${required} a candidate needs`;
    return report;
  }
  if (moving.length < required || baseline <= 0) return report;

  const epochs = [...new Set(moving.map((d) => d.epoch))].sort((a, b) => a - b);
  if (epochs.length < required) return report;

  const perEpoch = new Map();
  const epochMjd = new Map();
  for (const d of moving) {
    if (!perEpoch.has(d.epoch)) perEpoch.set(d.epoch, []);
    perEpoch.get(d.epoch).push(d);
    epochMjd.set(d.epoch, d.mjd);
  }
  // Twelve Earth positions, computed once. Without this the ephemeris series
  // is re-evaluated inside every hypothesis and dominates the whole search.
  const earthAt = new Map();
  for (const [ep, mjd] of epochMjd) earthAt.set(ep, earthState(mjdToJdTt(mjd)));

  // Seeding is deliberately loose. A grid node is a few percent off the true
  // distance at best, and a few percent of a 400 arcsec parallax is tens of
  // arcsec -- so seeding at the final match radius would link only the epochs
  // where the grid happened to be close, and report one object as three
  // fragments. Seed wide; the refinement below tightens it.
  const seedArcsec = matchArcsec * seedSlack;
  const minGap = baseline * minBaselineFraction;

  // Two tables, built once, that the hypothesis loop would otherwise rebuild
  // millions of times.
  //
  // `helio` is the heliocentric direction each detection implies at each trial
  // distance. It depends on the detection and the distance and nothing else,
  // so there are `detections x distances` distinct values -- a few thousand --
  // against the ~2 million times the loop asks for one.
  const helio = new Map();
  for (const d of moving) {
    const row = new Array(grid.length);
    for (let k = 0; k < grid.length; k++) {
      row[k] = helioDirection(d.raDeg, d.decDeg, grid[k], d.mjd, earthAt.get(d.epoch));
    }
    helio.set(d, row);
  }
  // `index` buckets each epoch's detections on the sky so matching a predicted
  // position is a lookup in nine cells rather than a scan of the whole epoch.
  // One slot per epoch, in `perEpoch`'s order, with everything a hypothesis
  // asks of that epoch at hand rather than behind four map lookups.
  const slots = [...perEpoch].map(([ep, rows]) => ({
    mjd: epochMjd.get(ep), earth: earthAt.get(ep), index: buildIndex(rows, seedArcsec),
  }));

  // The fastest drift a bound body at each trial distance may show; see tryPair.
  const maxDrift = grid.map((d) => driftTolerance * orbitalRateArcsecPerYear(d));
  const seeds = new Map();
  let pairs = 0;
  for (let ia = 0; ia < epochs.length; ia++) {
    for (let ib = ia + 1; ib < epochs.length; ib++) {
      const ea = epochs[ia], eb = epochs[ib];
      if (Math.abs(epochMjd.get(eb) - epochMjd.get(ea)) < minGap) continue;
      for (const a of perEpoch.get(ea)) {
        const ha = helio.get(a);
        for (const b of perEpoch.get(eb)) {
          const hb = helio.get(b);
          for (let k = 0; k < grid.length; k++) {
            const t = tryPair(a, b, grid[k], ha[k], hb[k], maxDrift[k], slots, seedArcsec,
                              required);
            if (!t) continue;
            const prev = seeds.get(t.key);
            if (!prev || t.rmsArcsec < prev.rmsArcsec) seeds.set(t.key, t);
          }
          if (onProgress && (++pairs & 255) === 0) onProgress(pairs);
        }
      }
    }
  }

  const grown = [];
  for (const t of seeds.values()) {
    const g = grow(t, perEpoch, epochMjd, matchArcsec, sigmaArcsec, required);
    if (g) grown.push(g);
  }
  // A candidate whose reduced chi-squared is 12 is not "a detection with a
  // caveat" -- it is the statement that these detections do not lie on one
  // trajectory, which is the definition of not being an object. Listing those
  // anyway produced 163 candidates on a field with nothing in it, which is
  // indistinguishable from no search at all.
  const credible = grown.filter((t) => {
    const s = t.solution;
    return s && s.separable && Number.isFinite(s.reducedChi2)
      && s.reducedChi2 <= maxReducedChi2 && s.significance >= minSignificance
      && s.distanceAu > 0;
  });
  // The ML linking gate, when there is one, judges candidates before
  // duplicates are merged and not after. The loosened search proposes false
  // links that share detections with a real body and outnumber it by a member
  // or two; merged first, the false one survives as the representative, the
  // gate then rejects it, and the real body goes with it.
  let kept = credible;
  if (gate) {
    report.gate = { before: dedupe(credible.slice()).length };
    kept = credible.filter((t) => passesGate(t, gate));
  }
  report.tracklets = dedupe(kept).sort(
    (a, b) => (b.members.length - a.members.length) || (a.rmsArcsec - b.rmsArcsec));
  if (gate) report.gate.kept = report.tracklets.length;
  return report;
}

function keyOf(members) {
  return members.map((m) => `${m.epoch}:${m.raDeg.toFixed(7)}:${m.decDeg.toFixed(7)}`)
    .sort().join('|');
}


/** Bucket one epoch's detections on the sky, in cells of the match radius.
 *
 *  Same trick as the static cut, for the same reason: the hypothesis loop asks
 *  "is there a detection near here" a few million times, and a linear scan of
 *  every detection in the epoch is most of the cost of the search. Cells at
 *  least as wide as the radius mean the answer is always inside the nine cells
 *  around the query, so this is an exact speed-up and not an approximation. */
function buildIndex(rows, radiusArcsec) {
  const cells = skyCells(rows.map((d) => d.decDeg), radiusArcsec);
  return { grid: bucketCells(rows, cells, (d) => d.decDeg, (d) => d.raDeg), cells };
}

/** Nearest detection to (ra, dec) within `radius`, or null. `radius` must not
 *  exceed the one the index was built with. */
function nearest(idx, ra, dec, radius) {
  const { cells } = idx;
  const ky = cells.row(dec), kx = cells.col(ra);
  let hit = null, hitSep = radius;
  for (let dy = -1; dy <= 1; dy++) {
    const line = idx.grid.get(ky + dy);
    if (!line) continue;
    for (let k = 0; k < 3; k++) {
      const cx = cells.near(kx, k);
      if (cx < 0) break;
      const bucket = line.get(cx);
      if (!bucket) continue;
      for (const s of bucket) {
        const sp = sepArcsec(ra, dec, s.raDeg, s.decDeg);
        if (sp <= hitSep) { hit = s; hitSep = sp; }
      }
    }
  }
  return hit ? { hit, sep: hitSep } : null;
}

/** One (pair, distance) hypothesis. null if it does not hold up. */
function tryPair(a, b, dAu, h1, h2, maxDrift, slots, matchArcsec, minEpochs) {
  if (!h1 || !h2) return null;
  const dtYr = (b.mjd - a.mjd) / YEAR_DAYS;
  if (dtYr === 0) return null;

  // A physical prune, and a strong one. A bound body at d AU sweeps a known
  // rate on a circular orbit; `maxDrift` is `driftTolerance` (three) times that,
  // which covers eccentric and retrograde cases with room to spare. Faster than
  // that is not an object at this distance -- it is two unrelated detections
  // and a wrong guess.
  const sep = sepArcsec(h1[0], h1[1], h2[0], h2[1]);
  if (sep / Math.abs(dtYr) > maxDrift) return null;

  const dra = (((h2[0] - h1[0] + 540) % 360) - 180) / dtYr;
  const ddec = (h2[1] - h1[1]) / dtYr;

  // At most one member per epoch, since each slot is one epoch: the count of
  // members is the count of epochs they span.
  const members = [];
  let resid2 = 0;
  let left = slots.length;
  for (const slot of slots) {
    // Bail as soon as every remaining epoch hitting would still not reach the
    // minimum. Most hypotheses are wrong, and most of those are wrong by the
    // third epoch, so this skips the majority of the work.
    if (members.length + left < minEpochs) return null;
    left--;
    const yr = (slot.mjd - a.mjd) / YEAR_DAYS;
    const [ra, dec] = apparentRaDec(h1[0] + dra * yr, h1[1] + ddec * yr, dAu, slot.earth);
    const found = nearest(slot.index, ra, dec, matchArcsec);
    if (found) { members.push(found.hit); resid2 += found.sep * found.sep; }
  }
  if (members.length < minEpochs) return null;
  return {
    members, hypothesisAu: dAu, rmsArcsec: Math.sqrt(resid2 / members.length),
    key: keyOf(members), solution: null, note: '',
  };
}

/** Fit, re-predict from the fit, re-match. Repeat until the members settle.
 *
 *  This is where the distance grid stops mattering: the seed only has to be
 *  close enough to associate three detections, and from there the solution
 *  predicts every epoch far better than any grid node can. */
function grow(t, perEpoch, epochMjd, matchArcsec, sigmaArcsec, minEpochs) {
  let cur = t;
  for (let pass = 0; pass < 4; pass++) {
    fit(cur, sigmaArcsec);
    if (!cur.solution) return null;
    const members = [];
    let resid2 = 0;
    for (const [ep, rows] of perEpoch) {
      const [ra, dec] = predict(cur.solution, epochMjd.get(ep));
      let hit = null, hitSep = matchArcsec;
      for (const s of rows) {
        const sp = sepArcsec(ra, dec, s.raDeg, s.decDeg);
        if (sp <= hitSep) { hit = s; hitSep = sp; }
      }
      if (hit) { members.push(hit); resid2 += hitSep * hitSep; }
    }
    // One member at most per epoch, so this is the number of epochs.
    if (members.length < minEpochs) return null;
    const key = keyOf(members);
    const settled = key === cur.key;
    cur = {
      members, hypothesisAu: cur.hypothesisAu,
      rmsArcsec: Math.sqrt(resid2 / members.length), key, solution: null, note: '',
    };
    if (settled) break;
  }
  fit(cur, sigmaArcsec);
  return cur.solution ? cur : null;
}

function fit(t, sigmaArcsec) {
  const obs = [...t.members].sort((a, b) => a.mjd - b.mjd)
    .map((m) => ({ mjd: m.mjd, ra: m.raDeg, dec: m.decDeg, sigma: sigmaArcsec }));
  try {
    t.solution = solveAstrometry(obs);
  } catch (err) {
    t.solution = null;
    t.note = `fit failed: ${err.message}`;
    return;
  }
  const s = t.solution;
  if (!s.separable) t.note = 'parallax and drift are not separable at these epochs';
  else if (s.reducedChi2 > 10) t.note = 'poor fit, probably not one object';
  else if (s.significance < 3) t.note = 'parallax not significant; a lower bound at best';
  else t.note = '';
}

/** One object, one candidate.
 *
 *  Every pair of a body's own detections proposes that body, so a real object
 *  arrives here dozens of times. A subset test is not enough: after refinement
 *  two proposals of the same object can differ by one member each, so neither
 *  contains the other. Sharing two detections is the test that means "same
 *  object", because a detection belongs to one body. */
function dedupe(tracklets, share = 2) {
  const ids = (t) => new Set(t.members.map((m) => `${m.epoch}:${m.raDeg}:${m.decDeg}`));
  tracklets.sort((a, b) => (b.members.length - a.members.length) || (a.rmsArcsec - b.rmsArcsec));
  const kept = [];
  for (const t of tracklets) {
    const mine = ids(t);
    let dup = false;
    for (const k of kept) {
      let n = 0;
      for (const id of ids(k)) if (mine.has(id)) n++;
      if (n >= share) { dup = true; break; }
    }
    if (!dup) kept.push(t);
  }
  return kept;
}
