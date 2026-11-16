// Apparent motion of a distant body, and the inverse problem: distance from
// motion. A port of src/tracklet/physics/ephemeris.py, checked against golden
// vectors generated from it.
//
// The one idea the whole tool is built on: SPHEREx maps the entire sky every six
// months, and six months is exactly the baseline over which Earth swaps sides of
// its orbit. So consecutive all-sky maps are a parallax pair by construction,
// and the displacement of an object between them measures its distance:
//
//     parallax amplitude   pi = 206265 / d          arcsec
//     circular-orbit rate  mu = 1.296e6 / d^1.5     arcsec/yr
//     ratio over 6 months  0.6366 * sqrt(d)  ->  parallax wins beyond 2.47 AU
//
// At 500 AU the parallax swing is 825 arcsec over six months -- 134 SPHEREx
// pixels -- against 58 arcsec of orbital drift. That is not a subtle effect
// needing a pipeline to dig out. It is a shift you can see by blinking two
// images, which is what this tool asks people to do.

const DEG = Math.PI / 180;
export const ARCSEC_PER_RAD = 648000 / Math.PI;
export const LIGHT_DAYS_PER_AU = 1.495978707e11 / 299792458 / 86400;
export const YEAR_DAYS = 365.256363004;
export const TT_MINUS_UTC_S = 69.184;
export const OBLIQUITY_J2000_DEG = 84381.406 / 3600;
export const PRECESSION_DEG_PER_CENTURY = 5029.0966 / 3600;
const MJD_TO_JD = 2400000.5;
const J2000_JD = 2451545.0;

export const mjdToJdTt = (mjdUtc) => mjdUtc + MJD_TO_JD + TT_MINUS_UTC_S / 86400;

/** Earth's heliocentric equatorial position (AU), from the Meeus Ch.25 solar
 *  series, precessed to J2000 and rotated by the J2000 mean obliquity so it
 *  lands in the same frame as a SPHEREx RADESYS='ICRS' WCS.
 *
 *  Accuracy, measured against astropy/ERFA over the real 479-day NEP baseline:
 *  max position error 1.7e-4 AU, which for a body at 500 AU is 0.07 arcsec --
 *  0.011 SPHEREx pixels. The error enters predictions as delta/d, so it gets
 *  *better* the further away the target is, which is the regime that matters. */
export function earthState(jdTt) {
  const t = (jdTt - J2000_JD) / 36525;
  const l0 = 280.46646 + 36000.76983 * t + 0.0003032 * t * t;
  const m = 357.52911 + 35999.05029 * t - 0.0001537 * t * t;
  const e = 0.016708634 - 0.000042037 * t - 0.0000001267 * t * t;
  const mr = ((m % 360) + 360) % 360 * DEG;
  const c = (1.914602 - 0.004817 * t - 0.000014 * t * t) * Math.sin(mr)
          + (0.019993 - 0.000101 * t) * Math.sin(2 * mr)
          + 0.000289 * Math.sin(3 * mr);
  const nu = mr + c * DEG;
  const radius = 1.000001018 * (1 - e * e) / (1 + e * Math.cos(nu));
  const sunLonJ2000 = l0 + c - PRECESSION_DEG_PER_CENTURY * t;
  const lonE = (((sunLonJ2000 + 180) % 360) + 360) % 360;
  const lr = lonE * DEG;
  const xe = radius * Math.cos(lr), ye = radius * Math.sin(lr);
  const eps = OBLIQUITY_J2000_DEG * DEG;
  return {
    jdTt, x: xe, y: ye * Math.cos(eps), z: ye * Math.sin(eps),
    eclipticLonDeg: lonE, radiusAu: radius,
  };
}

export const parallaxArcsec = (dAu) => ARCSEC_PER_RAD / dAu;
export const distanceFromParallax = (piArcsec) => ARCSEC_PER_RAD / piArcsec;
export const orbitalRateArcsecPerYear = (dAu) => 2 * Math.PI * ARCSEC_PER_RAD / Math.pow(dAu, 1.5);

/** Heliocentric direction after `years` of uniform drift from (ra, dec).
 *
 *  A straight line in the tangent plane at the start: a great circle on the
 *  sky, and the motion `solveAstrometry` fits. Adding pmRa*t/cos(dec) to RA
 *  walks a small circle instead: 0.1" from this at 500 AU over the North
 *  Ecliptic Pole's 440 days, 109" at 50 AU, where a body planted that way came
 *  back from the fit at chi2/nu 3,700. Mirrors `drift_direction` in the Python. */
export function driftDirection(raDeg, decDeg, pmRaArcsecYr, pmDecArcsecYr, years) {
  return tanDeproject(pmRaArcsecYr * years / 3600, pmDecArcsecYr * years / 3600, raDeg, decDeg);
}

export function parallaxDominance(dAu, baselineDays = YEAR_DAYS / 2) {
  const years = baselineDays / YEAR_DAYS;
  const swing = 2 * parallaxArcsec(dAu) * Math.sin(Math.PI * years);
  const drift = orbitalRateArcsecPerYear(dAu) * years;
  return drift ? swing / drift : Infinity;
}

/** [semiMajor, semiMinor] of the annual wobble, arcsec.
 *  The **axis ratio is sin|beta| regardless of distance**, so the shape of the
 *  path says where you are looking and its size says how far the object is.
 *  Those two facts are independent, which is what makes the measurement work. */
export function parallacticEllipse(dAu, eclipticLatDeg) {
  const pi = parallaxArcsec(dAu);
  return [pi, pi * Math.abs(Math.sin(eclipticLatDeg * DEG))];
}

export function eclipticLatitudeDeg(raDeg, decDeg) {
  const eps = OBLIQUITY_J2000_DEG * DEG, a = raDeg * DEG, d = decDeg * DEG;
  return Math.asin(Math.max(-1, Math.min(1,
    Math.sin(d) * Math.cos(eps) - Math.cos(d) * Math.sin(eps) * Math.sin(a)))) / DEG;
}

export function eclipticLongitudeDeg(raDeg, decDeg) {
  const eps = OBLIQUITY_J2000_DEG * DEG, a = raDeg * DEG, d = decDeg * DEG;
  const y = Math.sin(a) * Math.cos(eps) + Math.tan(d) * Math.sin(eps);
  return ((Math.atan2(y, Math.cos(a)) / DEG) % 360 + 360) % 360;
}

/** [raDeg, decDeg] for an ecliptic direction, J2000.
 *
 *  The exact inverse of the two above, and the direction the *drawing* code
 *  needs: a globe or a diagram is given a latitude and has to find out where
 *  that lands on the sky, not the other way round. */
export function equatorialFromEcliptic(lonDeg, latDeg) {
  const eps = OBLIQUITY_J2000_DEG * DEG, b = latDeg * DEG, l = lonDeg * DEG;
  const sd = Math.sin(b) * Math.cos(eps) + Math.cos(b) * Math.sin(eps) * Math.sin(l);
  const dec = Math.asin(Math.max(-1, Math.min(1, sd))) / DEG;
  const y = Math.cos(b) * Math.sin(l) * Math.cos(eps) - Math.sin(b) * Math.sin(eps);
  const x = Math.cos(b) * Math.cos(l);
  return [((Math.atan2(y, x) / DEG) % 360 + 360) % 360, dec];
}

/** [F_alpha*, F_delta]: parallax displacement per unit parallax.
 *  F_a* is already the great-circle component, so there is no 1/cos(dec) here.
 *  Adding one is the classic bug, and it blows up exactly at the ecliptic poles
 *  where SPHEREx's deepest fields are. */
export function parallaxFactors(raDeg, decDeg, earth) {
  const a = raDeg * DEG, d = decDeg * DEG;
  const { x, y, z } = earth;
  return [
    x * Math.sin(a) - y * Math.cos(a),
    x * Math.cos(a) * Math.sin(d) + y * Math.sin(a) * Math.sin(d) - z * Math.cos(d),
  ];
}

const unitFromRaDec = (ra, dec) => {
  const a = ra * DEG, d = dec * DEG, cd = Math.cos(d);
  return [cd * Math.cos(a), cd * Math.sin(a), Math.sin(d)];
};

const raDecFromVec = ([x, y, z]) => {
  const n = Math.hypot(x, y, z);
  return [((Math.atan2(y, x) / DEG) % 360 + 360) % 360,
          Math.asin(Math.max(-1, Math.min(1, z / n))) / DEG];
};

/** [raDeg, decDeg] of `apparentPlace`, and nothing else: the same arithmetic
 *  to the bit, without the range, light time and parallax offset that the
 *  linker's millions of predictions never read. `earth` is required. */
export function apparentRaDec(raHelio, decHelio, dAu, earth) {
  const a = raHelio * DEG, d = decHelio * DEG, cd = Math.cos(d);
  return raDecFromVec([dAu * (cd * Math.cos(a)) - earth.x, dAu * (cd * Math.sin(a)) - earth.y,
                       dAu * Math.sin(d) - earth.z]);
}

/** Where a body at heliocentric direction (ra,dec) and distance d appears.
 *  Exact in the geometry -- full vector difference, no 1/d expansion. */
export function apparentPlace(raHelio, decHelio, dAu, mjdUtc, earth = null) {
  const s = unitFromRaDec(raHelio, decHelio);
  // `earth` is an optional precomputed state for this instant. The linker
  // evaluates this millions of times across a handful of distinct epochs, and
  // earthState is a dozen trig calls; passing it in turns the ephemeris from
  // the dominant cost into twelve calls. Explicit rather than a hidden memo,
  // so nothing caches a value keyed on a float nobody is tracking.
  const e = earth || earthState(mjdToJdTt(mjdUtc));
  const rho = [dAu * s[0] - e.x, dAu * s[1] - e.y, dAu * s[2] - e.z];
  const range = Math.hypot(rho[0], rho[1], rho[2]);
  const [ra, dec] = raDecFromVec(rho);
  const dot = (rho[0] * s[0] + rho[1] * s[1] + rho[2] * s[2]) / range;
  return {
    raDeg: ra, decDeg: dec, rangeAu: range,
    lightTimeDays: range * LIGHT_DAYS_PER_AU,
    parallaxOffsetArcsec: Math.acos(Math.max(-1, Math.min(1, dot))) / DEG * 3600,
  };
}

// `apparentTrack` was here: a map over `apparentPlace` returning bare
// (ra, dec) pairs, mirroring `apparent_track` in the Python. It had no callers.
// Every place this app draws a track builds it inline instead -- render.js in
// three places, plus figures.js and app.js -- because each needs a
// different shape out of it: arcsec offsets from the field centre, canvas
// coordinates, or a curve sub-sampled far finer than the epoch list. Bare
// (ra, dec) pairs are the one thing none of them wanted, which is why it was
// never used. An uncovered mirror with no callers is only somewhere for the two
// languages to drift unobserved, so it is gone rather than pinned.

// -- linear algebra, just enough --------------------------------------- //

/** Solve A x = b and return [x, A^-1] for symmetric positive-definite A.
 *  Cholesky rather than Gaussian elimination because it fails loudly on a
 *  negative pivot -- which happens exactly when the design is degenerate, and
 *  that is the case the caller must be told about rather than handed a
 *  plausible-looking number for. */
function choleskySolveAndInvert(a, b) {
  const n = a.length;
  const L = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let s = a[i][j];
      for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
      if (i === j) {
        if (s <= 0) throw new Error('normal matrix is not positive definite');
        L[i][j] = Math.sqrt(s);
      } else L[i][j] = s / L[j][j];
    }
  }
  const fwd = (rhs) => {
    const y = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let s = rhs[i];
      for (let k = 0; k < i; k++) s -= L[i][k] * y[k];
      y[i] = s / L[i][i];
    }
    return y;
  };
  const back = (y) => {
    const x = new Float64Array(n);
    for (let i = n - 1; i >= 0; i--) {
      let s = y[i];
      for (let k = i + 1; k < n; k++) s -= L[k][i] * x[k];
      x[i] = s / L[i][i];
    }
    return x;
  };
  const x = back(fwd(b));
  const inv = Array.from({ length: n }, () => new Float64Array(n));
  for (let j = 0; j < n; j++) {
    const e = new Float64Array(n); e[j] = 1;
    const col = back(fwd(e));
    for (let i = 0; i < n; i++) inv[i][j] = col[i];
  }
  return [x, inv];
}

// -- the inverse problem ------------------------------------------------ //

import { tanProject, tanDeproject, angularSeparation } from './wcs.js';

/** Weighted least squares for [ra0, dec0, pmRa, pmDec, parallax].
 *
 *  Stage 1 is linear, because the parallax factors depend only on Earth's known
 *  position -- the same five-parameter fit Hipparcos and Gaia do for stars,
 *  applied to a solar-system body, with a unique solution and a real covariance
 *  matrix.
 *
 *  Stage 2 matters: those equations are first order in 1/d, so they carry a
 *  fractional error of order r_E/d -- 0.2% at 500 AU but 2.5% at 40 AU, which
 *  against 0.15 arcsec astrometry is a reduced chi-square of order 1e6 while the
 *  fitted distance still looks plausible. So the linear answer seeds
 *  Gauss-Newton against apparentPlace(), the exact model.
 *
 *  Needs 3 epochs. Two do not separate parallax from proper motion, so this
 *  throws instead of guessing -- twoEpochDistance() returns the range a
 *  two-epoch pair can actually support. */
export function solveAstrometry(obs, { refine = true } = {}) {
  if (obs.length < 3) {
    throw new Error(`need >= 3 epochs to separate parallax from proper motion, got ${obs.length}`);
  }
  // Refused by name, as solve_astrometry refuses them: a NaN position or a
  // zero uncertainty came back as a solution of NaNs with an infinite
  // significance, which anything reading the significance alone believes.
  obs.forEach((o, k) => {
    if (![o.mjd, o.ra, o.dec].every(Number.isFinite)) {
      throw new Error(`observation ${k} has a time or position that is not a number`);
    }
    if (Math.abs(o.dec) > 90) throw new Error(`observation ${k} has declination ${o.dec}, beyond a pole`);
    if (!(Number.isFinite(o.sigma) && o.sigma > 0)) {
      throw new Error(`observation ${k} has uncertainty ${o.sigma}; it must be a positive number of arcsec`);
    }
  });
  const w = obs.map((o) => 1 / (o.sigma * o.sigma));
  const wsum = w.reduce((a, b) => a + b, 0);
  const t0 = obs.reduce((a, o, i) => a + w[i] * o.mjd, 0) / wsum;

  const acc = [0, 0, 0];
  obs.forEach((o, i) => {
    const u = unitFromRaDec(o.ra, o.dec);
    for (let k = 0; k < 3; k++) acc[k] += w[i] * u[k];
  });
  const [raRef, decRef] = raDecFromVec(acc);

  const obsFlat = [];
  const design = [];
  for (const o of obs) {
    const [xw, yw] = tanProject(o.ra, o.dec, raRef, decRef);
    obsFlat.push(xw * 3600, yw * 3600);
    const dt = (o.mjd - t0) / YEAR_DAYS;
    const [fa, fd] = parallaxFactors(raRef, decRef, earthState(mjdToJdTt(o.mjd)));
    design.push([1, 0, dt, 0, fa]);
    design.push([0, 1, 0, dt, fd]);
  }

  const n = 5;
  const ata = Array.from({ length: n }, () => new Float64Array(n));
  design.forEach((row, k) => {
    const iw = 1 / (obs[k >> 1].sigma ** 2);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) ata[i][j] += iw * row[i] * row[j];
  });

  const normalSolve = (values) => {
    const atb = new Float64Array(n);
    design.forEach((row, k) => {
      const iw = 1 / (obs[k >> 1].sigma ** 2);
      for (let i = 0; i < n; i++) atb[i] += iw * row[i] * values[k];
    });
    return choleskySolveAndInvert(ata, atb);
  };

  let [params, cov] = normalSolve(obsFlat);

  const modelOffsets = (p) => {
    const [a0, d0, muA, muD, pi] = p;
    if (pi <= 0) throw new Error('fit drove the parallax non-positive');
    const dAu = ARCSEC_PER_RAD / pi;
    const out = [];
    for (const o of obs) {
      const dt = (o.mjd - t0) / YEAR_DAYS;
      const [raH, decH] = tanDeproject((a0 + muA * dt) / 3600, (d0 + muD * dt) / 3600, raRef, decRef);
      const ap = apparentPlace(raH, decH, dAu, o.mjd);
      const [xw, yw] = tanProject(ap.raDeg, ap.decDeg, raRef, decRef);
      out.push(xw * 3600, yw * 3600);
    }
    return out;
  };

  let iterations = 1;
  if (refine) {
    for (iterations = 1; iterations <= 12; iterations++) {
      const pred = modelOffsets(params);
      const resid = obsFlat.map((v, k) => v - pred[k]);
      const [delta, c2] = normalSolve(resid);
      cov = c2;
      params = params.map((v, i) => v + delta[i]);
      if (Math.max(Math.abs(delta[4]), Math.abs(delta[2]), Math.abs(delta[3])) < 1e-9) break;
    }
    iterations = Math.min(iterations, 12);
  }

  const [a0, d0, muA, muD, pi] = params;
  const final = refine ? modelOffsets(params)
    : design.map((row) => row.reduce((s, c, i) => s + c * params[i], 0));
  let chi2 = 0;
  for (let k = 0; k < obsFlat.length; k++) {
    chi2 += ((final[k] - obsFlat[k]) / obs[k >> 1].sigma) ** 2;
  }
  const dof = obsFlat.length - n;
  const reducedChi2 = dof > 0 ? chi2 / dof : NaN;
  const errScale = Number.isFinite(reducedChi2) && reducedChi2 > 1 ? Math.sqrt(reducedChi2) : 1;

  const pm = Math.hypot(muA, muD);

  // Multiple correlation between parallax and proper motion, not a projection
  // onto the fitted direction. Projecting onto mu/|mu| is undefined exactly
  // where it matters: a body with little fitted drift has |mu| at the noise
  // level, so the direction is random and so is the answer. R^2 = c' S^-1 c /
  // var(pi) is the largest correlation any linear combination of the two
  // components achieves -- well defined however mu came out, and the quantity
  // the question actually asks.
  let corr = 0;
  {
    const a = cov[2][2], b = cov[2][3], d2 = cov[3][3];
    const c0 = cov[4][2], c1 = cov[4][3];
    const det = a * d2 - b * b;
    const varPi = cov[4][4];
    if (det > 0 && varPi > 0) {
      const quad = (d2 * c0 * c0 - 2 * b * c0 * c1 + a * c1 * c1) / det;
      corr = Math.sqrt(Math.max(0, Math.min(1, quad / varPi)));
    }
  }

  const [raFit, decFit] = tanDeproject(a0 / 3600, d0 / 3600, raRef, decRef);
  const errPi = Math.sqrt(Math.max(cov[4][4], 0));
  const errPiQuoted = errPi * errScale;
  const distanceAu = distanceFromParallax(pi);
  const loPi = pi - errPiQuoted, hiPi = pi + errPiQuoted;

  return {
    t0Mjd: t0, ra0Deg: raFit, dec0Deg: decFit,
    // The tangent plane the motion is a straight line in, and the reference
    // position as an offset in it. `solutionPlace` needs all four.
    raRefDeg: raRef, decRefDeg: decRef, x0Arcsec: a0, y0Arcsec: d0,
    pmRaArcsecYr: muA, pmDecArcsecYr: muD, parallaxArcsec: pi,
    errRaArcsec: Math.sqrt(Math.max(cov[0][0], 0)),
    errDecArcsec: Math.sqrt(Math.max(cov[1][1], 0)),
    errPmRaArcsecYr: Math.sqrt(Math.max(cov[2][2], 0)),
    errPmDecArcsecYr: Math.sqrt(Math.max(cov[3][3], 0)),
    errParallaxArcsec: errPi,
    errParallaxQuoted: errPiQuoted,
    chi2, nObs: obs.length, dof, reducedChi2, errScale,
    corrParallaxPm: Math.max(0, Math.min(1, corr)),
    nIterations: iterations,
    distanceAu,
    distanceErrAu: ARCSEC_PER_RAD * errPiQuoted / (pi * pi),
    distanceIntervalAu: [hiPi > 0 ? ARCSEC_PER_RAD / hiPi : Infinity,
                         loPi <= 0 ? Infinity : ARCSEC_PER_RAD / loPi],
    significance: errPiQuoted ? pi / errPiQuoted : Infinity,
    separable: dof > 0 && Math.abs(corr) < 0.98,
    pmTotalArcsecYr: pm,
  };
}

/** Where a solution puts the body at `mjd`: the heliocentric direction, moved
 *  along a straight line in the fit's own tangent plane, and the apparent place
 *  seen from Earth.
 *
 *  The model `solveAstrometry` minimised, evaluated in one place, so the
 *  linker's predictions, the analysis page's residuals and the overlay's fitted
 *  track are all residuals against what was actually fitted. It used to drift
 *  ra0 linearly in RA instead, which is a different model: identical for a slow
 *  body and, for a fast one, tens of arcseconds from the fit it claimed to
 *  draw. Mirrors `AstrometricSolution.place` in the Python. */
export function solutionPlace(sol, mjd) {
  const yr = (mjd - sol.t0Mjd) / YEAR_DAYS;
  const [raH, decH] = tanDeproject((sol.x0Arcsec + sol.pmRaArcsecYr * yr) / 3600,
    (sol.y0Arcsec + sol.pmDecArcsecYr * yr) / 3600, sol.raRefDeg, sol.decRefDeg);
  const ap = apparentPlace(raH, decH, sol.distanceAu, mjd);
  return { raH, decH, ra: ap.raDeg, dec: ap.decDeg };
}

/** χ² with the distance held fixed and everything else refitted.
 *
 *  The profile of the fit along the one parameter anyone asks about. Its
 *  minimum is the fitted distance; where it has risen by one (times χ²/ν, when
 *  the errors were widened by that) is the 1σ interval, read off the data
 *  rather than off a linearised covariance. The two should agree when the fit
 *  is well behaved, and a profile that is flat and lopsided -- a short
 *  baseline, where parallax and drift trade off -- shows why a quoted ± alone
 *  undersells the problem.
 *
 *  Four parameters (position and drift) at each distance, Gauss-Newton against
 *  the exact apparent place, seeded from the linear solution: with parallax
 *  fixed the model is linear in those four to within r_E/d, so two or three
 *  iterations settle it. */
export function profileChi2(obs, dAu, { iterations = 4 } = {}) {
  const w = obs.map((o) => 1 / (o.sigma * o.sigma));
  const wsum = w.reduce((a, b) => a + b, 0);
  const t0 = obs.reduce((a, o, i) => a + w[i] * o.mjd, 0) / wsum;
  const acc = [0, 0, 0];
  obs.forEach((o, i) => {
    const u = unitFromRaDec(o.ra, o.dec);
    for (let k = 0; k < 3; k++) acc[k] += w[i] * u[k];
  });
  const [raRef, decRef] = raDecFromVec(acc);
  const pi = ARCSEC_PER_RAD / dAu;

  const obsFlat = [];
  const design = [];
  const fac = [];
  for (const o of obs) {
    const [xw, yw] = tanProject(o.ra, o.dec, raRef, decRef);
    obsFlat.push(xw * 3600, yw * 3600);
    const dt = (o.mjd - t0) / YEAR_DAYS;
    fac.push(parallaxFactors(raRef, decRef, earthState(mjdToJdTt(o.mjd))));
    design.push([1, 0, dt, 0], [0, 1, 0, dt]);
  }
  const n = 4;
  const ata = Array.from({ length: n }, () => new Float64Array(n));
  design.forEach((row, k) => {
    const iw = 1 / (obs[k >> 1].sigma ** 2);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) ata[i][j] += iw * row[i] * row[j];
  });
  const solve = (values) => {
    const atb = new Float64Array(n);
    design.forEach((row, k) => {
      const iw = 1 / (obs[k >> 1].sigma ** 2);
      for (let i = 0; i < n; i++) atb[i] += iw * row[i] * values[k];
    });
    return choleskySolveAndInvert(ata, atb)[0];
  };
  // Linear seed: the parallax part is known once d is, so subtract it.
  let p = Array.from(solve(obsFlat.map((v, k) => v - pi * fac[k >> 1][k & 1])));
  const model = (q) => {
    const out = [];
    for (const o of obs) {
      const dt = (o.mjd - t0) / YEAR_DAYS;
      const [raH, decH] = tanDeproject((q[0] + q[2] * dt) / 3600, (q[1] + q[3] * dt) / 3600,
        raRef, decRef);
      const ap = apparentPlace(raH, decH, dAu, o.mjd);
      const [xw, yw] = tanProject(ap.raDeg, ap.decDeg, raRef, decRef);
      out.push(xw * 3600, yw * 3600);
    }
    return out;
  };
  for (let it = 0; it < iterations; it++) {
    const pred = model(p);
    const delta = solve(obsFlat.map((v, k) => v - pred[k]));
    p = p.map((v, i) => v + delta[i]);
    if (Math.max(...Array.from(delta, Math.abs)) < 1e-9) break;
  }
  const pred = model(p);
  let chi2 = 0;
  for (let k = 0; k < obsFlat.length; k++) chi2 += ((obsFlat[k] - pred[k]) / obs[k >> 1].sigma) ** 2;
  return chi2;
}

/** What this field's cadence and geometry could do, before any data.
 *
 *  Plants a noiseless body at `dAu`, samples it at the epochs the field
 *  actually has, and solves. The distance comes back exact by construction --
 *  the numbers worth reading are `distanceErrAu` and `corrParallaxPm`, which
 *  depend only on *where* and *when* the field was observed.
 *
 *  A forecast, then, not a result. Worth showing because the two things that
 *  set it are independent and constantly conflated: on the baked fields the
 *  two ecliptic poles have identical geometry and differ by a factor of nine
 *  and a half, entirely because one was revisited over 440 days and the other
 *  over 93. */
export function forecastPrecision(raDeg, decDeg, mjds, dAu = 500, sigmaArcsec = 0.15) {
  const obs = mjds.map((m) => {
    const ap = apparentPlace(raDeg, decDeg, dAu, m);
    return { mjd: m, ra: ap.raDeg, dec: ap.decDeg, sigma: sigmaArcsec };
  });
  return solveAstrometry(obs);
}

/** Bound the distance from exactly two positions.
 *
 *  Two epochs give four numbers against five parameters, so something must be
 *  assumed. The weakest useful assumption: the body is on a bound, roughly
 *  circular heliocentric orbit, which fixes the *magnitude* of its drift but
 *  says nothing about its *direction*. By the triangle inequality the observable
 *  displacement then satisfies
 *
 *      |piDisp(d) - orbital(d)|  <=  displacement  <=  piDisp(d) + orbital(d)
 *
 *  and the consistent set of distances is every d satisfying that. Returning
 *  that interval is honest; returning its midpoint alone would not be.
 *
 *  A log grid is scanned rather than bisected because |piDisp - orbital| is not
 *  monotone in d -- it peaks near 5.5 AU for a six-month baseline -- and a
 *  bisection assuming monotonicity returns a confident wrong endpoint in the
 *  main-belt regime. */
export function twoEpochDistance(a, b, { dLo = 1, dHi = 1e5, grid = 2000 } = {}) {
  const sep = angularSeparation(a.ra, a.dec, b.ra, b.dec) * 3600;
  const dtDays = Math.abs(b.mjd - a.mjd);
  if (dtDays <= 0) throw new Error('the two epochs must differ in time');
  if (sep <= 0) throw new Error('no measurable displacement between the two epochs');
  const years = dtDays / YEAR_DAYS;

  const ua = unitFromRaDec(a.ra, a.dec), ub = unitFromRaDec(b.ra, b.dec);
  const [raM, decM] = raDecFromVec([ua[0] + ub[0], ua[1] + ub[1], ua[2] + ub[2]]);
  const [fa1, fd1] = parallaxFactors(raM, decM, earthState(mjdToJdTt(a.mjd)));
  const [fa2, fd2] = parallaxFactors(raM, decM, earthState(mjdToJdTt(b.mjd)));
  const leverage = Math.hypot(fa2 - fa1, fd2 - fd1);
  if (leverage <= 0) throw new Error('this epoch pair has no parallax leverage at all');

  const parts = (d) => [leverage * parallaxArcsec(d), orbitalRateArcsecPerYear(d) * years];
  const consistent = (d) => {
    const [p, o] = parts(d);
    return Math.abs(p - o) - 1e-12 <= sep && sep <= p + o + 1e-12;
  };

  const ratio = Math.pow(dHi / dLo, 1 / (grid - 1));
  let dMin = Infinity, dMax = -Infinity, any = false;
  for (let i = 0; i < grid; i++) {
    const d = dLo * Math.pow(ratio, i);
    if (consistent(d)) { any = true; if (d < dMin) dMin = d; if (d > dMax) dMax = d; }
  }
  if (!any) {
    const [p0, o0] = parts(dLo);
    throw new Error(`displacement ${sep.toFixed(2)}" is not reachable by a bound orbit `
      + `between ${dLo} and ${dHi} AU (max ${(p0 + o0).toFixed(2)}") -- probably not the same object`);
  }
  const refineEdge = (inside, outside) => {
    for (let i = 0; i < 60; i++) {
      const mid = Math.sqrt(inside * outside);
      if (consistent(mid)) inside = mid; else outside = mid;
    }
    return inside;
  };
  if (dMin > dLo * ratio) dMin = refineEdge(dMin, dMin / ratio);
  if (dMax < dHi / ratio) dMax = refineEdge(dMax, dMax * ratio);

  const solveMonotone = (f) => {
    let lo = dLo, hi = dHi;
    if (f(lo) < sep) return lo;
    if (f(hi) > sep) return hi;
    for (let i = 0; i < 200; i++) {
      const mid = Math.sqrt(lo * hi);
      if (f(mid) > sep) lo = mid; else hi = mid;
    }
    return Math.sqrt(lo * hi);
  };
  const dAligned = solveMonotone((d) => parts(d)[0] + parts(d)[1]);
  const dParOnly = solveMonotone((d) => parts(d)[0]);
  const [par, orb] = parts(dAligned);
  const fracErr = Math.hypot(a.sigma, b.sigma) / sep;
  return {
    dMinAu: dMin, dMaxAu: dMax, dAlignedAu: dAligned, dParallaxOnlyAu: dParOnly,
    displacementArcsec: sep, parallaxPartArcsec: par, orbitalPartArcsec: orb,
    parallaxLeverageAu: leverage, baselineDays: dtDays,
    fracErrMeasurement: fracErr,
    distanceAu: Math.sqrt(dMin * dMax),
    intervalAu: [dMin * (1 - Math.min(0.9, fracErr)), dMax / Math.max(1e-9, 1 - Math.min(0.9, fracErr))],
    isDegenerate: dMax / dMin > 2,
  };
}
