// Candidate evidence, and the ML linking gate that reads it.
//
// The linker keeps a candidate when it clears fixed cuts -- enough visits, a
// good enough fit, a parallax far enough from zero -- and each cut is judged
// alone. A real faint body often fails one of them narrowly while passing the
// rest comfortably, and a chance alignment of noise passes all of them
// narrowly. A score over all the evidence at once can tell those apart where
// the cuts cannot, which is what lets the cuts be loosened: more faint bodies
// through, the extra false links sorted out afterwards.
//
// The evidence, per candidate: the fraction of visits it was linked in, and
// that count against what the default search requires; the fit's chi-squared
// and parallax significance; how much of the parallax the drift could imitate; the residual scatter against the centroid
// error; whether its drift is possible at its distance -- a body on a bound
// orbit cannot move faster than escape speed, and false links routinely imply
// more; the distance; the real/bogus network's view of its detections, median
// and worst; and how steady its detections' brightness is.
//
// A logistic regression over these twelve was trained first and did not
// transfer between fields -- held out, it let 279 false candidates through on
// the North Ecliptic Pole. What ships is the gate below: two of the twelve,
// each with a reason, their thresholds chosen by `scripts/eval_mllink.py`. The
// features stay, because that script's candidates are described by them.

import { orbitalRateArcsecPerYear } from './ephem.js';

export const FEATURES = [
  'linked_frac', 'n_over_required', 'log_chi2nu', 'log_significance', 'corr_pi_mu', 'rms_over_sigma',
  'drift_over_circular', 'log_distance', 'rb_median', 'rb_min', 'log_snr_mean', 'snr_cv',
];

/** The feature vector for a tracklet with a solution. `rb` on the members is
 *  the real/bogus score; missing scores count as 0.5, no opinion. */
export function candidateFeatures(t, { nEpochs, required, sigmaArcsec }) {
  const s = t.solution;
  const n = new Set(t.members.map((m) => m.epoch)).size;
  const rate = orbitalRateArcsecPerYear(Math.max(1, s.distanceAu));
  const rbs = t.members.map((m) => (Number.isFinite(m.rb) ? m.rb : 0.5)).sort((a, b) => a - b);
  const snrs = t.members.map((m) => m.snr).filter((v) => v > 0);
  const mean = snrs.reduce((a, v) => a + v, 0) / Math.max(1, snrs.length);
  const sd = Math.sqrt(snrs.reduce((a, v) => a + (v - mean) ** 2, 0) / Math.max(1, snrs.length));
  return [
    n / Math.max(1, nEpochs),
    // Against what the default search requires, not as a count: six visits is
    // everything on one field and half on another, and a count taught the
    // first model the twelve-visit fields' idea of plenty.
    n / Math.max(1, required),
    Math.log10(Math.max(1e-3, s.reducedChi2)),
    Math.log10(Math.max(1e-3, s.significance)),
    s.corrParallaxPm,
    (t.rmsArcsec || 0) / sigmaArcsec,
    Math.min(10, s.pmTotalArcsecYr / rate),
    Math.log10(Math.max(1, s.distanceAu)),
    rbs.length ? rbs[rbs.length >> 1] : 0.5,
    rbs.length ? rbs[0] : 0.5,
    Math.log10(Math.max(1, mean)),
    mean > 0 ? sd / mean : 0,
  ];
}

/** How far a candidate's drift exceeds what a bound orbit at its distance
 *  allows, in units of the drift's own measured error: (|mu| - 1.5 x circular
 *  rate) / sigma_mu, negative when it is within the limit. Escape speed is
 *  sqrt(2) = 1.41 times circular; 1.5 leaves room for the eccentric. Measured
 *  in sigma rather than outright because a short baseline measures drift
 *  poorly: on the 93-day South Ecliptic Pole a flat cut threw out a third of
 *  the real bodies. */
export function driftExcessSigma(s, { factor = 1.5 } = {}) {
  const rate = orbitalRateArcsecPerYear(Math.max(1, s.distanceAu));
  const mu = s.pmTotalArcsecYr;
  const err = mu > 0 ? s.errScale * Math.hypot(s.pmRaArcsecYr * s.errPmRaArcsecYr,
    s.pmDecArcsecYr * s.errPmDecArcsecYr) / mu : Math.hypot(s.errPmRaArcsecYr, s.errPmDecArcsecYr);
  return err > 0 ? (mu - factor * rate) / err : (mu > factor * rate ? Infinity : -Infinity);
}

/** e^-z I0(z), the exponentially scaled modified Bessel function of order
 *  zero (Abramowitz and Stegun 9.8.1-2, to 2e-7). */
function i0e(z) {
  const x = Math.abs(z);
  if (x < 3.75) {
    const t = (x / 3.75) ** 2;
    return Math.exp(-x) * (1 + t * (3.5156229 + t * (3.0899424 + t * (1.2067492
      + t * (0.2659732 + t * (0.0360768 + t * 0.0045813))))));
  }
  const t = 3.75 / x;
  return (0.39894228 + t * (0.01328592 + t * (0.00225319 + t * (-0.00157565 + t * (0.00916281
    + t * (-0.02057706 + t * (0.02635537 + t * (-0.01647633 + t * 0.00392377)))))))) / Math.sqrt(x);
}

/** Marcum's Q1(a, b): the chance that the length of a 2-D vector with unit
 *  Gaussian errors about a true vector of length a exceeds b -- the Rice
 *  distribution's tail. Integrated directly; the integrand is
 *  x exp(-(x - a)^2 / 2) e^-ax I0(ax), which stays finite however large a is. */
export function marcumQ1(a, b) {
  if (!(b > 0)) return 1;
  const hi = Math.max(b, a) + 12;
  const n = 400, h = (hi - b) / n;
  const f = (x) => x * Math.exp(-0.5 * (x - a) ** 2) * i0e(a * x);
  let sum = f(b) + f(hi);
  for (let k = 1; k < n; k++) sum += (k % 2 ? 4 : 2) * f(b + k * h);
  return Math.min(1, Math.max(0, sum * h / 3));
}

/** The chance that a body on a bound orbit -- drifting at most `factor`
 *  times the circular rate at the fitted distance -- would show a drift at
 *  least as large as this candidate's, given how well the fit measured it.
 *
 *  driftExcessSigma asks the same question as a number of sigma, which is
 *  right when the drift is measured well and wrong when it is not. The drift
 *  is a 2-D vector and its length is biased: with the true drift near zero
 *  and an error sigma, the measured length exceeds sigma 61% of the time and
 *  2 sigma 14% of the time (the Rayleigh distribution). On the 93-day South
 *  Ecliptic Pole, where drift and parallax are hard to separate, that is
 *  most bodies: at 1700 AU a fixed one-sigma cut rejected real ones at
 *  1.5 sigma. The tail of the Rice distribution is the calibrated version:
 *  the worst case over every bound drift is one on the limit, and the chance
 *  there is Marcum's Q1(limit / sigma, measured / sigma). Errors are taken as
 *  round, at the mean of the two axes', widened by the fit's error scale. */
export function driftPValue(s, { factor = 1.5 } = {}) {
  const limit = factor * orbitalRateArcsecPerYear(Math.max(1, s.distanceAu));
  const sig = (s.errScale || 1) * Math.sqrt((s.errPmRaArcsecYr ** 2 + s.errPmDecArcsecYr ** 2) / 2);
  if (!(sig > 0)) return s.pmTotalArcsecYr > limit ? 0 : 1;
  return marcumQ1(limit / sig, s.pmTotalArcsecYr / sig);
}

/** The median real/bogus score of a candidate's detections; 0.5 where none
 *  was scored. */
export function rbMedian(t) {
  const rbs = t.members.map((m) => m.rb).filter(Number.isFinite).sort((a, b) => a - b);
  return rbs.length ? rbs[rbs.length >> 1] : 0.5;
}

/** Whether a candidate from the loosened search passes the ML linking gate:
 *  its detections look real to the network, at the median, and its drift is
 *  one a bound orbit at its distance could have. The two thresholds are
 *  `scripts/eval_mllink.py`'s, from data/model/mllink.json. */
export function passesGate(t, gate) {
  if (!t.solution || !(rbMedian(t) >= gate.rb_min)) return false;
  const factor = gate.drift_factor || 1.5;
  // A gate written with a drift p-value uses the calibrated test; an older
  // one, the sigma cut it was chosen with.
  return gate.drift_p != null
    ? driftPValue(t.solution, { factor }) >= gate.drift_p
    : driftExcessSigma(t.solution, { factor }) <= gate.drift_sigma;
}

let cached = null;
export async function loadGate(url = 'data/model/mllink.json') {
  if (cached !== null) return cached;
  try {
    const r = await fetch(url);
    cached = r.ok ? await r.json() : false;
  } catch { cached = false; }
  return cached || null;
}
