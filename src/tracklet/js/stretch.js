// Display stretch. A port of src/tracklet/frames/stretch.py; the Python is the
// authority and tests/js/run-golden.mjs checks this against vectors from it.
//
// Display stretch is where astronomical images get quietly falsified. A
// percentile clip picked to look good erases a faint moving source, or invents
// one out of read noise. So: algorithms with citable definitions, computed from
// the data, and invertible -- invertLevel() takes a screen grey back to MJy/sr,
// which is how the readout can always answer what a pixel is actually worth.

export const ZSCALE_NSAMPLES = 1000;
export const ZSCALE_CONTRAST = 0.25;
export const ZSCALE_KREJ = 2.5;
export const ZSCALE_MAX_ITER = 5;
export const ZSCALE_MIN_NPIXELS = 5;
export const ASINH_Q_DEFAULT = 8;

/** Grid-subsample a Float32Array image, dropping NaN, and sort.
 *  A grid stride rather than a random draw so the result is deterministic: two
 *  viewers blinking the same pair must get the same stretch, or what they are
 *  comparing is the stretch and not the sky. */
function sample(data, w, h, nsamples) {
  const total = w * h;
  const stride = Math.max(1, Math.floor(Math.sqrt(total / Math.max(1, nsamples))));
  const out = [];
  for (let y = 0; y < h; y += stride) {
    for (let x = 0; x < w; x += stride) {
      const v = data[y * w + x];
      if (Number.isFinite(v)) out.push(v);
    }
  }
  out.sort((a, b) => a - b);
  return out;
}

/** Median of an already-sorted array; 0 for an empty one. */
export function medianSorted(a) {
  const n = a.length;
  if (!n) return 0;
  const m = n >> 1;
  return n % 2 ? a[m] : 0.5 * (a[m - 1] + a[m]);
}

/** First index of sorted `a` whose value is >= x. */
export function lowerBound(a, x) {
  let lo = 0, hi = a.length;
  while (lo < hi) { const m = (lo + hi) >> 1; if (a[m] < x) lo = m + 1; else hi = m; }
  return lo;
}

/** First index of sorted `a` whose value is > x. */
export function upperBound(a, x) {
  let lo = 0, hi = a.length;
  while (lo < hi) { const m = (lo + hi) >> 1; if (a[m] <= x) lo = m + 1; else hi = m; }
  return lo;
}

const LITTLE_ENDIAN = new Uint8Array(new Uint32Array([1]).buffer)[0] === 1;
const radixCount = new Uint32Array(2048);
let radixKeys = [new Uint32Array(0), new Uint32Array(0), new Uint32Array(0), new Uint32Array(0)];

/** Sort a Float64Array in place, numerically, as its own `.sort()` does -- to
 *  the same order bit for bit, -0 before +0 included -- in less than half the
 *  time on the arrays this page sorts most: a stamp's 37,000 pixels, several
 *  times a detection, where the native sort was 60% of the detector's time.
 *
 *  A least-significant-digit radix sort on each value's bits, turned into a
 *  key that orders as the number does (negative: every bit flipped; positive:
 *  the sign bit set), in six passes of 11 bits, skipping any pass on which
 *  every key agrees. No NaN may be in `a`; every caller has filtered it out.
 *  Short arrays, and the rare big-endian or misaligned one, take the native
 *  sort, which is quicker there. */
export function sortFloat64(a) {
  const n = a.length;
  if (n < 4096 || !LITTLE_ENDIAN || (a.byteOffset & 7)) return a.sort();
  if (radixKeys[0].length < n) radixKeys = radixKeys.map(() => new Uint32Array(n));
  let [lo, hi, lo2, hi2] = radixKeys;
  const w = new Uint32Array(a.buffer, a.byteOffset, 2 * n);   // [low word, high word] per value
  for (let i = 0; i < n; i++) {
    const l = w[2 * i], h = w[2 * i + 1];
    if (h & 0x80000000) { lo[i] = ~l >>> 0; hi[i] = ~h >>> 0; } else { lo[i] = l; hi[i] = (h | 0x80000000) >>> 0; }
  }
  for (const [word, shift, mask] of RADIX_DIGITS) {
    const src = word ? hi : lo;
    radixCount.fill(0);
    for (let i = 0; i < n; i++) radixCount[(src[i] >>> shift) & mask]++;
    if (radixCount[(src[0] >>> shift) & mask] === n) continue;
    for (let b = 0, sum = 0; b <= mask; b++) { const c = radixCount[b]; radixCount[b] = sum; sum += c; }
    for (let i = 0; i < n; i++) {
      const p = radixCount[(src[i] >>> shift) & mask]++;
      lo2[p] = lo[i]; hi2[p] = hi[i];
    }
    [lo, lo2, hi, hi2] = [lo2, lo, hi2, hi];
  }
  for (let i = 0; i < n; i++) {
    const l = lo[i], h = hi[i];
    if (h & 0x80000000) { w[2 * i] = l; w[2 * i + 1] = h & 0x7fffffff; } else { w[2 * i] = ~l >>> 0; w[2 * i + 1] = ~h >>> 0; }
  }
  return a;
}
// [word (0 low, 1 high), shift, mask]: 11 + 11 + 10 bits of each word.
const RADIX_DIGITS = [[0, 0, 0x7ff], [0, 11, 0x7ff], [0, 22, 0x3ff], [1, 0, 0x7ff], [1, 11, 0x7ff], [1, 22, 0x3ff]];

/** Scale from median absolute deviation to a Gaussian sigma, 1/Phi^-1(0.75). */
export const MAD_TO_SIGMA = 1.4826;

/** The r-th smallest (from 0) of |v - med| over the sorted array `sorted`.
 *
 *  Without computing and sorting the deviations: measured from any value, a
 *  sorted array's deviations are two runs already in order -- falling towards
 *  `med` on its left, rising away on its right -- so the r-th of them is found
 *  by bisecting how many to take from each. The same element, so the same
 *  float, as sorting them; logarithmic rather than n log n. Mirrors
 *  `kth_deviation` in stretch.py. */
export function kthDeviation(sorted, med, r) {
  const k = lowerBound(sorted, med), m = sorted.length - k;
  const left = (t) => Math.abs(sorted[k - 1 - t] - med);   // t-th smallest below med
  const right = (t) => Math.abs(sorted[k + t] - med);      // and at or above it
  // The r + 1 smallest are the a smallest on the left and the rest on the right.
  const c = r + 1;
  let lo = Math.max(0, c - m), hi = Math.min(c, k);
  while (lo < hi) {
    const a = (lo + hi) >> 1;
    if (left(a) < right(c - a - 1)) lo = a + 1; else hi = a;
  }
  return Math.max(lo ? left(lo - 1) : -Infinity, c - lo ? right(c - lo - 1) : -Infinity);
}

/** MAD_TO_SIGMA times the median of |v - med| over the sorted array `sorted`;
 *  0 for an empty one. */
export function madSigma(sorted, med) {
  const n = sorted.length;
  if (!n) return 0;
  const h = n >> 1;
  return MAD_TO_SIGMA * (n % 2 ? kthDeviation(sorted, med, h)
    : 0.5 * (kthDeviation(sorted, med, h - 1) + kthDeviation(sorted, med, h)));
}

/** IRAF/DS9 zscale (Tody 1986).
 *  Fits a line to the *sorted* sample with iterative sigma rejection, then
 *  extrapolates that slope to the ends and divides by contrast. Because the fit
 *  is to the sorted distribution it locks onto the long flat run in the middle
 *  -- the sky -- and ignores both cosmic rays at the top and masked pixels at
 *  the bottom. On a background-limited SPHEREx frame (zodi/sigma ~ 12) that is
 *  exactly the wanted behaviour. */
export function zscale(data, w, h, opts = {}) {
  const nsamples = opts.nsamples ?? ZSCALE_NSAMPLES;
  const contrast = opts.contrast ?? ZSCALE_CONTRAST;
  const krej = opts.krej ?? ZSCALE_KREJ;
  const maxIter = opts.maxIterations ?? ZSCALE_MAX_ITER;

  const s = sample(data, w, h, nsamples);
  const n = s.length;
  if (!n) return { z1: 0, z2: 1, median: 0, slope: 0, npoints: 0, nrejected: 0, iterations: 0 };
  const vmin = s[0], vmax = s[n - 1];
  const median = medianSorted(s);
  if (n < ZSCALE_MIN_NPIXELS) {
    return { z1: vmin, z2: vmax, median, slope: 0, npoints: n, nrejected: 0, iterations: 0 };
  }

  const xs = new Float64Array(n);
  for (let i = 0; i < n; i++) xs[i] = i - 0.5 * (n - 1);
  let keep = new Uint8Array(n).fill(1);
  let slope = 0, intercept = median, iterations = 0, ngood = n;

  // Counts like Python's `for i in range(1, maxIter+1)`: a run that uses every
  // iteration reports maxIter, not maxIter+1. The JS for-loop's trailing
  // increment would report one too many, and the golden-vector test compares
  // this field across the two languages.
  for (let it = 1; it <= maxIter; it++) {
    iterations = it;
    let sx = 0, sy = 0, sxx = 0, sxy = 0;
    ngood = 0;
    for (let i = 0; i < n; i++) {
      if (!keep[i]) continue;
      sx += xs[i]; sy += s[i]; sxx += xs[i] * xs[i]; sxy += xs[i] * s[i];
      ngood++;
    }
    if (ngood < 3) break;
    const denom = ngood * sxx - sx * sx;
    if (denom === 0) break;
    slope = (ngood * sxy - sx * sy) / denom;
    intercept = (sy - slope * sx) / ngood;

    let ss = 0;
    const resid = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      resid[i] = s[i] - (intercept + slope * xs[i]);
      if (keep[i]) ss += resid[i] * resid[i];
    }
    const sigma = Math.sqrt(ss / ngood);
    if (sigma === 0) break;
    const limit = krej * sigma;
    const next = new Uint8Array(n);
    let changed = false;
    for (let i = 0; i < n; i++) {
      next[i] = keep[i] && Math.abs(resid[i]) <= limit ? 1 : 0;
      if (next[i] !== keep[i]) changed = true;
    }
    if (!changed) break;
    keep = next;
  }

  const nrejected = n - ngood;
  if (slope === 0) {
    return { z1: vmin, z2: vmax, median, slope: 0, npoints: n, nrejected, iterations };
  }
  const c = contrast > 0 ? contrast : 1;
  const midpoint = (n + 1) >> 1;
  let z1 = Math.max(vmin, intercept - (midpoint - 1) * slope / c);
  let z2 = Math.min(vmax, intercept + (n - midpoint) * slope / c);
  if (z2 <= z1) { z1 = vmin; z2 = vmax; }
  return { z1, z2, median, slope, npoints: n, nrejected, iterations };
}

export function percentileLimits(data, w, h, lo = 0.5, hi = 99.5) {
  const vals = [];
  for (let i = 0; i < w * h; i++) if (Number.isFinite(data[i])) vals.push(data[i]);
  if (!vals.length) return [0, 1];
  vals.sort((a, b) => a - b);
  const n = vals.length;
  const at = (p) => {
    if (n === 1) return vals[0];
    const k = Math.max(0, Math.min(1, p / 100)) * (n - 1);
    const i = Math.floor(k), f = k - i;
    return i >= n - 1 ? vals[n - 1] : vals[i] * (1 - f) + vals[i + 1] * f;
  };
  const a = at(lo), b = at(hi);
  return b > a ? [a, b] : [vals[0], vals[n - 1] > vals[0] ? vals[n - 1] : vals[0] + 1];
}

const clamp01 = (t) => (t < 0 ? 0 : t > 1 ? 1 : t);

export const TRANSFERS = ['linear', 'asinh', 'sqrt', 'squared', 'log', 'sinh', 'power'];

/** Named [0,1] -> [0,1] transfer. asinh is normalised so f(0)=0 and f(1)=1
 *  exactly -- an unnormalised asinh silently moves the white point, and then
 *  two stretches of the same data are no longer comparable. */
export function transfer(name, q = ASINH_Q_DEFAULT, gamma = 2.2) {
  switch (name) {
    case 'linear': return clamp01;
    case 'asinh': {
      const qq = Math.max(1e-6, q), d = Math.asinh(qq);
      return (t) => clamp01(Math.asinh(clamp01(t) * qq) / d);
    }
    case 'sinh': {
      const qq = Math.max(1e-6, q), d = Math.sinh(qq);
      return (t) => clamp01(Math.sinh(clamp01(t) * qq) / d);
    }
    case 'sqrt': return (t) => Math.sqrt(clamp01(t));
    case 'squared': return (t) => clamp01(t) ** 2;
    case 'log': {
      const a = 1000;
      return (t) => clamp01(Math.log10(a * clamp01(t) + 1) / Math.log10(a + 1));
    }
    case 'power': {
      const g = Math.max(1e-6, gamma);
      return (t) => clamp01(t) ** (1 / g);
    }
    default: throw new Error(`unknown transfer function ${name}`);
  }
}

/** Display level in [0,1] back to data units. Analytic where possible. */
export function invertLevel(level, z1, z2, mode = 'asinh', q = ASINH_Q_DEFAULT) {
  const span = (z2 - z1) || 1;
  const t = clamp01(level);
  if (mode === 'linear') return z1 + t * span;
  if (mode === 'asinh') {
    const qq = Math.max(1e-6, q);
    return z1 + span * Math.sinh(t * Math.asinh(qq)) / qq;
  }
  if (mode === 'sqrt') return z1 + span * t * t;
  if (mode === 'squared') return z1 + span * Math.sqrt(t);
  const f = transfer(mode, q);
  let lo = 0, hi = 1;
  for (let i = 0; i < 40; i++) {
    const mid = 0.5 * (lo + hi);
    if (f(mid) < t) lo = mid; else hi = mid;
  }
  return z1 + span * 0.5 * (lo + hi);
}

export function histogram(data, w, h, bins = 128, limits = null) {
  // Range and counts in loops, not `Math.min(...values)`: spreading an array
  // into arguments throws past ~120k of them, so a stamp much over 350 px on
  // a side would have stopped the page painting.
  const total = w * h;
  let n = 0, min = Infinity, max = -Infinity;
  for (let i = 0; i < total; i++) {
    const v = data[i];
    if (!Number.isFinite(v)) continue;
    n++;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  if (!n) return { edges: [], counts: [], nFinite: 0, nMasked: total, peak: 0 };
  let [lo, hi] = limits || [min, max];
  if (hi <= lo) hi = lo + 1;
  const width = (hi - lo) / bins;
  const counts = new Int32Array(bins);
  for (let i = 0; i < total; i++) {
    const v = data[i];
    if (!Number.isFinite(v)) continue;
    let k = Math.floor((v - lo) / width);
    if (k < 0) k = 0; else if (k >= bins) k = bins - 1;
    counts[k]++;
  }
  const edges = new Float64Array(bins + 1);
  for (let i = 0; i <= bins; i++) edges[i] = lo + i * width;
  let peak = 0;
  for (const c of counts) if (c > peak) peak = c;
  return { edges, counts, nFinite: n, nMasked: total - n, peak };
}

/** 1.4826 * MAD. Used rather than the sample standard deviation because a
 *  cutout containing an actual source -- the interesting case -- has a standard
 *  deviation set by the source, not by the noise we are trying to measure. */
export function robustSigma(values) {
  const v = finiteSorted(values);
  return madSigma(v, medianSorted(v));
}

/** The finite values, sorted as a Float64Array: numerically, in native code,
 *  with no comparator called per comparison. Where the order of equal values
 *  can show -- a median of zero, whose sign is the only thing that would --
 *  the caller must not rely on it; a MAD cannot see it. */
function finiteSorted(values) {
  const out = new Float64Array(values.length);
  let n = 0;
  for (let i = 0; i < values.length; i++) if (Number.isFinite(values[i])) out[n++] = values[i];
  return sortFloat64(out.subarray(0, n));
}
