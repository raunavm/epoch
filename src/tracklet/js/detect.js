// Matched-filter source detection, in the browser.
//
// The mirror of `src/tracklet/search/detect.py`, and the Python side is the
// authority: `scripts/emit_golden.py` emits vectors from it and
// `tests/js/run-golden.mjs` replays them through here. Read that module's
// docstring for why a matched filter and not a network, why the background is a
// clipped median, and why the filter's noise is not the image's noise.
//
// Operates on the flat Float32Array the FITS reader produces, rather than on
// nested arrays: a 193x193 stamp convolved with a 5x5 kernel is a million
// multiply-adds, and it runs while someone is waiting.

import { lowerBound, madSigma, medianSorted, sortFloat64, upperBound } from './stretch.js';

/** Gaussian sigma of the detection kernel, px. n_eff = 4*pi*sigma^2 = 4 gives
 *  0.56; rounded up, because a filter narrower than the source costs far more
 *  signal-to-noise than one slightly wider. */
export const PSF_SIGMA_PX = 0.6;

/** Sigma-clipped median and MAD sigma over finite values. */
export function estimateBackground(values, { clip = 3, iters = 5 } = {}) {
  // Sorted as a Float64Array, which sorts numerically in native code rather
  // than calling a comparator per comparison; then every clip keeps one slice.
  const all = new Float64Array(values.length);
  let n = 0, negZero = false;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (Number.isFinite(v)) {
      all[n++] = v;
      if (v === 0 && 1 / v < 0) negZero = true;
    }
  }
  if (!n) return { background: 0, sigma: 0, nUsed: 0, nClipped: 0 };
  let keep = sortFloat64(all.subarray(0, n));
  if (negZero) {
    // A native sort puts -0 before +0. The comparator sort this replaced left
    // equal values in the order they came, as Python's does, so the zeros are
    // put back in that order and a median of zero keeps the same sign.
    let z = lowerBound(keep, 0);
    for (let i = 0; i < values.length; i++) if (values[i] === 0) keep[z++] = values[i];
  }
  let med = 0, sigma = 0;
  for (let it = 0; it < iters; it++) {
    if (!keep.length) break;
    med = medianSorted(keep);
    sigma = madSigma(keep, med);
    if (!(sigma > 0)) break;
    const a = lowerBound(keep, med - clip * sigma), b = upperBound(keep, med + clip * sigma);
    if (b - a === keep.length) break;
    keep = keep.subarray(a, b);
  }
  return { background: med, sigma, nUsed: keep.length, nClipped: n - keep.length };
}

/** Side of the background mesh, px.
 *
 *  Must be large compared to a source and small compared to the gradient. A
 *  SPHEREx stamp is dominated by zodiacal light, which ramps across the frame
 *  by several times the pixel noise, so one number for "the background" is
 *  wrong in both directions at once: it invents sources in the bright half and
 *  deletes them from the faint half. */
export const BACKGROUND_BOX_PX = 32;

/** A background *surface*, not a level, plus the noise about it.
 *
 *  Clipped median per mesh cell, bilinearly interpolated back to full
 *  resolution -- and **extrapolated**, not clamped, in the half-cell border.
 *  Holding the surface flat out to the edge leaves a pedestal of half a cell's
 *  gradient under everything there, which is enough to drive a real source's
 *  aperture flux negative and have the sharpness cut delete it. The border is
 *  where a moving object spends most of its time. */
export function backgroundMap(data, width, height, box = BACKGROUND_BOX_PX) {
  if (!width || !height) return { bg: new Float64Array(0), sigma: 0, nUsed: 0, nClipped: 0, level: 0 };
  const gx = Math.max(1, Math.ceil(width / box));
  const gy = Math.max(1, Math.ceil(height / box));
  const { background: level } = estimateBackground(data);

  const mesh = new Float64Array(gx * gy).fill(level);
  const cell = [];
  for (let cy = 0; cy < gy; cy++) {
    for (let cx = 0; cx < gx; cx++) {
      cell.length = 0;
      for (let j = cy * box; j < Math.min(height, (cy + 1) * box); j++) {
        for (let i = cx * box; i < Math.min(width, (cx + 1) * box); i++) {
          const v = data[j * width + i];
          if (Number.isFinite(v)) cell.push(v);
        }
      }
      // A cell that is entirely masked inherits the global level rather than a
      // zero, which would carve a hole in the surface.
      if (cell.length) mesh[cy * gx + cx] = estimateBackground(cell).background;
    }
  }

  const bg = new Float64Array(width * height);
  for (let j = 0; j < height; j++) {
    const fy = (j - 0.5 * box + 0.5) / box;
    const jy = gy < 2 ? 0 : Math.min(gy - 2, Math.max(0, Math.floor(fy)));
    const jy1 = Math.min(gy - 1, jy + 1);
    const ty = gy < 2 ? 0 : fy - jy;
    for (let i = 0; i < width; i++) {
      const fx = (i - 0.5 * box + 0.5) / box;
      const ix = gx < 2 ? 0 : Math.min(gx - 2, Math.max(0, Math.floor(fx)));
      const ix1 = Math.min(gx - 1, ix + 1);
      const tx = gx < 2 ? 0 : fx - ix;
      const a = mesh[jy * gx + ix] * (1 - tx) + mesh[jy * gx + ix1] * tx;
      const b = mesh[jy1 * gx + ix] * (1 - tx) + mesh[jy1 * gx + ix1] * tx;
      bg[j * width + i] = a * (1 - ty) + b * ty;
    }
  }

  const resid = [];
  for (let p = 0; p < bg.length; p++) {
    if (Number.isFinite(data[p])) resid.push(data[p] - bg[p]);
  }
  const r = estimateBackground(resid);
  return { bg, sigma: r.sigma, nUsed: r.nUsed, nClipped: r.nClipped, level };
}

/** Normalised 2-D Gaussian and its radius. */
export function gaussianKernel(sigma = PSF_SIGMA_PX) {
  // floor(x + 0.5), not Math.round, so the radius does not depend on which
  // language is asking. Python's round() is round-half-to-even while this was
  // Math.round: at sigma = 1.0 the argument is exactly 2.5, so JavaScript gave
  // r = 3 where Python gave 2 -- different kernels, and a matched-filter gain
  // of 1.772637 against 1.772390. See `gaussian_kernel` in the Python mirror.
  const r = Math.max(1, Math.floor(2.5 * sigma + 0.5));
  const n = 2 * r + 1;
  const k = new Float64Array(n * n);
  let s = 0;
  for (let dy = -r; dy <= r; dy++) {
    for (let dx = -r; dx <= r; dx++) {
      const v = Math.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma));
      k[(dy + r) * n + (dx + r)] = v;
      s += v;
    }
  }
  for (let i = 0; i < k.length; i++) k[i] /= s;
  return { kernel: k, r, n };
}

/** `f`, kept per PSF width. The three constants below depend on the width
 *  alone, and the Depth page converts magnitudes through them thousands of
 *  times a draw -- each call rebuilding the kernel -- where the width is only
 *  ever one or two numbers. */
function perWidth(f) {
  const seen = new Map();
  return (psfSigma = PSF_SIGMA_PX) => {
    if (!seen.has(psfSigma)) seen.set(psfSigma, f(psfSigma));
    return seen.get(psfSigma);
  };
}

/** How much bilinear resampling widens the matched filter's noise.
 *
 *  Output-pixel correlations for uniform-phase bilinear weights are 1/4 for
 *  axis neighbours and 1/16 diagonally, so the widening is sqrt(k'Ck / k'k).
 *  Evaluates to 1.22 for the shipped 0.6-px kernel; white noise through the
 *  real resampler measures 1.28. NOT 1/sqrt(shrink) (1.50 here) -- the
 *  smoothing that correlates the noise also removes the high-frequency part
 *  the filter would otherwise integrate. Mirrors detect.py.
 */
export const resampledFilterNorm = perWidth((psfSigma) => {
  const { kernel, n } = gaussianKernel(psfSigma);
  const at = (j, i) => (j >= 0 && j < n && i >= 0 && i < n ? kernel[j * n + i] : 0);
  const rho = [[0, 0, 1], [1, 0, .25], [-1, 0, .25], [0, 1, .25], [0, -1, .25],
               [1, 1, 1 / 16], [1, -1, 1 / 16], [-1, 1, 1 / 16], [-1, -1, 1 / 16]];
  let q = 0, ktck = 0;
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const k = kernel[j * n + i];
      q += k * k;
      for (const [dj, di, p] of rho) ktck += k * at(j + dj, i + di) * p;
    }
  }
  return q > 0 ? Math.sqrt(ktck / q) : 1;
});

/** Peak matched-filter significance produced by a unit-amplitude point source.
 *
 *  Invert it and you get the amplitude to plant for a stated signal-to-noise:
 *  `A = target * sigma / matchedFilterGain(psfSigma)`. Computed from the same
 *  kernel the detector uses rather than from the closed form, so an injection
 *  cannot claim a signal-to-noise the detector would not agree with -- which is
 *  the one way an injection-recovery test can quietly lie to you. */
export const matchedFilterGain = perWidth((psfSigma) => {
  const { kernel, r, n } = gaussianKernel(psfSigma);
  let acc = 0, q = 0;
  for (let dy = -r; dy <= r; dy++) {
    for (let dx = -r; dx <= r; dx++) {
      const w = kernel[(dy + r) * n + (dx + r)];
      acc += w * Math.exp(-(dx * dx + dy * dy) / (2 * psfSigma * psfSigma));
      q += w * w;
    }
  }
  return acc / Math.sqrt(q);
});

/** Total flux of a unit-peak Gaussian PSF, in pixels: the factor that turns a
 *  **peak** pixel value into a flux. 2*pi*sigma^2 in the continuum (2.2619 px
 *  at 0.6); this is its discrete sum over the same radius `plantSource` lays
 *  light down over, so a planted flux and a measured one agree exactly.
 *
 *  Not n_eff. n_eff = (sum g)^2/sum g^2 = 4.07 px is the *noise* equivalent
 *  area -- what a per-pixel sigma is multiplied by, in feasibility_gate.py --
 *  and using it on a peak value instead reports a point source 0.61 mag too
 *  bright. Mirrors `psf_flux_integral` in detect.py. */
export const psfFluxIntegral = perWidth((psfSigma) => {
  const r = Math.ceil(4 * psfSigma);
  const twoS2 = 2 * psfSigma * psfSigma;
  let sum = 0;
  for (let dj = -r; dj <= r; dj++) for (let di = -r; di <= r; di++) sum += Math.exp(-(di * di + dj * dj) / twoS2);
  return sum;
});

/** Add a Gaussian point source of the given peak amplitude, in place.
 *
 *  Sub-pixel centre, because a source that always lands on a pixel centre is a
 *  softer test than reality: the centroid error a real detection carries comes
 *  mostly from where in the pixel the light fell. */
export function plantSource(data, width, height, x, y, amplitude, psfSigma = PSF_SIGMA_PX) {
  const r = Math.ceil(4 * psfSigma);
  const i0 = Math.max(0, Math.round(x) - r), i1 = Math.min(width - 1, Math.round(x) + r);
  const j0 = Math.max(0, Math.round(y) - r), j1 = Math.min(height - 1, Math.round(y) + r);
  let added = 0;
  for (let j = j0; j <= j1; j++) {
    for (let i = i0; i <= i1; i++) {
      const p = j * width + i;
      if (!Number.isFinite(data[p])) continue;   // a masked pixel stays masked
      const d2 = (i - x) ** 2 + (j - y) ** 2;
      data[p] += amplitude * Math.exp(-d2 / (2 * psfSigma * psfSigma));
      added++;
    }
  }
  return added;
}

/**
 * Find point sources in one stamp.
 *
 * `data` is row-major with `data[j*width + i]`, NaN for masked pixels, in the
 * frame's own units. Returned centroids are fractional array indices, so the
 * caller converts to sky with that frame's WCS and nothing here knows any
 * astronomy.
 *
 * `minPixels` defaults to one. See the Python for why that is right and not a
 * slip: the matched filter has already imposed the "several adjacent pixels
 * agree" requirement that an area cut is for, and what actually separates a
 * cosmic ray from a star is shape, which is `maxSharpness`.
 */
export function detect(data, width, height, {
  thresholdSigma = 5,
  minPixels = 1,
  psfSigma = PSF_SIGMA_PX,
  maxSharpness = 0.92,
  sharpnessMinSnr = 10,
  maxSources = 500,
  backgroundBox = BACKGROUND_BOX_PX,
  // mean(sum w^2) of whatever resampling produced these pixels; the bake
  // records it per epoch and `fits.reproject` measures it for live reads.
  // Resampling shrinks the measured per-pixel scatter by sqrt(this) while
  // correlating neighbours to match, so a matched filter assuming independent
  // pixels overstates every significance by the inverse. Dividing the sigma
  // back out keeps "n sigma" meaning n sigma of noise. Mirrors detect.py.
  pixelNoiseShrink = 1,
  // This image's own `backgroundMap`, when the caller has already computed it
  // and will use it again: the ML screen's features read the same surface.
  background = null,
} = {}) {
  if (!width || !height) {
    return { background: 0, sigma: 0, thresholdSigma, nUsed: 0, nClipped: 0, sources: [] };
  }
  let { bg: bgMap, sigma, nUsed, nClipped, level } = background
    || backgroundMap(data, width, height, backgroundBox);
  // Filter-level widening, not 1/sqrt(shrink); see resampledFilterNorm.
  if (pixelNoiseShrink > 0 && pixelNoiseShrink < 1) sigma *= resampledFilterNorm(psfSigma);
  const base = { background: level, sigma, thresholdSigma, nUsed, nClipped, sources: [] };
  if (!(sigma > 0)) return base;

  const { kernel, r, n: kn } = gaussianKernel(psfSigma);

  const excess = new Float64Array(width * height);
  for (let p = 0; p < excess.length; p++) {
    const v = data[p];
    excess[p] = Number.isFinite(v) ? v - bgMap[p] : NaN;
  }

  // The amplitude estimate is sum(k*x)/sum(k) and its noise is
  // sigma*sqrt(sum(k^2))/sum(k), so the significance -- which is what a
  // threshold is stated in -- is sum(k*x)/(sigma*sqrt(sum(k^2))). Written that
  // way, partial kernels at the frame edge and around NaN need no special case:
  // accumulate over the pixels that exist and both sums shrink together.
  const snr = new Float64Array(width * height);
  for (let j = 0; j < height; j++) {
    for (let i = 0; i < width; i++) {
      let acc = 0, q = 0;
      for (let dj = -r; dj <= r; dj++) {
        const jj = j + dj;
        if (jj < 0 || jj >= height) continue;
        const krow = (dj + r) * kn + r;
        const erow = jj * width;
        for (let di = -r; di <= r; di++) {
          const ii = i + di;
          if (ii < 0 || ii >= width) continue;
          const e = excess[erow + ii];
          if (!(e === e)) continue;
          const w = kernel[krow + di];
          acc += w * e;
          q += w * w;
        }
      }
      snr[j * width + i] = q > 0 ? acc / (sigma * Math.sqrt(q)) : 0;
    }
  }

  const sources = extract(snr, excess, width, height, thresholdSigma, minPixels, r)
    // The shape cut only applies where shape is measured. Sharpness is
    // peak/flux and flux is a sum over ~25 pixels, so its own noise is 5 sigma
    // of pixel noise -- about 28% on a 9-sigma source. Applied blindly it then
    // deletes faint *real* sources at random, which is the one kind of
    // incompleteness you can neither afford nor write down. It is also
    // unnecessary: a cosmic ray is bright by construction.
    .filter((s) => s.snr < sharpnessMinSnr || s.sharpness <= maxSharpness)
    .sort((a, b) => b.snr - a.snr)
    .slice(0, maxSources);
  return { ...base, sources };
}

/** Matched-filter significance at a position given in advance: forced
 *  photometry, with the PSF centred on (x, y) to the sub-pixel and the same
 *  background, noise and resampling correction as `detect`, so at a whole-pixel
 *  position it is exactly the value `detect` thresholds there. For positions a
 *  fit predicted, where a threshold would join the body to its neighbours; and
 *  centred on (x, y) rather than its nearest pixel, which made the answer
 *  depend on rounding -- 16.8 and 20.6 sigma for one body read a quarter pixel
 *  apart. NaN off the array or on a masked pixel. Mirrors `forced_snr`. */
export function forcedSnr(data, width, height, x, y, {
  psfSigma = PSF_SIGMA_PX,
  backgroundBox = BACKGROUND_BOX_PX,
  pixelNoiseShrink = 1,
  background = null,
} = {}) {
  const i0 = Math.round(x), j0 = Math.round(y);
  if (!(i0 >= 0 && i0 < width && j0 >= 0 && j0 < height)) return NaN;
  if (!Number.isFinite(data[j0 * width + i0])) return NaN;
  // `background`: this image's `backgroundMap`, when a caller reading many
  // positions has already computed it.
  let { bg: bgMap, sigma } = background || backgroundMap(data, width, height, backgroundBox);
  if (pixelNoiseShrink > 0 && pixelNoiseShrink < 1) sigma *= resampledFilterNorm(psfSigma);
  if (!(sigma > 0)) return NaN;
  // One pixel wider than the detection kernel, so a Gaussian shifted by up to
  // half a pixel is not cut off on its far side.
  const r = gaussianKernel(psfSigma).r + 1;
  const twoS2 = 2 * psfSigma * psfSigma;
  let acc = 0, q = 0;
  for (let jj = Math.max(0, j0 - r); jj < Math.min(height, j0 + r + 1); jj++) {
    for (let ii = Math.max(0, i0 - r); ii < Math.min(width, i0 + r + 1); ii++) {
      const v = data[jj * width + ii];
      if (!Number.isFinite(v)) continue;
      const w = Math.exp(-((ii - x) ** 2 + (jj - y) ** 2) / twoS2);
      acc += w * (v - bgMap[jj * width + ii]);
      q += w * w;
    }
  }
  return q > 0 ? acc / (sigma * Math.sqrt(q)) : NaN;
}

/** PSF amplitude at a known position and its 1-sigma error, frame units:
 *  the least-squares fit sum(g*(v-bg))/sum(g*g) of a Gaussian centred on
 *  (x, y), error sigma/sqrt(sum(g*g)). Total flux is amplitude times
 *  2*pi*psfSigma^2. On resampled pixels the error is widened exactly as
 *  forcedSnr widens its noise -- the fit sums the same correlated neighbours --
 *  so amplitude/error is the same significance. Mirrors `forced_photometry` in
 *  detect.py, whose docstring has the reason. */
export function forcedPhotometry(data, width, height, x, y, {
  psfSigma = PSF_SIGMA_PX, backgroundBox = BACKGROUND_BOX_PX, background = null,
  pixelNoiseShrink = 1,
} = {}) {
  const i0 = Math.round(x), j0 = Math.round(y);
  if (!(i0 >= 0 && i0 < width && j0 >= 0 && j0 < height)) return { amplitude: NaN, error: NaN };
  if (!Number.isFinite(data[j0 * width + i0])) return { amplitude: NaN, error: NaN };
  // `background`, when given, is this image's own `backgroundMap`, computed
  // once by a caller measuring many positions on it.
  let { bg: bgMap, sigma } = background || backgroundMap(data, width, height, backgroundBox);
  if (pixelNoiseShrink > 0 && pixelNoiseShrink < 1) sigma *= resampledFilterNorm(psfSigma);
  const r = gaussianKernel(psfSigma).r + 1;
  const twoS2 = 2 * psfSigma * psfSigma;
  let acc = 0, gg = 0;
  for (let jj = Math.max(0, j0 - r); jj < Math.min(height, j0 + r + 1); jj++) {
    for (let ii = Math.max(0, i0 - r); ii < Math.min(width, i0 + r + 1); ii++) {
      const v = data[jj * width + ii];
      if (!Number.isFinite(v)) continue;
      const g = Math.exp(-((ii - x) ** 2 + (jj - y) ** 2) / twoS2);
      acc += g * (v - bgMap[jj * width + ii]);
      gg += g * g;
    }
  }
  return gg > 0 ? { amplitude: acc / gg, error: sigma / Math.sqrt(gg) } : { amplitude: NaN, error: NaN };
}

/** 8-connected components above threshold. Iterative: a bright blob is
 *  thousands of pixels and a recursive fill on that is a stack overflow. */
function extract(snr, excess, width, height, thresh, minPixels, r) {
  const seen = new Uint8Array(width * height);
  const out = [];
  const stack = [];
  for (let j0 = 0; j0 < height; j0++) {
    for (let i0 = 0; i0 < width; i0++) {
      const p0 = j0 * width + i0;
      if (seen[p0] || snr[p0] < thresh) continue;
      seen[p0] = 1;
      stack.length = 0;
      stack.push(p0);
      const pix = [];
      while (stack.length) {
        const p = stack.pop();
        pix.push(p);
        const i = p % width, j = (p - i) / width;
        for (let dj = -1; dj <= 1; dj++) {
          const jj = j + dj;
          if (jj < 0 || jj >= height) continue;
          for (let di = -1; di <= 1; di++) {
            const ii = i + di;
            if (ii < 0 || ii >= width) continue;
            const q = jj * width + ii;
            if (!seen[q] && snr[q] >= thresh) { seen[q] = 1; stack.push(q); }
          }
        }
      }
      if (pix.length < minPixels) continue;
      out.push(measure(pix, snr, excess, width, height, r));
    }
  }
  return out;
}

/** Centroid and photometry in a fixed aperture around the peak.
 *
 *  Fixed, not over the footprint: a source detected on one pixel has all its
 *  footprint flux in that pixel by definition, so peak/flux would be exactly 1
 *  and the sharpness cut would discard every marginal detection while keeping
 *  the cosmic rays it exists to remove.
 *
 *  Weighted by the *unfiltered* excess -- the filter is a smoothing kernel, and
 *  centroiding a smoothed image throws away the sub-pixel information the
 *  smoothing spread out. */
function measure(pix, snr, excess, width, height, r) {
  let best = -Infinity, bp = pix[0];
  for (const p of pix) if (snr[p] > best) { best = snr[p]; bp = p; }
  const bi = bp % width, bj = (bp - bi) / width;

  let wsum = 0, xsum = 0, ysum = 0, flux = 0;
  let peak = excess[bp];
  if (!(peak === peak)) peak = 0;
  for (let j = Math.max(0, bj - r); j <= Math.min(height - 1, bj + r); j++) {
    for (let i = Math.max(0, bi - r); i <= Math.min(width - 1, bi + r); i++) {
      const e = excess[j * width + i];
      if (!(e === e)) continue;
      flux += e;
      const w = e > 0 ? e : 0;
      wsum += w; xsum += w * i; ysum += w * j;
    }
  }
  const cx = wsum > 0 ? xsum / wsum : bi;
  const cy = wsum > 0 ? ysum / wsum : bj;
  const sharp = flux > 0 ? peak / flux : 1;
  return {
    x: cx, y: cy, snr: best, flux, peak, npix: pix.length,
    sharpness: Math.min(1, Math.max(0, sharp)),
  };
}
