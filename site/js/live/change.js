// Wavelength-aware change model (docs/INTERFACES.md, "Wavelength-aware change model").
// Pure functions; frames are already resampled onto a common S x S grid.

export function median(a) {
  const s = Float64Array.from(a).sort(), n = s.length;
  if (!n) return NaN;
  return n % 2 ? s[n >> 1] : (s[n / 2 - 1] + s[n / 2]) / 2;
}
const finite = arr => { const o = []; for (const x of arr) if (Number.isFinite(x)) o.push(x); return o; };

// Sigma-clipped median: 3 sigma, 3 iterations (centre = median, scale = standard deviation).
export function clippedMedian(arr) {
  let x = finite(arr);
  for (let it = 0; it < 3 && x.length > 2; it++) {
    const m = median(x), sd = Math.sqrt(x.reduce((s, v) => s + (v - m) ** 2, 0) / x.length);
    const k = x.filter(v => Math.abs(v - m) <= 3 * sd);
    if (k.length === x.length) break;
    x = k;
  }
  return median(x);
}
export function madSigma(arr) {
  const x = finite(arr), m = median(x);
  return 1.4826 * median(x.map(v => Math.abs(v - m)));
}

// Weighted LSQ of v = a + b (lam - lbar) with weights w; returns [a, b, lbar] (b = 0 if fewer than 4 points).
function wfit(v, lam, w, idx) {
  let sw = 0, sl = 0;
  for (const i of idx) { sw += w[i]; sl += w[i] * lam[i]; }
  const lbar = sl / sw;
  let sv = 0, sxx = 0, sxv = 0;
  for (const i of idx) { const x = lam[i] - lbar; sv += w[i] * v[i]; sxx += w[i] * x * x; sxv += w[i] * x * v[i]; }
  const a = sv / sw;
  const b = idx.length >= 4 && sxx > 0 ? sxv / sxx : 0;
  return [a, b, lbar];
}

/**
 * frames: [{v: Float32Array(S*S) (NaN = no data), lam: Float32Array(S*S) µm, det, mjd}]
 * Returns {bg[], sigma[], vs[] (background-subtracted), model[], corr[], naive[] (null if no earlier frame of
 * the same detector), nCorr, nNaive, finding}
 */
export function changeModel(frames, S) {
  const N = frames.length, P = S * S;
  const bg = frames.map(f => clippedMedian(f.v));
  const vs = frames.map((f, e) => f.v.map(x => x - bg[e]));
  const sigma = vs.map(madSigma);
  const model = frames.map(() => new Float32Array(P).fill(NaN));
  const dets = [...new Set(frames.map(f => f.det))];
  for (const d of dets) {
    const E = frames.map((f, e) => (f.det === d ? e : -1)).filter(e => e >= 0);
    const v = new Float64Array(N), lam = new Float64Array(N), w = new Float64Array(N);
    for (const e of E) w[e] = 1 / sigma[e] ** 2;
    for (let p = 0; p < P; p++) {
      let idx = [];
      for (const e of E) { v[e] = vs[e][p]; lam[e] = frames[e].lam[p]; if (Number.isFinite(v[e]) && Number.isFinite(lam[e]) && w[e] > 0 && Number.isFinite(w[e])) idx.push(e); }
      if (!idx.length) continue;
      let fit = wfit(v, lam, w, idx);
      for (let round = 0; round < 2; round++) {
        const keep = idx.filter(e => Math.abs(v[e] - (fit[0] + fit[1] * (lam[e] - fit[2]))) <= 4 * sigma[e]);
        if (keep.length === idx.length || !keep.length) break;
        idx = keep; fit = wfit(v, lam, w, idx);
      }
      for (const e of E) if (Number.isFinite(lam[e])) model[e][p] = fit[0] + fit[1] * (lam[e] - fit[2]);
    }
  }
  const corr = vs.map((x, e) => x.map((y, p) => (y - model[e][p]) / sigma[e]));
  const order = frames.map((_, e) => e).sort((a, b) => frames[a].mjd - frames[b].mjd);
  const naive = new Array(N).fill(null), last = {};
  for (const e of order) {
    const d = frames[e].det;
    if (last[d] != null) { const q = last[d], s = Math.hypot(sigma[e], sigma[q]); naive[e] = vs[e].map((y, p) => (y - vs[q][p]) / s); }
    last[d] = e;
  }
  let nCorr = 0, nNaive = 0;
  for (let e = 0; e < N; e++) {
    nCorr += countChanges(corr[e], S);
    if (naive[e]) nNaive += countChanges(naive[e], S);
  }
  return { bg, sigma, vs, model, corr, naive, nCorr, nNaive, finding: nNaive ? 1 - nCorr / nNaive : NaN };
}

// Connected groups (8-connectivity) of >= minPix pixels with |value| > thr.
export function countChanges(map, S, thr = 5, minPix = 2) {
  const seen = new Uint8Array(S * S); let n = 0;
  for (let p = 0; p < S * S; p++) {
    if (seen[p] || !(Math.abs(map[p]) > thr)) continue;
    let size = 0; const stack = [p]; seen[p] = 1;
    while (stack.length) {
      const q = stack.pop(), x = q % S, y = (q - x) / S; size++;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const X = x + dx, Y = y + dy, r = Y * S + X;
        if (X < 0 || Y < 0 || X >= S || Y >= S || seen[r] || !(Math.abs(map[r]) > thr)) continue;
        seen[r] = 1; stack.push(r);
      }
    }
    if (size >= minPix) n++;
  }
  return n;
}
