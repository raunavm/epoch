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
 * Change model, revision 2 (docs/INTERFACES.md "Change model — revision 2").
 * frames: [{v: Float32Array(S*S) (NaN = no data), lam: Float32Array(S*S) µm, det, mjd}]; f: fractional noise floor.
 * Per-pixel noise sigma_e,p^2 = sigma_e^2 + (f |S_p|)^2, where S_p is the static model (corrected maps) or the larger
 * |background-subtracted value| of the two frames (naive maps). Same noise model for both, so the comparison is fair.
 * Returns {bg[], sigma[], vs[], model[], corr[], naiveAll[], naiveDet[], nCorr, nNaiveAll, nNaiveDet}.
 * Counts use countChanges after a 3x3 median single-pixel spike rejection (live mode reads no FLAGS plane).
 */
export function changeModel(frames, S, f = 0.1) {
  const N = frames.length, P = S * S;
  const bg = frames.map(fr => clippedMedian(fr.v));
  const vs = frames.map((fr, e) => fr.v.map(x => x - bg[e]));
  const sigma = vs.map(madSigma);
  const noise = (e, Sp) => Math.hypot(sigma[e], f * Math.abs(Sp));
  const model = frames.map(() => new Float32Array(P).fill(NaN));
  for (const d of new Set(frames.map(fr => fr.det))) {
    const E = frames.map((fr, e) => (fr.det === d ? e : -1)).filter(e => e >= 0);
    const v = new Float64Array(N), lam = new Float64Array(N), w = new Float64Array(N);
    for (const e of E) w[e] = 1 / sigma[e] ** 2;
    for (let p = 0; p < P; p++) {
      let idx = [];
      for (const e of E) { v[e] = vs[e][p]; lam[e] = frames[e].lam[p]; if (Number.isFinite(v[e]) && Number.isFinite(lam[e]) && Number.isFinite(w[e]) && w[e] > 0) idx.push(e); }
      if (!idx.length) continue;
      let fit = wfit(v, lam, w, idx);
      for (let round = 0; round < 2 && idx.length > 1; round++) { // drop only the single worst frame, if > 4 sigma
        let worst = -1, zmax = 4;
        for (const e of idx) { const m = fit[0] + fit[1] * (lam[e] - fit[2]), z = Math.abs(v[e] - m) / noise(e, m); if (z > zmax) { zmax = z; worst = e; } }
        if (worst < 0) break;
        idx = idx.filter(e => e !== worst); fit = wfit(v, lam, w, idx);
      }
      for (const e of E) if (Number.isFinite(lam[e])) model[e][p] = fit[0] + fit[1] * (lam[e] - fit[2]);
    }
  }
  const corr = vs.map((x, e) => x.map((y, p) => (y - model[e][p]) / noise(e, model[e][p])));
  const diff = (e, q) => vs[e].map((y, p) => {
    const z = vs[q][p], Sp = Math.max(Math.abs(y), Math.abs(z));
    return (y - z) / Math.hypot(noise(e, Sp), noise(q, Sp));
  });
  const order = frames.map((_, e) => e).sort((a, b) => frames[a].mjd - frames[b].mjd);
  const naiveAll = new Array(N).fill(null), naiveDet = new Array(N).fill(null), last = {};
  order.forEach((e, k) => {
    if (k) naiveAll[e] = diff(e, order[k - 1]);
    const d = frames[e].det;
    if (last[d] != null) naiveDet[e] = diff(e, last[d]);
    last[d] = e;
  });
  const count = maps => maps.reduce((n, m) => n + (m ? countChanges(despike(m, S), S) : 0), 0);
  return { bg, sigma, vs, model, corr, naiveAll, naiveDet, nCorr: count(corr), nNaiveAll: count(naiveAll), nNaiveDet: count(naiveDet) };
}

// Single-pixel spike rejection: a pixel with |m| > thr whose 3x3 neighbourhood median of |m| is below thr/2 is set
// to 0. A resampled single detector pixel lights up at most ~4 of the 9 pixels, so its median stays at noise level;
// a real source (PSF FWHM ~1.2 grid pixels) keeps most of its 3x3 neighbourhood above thr/2.
export function despike(m, S, thr = 5) {
  const out = Float32Array.from(m);
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
    const p = y * S + x;
    if (!(Math.abs(m[p]) > thr)) continue;
    const nb = [];
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      const X = x + dx, Y = y + dy;
      if (X >= 0 && Y >= 0 && X < S && Y < S && Number.isFinite(m[Y * S + X])) nb.push(Math.abs(m[Y * S + X]));
    }
    if (median(nb) < thr / 2) out[p] = 0;
  }
  return out;
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
