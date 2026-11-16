// E30 two-stage change detector (docs/protocols/E30.md). Pure computation: no DOM, no dependencies.
// Mirrors src/orrery/detect.py (parity: site/tests/detect.test.mjs with site/tests/fixtures/e30_parity.json).
// Stage 1: per-detector median of the other pictures (M3b: rejection of the single worst frame > 4 sigma, at most
// two rounds; leave-one-out per-pixel scale c). Candidates: 3x3 local maxima with z >= zmin.
// Stage 2: a 9x9 five-channel cutout scored by the model in site/data/model/e30.json.

const C_MIN_FRAMES = 5, HALF = 4, W = 2 * HALF + 1, NDET = 6;

// Median of the finite values a[k] for k in idx where keep[k]; NaN if none. buf: scratch Float64Array.
function medianKept(a, idx, keep, buf) {
  let n = 0;
  for (const k of idx) if (keep[k]) buf[n++] = a[k];
  if (!n) return NaN;
  const s = buf.subarray(0, n).sort();
  return n % 2 ? s[n >> 1] : (s[n / 2 - 1] + s[n / 2]) / 2;
}

/**
 * M3b reference (scripts/run_e27.py m3b_maps). vs: per-picture Float32Array/Float64Array(S*S) (background-subtracted,
 * NaN = no data); sigma: per-picture noise; dets: per-picture detector; f: fractional floor.
 * Returns {med, z, c}: per picture Float64Array(S*S), NaN where the picture has no map (its detector has one picture).
 */
export function medianReference(vs, sigma, dets, S, f, onProgress) {
  const N = vs.length, P = S * S;
  const med = [], z = [], c = [];
  for (let e = 0; e < N; e++) { med.push(new Float64Array(P).fill(NaN)); z.push(new Float64Array(P).fill(NaN)); c.push(new Float64Array(P).fill(NaN)); }
  const groups = [...new Set(dets)].sort((a, b) => a - b);
  groups.forEach((d, gi) => {
    const E = []; for (let e = 0; e < N; e++) if (dets[e] === d) E.push(e);
    const n = E.length;
    if (n < 2) return;
    const s2 = E.map(e => sigma[e] * sigma[e]);
    const V = new Float64Array(n), fin = new Uint8Array(n), keep = new Uint8Array(n), buf = new Float64Array(n);
    const others = E.map((_, i) => E.map((_, k) => k).filter(k => k !== i));
    for (let p = 0; p < P; p++) {
      for (let k = 0; k < n; k++) { V[k] = vs[E[k]][p]; fin[k] = Number.isFinite(V[k]) ? 1 : 0; }
      for (let i = 0; i < n; i++) {
        const o = others[i];
        for (const k of o) keep[k] = fin[k];
        let m = medianKept(V, o, keep, buf);
        for (let round = 0; round < 2; round++) {
          let worst = -1, zw = -Infinity, cnt = 0;
          for (const k of o) {
            if (keep[k]) cnt++;
            const fm = f * m, zz = keep[k] ? Math.abs(V[k] - m) / Math.sqrt(s2[k] + fm * fm) : -1;
            if (zz > zw) { zw = zz; worst = k; }
          }
          if (!(zw > 4 && cnt > 1)) break;
          keep[worst] = 0;
          m = medianKept(V, o, keep, buf);
        }
        const fm = f * m, e = E[i];
        const r = (V[i] - m) / Math.sqrt(s2[i] + fm * fm);
        let nk = 0;
        for (const k of o) if (keep[k]) buf[nk++] = Math.abs(V[k] - m) / Math.sqrt(s2[k] + fm * fm);
        let cc = 1;
        if (nk >= C_MIN_FRAMES) {
          const s = buf.subarray(0, nk).sort();
          cc = Math.max(1, 1.4826 * (nk % 2 ? s[nk >> 1] : (s[nk / 2 - 1] + s[nk / 2]) / 2));
        }
        med[e][p] = m; c[e][p] = cc; z[e][p] = r / cc;
      }
    }
    if (onProgress) onProgress((gi + 1) / groups.length / 2);
  });
  return { med, z, c };
}

/** Pixels with finite z >= zmin equal to the maximum of the finite values in their 3x3 neighbourhood; row-major. */
export function findCandidates(z, c, S, zmin = 3) {
  const out = [];
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
    const v = z[y * S + x];
    if (!(Number.isFinite(v) && v >= zmin)) continue;
    let top = true;
    for (let dy = -1; dy <= 1 && top; dy++) for (let dx = -1; dx <= 1; dx++) {
      const X = x + dx, Y = y + dy;
      if (X < 0 || Y < 0 || X >= S || Y >= S) continue;
      const q = z[Y * S + X];
      if (Number.isFinite(q) && q > v) { top = false; break; }
    }
    if (top) out.push({ x, y, z: v, c: c ? c[y * S + x] : NaN });
  }
  return out;
}

const g = x => Math.sign(x) * Math.log1p(Math.abs(x));

/** 9x9 cutout, channels g(vs/sigma), g(med/sigma), g(z), log c, finite(z); NaN and off-grid -> 0. Float64Array(405). */
export function cutout(vs, med, z, c, sigma, x, y, S) {
  const out = new Float64Array(5 * W * W);
  for (let dy = -HALF; dy <= HALF; dy++) for (let dx = -HALF; dx <= HALF; dx++) {
    const X = x + dx, Y = y + dy;
    if (X < 0 || Y < 0 || X >= S || Y >= S) continue;
    const p = Y * S + X, o = (dy + HALF) * W + dx + HALF;
    const vals = [g(vs[p] / sigma), g(med[p] / sigma), g(z[p]), Math.log(c[p]), Number.isFinite(z[p]) ? 1 : 0];
    for (let k = 0; k < 5; k++) out[k * W * W + o] = Number.isFinite(vals[k]) ? vals[k] : 0;
  }
  return out;
}

export function onehot(det) {
  const o = new Float64Array(NDET); o[det - 1] = 1; return o;
}

const flatCache = new WeakMap();
function flat(model) {
  let w = flatCache.get(model);
  if (!w) {
    w = {};
    for (const [k, v] of Object.entries(model.weights)) w[k] = Array.isArray(v) ? Float64Array.from(v.flat(3)) : v;
    flatCache.set(model, w);
  }
  return w;
}

// Valid 3x3 cross-correlation: input [C][H][H] -> [O][H-2][H-2], then ReLU.
function conv(inp, C, H, Wt, b, O) {
  const h = H - 2, out = new Float64Array(O * h * h);
  for (let o = 0; o < O; o++) for (let i = 0; i < h; i++) for (let j = 0; j < h; j++) {
    let s = b[o];
    for (let ci = 0; ci < C; ci++) {
      const wb = (o * C + ci) * 9, ib = ci * H * H;
      for (let di = 0; di < 3; di++) for (let dj = 0; dj < 3; dj++) s += Wt[wb + di * 3 + dj] * inp[ib + (i + di) * H + j + dj];
    }
    out[(o * h + i) * h + j] = s > 0 ? s : 0;
  }
  return out;
}

/** 6 hand features of the logistic-regression ablation (orrery.detect.lr_features). */
export function lrFeatures(cut) {
  const Z = k => { const v = cut[2 * W * W + k]; return Math.sign(v) * Math.expm1(Math.abs(v)); };
  const ctr = HALF * W + HALF, zc = Z(ctr);
  let s3 = 0; for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) s3 += Z(ctr + dy * W + dx);
  let valid = 0; for (let k = 0; k < W * W; k++) valid += cut[4 * W * W + k];
  return [g(zc), g(s3 / 3), g(zc - (s3 - zc) / 8), cut[3 * W * W + ctr], cut[W * W + ctr], valid / (W * W)];
}

/** Logit of one cutout (Float64Array 405) with detector one-hot `scalars` (6). model: e30.json's model / ablation. */
export function forward(model, cut, scalars) {
  if (model.kind === 'zrank') { const v = cut[2 * W * W + HALF * W + HALF]; return Math.sign(v) * Math.expm1(Math.abs(v)); }
  const w = flat(model);
  if (model.kind === 'logreg') {
    const F = lrFeatures(cut);
    let s = +w.b; for (let k = 0; k < 6; k++) s += (F[k] - w.mean[k]) / w.std[k] * w.w[k];
    return s;
  }
  let a = conv(cut, 5, 9, w.c1w, w.c1b, 12);
  a = conv(a, 12, 7, w.c2w, w.c2b, 16);
  a = conv(a, 16, 5, w.c3w, w.c3b, 16);
  const hcat = new Float64Array(150); hcat.set(a); hcat.set(scalars, 144);
  let out = w.d2b[0];
  for (let u = 0; u < 16; u++) {
    let s = w.d1b[u];
    for (let k = 0; k < 150; k++) s += hcat[k] * w.d1w[k * 16 + u];
    if (s > 0) out += s * w.d2w[u];
  }
  return out;
}

// False candidates per pixel at a score, log-linear interpolation of the calibration table (clamped at its ends).
function perPixel(table, s) {
  const x = table.score, y = table.false_per_pixel, n = x.length, lg = v => Math.log(Math.max(v, 1e-12));
  if (!(s > x[0])) return y[0];
  if (s >= x[n - 1]) return y[n - 1];
  let k = 1; while (x[k] < s) k++;
  const t = (s - x[k - 1]) / (x[k] - x[k - 1] || 1);
  return Math.exp(lg(y[k - 1]) + t * (lg(y[k]) - lg(y[k - 1])));
}

const tierOf = (s, t) => (s > t.r1 ? 'r1' : s > t.r3 ? 'r3' : s > t.r10 ? 'r10' : null);

/**
 * Score every picture's candidates. frames: the live view's frames [{det, mjd, obs_id, wl, ...}]; cm: changeModel's
 * output (vs, sigma); model: the parsed e30.json (its `use` picks the learned model or the z ranking) or one of its
 * model blocks. Returns {perFrame: [[{x, y, z, score, falsePerPicture, tier}]], ranked: [{idx, obs_id, mjd, det, wl, x,
 * y, z, score, falsePerPicture, tier}], thresholds, model: {id, sha256}, ms}. Thresholds are false candidates per 48 x 48
 * px picture on the calibration fields; falsePerPicture scales the per-pixel table by the picture's valid pixels.
 * model = null: candidates ranked by M3b z with no calibration (falsePerPicture NaN, tier null, thresholds null).
 */
export function detectChanges(frames, cm, S, f, model, onProgress) {
  const t0 = (globalThis.performance ?? Date).now();
  const m = !model ? { kind: 'zrank', id: 'e30-zrank-uncalibrated' } : model.model ? (model.use === 'zrank' ? model.zrank : model.model) : model;
  const zmin = model?.zmin ?? 3;
  const dets = frames.map(fr => fr.det);
  const ref = medianReference(cm.vs, cm.sigma, dets, S, f, onProgress);
  const perFrame = [], ranked = [];
  frames.forEach((fr, e) => {
    const z = ref.z[e];
    let nValid = 0; for (let p = 0; p < S * S; p++) if (Number.isFinite(z[p])) nValid++;
    const list = [];
    if (nValid) for (const cd of findCandidates(z, ref.c[e], S, zmin)) {
      const score = m.kind === 'zrank' ? cd.z : forward(m, cutout(cm.vs[e], ref.med[e], z, ref.c[e], cm.sigma[e], cd.x, cd.y, S), onehot(fr.det));
      const falsePerPicture = m.rate_table ? perPixel(m.rate_table, score) * nValid : NaN;
      const tier = m.thresholds ? tierOf(score, m.thresholds) : null;
      list.push({ x: cd.x, y: cd.y, z: cd.z, score, falsePerPicture, tier });
      ranked.push({ idx: e, obs_id: fr.obs_id, mjd: fr.mjd, det: fr.det, wl: fr.wl, x: cd.x, y: cd.y, z: cd.z, score, falsePerPicture, tier });
    }
    perFrame.push(list);
    if (onProgress && (e % 25 === 0 || e === frames.length - 1)) onProgress(0.5 + 0.5 * (e + 1) / frames.length);
  });
  ranked.sort((a, b) => b.score - a.score);
  return { perFrame, ranked, thresholds: m.thresholds ?? null, model: { id: m.id, sha256: model?.weights_sha256 ?? null },
    ms: (globalThis.performance ?? Date).now() - t0 };
}
