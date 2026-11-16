// Live methane test on the quick-look spectrum (E36, docs/protocols/E36.md). One-sided by design: on browser
// spectra of real brown dwarfs, 0 of 63 known L dwarfs and 6 of 6 positive calls among 15 known T/Y dwarfs were
// right, but several faint T/Y dwarfs showed no clear dip, so only a dip is reported, never "no methane".
// Index = F(3.2-3.4 um) / mean(F(2.9-3.1), F(3.9-4.1)); dip if index <= 0.356 - 3 sigma (rule of E15/E22).
export const THRESHOLD = 0.35573814952203764;
const BINS = { c30: [2.9, 3.1], m33: [3.2, 3.4], c40: [3.9, 4.1] };

function binMean(rows, lo, hi) {
  const r = rows.filter(x => x.wl >= lo && x.wl < hi && Number.isFinite(x.flux) && x.err > 0);
  if (r.length < 3) return null;
  const sorted = r.map(x => x.flux).sort((a, b) => a - b), med = sorted[sorted.length >> 1];
  const mad = 1.4826 * [...r.map(x => Math.abs(x.flux - med))].sort((a, b) => a - b)[r.length >> 1];
  const k = r.filter(x => Math.abs(x.flux - med) < 5 * Math.max(mad, 1e-9));
  if (k.length < 3) return null;
  let sw = 0, swx = 0; for (const x of k) { const w = 1 / (x.err * x.err); sw += w; swx += w * x.flux; }
  const mu = swx / sw, sd = Math.sqrt(k.reduce((s, x) => s + (x.flux - mu) ** 2, 0) / (k.length - 1));
  return { mu, err: Math.max(Math.sqrt(1 / sw), sd / Math.sqrt(k.length)), n: k.length };
}

/** rows: [{wl, flux, err}] in mJy. Returns {index, err, dip: true|false|null}; null = not measurable. */
export function methaneTest(rows) {
  const b = Object.fromEntries(Object.entries(BINS).map(([k, [lo, hi]]) => [k, binMean(rows, lo, hi)]));
  if (!b.c30 || !b.m33 || !b.c40) return { index: NaN, err: NaN, dip: null };
  const cont = (b.c30.mu + b.c40.mu) / 2, ce = 0.5 * Math.hypot(b.c30.err, b.c40.err);
  if (!(cont > 2 * ce)) return { index: NaN, err: NaN, dip: null };
  const index = b.m33.mu / cont, err = Math.abs(index) * Math.hypot(b.m33.err / b.m33.mu, ce / cont);
  return { index, err, dip: Number.isFinite(err) ? index <= THRESHOLD - 3 * err : null };
}
