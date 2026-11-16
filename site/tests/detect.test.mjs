// Parity of site/js/live/detect.js with src/orrery/detect.py (fixture written by `scripts/run_e30.py fixture`).
//   node --test site/tests/
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { medianReference, findCandidates, cutout, forward, onehot, detectChanges } from '../js/live/detect.js';

const FIX = fileURLToPath(new URL('./fixtures/e30_parity.json', import.meta.url));
const MODEL = fileURLToPath(new URL('../data/model/e30.json', import.meta.url));
const ready = existsSync(FIX) && existsSync(MODEL);
const load = () => {
  const fx = JSON.parse(readFileSync(FIX));
  const num = a => Float64Array.from(a, v => (v === null ? NaN : v));
  return { fx, model: JSON.parse(readFileSync(MODEL)), vs: fx.vs.map(num), med: fx.med.map(num), z: fx.z.map(num), c: fx.c.map(num) };
};

function close(a, b, rel, what) {
  assert.equal(a.length, b.length, what);
  for (let p = 0; p < a.length; p++) {
    if (Number.isNaN(b[p])) { assert.ok(Number.isNaN(a[p]), `${what}[${p}] should be NaN`); continue; }
    assert.ok(Math.abs(a[p] - b[p]) <= rel * Math.max(1, Math.abs(b[p])), `${what}[${p}]: ${a[p]} vs ${b[p]}`);
  }
}

test('medianReference equals the Python M3b (med, z, c to 1e-5)', { skip: !ready }, () => {
  const { fx, vs, med, z, c } = load();
  const r = medianReference(vs, fx.sigma, fx.det, fx.S, fx.f);
  for (let e = 0; e < vs.length; e++) {
    close(r.med[e], med[e], 1e-5, `med ${e}`);
    close(r.z[e], z[e], 1e-5, `z ${e}`);
    close(r.c[e], c[e], 1e-5, `c ${e}`);
  }
});

test('identical candidates, cutouts and logits within 1e-3', { skip: !ready }, () => {
  const { fx, model, vs, med, z, c } = load();
  const got = [];
  for (let e = 0; e < vs.length; e++) for (const cd of findCandidates(z[e], c[e], fx.S, model.zmin)) got.push([e, cd.x, cd.y]);
  assert.deepEqual(got, fx.candidates.map(k => [k.e, k.x, k.y]));
  assert.ok(got.length > 0);
  for (const k of fx.candidates) {
    const cut = cutout(vs[k.e], med[k.e], z[k.e], c[k.e], fx.sigma[k.e], k.x, k.y, fx.S);
    close(cut, Float64Array.from(k.cutout, v => (v === null ? NaN : v)), 1e-9, 'cutout');
    const oh = onehot(fx.det[k.e]);
    assert.ok(Math.abs(forward(model.model, cut, oh) - k.logit_model) < 1e-3, `model logit ${k.e},${k.x},${k.y}`);
    assert.ok(Math.abs(forward(model.ablation, cut, oh) - k.logit_ablation) < 1e-3, `ablation logit ${k.e},${k.x},${k.y}`);
  }
});

test('detectChanges returns the interface, ranked by score', { skip: !ready }, () => {
  const { fx, model, vs } = load();
  const frames = fx.det.map((d, e) => ({ det: d, mjd: 61000 + e, obs_id: `o${e}`, wl: 1 + d }));
  const out = detectChanges(frames, { vs, sigma: fx.sigma }, fx.S, fx.f, model);
  assert.equal(out.perFrame.length, frames.length);
  assert.equal(out.ranked.length, fx.candidates.length);
  for (let k = 1; k < out.ranked.length; k++) assert.ok(out.ranked[k - 1].score >= out.ranked[k].score);
  const want = model.use === 'zrank' ? 'z' : 'logit_model';
  const byKey = new Map(fx.candidates.map(k => [`${k.e},${k.x},${k.y}`, k]));
  for (const r of out.ranked) {
    const k = byKey.get(`${r.idx},${r.x},${r.y}`);
    assert.ok(Math.abs(r.score - k[want]) < 1e-3);
    assert.ok(r.falsePerPicture > 0 && Number.isFinite(r.falsePerPicture));
    const t = out.thresholds;
    assert.equal(r.tier, r.score > t.r1 ? 'r1' : r.score > t.r3 ? 'r3' : r.score > t.r10 ? 'r10' : null);
  }
  assert.equal(out.model.sha256, model.weights_sha256);
  assert.ok(out.ms >= 0);
});

test('a picture never enters its own reference', () => {
  const S = 3, N = 6, vs = [], sigma = [], dets = [];
  for (let e = 0; e < N; e++) { vs.push(Float64Array.from({ length: S * S }, (_, p) => (e === 2 && p === 4 ? 100 : (e * 7 + p) % 5 - 2))); sigma.push(1); dets.push(1); }
  const r = medianReference(vs, sigma, dets, S, 0.1);
  assert.ok(r.z[2][4] > 10);                       // the spike is measured against the others
  assert.ok(Math.abs(r.z[3][4]) < 5);               // and is rejected from the others' reference
});

test('detectChanges without a model ranks M3b candidates by z', () => {
  const S = 12, frames = [], vs = [], sigma = [];
  let seed = 7; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647 - 0.5;
  for (let e = 0; e < 18; e++) {
    const v = Float64Array.from({ length: S * S }, () => rnd());
    if (e === 9) v[5 * S + 6] += 40;
    frames.push({ det: 1 + (e % 3), mjd: 61000 + e, obs_id: `o${e}`, wl: 1 }); vs.push(v); sigma.push(0.29);
  }
  const out = detectChanges(frames, { vs, sigma }, S, 0.1, null);
  assert.equal(out.model.id, 'e30-zrank-uncalibrated');
  assert.equal(out.thresholds, null);
  assert.deepEqual([out.ranked[0].idx, out.ranked[0].x, out.ranked[0].y], [9, 6, 5]);
  assert.equal(out.ranked[0].score, out.ranked[0].z);
  assert.ok(Number.isNaN(out.ranked[0].falsePerPicture) && out.ranked[0].tier === null);
});
