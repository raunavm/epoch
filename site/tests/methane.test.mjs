import test from 'node:test';
import assert from 'node:assert/strict';
import { methaneTest, THRESHOLD } from '../js/live/methane.js';
const spec = (m33) => { const r = []; for (let k = 0; k < 40; k++) { const wl = 2.8 + k * 0.035, f = wl >= 3.2 && wl < 3.4 ? m33 : 1; r.push({ wl, flux: f * (1 + 0.01 * Math.sin(k)), err: 0.02 }); } return r; };
test('deep 3.3 um dip is reported', () => { const t = methaneTest(spec(0.1)); assert.equal(t.dip, true); assert.ok(t.index < THRESHOLD); });
test('flat spectrum has no dip', () => { assert.equal(methaneTest(spec(1)).dip, false); });
test('too few points is not measurable', () => { assert.equal(methaneTest(spec(0.1).slice(0, 4)).dip, null); });
