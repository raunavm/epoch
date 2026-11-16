// Reflected light: how bright a body of a given size and albedo is, and how far
// away it can be before it drops below a limit.
//
// The mirror of `src/tracklet/physics/reflected.py`, which is the authority and
// carries the reasoning; the golden vectors hold the two to 1e-12. The console
// needs it to turn a completeness map -- what fraction of planted bodies the
// search recovers at each brightness and distance -- into the statement the
// project exists to make: which bodies this patch of sky rules out.

export const AU_KM = 1.495978707e8;
export const R_EARTH_KM = 6371.0;
export const AB_ZP_JY = 3631.0;

//: Sun's apparent AB magnitude at 1 AU, by band: Willmer (2018) absolute AB
//: magnitudes shifted by the distance modulus of 1 AU. These once ran the
//: Vega way, brighter from z to K, which is impossible in AB; reflected.py
//: carries the account and test_reflected.py pins the ordering.
export const M_SUN_AB = { z: -27.07, V: -26.76, J: -27.03, H: -26.91, K: -26.49 };

//: Plausible geometric albedo for a cold, methane-rich ice giant; z (~0.9 um)
//: is the SPHEREx channel-1 regime, least suppressed by methane.
export const ALBEDO_ICE_GIANT = { z: 0.30, V: 0.45, J: 0.10, H: 0.04, K: 0.02 };

export const radiusAu = (radiusEarth) => radiusEarth * R_EARTH_KM / AU_KM;

/** Apparent AB magnitude of a reflecting sphere, 1/d^4 with d the
 *  heliocentric distance and, by default, the geocentric distance too. */
export function apparentMag(radiusEarth, dAu, albedo, { band = 'J', geoAu = null, phase = 1 } = {}) {
  const D = geoAu == null ? dAu : geoAu;
  const r = radiusAu(radiusEarth);
  const ratio = albedo * phase * r * r / (dAu * dAu * D * D);
  return M_SUN_AB[band] - 2.5 * Math.log10(ratio);
}

/** Heliocentric distance at which the body falls to `mLimit`. */
export function maxDistance(radiusEarth, albedo, mLimit, { band = 'J' } = {}) {
  const r = radiusAu(radiusEarth);
  const d4 = albedo * r * r * 10 ** ((mLimit - M_SUN_AB[band]) / 2.5);
  return d4 ** 0.25;
}

export const fluxToAb = (fluxJy) => -2.5 * Math.log10(fluxJy / AB_ZP_JY);
export const abToFlux = (magAb) => AB_ZP_JY * 10 ** (-0.4 * magAb);
