// Challenge test targets. ICRS positions at epoch J2000.0 and proper motions (mas/yr, pmra includes cos dec),
// from SIMBAD (TAP via the simbad.harvard.edu mirror, queried 2026-11-15). Sources: Gaia DR3 (2020yCat.1350....0G)
// except WISE 0855 (Kirkpatrick et al. 2021, 2021ApJS..253....7K) and RW Aur (van Leeuwen 2007, 2007A&A...474..653V).
import { project } from './wcs.js';

export const TARGETS = [
  { id: 'barnard', name: "Barnard's Star", ra: 269.4520769586187, dec: 4.693364966576667, pmra: -801.551, pmdec: 10362.394, note: 'live.note_barnard' },
  { id: 'epsindib', name: 'Epsilon Indi B', ra: 331.04414658852124, dec: -56.782844195116105, pmra: 3981.977, pmdec: -2466.832, note: 'live.note_epsind' },
  { id: 'wise0855', name: 'WISE 0855−0714', ra: 133.81890825, dec: -7.247048000000001, pmra: -8123.7, pmdec: 673.2, note: 'live.note_wise' },
  { id: 'rwaur', name: 'RW Aurigae', ra: 76.95652603458333, dec: 30.401438173888888, pmra: 7.23, pmdec: -20.33, note: 'live.note_variable' },
  { id: 'rdra', name: 'R Draconis', ra: 248.16761083295, dec: 66.75497941722, pmra: -20.643, pmdec: 6.149, note: 'live.note_variable' },
];
export const J2000_MJD = 51544.5;

// Linear proper-motion propagation from J2000 to `mjd`. Returns [ra, dec] in degrees.
export function propagate(tg, mjd) {
  const yr = (mjd - J2000_MJD) / 365.25;
  const dec = tg.dec + tg.pmdec * yr / 3.6e6;
  const ra = tg.ra + tg.pmra * yr / 3.6e6 / Math.cos(tg.dec * Math.PI / 180);
  return [((ra % 360) + 360) % 360, dec];
}

// The target, if any, that has moved more than MOVED_ARCMIN between epoch 2000 and `mjd` and whose path over that time
// passes within TRACK_ARCMIN of (ra, dec). A catalogue position of a fast-moving star (epoch 2000, 2010, 2016…) lies on
// that path, in a patch of sky the star has since left.
const MOVED_ARCMIN = 0.5, TRACK_ARCMIN = 1;
export function movedTarget(ra, dec, mjd) {
  return TARGETS.find(tg => {
    const a = project(tg.ra, tg.dec, ra, dec), b = project(...propagate(tg, mjd), ra, dec);
    if (!a || !b) return false;
    const dx = b[0] - a[0], dy = b[1] - a[1], len2 = dx * dx + dy * dy;
    if (Math.sqrt(len2) * 60 < MOVED_ARCMIN) return false;
    const u = Math.max(0, Math.min(1, -(a[0] * dx + a[1] * dy) / len2));
    return Math.hypot(a[0] + u * dx, a[1] + u * dy) * 60 < TRACK_ARCMIN;
  }) || null;
}
