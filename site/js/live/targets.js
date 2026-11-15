// Challenge test targets. ICRS positions at epoch J2000.0 and proper motions (mas/yr, pmra includes cos dec),
// from SIMBAD (TAP via the simbad.harvard.edu mirror, queried 2026-11-15). Sources: Gaia DR3 (2020yCat.1350....0G)
// except WISE 0855 (Kirkpatrick et al. 2021, 2021ApJS..253....7K) and RW Aur (van Leeuwen 2007, 2007A&A...474..653V).
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
