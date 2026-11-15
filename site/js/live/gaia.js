// Gaia DR3 stars for a field, from public catalogue services that allow access from a browser. Pure functions.
import { project } from './wcs.js';
import { SCALE_ARCSEC } from './grid.js';

export const GAIA_EPOCH_MJD = 57388.5; // J2016.0, the reference epoch of Gaia DR3 positions
export const MAX_STARS = 400; // brightest stars asked for; fainter ones are far below what SPHEREx can show

// Each service returns text rows of: source id, RA, Dec (deg, ICRS, J2016.0), pmRA (mas/yr, includes cos dec), pmDec, G mag;
// brightest first. Tried in order (both checked for cross-origin access on 2026-11-15).
export const SERVICES = [
  { name: 'VizieR I/355/gaiadr3 (CDS; CfA mirror)', timeout: 12000,
    url: (ra, dec, r) => `https://vizier.cfa.harvard.edu/viz-bin/asu-tsv?-source=I/355/gaiadr3&-c=${ra}%20${dec < 0 ? '' : '%2B'}${dec}&-c.rm=${(r * 60).toFixed(3)}` +
      `&-out=Source,RA_ICRS,DE_ICRS,pmRA,pmDE,Gmag&-out.max=${MAX_STARS}&-sort=Gmag` },
  { name: 'GAVO Data Center TAP', timeout: 45000, // slower in crowded fields
    url: (ra, dec, r) => 'https://dc.zah.uni-heidelberg.de/tap/sync?REQUEST=doQuery&LANG=ADQL&FORMAT=csv&QUERY=' + encodeURIComponent(
      `SELECT TOP ${MAX_STARS} source_id,ra,dec,pmra,pmdec,phot_g_mean_mag FROM gaia.dr3lite WHERE 1=CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',${ra},${dec},${r})) ORDER BY phot_g_mean_mag`) },
];

// CSV or TSV rows -> [{id (string: 19 digits do not fit a JS number), ra, dec, pmra, pmdec, g}]; pm and g are null when the catalogue has none.
export function parseStars(text) {
  const num = x => (x === '' || !Number.isFinite(+x) ? null : +x), out = [];
  for (const line of text.split('\n')) {
    const c = line.split(/[\t,]/).map(x => x.trim());
    if (c.length < 6 || !/^\d+$/.test(c[0]) || num(c[1]) == null || num(c[2]) == null) continue; // header, unit and comment lines
    out.push({ id: c[0], ra: +c[1], dec: +c[2], pmra: num(c[3]), pmdec: num(c[4]), g: num(c[5]) });
  }
  return out;
}

// Stars within radiusDeg of (ra, dec). Returns {stars, service}; throws when no service answers.
export async function fetchGaia(ra, dec, radiusDeg, fetchFn = fetch) {
  for (const s of SERVICES) {
    try {
      const r = await Promise.race([fetchFn(s.url(ra.toFixed(6), dec.toFixed(6), radiusDeg.toFixed(4))),
        new Promise((_, rej) => setTimeout(() => rej(new Error('timeout')), s.timeout))]);
      const text = r.ok ? await r.text() : '';
      if (/source/i.test(text)) return { stars: parseStars(text), service: s.name }; // a table header came back
    } catch { /* try the next service */ }
  }
  throw new Error('no Gaia service reachable');
}

// Position [x, y] of a star on the S x S live grid (pixel units, fractional) at time `mjd`:
// its J2016.0 position moved by its proper motion. Stars without a measured motion stay put.
export function starXY(st, mjd, S, ra0, dec0) {
  const yr = (mjd - GAIA_EPOCH_MJD) / 365.25;
  const dec = st.dec + (st.pmdec || 0) * yr / 3.6e6, ra = st.ra + (st.pmra || 0) * yr / 3.6e6 / Math.cos(st.dec * Math.PI / 180);
  const p = project(ra, dec, ra0, dec0), c = (S - 1) / 2, s = SCALE_ARCSEC / 3600;
  return p ? [c - p[0] / s, c - p[1] / s] : null;
}
// Circle radius in grid pixels: brighter stars (smaller G) get bigger circles.
export const starRadius = g => Math.min(3, Math.max(0.8, 1 + (16 - (g ?? 21)) * 0.25));
// Total proper motion in arcsec per year (0 when not measured).
export const starPM = st => Math.hypot(st.pmra || 0, st.pmdec || 0) / 1000;
