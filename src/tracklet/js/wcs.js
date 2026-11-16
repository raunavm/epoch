// TAN + SIP world coordinates. A port of src/tracklet/frames/wcs.py, and the
// Python is the authority: tests/golden/ holds vectors generated from it, and
// tests/js/run-golden.mjs fails if this file drifts from them by more than
// 1e-9 pixel. Two implementations of the same transform is a liability unless
// something keeps them honest, so something does.
//
// Conventions, identical to the Python side:
//   * pixel coordinates are 1-based FITS, centres on integers, matching CRPIX;
//   * intermediate world coordinates are degrees, x east-positive;
//   * every SIP coefficient present is applied, including A_0_0 and A_1_0.
//     SPHEREx ships A_0_0 = 0.483 and B_0_0 = -1.093, and dropping the
//     low-order terms -- which a lot of hand-rolled SIP code does, because they
//     are conventionally zero -- moves every position by 1.21 pixels, 7.5
//     arcsec, fifty times the mission's own astrometric error.

const DEG = Math.PI / 180;

export const INVERSE_TOL_PX = 1e-11;
export const INVERSE_MAX_ITER = 24;

/** Unpack {"1_2": v} coefficient maps into a flat [[p,q,v],...] list. */
function unpack(obj) {
  if (!obj) return [];
  return Object.entries(obj).map(([k, v]) => {
    const [p, q] = k.split('_').map(Number);
    return [p, q, v];
  });
}

function poly(terms, u, v) {
  let s = 0;
  for (let i = 0; i < terms.length; i++) {
    const [p, q, c] = terms[i];
    s += c * Math.pow(u, p) * Math.pow(v, q);
  }
  return s;
}

function dpoly(terms, u, v) {
  let du = 0, dv = 0;
  for (let i = 0; i < terms.length; i++) {
    const [p, q, c] = terms[i];
    if (p >= 1) du += c * p * Math.pow(u, p - 1) * Math.pow(v, q);
    if (q >= 1) dv += c * q * Math.pow(u, p) * Math.pow(v, q - 1);
  }
  return [du, dv];
}

export class Sip {
  constructor(spec) {
    this.a = unpack(spec && spec.a);
    this.b = unpack(spec && spec.b);
    this.ap = unpack(spec && spec.ap);
    this.bp = unpack(spec && spec.bp);
    this.order = (spec && spec.order) || 0;
    this.present = this.a.length > 0 || this.b.length > 0;
    this.hasInverse = this.ap.length > 0 || this.bp.length > 0;
  }

  forward(u, v) {
    return [u + poly(this.a, u, v), v + poly(this.b, u, v)];
  }

  jacobian(u, v) {
    const [adu, adv] = dpoly(this.a, u, v);
    const [bdu, bdv] = dpoly(this.b, u, v);
    return [1 + adu, adv, bdu, 1 + bdv];
  }

  /** AP/BP give a *fitted* inverse -- ~15 mas residual on real SPHEREx
   *  coefficients. Used only as the Newton starting point. */
  inverseGuess(up, vp) {
    if (!this.hasInverse) return [up, vp];
    return [up + poly(this.ap, up, vp), vp + poly(this.bp, up, vp)];
  }

  /** Newton on the forward map with its analytic Jacobian: ~1e-11 px in 2-3
   *  steps. The Jacobian is the identity plus ~1e-3 anywhere on a real
   *  detector, so no damping or line search is needed. */
  inverse(up, vp) {
    if (!this.present) return [up, vp];
    let [u, v] = this.inverseGuess(up, vp);
    for (let i = 0; i < INVERSE_MAX_ITER; i++) {
      const [fu, fv] = this.forward(u, v);
      const ru = fu - up, rv = fv - vp;
      const [j11, j12, j21, j22] = this.jacobian(u, v);
      const det = j11 * j22 - j12 * j21;
      if (det === 0) throw new Error('singular SIP Jacobian');
      const du = (ru * j22 - rv * j12) / det;
      const dv = (rv * j11 - ru * j21) / det;
      u -= du; v -= dv;
      if (Math.abs(du) < INVERSE_TOL_PX && Math.abs(dv) < INVERSE_TOL_PX) return [u, v];
    }
    throw new Error('SIP inverse did not converge');
  }
}

/** Orthonormal frame at a tangent point: [toward, east, north]. */
function basis(ra0, dec0) {
  const a = ra0 * DEG, d = dec0 * DEG;
  const ca = Math.cos(a), sa = Math.sin(a), cd = Math.cos(d), sd = Math.sin(d);
  return [
    [cd * ca, cd * sa, sd],
    [-sa, ca, 0],
    [-sd * ca, -sd * sa, cd],
  ];
}

/** Intermediate world coords (deg) -> [ra, dec] (deg), gnomonic.
 *  Vector form: no quadrant cases, and no denominator that degenerates as the
 *  tangent point approaches the pole -- which matters because SPHEREx's deep
 *  fields sit 23 degrees from the celestial pole. */
export function tanDeproject(xDeg, yDeg, ra0, dec0) {
  const [t, e, n] = basis(ra0, dec0);
  const xr = xDeg * DEG, yr = yDeg * DEG;
  const vx = t[0] + xr * e[0] + yr * n[0];
  const vy = t[1] + xr * e[1] + yr * n[1];
  const vz = t[2] + xr * e[2] + yr * n[2];
  const m = Math.hypot(vx, vy, vz);
  let ra = Math.atan2(vy, vx) / DEG;
  if (ra < 0) ra += 360;
  return [ra, Math.asin(Math.max(-1, Math.min(1, vz / m))) / DEG];
}

/** The right ascension and declination a north-up square stamp spans:
 *  `{ raLo, raHi, decLo, decHi }` in degrees, for a stamp centred on (ra0,
 *  dec0) reaching `halfDeg` from its centre to each edge.
 *
 *  Sampled along the edges rather than at the corners, because near a pole the
 *  widest right ascension is not at a corner, and a field holding the pole
 *  spans every right ascension (raLo 0, raHi 360). `raLo` may exceed `raHi`
 *  when the stamp straddles RA 0. */
export function fieldBounds(ra0, dec0, halfDeg) {
  const n = 16;
  let dLo = Infinity, dHi = -Infinity, aLo = Infinity, aHi = -Infinity;
  for (let k = 0; k <= n; k++) {
    const t = -halfDeg + (2 * halfDeg * k) / n;
    for (const [x, y] of [[t, -halfDeg], [t, halfDeg], [-halfDeg, t], [halfDeg, t]]) {
      const [ra, dec] = tanDeproject(x, y, ra0, dec0);
      dLo = Math.min(dLo, dec); dHi = Math.max(dHi, dec);
      const da = ((ra - ra0 + 540) % 360) - 180;
      aLo = Math.min(aLo, da); aHi = Math.max(aHi, da);
    }
  }
  const pole = Math.abs(dec0) + halfDeg * Math.SQRT2 >= 90
    && Math.abs(tanProject(0, Math.sign(dec0) * 90, ra0, dec0)[0]) <= halfDeg
    && Math.abs(tanProject(0, Math.sign(dec0) * 90, ra0, dec0)[1]) <= halfDeg;
  if (pole) return { raLo: 0, raHi: 360, decLo: dLo, decHi: dHi, pole: true };
  const wrap = (a) => ((a % 360) + 360) % 360;
  return { raLo: wrap(ra0 + aLo), raHi: wrap(ra0 + aHi), decLo: dLo, decHi: dHi, pole: false };
}

/** [ra, dec] (deg) -> intermediate world coords (deg). Throws beyond 90 deg. */
export function tanProject(ra, dec, ra0, dec0) {
  const [t, e, n] = basis(ra0, dec0);
  const a = ra * DEG, d = dec * DEG;
  const cd = Math.cos(d);
  const r = [cd * Math.cos(a), cd * Math.sin(a), Math.sin(d)];
  const w = r[0] * t[0] + r[1] * t[1] + r[2] * t[2];
  if (w <= 1e-12) throw new Error('direction is >=90 deg from the tangent point');
  const xe = r[0] * e[0] + r[1] * e[1] + r[2] * e[2];
  const yn = r[0] * n[0] + r[1] * n[1] + r[2] * n[2];
  return [(xe / w) / DEG, (yn / w) / DEG];
}

/** Great-circle separation in degrees. Haversine, not acos of a dot product:
 *  acos loses all its precision below an arcsecond, which is the entire regime
 *  this tool works in. */
export function angularSeparation(ra1, dec1, ra2, dec2) {
  const d1 = dec1 * DEG, d2 = dec2 * DEG;
  const dd = d2 - d1, da = (ra2 - ra1) * DEG;
  const h = Math.sin(dd / 2) ** 2 + Math.cos(d1) * Math.cos(d2) * Math.sin(da / 2) ** 2;
  return 2 * Math.asin(Math.min(1, Math.sqrt(h))) / DEG;
}

export class Wcs {
  /** From the flat JSON that scripts/bake_field.py writes. */
  constructor(spec) {
    this.crval1 = spec.crval[0];
    this.crval2 = spec.crval[1];
    this.crpix1 = spec.crpix[0];
    this.crpix2 = spec.crpix[1];
    this.cd = spec.cd;
    this.sip = new Sip(spec.sip);
    this.naxis1 = (spec.naxis && spec.naxis[0]) || 0;
    this.naxis2 = (spec.naxis && spec.naxis[1]) || 0;
    const [[a, b], [c, d]] = this.cd;
    this.det = a * d - b * c;
    this.scaleArcsec = Math.sqrt(Math.abs(this.det)) * 3600;
    this.pixelAreaSr = Math.pow(this.scaleArcsec * DEG / 3600, 2);
    this.positionAngleDeg = ((Math.atan2(b, d) / DEG) % 360 + 360) % 360;
    this.parityFlip = this.det > 0;
  }

  /** A plain TAN grid, north up and east left, centred on a position.
   *
   *  The frame a reprojected cutout lives in. SPHEREx rolls between visits --
   *  31 to 81 degrees between consecutive North Ecliptic Pole visits -- so
   *  cutouts taken as detector-aligned pixel boxes arrive at wildly different
   *  orientations, and the same star sits in a different corner every time.
   *  Every cutout is resampled onto a grid like this so a field's visits share
   *  one frame.
   *
   *  No SIP, and that is the point: the reprojection evaluates the source
   *  frame's distortion per output pixel, so what comes out is already
   *  undistorted. Mirrors `Wcs.north_up` in src/tracklet/frames/wcs.py. */
  static northUp(raDeg, decDeg, scaleArcsec, size) {
    const centre = (size + 1) / 2;
    const s = scaleArcsec / 3600;
    return new Wcs({
      crval: [raDeg, decDeg],
      crpix: [centre, centre],
      cd: [[-s, 0], [0, s]],
      naxis: [size, size],
    });
  }

  pixelToWorld(x, y) {
    const u = x - this.crpix1, v = y - this.crpix2;
    const [f, g] = this.sip.present ? this.sip.forward(u, v) : [u, v];
    const [[c11, c12], [c21, c22]] = this.cd;
    return tanDeproject(c11 * f + c12 * g, c21 * f + c22 * g, this.crval1, this.crval2);
  }

  worldToPixel(ra, dec) {
    const [xw, yw] = tanProject(ra, dec, this.crval1, this.crval2);
    const [[c11, c12], [c21, c22]] = this.cd;
    const det = c11 * c22 - c12 * c21;
    if (det === 0) throw new Error('singular CD matrix');
    const f = (xw * c22 - yw * c12) / det;
    const g = (yw * c11 - xw * c21) / det;
    const [u, v] = this.sip.present ? this.sip.inverse(f, g) : [f, g];
    return [u + this.crpix1, v + this.crpix2];
  }

  contains(x, y, margin = 0) {
    if (!this.naxis1 || !this.naxis2) return true;
    return x >= 0.5 + margin && x <= this.naxis1 + 0.5 - margin
        && y >= 0.5 + margin && y <= this.naxis2 + 0.5 - margin;
  }

  /** Worst pixel->world->pixel closure over the detector, in pixels. Shown in
   *  the UI so a viewer can see for themselves that the astrometry is not the
   *  limiting factor. */
  roundtripResidual(step = 509) {
    let worst = 0;
    const n1 = this.naxis1 || 2040, n2 = this.naxis2 || 2040;
    for (let x = 1; x <= n1; x += step) {
      for (let y = 1; y <= n2; y += step) {
        const [ra, dec] = this.pixelToWorld(x, y);
        const [xx, yy] = this.worldToPixel(ra, dec);
        worst = Math.max(worst, Math.hypot(xx - x, yy - y));
      }
    }
    return worst;
  }
}

// -- formatting ---------------------------------------------------------- //

// Sexagesimal, rounded once at the smallest digit shown and *then* decomposed.
//
// Decomposing first and rounding the seconds last is the obvious way round and
// it is wrong: `toFixed` carries into a field that has already been printed.
// RA 359.99999 came out as **23h59m60.00s** -- there is no sixtieth second --
// and any declination within ~0.00001 deg of an arcminute boundary came out as
// `...59'60.0"`, including positions a hair off the pole reading
// `+89°59'60.0"`. A sweep of 20,000 right ascensions hit it 33 times.
//
// Worth more than a cosmetic wobble: this is the coordinate a viewer copies out
// of the Cursor panel to look a position up elsewhere, and now the globe takes
// arbitrary positions rather than four baked ones, every boundary is reachable.
//
// Rounding to whole hundredths-of-a-second (RA) or tenths-of-an-arcsecond (Dec)
// first makes the carry arithmetic rather than string formatting: sixty seconds
// becomes a minute because 6,000 hundredths *is* a minute.

const CS_PER_HOUR = 3600 * 100;       // hundredths of a second of time
const CS_PER_MIN = 60 * 100;
const CS_PER_DAY = 24 * CS_PER_HOUR;

export function formatRa(deg) {
  const h = ((deg % 360) + 360) % 360 / 15;
  // Wraps at exactly 24h, so a value that rounds up through midnight returns
  // to 00h instead of printing an hour that does not exist.
  const cs = Math.round(h * CS_PER_HOUR) % CS_PER_DAY;
  const hh = Math.floor(cs / CS_PER_HOUR);
  const mm = Math.floor((cs % CS_PER_HOUR) / CS_PER_MIN);
  const ss = (cs % CS_PER_MIN) / 100;
  return `${String(hh).padStart(2, '0')}h${String(mm).padStart(2, '0')}m`
    + `${ss.toFixed(2).padStart(5, '0')}s`;
}

const DS_PER_DEG = 3600 * 10;         // tenths of an arcsecond
const DS_PER_ARCMIN = 60 * 10;

export function formatDec(deg) {
  const sign = deg < 0 ? '-' : '+';
  // No wrap: declination is bounded, and a value rounding up to the pole must
  // read +90°00'00.0" rather than +89°59'60.0".
  const ds = Math.round(Math.abs(deg) * DS_PER_DEG);
  const dd = Math.floor(ds / DS_PER_DEG);
  const mm = Math.floor((ds % DS_PER_DEG) / DS_PER_ARCMIN);
  const ss = (ds % DS_PER_ARCMIN) / 10;
  return `${sign}${String(dd).padStart(2, '0')}°${String(mm).padStart(2, '0')}'`
    + `${ss.toFixed(1).padStart(4, '0')}"`;
}
