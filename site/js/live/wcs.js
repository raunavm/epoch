// TAN-SIP world<->pixel for SPHEREx Level-2 IMAGE headers. Pixels are 0-based (x = column, y = row).
const D2R = Math.PI / 180;

function sipCoeffs(h, pre) {
  const order = h[`${pre}_ORDER`];
  if (!order) return null;
  const terms = [];
  for (let p = 0; p <= order; p++) for (let q = 0; q <= order - p; q++) {
    const v = h[`${pre}_${p}_${q}`];
    if (v) terms.push([p, q, v]);
  }
  return terms;
}
const poly = (t, u, v) => { let s = 0; if (t) for (const [p, q, c] of t) s += c * u ** p * v ** q; return s; };
const dpoly = (t, u, v) => { // [d/du, d/dv]
  let du = 0, dv = 0;
  if (t) for (const [p, q, c] of t) { if (p) du += c * p * u ** (p - 1) * v ** q; if (q) dv += c * q * u ** p * v ** (q - 1); }
  return [du, dv];
};

// Gnomonic projection about (ra0, dec0): returns standard coordinates (xi, eta) in degrees, or null behind the plane.
export function project(ra, dec, ra0, dec0) {
  const a = (ra - ra0) * D2R, d = dec * D2R, d0 = dec0 * D2R;
  const cc = Math.sin(d0) * Math.sin(d) + Math.cos(d0) * Math.cos(d) * Math.cos(a);
  if (cc <= 0) return null;
  return [Math.cos(d) * Math.sin(a) / cc / D2R, (Math.cos(d0) * Math.sin(d) - Math.sin(d0) * Math.cos(d) * Math.cos(a)) / cc / D2R];
}
export function deproject(xi, eta, ra0, dec0) {
  const x = xi * D2R, y = eta * D2R, d0 = dec0 * D2R, rho = Math.hypot(x, y);
  if (rho === 0) return [ra0, dec0];
  const c = Math.atan(rho);
  const dec = Math.asin(Math.cos(c) * Math.sin(d0) + y * Math.sin(c) * Math.cos(d0) / rho);
  const ra = ra0 + Math.atan2(x * Math.sin(c), rho * Math.cos(d0) * Math.cos(c) - y * Math.sin(d0) * Math.sin(c)) / D2R;
  return [((ra % 360) + 360) % 360, dec / D2R];
}

export function makeWCS(h) {
  const cd = [h.PC1_1 * (h.CDELT1 ?? 1), h.PC1_2 * (h.CDELT1 ?? 1), h.PC2_1 * (h.CDELT2 ?? 1), h.PC2_2 * (h.CDELT2 ?? 1)];
  const det = cd[0] * cd[3] - cd[1] * cd[2];
  const inv = [cd[3] / det, -cd[1] / det, -cd[2] / det, cd[0] / det];
  const A = sipCoeffs(h, 'A'), B = sipCoeffs(h, 'B'), AP = sipCoeffs(h, 'AP'), BP = sipCoeffs(h, 'BP');
  const ra0 = h.CRVAL1, dec0 = h.CRVAL2, cx = h.CRPIX1, cy = h.CRPIX2;
  return {
    pix2world(x, y) {
      const u = x + 1 - cx, v = y + 1 - cy;
      const U = u + poly(A, u, v), V = v + poly(B, u, v);
      return deproject(cd[0] * U + cd[1] * V, cd[2] * U + cd[3] * V, ra0, dec0);
    },
    world2pix(ra, dec) {
      const p = project(ra, dec, ra0, dec0);
      if (!p) return null;
      const U = inv[0] * p[0] + inv[1] * p[1], V = inv[2] * p[0] + inv[3] * p[1];
      let u = U + poly(AP, U, V), v = V + poly(BP, U, V);
      for (let it = 0; it < 3; it++) { // Newton on the forward SIP
        const fu = u + poly(A, u, v) - U, fv = v + poly(B, u, v) - V;
        const [a1, a2] = dpoly(A, u, v), [b1, b2] = dpoly(B, u, v);
        const j11 = 1 + a1, j12 = a2, j21 = b1, j22 = 1 + b2, dj = j11 * j22 - j12 * j21;
        u -= (j22 * fu - j12 * fv) / dj; v -= (-j21 * fu + j11 * fv) / dj;
      }
      return [u + cx - 1, v + cy - 1];
    },
  };
}

// Bilinear lookup in a WCS-WAVE node table {x[], y[] (1-based FITS), values[iy][ix] = [cwave, cband]} at 0-based pixel.
export function waveAt(tab, x, y) {
  const seg = (nodes, v) => {
    let i = 0; while (i < nodes.length - 2 && v > nodes[i + 1]) i++;
    const f = Math.min(1, Math.max(0, (v - nodes[i]) / (nodes[i + 1] - nodes[i])));
    return [i, f];
  };
  const [ix, fx] = seg(tab.x, x + 1), [iy, fy] = seg(tab.y, y + 1), V = tab.values;
  const out = [0, 0];
  for (let k = 0; k < 2; k++) {
    out[k] = (1 - fy) * ((1 - fx) * V[iy][ix][k] + fx * V[iy][ix + 1][k]) + fy * ((1 - fx) * V[iy + 1][ix][k] + fx * V[iy + 1][ix + 1][k]);
  }
  return out;
}
