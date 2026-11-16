// Orthographic projection of the celestial sphere (ported from Tracklet). DOM-free.

const DEG = Math.PI / 180;

/** Unit vector for a sky direction. */
export function unitFromRaDec(raDeg, decDeg) {
  const a = raDeg * DEG, d = decDeg * DEG, c = Math.cos(d);
  return [c * Math.cos(a), c * Math.sin(a), Math.sin(d)];
}

/** [raDeg, decDeg] for a unit vector. */
export function raDecFromUnit(v) {
  return [
    ((Math.atan2(v[1], v[0]) / DEG) % 360 + 360) % 360,
    Math.asin(Math.max(-1, Math.min(1, v[2]))) / DEG,
  ];
}

export const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

/** The camera frame for a viewpoint: toward the viewer, east, north.
 *
 *  `right` points along increasing RA, so the globe is seen from *outside* the
 *  celestial sphere and east is to the right. That is the opposite handedness
 *  from a finder chart, which shows the sky as seen from inside it. The choice
 *  is deliberate: a globe you drag has to turn the way an object turns, and
 *  inverting it to match chart convention makes the drag feel broken. The
 *  caption says so, and the two poles are labelled by name so nothing depends
 *  on the reader inferring which way round it is. */
export function cameraBasis(lon0Deg, lat0Deg) {
  const cl = Math.cos(lon0Deg * DEG), sl = Math.sin(lon0Deg * DEG);
  const cb = Math.cos(lat0Deg * DEG), sb = Math.sin(lat0Deg * DEG);
  return {
    fwd: [cb * cl, cb * sl, sb],
    right: [-sl, cl, 0],
    up: [-sb * cl, -sb * sl, cb],
  };
}

/** Sky -> unit disc. `depth > 0` is the near hemisphere. */
export function projectUnit(cam, v) {
  return { u: dot(v, cam.right), w: dot(v, cam.up), depth: dot(v, cam.fwd) };
}

/** Unit disc -> sky, exact and non-iterative. Null outside the disc.
 *
 *  Only the near hemisphere has an inverse -- every disc point corresponds to
 *  two sky directions and this returns the one facing the viewer, which is the
 *  one that was clicked. */
export function unprojectUnit(cam, u, w) {
  const q = u * u + w * w;
  if (q > 1) return null;
  const k = Math.sqrt(1 - q);
  return raDecFromUnit([
    u * cam.right[0] + w * cam.up[0] + k * cam.fwd[0],
    u * cam.right[1] + w * cam.up[1] + k * cam.fwd[1],
    u * cam.right[2] + w * cam.up[2] + k * cam.fwd[2],
  ]);
}
