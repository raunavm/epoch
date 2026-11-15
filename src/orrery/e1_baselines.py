"""E1 baseline M5 (docs/PROTOCOLS.md): wavelength-neighbour median reference.

Idea: the static sky at pixel p in frame e is the median, over the K other frames of the SAME detector whose
central wavelength at p is closest to frame e's, of their background-subtracted values; no model is fitted.
It is wavelength-local (like M4) but robust to several contaminated frames (like M3) and leave-one-out.

Inputs (same arrays orrery.change.change_model receives):
- frames_resampled: [N, S, S] float, frames on the common north-up 6.15" grid, indexed [y, x]; NaN = no data
  or masked. Raw resampled frames (background included) or already background-subtracted frames are both fine:
  each frame's 3-sigma clipped median (orrery.change.clipped_median) is subtracted here, which is a no-op on an
  already-subtracted frame (the clipped median is translation-invariant).
- wl: [N, S, S] per-pixel central wavelength (um).
- det: [N] detector number 1-6. mjd: [N] MJD (accepted for interface parity; not used).
- sigma_e: [N] per-frame noise (1.4826 * MAD after background subtraction, i.e. change_model's "sigma").
- f: frozen model-error floor (0.1512); per-pixel sigma^2 = sigma_e^2 + (f |S|)^2 with S = the reference value.
Output: list of N maps [S, S], (v_e - ref_e) / sigma; NaN where the frame is NaN or < MIN_REF valid neighbours.
"""
import numpy as np

from orrery.change import clipped_median

M5_NAME = "wavelength-neighbour median (K=15, same detector, leave-one-out)"
K_NEIGHBOURS = 15   # frozen a priori: smallest K whose median noise inflates the map noise by <= 5%
MIN_REF = 3         # fewer valid neighbours at a pixel -> NaN


def m5_map(frames_resampled, wl, det, mjd, sigma_e, f, k=K_NEIGHBOURS):
    v = np.asarray(frames_resampled, float)
    lam = np.asarray(wl, float)
    det, sigma_e = np.asarray(det), np.asarray(sigma_e, float)
    N, S, _ = v.shape
    vs = (v - np.array([clipped_median(x) for x in v])[:, None, None]).reshape(N, -1)
    lam = lam.reshape(N, -1)
    out = np.full((N, S * S), np.nan)
    for d in np.unique(det):
        E = np.flatnonzero(det == d)
        V, L = vs[E], lam[E]
        bad = ~np.isfinite(V) | ~np.isfinite(L)
        Vn = np.where(bad, np.nan, V)
        kk = min(k, len(E) - 1)
        if kk < MIN_REF:
            continue
        for i, e in enumerate(E):
            dist = np.abs(L - L[i])
            dist[bad] = np.inf
            dist[i] = np.inf                                   # leave the frame itself out
            idx = np.argpartition(dist, kk - 1, axis=0)[:kk]   # K nearest in wavelength, per pixel
            nb = np.take_along_axis(Vn, idx, axis=0)
            ok = (np.isfinite(nb).sum(0) >= MIN_REF) & ~bad[i]
            ref = np.nanmedian(np.where(ok, nb, 0.0), axis=0)
            sig = np.sqrt(sigma_e[e] ** 2 + (f * ref) ** 2)
            out[e] = np.where(ok, (V[i] - ref) / sig, np.nan)
    return list(out.reshape(N, S, S))
