"""ePSF evaluation (docs/INTERFACES.md, "ePSF evaluation").

The ePSF (effective PSF; Anderson & King 2000, PASP 112, 1360) is the
instrumental PSF already convolved with the pixel response, tabulated on a grid
oversampled by `oversample`. Its value at offset (dx, dy) *is* the fraction of
the source flux landing in a native pixel whose centre sits at that offset, so
no pixel integration is needed -- only interpolation.
"""
import numpy as np
from scipy.ndimage import map_coordinates


def psf_native(epsf, dx, dy, oversample=5):
    """ePSF value of a native pixel at offset (dx, dy) = pixel centre - source.

    Grid index (iy, ix) <-> offset ((ix - c)/oversample, (iy - c)/oversample),
    c = (N - 1)/2 (INTERFACES.md). Bicubic spline interpolation
    (map_coordinates order=3), zero outside the tabulated grid.
    """
    epsf = np.asarray(epsf, dtype=float)
    dx, dy = np.broadcast_arrays(np.asarray(dx, float), np.asarray(dy, float))
    c = (epsf.shape[0] - 1) / 2.0
    iy = c + oversample * dy  # row index on the oversampled grid
    ix = c + oversample * dx  # column index on the oversampled grid
    val = map_coordinates(epsf, [iy.ravel(), ix.ravel()], order=3, mode="constant", cval=0.0)
    outside = (iy < 0) | (iy > epsf.shape[0] - 1) | (ix < 0) | (ix > epsf.shape[1] - 1)
    return np.where(outside.ravel(), 0.0, val).reshape(dx.shape)


def psf_model(epsf, xs, ys, shape, oversample=5, box=3):
    """Normalised [H, W] PSF image for a source at stamp coords (xs, ys).

    Nonzero only on native pixels with |dx|, |dy| <= box; rescaled so the sum
    over *all* such pixels (including any that fall off the stamp) is exactly 1
    (INTERFACES.md normalisation). A source near the edge therefore has an
    on-stamp sum < 1, keeping A = "flux inside the ePSF box" position-independent.
    """
    # Every native pixel centre inside the box, independent of stamp extent.
    bx = np.arange(np.ceil(xs - box), np.floor(xs + box) + 1)
    by = np.arange(np.ceil(ys - box), np.floor(ys + box) + 1)
    gx, gy = np.meshgrid(bx, by)
    vals = psf_native(epsf, gx - xs, gy - ys, oversample)
    vals = vals / vals.sum()  # sum over the box = 1

    out = np.zeros(shape)
    H, W = shape
    on = (gx >= 0) & (gx < W) & (gy >= 0) & (gy < H)
    out[gy[on].astype(int), gx[on].astype(int)] = vals[on]
    return out
