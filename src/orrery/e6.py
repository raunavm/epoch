"""E6 / method M6 — native-pixel scene reference (docs/PROTOCOLS.md, E6; frozen 2026-11-15 10:10 PST).

Per frame, on the NATIVE cutout: model = sum_k A_k ePSF(x_k) over Gaia DR3 G < 19 stars (proper motion to the frame
epoch, zone ePSF nearest to each star, same rule as e1.epsf_zone) + background plane, one free amplitude per star,
weighted least squares exactly as orrery.fit.fit_stamp (pipeline variance weights, hard-bad FLAGS masked, columns
without support dropped, SVD pseudo-inverse with rcond 1e-12, plane pivot at the cutout centre) but with no
moving-source column. Several images sharing one design matrix (the unplanted cutout and its planted copies) are
solved in one call; the result is identical to separate fits (tests/test_e6.py).

Change map in native pixels: z = resid / sqrt(sigma_e^2 + (f6 |S|)^2), sigma_e = e1.native_sigma (Common noise
model), S = the static model = the star part sum_k A_k P_k (the plane plays the role of rev 2's background
subtraction). NaN where the pixel was not used in the fit. Then resampled with e1.resample (the harness resampler,
which also NaNs hard-bad + OUTLIER neighbours) and despiked (D1) like every other method.
"""
import numpy as np

from orrery.data import BAD_BITS
from orrery.e1 import GAIA_EPOCH, despike, epsf_zone, resample
from orrery.psf import psf_model

GMAX = 19.0
BOX = 3          # psf_model box, as in fit_stamp


def star_radec(gaia, mjd, gmax=GMAX):
    """(ra, dec, G) of Gaia G < gmax stars propagated with proper motion to mjd (same as e1.gaia_xy)."""
    g = gaia[gaia["phot_g_mean_mag"] < gmax]
    yr = (mjd - GAIA_EPOCH) / 365.25
    dec = g["dec"].values + np.nan_to_num(g["pmdec"].values) * yr / 3.6e6
    ra = g["ra"].values + np.nan_to_num(g["pmra"].values) * yr / 3.6e6 / np.cos(np.radians(g["dec"].values))
    return ra, dec, g["phot_g_mean_mag"].values


def star_xy_native(gaia, mjd, wcs, x0, y0, gmax=GMAX):
    """Cutout coords [n, 2] and G of stars at mjd."""
    ra, dec, G = star_radec(gaia, mjd, gmax)
    if len(ra) == 0:
        return np.zeros((0, 2)), G
    x, y = wcs.all_world2pix(ra, dec, 0)
    return np.stack([x - x0, y - y0], -1), G


def star_columns(star_xy, det, x0, y0, shape):
    """psf_model image per star with the zone ePSF nearest to the star's full-frame position."""
    cols = []
    for sx, sy in star_xy:
        epsf, ovs = epsf_zone(det, sx + x0, sy + y0)
        cols.append(psf_model(epsf, sx, sy, shape, oversample=ovs, box=BOX))
    return cols


def scene_fit(images, variance, flags, cols, bad_mask_bits=BAD_BITS):
    """fit_stamp's WLS for stars-only + plane, several images [n, H, W] at once.
    Returns resid [n, H, W] (image - model), stars [n, H, W] (sum_k A_k P_k), mask [H, W] (pixel used)."""
    images = np.asarray(images, float)
    variance = np.asarray(variance, float)
    n, H, W = images.shape
    yy, xx = np.indices((H, W))
    xm, ym = (W - 1) / 2.0, (H - 1) / 2.0
    fin = np.isfinite(images).all(0)
    assert (np.isfinite(images).any(0) == fin).all(), "images must share their NaN pattern"
    mask = ((np.asarray(flags) & bad_mask_bits) == 0) & np.isfinite(variance) & (variance > 0) & fin
    ns = len(cols)
    X_full = np.stack([c.ravel() for c in list(cols) + [np.ones((H, W)), xx - xm, yy - ym]], axis=1)
    m = mask.ravel()
    keep = np.any(X_full[m] != 0, axis=0)
    X = X_full[m][:, keep]
    sw = 1.0 / np.sqrt(variance.ravel()[m])
    U, S, Vt = np.linalg.svd(X * sw[:, None], full_matrices=False)
    good = S > S.max() * 1e-12
    U, S, Vt = U[:, good], S[good], Vt[good]
    D = images.reshape(n, -1)[:, m].T * sw[:, None]                 # [npix, n] whitened data
    beta = Vt.T @ ((U.T @ D) / S[:, None])                          # [nkeep, n]
    Xk = X_full[:, keep]
    model = (Xk @ beta).T.reshape(n, H, W)
    is_star = keep[:ns]
    stars = (Xk[:, :is_star.sum()] @ beta[:is_star.sum()]).T.reshape(n, H, W)
    return images - model, stars, mask


def native_z(resid, stars, mask, sigma_e, f6):
    """Residual / sigma in native pixels, sigma^2 = sigma_e^2 + (f6 |S|)^2; NaN where not fitted."""
    return np.where(mask, resid / np.sqrt(sigma_e ** 2 + (f6 * stars) ** 2), np.nan)


def m6_grid(z, fr):
    """Native z map -> 48 px grid with the harness resampler (e1.resample), then the D1 despike."""
    return despike(resample(z, fr["flg"], fr["gx"], fr["gy"])[0])
