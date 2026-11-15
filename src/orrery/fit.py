"""Stamp likelihood (docs/INTERFACES.md, "Stamp likelihood").

Linear model per pixel p (INTERFACES.md):
    mu_p = A*P_ast(p) + sum_k B_k*P_k(p) + c0 + c1*(x_p - xm) + c2*(y_p - ym)
fitted by weighted least squares with weights w_p = 1/variance_p, i.e. the
maximum-likelihood solution for independent Gaussian pixel noise
(e.g. Press et al., Numerical Recipes, 3rd ed., sec. 15.4).
"""
from dataclasses import dataclass

import numpy as np

from orrery.psf import psf_model

ARCSEC_PER_RAD = 206264.806


@dataclass
class FitResult:
    A: float              # moving-source amplitude (MJy/sr summed over pixels)
    A_err: float          # sqrt(Cov[A, A])
    coef: np.ndarray      # [A, B_1..B_K, c0, c1, c2]; NaN for dropped columns
    cov: np.ndarray       # covariance of coef; NaN rows/cols for dropped columns
    names: list           # column labels matching coef
    chi2: float
    dof: int
    model: np.ndarray     # [H, W]
    resid: np.ndarray     # [H, W] image - model
    mask: np.ndarray      # [H, W] bool, True = pixel used in the fit
    chi2_local: float     # chi2 over used pixels within 2 px of (xs, ys)
    dof_local: float      # effective dof there: n_local - sum of leverages


def flux_mJy(A, omega_arcsec2):
    """F = A * Omega; Omega[sr] = Omega[arcsec^2] / 206264.806^2; 1 MJy = 1e9 mJy (INTERFACES.md)."""
    return A * omega_arcsec2 / ARCSEC_PER_RAD**2 * 1e9


def fit_stamp(image, variance, flags, epsf, xs, ys, star_xy, bad_mask_bits, oversample=5):
    """Weighted linear fit of moving source + static stars + background plane.

    All positions are 0-based stamp coordinates (x = column, y = row).
    """
    image = np.asarray(image, float)
    variance = np.asarray(variance, float)
    H, W = image.shape
    yy, xx = np.indices((H, W))
    xm, ym = (W - 1) / 2.0, (H - 1) / 2.0  # plane pivot = stamp centre

    # Pixel selection (INTERFACES.md); non-finite image values cannot enter a fit either.
    mask = ((np.asarray(flags) & bad_mask_bits) == 0) & np.isfinite(variance) & (variance > 0) & np.isfinite(image)

    star_xy = np.asarray(star_xy, float).reshape(-1, 2)
    cols = [psf_model(epsf, xs, ys, (H, W), oversample, box=3)]
    cols += [psf_model(epsf, sx, sy, (H, W), oversample, box=3) for sx, sy in star_xy]
    cols += [np.ones((H, W)), xx - xm, yy - ym]
    names = ["A"] + [f"B{k}" for k in range(len(star_xy))] + ["c0", "c1", "c2"]
    X_full = np.stack([c.ravel() for c in cols], axis=1)

    # Drop columns with no support on the used pixels (e.g. star fully masked/off-stamp).
    m = mask.ravel()
    keep = np.any(X_full[m] != 0, axis=0)
    X = X_full[m][:, keep]

    # Whitened system: sqrt(w) X beta = sqrt(w) d, w = 1/var. Its least-squares solution solves
    # the normal equations (X^T W X) beta = X^T W d, with Cov(beta) = (X^T W X)^-1.
    # Solved via SVD sqrt(w) X = U S V^T for numerical robustness (Numerical Recipes sec. 15.4.2):
    # beta = V S^-1 U^T (sqrt(w) d),  Cov = V S^-2 V^T. Singular values below rcond are discarded
    # (pseudo-inverse) so an exactly degenerate pair of columns does not blow up.
    sw = 1.0 / np.sqrt(variance.ravel()[m])
    U, S, Vt = np.linalg.svd(X * sw[:, None], full_matrices=False)
    good = S > S.max() * 1e-12
    U, S, Vt = U[:, good], S[good], Vt[good]
    beta = Vt.T @ ((U.T @ (image.ravel()[m] * sw)) / S)
    cov_k = (Vt.T / S**2) @ Vt

    n_par = X_full.shape[1]
    coef = np.full(n_par, np.nan)
    coef[keep] = beta
    cov = np.full((n_par, n_par), np.nan)
    cov[np.ix_(keep, keep)] = cov_k

    model = (X_full[:, keep] @ beta).reshape(H, W)
    resid = image - model
    chi2_map = np.where(mask, resid**2 / np.where(mask, variance, 1.0), 0.0)  # chi2 = sum w r^2
    chi2 = float(chi2_map.sum())
    dof = int(m.sum() - good.sum())

    # Local chi2 within 2 px (Euclidean) of the source. For WLS, E[w_i r_i^2] = 1 - h_ii with
    # h_ii the leverage (diagonal of the hat matrix U U^T), so the effective local dof is
    # n_local - sum h_ii (e.g. Hoaglin & Welsch 1978, Am. Stat. 32, 17).
    lev = np.zeros(H * W)
    lev[m] = np.sum(U**2, axis=1)
    near = mask & ((xx - xs) ** 2 + (yy - ys) ** 2 <= 2.0**2)
    chi2_local = float(chi2_map[near].sum())
    dof_local = float(near.sum() - lev.reshape(H, W)[near].sum())

    return FitResult(A=coef[0], A_err=float(np.sqrt(cov[0, 0])), coef=coef, cov=cov, names=names,
                     chi2=chi2, dof=dof, model=model, resid=resid, mask=mask,
                     chi2_local=chi2_local, dof_local=dof_local)


def fit_stamp_obj(stamp, xs, ys, star_xy, bad_mask_bits):
    """fit_stamp on an orrery.data.Stamp-like object."""
    return fit_stamp(stamp.image, stamp.variance, stamp.flags, stamp.epsf, xs, ys, star_xy,
                     bad_mask_bits, oversample=stamp.epsf_oversample)


def refine_position(image, variance, flags, epsf, xs, ys, star_xy, bad_mask_bits, oversample=5,
                    half=1.5, step=0.1):
    """Grid-search the source position over offsets in [-half, half] (step `step`), INTERFACES.md.

    Returns (xs_best, ys_best, chi2_surface, offsets); chi2_surface[j, i] is the fit chi2
    at (xs + offsets[i], ys + offsets[j]).
    """
    n = int(round(2 * half / step)) + 1
    offsets = np.linspace(-half, half, n)
    surf = np.array([[fit_stamp(image, variance, flags, epsf, xs + ox, ys + oy, star_xy,
                                bad_mask_bits, oversample).chi2 for ox in offsets] for oy in offsets])
    j, i = np.unravel_index(np.argmin(surf), surf.shape)
    return xs + offsets[i], ys + offsets[j], surf, offsets


def refine_position_obj(stamp, xs, ys, star_xy, bad_mask_bits, **kw):
    """refine_position on an orrery.data.Stamp-like object."""
    return refine_position(stamp.image, stamp.variance, stamp.flags, stamp.epsf, xs, ys, star_xy,
                           bad_mask_bits, oversample=stamp.epsf_oversample, **kw)
