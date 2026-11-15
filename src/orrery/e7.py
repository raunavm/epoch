"""E7 — per-star change test with an empirical null (docs/PROTOCOLS.md E7, frozen 2026-11-15 11:05 PST).

Photometry: per picture, joint forced ePSF fit of every Gaia G < 19 star (+ target) and a background plane on the
native cutout (the E6 scene: orrery.e6.star_columns, fit_stamp's WLS), with hard-bad, OVERFLOW and NONLINEAR pixels
masked. A reported source's amplitude is the sum over its blend group (Gaia sources within BLEND_ARCSEC, which an
undersampled 6.15" pixel cannot separate); its error is sqrt(1' Cov 1) of the group.

Models per star: static = per detector c0 + c1 (l - lbar_d) + c2 (l - lbar_d)^2 (WLS); changing = static x (1 + a_w),
one a_w per visit window (split at gaps > 30 d), first window a = 0. T = chi2_static - chi2_changing.
"""
import numpy as np
from scipy.optimize import least_squares

from orrery.data import BAD_BITS

PHOT_BITS = BAD_BITS | (1 << 1) | (1 << 15)      # hard-bad + OVERFLOW + NONLINEAR (15 is already hard-bad)
SAT_BITS = (1 << 1) | (1 << 15)
BLEND_ARCSEC = 3.0
GAP_DAYS = 30.0
MIN_USABLE_PIX = 4                                # unmasked pixels with P > 0 under a group's ePSF box


def forced_photometry(image, variance, flags, cols, groups, bits=PHOT_BITS):
    """WLS of image on cols + plane (as orrery.fit.fit_stamp / orrery.e6.scene_fit). groups: list of index lists
    into cols. Returns per group (amp, err, n_usable_pix); amp = err = NaN if any member column has no support."""
    image = np.asarray(image, float)
    H, W = image.shape
    yy, xx = np.indices((H, W))
    mask = ((np.asarray(flags) & bits) == 0) & np.isfinite(variance) & (variance > 0) & np.isfinite(image)
    ns = len(cols)
    X_full = np.stack([c.ravel() for c in list(cols) + [np.ones((H, W)), xx - (W - 1) / 2, yy - (H - 1) / 2]], 1)
    m = mask.ravel()
    keep = np.any(X_full[m] != 0, axis=0)
    X = X_full[m][:, keep]
    sw = 1.0 / np.sqrt(np.asarray(variance, float).ravel()[m])
    U, S, Vt = np.linalg.svd(X * sw[:, None], full_matrices=False)
    good = S > S.max() * 1e-12
    U, S, Vt = U[:, good], S[good], Vt[good]
    beta = Vt.T @ ((U.T @ (image.ravel()[m] * sw)) / S)
    pos = np.full(ns + 3, -1)
    pos[np.flatnonzero(keep)] = np.arange(keep.sum())
    out = []
    for g in groups:
        p = pos[list(g)]
        npix = int(sum(((X_full[:, k] != 0) & m).sum() for k in g))
        if (p < 0).any():
            out.append((np.nan, np.nan, npix))
            continue
        u = (Vt[:, p].sum(1)) / S                          # Cov = V S^-2 V^T -> 1'Cov1 = |S^-1 V^T 1|^2
        out.append((float(beta[p].sum()), float(np.sqrt(u @ u)), npix))
    return out


def windows(mjd, gap=GAP_DAYS):
    """Window index per MJD (sorted order not assumed): a new window starts after a gap > `gap` days."""
    mjd = np.asarray(mjd, float)
    o = np.argsort(mjd)
    w = np.zeros(len(mjd), int)
    w[o] = np.concatenate([[0], np.cumsum(np.diff(mjd[o]) > gap)])
    return w


def _design(lam, det):
    """Per-detector quadratic design [n, 3 * n_det] and the detector list; lbar_d = mean lambda of detector d."""
    dets = np.unique(det)
    A = np.zeros((len(lam), 3 * len(dets)))
    for j, d in enumerate(dets):
        s = det == d
        x = lam[s] - lam[s].mean()
        A[s, 3 * j:3 * j + 3] = np.stack([np.ones(s.sum()), x, x * x], 1)
    return A, dets


def static_fit(flux, err, lam, det):
    """WLS static model. Returns model, chi2 per detector {d: (chi2, n)}, total chi2, n_par."""
    A, dets = _design(lam, det)
    w = 1.0 / err
    c, *_ = np.linalg.lstsq(A * w[:, None], flux * w, rcond=None)
    model = A @ c
    r2 = ((flux - model) / err) ** 2
    per = {int(d): (float(r2[det == d].sum()), int((det == d).sum())) for d in dets}
    npar = int(sum(min(3, n) for _, n in per.values()))
    return model, per, float(r2.sum()), npar


def change_fit(flux, err, lam, det, win):
    """static x (1 + a_w), a_w free for every window with data except the first (a = 0). Returns dict with chi2,
    a [n_win] (first = 0), a_err (first = 0; inf if undetermined), model, windows present."""
    A, dets = _design(lam, det)
    ws = np.unique(win)
    widx = np.searchsorted(ws, win)
    nc = A.shape[1]
    c0, *_ = np.linalg.lstsq(A / err[:, None], flux / err, rcond=None)

    def resid(p):
        a = np.concatenate([[0.0], p[nc:]])
        return (flux - (A @ p[:nc]) * (1 + a[widx])) / err

    p0 = np.concatenate([c0, np.zeros(len(ws) - 1)])
    r = least_squares(resid, p0, method="lm", xtol=1e-12, ftol=1e-12, max_nfev=20000) if len(ws) > 1 else None
    p = r.x if r is not None else p0
    chi2 = float((resid(p) ** 2).sum())
    a = np.concatenate([[0.0], p[nc:]])
    aerr = np.zeros(len(ws))
    if r is not None:
        J = r.jac
        cov = np.linalg.pinv(J.T @ J)
        d = np.diag(cov)[nc:]
        aerr[1:] = np.where(d > 0, np.sqrt(np.abs(d)), np.inf)
    model = (A @ p[:nc]) * (1 + a[widx])
    return dict(chi2=chi2, a=a, a_err=aerr, windows=ws, model=model, coef=p[:nc], dets=dets)
