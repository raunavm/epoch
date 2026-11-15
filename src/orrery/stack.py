"""Stacking along a known orbit (finding 2) and planted-object tests (finding 4).

Each frame's static scene (Gaia stars + background plane, fitted jointly with the mover by
orrery.fit) is subtracted, the residual is shifted so the predicted position sits at the stamp
centre, and frames are combined with weights w_e = p_e / σ_e² where p_e is the expected flux of the
object in that frame (shape known up to one scale; brief §10 'Stack', Kaiser-style matched filter).
The combined image is in units of "fraction of the expected brightness" (1 = as predicted).
The same weights give the flux-domain estimate  â = Σ p A/σ² / Σ p²/σ²,  σ_â = (Σ p²/σ²)^-1/2.
Stacking along known orbits is not new (NEOWISE: Masiero et al.; IRSA MOST tutorial); it is used
here as a measured demonstration and a depth calibration, not claimed as a method contribution.
"""
import numpy as np
from scipy.ndimage import shift as nd_shift

from .data import BAD_BITS
from .fit import fit_stamp, flux_mJy
from .psf import psf_model


def scene_residual(st, xs, ys, star_xy):
    r = fit_stamp(st.image, st.variance, st.flags, st.epsf, xs, ys, star_xy, BAD_BITS, st.epsf_oversample)
    pa = psf_model(st.epsf, xs, ys, st.image.shape, st.epsf_oversample)
    static = r.model - r.A * pa
    res = st.image - static
    res[~r.mask] = 0.0
    w = np.where(r.mask, 1.0 / st.variance, 0.0)
    return res, w, r


def centre(img, xs, ys, half):
    """Shift so (xs, ys) lands on (half, half) (cubic spline, zero outside)."""
    return nd_shift(img, (half - ys, half - xs), order=3, mode="constant", cval=0.0)


def combine(items, half):
    """items: list of dict(res, w, xs, ys, p, A, A_err) with p and A in the same amplitude units.
    Returns stack image (fraction of expected), its per-pixel S/N map, and flux-domain â ± σ."""
    num = np.zeros((2 * half + 1,) * 2)
    den = np.zeros_like(num)
    sa = sb = 0.0
    for it in items:
        r = centre(it["res"], it["xs"], it["ys"], half)
        w = centre(it["w"], it["xs"], it["ys"], half).clip(0)
        p = it["p"]
        num += w * p * r
        den += w * p * p
        sa += p * it["A"] / it["A_err"] ** 2
        sb += p * p / it["A_err"] ** 2
    img = np.where(den > 0, num / np.where(den > 0, den, 1), 0.0)
    snr_map = np.where(den > 0, num / np.sqrt(np.where(den > 0, den, 1)), 0.0)
    a_hat = sa / sb if sb > 0 else np.nan
    return img, snr_map, a_hat, 1 / np.sqrt(sb) if sb > 0 else np.nan


def inject(st, xs, ys, amp):
    """Plant a point source of amplitude amp (MJy/sr summed) at (xs, ys) using the frame's own ePSF,
    with Poisson-like extra noise neglected (faint planted sources are background-limited)."""
    out = st.image.copy()
    out += amp * psf_model(st.epsf, xs, ys, st.image.shape, st.epsf_oversample)
    return out


def amp_from_mJy(f_mJy, omega_arcsec2):
    return f_mJy / (omega_arcsec2 / 206264.806 ** 2 * 1e9)
