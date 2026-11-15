"""Reflected-light and thermal (NEATM) flux models for asteroids.

Units: wavelength in micron, flux density F_nu in mJy, distances in au, angles in degrees.

Constants and sources
---------------------
* Solar spectrum: CALSPEC ``sun_mod_001.fits`` (Kurucz special solar model, Teff/logg = 5777/4.4,
  R = 5000, normalised to observed solar fluxes at 680-770 nm; flux at 1 au), STScI CALSPEC archive
  https://archive.stsci.edu/hlsps/reference-atlases/cdbs/current_calspec/sun_mod_001.fits ;
  CALSPEC described in Bohlin, Gordon & Tremblay 2014, PASP 126, 711. Chosen over
  ``sun_reference_stis_002`` because the latter stops at 2.7 um and SPHEREx reaches 5 um.
  Its integral over 0.15-300 um is 1367.0 W m^-2 (0.45% above the IAU nominal TSI below).
  The spectrum is smoothed to resolving power SOLAR_R (Gaussian FWHM = lambda/R) so that point
  evaluations are not dominated by individual Fraunhofer lines.
* Total solar irradiance at 1 au: S_SUN = 1361 W m^-2 (IAU 2015 Resolution B3 nominal value;
  Kopp & Lean 2011, GRL 38, L01706).
* V-band zero point: f_nu(V=0) = 3.636e-20 erg s^-1 cm^-2 Hz^-1 = 3636 Jy
  (Bessell, Castelli & Plez 1998, A&A 333, 231, Table A2).
* Physical constants: astropy.constants (CODATA 2018), au = IAU 2012 Resolution B2.
"""

from functools import lru_cache
from pathlib import Path

import numpy as np
from astropy import constants as const
from astropy import units as u
from astropy.io import fits
from scipy.ndimage import gaussian_filter1d

SOLAR_FILE = Path(__file__).resolve().parents[2] / "data" / "ref" / "sun_mod_001.fits"
SOLAR_URL = "https://archive.stsci.edu/hlsps/reference-atlases/cdbs/current_calspec/sun_mod_001.fits"
SOLAR_R = 200.0  # smoothing resolving power for the solar spectrum (SPHEREx R ~ 35-130)
S_SUN = 1361.0  # W m^-2 at 1 au, IAU 2015 Resolution B3
V_ZERO_POINT_MJY = 3.636e6  # mJy for V = 0, Bessell, Castelli & Plez 1998 Table A2
V_WAVELENGTH_UM = 0.55  # reference wavelength of H (V band), Bowell et al. 1989

_H = const.h.si.value
_C = const.c.si.value
_K = const.k_B.si.value
_SIGMA = const.sigma_sb.si.value
_AU_M = u.au.to(u.m)
_KM_M = 1e3
_MJY = 1e-29  # W m^-2 Hz^-1 per mJy


@lru_cache(maxsize=1)
def _solar_table():
    """Smoothed solar spectrum at 1 au on a uniform ln(lambda) grid: (ln wl_um, ln F_nu[mJy])."""
    d = fits.open(SOLAR_FILE)[1].data
    wl_a = np.asarray(d["WAVELENGTH"], float)
    flam = np.asarray(d["FLUX"], float)  # erg s^-1 cm^-2 A^-1 at 1 au
    step = 1e-4  # native sampling is d ln(lambda) = 1.02e-4
    lnw = np.arange(np.log(wl_a[0]), np.log(wl_a[-1]), step)
    flam = np.interp(lnw, np.log(wl_a), flam)
    sigma = 1.0 / SOLAR_R / (2 * np.sqrt(2 * np.log(2))) / step  # FWHM = lambda/R, in samples
    flam = gaussian_filter1d(flam, sigma, mode="nearest")
    wl_a = np.exp(lnw)
    # F_nu = F_lambda * lambda^2 / c  (c in A/s gives erg s^-1 cm^-2 Hz^-1); 1 mJy = 1e-26 cgs
    fnu_mjy = flam * wl_a**2 / 2.99792458e18 * 1e26
    return lnw - np.log(1e4), np.log(fnu_mjy)


def solar_flux_mJy(wl_um, r_au=1.0, delta_au=None):
    """Solar flux density F_nu (mJy) at heliocentric distance r_au, inverse-square scaled from 1 au.

    ``delta_au`` is accepted for interface compatibility and not used (the solar flux incident on
    the object does not depend on the observer distance).
    """
    lnw_grid, lnf_grid = _solar_table()
    lnw = np.log(np.asarray(wl_um, float))
    if np.any(lnw < lnw_grid[0]) or np.any(lnw > lnw_grid[-1]):
        raise ValueError("wavelength outside solar model range (0.15-300 um)")
    return np.exp(np.interp(lnw, lnw_grid, lnf_grid)) / np.asarray(r_au, float) ** 2


def hg_phase_function(phase_deg, G):
    """IAU H,G phase function Phi(alpha) = (1-G) Phi1 + G Phi2, exponential approximation
    Phi_i = exp(-A_i tan(alpha/2)^B_i), A1=3.33, B1=0.63, A2=1.87, B2=1.22 (Bowell et al. 1989,
    in Asteroids II, pp. 524-556, appendix; valid for 0 <= alpha <= 120 deg)."""
    t = np.tan(np.radians(np.asarray(phase_deg, float)) / 2)
    phi1 = np.exp(-3.33 * t**0.63)
    phi2 = np.exp(-1.87 * t**1.22)
    return (1 - G) * phi1 + G * phi2


def v_magnitude(H, G, r_au, delta_au, phase_deg):
    """Apparent V = H + 5 log10(r Delta) - 2.5 log10 Phi(alpha) (Bowell et al. 1989)."""
    return H + 5 * np.log10(r_au * delta_au) - 2.5 * np.log10(hg_phase_function(phase_deg, G))


def reflected_mJy(wl_um, H, G, r_au, delta_au, phase_deg, refl_fn=None):
    """Reflected sunlight F_nu (mJy).

    F(lambda) = F_V * [S(lambda)/S(0.55 um)] * refl(lambda), with F_V = 3636 Jy * 10^(-0.4 V)
    (Bessell et al. 1998 zero point), V from the H,G system, S the solar F_nu spectrum and refl the
    relative reflectance normalised to 1 at 0.55 um (flat if refl_fn is None).
    """
    wl = np.asarray(wl_um, float)
    f_v = V_ZERO_POINT_MJY * 10 ** (-0.4 * v_magnitude(H, G, r_au, delta_au, phase_deg))
    shape = solar_flux_mJy(wl) / solar_flux_mJy(V_WAVELENGTH_UM)
    refl = 1.0 if refl_fn is None else refl_fn(wl)
    return f_v * shape * refl


def diameter_from_H(H, pV):
    """D [km] = 1329 / sqrt(pV) * 10^(-H/5) (Fowler & Chillemi 1992; Pravec & Harris 2007, Icarus 190, 250)."""
    return 1329.0 / np.sqrt(pV) * 10 ** (-np.asarray(H, float) / 5)


def subsolar_temperature(pV, eta, r_au, emissivity=0.9, G=0.15):
    """NEATM subsolar temperature T_ss = [(1-A) S / (eta eps sigma r^2)]^(1/4) (Harris 1998),
    with Bond albedo A = q pV and phase integral q = 0.290 + 0.684 G (Bowell et al. 1989)."""
    A = (0.290 + 0.684 * G) * pV
    return ((1 - A) * S_SUN / (eta * emissivity * _SIGMA * r_au**2)) ** 0.25


def _planck_nu(nu, T):
    """Planck B_nu (W m^-2 Hz^-1 sr^-1); zero where T == 0."""
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        x = _H * nu / (_K * T)
        b = 2 * _H * nu**3 / _C**2 / np.expm1(x)
    return np.where(T > 0, b, 0.0)


def neatm_mJy(wl_um, D_km, pV, eta, r_au, delta_au, phase_deg, emissivity=0.9, G=0.15, n_grid=96):
    """NEATM thermal F_nu (mJy) (Harris 1998, Icarus 131, 291).

    Sphere of diameter D; T(theta, phi) = T_ss (cos theta cos phi)^(1/4) on the day side, 0 on the
    night side (Harris 1998). Latitude theta, longitude phi measured from the subsolar point
    in the Sun-object-observer plane; observer at longitude alpha (phase angle). Flux (Harris 1998;
    written here for the full latitude range):
        F_nu = eps D^2 / (4 Delta^2) * Int_{-pi/2}^{pi/2} dtheta Int_{alpha-pi/2}^{pi/2} dphi
               B_nu(T) cos^2(theta) cos(phi - alpha)
    i.e. emission from the part of the sunlit hemisphere visible to the observer. Integrated by
    Gauss-Legendre quadrature with n_grid nodes in each angle over the exact visible-sunlit domain
    (convergence < 0.5% tested in tests/test_physics.py).
    """
    wl = np.atleast_1d(np.asarray(wl_um, float))
    alpha = np.radians(float(phase_deg))
    t_ss = subsolar_temperature(pV, eta, r_au, emissivity, G)

    x, w = np.polynomial.legendre.leggauss(n_grid)
    # latitude: integrand is even in theta, integrate [0, pi/2] and double
    th = (x + 1) * np.pi / 4
    w_th = w * np.pi / 4 * 2
    # longitude: visible and sunlit for alpha - pi/2 < phi < pi/2
    lo, hi = alpha - np.pi / 2, np.pi / 2
    ph = (x + 1) * (hi - lo) / 2 + lo
    w_ph = w * (hi - lo) / 2

    TH, PH = np.meshgrid(th, ph, indexing="ij")
    cos_sun = np.clip(np.cos(TH) * np.cos(PH), 0.0, None)
    T = t_ss * cos_sun**0.25
    weight = np.outer(w_th, w_ph) * np.cos(TH) ** 2 * np.cos(PH - alpha)

    nu = _C / (wl * 1e-6)
    integral = np.einsum("lij,ij->l", _planck_nu(nu[:, None, None], T[None]), weight)
    D = D_km * _KM_M
    delta = delta_au * _AU_M
    flux = emissivity * D**2 / (4 * delta**2) * integral / _MJY
    return flux if np.ndim(wl_um) else flux[0]
