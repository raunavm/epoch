import numpy as np
import pytest

from orrery import physics as p

C = p._C
K = p._K
H_PL = p._H
SIGMA = p._SIGMA
AU = p._AU_M
MJY = p._MJY


def _bolometric(D, pV, eta, r, delta, phase, emissivity=0.9, G=0.15):
    """Integrate NEATM F_nu over frequency -> W m^-2."""
    wl = np.logspace(np.log10(0.3), np.log10(5000), 1500)
    nu = C / (wl * 1e-6)
    f = p.neatm_mJy(wl, D, pV, eta, r, delta, phase, emissivity, G) * MJY
    return np.trapezoid(f * nu, np.log(nu)) * -1  # nu decreases with wl


# ---------------- solar spectrum ----------------

def test_solar_total_irradiance():
    # the smoothed CALSPEC model must integrate to the IAU 2015 TSI within 1%
    wl = np.logspace(np.log10(0.16), np.log10(290), 20000)
    nu = C / (wl * 1e-6)
    f = p.solar_flux_mJy(wl) * MJY
    tsi = -np.trapezoid(f, nu)
    assert tsi == pytest.approx(p.S_SUN, rel=0.01)


def test_solar_v_consistent_with_sun_magnitude():
    # Sun V = -26.76 (Willmer 2018, ApJS 236, 47) with the Bessell+1998 zero point must match
    # the solar model at 0.55 um within 0.05 mag (measured offset is 2.6%)
    f_pred = p.V_ZERO_POINT_MJY * 10 ** (0.4 * 26.76)
    assert abs(2.5 * np.log10(p.solar_flux_mJy(0.55) / f_pred)) < 0.05


def test_solar_inverse_square():
    assert p.solar_flux_mJy(1.5, r_au=2.0) == pytest.approx(p.solar_flux_mJy(1.5) / 4, rel=1e-12)


def test_solar_out_of_range():
    with pytest.raises(ValueError):
        p.solar_flux_mJy(0.1)


# ---------------- reflected light ----------------

def test_hg_phase_zero():
    assert p.hg_phase_function(0.0, 0.15) == pytest.approx(1.0)


def test_reflected_matches_independent_v():
    H, G, r, d, a = 10.0, 0.25, 2.3, 1.6, 18.0
    t = np.tan(np.radians(a) / 2)
    phi = (1 - G) * np.exp(-3.33 * t**0.63) + G * np.exp(-1.87 * t**1.22)
    V = H + 5 * np.log10(r * d) - 2.5 * np.log10(phi)
    f_v_mjy = 3636.0e3 * 10 ** (-0.4 * V)  # Bessell et al. 1998: 3636 Jy at V = 0
    assert p.reflected_mJy(0.55, H, G, r, d, a) == pytest.approx(f_v_mjy, rel=1e-10)


def test_reflected_spectral_shape_and_refl():
    H, G, r, d, a = 12.0, 0.15, 2.0, 1.2, 25.0
    wl = np.array([0.8, 1.6, 3.2])
    f0 = p.reflected_mJy(wl, H, G, r, d, a)
    ratio = p.solar_flux_mJy(wl) / p.solar_flux_mJy(0.55)
    assert f0 / f0[0] == pytest.approx(ratio / ratio[0], rel=1e-12)
    f1 = p.reflected_mJy(wl, H, G, r, d, a, refl_fn=lambda w: 1 + 0.1 * (w - 0.55))
    assert f1 / f0 == pytest.approx(1 + 0.1 * (wl - 0.55), rel=1e-12)


def test_diameter_from_H():
    assert p.diameter_from_H(0.0, 1.0) == pytest.approx(1329.0)
    # Ceres: H = 3.34, pV = 0.09 -> ~950 km (measured 939 km)
    assert p.diameter_from_H(3.34, 0.09) == pytest.approx(950.0, rel=0.02)


# ---------------- NEATM ----------------

@pytest.mark.parametrize("phase", [0.0, 20.0, 60.0, 90.0, 120.0])
def test_neatm_grid_convergence(phase):
    wl = np.array([1.0, 2.0, 3.0, 5.0, 10.0, 30.0, 100.0, 1000.0])
    f = p.neatm_mJy(wl, 100.0, 0.1, 1.0, 2.5, 1.8, phase)
    f_ref = p.neatm_mJy(wl, 100.0, 0.1, 1.0, 2.5, 1.8, phase, n_grid=1024)
    np.testing.assert_allclose(f, f_ref, rtol=5e-3)


def test_neatm_bolometric_phase0_analytic():
    # phase 0: F_bol = eps sigma T_ss^4 R^2 / (pi Delta^2) * Int cos^2(psi) dOmega = (2/3) eps sigma T^4 R^2/Delta^2
    D, pV, eta, r, d, eps = 50.0, 0.2, 1.0, 1.3, 0.7, 0.9
    T = p.subsolar_temperature(pV, eta, r, eps)
    R = D * 500.0
    expected = 2 / 3 * eps * SIGMA * T**4 * R**2 / (d * AU) ** 2
    assert _bolometric(D, pV, eta, r, d, 0.0, eps) == pytest.approx(expected, rel=0.01)


@pytest.mark.parametrize("eta", [1.0, 0.756])
def test_neatm_energy_balance(eta):
    # emitted power over all observer directions = absorbed power / eta (= absorbed for eta = 1)
    D, pV, r, d, eps, G = 50.0, 0.2, 1.3, 0.7, 0.9, 0.15
    x, w = np.polynomial.legendre.leggauss(40)
    alpha = (x + 1) * np.pi / 2
    w = w * np.pi / 2
    fbol = np.array([_bolometric(D, pV, eta, r, d, np.degrees(a), eps, G) for a in alpha])
    power = 2 * np.pi * (d * AU) ** 2 * np.sum(w * fbol * np.sin(alpha))
    A = (0.290 + 0.684 * G) * pV
    absorbed = (1 - A) * p.S_SUN / r**2 * np.pi * (D * 500.0) ** 2
    assert power == pytest.approx(absorbed / eta, rel=0.01)


def test_neatm_rayleigh_jeans_limit():
    # RJ: F = eps R^2/Delta^2 * 2 k nu^2 T_ss / c^2 * Int cos^(5/4)(psi) dOmega, integral = 2 pi / 2.25
    D, pV, eta, r, d, eps = 100.0, 0.1, 1.0, 2.0, 1.5, 0.9
    wl = 1e5  # 10 cm, h nu / k T_ss ~ 5e-4
    nu = C / (wl * 1e-6)
    T = p.subsolar_temperature(pV, eta, r, eps)
    expected = eps * (D * 500.0) ** 2 / (d * AU) ** 2 * 2 * K * nu**2 * T / C**2 * 2 * np.pi / 2.25
    assert p.neatm_mJy(wl, D, pV, eta, r, d, 0.0, eps) * MJY == pytest.approx(expected, rel=5e-3)


@pytest.mark.parametrize("x", [200.0, 400.0])
def test_neatm_wien_limit(x):
    # Wien, phase 0, Laplace expansion about the subsolar point (x = h nu / k T_ss):
    # F = eps R^2/Delta^2 * (2 h nu^3/c^2) e^-x * (8 pi / x) * (1 - 9/x + O(x^-2))
    D, pV, eta, r, d, eps = 100.0, 0.1, 1.0, 1.0, 1.0, 0.9
    T = p.subsolar_temperature(pV, eta, r, eps)
    nu = x * K * T / H_PL
    wl = C / nu * 1e6
    expected = (eps * (D * 500.0) ** 2 / (d * AU) ** 2 * 2 * H_PL * nu**3 / C**2 * np.exp(-x)
                * 8 * np.pi / x * (1 - 9 / x))
    ratio = p.neatm_mJy(wl, D, pV, eta, r, d, 0.0, eps) * MJY / expected
    assert abs(ratio - 1) < 100 / x**2


def test_neatm_diameter_scaling():
    wl = np.array([3.0, 5.0, 12.0])
    f1 = p.neatm_mJy(wl, 10.0, 0.1, 1.0, 2.5, 1.8, 20.0)
    f2 = p.neatm_mJy(wl, 30.0, 0.1, 1.0, 2.5, 1.8, 20.0)
    np.testing.assert_allclose(f2 / f1, 9.0, rtol=1e-12)


@pytest.mark.parametrize("eta", [0.756, 1.0])
def test_ceres_thermal_reflected_crossover(eta):
    # Ceres: H = 3.34, G = 0.12, D = 939.4 km, pV = 0.090 (JPL SBDB values: D from Park et al. 2016,
    # Nature 537, 515; pV from Li et al. 2006, Icarus 182, 143); r = 2.77, Delta = 1.8, alpha = 20 deg
    wl = np.linspace(1.0, 6.0, 5001)
    th = p.neatm_mJy(wl, 939.4, 0.090, eta, 2.77, 1.8, 20.0, G=0.12)
    rf = p.reflected_mJy(wl, 3.34, 0.12, 2.77, 1.8, 20.0)
    i = np.flatnonzero(np.diff(np.sign(th - rf)))
    assert len(i) == 1
    cross = wl[i[0]]
    print(f"Ceres eta={eta}: thermal = reflected at {cross:.3f} um")
    assert 2.8 < cross < 5.0
