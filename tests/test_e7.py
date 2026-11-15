import numpy as np

from orrery.e7 import change_fit, forced_photometry, static_fit, windows
from orrery.fit import fit_stamp
from orrery.psf import psf_model

from test_e6 import COLS, EPSF, OVS, SHAPE, STARS, scene


def test_photometry_matches_fit_stamp_and_group_sum():
    rng = np.random.default_rng(4)
    var = 1.0 + rng.uniform(0, 0.5, SHAPE)
    flags = np.zeros(SHAPE, np.int32)
    flags[8, 6] = 1 << 1                                       # OVERFLOW on a star core -> masked here only
    img = scene(rng, rng.uniform(100, 3000, len(STARS)))
    ph = forced_photometry(img, var, flags, COLS, [[0], [1], [1, 2]])
    r = fit_stamp(img, var, flags | ((flags & 2) << 9), EPSF, -100.0, -100.0, STARS, 1 << 10, OVS)   # same mask
    assert np.isclose(ph[0][0], r.coef[1]) and np.isclose(ph[0][1], np.sqrt(r.cov[1, 1]))
    assert np.isclose(ph[2][0], r.coef[2] + r.coef[3])
    assert np.isclose(ph[2][1], np.sqrt(r.cov[2, 2] + r.cov[3, 3] + 2 * r.cov[2, 3]))


def test_windows():
    assert list(windows([0, 1, 40, 41, 5, 200])) == [0, 0, 1, 1, 0, 2]


def test_static_and_change():
    rng = np.random.default_rng(5)
    n = 300
    lam = rng.uniform(0.75, 5.0, n)
    det = np.digitize(lam, [1.1, 1.6, 2.4, 3.8, 4.4]) + 1
    mjd = np.sort(rng.choice([0, 1, 2, 3, 100, 101, 102, 250, 251], n))
    win = windows(mjd)
    truth = 100 * (1 + 0.1 * lam)
    err = np.full(n, 1.0)
    f0 = truth + rng.normal(size=n)
    _, _, chi_s, npar = static_fit(f0, err, lam, det)
    c = change_fit(f0, err, lam, det, win)
    assert 0.7 < chi_s / (n - npar) < 1.3 and chi_s - c["chi2"] < 15
    f1 = truth * (1 + np.array([0, 0.05, -0.03])[win]) + rng.normal(size=n)
    c = change_fit(f1, err, lam, det, win)
    assert abs(c["a"][1] - 0.05) < 4 * c["a_err"][1] and abs(c["a"][2] + 0.03) < 4 * c["a_err"][2]
    assert static_fit(f1, err, lam, det)[2] - c["chi2"] > 100
