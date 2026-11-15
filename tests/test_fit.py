import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orrery.fit import fit_stamp, flux_mJy, refine_position  # noqa: E402
from orrery.psf import psf_model, psf_native  # noqa: E402

OVS = 5
SHAPE = (25, 27)
SRC = (13.37, 11.81)
STARS = [(5.2, 6.7), (20.6, 18.3)]
A_TRUE, B_TRUE, BG = 400.0, (900.0, 250.0), (3.0, 0.05, -0.08)


def gaussian_epsf(sigma_native=0.55, n=33, ovs=OVS):
    c = (n - 1) / 2
    iy, ix = np.indices((n, n))
    e = np.exp(-((ix - c) ** 2 + (iy - c) ** 2) / (2 * (sigma_native * ovs) ** 2))
    return e / e.sum()


EPSF = gaussian_epsf()


def truth_image(src=SRC, a=A_TRUE):
    H, W = SHAPE
    yy, xx = np.indices(SHAPE)
    img = a * psf_model(EPSF, *src, SHAPE, OVS)
    for (sx, sy), b in zip(STARS, B_TRUE):
        img += b * psf_model(EPSF, sx, sy, SHAPE, OVS)
    return img + BG[0] + BG[1] * (xx - (W - 1) / 2) + BG[2] * (yy - (H - 1) / 2)


def noisy(rng, var):
    return truth_image() + rng.normal(size=SHAPE) * np.sqrt(var)


def test_recovery_pulls_and_chi2():
    rng = np.random.default_rng(1)
    flags = np.zeros(SHAPE, np.int32)
    pulls, chi2dof, loc_c, loc_d = [], [], [], []
    for _ in range(300):
        var = rng.uniform(4.0, 16.0, SHAPE)  # heteroscedastic, known
        r = fit_stamp(noisy(rng, var), var, flags, EPSF, *SRC, STARS, bad_mask_bits=1, oversample=OVS)
        pulls.append((r.A - A_TRUE) / r.A_err)
        chi2dof.append(r.chi2 / r.dof)
        loc_c.append(r.chi2_local)
        loc_d.append(r.dof_local)
    pulls = np.array(pulls)
    n = len(pulls)
    assert np.mean(np.abs(pulls) < 3) >= 0.99           # within 3 sigma (expect 99.73%)
    assert abs(pulls.mean()) < 3 / np.sqrt(n)            # unbiased
    assert 0.9 < pulls.std(ddof=1) < 1.1                 # errors correctly sized
    assert abs(np.mean(chi2dof) - 1) < 0.02              # chi2/dof ~ 1 (per-trial sd ~0.06)
    assert abs(np.sum(loc_c) / np.sum(loc_d) - 1) < 0.1  # local chi2 vs effective dof


def test_masked_pixels_excluded():
    rng = np.random.default_rng(2)
    var = np.full(SHAPE, 9.0)
    img = noisy(rng, var)
    flags = np.zeros(SHAPE, np.int32)
    BAD, OUTLIER = 1 << 0, 1 << 5
    ref = fit_stamp(img, var, flags, EPSF, *SRC, STARS, BAD | 4, OVS)

    bad_img, bad_var, bad_flags = img.copy(), var.copy(), flags.copy()
    bad_px = [(12, 13), (11, 14), (3, 3), (20, 22)]  # (y, x); first two sit on the source
    for y, x in bad_px:
        bad_img[y, x] = 1e6
        bad_flags[y, x] = BAD
    bad_var[0, 0] = np.nan
    bad_var[0, 1] = 0.0
    bad_flags[5, 5] = OUTLIER  # not in bad_mask_bits: must stay in the fit
    r = fit_stamp(bad_img, bad_var, bad_flags, EPSF, *SRC, STARS, BAD | 4, OVS)

    expected_mask = np.ones(SHAPE, bool)
    for y, x in bad_px + [(0, 0), (0, 1)]:
        expected_mask[y, x] = False
    assert np.array_equal(r.mask, expected_mask)
    assert r.dof == ref.dof - 6
    # The same fit on clean data with those pixels flagged must agree exactly.
    clean_flags = flags.copy()
    for y, x in bad_px + [(0, 0), (0, 1)]:
        clean_flags[y, x] = BAD
    r_clean = fit_stamp(img, var, clean_flags, EPSF, *SRC, STARS, BAD | 4, OVS)
    np.testing.assert_allclose(r.coef, r_clean.coef, rtol=1e-10)
    assert abs(r.A - A_TRUE) < 4 * r.A_err


def test_fully_masked_star_is_dropped():
    rng = np.random.default_rng(3)
    var = np.full(SHAPE, 9.0)
    img = noisy(rng, var)
    off = [(60.0, 60.0)]  # star far off the stamp: zero column
    r = fit_stamp(img, var, np.zeros(SHAPE, np.int32), EPSF, *SRC, STARS + off, 1, OVS)
    assert np.isnan(r.coef[3]) and np.isfinite(r.A_err)


def test_refine_position_recovers_offset():
    rng = np.random.default_rng(4)
    var = np.full(SHAPE, 1.0)
    img = truth_image(a=5000.0) + rng.normal(size=SHAPE)
    xb, yb, surf, offs = refine_position(img, var, np.zeros(SHAPE, np.int32), EPSF,
                                         SRC[0] - 0.3, SRC[1] + 0.3, STARS, 1, OVS)
    assert surf.shape == (31, 31) and np.isclose(offs[0], -1.5) and np.isclose(offs[-1], 1.5)
    assert abs(xb - SRC[0]) < 0.05 and abs(yb - SRC[1]) < 0.05


def test_psf_model_normalisation():
    rng = np.random.default_rng(5)
    H, W = SHAPE
    yy, xx = np.indices(SHAPE)
    pos = list(rng.uniform(5, 19, (500, 2))) + [(10.0, 10.0), (10.5, 10.0), (10.0, 12.5)]
    for xs, ys in pos:
        p = psf_model(EPSF, xs, ys, SHAPE, OVS, box=3)
        inbox = (np.abs(xx - xs) <= 3) & (np.abs(yy - ys) <= 3)
        assert abs(p[inbox].sum() - 1) < 1e-12
        assert np.all(p[~inbox] == 0)


def test_psf_native_grid_and_orientation():
    e = np.random.default_rng(6).random((33, 33))
    # Spline interpolation reproduces grid nodes; x offset moves along columns (ix).
    assert np.isclose(psf_native(e, 0.0, 0.0), e[16, 16])
    assert np.isclose(psf_native(e, 0.4, -0.2), e[15, 18])
    assert psf_native(e, 3.3, 0.0) == 0.0  # outside the +-3.2 px grid


def test_flux_mJy():
    # 1 MJy/sr over 1 sr -> 1e9 mJy
    assert np.isclose(flux_mJy(1.0, 206264.806**2), 1e9)
