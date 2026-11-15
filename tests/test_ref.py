import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import jax  # noqa: E402

from orrery_ref import epsf_eval, ref_fit, ref_flux_mJy, ref_loglike  # noqa: E402

OS = 5
SIG = 0.55  # native px
H, W = 21, 23
SRC = (11.3, 9.6)
STARS = [(4.2, 15.1), (17.7, 4.4)]
A_TRUE, B_TRUE = 60.0, [120.0, 35.0]
BG = (2.0, 0.05, -0.03)  # c0, c1, c2 about the stamp centre
BAD = 1 | 4


def make_epsf():
    c = 16
    g = (np.arange(33) - c) / OS
    e = np.exp(-0.5 * (g[None, :] ** 2 + g[:, None] ** 2) / SIG**2)
    return e / e.sum()


def gauss_psf(x, y):
    """Analytic truth: Gaussian at pixel centres, normalised over the |d|<=3 box."""
    yy, xx = np.mgrid[0:H, 0:W]
    img = np.exp(-0.5 * ((xx - x) ** 2 + (yy - y) ** 2) / SIG**2)
    k = np.arange(-4, 5)
    bx, by = np.meshgrid(np.round(x) + k - x, np.round(y) + k - y)
    inbox = (np.abs(bx) <= 3) & (np.abs(by) <= 3)
    norm = np.exp(-0.5 * (bx**2 + by**2) / SIG**2)[inbox].sum()
    return img / norm


def truth_mean():
    yy, xx = np.mgrid[0:H, 0:W]
    mu = A_TRUE * gauss_psf(*SRC)
    for b, s in zip(B_TRUE, STARS):
        mu += b * gauss_psf(*s)
    mu += BG[0] + BG[1] * (xx - (W - 1) / 2) + BG[2] * (yy - (H - 1) / 2)
    return mu


def make_stamp(rng):
    mu = truth_mean()
    var = rng.uniform(0.15, 0.35, size=(H, W))
    img = mu + rng.normal(size=(H, W)) * np.sqrt(var)
    return img, var, np.zeros((H, W), np.int32)


def test_epsf_eval_interpolates_nodes_and_zero_outside():
    e = make_epsf()
    iy, ix = np.mgrid[0:33, 0:33]
    v = np.asarray(epsf_eval(e, (ix - 16) / OS, (iy - 16) / OS, OS))
    assert np.allclose(v, e, atol=1e-15)
    assert float(epsf_eval(e, np.array(3.3), np.array(0.0), OS)) == 0.0


def test_flux_unbiased_and_pulls():
    e = make_epsf()
    rng = np.random.default_rng(12345)
    A, Aerr, chi2r = [], [], []
    for _ in range(300):
        img, var, fl = make_stamp(rng)
        r = ref_fit(img, var, fl, e, *SRC, STARS, BAD, OS)
        A.append(r["A"])
        Aerr.append(r["A_err"])
        chi2r.append(r["chi2"] / r["dof"])
    A, Aerr = np.array(A), np.array(Aerr)
    pull = (A - A_TRUE) / Aerr
    n = len(A)
    assert abs(A.mean() - A_TRUE) < 3 * A.std(ddof=1) / np.sqrt(n)
    assert abs(pull.mean()) < 3 / np.sqrt(n)
    assert 0.9 < pull.std(ddof=1) < 1.1
    assert abs(np.mean(chi2r) - 1) < 0.05


def test_flux_conversion():
    omega = 6.2**2
    assert ref_flux_mJy(1.0, omega) == pytest.approx(omega / 206264.806**2 * 1e9)


def test_gradient_points_to_truth():
    e = make_epsf()
    rng = np.random.default_rng(7)
    img, var, fl = make_stamp(rng)
    g = jax.grad(ref_loglike, argnums=(0, 1))
    for off in [(0.4, -0.3), (-0.25, 0.35), (0.6, 0.5)]:
        x0, y0 = SRC[0] + off[0], SRC[1] + off[1]
        gx, gy = (float(v) for v in g(x0, y0, img, var, fl, e, STARS, BAD, OS))
        # each component points back toward the true position
        assert gx * off[0] < 0 and gy * off[1] < 0
    # finite-difference check of the gradient
    x0, y0 = SRC[0] + 0.2, SRC[1] - 0.1
    gx, _ = g(x0, y0, img, var, fl, e, STARS, BAD, OS)
    h = 1e-5
    fd = (
        ref_loglike(x0 + h, y0, img, var, fl, e, STARS, BAD, OS)
        - ref_loglike(x0 - h, y0, img, var, fl, e, STARS, BAD, OS)
    ) / (2 * h)
    assert float(gx) == pytest.approx(float(fd), rel=1e-5)
    # loglike agrees with ref_fit chi2
    r = ref_fit(img, var, fl, e, x0, y0, STARS, BAD, OS)
    assert float(ref_loglike(x0, y0, img, var, fl, e, STARS, BAD, OS)) == pytest.approx(
        -0.5 * r["chi2"], rel=1e-10
    )


def test_masked_pixels():
    e = make_epsf()
    rng = np.random.default_rng(3)
    img, var, fl = make_stamp(rng)
    ref = ref_fit(img, var, fl, e, *SRC, STARS, BAD, OS)

    img2, var2, fl2 = img.copy(), var.copy(), fl.copy()
    # bad-bit pixels with garbage values, including on the source core
    bad = [(9, 11), (10, 11), (3, 3), (15, 20)]  # (y, x)
    for y, x in bad:
        fl2[y, x] = 4
        img2[y, x] = 1e6
    img2[0, 0] = np.nan
    fl2[0, 0] = 1
    var2[5, 5] = 0.0
    var2[6, 6] = np.nan
    var2[7, 7] = -1.0
    img2[7, 7] = 1e9
    # non-bad bits (e.g. OUTLIER/SOURCE) must NOT be masked
    fl2[12, 12] = 2 | 8
    r2 = ref_fit(img2, var2, fl2, e, *SRC, STARS, BAD, OS)
    n_masked = len(bad) + 4
    assert r2["dof"] == ref["dof"] - n_masked
    assert r2["mask"].sum() == H * W - n_masked and r2["mask"][12, 12]
    assert np.all(np.isfinite(r2["coeffs"])) and np.isfinite(r2["chi2"])

    # identical to fitting only the good pixels: compare with clean values masked
    img3 = img.copy()
    var3, fl3 = var2.copy(), fl2.copy()
    var3[[5, 6, 7], [5, 6, 7]] = var[[5, 6, 7], [5, 6, 7]]
    fl3[[5, 6, 7], [5, 6, 7]] = 1
    r3 = ref_fit(img3, var3, fl3, e, *SRC, STARS, BAD, OS)
    assert np.allclose(r2["coeffs"], r3["coeffs"], rtol=1e-12, atol=1e-12)
    assert r2["chi2"] == pytest.approx(r3["chi2"], rel=1e-12)
    # masking core pixels costs precision
    assert r2["A_err"] > ref["A_err"]
    assert r2["model"].shape == (H, W) and r2["resid"].shape == (H, W)
