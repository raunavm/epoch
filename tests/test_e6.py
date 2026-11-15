import numpy as np

from orrery.data import BAD_BITS
from orrery.e6 import native_z, scene_fit
from orrery.fit import fit_stamp
from orrery.psf import psf_model

OVS = 5
SHAPE = (41, 43)
STARS = [(6.2, 7.7), (30.6, 28.3), (20.1, 10.4), (12.5, 33.0), (35.3, 9.8), (22.4, 22.9)]


def gaussian_epsf(sigma_native=0.55, n=33, ovs=OVS):
    c = (n - 1) / 2
    iy, ix = np.indices((n, n))
    e = np.exp(-((ix - c) ** 2 + (iy - c) ** 2) / (2 * (sigma_native * ovs) ** 2))
    return e / e.sum()


EPSF = gaussian_epsf()
COLS = [psf_model(EPSF, x, y, SHAPE, OVS) for x, y in STARS]


def scene(rng, amps, sig=1.0):
    H, W = SHAPE
    yy, xx = np.indices(SHAPE)
    img = sum(a * c for a, c in zip(amps, COLS))
    img = img + rng.normal(50, 5) + rng.normal(0, 0.05) * (xx - (W - 1) / 2) + rng.normal(0, 0.05) * (yy - (H - 1) / 2)
    return img + rng.normal(size=SHAPE) * sig


def test_identical_to_fit_stamp_and_multi_rhs():
    rng = np.random.default_rng(1)
    var = np.full(SHAPE, 1.0) + rng.uniform(0, 0.5, SHAPE)
    flags = np.zeros(SHAPE, np.int32)
    flags[3, 5] = 1 << 10                                     # HOT -> masked
    flags[17, 20] = 1 << 19                                   # OUTLIER is not hard-bad -> kept in the fit
    imgs = [scene(rng, rng.uniform(0, 3000, len(STARS))) for _ in range(3)]
    resid, stars, mask = scene_fit(imgs, var, flags, COLS)
    assert not mask[3, 5] and mask[17, 20]
    for i, im in enumerate(imgs):                            # off-stamp moving source -> its column is dropped
        r = fit_stamp(im, var, flags, EPSF, -100.0, -100.0, STARS, BAD_BITS, OVS)
        assert np.allclose(resid[i], r.resid, atol=1e-8)
        B = r.coef[1:1 + len(STARS)]
        assert np.allclose(stars[i], sum(b * c for b, c in zip(B, COLS)), atol=1e-8)


def test_static_stars_with_arbitrary_amplitudes_vanish():
    rng = np.random.default_rng(2)
    z = []
    for _ in range(30):
        amps = rng.choice([0.0, 50.0, 5e3, 3e4], len(STARS)) * rng.uniform(0.5, 1.5, len(STARS))
        r, s, m = scene_fit([scene(rng, amps)], np.ones(SHAPE), np.zeros(SHAPE, np.int32), COLS)
        z.append(native_z(r[0], s[0], m, 1.0, 0.0))
    z = np.array(z)
    assert 0.85 < np.nanmean(z ** 2) < 1.05                   # pure noise left (dof loss ~ 9/1763)
    assert np.nanmax(np.abs(z)) < 5.5


def test_new_source_survives():
    rng = np.random.default_rng(3)
    amps = rng.uniform(1e3, 3e4, len(STARS))
    xs, ys = 27.4, 16.2                                       # >= 3 px from every star, not in the star list
    assert min(np.hypot(np.array(STARS)[:, 0] - xs, np.array(STARS)[:, 1] - ys)) >= 3
    P = psf_model(EPSF, xs, ys, SHAPE, OVS)
    flux = 10.0 / np.sqrt((P ** 2).sum())                     # S/N_frame = 10 at sigma = 1
    base = scene(rng, amps)
    r, s, m = scene_fit([base, base + flux * P], np.ones(SHAPE), np.zeros(SHAPE, np.int32), COLS)
    z0, z1 = native_z(r[0], s[0], m, 1.0, 0.0), native_z(r[1], s[1], m, 1.0, 0.0)
    iy, ix = int(round(ys)), int(round(xs))
    assert z1[iy, ix] - z0[iy, ix] > 0.9 * flux * P[iy, ix]   # >= 90% of the peak survives the star fit
    assert z1[iy, ix] > 5
