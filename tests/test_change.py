"""Synthetic tests of the wavelength-aware change model (docs/INTERFACES.md)."""
import numpy as np
from scipy.stats import norm

from orrery.change import change_model, count_changes, find_changes, grid_world, world_to_grid

# SPHEREx-like band edges (um) per detector.
BANDS = {1: (0.75, 1.09), 2: (1.10, 1.62), 3: (1.63, 2.41), 4: (2.42, 3.82), 5: (3.83, 4.41), 6: (4.42, 5.00)}


def gauss_psf(S, x0, y0, fwhm=1.6):
    s = fwhm / 2.355
    j, i = np.mgrid[0:S, 0:S]
    g = np.exp(-((i - x0) ** 2 + (j - y0) ** 2) / (2 * s * s))
    return g / g.sum()


def make_frames(rng, S, n_per_det, dets=(1, 4), noise=1.0):
    """Frames ordered in time, alternating detector pairs (taken simultaneously), random wavelength per visit,
    with a per-pixel wavelength gradient across the field (the LVF) and a random background level."""
    det, mjd, lam = [], [], []
    for k in range(n_per_det):
        for d in dets:
            lo, hi = BANDS[d]
            l0 = rng.uniform(lo, hi - 0.03 * (hi - lo))
            grad = np.linspace(0, 0.03 * (hi - lo), S)[None, :] * np.ones((S, 1))
            lam.append(l0 + grad)
            det.append(d)
            mjd.append(60800 + k * 0.7 + (0 if k < n_per_det // 2 else 180))
    lam = np.array(lam)
    N = len(det)
    v = rng.normal(0, noise, (N, S, S)) + rng.uniform(-50, 50, N)[:, None, None]
    return v, lam, np.array(det), np.array(mjd)


def test_grid_roundtrip_north_up():
    S, ra0, dec0 = 48, 120.0, -30.0
    ra, dec = grid_world(S, ra0, dec0)
    i, j = world_to_grid(ra, dec, S, ra0, dec0)
    jj, ii = np.mgrid[0:S, 0:S]
    assert np.allclose(i, ii, atol=1e-6) and np.allclose(j, jj, atol=1e-6)
    assert dec[0, 24] > dec[-1, 24]          # row 0 is north
    assert ra[24, 0] > ra[24, -1]            # column 0 is east


def test_steep_spectrum_star_naive_many_corrected_none():
    rng = np.random.default_rng(1)
    S = 32
    v, lam, det, mjd = make_frames(rng, S, 24)
    psf = gauss_psf(S, 15.3, 16.1)
    # Red, steep spectrum: flux doubles across band 1 and rises x4 across band 4 (in each band nearly linear).
    peak_snr = 60.0
    for e in range(len(det)):
        lo, hi = BANDS[det[e]]
        f = 1 + (lam[e] - lo) / (hi - lo) * (1 if det[e] == 1 else 3)
        v[e] += peak_snr * f * psf / psf.max()
    r = change_model(v, lam, det, mjd)
    n_naive, n_corr = r["n_naive_det"].sum(), r["n_corr"].sum()
    assert n_naive >= 20, n_naive
    assert r["n_naive_all"].sum() >= n_naive  # mixing detectors (different bands) only adds colour "changes"
    assert n_corr <= 1, n_corr
    assert 1 - n_corr / n_naive > 0.95


def test_moving_source_survives_correction():
    rng = np.random.default_rng(2)
    S = 32
    v, lam, det, mjd = make_frames(rng, S, 20)
    # static star with a spectrum, plus a mover crossing the field over all visits
    star = gauss_psf(S, 8, 8)
    N = len(det)
    pos = []
    for e in range(N):
        v[e] += 40 * (1 + 0.5 * (lam[e, 8, 8] - BANDS[det[e]][0])) * star / star.max()
        x, y = 4 + 24 * e / (N - 1), 24 - 12 * e / (N - 1)
        pos.append((x, y))
        p = gauss_psf(S, x, y)
        v[e] += 25 * p / p.max()
    r = change_model(v, lam, det, mjd)
    hits = 0
    for e in range(N):
        ch = find_changes(r["corr"][e])
        if any(np.hypot(c["x"] - pos[e][0], c["y"] - pos[e][1]) < 2 and c["peak"] > 0 for c in ch):
            hits += 1
    assert hits >= 0.9 * N, (hits, N)
    # nothing left at the static star
    assert all(not any(np.hypot(c["x"] - 8, c["y"] - 8) < 2 for c in find_changes(c_)) for c_ in r["corr"])


def _expected_pairs(S, p):
    """Expected number of 8-connected groups of >= 2 pixels in an S x S map of independent pixels exceeding
    with probability p, to leading order (adjacent pairs; groups of 3+ are O(p^3))."""
    n_adj = 2 * S * (S - 1) + 2 * (S - 1) ** 2
    return n_adj * p * p


def test_pure_noise_matches_gaussian_expectation():
    rng = np.random.default_rng(3)
    S = 96
    v, lam, det, mjd = make_frames(rng, S, 30)
    r = change_model(v, lam, det, mjd)
    naive = [m for m in r["naive_det"] if m is not None] + [m for m in r["naive_all"] if m is not None]
    # naive maps are unit-variance Gaussian (MAD noise estimate is unbiased for Gaussian noise)
    sd = np.std(np.concatenate([m.ravel() for m in naive]))
    assert abs(sd - 1) < 0.02, sd
    # corrected maps: residuals of a per-pixel fit, so slightly below unit variance
    sdc = np.nanstd(r["corr"])
    assert 0.9 < sdc <= 1.0, sdc
    # at the change threshold (5 sigma, >= 2 px) the Gaussian expectation is ~1e-6 per map
    assert r["n_naive_det"].sum() == 0 and r["n_naive_all"].sum() == 0 and r["n_corr"].sum() == 0
    exp5 = len(naive) * _expected_pairs(S, 2 * norm.sf(5))
    assert exp5 < 1e-3
    # at a lower threshold the count must match the Gaussian expectation (Poisson, 4 sigma)
    thr = 3.0
    obs = sum(count_changes(m, thr=thr) for m in naive)
    exp = len(naive) * _expected_pairs(S, 2 * norm.sf(thr))
    assert abs(obs - exp) < 4 * np.sqrt(exp) + 1, (obs, exp)


def test_count_changes_connectivity():
    m = np.zeros((10, 10))
    m[2, 2] = m[3, 3] = 6        # diagonal pair -> one change (8-connectivity)
    m[7, 7] = -9                 # single pixel -> not a change
    m[5, 0] = m[5, 1] = -6       # negative pair -> one change
    m[0, 9] = np.nan
    assert count_changes(m) == 2
    ch = find_changes(m)
    assert {(c["x"], c["y"]) for c in ch} == {(2, 2), (0, 5)} or {(c["x"], c["y"]) for c in ch} == {(3, 3), (0, 5)}


def test_worst_frame_rejection_with_few_frames():
    """3 frames of one detector, one with a cosmic-ray-like spike: the spike is flagged, the other two are not."""
    rng = np.random.default_rng(4)
    S = 16
    v = rng.normal(0, 1, (3, S, S))
    lam = np.full((3, S, S), 1.0) + np.arange(3)[:, None, None] * 0.05
    v[1, 8, 8:10] += 30
    r = change_model(v, lam, np.ones(3, int), np.arange(3.0))
    assert r["n_corr"].tolist() == [0, 1, 0]
    assert abs(r["corr"][0, 8, 8]) < 4 and abs(r["corr"][2, 8, 8]) < 4


def test_noise_floor_scales_both_maps():
    """A bright static star with a frame-to-frame fractional error 3%: with f = 0 it makes changes in both maps,
    with f = 0.05 (the floor) neither map shows changes, and pure-background pixels are unaffected."""
    rng = np.random.default_rng(5)
    S = 24
    v, lam, det, mjd = make_frames(rng, S, 12)
    psf = gauss_psf(S, 12, 12)
    for e in range(len(det)):
        v[e] += 2000 * (1 + rng.normal(0, 0.03)) * psf / psf.max()
    r0 = change_model(v, lam, det, mjd, f=0.0)
    r1 = change_model(v, lam, det, mjd, f=0.05)
    assert r0["n_corr"].sum() > 5 and r0["n_naive_det"].sum() > 5
    assert r1["n_corr"].sum() == 0 and r1["n_naive_det"].sum() == 0
    assert np.allclose(r0["corr"][:, 0, 0], r1["corr"][:, 0, 0], rtol=1e-3, atol=1e-3)
