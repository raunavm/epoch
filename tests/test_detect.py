"""E30 detector (orrery.detect): the Python port equals run_e27.m3b_maps; exact component helpers equal orrery.e1."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from orrery import detect, e1  # noqa: E402
import run_e27  # noqa: E402


def stack(seed=0, N=30, S=16):
    rng = np.random.default_rng(seed)
    det = np.array([1 + e % 3 for e in range(N)])
    vs = rng.normal(0, 1, (N, S, S)) + 20 * (rng.random((N, S, S)) < 0.01)
    vs[3, 2:5, 2:5] = np.nan
    vs[:, 0, 0] = np.nan
    sigma = rng.uniform(0.8, 1.2, N)
    return vs, sigma, det


def test_m3b_equals_run_e27():
    for seed in range(3):
        vs, sigma, det = stack(seed)
        ref = run_e27.m3b_maps(vs, sigma, det, 0.15)
        med, z, c = detect.m3b(vs, sigma, det, 0.15)
        assert np.array_equal(ref, z, equal_nan=True)
        assert np.all(c[np.isfinite(c)] >= 1)


def test_m3b_single_frame_detector_and_use():
    vs, sigma, det = stack(1, N=7)
    det[6] = 4                                         # a detector with one picture has no map
    med, z, c = detect.m3b(vs, sigma, det, 0.1)
    assert np.isnan(z[6]).all()
    use = np.ones(7, bool)
    use[0] = False                                     # picture 0 is never anyone's reference but gets a map
    med2, z2, _ = detect.m3b(vs, sigma, det, 0.1, use)
    assert np.isfinite(z2[0]).any()
    assert np.array_equal(z2[0], z[0], equal_nan=True)  # its own reference never contained itself


def test_candidates_local_maxima():
    z = np.zeros((8, 8))
    z[2, 3], z[2, 4], z[6, 6] = 5.0, 4.0, 3.0
    z[0, 7] = np.nan
    x, y, v = detect.find_candidates(z)
    assert list(zip(x, y)) == [(3, 2), (6, 6)] and list(v) == [5.0, 3.0]


def test_component_helpers_exact():
    rng = np.random.default_rng(5)
    for t in range(20):
        m = rng.normal(0, 1.5, (12, 12))
        m[rng.random((12, 12)) < 0.05] = np.nan
        vals, d = detect.comp_events(m, 0.5)
        for k in (0.5, 1.0, 1.7, 2.3, 3.1):
            assert detect.count_events(vals, d, k) == e1.n_components(m, k)
        x, y = rng.uniform(2, 9, 2)
        km = detect.comp_kmax(m, x, y, 1.5)
        for k in (0.5, 1.0, 1.7, 2.3):
            lab = e1.components(m, k)
            jj, ii = np.mgrid[0:12, 0:12]
            assert ((lab[np.hypot(ii - x, jj - y) <= 1.5] > 0).any()) == (km > k)


def test_thresholds():
    s = np.array([5.0, 4.0, 3.0, 2.0])
    assert detect.threshold_scores(s, 100, 0.02) == 3.0          # two may pass
    assert detect.threshold_scores(s, 10, 1.0) == -np.inf
    vals, d = np.array([3.0, 2.0, 1.5]), np.array([1, 1, -1])
    assert detect.threshold_events(vals, d, 10, 0.1, 1.0) == 1.0  # count(1) = 1 <= 1
    assert detect.threshold_events(vals, d, 10, 0.0, 1.0) == 3.0  # count(2) = 1, count(3) = 0


def test_forward_shapes_and_zrank():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (4, 5, 9, 9))
    D = detect.onehot([1, 2, 3, 6])
    w = dict(c1w=rng.normal(0, .1, (12, 5, 3, 3)), c1b=np.zeros(12), c2w=rng.normal(0, .1, (16, 12, 3, 3)),
             c2b=np.zeros(16), c3w=rng.normal(0, .1, (16, 16, 3, 3)), c3b=np.zeros(16),
             d1w=rng.normal(0, .1, (150, 16)), d1b=np.zeros(16), d2w=rng.normal(0, .1, (16, 1)), d2b=np.zeros(1))
    assert detect.forward(dict(kind="cnn", weights=w), X, D).shape == (4,)
    z = detect.forward(dict(kind="zrank"), X, D)
    assert np.allclose(detect.g(z), X[:, 2, 4, 4])
