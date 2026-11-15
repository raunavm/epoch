"""Synthetic tests of E1 baseline M5 (wavelength-neighbour median reference)."""
import numpy as np

from orrery.change import find_changes, mad_sigma
from orrery.e1_baselines import M5_NAME, m5_map

BANDS = {1: (0.75, 1.09), 4: (2.42, 3.82)}


def gauss_psf(S, x0, y0, fwhm=1.6):
    s = fwhm / 2.355
    j, i = np.mgrid[0:S, 0:S]
    g = np.exp(-((i - x0) ** 2 + (j - y0) ** 2) / (2 * s * s))
    return g / g.max()


def make_frames(rng, S=32, n_per_det=40, noise=1.0):
    """Detector pairs 1/4 observed simultaneously, random wavelength per visit with an LVF gradient across the
    field, random background level per frame."""
    det, mjd, lam = [], [], []
    for k in range(n_per_det):
        for d in (1, 4):
            lo, hi = BANDS[d]
            l0 = rng.uniform(lo, hi - 0.03 * (hi - lo))
            lam.append(l0 + np.linspace(0, 0.03 * (hi - lo), S)[:, None] * np.ones((1, S)))
            det.append(d)
            mjd.append(60800 + k * 0.7)
    lam, N = np.array(lam), len(det)
    v = rng.normal(0, noise, (N, S, S)) + rng.uniform(-50, 50, N)[:, None, None]
    return v, lam, np.array(det), np.array(mjd)


def sig(v):
    return np.array([mad_sigma(x - np.nanmedian(x)) for x in v])


def test_name_and_shapes():
    rng = np.random.default_rng(0)
    v, lam, det, mjd = make_frames(rng, S=16, n_per_det=10)
    m = m5_map(v, lam, det, mjd, sig(v), 0.1512)
    assert isinstance(M5_NAME, str) and M5_NAME
    assert isinstance(m, list) and len(m) == len(det) and all(x.shape == (16, 16) for x in m)


def test_pure_noise_unit_scatter_and_no_false_changes():
    rng = np.random.default_rng(1)
    v, lam, det, mjd = make_frames(rng)
    m = np.array(m5_map(v, lam, det, mjd, sig(v), 0.1512))
    s = np.std(m)
    assert 1.0 < s < 1.12, s          # leave-one-out median of 15: expected ~1.05
    assert abs(np.mean(m)) < 0.02
    assert sum(len(find_changes(x)) for x in m) == 0


def test_background_level_and_presubtraction_invariant():
    rng = np.random.default_rng(2)
    v, lam, det, mjd = make_frames(rng, S=16, n_per_det=12)
    s = sig(v)
    a = np.array(m5_map(v, lam, det, mjd, s, 0.1512))
    b = np.array(m5_map(v + rng.uniform(-30, 30, len(det))[:, None, None], lam, det, mjd, s, 0.1512))
    vs = v - np.array([np.median(x) for x in v])[:, None, None]
    c = np.array(m5_map(vs, lam, det, mjd, s, 0.1512))
    assert np.allclose(a, b, atol=1e-9) and np.allclose(a, c, atol=1e-9)


def test_nonlinear_spectrum_star_is_static():
    """Bright star with a resolved emission feature at 3.3 um in band 4 (Gaussian sigma 0.08 um, about one SPHEREx
    resolution element; not linear in wavelength): no change components with the frozen floor f."""
    rng = np.random.default_rng(3)
    S = 32
    v, lam, det, mjd = make_frames(rng, S=S, n_per_det=60)
    psf = gauss_psf(S, 15.3, 16.1)
    for e in range(len(det)):
        lc = lam[e][16, 15]
        f = 1 + 3 * np.exp(-0.5 * ((lc - 3.3) / 0.08) ** 2) if det[e] == 4 else 1 + (lc - 0.75) / 0.34
        v[e] += 40.0 * f * psf
    m = m5_map(v, lam, det, mjd, sig(v), 0.1512)
    assert sum(len(find_changes(x)) for x in m) == 0


def test_transient_found_and_not_leaking():
    rng = np.random.default_rng(4)
    S = 32
    v, lam, det, mjd = make_frames(rng, S=S)
    e0 = 31
    v[e0] += 25.0 * gauss_psf(S, 20.4, 9.7)
    m = m5_map(v, lam, det, mjd, sig(v), 0.1512)
    hits = [c for c in find_changes(m[e0]) if np.hypot(c["x"] - 20.4, c["y"] - 9.7) < 1.5]
    assert hits and hits[0]["peak"] > 15
    others = [c for e, x in enumerate(m) if e != e0 for c in find_changes(x)]
    assert not others, others        # median reference: no negative ghost of the transient in other frames


def test_robust_to_several_contaminated_frames():
    """Same pixel hit in 4 frames of one detector (e.g. artefacts): every one is still flagged."""
    rng = np.random.default_rng(5)
    S = 24
    v, lam, det, mjd = make_frames(rng, S=S)
    E = np.flatnonzero(det == 4)
    hit = E[np.argsort(lam[E][:, 12, 12])[10:14]]           # 4 frames adjacent in wavelength
    for e in hit:
        v[e] += 25.0 * gauss_psf(S, 12, 12)
    m = m5_map(v, lam, det, mjd, sig(v), 0.1512)
    for e in hit:
        assert any(np.hypot(c["x"] - 12, c["y"] - 12) < 1.5 for c in find_changes(m[e]))


def test_detectors_kept_separate():
    """A star seen only by detector 1 (blue) must not appear as change in detector 4 maps, and vice versa."""
    rng = np.random.default_rng(6)
    S = 24
    v, lam, det, mjd = make_frames(rng, S=S)
    v[det == 1] += 200.0 * gauss_psf(S, 11.5, 12.2)
    m = m5_map(v, lam, det, mjd, sig(v), 0.1512)
    assert sum(len(find_changes(x)) for x in m) == 0


def test_noise_floor_term_and_nan():
    rng = np.random.default_rng(7)
    S = 24
    v, lam, det, mjd = make_frames(rng, S=S)
    psf = gauss_psf(S, 12, 12)
    v += 500.0 * psf * (1 + 0.1 * rng.normal(size=len(det)))[:, None, None]   # 10% frame-to-frame star wobble
    v[3, 0:4, 0:4] = np.nan
    s = sig(v)
    m0 = np.array(m5_map(v, lam, det, mjd, s, 0.0))
    m1 = np.array(m5_map(v, lam, det, mjd, s, 0.1512))
    assert np.nanmax(np.abs(m1[:, 12, 12])) < np.nanmax(np.abs(m0[:, 12, 12]))
    assert np.isnan(m1[3, 0:4, 0:4]).all() and np.isfinite(m1[3, 10:, 10:]).all()
    assert np.isfinite(m1[5, 0:4, 0:4]).all()                 # other frames simply skip the missing values


def test_few_frames_gives_nan_not_crash():
    rng = np.random.default_rng(8)
    v, lam, det, mjd = make_frames(rng, S=8, n_per_det=3)    # 3 per detector -> only 2 neighbours < MIN_REF
    m = np.array(m5_map(v, lam, det, mjd, sig(v), 0.1512))
    assert np.isnan(m).all()
