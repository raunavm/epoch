"""Synthetic checks of the E1 harness (orrery.e1)."""
import numpy as np

from orrery import e1


def synth_field(n=60, S=48, sigma=1.0, seed=0):
    """Pure-noise native cutouts (70x70) on a rotated affine grid; detectors 1..6 cycling; no flags."""
    rng = np.random.default_rng(seed)
    th = np.radians(17.0)
    j, i = np.mgrid[0:S, 0:S].astype(float)
    frames = []
    for e in range(n):
        det = 1 + e % 6
        dx, dy = rng.uniform(-0.5, 0.5, 2)
        gx = 35 + np.cos(th) * (i - 23.5) - np.sin(th) * (j - 23.5) + dx
        gy = 35 + np.sin(th) * (i - 23.5) + np.cos(th) * (j - 23.5) + dy
        frames.append(dict(key=str(e), obs_id=str(e), det=det, mjd=60000.0 + e * 0.1, x0=1000, y0=1000,
                           img=rng.normal(0, sigma, (70, 70)), flg=np.zeros((70, 70), np.int32), gx=gx, gy=gy))
    return e1.prepare(dict(name="synth", ra0=10.0, dec0=-20.0, S=S, frames=frames, gaia=None))


def test_native_sigma_and_planted_flux():
    fd = synth_field(n=6)
    fr = fd["frames"][0]
    assert abs(fr["sigma_native"] - 1.0) < 0.08
    img, inf = e1.plant(fr, 20.3, 25.7, 5)
    P = (img - fr["img"]) / inf["flux"]
    assert abs(P.sum() - 1) < 1e-6                       # whole ePSF box is on the cutout
    assert abs(inf["flux"] * np.sqrt((P ** 2).sum()) / fr["sigma_native"] - 5) < 1e-9


def test_positions_seeded_and_in_centre():
    fd = synth_field(n=20)
    a = e1.draw_positions(fd, 1, 3)
    b = e1.draw_positions(fd, 1, 3)
    assert np.array_equal(a, b)
    assert a.min() >= 3.5 and a.max() <= 43.5
    assert not np.array_equal(a, e1.draw_positions(fd, 2, 3))


def test_bright_detected_noise_not():
    fd = synth_field(n=60)
    k = {m: 5.0 for m in e1.all_methods()}
    res, _ = e1.run_level(fd, 0, 4, 0.1512, k)          # S/N 20
    for m in e1.all_methods():
        assert res[m]["n_detected"] / res[m]["n_trials"] > 0.4, (m, res[m])   # D1 despike also trims faint peaks
    v, lam, det, mjd = e1.frame_arrays(fd)
    maps, _ = e1.method_maps(v, lam, det, mjd, 0.1512)
    for m in e1.all_methods():
        rate, n = e1.false_rate_curve(maps[m], np.array([5.0]))
        assert rate[0] < 0.05


def test_s50():
    assert e1.s50([0, 0.2, 0.6, 1, 1]) == (5 + 0.3 / 0.4 * 3, None)
    assert e1.s50([0.7, 1, 1, 1, 1]) == (3.0, "<=min")
    assert e1.s50([0, 0, 0, 0, 0.4])[1] == ">max"


def test_despike_and_positive_only():
    m = np.zeros((10, 10))
    m[4, 4], m[4, 5] = 50, 40                    # 2-px spike: 3x3 medians are 0 -> zeroed
    m[1:4, 7:10] = 8                             # 3x3 source: kept
    d = e1.despike(m)
    assert d[4, 4] == 0 and d[4, 5] == 0 and d[2, 8] == 8 and d[1, 7] == 0   # corners of a 3x3 block: 4/9 lit
    assert e1.n_components(d, 5) == 1 and e1.n_components(-d, 5) == 0


def test_blink_d1_reference_noise():
    vs = np.array([np.full((2, 2), 10.0), np.zeros((2, 2))])
    m = e1.blink_d1(vs, np.array([1.0, 1.0]), np.array([-1, 0]), 0.1)
    assert m[0] is None and np.allclose(m[1], -10 / np.sqrt(2 + 1.0))
