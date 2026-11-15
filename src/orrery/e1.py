"""E1 — sensitivity at equal false-alarm rate (docs/PROTOCOLS.md, E1; frozen).

Methods, all on the rev-2 resampled 48 px grid with the identical noise model
sigma^2 = sigma_e^2 + (f |S|)^2 (f frozen in data/findings/f1_calibration.json):
  M1 consecutive blink, all frames in time order         (pairing of orrery.change.change_model prev_all)
  M2 consecutive blink, same detector                     (pairing prev_det)
  M3 per-pixel per-detector median of the OTHER frames, rev-2 rejection, no wavelength term
  M4 change model revision 2, exactly as implemented     (change_model -> corr)
  M5 optional: src/orrery/e1_baselines.py, included if present.

Deviation D1 (docs/PROTOCOLS.md, before any test field): blink S = |reference (earlier) frame value|; every map
is despiked with the live site's rule (despike) before counting; only positive components count.

Planting is done on the RAW native-pixel cutout with the frame's own ePSF zone, before the rev-2
bilinear resampling (scripts/build_field.py `cut`, reproduced here on the cutout).
"""
import importlib
import os

import numpy as np
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator

from orrery.change import bilinear, change_model, clipped_median, mad_sigma, world_to_grid
from orrery.data import BAD_BITS, _calib
from orrery.psf import psf_model

MASK_BITS = BAD_BITS | (1 << 19)            # rev 2: hard-bad + OUTLIER -> NaN before the change count
LEVELS = (3, 5, 8, 12, 20)                  # S/N_frame
SEED0 = 20261115
GAIA_EPOCH = 57388.5                        # J2016.0
K_GRID = np.round(np.arange(0.05, 20000.0001, 0.05), 2)   # protocol sets no upper end
TARGET_RATE = 0.01                          # false components per frame
DET_RADIUS = 1.5                            # px
EIGHT = np.ones((3, 3), bool)
METHODS = ("M1", "M2", "M3", "M4")


# ---------------------------------------------------------------- per-frame preparation
def wave_map(det, x, y, release="dr1"):
    """Per-pixel central wavelength at full-frame native coords (x, y): same as orrery.data.Frame.wave_at."""
    c = _calib(release, int(det))
    f = RegularGridInterpolator((c["wy"], c["wx"]), c["wv"], bounds_error=False, fill_value=None)
    return f(np.stack([np.ravel(y), np.ravel(x)], -1))[:, 0].reshape(np.shape(x))


def epsf_zone(det, x, y, release="dr1"):
    """ePSF of the zone whose centre is nearest to full-frame (x, y): same rule as Frame.epsf_at."""
    c = _calib(release, int(det))
    i = np.argmin((c["xc"] - x) ** 2 + (c["yc"] - y) ** 2)
    return np.asarray(c["epsf"][i], float), int(c["ovs"])


def resample(img, flg, gx, gy):
    """Rev-2 resampling (scripts/build_field.py `cut`) of a native cutout; gx, gy = cutout coords of grid pixels.
    Returns v (masked neighbour -> NaN) and v_raw."""
    v_raw = bilinear(img, gx, gy)
    bad = np.where(flg & MASK_BITS, np.nan, 0.0)
    return v_raw + bilinear(bad, gx, gy), v_raw


def native_sigma(img, flg, gx, gy):
    """sigma_e on the native grid: 1.4826 MAD after clipped-median background, over unmasked native pixels of the
    cutout inside the grid footprint."""
    H, W = img.shape
    x0, x1 = max(0, int(np.floor(np.nanmin(gx)))), min(W, int(np.ceil(np.nanmax(gx))) + 1)
    y0, y1 = max(0, int(np.floor(np.nanmin(gy)))), min(H, int(np.ceil(np.nanmax(gy))) + 1)
    a = np.where(flg[y0:y1, x0:x1] & MASK_BITS, np.nan, img[y0:y1, x0:x1].astype(float))
    return mad_sigma(a - clipped_median(a))


def prepare(field):
    """Add per frame: lam [S,S] (rev 2: NaN where the raw resample is NaN) and sigma_native."""
    for fr in field["frames"]:
        _, v_raw = resample(fr["img"], fr["flg"], fr["gx"], fr["gy"])
        lam = wave_map(fr["det"], fr["gx"] + fr["x0"], fr["gy"] + fr["y0"])
        fr["lam"] = np.where(np.isfinite(v_raw), lam, np.nan)
        fr["sigma_native"] = native_sigma(fr["img"], fr["flg"], fr["gx"], fr["gy"])
    return field


# ---------------------------------------------------------------- planting
def gaia_xy(gaia, mjd, S, ra0, dec0, gmax=19.0):
    """Grid positions of Gaia DR3 G < gmax stars propagated (proper motion) to mjd."""
    if gaia is None or len(gaia) == 0:
        return np.zeros((0, 2))
    g = gaia[gaia["phot_g_mean_mag"] < gmax]
    yr = (mjd - GAIA_EPOCH) / 365.25
    dec = g["dec"].values + np.nan_to_num(g["pmdec"].values) * yr / 3.6e6
    ra = g["ra"].values + np.nan_to_num(g["pmra"].values) * yr / 3.6e6 / np.cos(np.radians(g["dec"].values))
    i, j = world_to_grid(ra, dec, S, ra0, dec0)
    return np.stack([i, j], -1)


def draw_positions(field, level_index, field_index):
    """One position per frame (frames in MJD order), rng = default_rng(20261115 + 1000*level_index + field_index):
    uniform in the central 40x40 px, >= 3 px from every Gaia G < 19 star at the frame epoch and from the target."""
    rng = np.random.default_rng(SEED0 + 1000 * level_index + field_index)
    S = field["S"]
    lo, hi = (S - 40) / 2 - 0.5, (S + 40) / 2 - 0.5          # pixel centres 4..43 for S = 48
    out = []
    for fr in field["frames"]:
        avoid = gaia_xy(field.get("gaia"), fr["mjd"], S, field["ra0"], field["dec0"])
        if field.get("target_xy") is not None:
            avoid = np.vstack([avoid, np.atleast_2d(field["target_xy"](fr["mjd"]))])
        for _ in range(100000):
            x, y = rng.uniform(lo, hi), rng.uniform(lo, hi)
            if len(avoid) == 0 or np.min(np.hypot(avoid[:, 0] - x, avoid[:, 1] - y)) >= 3:
                break
        else:
            raise RuntimeError("no admissible planting position")
        out.append((x, y))
    return np.array(out)


def grid_to_cutout(fr, x, y):
    """Fractional grid (x, y) -> cutout coords, by bilinear interpolation of the per-pixel WCS maps gx, gy
    (TAN-SIP is linear to << 0.01 px over one 6.15" pixel)."""
    return float(bilinear(fr["gx"], x, y)), float(bilinear(fr["gy"], x, y))


def plant(fr, x, y, snr):
    """Add a point source with S/N_frame = snr at grid (x, y) to the raw cutout. Returns (img, info)."""
    xc, yc = grid_to_cutout(fr, x, y)
    epsf, ovs = epsf_zone(fr["det"], xc + fr["x0"], yc + fr["y0"])
    P = psf_model(epsf, xc, yc, fr["img"].shape, oversample=ovs)
    sumP2 = float((P ** 2).sum())
    flux = snr * fr["sigma_native"] / np.sqrt(sumP2)
    return fr["img"] + flux * P, dict(xc=xc, yc=yc, flux=flux, sumP2=sumP2)


# ---------------------------------------------------------------- maps
def m3_maps(vs, sigma, det, f):
    """M3: per pixel, per detector, median of the OTHER frames of that detector; rev-2 rejection (per round drop
    only the single frame with the largest |resid|/sigma if > 4, at most 2 rounds); residual / sigma with
    sigma^2 = sigma_e^2 + (f * model)^2."""
    N, S, _ = vs.shape
    out = np.full_like(vs, np.nan)
    for d in np.unique(det):
        E = np.flatnonzero(det == d)
        V = vs[E].reshape(len(E), -1)
        s2 = (sigma[E] ** 2)[:, None]
        cols = np.arange(V.shape[1])
        for i, e in enumerate(E):
            if len(E) < 2:
                break
            o = np.delete(np.arange(len(E)), i)
            Vo, s2o = V[o], s2[o]
            M = np.isfinite(Vo)
            with np.errstate(all="ignore"):
                med = np.nanmedian(np.where(M, Vo, np.nan), 0)
                for _ in range(2):
                    z = np.where(M, np.abs(Vo - med) / np.sqrt(s2o + (f * med) ** 2), -1)
                    worst = np.argmax(z, 0)
                    drop = (z[worst, cols] > 4) & (M.sum(0) > 1)
                    if not drop.any():
                        break
                    M[worst[drop], cols[drop]] = False
                    med = np.nanmedian(np.where(M, Vo, np.nan), 0)
                out[e] = ((V[i] - med) / np.sqrt(sigma[e] ** 2 + (f * med) ** 2)).reshape(S, S)
    return out


def _m5():
    try:
        return importlib.import_module("orrery.e1_baselines")
    except ImportError:
        return None


def all_methods():
    return METHODS + (("M5",) if _m5() is not None else ())


def blink_d1(vs, sigma, prev, f):
    """D1 blink map: (v_e - v_q) / sqrt(sigma_e^2 + sigma_q^2 + (f |v_q|)^2), q = reference (earlier) frame given by
    rev 2's pairing `prev` (-1 = no earlier frame -> None)."""
    return [None if q < 0 else (vs[e] - vs[q]) / np.sqrt(sigma[e] ** 2 + sigma[q] ** 2 + (f * vs[q]) ** 2)
            for e, q in enumerate(prev)]


def despike(m, thr=5.0):
    """D1 / site/js/live/change.js `despike`: a pixel with |m| > thr is set to 0 when the median of |m| over the
    finite pixels of its 3x3 neighbourhood (itself included, clipped at the edge) is < thr/2."""
    a = np.abs(np.asarray(m, float))
    S0, S1 = a.shape
    pad = np.pad(a, 1, constant_values=np.nan)
    nb = np.stack([pad[dy:dy + S0, dx:dx + S1] for dy in range(3) for dx in range(3)])
    cand = a > thr
    out = np.array(m, float)
    if cand.any():
        med = np.nanmedian(nb[:, cand], 0)
        z = np.zeros_like(cand)
        z[cand] = med < thr / 2
        out[z] = 0.0
    return out


def method_maps(v, lam, det, mjd, f):
    """{method: list of [S,S] maps or None (no map for that frame)}, plus the rev-2 output.
    Deviation D1 (docs/PROTOCOLS.md): blink S = |reference (earlier) frame value|; every map despiked."""
    r = change_model(v, lam, det, mjd, f=f)
    maps = {"M1": blink_d1(r["vs"], r["sigma"], r["prev_all"], f),
            "M2": blink_d1(r["vs"], r["sigma"], r["prev_det"], f),
            "M4": list(r["corr"]), "M3": list(m3_maps(r["vs"], r["sigma"], det, f))}
    mod = _m5()
    if mod is not None:
        maps["M5"] = list(mod.m5_map(v, lam, det, mjd, r["sigma"], f))
    maps = {k: [None if m is None else despike(m) for m in ms] for k, ms in maps.items()}
    return maps, r


# ---------------------------------------------------------------- components
def components(m, k):
    """Label image of 8-connected groups of >= 2 pixels with m > k (D1: positive only; NaN never exceeds)."""
    hit = np.nan_to_num(m, nan=0.0) > k
    lab, n = ndimage.label(hit, structure=EIGHT)
    if n:
        size = np.bincount(lab.ravel())
        lab[size[lab] < 2] = 0
    return lab


def n_components(m, k):
    return len(np.setdiff1d(np.unique(components(m, k)), [0]))


def detected(m, k, x, y):
    lab = components(m, k)
    S = m.shape[0]
    jj, ii = np.mgrid[0:S, 0:S]
    near = np.hypot(ii - x, jj - y) <= DET_RADIUS
    return bool((lab[near] > 0).any())


def kmax_detected(m, x, y, k_grid=K_GRID):
    """Diagnostic: largest grid k at which the planted position is still detected (0 if never). Detection is
    monotone in k (a >= 2 px component above k lies inside a >= 2 px component above any smaller k)."""
    if not detected(m, k_grid[0], x, y):
        return 0.0
    lo, hi = 0, len(k_grid) - 1
    top = np.nanmax(m)
    hi = min(hi, int(np.searchsorted(k_grid, top)))
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if detected(m, k_grid[mid], x, y):
            lo = mid
        else:
            hi = mid - 1
    return float(k_grid[lo])


def false_rate_curve(maps, k_grid=K_GRID):
    """Components per evaluated map for each k (maps that are None are not evaluated)."""
    ms = [m for m in maps if m is not None]
    # max map value bounds the useful k range: above it there are no components
    tot = np.zeros(len(k_grid))
    for m in ms:
        top = np.nanmax(m) if np.isfinite(m).any() else 0
        for i, k in enumerate(k_grid):
            if k >= top:
                break
            tot[i] += n_components(m, k)
    return tot / max(1, len(ms)), len(ms)


def pick_k(rate, k_grid=K_GRID, target=TARGET_RATE):
    """Smallest k on the grid with rate <= target (docs/PROTOCOLS.md, E1)."""
    ok = np.flatnonzero(rate <= target)
    return float(k_grid[ok[0]]) if len(ok) else None


# ---------------------------------------------------------------- one run
def frame_arrays(field, imgs=None):
    fr = field["frames"]
    v = np.array([resample(imgs[i] if imgs is not None else f["img"], f["flg"], f["gx"], f["gy"])[0]
                  for i, f in enumerate(fr)]).astype(np.float32)        # rev-2 cache stores float32
    lam = np.array([f["lam"] for f in fr]).astype(np.float32)
    det = np.array([f["det"] for f in fr])
    mjd = np.array([f["mjd"] for f in fr])
    return v, lam, det, mjd


def run_level(field, field_index, level_index, f, thresholds):
    """Plant one source per frame at LEVELS[level_index], build all maps, score detections.
    Returns {method: {n_trials, n_detected, ...}} and per-frame records."""
    snr = LEVELS[level_index]
    pos = draw_positions(field, level_index, field_index)
    imgs, info = [], []
    for fr, (x, y) in zip(field["frames"], pos):
        im, inf = plant(fr, x, y, snr)
        imgs.append(im)
        info.append(inf)
    v, lam, det, mjd = frame_arrays(field, imgs)
    maps, r = method_maps(v, lam, det, mjd, f)
    S = field["S"]
    res, per_frame = {}, []
    for e, (x, y) in enumerate(pos):
        ix, iy = int(np.clip(round(x), 0, S - 1)), int(np.clip(round(y), 0, S - 1))
        has_data = bool(np.isfinite(v[e, iy, ix]))
        rec = dict(x=round(float(x), 3), y=round(float(y), 3), has_data=has_data,
                   flux=info[e]["flux"], sumP2=info[e]["sumP2"],
                   sigma_native=field["frames"][e]["sigma_native"], sigma_grid=float(r["sigma"][e]))
        for meth, ms in maps.items():
            m = ms[e]
            if m is None or not has_data:
                rec[meth] = None
                continue
            rec[meth] = detected(m, thresholds[meth], x, y)
            rec[meth + "_kmax"] = kmax_detected(m, x, y)       # diagnostic only (not the protocol metric)
        per_frame.append(rec)
    for meth in maps:
        t = [p[meth] for p in per_frame if p[meth] is not None]
        res[meth] = dict(n_trials=len(t), n_detected=int(sum(t)),
                         n_no_map=sum(p[meth] is None and p["has_data"] for p in per_frame))
    return res, per_frame


# ---------------------------------------------------------------- statistics
def s50(eff, levels=LEVELS):
    """S/N_frame at 50% efficiency, linear interpolation at the first upward crossing.
    Returns (value, censor) with censor in {None, '<=min', '>max'}; '>max' -> inf."""
    eff = np.asarray(eff, float)
    if eff[0] >= 0.5:
        return float(levels[0]), "<=min"
    for i in range(1, len(levels)):
        if eff[i] >= 0.5:
            a, b = eff[i - 1], eff[i]
            return float(levels[i - 1] + (0.5 - a) / (b - a) * (levels[i] - levels[i - 1])), None
    return float("inf"), ">max"


def bootstrap(counts, methods, n_boot=2000, seed=SEED0):
    """counts[field][level][method] = (n_det, n_trials). Field bootstrap of pooled efficiencies and S/N50."""
    fields = sorted(counts)
    nf, nl = len(fields), len(LEVELS)
    D = {m: np.array([[counts[fd][l][m][0] for l in range(nl)] for fd in fields], float) for m in methods}
    T = {m: np.array([[counts[fd][l][m][1] for l in range(nl)] for fd in fields], float) for m in methods}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, nf, size=(n_boot, nf))
    eff_b = {m: D[m][idx].sum(1) / np.maximum(T[m][idx].sum(1), 1) for m in methods}     # [B, L]
    s50_b = {m: np.array([s50(e)[0] for e in eff_b[m]]) for m in methods}
    eff = {m: D[m].sum(0) / np.maximum(T[m].sum(0), 1) for m in methods}
    return eff, eff_b, s50_b


def q(a, p):
    """Quantile that tolerates inf (nearest-rank on the sorted sample)."""
    a = np.sort(np.asarray(a, float))
    return float(a[min(len(a) - 1, max(0, int(np.ceil(p * len(a))) - 1))])
