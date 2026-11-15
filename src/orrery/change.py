"""Wavelength-aware change model (finding 1). Spec: docs/INTERFACES.md, "Wavelength-aware change model".

Must stay numerically identical to site/js/live/change.js (same clipping, fit, rejection and counting rules).
Grid convention (matches site/js/live/grid.js): output pixel (i = column, j = row) sits at standard
coordinates xi = (c - i)*s, eta = (c - j)*s about the target, c = (S-1)/2, s = 6.15" -> north up, east left.
"""
import numpy as np
from scipy import ndimage

SCALE_ARCSEC = 6.15
THRESH = 5.0
MIN_PIX = 2
_EIGHT = np.ones((3, 3), bool)


# ---------------------------------------------------------------- grid + resampling
def deproject(xi, eta, ra0, dec0):
    """Inverse gnomonic projection; xi, eta in degrees -> (ra, dec) degrees."""
    x, y, d0 = np.radians(xi), np.radians(eta), np.radians(dec0)
    rho = np.hypot(x, y)
    c = np.arctan(rho)
    with np.errstate(invalid="ignore", divide="ignore"):
        dec = np.arcsin(np.cos(c) * np.sin(d0) + np.where(rho > 0, y * np.sin(c) * np.cos(d0) / rho, 0))
    ra = ra0 + np.degrees(np.arctan2(x * np.sin(c), rho * np.cos(d0) * np.cos(c) - y * np.sin(d0) * np.sin(c)))
    ra = np.where(rho > 0, ra, ra0)
    dec = np.where(rho > 0, dec, d0)
    return ra % 360, np.degrees(dec)


def project(ra, dec, ra0, dec0):
    """Gnomonic projection about (ra0, dec0) -> (xi, eta) in degrees."""
    a, d, d0 = np.radians(np.asarray(ra) - ra0), np.radians(dec), np.radians(dec0)
    cc = np.sin(d0) * np.sin(d) + np.cos(d0) * np.cos(d) * np.cos(a)
    xi = np.cos(d) * np.sin(a) / cc
    eta = (np.cos(d0) * np.sin(d) - np.sin(d0) * np.cos(d) * np.cos(a)) / cc
    return np.degrees(xi), np.degrees(eta)


def grid_world(S, ra0, dec0, scale=SCALE_ARCSEC):
    """(ra, dec) [S, S] of the output grid, indexed [j, i]."""
    c, s = (S - 1) / 2, scale / 3600
    j, i = np.mgrid[0:S, 0:S].astype(float)
    return deproject((c - i) * s, (c - j) * s, ra0, dec0)


def world_to_grid(ra, dec, S, ra0, dec0, scale=SCALE_ARCSEC):
    """Sky position -> fractional grid pixel (i, j)."""
    xi, eta = project(ra, dec, ra0, dec0)
    c, s = (S - 1) / 2, scale / 3600
    return c - xi / s, c - eta / s


def bilinear(img, x, y):
    """Bilinear sample of img[y, x] at fractional 0-based (x, y); NaN outside or if any neighbour is NaN."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    H, W = img.shape
    xi, yi = np.floor(x).astype(int), np.floor(y).astype(int)
    ok = (xi >= 0) & (yi >= 0) & (xi + 1 < W) & (yi + 1 < H) & np.isfinite(x) & np.isfinite(y)
    xi, yi = np.where(ok, xi, 0), np.where(ok, yi, 0)
    fx, fy = x - xi, y - yi
    v = ((1 - fy) * ((1 - fx) * img[yi, xi] + fx * img[yi, xi + 1])
         + fy * ((1 - fx) * img[yi + 1, xi] + fx * img[yi + 1, xi + 1]))
    return np.where(ok, v, np.nan)


# ---------------------------------------------------------------- per-frame statistics
def clipped_median(a):
    """Median after sigma clipping: 3 sigma about the median, scale = std, at most 3 iterations."""
    x = np.asarray(a, float).ravel()
    x = x[np.isfinite(x)]
    for _ in range(3):
        if x.size <= 2:
            break
        m, sd = np.median(x), np.sqrt(np.mean((x - np.median(x)) ** 2))
        k = x[np.abs(x - m) <= 3 * sd]
        if k.size == x.size:
            break
        x = k
    return float(np.median(x)) if x.size else np.nan


def mad_sigma(a):
    x = np.asarray(a, float).ravel()
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if x.size else np.nan


# ---------------------------------------------------------------- the model
def _wfit(v, lam, w, M):
    """Per-pixel WLS of v = a + b (lam - lbar) over frames where M; arrays [E, P], w [E, 1]."""
    W = w * M
    sw = W.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        lbar = (W * np.where(M, lam, 0)).sum(0) / sw
        x = np.where(M, lam - lbar, 0)
        a = (W * np.where(M, v, 0)).sum(0) / sw
        sxx, sxv = (W * x * x).sum(0), (W * x * np.where(M, v, 0)).sum(0)
        b = np.where((M.sum(0) >= 4) & (sxx > 0), sxv / sxx, 0.0)
    return a, b, lbar


def change_model(v, lam, det, mjd, f=0.0):
    """Revision 2 (docs/INTERFACES.md, "Change model — revision 2").
    v, lam: [N, S, S] resampled frames (NaN = no data / masked) and per-pixel central wavelength (um).
    det, mjd: [N]. f: frozen fractional model-error floor, sigma_ep^2 = sigma_e^2 + (f |S_p|)^2.
    Returns dict with bg, sigma, vs, model, corr [N,S,S]; naive_all / naive_det (lists, None where no earlier
    frame) against the previous frame in time (any detector) / of the same detector, prev_all / prev_det
    (indices or -1); n_corr, n_naive_all, n_naive_det [N]."""
    v, lam = np.asarray(v, float), np.asarray(lam, float)
    det, mjd = np.asarray(det), np.asarray(mjd, float)
    N, S, _ = v.shape
    bg = np.array([clipped_median(x) for x in v])
    vs = v - bg[:, None, None]
    sigma = np.array([mad_sigma(x) for x in vs])
    model = np.full_like(vs, np.nan)
    for d in np.unique(det):
        E = np.flatnonzero(det == d)
        V, L = vs[E].reshape(len(E), -1), lam[E].reshape(len(E), -1)
        s2 = (sigma[E] ** 2)[:, None]
        with np.errstate(divide="ignore"):
            w = 1 / s2
        M = np.isfinite(V) & np.isfinite(L) & np.isfinite(w) & (w > 0)
        a, b, lb = _wfit(V, L, w, M)
        cols = np.arange(V.shape[1])
        for _ in range(2):  # drop the single worst frame per pixel if > 4 sigma
            mod = a + b * (L - lb)
            with np.errstate(invalid="ignore"):
                z = np.where(M, np.abs(V - mod) / np.sqrt(s2 + (f * mod) ** 2), -1)
            worst = np.argmax(z, 0)
            drop = (z[worst, cols] > 4) & (M.sum(0) > 1)
            if not drop.any():
                break
            M[worst[drop], cols[drop]] = False
            a, b, lb = _wfit(V, L, w, M)
        mod = a + b * (L - lb)
        mod[:, ~np.isfinite(a)] = np.nan
        mod[~np.isfinite(L)] = np.nan
        model[E] = mod.reshape(len(E), S, S)
    s2 = (sigma ** 2)[:, None, None]
    corr = (vs - model) / np.sqrt(s2 + (f * model) ** 2)
    order = np.lexsort((det, mjd))
    out = dict(bg=bg, sigma=sigma, vs=vs, model=model, corr=corr)
    for name, key in (("all", lambda e: 0), ("det", lambda e: det[e])):
        maps, prev, last = [None] * N, np.full(N, -1), {}
        for e in order:
            q = last.get(key(e))
            if q is not None:
                big = np.maximum(np.abs(vs[e]), np.abs(vs[q]))
                maps[e] = (vs[e] - vs[q]) / np.sqrt(sigma[e] ** 2 + sigma[q] ** 2 + (f * big) ** 2)
                prev[e] = q
            last[key(e)] = e
        out[f"naive_{name}"], out[f"prev_{name}"] = maps, prev
        out[f"n_naive_{name}"] = np.array([count_changes(m) if m is not None else 0 for m in maps])
    out["n_corr"] = np.array([count_changes(c) for c in corr])
    return out


# ---------------------------------------------------------------- change detection
def find_changes(m, thr=THRESH, min_pix=MIN_PIX):
    """Connected groups (8-connectivity) of >= min_pix pixels with |value| > thr.
    Returns [{x, y (peak pixel), peak (signed value at max |value|), npix}]."""
    m = np.asarray(m, float)
    hit = np.abs(np.nan_to_num(m, nan=0.0)) > thr
    lab, n = ndimage.label(hit, structure=_EIGHT)
    out = []
    for k, sl in enumerate(ndimage.find_objects(lab), start=1):
        sub = lab[sl] == k
        npx = int(sub.sum())
        if npx < min_pix:
            continue
        vals = np.where(sub, np.abs(m[sl]), -1)
        iy, ix = np.unravel_index(np.argmax(vals), vals.shape)
        y, x = iy + sl[0].start, ix + sl[1].start
        out.append(dict(x=int(x), y=int(y), peak=float(m[y, x]), npix=npx))
    return out


def count_changes(m, thr=THRESH, min_pix=MIN_PIX):
    return len(find_changes(m, thr, min_pix))
