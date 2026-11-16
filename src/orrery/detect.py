"""E30 two-stage change detector (docs/protocols/E30.md); must stay numerically identical to site/js/live/detect.js.

Stage 1: M3b (scripts/run_e27.py m3b_maps) returning the reference and the per-pixel scale too; candidates are positive
3x3 local maxima with z >= ZMIN. Stage 2: a 9x9 five-channel cutout scored by a small CNN or a 6-feature logistic
regression (numpy forward from the weights in site/data/model/e30.json). Also exact helpers for the component rule
(E27 detection) and for thresholds at a false-candidate rate.
"""
import warnings

import numpy as np

ZMIN = 3.0
HALF = 4                     # 9 x 9 cutout
C_MIN_FRAMES = 5
NDET = 6


# ---------------------------------------------------------------- stage 1
def m3b(vs, sigma, det, f, use=None):
    """run_e27.m3b_maps with the reference restricted to pictures with use[e] (still leaving out the picture itself).
    Returns med, z, c [N, S, S] (NaN where no map). use=None: all pictures (== run_e27.m3b_maps)."""
    vs = np.asarray(vs, float)
    N, S, _ = vs.shape
    use = np.ones(N, bool) if use is None else np.asarray(use, bool)
    med_o, z_o, c_o = (np.full_like(vs, np.nan) for _ in range(3))
    for d in np.unique(det):
        E = np.flatnonzero(det == d)
        if len(E) < 2:
            continue
        V = vs[E].reshape(len(E), -1)
        s2 = (np.asarray(sigma, float)[E] ** 2)[:, None]
        cols = np.arange(V.shape[1])
        for i, e in enumerate(E):
            o = np.array([k for k in range(len(E)) if k != i and use[E[k]]], int)
            if len(o) == 0:
                continue
            Vo, s2o = V[o], s2[o]
            M = np.isfinite(Vo)
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                med = np.nanmedian(np.where(M, Vo, np.nan), 0)
                for _ in range(2):
                    zz = np.where(M, np.abs(Vo - med) / np.sqrt(s2o + (f * med) ** 2), -1)
                    worst = np.argmax(zz, 0)
                    drop = (zz[worst, cols] > 4) & (M.sum(0) > 1)
                    if not drop.any():
                        break
                    M[worst[drop], cols[drop]] = False
                    med = np.nanmedian(np.where(M, Vo, np.nan), 0)
                m = (V[i] - med) / np.sqrt(s2[i] + (f * med) ** 2)
                za = np.where(M, np.abs(Vo - med) / np.sqrt(s2o + (f * med) ** 2), np.nan)
                c = np.maximum(1.0, 1.4826 * np.nanmedian(za, 0))
                c = np.where(M.sum(0) >= C_MIN_FRAMES, c, 1.0)
            med_o[e], z_o[e], c_o[e] = med.reshape(S, S), (m / c).reshape(S, S), c.reshape(S, S)
    return med_o, z_o, c_o


def _nbmax(a, fill):
    """3 x 3 neighbourhood maximum (edge-clipped) of a 2-D array."""
    S0, S1 = a.shape
    p = np.pad(a, 1, constant_values=fill)
    return np.max(np.stack([p[dy:dy + S0, dx:dx + S1] for dy in range(3) for dx in range(3)]), 0)


def find_candidates(z, zmin=ZMIN):
    """Pixels with finite z >= zmin that equal the maximum of the finite values in their 3x3 neighbourhood.
    Row-major order. Returns int arrays x, y and float z."""
    a = np.where(np.isfinite(z), z, -np.inf)
    ok = np.isfinite(z) & (a >= zmin) & (a >= _nbmax(a, -np.inf))
    y, x = np.nonzero(ok)
    return x, y, a[y, x]


def g(x):
    return np.sign(x) * np.log1p(np.abs(x))


def cutouts(vs, med, z, c, sigma, x, y, h=HALF):
    """[n, 5, 9, 9] float64: g(vs/sigma), g(med/sigma), g(z), log c, finite(z); NaN and off-grid -> 0."""
    S = z.shape[0]
    w = 2 * h + 1
    with np.errstate(all="ignore"):
        ch = np.stack([g(vs / sigma), g(med / sigma), g(z), np.log(c), np.isfinite(z).astype(float)])
    ch = np.where(np.isfinite(ch), ch, 0.0)
    P = np.zeros((5, S + 2 * h, S + 2 * h))
    P[:, h:h + S, h:h + S] = ch
    out = np.empty((len(x), 5, w, w))
    for k, (xi, yi) in enumerate(zip(x, y)):
        out[k] = P[:, yi:yi + w, xi:xi + w]
    return out


def onehot(det):
    det = np.atleast_1d(det).astype(int)
    o = np.zeros((len(det), NDET))
    o[np.arange(len(det)), det - 1] = 1.0
    return o


# ---------------------------------------------------------------- stage 2 (numpy forward)
def lr_features(X):
    """6 hand features from cutouts [n, 5, 9, 9] (z recovered from the g(z) channel)."""
    gz = X[:, 2]
    zr = np.sign(gz) * np.expm1(np.abs(gz))
    h = X.shape[-1] // 2
    zc = zr[:, h, h]
    s3 = zr[:, h - 1:h + 2, h - 1:h + 2].sum((1, 2))
    return np.stack([g(zc), g(s3 / 3), g(zc - (s3 - zc) / 8), X[:, 3, h, h], X[:, 1, h, h], X[:, 4].mean((1, 2))], 1)


def _conv(x, W, b):
    """Valid 3x3 cross-correlation: x [n, C, H, W], W [O, C, 3, 3] -> [n, O, H-2, W-2]."""
    win = np.lib.stride_tricks.sliding_window_view(x, (3, 3), axis=(2, 3))       # n, C, H-2, W-2, 3, 3
    return np.einsum("nchwij,ocij->nohw", win, W, optimize=True) + b[None, :, None, None]


def forward(model, X, D):
    """Logits for cutouts X [n, 5, 9, 9] and detector one-hots D [n, 6]; model = e30.json 'model' / 'ablation' dict."""
    X = np.asarray(X, float)
    if model["kind"] == "cnn":
        w = {k: np.asarray(v, float) for k, v in model["weights"].items()}
        a = np.maximum(_conv(X, w["c1w"], w["c1b"]), 0)
        a = np.maximum(_conv(a, w["c2w"], w["c2b"]), 0)
        a = np.maximum(_conv(a, w["c3w"], w["c3b"]), 0)
        hh = np.concatenate([a.reshape(len(X), -1), np.asarray(D, float)], 1)
        hh = np.maximum(hh @ w["d1w"] + w["d1b"], 0)
        return (hh @ w["d2w"] + w["d2b"])[:, 0]
    if model["kind"] == "logreg":
        w = model["weights"]
        F = (lr_features(X) - np.asarray(w["mean"])) / np.asarray(w["std"])
        return F @ np.asarray(w["w"]) + float(w["b"])
    if model["kind"] == "zrank":
        gz = X[:, 2, HALF, HALF]
        return np.sign(gz) * np.expm1(np.abs(gz))
    raise ValueError(model["kind"])


# ---------------------------------------------------------------- counting helpers
def threshold_scores(scores, n_pictures, rate):
    """Smallest t with #(scores > t) / n_pictures <= rate (-inf when every candidate may pass)."""
    s = np.sort(np.asarray(scores, float))[::-1]
    m = int(np.floor(rate * n_pictures + 1e-9))
    return float(s[m]) if len(s) > m else -np.inf


def comp_kmax(m, x, y, radius):
    """Largest k at which a positive 8-connected >= 2 px component of {m > k} has a pixel within `radius` of (x, y):
    max over those pixels p of min(m_p, max over the 8 neighbours q of m_q) (NaN as 0, as orrery.e1.components)."""
    a = np.nan_to_num(np.asarray(m, float), nan=0.0)
    S0, S1 = a.shape
    p = np.pad(a, 1, constant_values=-np.inf)
    nb = np.max(np.stack([p[dy:dy + S0, dx:dx + S1] for dy in range(3) for dx in range(3) if (dy, dx) != (1, 1)]), 0)
    jj, ii = np.mgrid[0:S0, 0:S1]
    near = np.hypot(ii - x, jj - y) <= radius
    return float(np.max(np.minimum(a, nb)[near])) if near.any() else -np.inf


def comp_events(m, kmin):
    """Exact component-count curve of {m > k} (8-connected, >= 2 px) for k >= kmin, as events (value, delta):
    count(k) = sum of delta over events with value > k."""
    a = np.nan_to_num(np.asarray(m, float), nan=0.0)
    S0, S1 = a.shape
    idx = np.flatnonzero(a.ravel() > kmin)
    if not len(idx):
        return np.zeros(0), np.zeros(0, np.int32)
    order = idx[np.argsort(-a.ravel()[idx], kind="stable")]
    parent, size = {}, {}

    def root(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]
            u = parent[u]
        return u
    vals, deltas = a.ravel()[order], np.zeros(len(order), np.int32)
    for n, p in enumerate(order):
        py, px = divmod(int(p), S1)
        R = set()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                qy, qx = py + dy, px + dx
                if (dy or dx) and 0 <= qy < S0 and 0 <= qx < S1 and qy * S1 + qx in parent:
                    R.add(root(qy * S1 + qx))
        deltas[n] = (1 if R else 0) - sum(size[r] >= 2 for r in R)
        parent[p], size[p] = p, 1 + sum(size[r] for r in R)
        for r in R:
            parent[r] = p
    return vals, deltas


def threshold_events(vals, deltas, n_maps, rate, kmin):
    """Smallest k >= kmin with count(k) / n_maps <= rate, count(k) = sum(deltas[vals > k])."""
    T = rate * n_maps + 1e-9
    if not len(vals):
        return float(kmin)
    u, inv = np.unique(vals, return_inverse=True)
    d = np.bincount(inv, weights=deltas, minlength=len(u))
    total = d.sum()
    if total <= T:
        return float(kmin)
    above = total - np.cumsum(d)                  # count(k = u_i): events with value > u_i
    ok = np.flatnonzero(above <= T)
    return float(u[ok[0]])


def count_events(vals, deltas, k):
    return int(np.asarray(deltas)[np.asarray(vals) > k].sum())
