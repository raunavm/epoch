"""E9 library: wing (annulus) photometry for saturated stars, estimator A, and the per-detector window model.
Choices and their values are recorded in docs/protocols/E9.md."""
import os, sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from orrery.e7 import PHOT_BITS, SAT_BITS  # noqa: E402
from orrery.fit import fit_stamp, flux_mJy  # noqa: E402

HALF = 16
R_IN, R_OUT = (3, 4, 5, 6, 7, 8, 9), (7, 9, 12)
BG = (13.0, 16.0)
GAIA_EPOCH = 57388.5
MJD0, BIN_DAYS = 60780.0, 30.0             # fixed 30-day bin grid for stars with continuous coverage
WAVE_EDGES = {1: (0.74, 1.12), 2: (1.10, 1.65), 3: (1.64, 2.42), 4: (2.42, 3.82), 5: (3.81, 4.42), 6: (4.41, 5.00)}


def load_star(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


def load_target(field, gaia_row):
    """Target stamps from the E7 cache (data/efields/e7_<field>, 77 px cutouts) cropped to the comparison-star
    format (HALF around the pixel nearest the star)."""
    import json
    from astropy.io import fits
    from astropy.wcs import WCS
    from scipy.ndimage import map_coordinates
    from orrery.data import load_stamp
    root = f"data/efields/e7_{field}"
    fl = json.load(open(f"{root}/field.json"))["frame_list"]
    d = {k: [] for k in ("image", "variance", "flags", "mjd", "detector", "x0", "y0", "xs", "ys", "lam", "band",
                         "omega", "epsf", "key")}
    for f in fl:
        st = load_stamp(f"{root}/{f['file']}")
        yr = (st.mjd - GAIA_EPOCH) / 365.25
        r = gaia_row["ra"] + gaia_row["pmra"] * yr / 3.6e6 / np.cos(np.radians(gaia_row["dec"]))
        de = gaia_row["dec"] + gaia_row["pmdec"] * yr / 3.6e6
        x, y = WCS(fits.Header.fromstring(st.wcs_header)).all_world2pix([r], [de], 0)
        xs, ys = float(x[0]), float(y[0])
        ix, iy = int(round(xs)) - st.x0, int(round(ys)) - st.y0
        if ix - HALF < 0 or iy - HALF < 0 or ix + HALF + 1 > st.image.shape[1] or iy + HALF + 1 > st.image.shape[0]:
            continue
        sl = (slice(iy - HALF, iy + HALF + 1), slice(ix - HALF, ix + HALF + 1))
        c = [[ys - st.y0], [xs - st.x0]]
        d["image"].append(st.image[sl]), d["variance"].append(st.variance[sl]), d["flags"].append(st.flags[sl])
        d["mjd"].append(st.mjd), d["detector"].append(st.detector)
        d["x0"].append(st.x0 + ix - HALF), d["y0"].append(st.y0 + iy - HALF), d["xs"].append(xs), d["ys"].append(ys)
        d["lam"].append(float(map_coordinates(st.cwave.astype(float), c, order=1, mode="nearest")[0]))
        d["band"].append(float(map_coordinates(st.cband.astype(float), c, order=1, mode="nearest")[0]))
        d["omega"].append(st.omega_arcsec2), d["epsf"].append(st.epsf.astype(np.float32)), d["key"].append(f["key"])
    return {k: np.array(v) for k, v in d.items()}


def _clip_plane(x, y, v, n_iter=3, k=3.0):
    """Sigma-clipped plane fit; returns coefficients (c0, cx, cy) about (0, 0) and the robust scatter."""
    keep = np.ones(len(v), bool)
    c = np.array([np.median(v), 0.0, 0.0])
    for _ in range(n_iter):
        if keep.sum() < 10:
            break
        A = np.stack([np.ones(keep.sum()), x[keep], y[keep]], 1)
        c, *_ = np.linalg.lstsq(A, v[keep], rcond=None)
        r = v - (c[0] + c[1] * x + c[2] * y)
        s = 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep])))
        keep = np.abs(r) < k * max(s, 1e-12)
    return c, s


def wing_phot(img, flg, cx, cy):
    """Estimator B on one picture. (cx, cy) = star position in stamp pixels. VARIANCE is not used.
    Background: sigma-clipped plane through unflagged pixels at BG radii. Ring profile: median of usable,
    background-subtracted pixels in 1-px rings. Variants per (r_in, r_out):
      fill = sum of usable pixels + ring median for unusable ones;  med = sum over rings of n_pix x ring median.
    Two masks: '' = E7 photometry mask (hard-bad, OVERFLOW, NONLINEAR); 'b' = the same but BLOOM (bit 26) allowed.
    Returns a flat dict (MJy/sr x pixels)."""
    img = np.asarray(img, float)
    H, W = img.shape
    yy, xx = np.indices((H, W))
    dx, dy = xx - cx, yy - cy
    r = np.hypot(dx, dy)
    fin = np.isfinite(img)
    good0 = ((flg & PHOT_BITS) == 0) & fin
    b = good0 & (r >= BG[0]) & (r < BG[1])
    out = dict(n_bg=int(b.sum()))
    if b.sum() < 30:
        return None
    c, s = _clip_plane(dx[b], dy[b], img[b])
    sub = img - (c[0] + c[1] * dx + c[2] * dy)
    out.update(bg=float(c[0]), bg_gx=float(c[1]), bg_gy=float(c[2]), bg_rms=float(s),
               bg_const=float(np.median(img[b])))
    kmax = int(max(R_OUT))
    ring = np.floor(r).astype(int)
    sat = (flg & SAT_BITS) != 0
    out["core_flag"] = bool(sat[r <= 1.0].any())
    out["n_sat"] = int((sat & (r < kmax)).sum())
    out["r_sat"] = float(r[sat & (r < kmax)].max()) if out["n_sat"] else 0.0
    bad0 = ~good0 & (r < kmax)
    out["n_masked"] = int(bad0.sum())
    out["r_masked90"] = float(np.percentile(r[bad0], 90)) if bad0.sum() else 0.0
    core = r < 3.0                                     # plain core sum (only if no pixel there is flagged)
    out["core_sum"] = float(sub[core].sum()) if good0[core].all() else np.nan
    for sfx, good in (("", good0), ("b", ((flg & (PHOT_BITS & ~(1 << 26))) == 0) & fin)):
        med = np.full(kmax, np.nan)
        ngood = np.zeros(kmax, int)
        nall = np.zeros(kmax, int)
        for k in range(kmax):
            m = ring == k
            nall[k] = m.sum()
            g = m & good
            ngood[k] = g.sum()
            if g.sum() >= 3:
                med[k] = np.median(sub[g])
        if sfx == "":
            for k in range(kmax):
                out[f"p{k}"] = med[k]
        for ri in R_IN:
            for ro in R_OUT:
                if ro <= ri:
                    continue
                ks = np.arange(ri, ro)
                ann = (r >= ri) & (r < ro)
                tag = f"{ri}_{ro}"
                out[f"mf{sfx}_{tag}"] = 1.0 - ngood[ks].sum() / nall[ks].sum()
                if np.isnan(med[ks]).any():
                    out[f"fill{sfx}_{tag}"] = out[f"med{sfx}_{tag}"] = np.nan
                    continue
                out[f"med{sfx}_{tag}"] = float((nall[ks] * med[ks]).sum())
                out[f"fill{sfx}_{tag}"] = float(sub[ann & good].sum() + med[ring[ann & ~good]].sum())
    return out


def measure(d, s_d=None, do_a=True):
    """All pictures of one star -> DataFrame (one row per picture)."""
    rows = []
    for i in range(len(d["mjd"])):
        cx, cy = float(d["xs"][i] - d["x0"][i]), float(d["ys"][i] - d["y0"][i])
        o = wing_phot(d["image"][i], d["flags"][i], cx, cy)
        if o is None:
            continue
        det = int(d["detector"][i])
        o.update(mjd=float(d["mjd"][i]), det=det, lam=float(d["lam"][i]), band=float(d["band"][i]),
                 x=float(d["xs"][i]), y=float(d["ys"][i]), omega=float(d["omega"][i]), key=str(d["key"][i]))
        if do_a and "variance" in d and "epsf" in d:
            # estimator A: forced ePSF fit (E7 mask bits) on the central 15 x 15 px, no neighbour stars
            h = 7
            ix, iy = int(round(cx)), int(round(cy))
            sl = (slice(iy - h, iy + h + 1), slice(ix - h, ix + h + 1))
            try:
                r = fit_stamp(d["image"][i][sl], d["variance"][i][sl], d["flags"][i][sl], d["epsf"][i].astype(float),
                              cx - (ix - h), cy - (iy - h), np.zeros((0, 2)), PHOT_BITS)
                sc = (s_d or {}).get(str(det), 1.0)
                o.update(A_mJy=flux_mJy(r.A, o["omega"]), A_err_mJy=flux_mJy(r.A_err, o["omega"]) * sc)
            except Exception:
                o.update(A_mJy=np.nan, A_err_mJy=np.nan)
        rows.append(o)
    return pd.DataFrame(rows)


def window_index(mjd, mode):
    """mode 'gap': windows split at gaps > 30 d (orrery.e7.windows). mode 'bin': fixed 30-day bins from MJD0."""
    from orrery.e7 import windows
    mjd = np.asarray(mjd, float)
    return windows(mjd) if mode == "gap" else np.floor((mjd - MJD0) / BIN_DAYS).astype(int)


def _poly(lam, deg):
    x = lam - lam.mean()
    return np.stack([x ** k for k in range(deg + 1)], 1)


def fit_detector(lnf, lam, win, method="joint", deg=2, x=None, n_clip=2, k_clip=4.0):
    """One star, one detector. ln flux = polynomial(lam) [+ linear term in detector x] + offset per window.
    method 'static': polynomial fitted to all pictures, a_w = exp(median residual in window) - 1 (E7's static model).
    method 'joint' : polynomial and window offsets fitted together; offsets shifted to picture-weighted mean zero.
    Degree drops (2 -> 1 -> 0) while pictures < parameters + 3. Robust: n_clip rounds of k_clip x MAD rejection.
    Returns {window: (a, n_pictures)}, residual scatter (MAD, after window offsets), residuals."""
    n = len(lnf)
    ws = np.unique(win)
    nw = len(ws) if method == "joint" else 1
    while deg > 0 and n < (deg + 1) + (nw - 1) + (x is not None) + 3:
        deg -= 1
    P = _poly(lam, deg)
    if x is not None:
        P = np.concatenate([P, (x - 1020.0)[:, None] / 1000.0], 1)
    if method == "joint" and nw > 1:
        Wm = (win[:, None] == ws[None, 1:]).astype(float)
        A = np.concatenate([P, Wm], 1)
    else:
        A = P
    keep = np.isfinite(lnf)
    res = np.full(n, np.nan)
    for _ in range(n_clip + 1):
        if keep.sum() < A.shape[1] + 1:
            return None
        c, *_ = np.linalg.lstsq(A[keep], lnf[keep], rcond=None)
        base = P @ c[:P.shape[1]]
        res = lnf - base                                   # residual about the static spectrum (offsets left in)
        full = lnf - A @ c
        s = 1.4826 * np.nanmedian(np.abs(full[keep] - np.nanmedian(full[keep])))
        new = np.isfinite(lnf) & (np.abs(full) < k_clip * max(s, 1e-6))
        if (new == keep).all():
            break
        keep = new
    out = {}
    if method == "joint" and nw > 1:
        off = np.concatenate([[0.0], c[P.shape[1]:]])
        cnt = np.array([(keep & (win == w)).sum() for w in ws])
        ok = cnt > 0
        off = off - (off[ok] * cnt[ok]).sum() / cnt[ok].sum()
        for w, o, m in zip(ws, off, cnt):
            if m > 0:
                out[int(w)] = (float(np.expm1(o)), int(m))
    else:
        r0 = res - np.nanmedian(res[keep])
        for w in ws:
            m = keep & (win == w)
            if m.sum() > 0:
                out[int(w)] = (float(np.expm1(np.median(r0[m]))), int(m.sum()))
    return out, float(s), res, keep
