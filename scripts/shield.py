"""False-discovery shield: does a feature follow the DETECTOR position or the SKY position?

The check that would have prevented a withdrawn claim (the VY CMa "shell", Banerjee et al. 2026, arXiv:2609.18895,
withdrawn; shown to be a Band-1 detector artefact by Karambelkar, Sharma & De 2026, arXiv:2609.28612). Replicated on
DR1 in E19 (docs/protocols/E19.md, data/findings/e19.json).

Usage (from the repo root; reads only headers and a band of image rows per picture from the public S3 bucket):
  # 1. list every DR1 picture (detector 1) containing a star, with the star's detector x, y from each header WCS
  .venv/bin/python scripts/shield.py --name "VY CMa"
  .venv/bin/python scripts/shield.py --ra 110.743026 --dec -25.767554 --detector 1
  # 2. test a suspected feature: measure every picture and ask whether it follows the detector position
  .venv/bin/python scripts/shield.py --name "VY CMa" --test --spot 1648 1752 --others "eps Oct" "GY UMa" --max 30
What --test reports, per picture: d_spot (distance of the star from the suspect detector position), S (smooth ring
excess 30-50 px vs 60-80 px, noise units), Z (radial-spike structure 20-70 px of the glow-subtracted image) and
match = pixel correlation, in DETECTOR coordinates, of the glow-subtracted 20-70 px pattern with the reference picture
(the target's picture closest to the spot, or --ref OBS_ID). Then the summary:
  - other stars (--others) near the spot (< 15 px): if their pictures match the reference much better than pictures
    elsewhere, the same pattern appears around DIFFERENT stars at the SAME detector pixels -> detector artefact.
  - the target elsewhere: no match is expected for an artefact. (A real sky feature would also not match in detector
    coordinates when the picture is rotated, so this alone is not decisive; the other-stars-at-the-spot test is.)
  - blank-spot control (--blank N): the same detector pixels in N pictures where no bright star is there; matches here
    would mean fixed detector patterns (bad pixels, flat residue), not a star-induced artefact.
Choose --others as bright stars that DR1 places near the spot (e.g. found with mode 1 or an archive search), and
compare against stars of similar brightness elsewhere before believing any feature.
Options: --detector (1-6), --spot X Y (0-based), --max N per star (all pictures < 150 px from the spot are kept; the
rest spread evenly in distance), --ref OBS_ID, --blank N, --workers (<= 3 recommended), --out file.json.
"""
import argparse, json, os, sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from astropy.wcs import WCS  # noqa: E402
from e2_cache import _get, _header  # noqa: E402
from orrery.change import project  # noqa: E402
from orrery.data import BUCKET  # noqa: E402

SPOT = (1648.0, 1752.0)        # Karambelkar et al. 2026: VY CMa position in the two "shell" pictures (detector 1)
HALF = 90
NPIX = 2040
INDEX = "data/index/frames_dr1.parquet"


def d1_index(detector=1):
    df = pd.read_parquet(INDEX)
    return df[(df.detector == detector) & (df.dqa == "Pass")].reset_index(drop=True)


def pictures_containing(idx, ra, dec):
    """Index rows whose footprint contains (ra, dec), with approximate x, y from a 4-corner homography (the corners
    are pixels (0,0), (2039,0), (2039,2039), (0,2039)); exact values come from detector_xy()."""
    a1, d1, a2, d2 = map(np.radians, (idx.ra.values, idx.dec.values, ra, dec))
    sep = np.degrees(2 * np.arcsin(np.sqrt(np.sin((d1 - d2) / 2) ** 2 + np.cos(d1) * np.cos(d2) * np.sin((a1 - a2) / 2) ** 2)))
    sub = idx[sep < 2.8].copy()
    if not len(sub):
        return sub.assign(x_approx=[], y_approx=[], d_approx=[])
    xi, eta = project(np.stack(sub.c_ra.values), np.stack(sub.c_dec.values), ra, dec)
    X = np.array([0, NPIX - 1, NPIX - 1, 0], float)
    Y = np.array([0, 0, NPIX - 1, NPIX - 1], float)
    n = len(sub)
    A = np.zeros((n, 8, 8))
    b = np.zeros((n, 8))
    for i in range(4):
        u, v = xi[:, i], eta[:, i]
        A[:, 2 * i] = np.stack([u, v, np.ones(n), 0 * u, 0 * u, 0 * u, -u * X[i], -v * X[i]], 1)
        A[:, 2 * i + 1] = np.stack([0 * u, 0 * u, 0 * u, u, v, np.ones(n), -u * Y[i], -v * Y[i]], 1)
        b[:, 2 * i], b[:, 2 * i + 1] = X[i], Y[i]
    h = np.linalg.solve(A, b[..., None])[..., 0]                  # star at (xi, eta) = (0, 0) -> x = h2 / 1, y = h5 / 1
    sub["x_approx"], sub["y_approx"] = h[:, 2], h[:, 5]
    sub = sub[(sub.x_approx > -5) & (sub.x_approx < NPIX + 4) & (sub.y_approx > -5) & (sub.y_approx < NPIX + 4)]
    sub["d_approx"] = np.hypot(sub.x_approx - SPOT[0], sub.y_approx - SPOT[1])
    return sub


def _image_header(key):
    url = f"{BUCKET}/{key}"
    _, o = _header(url, 0)
    h, img_off = _header(url, o)
    assert h["EXTNAME"] == "IMAGE" and h["BITPIX"] == -32 and h["NAXIS1"] == NPIX
    return url, h, img_off


def detector_xy(key, ra, dec):
    """0-based detector x, y of (ra, dec) from the picture's own TAN-SIP header WCS."""
    _, h, _ = _image_header(key)
    x, y = WCS(h).all_world2pix([ra], [dec], 0)
    return float(x[0]), float(y[0])


def cutout(key, x, y, half=HALF):
    """IMAGE cutout (2*half+1)^2 centred on the rounded (x, y), NaN-padded off the array (MJy/sr)."""
    url, h, img_off = _image_header(key)
    ix, iy = int(round(x)), int(round(y))
    y0, y1 = max(0, iy - half), min(NPIX, iy + half + 1)
    rows = np.frombuffer(_get(url, img_off + y0 * NPIX * 4, img_off + y1 * NPIX * 4), ">f4").reshape(-1, NPIX)
    out = np.full((2 * half + 1, 2 * half + 1), np.nan, np.float32)
    x0, x1 = max(0, ix - half), min(NPIX, ix + half + 1)
    out[y0 - (iy - half):y1 - (iy - half), x0 - (ix - half):x1 - (ix - half)] = rows[:, x0:x1]
    return out


def ring_stat(img, dx=0.0, dy=0.0, r_in=30, r_out=50, bg=(60, 80), nsec=16):
    """Ring excess S (units of local pixel noise) and its azimuthal-scatter error; see module docstring.
    (dx, dy) = exact source position minus the cutout centre pixel."""
    c = (img.shape[0] - 1) / 2
    yy, xx = np.mgrid[:img.shape[0], :img.shape[1]]
    X, Y = xx - c - dx, yy - c - dy
    r, phi = np.hypot(X, Y), np.arctan2(Y, X)
    ok = np.isfinite(img)
    bgm = ok & (r >= bg[0]) & (r < bg[1])
    b = np.median(img[bgm])
    s = 1.4826 * np.median(np.abs(img[bgm] - b))
    ring = ok & (r >= r_in) & (r < r_out)
    sec = ((phi + np.pi) / (2 * np.pi) * nsec).astype(int) % nsec
    m = np.array([np.median(img[ring & (sec == k)]) if (ring & (sec == k)).sum() > 10 else np.nan for k in range(nsec)])
    m = m[np.isfinite(m)]
    S = (m.mean() - b) / s
    S_err = m.std(ddof=1) / np.sqrt(len(m)) / s
    F = float((img[ring] > b + 3 * s).mean() - (img[bgm] > b + 3 * s).mean())
    return dict(S=float(S), S_err=float(S_err), F=F, bg=float(b), sigma_pix=float(s),
                nan_frac_ring=float(1 - ring.sum() / ((r >= r_in) & (r < r_out)).sum()))


def _polar(shape, dx, dy):
    c = (shape[0] - 1) / 2
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    X, Y = xx - c - dx, yy - c - dy
    return np.hypot(X, Y), np.arctan2(Y, X)


def halo_subtract(img, dx=0.0, dy=0.0):
    """Image minus its azimuthal median profile (1-px radius bins): removes the smooth stellar halo."""
    r, _ = _polar(img.shape, dx, dy)
    ri = r.astype(int)
    ok = np.isfinite(img)
    prof = np.full(ri.max() + 1, np.nan)
    for k in np.unique(ri[ok]):
        prof[k] = np.median(img[ok & (ri == k)])
    return img - prof[ri]


def spike_stat(img, dx=0.0, dy=0.0, r_in=20, r_out=70, noise=(75, 88), nbin=120):
    """Azimuthal-structure statistic Z of the halo-subtracted image (docs/protocols/E19.md, Deviation 1):
    a_k = median over radius (r_in..r_out) of azimuth bin k; Z = (P90 - P10) of a_k / (2.563 x expected noise)."""
    R = halo_subtract(img, dx, dy)
    r, phi = _polar(img.shape, dx, dy)
    ok = np.isfinite(R)
    nz = ok & (r >= noise[0]) & (r < noise[1])
    s = 1.4826 * np.median(np.abs(R[nz] - np.median(R[nz])))
    ann = ok & (r >= r_in) & (r < r_out)
    k = ((phi + np.pi) / (2 * np.pi) * nbin).astype(int) % nbin
    a, sa = [], []
    for j in range(nbin):
        m = ann & (k == j)
        if m.sum() >= 10:
            a.append(np.median(R[m]))
            sa.append(1.2533 * s / np.sqrt(m.sum()))
    a, den = np.array(a), 2.563 * np.median(sa)
    Z = (np.percentile(a, 90) - np.percentile(a, 10)) / den
    bs = np.random.default_rng(0).choice(a, (200, len(a)))       # bootstrap over azimuth bins -> error of Z
    Zb = (np.percentile(bs, 90, axis=1) - np.percentile(bs, 10, axis=1)) / den
    return dict(Z=float(Z), Z_err=float(Zb.std()), sigma_R=float(s))


def pattern(img, dx=0.0, dy=0.0, r_in=20, r_out=70, noise=(75, 88)):
    """Detector-frame pattern for correlation tests: halo-subtracted image / its noise, clipped to +-5, NaN outside
    the r_in..r_out annulus (docs/protocols/E19.md, Deviation 2)."""
    R = halo_subtract(img, dx, dy)
    r, _ = _polar(img.shape, dx, dy)
    nz = np.isfinite(R) & (r >= noise[0]) & (r < noise[1])
    s = 1.4826 * np.median(np.abs(R[nz] - np.median(R[nz])))
    P = np.clip(R / s, -5, 5)
    P[(r < r_in) | (r >= r_out)] = np.nan
    return P


def corr(P, Q):
    m = np.isfinite(P) & np.isfinite(Q)
    return float(np.corrcoef(P[m], Q[m])[0, 1])


def _resolve(name):
    from astropy.coordinates import SkyCoord
    c = SkyCoord.from_name(name)
    return float(c.ra.deg), float(c.dec.deg)


def _pictures(idx, ra, dec, spot, nmax, ex):
    cand = pictures_containing(idx, ra, dec)
    cand["d_approx"] = np.hypot(cand.x_approx - spot[0], cand.y_approx - spot[1])
    if nmax and len(cand) > nmax:                       # keep all near the spot, then spread evenly in distance
        near, far = cand[cand.d_approx < 200], cand[cand.d_approx >= 200].sort_values("d_approx")
        k = max(0, nmax - len(near))
        cand = pd.concat([near, far.iloc[np.linspace(0, len(far) - 1, k).round().astype(int)] if k else far.iloc[:0]])
    xy = list(ex.map(lambda k: detector_xy(k, ra, dec), cand.key))
    cand["x"], cand["y"] = [p[0] for p in xy], [p[1] for p in xy]
    cand["d_spot"] = np.hypot(cand.x - spot[0], cand.y - spot[1])
    return cand.sort_values("d_spot")


def main():
    ap = argparse.ArgumentParser(description="False-discovery shield (see module docstring)")
    ap.add_argument("--ra", type=float)
    ap.add_argument("--dec", type=float)
    ap.add_argument("--name")
    ap.add_argument("--detector", type=int, default=1)
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--spot", type=float, nargs=2, default=SPOT)
    ap.add_argument("--others", nargs="*", default=[])
    ap.add_argument("--ref")
    ap.add_argument("--blank", type=int, default=0)
    ap.add_argument("--max", type=int, default=30)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.name:
        a.ra, a.dec = _resolve(a.name)
    spot = tuple(a.spot)
    idx = d1_index(a.detector)
    lim = HALF + 1
    with ThreadPoolExecutor(a.workers) as ex:
        tgt = _pictures(idx, a.ra, a.dec, spot, a.max if a.test else 0, ex).assign(star=a.name or "target")
        print(f"{len(tgt)} DR1 detector-{a.detector} pictures of RA {a.ra:.5f} Dec {a.dec:+.5f} "
              f"({'sampled' if a.test else 'all'}); distance from spot {spot}:")
        if not a.test:
            print(tgt[["obs_id", "mjd", "x", "y", "d_spot"]].to_string(index=False, float_format=lambda v: f"{v:.1f}"))
            if a.out:
                json.dump(dict(ra=a.ra, dec=a.dec, detector=a.detector, spot=spot,
                               pictures=tgt[["obs_id", "key", "mjd", "x", "y", "d_spot"]].to_dict("records")),
                          open(a.out, "w"), indent=1, default=float)
            return
        tabs = [tgt]
        for o in a.others:
            ra, dec = _resolve(o)
            tabs.append(_pictures(idx, ra, dec, spot, a.max, ex).assign(star=o))
        if a.blank:
            b = tgt[tgt.d_spot > 300].iloc[:a.blank].copy()
            b["x"], b["y"], b["d_spot"], b["star"] = spot[0], spot[1], 0.0, "blank spot (no bright star)"
            tabs.append(b)
        allp = pd.concat(tabs, ignore_index=True)
        allp = allp[(allp.x >= lim) & (allp.x <= NPIX - 1 - lim) & (allp.y >= lim) & (allp.y <= NPIX - 1 - lim)]
        allp = allp.reset_index(drop=True)
        imgs = list(ex.map(lambda r: cutout(r.key, r.x, r.y), allp.itertuples()))
    off = [(r.x - round(r.x), r.y - round(r.y)) for r in allp.itertuples()]
    allp["S"] = [ring_stat(im, *o)["S"] for im, o in zip(imgs, off)]
    allp["Z"] = [spike_stat(im, *o)["Z"] for im, o in zip(imgs, off)]
    P = [pattern(im, *o) for im, o in zip(imgs, off)]
    t = allp[allp.star == (a.name or "target")]
    iref = t.index[t.obs_id == a.ref][0] if a.ref else t.d_spot.idxmin()
    allp["match"] = [corr(P[i], P[iref]) if i != iref else np.nan for i in allp.index]
    print(f"reference picture: {allp.obs_id[iref]} (target at d = {allp.d_spot[iref]:.1f} px)")
    print(allp[["star", "obs_id", "x", "y", "d_spot", "S", "Z", "match"]].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    el = allp[(allp.d_spot > 150) & (allp.star != "blank spot (no bright star)")].match.dropna()
    med, sig = float(el.median()), float(1.4826 * np.median(np.abs(el - el.median())))
    summ = dict(reference=allp.obs_id[iref], elsewhere_match_median=med, elsewhere_match_robust_sigma=sig, n_elsewhere=int(len(el)))
    print(f"\nmatch elsewhere (> 150 px from spot, all stars): median {med:+.3f} ± {sig:.3f} (robust spread, n = {len(el)})")
    on = allp[(allp.d_spot < 15) & (allp.star != (a.name or "target")) & (allp.star != "blank spot (no bright star)")]
    if len(on):
        z = (on.match - med) / sig
        summ.update(other_stars_at_spot=[dict(star=r.star, obs_id=r.obs_id, d_spot=r.d_spot, match=r.match, n_sigma=zz)
                                         for r, zz in zip(on.itertuples(), z)])
        for r, zz in zip(on.itertuples(), z):
            print(f"  {r.star} at the spot ({r.obs_id}, d = {r.d_spot:.1f} px): match {r.match:+.3f} = {zz:+.1f} sigma")
        verdict = ("FOLLOWS THE DETECTOR: different stars at the same detector spot share the pattern (> 5 sigma)"
                   if (z > 5).any() else "no other star at the spot shares the pattern above 5 sigma")
    else:
        verdict = "no other star at the spot (< 15 px) among --others: the decisive test could not be run"
    bl = allp[allp.star == "blank spot (no bright star)"].match
    if len(bl):
        summ["blank_spot_match_median"] = float(bl.median())
        print(f"blank spot (same pixels, no bright star): median match {bl.median():+.3f} (n = {len(bl)})")
    print("VERDICT:", verdict)
    summ["verdict"] = verdict
    if a.out:
        json.dump(dict(ra=a.ra, dec=a.dec, detector=a.detector, spot=spot, summary=summ,
                       pictures=allp.drop(columns=["c_ra", "c_dec"], errors="ignore").to_dict("records")),
                  open(a.out, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


if __name__ == "__main__":
    main()
