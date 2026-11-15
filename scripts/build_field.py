"""Time Machine field: raw and wavelength-corrected blink of one sky patch, plus finding-1 change counts.

Method: docs/INTERFACES.md, "Wavelength-aware change model" and "Change model — revision 2" (orrery.change).
Pixels come from the public S3 bucket (one range request per IMAGE row block, two per FLAGS row block).

  .venv/bin/python scripts/build_field.py --name calib --ra 7.79 --dec -65.37 --size 48 --calibrate
  .venv/bin/python scripts/build_field.py --name barnard --ra 269.452077 --dec 4.693365 --size 48 \
      --target-name "Barnard's Star" --kind star --pmra -801.551 --pmdec 10362.394 --plx 546.976

--ra/--dec are ICRS at epoch J2000.0 when proper motions are given (mas/yr, pmra includes cos dec).
Outputs: site/data/timemachine/<name>.json + <name>/*.png, site/data/timemachine/index.json,
data/findings/f1_<name>.json, data/findings/contact_<name>.png; --calibrate writes data/findings/f1_calibration.json.
"""
import argparse, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from astropy.time import Time  # noqa: E402
from astropy.coordinates import get_body_barycentric  # noqa: E402
from orrery.change import (SCALE_ARCSEC, bilinear, change_model, find_changes, grid_world, project,  # noqa: E402
                           world_to_grid)
from orrery.data import BAD_BITS, FLAG_NAMES, Frame  # noqa: E402
from astropy.io.fits.hdu.compressed._codecs import Rice1  # noqa: E402

MASK_BITS = BAD_BITS | (1 << 19)          # revision 2: hard-bad + OUTLIER
J2000, GAIA_EPOCH = 51544.5, 57388.5      # MJD of J2000.0 and Gaia DR3 epoch J2016.0
CALIB = "data/findings/f1_calibration.json"
META_KEYS = ["PSF_FWHM", "L2DQAFLG", "SPS_EPA", "L2 N_OUTLIER", "L2 N_PERSIST", "L2 SATSTRKS", "L2 N_GHOST"]


# ---------------------------------------------------------------- astrometry
def position(ra, dec, pmra, pmdec, plx, epoch, mjd):
    """Linear proper motion from `epoch` (MJD) plus annual parallax (Earth barycentric position). Degrees."""
    yr = (np.asarray(mjd, float) - epoch) / 365.25
    da, dd = pmra * yr, pmdec * yr                              # mas
    if plx:
        e = get_body_barycentric("earth", Time(np.atleast_1d(mjd), format="mjd")).xyz.to_value("au")
        a, d = np.radians(ra), np.radians(dec)
        X, Y, Z = e
        da = da + plx * (X * np.sin(a) - Y * np.cos(a)).reshape(np.shape(yr))
        dd = dd + plx * (X * np.cos(a) * np.sin(d) + Y * np.sin(a) * np.sin(d) - Z * np.cos(d)).reshape(np.shape(yr))
    return ra + da / 3.6e6 / np.cos(np.radians(dec)), dec + dd / 3.6e6


# ---------------------------------------------------------------- index search
def load_index():
    full = "data/index/frames_dr1.parquet"
    d = "data/index/parts_dr1"
    parts = pd.concat([pd.read_parquet(f"{d}/{f}") for f in sorted(os.listdir(d))], ignore_index=True)
    df = pd.read_parquet(full) if os.path.exists(full) else parts
    src = full
    if len(parts) > len(df):
        df, src = parts, d
    if "error" in df:
        df = df[df["error"].isna()]
    return df[df["dqa"] == "Pass"].reset_index(drop=True), src


def candidates(df, ra, dec):
    a1, d1, a2, d2 = map(np.radians, (df.ra.values, df.dec.values, ra, dec))
    sep = np.degrees(2 * np.arcsin(np.sqrt(np.sin((d1 - d2) / 2) ** 2 + np.cos(d1) * np.cos(d2) * np.sin((a1 - a2) / 2) ** 2)))
    sub = df[sep < 2.8]
    keep = []
    for k, r in sub.iterrows():
        xi, eta = project(np.asarray(r.c_ra), np.asarray(r.c_dec), ra, dec)
        ang = np.argsort(np.arctan2(eta - eta.mean(), xi - xi.mean()))
        px, py = xi[ang], eta[ang]
        cross = [(px[(i + 1) % 4] - px[i]) * (0 - py[i]) - (py[(i + 1) % 4] - py[i]) * (0 - px[i]) for i in range(4)]
        if all(c > 0 for c in cross) or all(c < 0 for c in cross):
            keep.append(k)
    return sub.loc[keep]


# ---------------------------------------------------------------- pixel reads
def _get(url, a, b):
    for i in range(4):
        try:
            r = requests.get(url, headers={"Range": f"bytes={a}-{b - 1}"}, timeout=60)
            r.raise_for_status()
            return r.content
        except Exception:
            if i == 3:
                raise
            time.sleep(2 * (i + 1))


def read_rows(fr, y0, y1):
    """IMAGE float32 and FLAGS int32 rows [y0, y1) at full width."""
    im = fr.hdul["IMAGE"]
    img = np.frombuffer(_get(fr.url, im._data_offset + y0 * 8160, im._data_offset + y1 * 8160), ">f4").reshape(-1, 2040)
    fl = fr.hdul["FLAGS"]
    off = fl._data_offset
    desc = np.frombuffer(_get(fr.url, off + 8 * y0, off + 8 * y1), ">i4").reshape(-1, 2)   # (nbytes, heap offset)
    heap = off + 8 * 2040                                     # THEAP default = NAXIS1 * NAXIS2
    lo, hi = int(desc[:, 1].min()), int((desc[:, 1] + desc[:, 0]).max())
    buf = _get(fr.url, heap + lo, heap + hi)
    codec = Rice1(blocksize=32, bytepix=4, tilesize=2040)
    flg = np.stack([codec.decode(np.frombuffer(buf[o - lo:o - lo + n], np.uint8)) for n, o in desc]).astype(np.int32)
    return img.astype(np.float32), flg


def cut(key, gra, gdec):
    """Resample one frame onto the grid. Returns None if the grid centre is off the frame."""
    fr = Frame(key)
    S = gra.shape[0]
    x, y = fr.wcs.all_world2pix(gra.ravel(), gdec.ravel(), 0)
    x, y = x.reshape(S, S), y.reshape(S, S)
    c = S // 2
    if not (0 <= x[c, c] <= 2039 and 0 <= y[c, c] <= 2039):
        return None
    y0, y1 = max(0, int(np.floor(np.nanmin(y))) - 1), min(2040, int(np.ceil(np.nanmax(y))) + 2)
    img, flg = read_rows(fr, y0, y1)
    v_raw = bilinear(img, x, y - y0)
    bad = np.where(flg & MASK_BITS, np.nan, 0.0)
    v = v_raw + bilinear(bad, x, y - y0)                    # any masked neighbour -> NaN
    lam, _ = fr.wave_at(x, y)
    lam = np.where(np.isfinite(v_raw), lam, np.nan)
    ix = np.clip(np.round(x).astype(int), 0, 2039)
    iy = np.clip(np.round(y).astype(int) - y0, 0, y1 - y0 - 1)
    inside = np.isfinite(v_raw)
    fg = flg[iy, ix][inside]
    flags = {FLAG_NAMES.get(b, str(b)): int(((fg >> b) & 1).sum()) for b in range(30) if ((fg >> b) & 1).any()}
    meta = {k: fr.header.get(k) for k in META_KEYS}
    meta = {k: (float(v_) if isinstance(v_, (int, float)) and not isinstance(v_, bool) else v_) for k, v_ in meta.items()}
    out = dict(key=key, obs_id=fr.obs_id, detector=fr.detector, mjd=fr.mjd, v=v.astype(np.float32),
               v_raw=v_raw.astype(np.float32), lam=lam.astype(np.float32), flags=flags, meta=meta, url=fr.url)
    fr.close()
    return out


def gather(name, ra0, dec0, S, threads=8):
    cache = f"data/cache/fields/{name}_S{S}.npz"
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        print(f"cache {cache}: {len(z['mjd'])} frames")
        return {k: z[k] for k in z.files}
    df, src = load_index()
    cand = candidates(df, ra0, dec0)
    print(f"index {src}: {len(df)} frames, {len(cand)} polygon candidates")
    gra, gdec = grid_world(S, ra0, dec0)
    res, fails = [], []
    t = time.time()
    with ThreadPoolExecutor(threads) as ex:
        futs = {ex.submit(cut, k, gra, gdec): k for k in cand.key}
        for i, fu in enumerate(futs):
            try:
                r = fu.result()
                if r is not None:
                    res.append(r)
            except Exception as e:
                fails.append((futs[fu], repr(e)))
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(futs)} read, {time.time() - t:.0f}s", flush=True)
    if fails:
        print(f"WARNING {len(fails)} read failures, e.g. {fails[0]}")
    res.sort(key=lambda r: (r["mjd"], r["detector"]))
    d = {k: np.array([r[k] for r in res]) for k in ("key", "obs_id", "detector", "mjd", "v", "v_raw", "lam", "url")}
    d["flags"] = np.array([json.dumps(r["flags"]) for r in res])
    d["meta"] = np.array([json.dumps(r["meta"]) for r in res])
    d["n_candidates"] = np.array(len(cand))
    d["n_fail"] = np.array(len(fails))
    d["index_src"] = np.array(src)
    np.savez_compressed(cache, **d)
    return d


def gaia(name, ra0, dec0, S):
    path = f"data/cache/fields/{name}_S{S}_gaia.json"
    if os.path.exists(path):
        return pd.read_json(path)
    import pyvo
    r = S * SCALE_ARCSEC / 2 * 1.5 / 3600
    q = (f"SELECT source_id, ra, dec, pmra, pmdec, parallax, phot_g_mean_mag FROM gaiadr3.gaia_source "
         f"WHERE 1=CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra0}, {dec0}, {r})) AND phot_g_mean_mag < 19")
    t = pyvo.dal.TAPService("https://gaia.aip.de/tap").run_sync(q, maxrec=100000).to_table().to_pandas()
    t.to_json(path)
    return t


def passes(mjd, gap=30):
    cuts = np.flatnonzero(np.diff(mjd) > gap) + 1
    return np.split(np.arange(len(mjd)), cuts)


# ---------------------------------------------------------------- calibration of f
def star_pixels(g, mjd_ref, S, ra0, dec0, gmax=16):
    s = g[g.phot_g_mean_mag < gmax]
    ra, dec = position(s.ra.values, s.dec.values, np.nan_to_num(s.pmra.values), np.nan_to_num(s.pmdec.values), 0,
                       GAIA_EPOCH, mjd_ref)
    i, j = world_to_grid(ra, dec, S, ra0, dec0)
    jj, ii = np.mgrid[0:S, 0:S]
    m = np.zeros((S, S), bool)
    for a, b in zip(i, j):
        m |= np.hypot(ii - a, jj - b) <= 1.0
    return m, len(s)


def calibrate(d, g, S, ra0, dec0, target=1.0):
    m, nstar = star_pixels(g, d["mjd"].mean(), S, ra0, dec0)
    def med(f):
        r = change_model(d["v"], d["lam"], d["detector"], d["mjd"], f=f)
        return float(np.nanmedian(r["corr"][:, m] ** 2))
    lo, hi = 1e-4, 1.0
    if med(0.0) <= target:
        return 0.0, nstar, int(m.sum()), med(0.0)
    for _ in range(40):
        mid = np.sqrt(lo * hi)
        lo, hi = (mid, hi) if med(mid) > target else (lo, mid)
    return float(hi), nstar, int(m.sum()), med(hi)


# ---------------------------------------------------------------- images
def asinh_png(v, path):
    """Per-frame asinh stretch, identical to site/js/live/grid.js asinhRGBA."""
    from PIL import Image
    f = np.sort(v[np.isfinite(v)])
    q = lambda t: f[min(len(f) - 1, int(np.floor(t * (len(f) - 1))))] if len(f) else 0.0
    lo, sd = q(0.5), max(1e-12, (q(0.84) - q(0.16)) / 2)
    hi = max(q(0.995) - lo, 3 * sd)
    g = 255 * np.clip(np.arcsinh((v - lo) / sd) / np.arcsinh(hi / sd), 0, 1)
    rgb = np.stack([g, g, g], -1)
    rgb[~np.isfinite(v)] = (30, 34, 48)
    Image.fromarray(rgb.astype(np.uint8)).save(path, optimize=True)


def diverging_png(v, path, clip=8):
    """Blue (#56B4E9) <- black -> orange (#E69F00), clipped at +-8; identical to grid.js divergingRGBA."""
    from PIL import Image
    t = np.clip(np.abs(np.nan_to_num(v)) / clip, 0, 1)[..., None]
    c = np.where((v < 0)[..., None], np.array([86, 180, 233.0]), np.array([230, 159, 0.0]))
    rgb = c * t
    rgb[~np.isfinite(v)] = (30, 34, 48)
    Image.fromarray(np.round(rgb).astype(np.uint8)).save(path, optimize=True)


def contact_sheet(name, title, d, r, tpos, path, n_per_pass=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    P = passes(d["mjd"])
    sel = [p[np.linspace(0, len(p) - 1, min(n_per_pass, len(p))).astype(int)] for p in P]
    rows = [e for s in sel for e in s]
    fig, ax = plt.subplots(len(rows), 3, figsize=(6.6, 2.2 * len(rows)))
    ax = np.atleast_2d(ax)
    for k, e in enumerate(rows):
        vs = r["vs"][e]
        lo, hi = np.nanpercentile(vs, [50, 99.5])
        sd = max(1e-9, np.nanpercentile(vs, 84) - np.nanpercentile(vs, 16)) / 2
        ax[k, 0].imshow(np.arcsinh((vs - lo) / sd), cmap="gray", vmin=0, vmax=np.arcsinh(max(hi - lo, 3 * sd) / sd))
        nv = r["naive_all"][e]
        ax[k, 1].imshow(nv if nv is not None else np.full_like(vs, np.nan), cmap="PuOr_r", vmin=-8, vmax=8)
        ax[k, 2].imshow(r["corr"][e], cmap="PuOr_r", vmin=-8, vmax=8)
        for a in ax[k]:
            a.set_xticks([]); a.set_yticks([])
            if tpos is not None:
                a.plot(*tpos[e], "+", color="lime", ms=10, mew=0.8)
        ax[k, 0].set_ylabel(f"{Time(d['mjd'][e], format='mjd').iso[:10]}\nD{d['detector'][e]} "
                            f"{np.nanmean(d['lam'][e]):.2f}um", fontsize=7)
        ax[k, 1].set_title(f"naive(all) n={r['n_naive_all'][e]}", fontsize=7)
        ax[k, 2].set_title(f"corrected n={r['n_corr'][e]}", fontsize=7)
    ax[0, 0].set_title("raw (asinh)", fontsize=7)
    fig.suptitle(f"{title}: {len(d['mjd'])} frames, {len(P)} passes (PuOr, +-8 sigma)", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------- target diagnostics
def target_check(d, r, tpos, P):
    """Per pass: mean measured centroid offset of the target (5x5 first moment of the background-subtracted
    image about the predicted mean position) and the predicted mean position; 3x3 data/model flux ratio."""
    S = r["vs"].shape[1]
    ref = np.nanmean(tpos, 0)
    cx, cy = int(round(ref[0])), int(round(ref[1]))
    out = []
    for p in P:
        cen, rat, cm = [], [], []
        for e in p:
            w = r["vs"][e, cy - 2:cy + 3, cx - 2:cx + 3]
            sn = np.nansum(w) / (r["sigma"][e] * 5)
            if np.isfinite(w).all() and sn > 10:
                ww = np.clip(w, 0, None)
                jj, ii = np.mgrid[cy - 2:cy + 3, cx - 2:cx + 3]
                cen.append(((ww * ii).sum() / ww.sum(), (ww * jj).sum() / ww.sum()))
            a, m = r["vs"][e, cy - 1:cy + 2, cx - 1:cx + 2], r["model"][e, cy - 1:cy + 2, cx - 1:cx + 2]
            if np.isfinite(a).all() and np.isfinite(m).all() and m.sum() > 5 * r["sigma"][e] * 3:
                rat.append(a.sum() / m.sum())
            c = r["corr"][e, cy - 1:cy + 2, cx - 1:cx + 2]
            if np.isfinite(c).any():
                cm.append(float(np.nanmean(c)))
        cen = np.array(cen)
        rec = dict(n=len(p), mjd_mean=float(d["mjd"][p].mean()),
                   predicted_xy=np.nanmean(tpos[p], 0).round(3).tolist(),
                   n_centroid=int(len(cen)), corr_mean_3x3_median=float(np.median(cm)) if cm else None,
                   flux_ratio_data_over_model_median=float(np.median(rat)) if rat else None,
                   flux_ratio_n=len(rat))
        if len(cen) >= 3:
            rec["measured_xy"] = cen.mean(0).round(3).tolist()
            rec["measured_xy_err"] = (cen.std(0, ddof=1) / np.sqrt(len(cen))).round(3).tolist()
        out.append(rec)
    res = {"per_pass": out, "box_centre_xy": [cx, cy]}
    ok = [o for o in out if "measured_xy" in o]
    if len(ok) >= 2:
        a, b = ok[0], ok[-1]
        res["shift_measured_px"] = (np.subtract(b["measured_xy"], a["measured_xy"])).round(3).tolist()
        res["shift_measured_err_px"] = np.hypot(a["measured_xy_err"], b["measured_xy_err"]).round(3).tolist()
        res["shift_predicted_px"] = (np.subtract(b["predicted_xy"], a["predicted_xy"])).round(3).tolist()
    return res


# ---------------------------------------------------------------- main
def main(a):
    S = a.size
    has_t = a.target_name is not None
    # grid centre: target propagated to mid-DR1 (MJD 60982; one fixed grid for all frames)
    if has_t:
        ra0, dec0 = position(a.ra, a.dec, a.pmra, a.pmdec, a.plx, J2000, 60982.0)
        ra0, dec0 = float(np.ravel(ra0)[0]), float(np.ravel(dec0)[0])
    else:
        ra0, dec0 = a.ra, a.dec
    d = gather(a.name, ra0, dec0, S)
    N = len(d["mjd"])
    print(f"{N} frames on the grid (of {int(d['n_candidates'])} candidates, {int(d['n_fail'])} read failures)")
    if N == 0:
        sys.exit("no frames")
    g = gaia(a.name, ra0, dec0, S)
    os.makedirs("data/findings", exist_ok=True)

    if a.calibrate:
        f, nstar, npix, medchi2 = calibrate(d, g, S, ra0, dec0, 1.0)
        f_gauss, _, _, med_g = calibrate(d, g, S, ra0, dec0, 0.4549)
        rec = dict(f=f, field=a.name, ra=ra0, dec=dec0, size_px=S, n_frames=N, n_gaia_g_lt_16=nstar,
                   n_star_pixels=npix, median_corr_chi2_at_f=medchi2,
                   rule="f such that the median corrected chi^2 over grid pixels within 1 px of Gaia DR3 G<16 stars "
                        "equals 1 (docs/INTERFACES.md, revision 2)",
                   note_gaussian_median=dict(f=f_gauss, median_chi2=med_g,
                                             comment="for reference only: f that would make the median chi^2 equal "
                                                     "0.455, the median of a 1-dof chi^2; not used"),
                   frozen_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), release="dr1",
                   index_src=str(d["index_src"]))
        if os.path.exists(CALIB):
            sys.exit(f"{CALIB} exists (frozen); refusing to overwrite")
        json.dump(rec, open(CALIB, "w"), indent=1)
        print(json.dumps(rec, indent=1))
        return

    f = json.load(open(CALIB))["f"]
    r = change_model(d["v"], d["lam"], d["detector"], d["mjd"], f=f)
    P = passes(d["mjd"])

    tpos = None
    if has_t:
        tra, tdec = position(a.ra, a.dec, a.pmra, a.pmdec, a.plx, J2000, d["mjd"])
        tpos = np.stack(world_to_grid(tra, tdec, S, ra0, dec0), -1)
    pm = np.hypot(g.pmra.fillna(0), g.pmdec.fillna(0))
    hp = g[pm > 100]
    if has_t:
        tra16, tdec16 = position(a.ra, a.dec, a.pmra, a.pmdec, 0, J2000, GAIA_EPOCH)
        hp = hp[np.hypot((hp.ra - tra16) * np.cos(np.radians(dec0)), hp.dec - tdec16) * 3600 > 15]
    hpos = []
    for _, s in hp.iterrows():
        sra, sdec = position(s.ra, s.dec, s.pmra, s.pmdec, s.parallax if np.isfinite(s.parallax) else 0, GAIA_EPOCH, d["mjd"])
        hpos.append((f"Gaia DR3 {s.source_id}", np.stack(world_to_grid(sra, sdec, S, ra0, dec0), -1)))

    out_dir = f"site/data/timemachine/{a.name}"
    os.makedirs(out_dir, exist_ok=True)
    frames, n_known, n_unexp = [], 0, 0
    for e in range(N):
        labels = []
        known = []
        if tpos is not None:
            known.append(dict(name=a.target_name, kind=a.kind, label="variable star" if a.kind == "variable star" else "target",
                              xy=tpos[e]))
        for nm, xy in hpos:
            known.append(dict(name=nm, kind="star", label="known star", xy=xy[e]))
        ch = find_changes(r["corr"][e])
        used = set()
        for k in known:
            m = [c for c in ch if np.hypot(c["x"] - k["xy"][0], c["y"] - k["xy"][1]) <= 2]
            used |= {id(c) for c in m}
            pk = max((c["peak"] for c in m), key=abs, default=None)
            if -1 <= k["xy"][0] <= S and -1 <= k["xy"][1] <= S:
                labels.append(dict(x=round(float(k["xy"][0]), 2), y=round(float(k["xy"][1]), 2), name=k["name"],
                                   kind=k["kind"], label=k["label"], confidence=None, change=bool(m),
                                   peak=round(pk, 2) if pk is not None else None))
            n_known += len(m)
        for c in ch:
            if id(c) in used:
                continue
            n_unexp += 1
            labels.append(dict(x=c["x"], y=c["y"], name=None, kind="unknown", label="unexplained",
                               confidence=round(float(np.clip(0.5 + (abs(c["peak"]) - 5) / 20, 0, 1)), 3),
                               change=True, peak=round(c["peak"], 2), npix=c["npix"]))
        raw_p, corr_p = f"{out_dir}/raw_{e:03d}.png", f"{out_dir}/corr_{e:03d}.png"
        asinh_png(d["v_raw"][e], raw_p)
        diverging_png(r["corr"][e], corr_p)
        rel = lambda p: os.path.relpath(p, "site/data")
        frames.append(dict(obs_id=str(d["obs_id"][e]), mjd=round(float(d["mjd"][e]), 6),
                           date=Time(d["mjd"][e], format="mjd").iso[:10], detector=int(d["detector"][e]),
                           wl_um=round(float(d["lam"][e][S // 2, S // 2]), 4) if np.isfinite(d["lam"][e][S // 2, S // 2])
                           else round(float(np.nanmean(d["lam"][e])), 4),
                           pass_index=int(next(i for i, p in enumerate(P) if e in p)),
                           img_png=rel(raw_p), corr_png=rel(corr_p), resid_png=rel(corr_p), labels=labels,
                           n_changes_corr=int(r["n_corr"][e]), n_changes_naive_all=int(r["n_naive_all"][e]),
                           n_changes_naive_det=int(r["n_naive_det"][e]),
                           flags=json.loads(str(d["flags"][e])), meta=json.loads(str(d["meta"][e])),
                           s3_url=str(d["url"][e])))
    title = a.title or a.target_name or a.name
    field = dict(field=a.name, name=title, ra=round(ra0, 6), dec=round(dec0, 6), size_px=S, scale_arcsec=SCALE_ARCSEC,
                 release="dr1", noise_floor_f=f, method="docs/INTERFACES.md#wavelength-aware-change-model (revision 2)",
                 corr_colormap="blue #56B4E9 (negative) - black - orange #E69F00 (positive), clipped at +-8 sigma",
                 confidence_rule="unexplained: clip(0.5 + (|peak| - 5)/20, 0, 1)",
                 rerun_command=" ".join(["scripts/build_field.py"] + sys.argv[1:]),
                 n_passes=len(P), frames=frames)
    if has_t:
        field["target"] = dict(name=a.target_name, kind=a.kind, ra_j2000=a.ra, dec_j2000=a.dec, pmra=a.pmra,
                               pmdec=a.pmdec, plx=a.plx, source=a.source)
    json.dump(field, open(f"site/data/timemachine/{a.name}.json", "w"), separators=(",", ":"))
    ip = "site/data/timemachine/index.json"
    idx = json.load(open(ip)) if os.path.exists(ip) else []
    idx = [x for x in idx if x["field"] != a.name] + [dict(field=a.name, name=title, ra=round(ra0, 6), dec=round(dec0, 6))]
    json.dump(idx, open(ip, "w"), indent=1)

    by_det = {}
    for dd in sorted(set(d["detector"].tolist())):
        m = d["detector"] == dd
        by_det[str(dd)] = dict(n_frames=int(m.sum()), naive_all=int(r["n_naive_all"][m].sum()),
                               naive_det=int(r["n_naive_det"][m].sum()), corrected=int(r["n_corr"][m].sum()))
    f1 = dict(field=a.name, name=title, ra=ra0, dec=dec0, size_px=S, n_frames=N, n_candidates=int(d["n_candidates"]),
              n_read_failures=int(d["n_fail"]), n_passes=len(P),
              passes=[dict(n=len(p), start=Time(d["mjd"][p[0]], format="mjd").iso[:10],
                           end=Time(d["mjd"][p[-1]], format="mjd").iso[:10]) for p in P],
              f=f, n_naive_all=int(r["n_naive_all"].sum()), n_naive_det=int(r["n_naive_det"].sum()),
              n_corrected=int(r["n_corr"].sum()), corrected_matched_known=n_known, corrected_unexplained=n_unexp,
              n_gaia_hpm=len(hpos), by_detector=by_det, index_src=str(d["index_src"]),
              method="docs/INTERFACES.md#wavelength-aware-change-model (revision 2)")
    if tpos is not None:
        f1["target_check"] = target_check(d, r, tpos, P)
    json.dump(f1, open(f"data/findings/f1_{a.name}.json", "w"), indent=1)
    contact_sheet(a.name, title, d, r, tpos, f"data/findings/contact_{a.name}.png")
    print(json.dumps({k: v for k, v in f1.items() if k != "target_check"}, indent=1))
    if tpos is not None:
        print(json.dumps(f1["target_check"], indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--ra", type=float, required=True)
    ap.add_argument("--dec", type=float, required=True)
    ap.add_argument("--size", type=int, default=48)
    ap.add_argument("--title")
    ap.add_argument("--target-name")
    ap.add_argument("--kind", default="star")
    ap.add_argument("--pmra", type=float, default=0.0)
    ap.add_argument("--pmdec", type=float, default=0.0)
    ap.add_argument("--plx", type=float, default=0.0)
    ap.add_argument("--source", default="")
    ap.add_argument("--calibrate", action="store_true")
    main(ap.parse_args())
