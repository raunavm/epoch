"""E18 NEP deep field extraction (docs/protocols/E18.md). Run from the repo root:
  .venv/bin/python scripts/run_e18.py catalogue   # -> data/phase2/e18/sources.parquet
  .venv/bin/python scripts/run_e18.py frames      # -> data/phase2/e18/frames.parquet (candidates, tiers, order)
  .venv/bin/python scripts/run_e18.py phot        # -> data/phase2/e18/phot/part-XXXX.parquet (resumable)
"""
import os, sys, time, glob, json
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT = "data/phase2/e18"
RA0, DEC0, R, MARGIN = 270.0, 66.56, 0.40, 0.06
PIX = 6.15 / 3600
SEED = 20261115
CUTOFF = "17:15"
CORE, PAD = 32, 8
BLEND = 3.0 / 3600


def sep_deg(ra1, dec1, ra2, dec2):
    a1, d1, a2, d2 = map(np.radians, (ra1, dec1, ra2, dec2))
    return np.degrees(2 * np.arcsin(np.sqrt(np.sin((d1 - d2) / 2) ** 2 + np.cos(d1) * np.cos(d2) * np.sin((a1 - a2) / 2) ** 2)))


# ------------------------------------------------------------------ catalogue
def catalogue():
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    g = pd.read_parquet(f"{OUT}/gaia_raw.parquet")      # AIP TAP query, radius R + 0.16 deg, G < 19.5
    w = pd.read_parquet(f"{OUT}/allwise_raw.parquet")   # CfA VizieR II/328/allwise, radius R + 0.06, W1 < 16
    g = g[sep_deg(g.ra, g.dec, RA0, DEC0) < R + MARGIN].reset_index(drop=True)
    dt = 2010.5 - 2016.0                                # AllWISE mean epoch ~ J2010.5
    gdec = g.dec + g.pmdec.fillna(0) * dt / 3.6e6
    gra = g.ra + g.pmra.fillna(0) * dt / 3.6e6 / np.cos(np.radians(g.dec))
    cg, cw = SkyCoord(gra.values * u.deg, gdec.values * u.deg), SkyCoord(w.RAJ2000.values * u.deg, w.DEJ2000.values * u.deg)
    i, d2, _ = cw.match_to_catalog_sky(cg)
    has_gaia = d2.arcsec < 2.0
    # WISE / 2MASS photometry for Gaia sources: nearest AllWISE within 2" (each WISE row used once, nearest Gaia)
    j, d2g, _ = cg.match_to_catalog_sky(cw)
    okg = (d2g.arcsec < 2.0) & (i[j] == np.arange(len(g)))
    cols = ["W1mag", "e_W1mag", "W2mag", "e_W2mag", "W3mag", "W4mag", "Jmag", "Hmag", "Kmag"]
    gs = pd.DataFrame(dict(id=g.source_id.astype(str), kind="gaia", ra=g.ra, dec=g.dec, pmra=g.pmra, pmdec=g.pmdec,
                           G=g.phot_g_mean_mag, bp_rp=g.bp_rp, parallax=g.parallax, parallax_error=g.parallax_error,
                           ruwe=g.ruwe, phot_variable_flag=g.phot_variable_flag,
                           classprob_quasar=g.classprob_dsc_combmod_quasar, classprob_galaxy=g.classprob_dsc_combmod_galaxy,
                           classprob_star=g.classprob_dsc_combmod_star, vari_class=g.best_class_name.where(g.best_class_name.fillna("") != "", None),
                           vari_class_score=g.best_class_score))
    gs["allwise"] = np.where(okg, w.AllWISE.values[j], None)
    for c in cols:
        gs[c] = np.where(okg, w[c].values[j], np.nan)
    wo = w[~has_gaia].reset_index(drop=True)
    ws = pd.DataFrame(dict(id="AllWISE " + wo.AllWISE, kind="allwise_only", ra=wo.RAJ2000, dec=wo.DEJ2000, allwise=wo.AllWISE))
    for c in cols:
        ws[c] = wo[c].values
    s = pd.concat([gs, ws], ignore_index=True)
    s["ccf"] = None
    s.loc[s.kind == "allwise_only", "ccf"] = wo.ccf.values
    s["r_deg"] = sep_deg(s.ra, s.dec, RA0, DEC0)
    s["in_region"] = s.r_deg < R
    s["rank_mag"] = np.where(s.kind == "gaia", s.G, s.W1mag + 3.0)   # ordering only (blend primary, brighter neighbour)
    s = s.sort_values("rank_mag", kind="stable").reset_index(drop=True)
    c = SkyCoord(s.ra.values * u.deg, s.dec.values * u.deg)
    # blend groups: friends-of-friends at 3", primary = brightest member
    a, b, _, _ = c.search_around_sky(c, BLEND * u.deg)
    par = np.arange(len(s))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for p, q in zip(a, b):
        rp, rq = find(p), find(q)
        if rp != rq:
            par[max(rp, rq)] = min(rp, rq)                 # rows sorted by brightness -> root = brightest
    root = np.array([find(x) for x in range(len(s))])
    s["group"] = root
    s["blend_n"] = pd.Series(root).map(pd.Series(root).value_counts()).values
    s["is_primary"] = root == np.arange(len(s))
    # nearest brighter catalogue neighbour (rows sorted by rank_mag), in arcsec; and nearest any within 15"
    a, b, d, _ = c.search_around_sky(c, 60 * u.arcsec)
    m = b < a
    nb = pd.Series(d.arcsec[m]).groupby(a[m]).min()
    s["d_brighter_arcsec"] = nb.reindex(range(len(s))).values
    m = a != b
    s["d_nearest_arcsec"] = pd.Series(d.arcsec[m]).groupby(a[m]).min().reindex(range(len(s))).values
    s["idx"] = np.arange(len(s))
    s.to_parquet(f"{OUT}/sources.parquet", index=False)
    print(s.kind.value_counts(), s[s.in_region].kind.value_counts(), (s.in_region & s.is_primary).sum(), "reported groups")


def kim():
    """Add neowise_var_kim2026: AllWISE/Gaia source within 2" of a Kim M. et al. 2026 (arXiv:2604.05332) NEP NEOWISE
    variable (their tab1.tex in the arXiv source, parsed to kim2026_neowise_var.parquet; P_var > 0.95 in W1 and W2, r > 0.7)."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    s = pd.read_parquet(f"{OUT}/sources.parquet")
    k = pd.read_parquet(f"{OUT}/kim2026_neowise_var.parquet")
    k = k[k.dec > 0]
    ck = SkyCoord(k.ra.values * u.deg, k.dec.values * u.deg)
    # WISE-epoch positions: AllWISE coordinates where matched, else Gaia J2016 (pm offset < 1" for almost all)
    i, d2, _ = SkyCoord(s.ra.values * u.deg, s.dec.values * u.deg).match_to_catalog_sky(ck)
    s["neowise_var_kim2026"] = d2.arcsec < 2.0
    s["kim2026_name"] = np.where(s.neowise_var_kim2026, k.name.values[i], None)
    # Milliquas (VizieR VII/294 via the CfA mirror, cone R + 0.06 deg): nearest within 2"
    q = pd.read_parquet(f"{OUT}/milliquas_raw.parquet")
    i, d2, _ = SkyCoord(s.ra.values * u.deg, s.dec.values * u.deg).match_to_catalog_sky(
        SkyCoord(q.RAJ2000.values * u.deg, q.DEJ2000.values * u.deg))
    ok = d2.arcsec < 2.0
    s["milliquas_name"] = np.where(ok, q.Name.values[i], None)
    s["milliquas_type"] = np.where(ok, q.Type.values[i], None)
    s["milliquas_z"] = np.where(ok, q.z.values[i], np.nan)
    s.to_parquet(f"{OUT}/sources.tmp", index=False)
    os.replace(f"{OUT}/sources.tmp", f"{OUT}/sources.parquet")
    print(ok.sum(), "Milliquas matches;", s.neowise_var_kim2026.sum(), (s.neowise_var_kim2026 & s.in_region).sum(), "in region;",
          len(k), "Kim NEP rows")


# ------------------------------------------------------------------ frames
def frames():
    df = pd.read_parquet("data/index/frames_dr1.parquet")
    df = df[(df.dqa == "Pass") & (sep_deg(df.ra, df.dec, RA0, DEC0) < 1.6)].copy()
    df["bin10"] = np.floor((df.mjd - 60789) / 10).astype(int)
    df = df.sort_values(["detector", "bin10", "mjd"], kind="stable")
    df["k"] = df.groupby(["detector", "bin10"]).cumcount()
    df["tier"] = np.where(df.k % 4 == 0, 0, np.where(df.k % 4 == 2, 1, 2))
    rng = np.random.default_rng(SEED)
    df["rnd"] = rng.permutation(len(df))
    df = df.sort_values(["tier", "rnd"], kind="stable").reset_index(drop=True)
    df["order"] = np.arange(len(df))
    df[["order", "key", "obs_id", "detector", "mjd", "ra", "dec", "psf_fwhm", "bin10", "k", "tier"]].to_parquet(
        f"{OUT}/frames.parquet", index=False)
    print(len(df), df.tier.value_counts().sort_index().to_dict())


# ------------------------------------------------------------------ pixels
def read_rows(key, half):
    """Header + rows covering the NEP +- half px (IMAGE, FLAGS, VARIANCE), columns cropped; None if NEP off frame.
    Same byte layout and reads as e2_cache.fast_stamp, without the per-pixel wavelength map."""
    from astropy.io.fits.hdu.compressed._codecs import Rice1
    from astropy.wcs import WCS
    from e2_cache import BUCKET, _get, _header, _pad
    url = f"{BUCKET}/{key}"
    _, o = _header(url, 0)
    h, img_off = _header(url, o)
    assert h["EXTNAME"] == "IMAGE" and h["BITPIX"] == -32 and h["NAXIS1"] == 2040 and "BZERO" not in h
    wcs = WCS(h)
    x, y = wcs.all_world2pix([RA0], [DEC0], 0)
    x, y = float(x[0]), float(y[0])
    if not (-half <= x <= 2039 + half and -half <= y <= 2039 + half):
        return None
    ix, iy = int(round(x)), int(round(y))
    x0, y0, x1, y1 = max(0, ix - half), max(0, iy - half), min(2040, ix + half + 1), min(2040, iy + half + 1)
    if x1 - x0 < 10 or y1 - y0 < 10:
        return None
    img = np.frombuffer(_get(url, img_off + y0 * 8160, img_off + y1 * 8160), ">f4").reshape(-1, 2040)
    fh, fl_off = _header(url, img_off + _pad(2040 * 2040 * 4))
    assert fh["EXTNAME"] == "FLAGS" and fh["ZCMPTYPE"] == "RICE_1" and fh["ZTILE1"] == 2040 and fh["ZTILE2"] == 1
    desc = np.frombuffer(_get(url, fl_off + 8 * y0, fl_off + 8 * y1), ">i4").reshape(-1, 2)
    heap = fl_off + 8 * 2040
    lo, hi = int(desc[:, 1].min()), int((desc[:, 1] + desc[:, 0]).max())
    buf = _get(url, heap + lo, heap + hi)
    codec = Rice1(blocksize=fh.get("ZVAL1", 32), bytepix=4, tilesize=2040)
    flg = np.stack([codec.decode(np.frombuffer(buf[o_ - lo:o_ - lo + n], np.uint8)) for n, o_ in desc]).astype(np.int32)
    vh, var_off = _header(url, fl_off + _pad(8 * 2040 + fh["PCOUNT"]))
    assert vh["EXTNAME"] == "VARIANCE" and vh["BITPIX"] == -32 and "BZERO" not in vh and "BSCALE" not in vh
    var = np.frombuffer(_get(url, var_off + y0 * 8160, var_off + y1 * 8160), ">f4").reshape(-1, 2040)
    return dict(h=h, wcs=wcs, x0=x0, y0=y0, img=np.ascontiguousarray(img[:, x0:x1], np.float32),
                var=np.ascontiguousarray(var[:, x0:x1], np.float32), flg=np.ascontiguousarray(flg[:, x0:x1]))


# ------------------------------------------------------------------ photometry
_S = None


def _sources():
    global _S
    if _S is None:
        _S = pd.read_parquet(f"{OUT}/sources.parquet")
    return _S


def psf_boxes(det, xs, ys):
    """Per source: 7x7 normalised ePSF box values and the box's integer origin (full-frame coords)."""
    from orrery.e1 import epsf_zone
    from orrery.psf import psf_native
    n = len(xs)
    bx0 = np.ceil(xs - 3).astype(int)
    by0 = np.ceil(ys - 3).astype(int)
    out = np.zeros((n, 7, 7))
    for k in range(n):
        ep, ovs = epsf_zone(det, xs[k], ys[k])
        gx = bx0[k] + np.arange(7)[None, :] + 0 * np.arange(7)[:, None]
        gy = by0[k] + np.arange(7)[:, None] + 0 * np.arange(7)[None, :]
        inb = (gx <= xs[k] + 3) & (gy <= ys[k] + 3)          # box = pixel centres within |d| <= 3
        v = np.where(inb, psf_native(ep, gx - xs[k], gy - ys[k], ovs), 0.0)
        out[k] = v / v.sum()
    return out, bx0, by0


def measure(key, fr=None):
    from orrery.data import _calib
    from orrery.e7 import PHOT_BITS
    from orrery.fit import flux_mJy
    from scipy.interpolate import RegularGridInterpolator
    S = _sources()
    if fr is None:
        fr = read_rows(key, HALF)
    if fr is None:
        return None, "off_frame"
    h, wcs = fr["h"], fr["wcs"]
    mjd, det, omega = float(h["MJD-AVG"]), int(h["DETECTOR"]), float(h["OMEGA_MEDIAN"])
    yr = (mjd - 57388.5) / 365.25
    dec = S.dec.values + np.nan_to_num(S.pmdec.values) * yr / 3.6e6
    ra = S.ra.values + np.nan_to_num(S.pmra.values) * yr / 3.6e6 / np.cos(np.radians(S.dec.values))
    X, Y = wcs.all_world2pix(ra, dec, 0)
    img, var, flg = fr["img"], fr["var"], fr["flg"]
    H, W = img.shape
    x0, y0 = fr["x0"], fr["y0"]
    xs, ys = X - x0, Y - y0
    on = (xs > -4) & (xs < W + 3) & (ys > -4) & (ys < H + 3)
    idx = np.flatnonzero(on)
    boxes, bx0, by0 = psf_boxes(det, X[idx], Y[idx])
    bx0, by0 = bx0 - x0, by0 - y0
    good = ((flg & PHOT_BITS) == 0) & np.isfinite(var) & (var > 0) & np.isfinite(img)
    grp = S.group.values[idx]
    prim = S.is_primary.values[idx] & S.in_region.values[idx]
    sx, sy = xs[idx], ys[idx]
    rows = []
    # tiles over the bounding box of reported primaries
    if not prim.any():
        return None, "no_sources"
    xa, xb = int(np.floor(sx[prim].min())), int(np.ceil(sx[prim].max())) + 1
    ya, yb = int(np.floor(sy[prim].min())), int(np.ceil(sy[prim].max())) + 1
    for ty in range(ya, yb, CORE):
        for tx in range(xa, xb, CORE):
            core_p = prim & (sx >= tx - 0.5) & (sx < tx + CORE - 0.5) & (sy >= ty - 0.5) & (sy < ty + CORE - 0.5)
            if not core_p.any():
                continue
            wx0, wy0 = max(0, tx - PAD), max(0, ty - PAD)
            wx1, wy1 = min(W, tx + CORE + PAD), min(H, ty + CORE + PAD)
            mem = np.flatnonzero((bx0 + 6 >= wx0) & (bx0 < wx1) & (by0 + 6 >= wy0) & (by0 < wy1))
            hw, ww = wy1 - wy0, wx1 - wx0
            npx = hw * ww
            A = np.zeros((npx, len(mem) + 3))
            for c, k in enumerate(mem):
                gy, gx = np.mgrid[by0[k]:by0[k] + 7, bx0[k]:bx0[k] + 7]
                ins = (gx >= wx0) & (gx < wx1) & (gy >= wy0) & (gy < wy1)
                A[(gy[ins] - wy0) * ww + (gx[ins] - wx0), c] = boxes[k][ins]
            yy, xx = np.mgrid[0:hw, 0:ww]
            A[:, -3] = 1
            A[:, -2] = (xx.ravel() - (ww - 1) / 2) / ww
            A[:, -1] = (yy.ravel() - (hw - 1) / 2) / hw
            m = good[wy0:wy1, wx0:wx1].ravel()
            d = img[wy0:wy1, wx0:wx1].ravel()[m].astype(float)
            sw = 1 / np.sqrt(var[wy0:wy1, wx0:wx1].ravel()[m].astype(float))
            keep = np.any(A[m] != 0, 0)
            U, Sv, Vt = np.linalg.svd(A[m][:, keep] * sw[:, None], full_matrices=False)
            gd = Sv > Sv.max() * 1e-12
            U, Sv, Vt = U[:, gd], Sv[gd], Vt[gd]
            beta = Vt.T @ ((U.T @ (d * sw)) / Sv)
            pos = np.full(A.shape[1], -1)
            pos[np.flatnonzero(keep)] = np.arange(keep.sum())
            r2 = np.zeros(npx)
            r2[m] = ((d - A[m][:, keep] @ beta) * sw) ** 2
            lev = np.zeros(npx)
            lev[m] = (U ** 2).sum(1)
            mm = np.zeros(npx, bool)
            mm[m] = True
            for c in np.flatnonzero(core_p[mem]):
                k = mem[c]
                cg = [cc for cc in range(len(mem)) if grp[mem[cc]] == grp[k]]
                p = pos[cg]
                gy, gx = np.mgrid[by0[k]:by0[k] + 7, bx0[k]:bx0[k] + 7]
                ins = (gx >= wx0) & (gx < wx1) & (gy >= wy0) & (gy < wy1)
                sup = (boxes[k] != 0) & ins
                nmask = int((boxes[k] != 0).sum() - mm[(gy[sup] - wy0) * ww + (gx[sup] - wx0)].sum())
                near = (gx - sx[k]) ** 2 + (gy - sy[k]) ** 2 <= 4
                pn = (gy[ins & near] - wy0) * ww + (gx[ins & near] - wx0)
                c2, dl = float(r2[pn][mm[pn]].sum()), float((1 - lev[pn])[mm[pn]].sum())
                ixc, iyc = int(round(sx[k])), int(round(sy[k]))
                fcore = int(np.bitwise_or.reduce(flg[max(0, iyc - 1):iyc + 2, max(0, ixc - 1):ixc + 2].ravel())) \
                    if 0 <= ixc < W and 0 <= iyc < H else -1
                if (p < 0).any():
                    a = e = np.nan
                else:
                    u = Vt[:, p].sum(1) / Sv
                    a, e = float(beta[p].sum()), float(np.sqrt(u @ u))
                rows.append((idx[k], sx[k] + x0, sy[k] + y0, a, e, nmask, c2, dl, fcore))
    c = _calib("dr1", det)
    f = RegularGridInterpolator((c["wy"], c["wx"]), c["wv"], bounds_error=False, fill_value=None)
    o = pd.DataFrame(rows, columns=["src", "x", "y", "amp", "amp_err", "n_masked_box", "chi2_local", "dof_local", "flags_core"])
    wv = f(np.stack([o.y.values, o.x.values], -1))
    o.insert(0, "id", S.id.values[o.src.values])
    o["flux_mJy"] = flux_mJy(o.amp.values, omega).astype(np.float32)
    o["flux_err_mJy"] = flux_mJy(o.amp_err.values, omega).astype(np.float32)
    o["wl_um"] = wv[:, 0].astype(np.float32)
    o["bw_um"] = wv[:, 1].astype(np.float32)
    o["chi2dof_local"] = (o.chi2_local / o.dof_local).astype(np.float32)
    o["mjd"], o["detector"], o["obs_id"] = mjd, np.int8(det), h["OBSID"]
    o["d_brighter_arcsec"] = S.d_brighter_arcsec.values[o.src.values].astype(np.float32)
    o["blend_n"] = S.blend_n.values[o.src.values].astype(np.int16)
    o["edge_px"] = np.minimum(np.minimum(o.x, 2039 - o.x), np.minimum(o.y, 2039 - o.y)).astype(np.float32)
    o = o.drop(columns=["amp", "amp_err", "chi2_local"])
    for k in ("x", "y", "dof_local"):
        o[k] = o[k].astype(np.float32)
    meta = dict(key=key, obs_id=h["OBSID"], detector=det, mjd=mjd, omega=omega, x0=x0, y0=y0, H=H, W=W,
                n_rows=len(o), frac_masked_region=float(1 - good.mean()),
                zodi_med=float(np.nanmedian(img)), var_med=float(np.nanmedian(var)))
    return o, meta


HALF = int(np.ceil((R + MARGIN) / PIX)) + 4


def _work(chunk):
    """Pictures of a chunk in sequence; the next picture's pixels are read in a thread while the current one is fitted."""
    from concurrent.futures import ThreadPoolExecutor
    out, metas = [], []
    with ThreadPoolExecutor(1) as io:
        nxt = io.submit(read_rows, chunk[0], HALF)
        for i, key in enumerate(chunk):
            t = time.time()
            cur = nxt
            if i + 1 < len(chunk):
                nxt = io.submit(read_rows, chunk[i + 1], HALF)
            try:
                fr = cur.result()
                o, meta = measure(key, fr) if fr is not None else (None, "off_frame")
                if o is None:
                    metas.append(dict(key=key, status=meta))
                    continue
                out.append(o)
                metas.append(dict(meta, status="ok", sec=time.time() - t))
            except Exception as e:
                metas.append(dict(key=key, status=f"error: {e!r}"[:300]))
    return out, metas


def phot():
    os.makedirs(f"{OUT}/phot", exist_ok=True)
    os.makedirs(f"{OUT}/meta", exist_ok=True)
    fr = pd.read_parquet(f"{OUT}/frames.parquet")
    done = set()
    for p in glob.glob(f"{OUT}/meta/meta-*.parquet"):
        mt = pd.read_parquet(p)
        done |= set(mt.key[~mt.status.str.startswith("error")])
    todo = [k for k in fr.key if k not in done]
    nmax = max([int(os.path.basename(p)[5:9]) for p in glob.glob(f"{OUT}/meta/meta-*.parquet")] + [-1])
    chunks = [todo[i:i + 12] for i in range(0, len(todo), 12)]
    cut = time.mktime(time.strptime(time.strftime("%Y-%m-%d ") + CUTOFF, "%Y-%m-%d %H:%M"))
    t0 = time.time()
    print(f"{len(done)} done, {len(todo)} to do, {len(chunks)} chunks", flush=True)
    nworkers = int(os.environ.get("E18_WORKERS", 6))
    with ProcessPoolExecutor(nworkers) as ex:
        it = iter(chunks)
        futs = set()
        for _ in range(nworkers * 2):
            c = next(it, None)
            if c is not None:
                futs.add(ex.submit(_work, c))
        n, nd = nmax, 0
        while futs:
            fu = next(as_completed(futs))
            futs.remove(fu)
            out, metas = fu.result()
            n += 1
            if out:
                pd.concat(out, ignore_index=True).to_parquet(f"{OUT}/phot/part-{n:04d}.tmp", index=False)
                os.replace(f"{OUT}/phot/part-{n:04d}.tmp", f"{OUT}/phot/part-{n:04d}.parquet")
            pd.DataFrame(metas).to_parquet(f"{OUT}/meta/meta-{n:04d}.tmp", index=False)
            os.replace(f"{OUT}/meta/meta-{n:04d}.tmp", f"{OUT}/meta/meta-{n:04d}.parquet")
            nd += len(metas)
            if n % 10 == 0:
                print(f"{time.strftime('%H:%M:%S')} part {n}: {nd} pictures this run, "
                      f"{nd / (time.time() - t0):.2f}/s", flush=True)
            if time.time() < cut:
                c = next(it, None)
                if c is not None:
                    futs.add(ex.submit(_work, c))
    print("done", time.strftime("%H:%M:%S"), flush=True)


def load_phot():
    return pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f"{OUT}/phot/part-*.parquet"))], ignore_index=True)


def qc():
    from orrery.e7 import static_fit
    S = pd.read_parquet(f"{OUT}/sources.parquet")
    P = load_phot()
    M = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f"{OUT}/meta/meta-*.parquet"))], ignore_index=True)
    usable = (np.isfinite(P.flux_mJy) & np.isfinite(P.flux_err_mJy) & (P.n_masked_box <= 3)
              & ((P.flags_core < 0) | ((P.flags_core & 2) == 0)))
    out = dict(protocol="docs/protocols/E18.md (QC)", generated=time.strftime("%Y-%m-%d %H:%M:%S %Z"),
               n_pictures_candidates=int(len(pd.read_parquet(f"{OUT}/frames.parquet"))),
               n_pictures_attempted=int(len(M)), pictures_status=M.status.str[:40].value_counts().to_dict(),
               n_rows=int(len(P)), n_sources_reported=int(P.id.nunique()))
    ok = M[M.status == "ok"]
    out["pictures_ok_per_detector"] = {int(k): int(v) for k, v in ok.detector.value_counts().sort_index().items()}
    # (c) losses
    out["c_losses"] = dict(
        frac_pictures_not_ok=round(float(1 - len(ok) / len(M)), 5),
        frac_rows_flux_nan=round(float(np.mean(~np.isfinite(P.flux_mJy))), 5),
        frac_rows_any_masked_box_pixel=round(float(np.mean(P.n_masked_box > 0)), 5),
        frac_rows_overflow_in_core=round(float(np.mean((P.flags_core >= 0) & ((P.flags_core & 2) > 0))), 5),
        frac_rows_nonlinear_in_core=round(float(np.mean((P.flags_core >= 0) & ((P.flags_core & (1 << 15)) > 0))), 5),
        frac_rows_usable=round(float(usable.mean()), 5),
        median_frac_masked_pixels_per_picture=round(float(ok.frac_masked_region.median()), 5),
        pictures_frac_masked_gt_0p2=int((ok.frac_masked_region > 0.2).sum()))
    # (a) error scale per detector on constant-looking stars
    const = S[(S.kind == "gaia") & (S.G < 19.5) & (S.phot_variable_flag != "VARIABLE") & S.vari_class.isna()
              & (S.ruwe < 1.4) & (S.d_nearest_arcsec.fillna(99) > 15) & (S.blend_n == 1) & S.in_region]
    Pc = P[usable & P.id.isin(const.id)]
    red, pulls, ratios = {}, {}, []
    for sid, g in Pc.groupby("id"):
        f, e, lam, det = g.flux_mJy.values.astype(float), g.flux_err_mJy.values.astype(float), g.wl_um.values.astype(float), g.detector.values
        keep = np.isin(det, [d for d in np.unique(det) if (det == d).sum() >= 20])
        if keep.sum() < 20:
            continue
        f, e, lam, det = f[keep], e[keep], lam[keep], det[keep]
        model, per, _, _ = static_fit(f, e, lam, det)
        G = float(S.G[S.id == sid].iloc[0])
        for d, (c2, n) in per.items():
            red.setdefault(d, []).append((c2 / (n - 3), G))
        pr = (f - model) / e
        if G >= 16:                                       # faint stars: only for s_d_by_G (information)
            continue
        for d in np.unique(det):
            pulls.setdefault(int(d), []).extend(pr[det == d].tolist())
        ratios.append(pd.DataFrame(dict(obs_id=g.obs_id.values[keep], detector=det, ratio=f / model, pull=pr,
                                        r2=(f - model) ** 2, e2=e ** 2, m2=model ** 2)))
    s_d = {int(d): float(np.sqrt(np.median([r for r, G in v if G < 16]))) for d, v in sorted(red.items())}
    bins = [(0, 13), (13, 14.5), (14.5, 16), (16, 17.5), (17.5, 19.5)]
    out["a_error_scale"] = dict(
        rule="per detector d: s_d = sqrt(median over constant-looking stars of chi2_d/(n_d-3)) of orrery.e7.static_fit "
             "(per-detector quadratic in wavelength); stars: Gaia G<16, not VARIABLE, no vari class, RUWE<1.4, no catalogue "
             "neighbour within 15\", blend_n=1, in region; usable rows (finite, n_masked_box<=3, no OVERFLOW in the 3x3 core); >=20 pictures per detector",
        n_stars=int(len(set(Pc.id[Pc.id.isin(const.id[const.G < 16])]))), s_d=s_d,
        n_star_terms={int(d): sum(G < 16 for _, G in v) for d, v in red.items()},
        s_d_by_G_note="bins 16-19.5 use the same cuts on fainter Gaia stars; information only (not part of s_d)",
        s_d_by_G={f"{a}-{b}": {int(d): (round(float(np.sqrt(np.median([r for r, G in v if a <= G < b]))), 3)
                                         if any(a <= G < b for _, G in v) else None) for d, v in sorted(red.items())}
                  for a, b in bins},
        pull_sigma_mad={int(d): round(float(1.4826 * np.median(np.abs(np.array(v) - np.median(v)))), 3) for d, v in sorted(pulls.items())},
        pull_frac_abs_gt5={int(d): round(float(np.mean(np.abs(np.array(v)) > 5)), 4) for d, v in sorted(pulls.items())})
    # per-picture zero-point check: median flux/static-model of constant stars in each picture
    if ratios:
        R_ = pd.concat(ratios)
        # two-parameter error model per detector: E[r^2] = a^2 sigma^2 + (b F)^2, fitted (non-negative LS) to
        # median(r^2)/0.4549 (median of chi2_1) in 10 quantile bins of S/N = F/sigma
        from scipy.optimize import nnls
        em = {}
        for d, g in R_.groupby("detector"):
            q = pd.qcut(np.sqrt(g.m2 / g.e2), 10, labels=False, duplicates="drop")
            t = g.groupby(q).agg(r2=("r2", "median"), e2=("e2", "mean"), m2=("m2", "mean"))
            c, _ = nnls(np.stack([t.e2, t.m2], 1) / t.e2.values[:, None], (t.r2 / 0.4549 / t.e2).values)
            em[int(d)] = dict(a=round(float(np.sqrt(c[0])), 3), b_frac=round(float(np.sqrt(c[1])), 4), n_rows=int(len(g)))
        out["a_error_scale"]["two_param_model"] = dict(
            rule="sigma_total^2 = (a sigma_formal)^2 + (b_frac F)^2 per detector, NNLS on S/N-decile medians of "
                 "(flux - static model)^2 / 0.4549 of the same constant stars", per_detector=em)
        fq = R_.groupby(["obs_id", "detector"]).agg(n_const=("ratio", "size"), zp_ratio_median=("ratio", "median"),
                                                     pull_median=("pull", "median")).reset_index()
        mad = 1.4826 * np.median(np.abs(fq.zp_ratio_median - 1))
        fq["zp_outlier_5mad"] = np.abs(fq.zp_ratio_median - 1) > 5 * mad
        fq.to_parquet(f"{OUT}/frame_qc.parquet", index=False)
        out["picture_zero_point"] = dict(file="frame_qc.parquet", mad_of_median_ratio=round(float(mad), 5),
                                         n_pictures=int(len(fq)), n_outliers_5mad=int(fq.zp_outlier_5mad.sum()),
                                         median_n_const_stars_per_picture=float(fq.n_const.median()))
    # (b) mean spectra of bright isolated stars vs 2MASS / WISE (Vega zero points, Jy: 2MASS Cohen+2003, WISE Jarrett+2011)
    zp = {"J": (1.235, 1594.0, 1.11, 1.36), "H": (1.662, 1024.0, 1.50, 1.80), "Ks": (2.159, 666.7, 1.99, 2.31),
          "W1": (3.35, 309.54, 3.0, 3.7), "W2": (4.60, 171.79, 4.2, 5.0)}
    col = {"J": "Jmag", "H": "Hmag", "Ks": "Kmag", "W1": "W1mag", "W2": "W2mag"}
    ref = const[(const.G > 10) & const[list(col.values())].notna().all(1)].sort_values("G").head(8)
    b = []
    for _, s in ref.iterrows():
        g = P[usable & (P.id == s.id)]
        rec = dict(id=s.id, G=round(float(s.G), 3), bp_rp=round(float(s.bp_rp), 3))
        for k, (lc, z, lo, hi) in zp.items():
            sel = (g.wl_um >= lo) & (g.wl_um <= hi)
            cat = z * 1e3 * 10 ** (-0.4 * float(s[col[k]]))
            sph = float(np.median(g.flux_mJy[sel])) if sel.sum() else np.nan
            rec[k] = dict(catalog_mJy=round(cat, 3), spherex_median_mJy=round(sph, 3), ratio=round(sph / cat, 3), n=int(sel.sum()))
        b.append(rec)
    out["b_spectra_vs_2mass_wise"] = dict(
        note="SPHEREx = median flux over usable pictures with wl in the band's range; flux is 'inside the 7x7 ePSF box' "
             "(no aperture correction), so ratios < 1 by the wing fraction are expected; broad-band vs narrow-band "
             "colour terms not corrected", stars=b,
        median_ratio={k: round(float(np.nanmedian([r[k]["ratio"] for r in b])), 3) for k in zp})
    # (d) timing
    out["d_timing"] = dict(sec_per_picture_per_worker_median=round(float(ok.sec.median()), 2),
                           workers=6, phot_log="data/phase2/e18/phot.log")
    json.dump(out, open(f"{OUT}/qc.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps(out, indent=1, default=str)[:6000])


if __name__ == "__main__":
    {"catalogue": catalogue, "kim": kim, "frames": frames, "phot": phot, "qc": qc}[sys.argv[1]]()
