"""E17 - blind search for moving objects by shift-and-stack (docs/PROTOCOLS.md PHASE 3 E17; frozen choices in
docs/protocols/E17.md). Steps (each resumable, partial results on disk per field):

  .venv/bin/python scripts/run_e17.py draw            # fields + windows from the frame index (no pixels)
  .venv/bin/python scripts/run_e17.py cache  <field>  # pictures resampled onto the field grid -> data/phase2/e17/<field>/
  .venv/bin/python scripts/run_e17.py null   <field>  # 10 time-permuted searches -> nulls.json
  .venv/bin/python scripts/run_e17.py inject <field>  # 200 planted movers -> inject.json
  .venv/bin/python scripts/run_e17.py real   <field>  # the real search (refuses before docs/protocols/E17.md is committed)
  .venv/bin/python scripts/run_e17.py known  <field>  # SkyBoT objects in the patch at t_ref + forced stack S/N
  .venv/bin/python scripts/run_e17.py final           # threshold, recovery, candidates -> data/findings/e17.json + figures
"""
import json, os, subprocess, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

ROOT = "data/phase2/e17"
OUT = "data/findings/e17.json"
PROTO = "docs/protocols/E17.md"
SEED_FIELDS, SEED_PERM, SEED_INJ = 1717, 171717, 17171717
N_FIELDS = 6
SCALE = 6.2                       # arcsec per grid pixel (TAN, north up, east left)
PATCH = 232                       # 0.4 deg search patch (objects inside it at t_ref)
DMAX_DEG = 0.15                   # each trajectory uses the pictures where it is <= 0.15 deg from its t_ref position
DMAX = DMAX_DEG * 3600 / SCALE    # 87.1 px
MARGIN = int(np.ceil(DMAX)) + 4   # stamp = patch + margin on every side
S = PATCH + 2 * MARGIN            # 414 px
HALF_SPAN_D = 3.0                 # pictures within +-3 d of t_ref (0.05 deg/day x 3 d = DMAX)
VMIN, VMAX = 0.05, 1.0            # deg/day
TRAIL = 0.7                       # px: max trail error over the span a trajectory uses
STEP = TRAIL * np.sqrt(2) / DMAX  # relative grid step in speed and in angle (vector error <= 0.7 px)
FWHM_PX = 1.55                    # matched filter: Gaussian, ePSF FWHM 7.5" (1.22 px) (+) bilinear resampling
CLIP = 5.0                        # per-epoch |S/N| clip before stacking
PRE = 5.0                         # S/N pre-threshold for storing (x, y, rate) peaks
NMS_PX, NMS_RATE = 3.0, 0.05      # candidates: greedy non-maximum suppression (px at t_ref, deg/day)
MATCH_PX, MATCH_RATE = 15 / SCALE, 0.05
FALSE_PER_FIELD = 0.1
N_PERM, N_INJ = 10, 200
EXCLUDE_DEG = 1.0                 # distance from E3/E3b field centres


def git_committed(path):
    r = subprocess.run(["git", "log", "--format=%H %ci", "-1", "--", path], capture_output=True, text=True)
    return r.stdout.strip()


# ---------------------------------------------------------------- fields
def time_plan(mjd):
    """mjd: sorted picture times of one visit window. t_ref = the picture time with the most pictures within
    +-0.5 d (earliest on ties); used = pictures within +-HALF_SPAN_D of t_ref."""
    n05 = np.array([np.sum(np.abs(mjd - t) <= 0.5) for t in mjd])
    t_ref = float(mjd[int(np.argmax(n05))])
    used = np.abs(mjd - t_ref) <= HALF_SPAN_D
    return t_ref, used


def windows(mjd, gap=2.0):
    m = np.sort(mjd)
    br = np.where(np.diff(m) > gap)[0]
    return [w for w in np.split(m, br + 1)]


def draw():
    path = f"{ROOT}/fields.json"
    if os.path.exists(path):
        return json.load(open(path))
    import build_field as bf
    from astropy.coordinates import SkyCoord, BarycentricMeanEcliptic
    import astropy.units as u
    df, src = bf.load_index()
    old = [(f["ra"], f["dec"]) for k in ("e3_fields", "e3b_fields") for f in json.load(open(f"data/findings/{k}.json"))["fields"]]
    rng = np.random.default_rng(SEED_FIELDS)
    acc, rej, n = [], [], 0

    def sep(a1, d1, a2, d2):
        return SkyCoord(a1, d1, unit="deg").separation(SkyCoord(a2, d2, unit="deg")).deg

    while len(acc) < N_FIELDS:
        lon, lat = float(rng.uniform(0, 360)), float(rng.uniform(-3, 3))
        n += 1
        c = SkyCoord(lon * u.deg, lat * u.deg, frame=BarycentricMeanEcliptic()).icrs
        ra, dec = float(c.ra.deg), float(c.dec.deg)
        b = float(c.galactic.b.deg)
        rec = dict(draw=n, ecl_lon=lon, ecl_lat=lat, ra=ra, dec=dec, b=b)
        if abs(b) < 15:
            rej.append(dict(rec, reason="|b| < 15")); continue
        if any(sep(ra, dec, a["ra"], a["dec"]) < 20 for a in acc):
            rej.append(dict(rec, reason="< 20 deg from an accepted field")); continue
        if any(sep(ra, dec, a, d) < EXCLUDE_DEG for a, d in old):
            rej.append(dict(rec, reason="E3/E3b field")); continue
        cand = bf.candidates(df, ra, dec)
        best = None
        for w in windows(cand.mjd.values):
            if w[-1] - w[0] > 10:          # the visit window must be <= 10 d: use its first 10 d
                w = w[w <= w[0] + 10]
            t_ref, used = time_plan(w)
            if best is None or used.sum() > best[2]:
                best = (float(w[0]), float(w[-1]), int(used.sum()), t_ref)
        if best is None or best[2] < 30:
            rej.append(dict(rec, reason="< 30 pictures in +-3 d of t_ref in any visit window",
                            n_best=None if best is None else best[2])); continue
        sel = cand[np.abs(cand.mjd.values - best[3]) <= HALF_SPAN_D].sort_values("mjd", kind="stable")
        acc.append(dict(rec, field=f"e17_{len(acc):02d}", window=[best[0], best[1]], t_ref=best[3],
                        n_pictures=len(sel), keys=sel.key.tolist()))
    out = dict(protocol="docs/protocols/E17.md", seed=SEED_FIELDS, index=src, n_draws=n, fields=acc, rejected=rej,
               rule="rng = default_rng(1717); per draw ecliptic lon U(0,360) then lat U(-3,3) deg (BarycentricMeanEcliptic"
                    " J2000) -> ICRS; reject |b|<15, <20 deg from an accepted field, <1 deg from an E3/E3b field centre, or "
                    "fewer than 30 index pictures (dqa Pass, footprint contains the centre) within +-3 d of t_ref in a "
                    "visit window (gaps > 2 d split windows; first 10 d of a window); t_ref = picture time with the most "
                    "pictures within +-0.5 d; the window with most pictures is used")
    os.makedirs(ROOT, exist_ok=True)
    json.dump(out, open(path, "w"), indent=1)
    return out


def cmd_draw():
    out = draw()
    import build_field as bf
    df, _ = bf.load_index()
    for f in out["fields"]:
        sub = df[df.key.isin(f["keys"])]
        dt = np.sort(sub.mjd.values) - f["t_ref"]
        g = windows(sub.mjd.values, gap=0.05)
        print(f["field"], f"ra {f['ra']:.3f} dec {f['dec']:.3f} b {f['b']:.1f} n {f['n_pictures']} groups {len(g)} "
              f"span {dt.min():+.2f}..{dt.max():+.2f} d  dets {dict(zip(*np.unique(sub.detector, return_counts=True)))}")
        print("    group times rel t_ref:", np.round([w.mean() - f["t_ref"] for w in g], 2).tolist())
        print("    pictures per group:", [len(w) for w in g])
    print(out["n_draws"], "draws")


# ---------------------------------------------------------------- pictures on the field grid
def field(name):
    for i, f in enumerate(draw()["fields"]):
        if f["field"] == name:
            return i, f
    raise KeyError(name)


def grid_radec(f):
    from orrery.change import grid_world
    return grid_world(S, f["ra"], f["dec"], SCALE)


def read_picture(key, gra, gdec):
    """Native rows covering the grid (range reads, scripts/e2_cache helpers), bilinear onto the grid."""
    import e2_cache as ec
    from astropy.wcs import WCS
    from orrery.data import BUCKET, BAD_BITS, Frame, _calib, release_of
    url = f"{BUCKET}/{key}"
    p, o = ec._header(url, 0)
    h, img_off = ec._header(url, o)
    assert h["EXTNAME"] == "IMAGE" and h["BITPIX"] == -32 and h["NAXIS1"] == 2040 and "BZERO" not in h
    w = WCS(h)
    x, y = w.all_world2pix(gra.ravel(), gdec.ravel(), 0)
    x, y = x.reshape(gra.shape).astype(np.float32), y.reshape(gra.shape).astype(np.float32)
    ins = (x >= 0) & (x <= 2038.999) & (y >= 0) & (y <= 2038.999)
    if ins.sum() < 1000:
        return None
    y0, y1 = int(np.floor(y[ins].min())), min(2040, int(np.ceil(y[ins].max())) + 2)
    img = np.frombuffer(ec._get(url, img_off + y0 * 8160, img_off + y1 * 8160), ">f4").reshape(-1, 2040)
    fh, fl_off = ec._header(url, img_off + ec._pad(2040 * 2040 * 4))
    assert fh["EXTNAME"] == "FLAGS" and fh["ZCMPTYPE"] == "RICE_1" and fh["ZTILE1"] == 2040 and fh["ZTILE2"] == 1
    desc = np.frombuffer(ec._get(url, fl_off + 8 * y0, fl_off + 8 * y1), ">i4").reshape(-1, 2)
    heap = fl_off + 8 * 2040
    lo, hi = int(desc[:, 1].min()), int((desc[:, 1] + desc[:, 0]).max())
    buf = ec._get(url, heap + lo, heap + hi)
    from astropy.io.fits.hdu.compressed._codecs import Rice1
    codec = Rice1(blocksize=fh.get("ZVAL1", 32), bytepix=4, tilesize=2040)
    flg = np.stack([codec.decode(np.frombuffer(buf[o_ - lo:o_ - lo + n], np.uint8)) for n, o_ in desc]).astype(np.int32)
    vh, var_off = ec._header(url, fl_off + ec._pad(8 * 2040 + fh["PCOUNT"]))
    assert vh["EXTNAME"] == "VARIANCE" and vh["BITPIX"] == -32
    var = np.frombuffer(ec._get(url, var_off + y0 * 8160, var_off + y1 * 8160), ">f4").reshape(-1, 2040)
    # bilinear onto the grid; any bad native neighbour (or off-detector) masks the grid pixel
    xi = np.clip(np.floor(x).astype(int), 0, 2038)
    yi = np.clip(np.floor(y).astype(int) - y0, 0, y1 - y0 - 2)
    fx, fy = x - xi, (y - y0) - yi
    bad = (flg & BAD_BITS) != 0
    bad |= ~np.isfinite(img) | ~np.isfinite(var) | (var <= 0)

    def bil(a):
        return ((1 - fx) * (1 - fy) * a[yi, xi] + fx * (1 - fy) * a[yi, xi + 1] + (1 - fx) * fy * a[yi + 1, xi]
                + fx * fy * a[yi + 1, xi + 1])
    m = ~ins | bad[yi, xi] | bad[yi, xi + 1] | bad[yi + 1, xi] | bad[yi + 1, xi + 1]
    im = np.where(m, 0, bil(np.where(bad, 0, img))).astype(np.float32)
    va = np.where(m, 0, bil(np.where(bad, 0, var))).astype(np.float32)
    c = _calib(release_of(key), int(h["DETECTOR"]))
    fr = Frame.__new__(Frame)
    fr.wx, fr.wy, fr.wv = c["wx"], c["wy"], c["wv"]
    sub = (slice(4, S, 8), slice(4, S, 8))           # wavelength on every 8th grid pixel (it varies slowly)
    lam, _ = fr.wave_at(x[sub].astype(float), y[sub].astype(float))
    xc, yc = float(x[S // 2, S // 2]), float(y[S // 2, S // 2])
    zone = int(np.argmin((c["xc"] - xc) ** 2 + (c["yc"] - yc) ** 2))
    return dict(img=im, var=va, mask=np.packbits(m), lam8=lam.astype(np.float32), mjd=float(h["MJD-AVG"]),
                obs_id=h["OBSID"], detector=int(h["DETECTOR"]), omega=float(h["OMEGA_MEDIAN"]), zone=zone,
                wcs=w.to_header(relax=True).tostring(), key=key, xc=xc, yc=yc)


def cmd_cache(name):
    from concurrent.futures import ThreadPoolExecutor
    i, f = field(name)
    d = f"{ROOT}/{name}/pics"
    os.makedirs(d, exist_ok=True)
    gra, gdec = grid_radec(f)

    def one(key):
        p = f"{d}/{key.split('/')[-1].replace('.fits', '')}.npz"
        if os.path.exists(p):
            return "ok"
        r = read_picture(key, gra, gdec)
        if r is None:
            open(p[:-4] + ".off", "w").write("off\n")
            return "off"
        np.savez(p[:-4] + ".tmp.npz", **r)
        os.replace(p[:-4] + ".tmp.npz", p)
        return "ok"
    t = time.time()
    with ThreadPoolExecutor(4) as ex:
        st = {}
        for rnd in range(3):
            todo = [k for k in f["keys"] if st.get(k) not in ("ok", "off")]
            for k, fu in [(k, ex.submit(one, k)) for k in todo]:
                try:
                    st[k] = fu.result()
                except Exception as e:
                    st[k] = f"error {e!r}"[:200]
            if all(v in ("ok", "off") for v in st.values()):
                break
            time.sleep(10)
    print(name, {v: list(st.values()).count(v) for v in set(st.values())}, f"{time.time() - t:.0f}s", flush=True)
    if all(v in ("ok", "off") for v in st.values()):
        open(f"{ROOT}/{name}/CACHED", "w").write(time.strftime("%Y-%m-%dT%H:%M:%S%z\n"))


# ---------------------------------------------------------------- residual exposure maps
def load_pics(name):
    i, f = field(name)
    d = f"{ROOT}/{name}/pics"
    P = []
    for k in f["keys"]:
        p = f"{d}/{k.split('/')[-1].replace('.fits', '')}.npz"
        if not os.path.exists(p):
            continue
        z = np.load(p)
        r = {k_: z[k_] for k_ in z.files}
        r["mask"] = np.unpackbits(r["mask"])[:S * S].reshape(S, S).astype(bool)
        for k_ in ("mjd", "omega", "xc", "yc"):
            r[k_] = float(r[k_])
        for k_ in ("detector", "zone"):
            r[k_] = int(r[k_])
        for k_ in ("obs_id", "wcs", "key"):
            r[k_] = str(r[k_])
        P.append(r)
    P.sort(key=lambda r: (r["mjd"], r["detector"]))
    return f, P


def group_ids(mjd, gap=0.05):
    o = np.argsort(mjd)
    g = np.zeros(len(mjd), int)
    gid, last = 0, None
    for e in o:
        if last is not None and mjd[e] - last > gap:
            gid += 1
        g[e] = gid
        last = mjd[e]
    return g


def colour_map(lam8):
    """S(lambda)/S(0.55 um) on the full grid from the every-8th-pixel wavelength map."""
    from scipy.ndimage import map_coordinates
    from orrery.physics import solar_flux_mJy
    jj, ii = np.mgrid[0:S, 0:S].astype(float)
    lam = map_coordinates(lam8, [(jj - 4) / 8, (ii - 4) / 8], order=1, mode="nearest")
    return (solar_flux_mJy(lam) / solar_flux_mJy(0.55)).astype(np.float32), lam


_KERN = None


def kernel():
    global _KERN
    if _KERN is None:
        s = FWHM_PX / 2.3548
        r = np.arange(-3, 4)
        g = np.exp(-0.5 * (r[:, None] ** 2 + r[None] ** 2) / s ** 2)
        _KERN = (g / g.sum()).astype(np.float32)
    return _KERN


def block_bg(r, m, b=23):
    """Median of unmasked pixels in b x b blocks, bilinearly interpolated between block centres."""
    from scipy.ndimage import map_coordinates
    n = S // b
    a = np.where(m, np.nan, r)[: n * b, : n * b].reshape(n, b, n, b).transpose(0, 2, 1, 3).reshape(n, n, b * b)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            B = np.nanmedian(a, axis=2)
    B = np.where(np.isfinite(B), B, np.nanmedian(B) if np.isfinite(B).any() else 0.0)
    jj, ii = np.mgrid[0:S, 0:S].astype(float)
    return map_coordinates(B, [(jj - (b - 1) / 2) / b, (ii - (b - 1) / 2) / b], order=1, mode="nearest").astype(np.float32)


def exposures(P, extra=None):
    """Residual, matched-filtered, V-equivalent exposure maps.
    extra: optional list (per picture) of planted surface brightness [S, S] added before static-sky removal.
    Returns dict with per-picture (F, w) and per-exposure (Fw clipped, w, t, group, detectors) arrays."""
    import warnings
    from scipy.ndimage import correlate
    mjd = np.array([p["mjd"] for p in P])
    det = np.array([p["detector"] for p in P])
    grp = group_ids(mjd)
    img = [p["img"] + (extra[e] if extra is not None else 0) for e, p in enumerate(P)]
    keep, ref = np.zeros(len(P), bool), [None] * len(P)
    for dd in np.unique(det):
        for gg in np.unique(grp[det == dd]):
            others = np.where((det == dd) & (grp != gg))[0]
            mine = np.where((det == dd) & (grp == gg))[0]
            if len(others) < 3:
                continue
            st = np.stack([np.where(P[o]["mask"], np.nan, img[o]) for o in others])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                med = np.nanmedian(st, axis=0)
            for e in mine:
                ref[e], keep[e] = med, True
    g, g2 = kernel(), kernel() ** 2
    pic = []
    for e, p in enumerate(P):
        if not keep[e]:
            pic.append(None)
            continue
        m = p["mask"] | ~np.isfinite(ref[e])
        r = np.where(m, 0, img[e] - np.where(m, 0, ref[e])).astype(np.float32)
        r = np.where(m, 0, r - block_bg(r, m))
        w = np.where(m, 0, 1 / np.where(p["var"] > 0, p["var"], np.inf)).astype(np.float32)
        num = correlate(w * r, g, mode="constant")
        den = correlate(w, g2, mode="constant")
        full = correlate((~m).astype(np.float32), np.ones_like(g), mode="constant") > 48.5   # all 49 pixels valid
        A = np.where(den > 0, num / np.where(den > 0, den, 1), 0)
        snr = A * np.sqrt(den)
        s = 1.4826 * np.median(np.abs(snr[full] - np.median(snr[full]))) if full.sum() > 1000 else np.nan
        if not np.isfinite(s) or s <= 0:
            pic.append(None); keep[e] = False
            continue
        col, _ = colour_map(p["lam8"])
        conv = (SCALE / 206264.806) ** 2 * 1e9                 # MJy/sr x grid pixel -> mJy
        F = A * conv / col                                     # V-equivalent mJy
        wF = np.where(full, den / s ** 2 / (conv / col) ** 2, 0).astype(np.float32)   # 1/var of F
        pic.append(dict(F=F.astype(np.float32), w=wF, scale=float(s)))
    # exposures = the simultaneous pictures (same obs_id)
    obs = np.array([p["obs_id"] for p in P])
    ex = []
    for o in dict.fromkeys(obs[keep]):
        ee = [e for e in np.where(obs == o)[0] if keep[e]]
        W = sum(pic[e]["w"] for e in ee)
        F = np.where(W > 0, sum(pic[e]["F"] * pic[e]["w"] for e in ee) / np.where(W > 0, W, 1), 0)
        ex.append(dict(F=F, W=W, t=float(mjd[ee[0]]), group=int(grp[ee[0]]), pics=ee, dets=[int(det[e]) for e in ee]))
    # per-pixel temporal scatter of the exposure S/N (static-sky residuals at stars vary from picture to picture;
    # a mover touches a pixel in only a few exposures, so the median absolute deviation ignores it)
    sn = np.stack([np.where(x["W"] > 0, x["F"] * np.sqrt(x["W"]), np.nan) for x in ex])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        med = np.nanmedian(sn, axis=0)
        tau = 1.4826 * np.nanmedian(np.abs(sn - med), axis=0)
    nval = np.isfinite(sn).sum(0)
    tau = np.where((nval >= 5) & np.isfinite(tau), np.maximum(tau, 1.0), np.inf).astype(np.float32)
    for x in ex:
        W = np.where(np.isfinite(tau), x["W"] / tau ** 2, 0).astype(np.float32)
        lim = CLIP / np.sqrt(np.where(W > 0, W, np.inf))
        Fc = np.clip(x.pop("F"), -lim, lim)
        x.pop("W")
        x["Fw"], x["w"] = (Fc * W).astype(np.float32), W
    return dict(pic=pic, ex=ex, keep=keep, grp=grp, mjd=mjd, det=det, tau=tau)


# ---------------------------------------------------------------- rate grid and search
def rate_grid():
    nsp = int(np.ceil(np.log(VMAX / VMIN) / np.log(1 + STEP))) + 1
    sp = VMIN * (1 + STEP) ** np.arange(nsp)
    nan = int(np.ceil(2 * np.pi / STEP))
    th = np.arange(nan) * 2 * np.pi / nan
    v = np.repeat(sp, nan)
    a = np.tile(th, nsp)
    pxd = v * 3600 / SCALE
    return dict(speed=v, angle=a, vx=pxd * np.cos(a), vy=pxd * np.sin(a), nsp=nsp, nan=nan, speeds=sp, angles=th)


def shifts(G, t, t_ref, grp, sel=None):
    """Integer shifts (n_rate, n_exp) and use mask for the rate subset `sel`."""
    vx, vy = (G["vx"], G["vy"]) if sel is None else (G["vx"][sel], G["vy"][sel])
    dt = (np.asarray(t) - t_ref)[None]
    sp = np.hypot(vx, vy)[:, None]
    use = (sp * np.abs(dt) <= DMAX) & (np.abs(dt) <= HALF_SPAN_D)
    sx = np.rint(vx[:, None] * dt).astype(np.int16)
    sy = np.rint(vy[:, None] * dt).astype(np.int16)
    gmat = np.where(use, np.asarray(grp)[None], -1)
    ngr = np.array([len(set(r[r >= 0])) for r in gmat])
    return sx, sy, use & (ngr >= 2)[:, None], ngr


def stack_one(EX, sx, sy, use):
    num = np.zeros((PATCH, PATCH), np.float32)
    den = np.zeros((PATCH, PATCH), np.float32)
    for e in np.where(use)[0]:
        y0, x0 = MARGIN + sy[e], MARGIN + sx[e]
        num += EX[e]["Fw"][y0:y0 + PATCH, x0:x0 + PATCH]
        den += EX[e]["w"][y0:y0 + PATCH, x0:x0 + PATCH]
    snr = np.where(den > 0, num / np.sqrt(np.where(den > 0, den, 1)), 0)
    return snr, num, den


def search(EX, t, t_ref, grp, G, sel=None, log=None):
    """Full search over the rate subset `sel` (indices into the grid; default all). Returns peaks array
    (x, y, rate_index, snr) of local (x, y) maxima with S/N >= PRE, after skipping duplicate shift patterns."""
    from scipy.ndimage import maximum_filter
    idx = np.arange(len(G["vx"])) if sel is None else np.asarray(sel)
    out, seen, t0 = [], set(), time.time()
    for c0 in range(0, len(idx), 4096):
        ch = idx[c0:c0 + 4096]
        sx, sy, use, _ = shifts(G, t, t_ref, grp, ch)
        for k in range(len(ch)):
            if not use[k].any():
                continue
            key = (use[k].tobytes(), np.where(use[k], sx[k], 0).tobytes(), np.where(use[k], sy[k], 0).tobytes())
            if key in seen:
                continue
            seen.add(key)
            snr, _, _ = stack_one(EX, sx[k], sy[k], use[k])
            hit = snr >= PRE
            if hit.any():
                mx = maximum_filter(snr, 3, mode="constant")
                yy, xx = np.nonzero(hit & (snr >= mx))
                for y_, x_ in zip(yy, xx):
                    out.append((x_, y_, ch[k], snr[y_, x_]))
        if log and (c0 // 4096) % 8 == 0:
            print(f"  {log}: {c0 + len(ch)}/{len(idx)} rates, {len(seen)} distinct, {len(out)} peaks, "
                  f"{time.time() - t0:.0f}s", flush=True)
    return np.array(out, float).reshape(-1, 4), len(seen)


def nms(peaks, G):
    """Greedy non-maximum suppression in (x, y, velocity): candidates."""
    if len(peaks) == 0:
        return np.zeros((0, 4))
    o = np.argsort(-peaks[:, 3])
    pk = peaks[o]
    ri = pk[:, 2].astype(int)
    vdx = G["speed"][ri] * np.cos(G["angle"][ri])
    vdy = G["speed"][ri] * np.sin(G["angle"][ri])
    from scipy.spatial import cKDTree
    tree = cKDTree(pk[:, :2])
    acc = []
    alive = np.ones(len(pk), bool)
    for i in range(len(pk)):           # same greedy rule as before, neighbours found with a k-d tree
        if not alive[i]:
            continue
        acc.append(i)
        nb = np.array(tree.query_ball_point(pk[i, :2], NMS_PX + 1e-9), int)
        dv = np.hypot(vdx[nb] - vdx[i], vdy[nb] - vdy[i])
        alive[nb[dv <= NMS_RATE]] = False
    return pk[acc]


def ex_arrays(X, perm=None):
    EX = X["ex"]
    t = np.array([e["t"] for e in EX])
    grp = np.array([e["group"] for e in EX])
    if perm is not None:
        t, grp = t[perm], grp[perm]
    return EX, t, grp


def ex_cached(name):
    """Real exposure maps, cached per field (they do not depend on time order)."""
    p = f"{ROOT}/{name}/ex.npz"
    f, P = load_pics(name)
    if os.path.exists(p):
        z = np.load(p)
        EX = [dict(Fw=z["Fw"][e], w=z["w"][e], t=float(z["t"][e]), group=int(z["group"][e])) for e in range(len(z["t"]))]
        return f, P, dict(ex=EX)
    X = exposures(P)
    EX = X["ex"]
    np.savez(p[:-4] + ".tmp.npz", Fw=np.stack([e["Fw"] for e in EX]), w=np.stack([e["w"] for e in EX]),
             t=np.array([e["t"] for e in EX]), group=np.array([e["group"] for e in EX]))
    os.replace(p[:-4] + ".tmp.npz", p)
    json.dump(dict(n_pictures=len(P), n_used=int(X["keep"].sum()), n_exposures=len(EX),
                   dropped=[P[e]["obs_id"] + f"_D{P[e]['detector']}" for e in np.where(~X["keep"])[0]],
                   scales=[None if q is None else q["scale"] for q in X["pic"]],
                   exposures=[dict(t=e["t"], group=e["group"], dets=e["dets"]) for e in EX]),
              open(f"{ROOT}/{name}/ex_meta.json", "w"), indent=1)
    return f, P, X


def cand_records(c, G):
    ri = c[:, 2].astype(int)
    return [dict(x=int(a), y=int(b), rate=int(r), speed=round(float(G["speed"][r]), 5),
                 angle_deg=round(float(np.degrees(G["angle"][r])), 3), snr=round(float(s), 3))
            for a, b, r, s in zip(c[:, 0], c[:, 1], ri, c[:, 3])]


def cmd_null(name, ks):
    fi, _ = field(name)
    f, P, X = ex_cached(name)
    EX, t, grp = ex_arrays(X)
    G = rate_grid()
    for k in ks:
        p = f"{ROOT}/{name}/null_{k}.json"
        if os.path.exists(p):
            continue
        perm = np.random.default_rng(SEED_PERM + 1000 * fi + k).permutation(len(EX))
        t0 = time.time()
        peaks, nd = search(EX, t[perm], f["t_ref"], grp[perm], G, log=f"{name} null {k}")
        c = nms(peaks, G)
        json.dump(dict(field=name, k=k, perm=perm.tolist(), n_rates_distinct=nd, n_peaks=len(peaks),
                       seconds=round(time.time() - t0), candidates=cand_records(c, G)), open(p, "w"))
        print(f"{name} null {k}: {len(c)} cands, max {c[:, 3].max() if len(c) else 0:.2f}, {time.time() - t0:.0f}s", flush=True)


def robust_snr(EX, t, grp, t_ref, G, x, y, rate):
    """Detection statistic (deviation 14:11): stack S/N after removing the group (visit) with the largest summed
    contribution, so that one visit alone (e.g. a bright object that is static within its own visit) cannot make a
    detection. Returns (mean-stack S/N, leave-one-group-out S/N, groups with data)."""
    sx, sy, use, _ = shifts(G, t, t_ref, grp, np.array([rate]))
    ee = np.where(use[0])[0]
    if len(ee) == 0:
        return np.nan, np.nan, 0
    fw = np.array([EX[e]["Fw"][MARGIN + y + sy[0][e], MARGIN + x + sx[0][e]] for e in ee], float)
    ww = np.array([EX[e]["w"][MARGIN + y + sy[0][e], MARGIN + x + sx[0][e]] for e in ee], float)
    gg = np.asarray(grp)[ee]
    ug = [g for g in np.unique(gg) if ww[gg == g].sum() > 0]
    if ww.sum() <= 0:
        return np.nan, np.nan, 0
    full = fw.sum() / np.sqrt(ww.sum())
    if len(ug) < 2:
        return full, np.nan, len(ug)
    gmax = max(ug, key=lambda g: fw[gg == g].sum())
    k = gg != gmax
    return full, (fw[k].sum() / np.sqrt(ww[k].sum()) if ww[k].sum() > 0 else np.nan), len(ug)


def min_snr(EX, t, grp, t_ref, G, x, y, rate):
    """POST HOC variant (15:04): minimum over visits of the stack S/N with that visit removed."""
    sx, sy, use, _ = shifts(G, t, t_ref, grp, np.array([rate]))
    ee = np.where(use[0])[0]
    if len(ee) == 0:
        return np.nan
    fw = np.array([EX[e]["Fw"][MARGIN + y + sy[0][e], MARGIN + x + sx[0][e]] for e in ee], float)
    ww = np.array([EX[e]["w"][MARGIN + y + sy[0][e], MARGIN + x + sx[0][e]] for e in ee], float)
    gg = np.asarray(grp)[ee]
    ug = [g for g in np.unique(gg) if ww[gg == g].sum() > 0]
    if len(ug) < 2:
        return np.nan
    v = [fw[gg != g].sum() / np.sqrt(ww[gg != g].sum()) for g in ug if ww[gg != g].sum() > 0]
    return float(min(v)) if v else np.nan


def cmd_posthoc(name):
    """Add S/N_min to stored null/real candidates, injected movers (re-planted) and known objects."""
    global f_ra, f_dec
    fi, f = field(name)
    f_ra, f_dec = f["ra"], f["dec"]
    _, P, X = ex_cached(name)
    EX, t, grp = ex_arrays(X)
    G = rate_grid()
    for pth, is_null in [(f"{ROOT}/{name}/null_{k}.json", True) for k in range(N_PERM)] + [(f"{ROOT}/{name}/real.json", False)]:
        d = json.load(open(pth))
        if d.get("min_done"):
            continue
        tt, gg = (t[np.array(d["perm"])], grp[np.array(d["perm"])]) if is_null else (t, grp)
        for c in d["candidates"]:
            if c.get("snr_logo") is not None or c["snr"] >= 5:
                v = min_snr(EX, tt, gg, f["t_ref"], G, c["x"], c["y"], c["rate"])
                c["snr_min"] = None if not np.isfinite(v) else round(v, 3)
        d["min_done"] = True
        json.dump(d, open(pth + ".tmp", "w")); os.replace(pth + ".tmp", pth)
    kp = f"{ROOT}/{name}/known.json"
    K = json.load(open(kp))
    for k in K["objects"]:
        fo = k.get("forced")
        if fo and fo.get("rate") is not None:
            v = min_snr(EX, t, grp, f["t_ref"], G, fo["x"], fo["y"], fo["rate"])
            fo["snr_min"] = None if not np.isfinite(v) else v
    json.dump(K, open(kp, "w"), indent=1)
    # injections: same movers (same seeds), re-planted, S/N_min at the stored best (x, y, rate)
    for b in range(4):
        pth = f"{ROOT}/{name}/inject_{b}.json"
        d = json.load(open(pth))
        if d.get("min_done"):
            continue
        mv = d["movers"]
        Xi = exposures(P, plant(P, mv, f["t_ref"]))
        EXi, ti, gi = ex_arrays(Xi)
        for m in mv:
            if m.get("rec_rate") is not None:
                v = min_snr(EXi, ti, gi, f["t_ref"], G, m["rec_x"], m["rec_y"], m["rec_rate"])
                m["rec_snr_min"] = None if not np.isfinite(v) else v
        d["min_done"] = True
        json.dump(d, open(pth + ".tmp", "w")); os.replace(pth + ".tmp", pth)
    print(name, "posthoc S/N_min done", flush=True)


def neighbours(G, speed, angle):
    ks = int(np.rint(np.log(speed / VMIN) / np.log(1 + STEP)))
    ka = int(np.rint((angle % (2 * np.pi)) / (2 * np.pi / G["nan"])))
    out = []
    for a in (-1, 0, 1):
        for b in (-1, 0, 1):
            s_ = ks + a
            if 0 <= s_ < G["nsp"]:
                out.append(s_ * G["nan"] + (ka + b) % G["nan"])
    return out


def forced(EX, t, t_ref, grp, G, x, y, speed, angle, rad=MATCH_PX):
    """Best stack S/N over the 9 grid rates around (speed, angle) within rad px of patch position (x, y)."""
    best = dict(snr=-np.inf)
    nb = np.array(neighbours(G, speed, angle))
    sx, sy, use, ngr = shifts(G, t, t_ref, grp, nb)
    yy, xx = np.mgrid[0:PATCH, 0:PATCH]
    near = np.hypot(xx - x, yy - y) <= rad
    if not near.any():
        return dict(snr=np.nan, n_exp=0)
    for k, r in enumerate(nb):
        if not use[k].any():
            continue
        snr, num, den = stack_one(EX, sx[k], sy[k], use[k])
        s = np.where(near, snr, -np.inf)
        j = np.unravel_index(np.argmax(s), s.shape)
        if s[j] > best["snr"]:
            best = dict(snr=float(s[j]), x=int(j[1]), y=int(j[0]), rate=int(r), n_exp=int(use[k].sum()),
                        n_groups=int(ngr[k]), flux_V_mJy=float(num[j] / den[j]) if den[j] > 0 else np.nan,
                        flux_err=float(1 / np.sqrt(den[j])) if den[j] > 0 else np.nan)
    if not np.isfinite(best["snr"]):
        return dict(snr=np.nan, n_exp=0, snr_logo=np.nan)
    _, best["snr_logo"], best["n_groups_data"] = robust_snr(EX, t, grp, t_ref, G, best["x"], best["y"], best["rate"])
    return best


def plant(P, movers, t_ref):
    """Planted surface brightness per picture: each mover's ePSF placed in native pixels (orrery.psf, normalised in
    the 7x7 box), sampled with the picture's own bilinear grid map. movers: list of dict(x, y [grid px at t_ref],
    vx, vy [grid px/day], V)."""
    from astropy.wcs import WCS
    from astropy.io.fits import Header
    from orrery.change import deproject
    from orrery.data import _calib
    from orrery.physics import solar_flux_mJy
    from orrery.psf import psf_native
    c0, s = (S - 1) / 2, SCALE / 3600
    out = []
    bj, bi = np.mgrid[-4:5, -4:5]
    for p in P:
        add = np.zeros((S, S), np.float32)
        w = WCS(Header.fromstring(p["wcs"]))
        ep = _calib("dr1", p["detector"])["epsf"][p["zone"]]
        col, lam = colour_map(p["lam8"])
        dt = p["mjd"] - t_ref
        for m in movers:
            gx, gy = m["x"] + m["vx"] * dt, m["y"] + m["vy"] * dt
            if not (6 <= gx <= S - 7 and 6 <= gy <= S - 7):
                continue
            ra, dec = deproject((c0 - gx) * s, (c0 - gy) * s, f_ra, f_dec)
            xs, ys = w.all_world2pix([ra], [dec], 0)
            xs, ys = float(xs[0]), float(ys[0])
            ii, jj = int(round(gx)) + bi, int(round(gy)) + bj
            bra, bdec = deproject((c0 - ii) * s, (c0 - jj) * s, f_ra, f_dec)
            nx, ny = w.all_world2pix(bra.ravel(), bdec.ravel(), 0)
            # native-box normalisation (sum over native pixels with |dx|, |dy| <= 3 equals 1)
            ax = np.arange(np.ceil(xs - 3), np.floor(xs + 3) + 1)
            ay = np.arange(np.ceil(ys - 3), np.floor(ys + 3) + 1)
            AX, AY = np.meshgrid(ax, ay)
            norm = psf_native(ep, AX - xs, AY - ys).sum()
            x0, y0 = np.floor(nx), np.floor(ny)
            fx, fy = nx - x0, ny - y0
            val = np.zeros(nx.shape)
            for ox, oy, wt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
                dx, dy = x0 + ox - xs, y0 + oy - ys
                inb = (np.abs(dx) <= 3) & (np.abs(dy) <= 3)
                val += wt * np.where(inb, psf_native(ep, dx, dy), 0)
            lam_here = float(lam[int(round(gy)), int(round(gx))])
            Vt = m["V"] + m.get("amp_mag", 0.0) * np.sin(2 * np.pi * dt * 24 / m.get("period_h", 1.0) + m.get("phase", 0.0))
            refl = max(0.2, 1 + m.get("slope_per_um", 0.0) * (lam_here - 0.55))      # reflectance relative to V
            F = 3.636e6 * 10 ** (-0.4 * Vt) * refl * solar_flux_mJy(lam_here) / solar_flux_mJy(0.55)
            amp = F / (p["omega"] / 206264.806 ** 2 * 1e9)          # MJy/sr summed over native pixels
            add[jj, ii] += (amp * val / norm).reshape(jj.shape).astype(np.float32)
        out.append(add)
    return out


f_ra = f_dec = None


def cmd_inject(name, batches):
    global f_ra, f_dec
    fi, f = field(name)
    f_ra, f_dec = f["ra"], f["dec"]
    _, P = load_pics(name)
    G = rate_grid()
    rng = np.random.default_rng(SEED_INJ + fi)
    allm = []
    for k in range(N_INJ):
        v = float(VMIN * np.exp(rng.uniform(0, np.log(VMAX / VMIN))))
        a = float(rng.uniform(0, 2 * np.pi))
        allm.append(dict(id=k, x=float(MARGIN + rng.uniform(5, PATCH - 6)), y=float(MARGIN + rng.uniform(5, PATCH - 6)),
                         speed=v, angle=a, vx=v * 3600 / SCALE * np.cos(a), vy=v * 3600 / SCALE * np.sin(a),
                         V=float(rng.uniform(16.5, 21.5))))
    # deviation 14:11 (lead's request): odd ids get a rotation light curve and a red reflectance slope
    rng2 = np.random.default_rng(SEED_INJ + 100 + fi)
    for m in allm:
        a_, p_, ph_, sl_ = rng2.uniform(0, 0.5), rng2.uniform(2, 30), rng2.uniform(0, 2 * np.pi), rng2.uniform(0, 0.3)
        if m["id"] % 2 == 1:
            m.update(modulated=True, amp_mag=float(a_), period_h=float(p_), phase=float(ph_), slope_per_um=float(sl_))
        else:
            m["modulated"] = False
    nb = N_INJ // 4
    for b in batches:
        p = f"{ROOT}/{name}/inject_{b}.json"
        if os.path.exists(p):
            continue
        t0 = time.time()
        mv = allm[b * nb:(b + 1) * nb]
        X = exposures(P, plant(P, mv, f["t_ref"]))
        EX, t, grp = ex_arrays(X)
        res = []
        for m in mv:
            r = forced(EX, t, f["t_ref"], grp, G, m["x"] - MARGIN, m["y"] - MARGIN, m["speed"], m["angle"])
            res.append(dict(m, **{f"rec_{k_}": v_ for k_, v_ in r.items()}))
        json.dump(dict(field=name, batch=b, movers=res, seconds=round(time.time() - t0)), open(p, "w"))
        s_ = np.array([r_["rec_snr"] for r_ in res])
        print(f"{name} inject {b}: median S/N {np.nanmedian(s_):.1f}, {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- known objects (IMCCE SkyBoT)
SKYBOT = "https://ssp.imcce.fr/webservices/skybot/api/conesearch.php"


def skybot(name, f, sr=0.3):
    import requests
    p = f"{ROOT}/{name}/skybot_sr{sr}.json"
    if os.path.exists(p):
        return json.load(open(p))
    jd = f["t_ref"] + 2400000.5
    q = {"EPOCH": f"{jd:.6f}", "RA": f["ra"], "DEC": f["dec"], "SR": sr, "LOC": "500", "-mime": "json", "-output": "basic"}
    for i in range(6):
        try:
            r = requests.get(SKYBOT, params=q, timeout=120)
            if r.status_code == 200:
                objs = r.json(); break
            if r.status_code == 204:
                objs = []; break
        except Exception:
            pass
        time.sleep(5 * (i + 1))
    else:
        raise RuntimeError("SkyBoT failed")
    rec = dict(url=SKYBOT, params=q, queried=time.strftime("%Y-%m-%dT%H:%M:%S%z"), objects=objs)
    json.dump(rec, open(p, "w"), indent=1)
    return rec


def known_list(name, f):
    from astropy.coordinates import Angle
    from orrery.change import world_to_grid
    out = []
    for o in skybot(name, f)["objects"]:
        ra, dec = Angle(o["RA (hms)"], "hourangle").deg, Angle(o["DEC (dms)"], "deg").deg
        pos = []
        for h in (-12.0, 12.0):
            r_ = ra + o["dRA (arcsec/h)"] * h / 3600 / np.cos(np.radians(dec))
            d_ = dec + o["dDEC (arcsec/h)"] * h / 3600
            pos.append(world_to_grid(r_, d_, S, f["ra"], f["dec"], SCALE))
        gx, gy = world_to_grid(ra, dec, S, f["ra"], f["dec"], SCALE)
        vx, vy = pos[1][0] - pos[0][0], pos[1][1] - pos[0][1]          # grid px/day
        sp = float(np.hypot(vx, vy) * SCALE / 3600)
        x, y = float(gx) - MARGIN, float(gy) - MARGIN
        out.append(dict(name=o["Name"], number=o.get("Num"), cls=o["Class"], V=float(o["VMag (mag)"]),
                        err_arcsec=float(o["Err (arcsec)"]), ra=ra, dec=dec, x=x, y=y, vx=float(vx), vy=float(vy),
                        speed=sp, angle=float(np.arctan2(vy, vx) % (2 * np.pi)),
                        in_patch=bool(3 <= x <= PATCH - 4 and 3 <= y <= PATCH - 4),
                        rate_in_grid=bool(VMIN <= sp <= VMAX)))
    return out


def cmd_known(name):
    fi, f = field(name)
    _, _, X = ex_cached(name)
    EX, t, grp = ex_arrays(X)
    G = rate_grid()
    ks = known_list(name, f)
    for k in ks:
        if k["in_patch"] and k["rate_in_grid"]:
            _, _, use, ngr = shifts(G, t, f["t_ref"], grp, np.array(neighbours(G, k["speed"], k["angle"])))
            k["trackable"] = bool(use.any())
            k["forced"] = forced(EX, t, f["t_ref"], grp, G, k["x"], k["y"], k["speed"], k["angle"]) if k["trackable"] else None
    json.dump(dict(field=name, t_ref=f["t_ref"], objects=ks), open(f"{ROOT}/{name}/known.json", "w"), indent=1)
    s = [k for k in ks if k.get("trackable")]
    print(name, f"{len(ks)} SkyBoT objects, {len(s)} searchable+trackable:",
          [(k["name"], round(k["V"], 1), round(k["speed"], 3), round(k["forced"]["snr"], 1)) for k in s], flush=True)


def cmd_real(name):
    c = git_committed(PROTO)
    if not c:
        sys.exit(f"{PROTO} not committed; refusing the real search")
    p = f"{ROOT}/{name}/real.json"
    if os.path.exists(p):
        return
    fi, f = field(name)
    _, _, X = ex_cached(name)
    EX, t, grp = ex_arrays(X)
    G = rate_grid()
    t0 = time.time()
    started = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    peaks, nd = search(EX, t, f["t_ref"], grp, G, log=f"{name} real")
    cc = nms(peaks, G)
    json.dump(dict(field=name, started=started, protocol_commit=c, n_rates_distinct=nd, n_peaks=len(peaks),
                   seconds=round(time.time() - t0), candidates=cand_records(cc, G)), open(p, "w"))
    print(f"{name} real: {len(cc)} cands >= {PRE}, max {cc[:, 3].max() if len(cc) else 0:.2f}", flush=True)


def cmd_robust(name):
    """Add S/N_LOGO to every stored candidate of the null searches and (if present) the real search."""
    fi, f = field(name)
    _, _, X = ex_cached(name)
    EX, t, grp = ex_arrays(X)
    G = rate_grid()
    files = [(f"{ROOT}/{name}/null_{k}.json", True) for k in range(N_PERM)] + [(f"{ROOT}/{name}/real.json", False)]
    for pth, is_null in files:
        if not os.path.exists(pth):
            continue
        d = json.load(open(pth))
        if d.get("robust_done"):
            continue
        if is_null:
            perm = np.array(d["perm"])
            tt, gg = t[perm], grp[perm]
        else:
            tt, gg = t, grp
        for c in d["candidates"]:
            full, logo, ng = robust_snr(EX, tt, gg, f["t_ref"], G, c["x"], c["y"], c["rate"])
            c["snr_logo"] = None if not np.isfinite(logo) else round(float(logo), 3)
            c["n_groups_data"] = int(ng)
        d["robust_done"] = True
        json.dump(d, open(pth + ".tmp", "w"))
        os.replace(pth + ".tmp", pth)
        lo = [c["snr_logo"] for c in d["candidates"] if c["snr_logo"] is not None]
        print(name, os.path.basename(pth), "max LOGO", max(lo) if lo else None, flush=True)


# ---------------------------------------------------------------- final: threshold, recovery, filters
def gaia_bright(name, f, r=0.65, gmax=14):
    import pandas as pd
    p = f"{ROOT}/{name}/gaia_g{gmax}.parquet"
    if os.path.exists(p):
        return pd.read_parquet(p)
    import pyvo
    q = (f"SELECT source_id, ra, dec, phot_g_mean_mag FROM gaiadr3.gaia_source WHERE 1=CONTAINS(POINT('ICRS',ra,dec),"
         f"CIRCLE('ICRS',{f['ra']},{f['dec']},{r})) AND phot_g_mean_mag < {gmax}")
    for i in range(4):
        try:
            t = pyvo.dal.TAPService("https://gaia.aip.de/tap").run_sync(q, maxrec=200000).to_table().to_pandas()
            t.to_parquet(p)
            return t
        except Exception as e:
            print("gaia retry", e, flush=True)
            time.sleep(10 * (i + 1))
    raise RuntimeError("gaia failed")


def threshold(fields):
    s, n = [], 0
    per = {}
    for nm in fields:
        for k in range(N_PERM):
            d = json.load(open(f"{ROOT}/{nm}/null_{k}.json"))
            n += 1
            v = [c["snr_logo"] for c in d["candidates"] if c.get("snr_logo") is not None]
            s += v
            per.setdefault(nm, []).append(v)
    s = np.sort(s)[::-1]
    kmax = int(np.floor(FALSE_PER_FIELD * n + 1e-9))
    T = float(s[kmax]) if len(s) > kmax else float(s[-1])          # detection: S/N_LOGO strictly above T
    all_mean = []
    for nm in fields:
        for k in range(N_PERM):
            all_mean += [c["snr"] for c in json.load(open(f"{ROOT}/{nm}/null_{k}.json"))["candidates"]]
    return T, dict(n_null_searches=n, allowed_above=kmax, top_null_snr=[round(float(x), 3) for x in s[:20]],
                   per_field_above_T={nm: int(sum(np.sum(np.array(v) > T) for v in vv)) for nm, vv in per.items()},
                   per_field_max={nm: [round(max(v) if v else 0, 3) for v in vv] for nm, vv in per.items()},
                   hist_edges=list(np.arange(5, 10.01, 0.25).round(2)),
                   hist_counts=np.histogram(s, np.arange(5, 10.01, 0.25))[0].tolist(),
                   statistic="S/N_LOGO (stack S/N without the visit that contributes most)",
                   top_null_mean_snr=[round(float(x), 3) for x in np.sort(all_mean)[::-1][:20]])


def track_pixels(EX, t, t_ref, G, c):
    """Per exposure: used?, grid (x, y) of the trajectory (integer shifts as in the search)."""
    r = c["rate"]
    sx, sy, use, _ = shifts(G, t, t_ref, np.array([e["group"] for e in EX]), np.array([r]))
    return use[0], MARGIN + c["x"] + sx[0], MARGIN + c["y"] + sy[0]


def checks(name, f, P, X, c, G, gaia):
    """Frozen unknown-candidate filters (1)-(6) for one candidate (patch x, y, grid rate)."""
    from astropy.wcs import WCS
    from astropy.io.fits import Header
    from scipy.ndimage import label
    from orrery.change import deproject
    EX, t, grp = ex_arrays(X)
    use, gx, gy = track_pixels(EX, t, f["t_ref"], G, c)
    ee = np.where(use)[0]
    fw = np.array([EX[e]["Fw"][gy[e], gx[e]] for e in ee])
    ww = np.array([EX[e]["w"][gy[e], gx[e]] for e in ee])
    ok = ww > 0
    res = dict(n_exp=int(ok.sum()))
    halves = []
    for h in (0, 1):
        sel = ok & (np.arange(len(ee)) % 2 == h)
        halves.append(float(fw[sel].sum() / np.sqrt(ww[sel].sum())) if ww[sel].sum() > 0 else 0.0)
    res["halves_snr"] = [round(x, 2) for x in halves]
    res["f1_halves"] = bool(min(halves) >= 3)
    # (2) per-detector partial stacks from the unmerged pictures (unclipped, same tau weighting)
    tau = X["tau"]
    v = np.array([G["vx"][c["rate"]], G["vy"][c["rate"]]])
    per_det = {}
    for e, p in enumerate(P):
        q = X["pic"][e]
        if q is None:
            continue
        dt = p["mjd"] - f["t_ref"]
        if np.hypot(*v) * abs(dt) > DMAX or abs(dt) > HALF_SPAN_D:
            continue
        x_, y_ = MARGIN + c["x"] + int(np.rint(v[0] * dt)), MARGIN + c["y"] + int(np.rint(v[1] * dt))
        w_ = q["w"][y_, x_] / tau[y_, x_] ** 2 if np.isfinite(tau[y_, x_]) else 0
        if w_ > 0:
            a = per_det.setdefault(p["detector"], [0.0, 0.0])
            a[0] += q["F"][y_, x_] * w_
            a[1] += w_
    det_snr = {int(k): round(float(a[0] / np.sqrt(a[1])), 2) for k, a in per_det.items()}
    res["detector_snr"] = det_snr
    res["f2_detectors"] = bool(sum(s_ >= 2 for s_ in det_snr.values()) >= 2)
    # (5) not dominated by 1-2 exposures
    o = np.argsort(-fw)
    rest = o[2:]
    res["snr_without_top2"] = round(float(fw[rest].sum() / np.sqrt(ww[rest].sum())), 2) if ww[rest].sum() > 0 else 0.0
    res["median_exposure_flux"] = float(np.median(fw[ok] / ww[ok])) if ok.any() else 0.0
    res["f5_not_dominated"] = bool(res["snr_without_top2"] >= 4 and res["median_exposure_flux"] > 0)
    tot = fw[ok].sum() / np.sqrt(ww[ok].sum()) if ok.any() else 0.0
    res["max_single_share"] = round(float(fw[ok].max() / np.sqrt(ww[ok].sum()) / tot), 3) if tot > 0 else None
    res["f8_single_share"] = bool(tot > 0 and res["max_single_share"] <= 0.3)
    te = t[ee]
    el = []
    for sel in (ok & (te <= np.median(te)), ok & (te > np.median(te))):
        el.append(float(fw[sel].sum() / np.sqrt(ww[sel].sum())) if ww[sel].sum() > 0 else 0.0)
    res["early_late_snr"] = [round(x, 2) for x in el]
    res["f9_early_late"] = bool(min(el) >= 2.5)
    # (3), (4), (6) per contributing picture
    c0, s = (S - 1) / 2, SCALE / 3600
    gra, gdec = gaia.ra.values, gaia.dec.values
    gG = gaia.phot_g_mean_mag.values
    near_star, near_cluster, xtalk, bloom = False, False, False, False
    for e, w_e in zip(ee, ww):
        if w_e <= 0:                       # deviation 15:04: only pictures that contribute to the stack
            continue
        for pi in X["ex"][e]["pics"] if "pics" in X["ex"][e] else []:
            p = P[pi]
            x_, y_ = gx[e], gy[e]
            ra, dec = deproject((c0 - x_) * s, (c0 - y_) * s, f["ra"], f["dec"])
            cosd = np.cos(np.radians(dec))
            sep = np.hypot((gra - ra) * cosd, gdec - dec) * 3600
            if np.any((sep < 15) & (gG < 14)):
                near_star = True
            m = p["mask"][max(0, y_ - 6):y_ + 7, max(0, x_ - 6):x_ + 7]
            lab, n = label(m)
            if n:
                yy, xx = np.mgrid[max(0, y_ - 6):y_ + 7, max(0, x_ - 6):x_ + 7]
                for k in range(1, n + 1):
                    sel = lab == k
                    # deviation 15:04: one bad native pixel masks a 2x2 grid block (bilinear), so a cluster of >= 3
                    # native pixels is >= 8 connected grid pixels
                    if sel.sum() >= 8 and np.min(np.hypot(xx[sel] - x_, yy[sel] - y_)) <= MATCH_PX:
                        near_cluster = True
            w = WCS(Header.fromstring(p["wcs"]))
            cx, cy = w.all_world2pix([ra], [dec], 0)
            b = gG < 12
            if b.any():
                sx_, sy_ = w.all_world2pix(gra[b], gdec[b], 0)
                dx, dy = cx[0] - sx_, cy[0] - sy_
                for off in (32, -32, 64, -64, 128, -128, 192, -192):
                    if np.any((np.abs(dx - off) <= 2) & (np.abs(dy) <= 2)) or np.any((np.abs(dy - off) <= 2) & (np.abs(dx) <= 2)):
                        xtalk = True
                vb = gG[b] < 7
                if vb.any() and np.any(np.hypot(dx[vb], dy[vb]) <= 100):
                    bloom = True
    res["f3_gaia"] = not near_star
    res["f4_flag_cluster"] = not near_cluster
    res["f6_crosstalk_bloom"] = not (xtalk or bloom)
    res["pass_all"] = all(res[k] for k in ("f1_halves", "f2_detectors", "f3_gaia", "f4_flag_cluster",
                                             "f5_not_dominated", "f6_crosstalk_bloom", "f8_single_share",
                                             "f9_early_late"))
    return res


def logistic_fit(V, rec):
    from scipy.optimize import minimize
    V, rec = np.asarray(V, float), np.asarray(rec, float)

    def nll(p):
        cmax, v50, w = 1 / (1 + np.exp(-p[0])), p[1], np.exp(p[2])
        pr = np.clip(cmax / (1 + np.exp((V - v50) / w)), 1e-6, 1 - 1e-6)
        return -np.sum(rec * np.log(pr) + (1 - rec) * np.log(1 - pr))
    r = minimize(nll, [2.0, 18.5, np.log(0.3)], method="Nelder-Mead")
    cmax, v50, w = 1 / (1 + np.exp(-r.x[0])), r.x[1], np.exp(r.x[2])
    # bootstrap error on V50
    rng = np.random.default_rng(5)
    bs = []
    for _ in range(200):
        i = rng.integers(0, len(V), len(V))
        rr = minimize(lambda p: nll_b(p, V[i], rec[i]), r.x, method="Nelder-Mead")
        bs.append(rr.x[1])
    return dict(cmax=float(cmax), V50=float(v50), width=float(w), V50_err=float(np.std(bs)))


def nll_b(p, V, rec):
    cmax, v50, w = 1 / (1 + np.exp(-p[0])), p[1], np.exp(p[2])
    pr = np.clip(cmax / (1 + np.exp((V - v50) / w)), 1e-6, 1 - 1e-6)
    return -np.sum(rec * np.log(pr) + (1 - rec) * np.log(1 - pr))


def matches_known(c, ks, G):
    for k in ks:
        if not k["in_patch"] and not (-5 <= k["x"] <= PATCH + 5 and -5 <= k["y"] <= PATCH + 5):
            continue
        d = np.hypot(c["x"] - k["x"], c["y"] - k["y"])
        a, b = np.radians(c["angle_deg"]), k["angle"]
        dv = np.hypot(c["speed"] * np.cos(a) - k["speed"] * np.cos(b), c["speed"] * np.sin(a) - k["speed"] * np.sin(b))
        if d <= MATCH_PX and dv <= MATCH_RATE:
            return k["name"]
    return None


def complete(nm):
    d = f"{ROOT}/{nm}"
    return all(os.path.exists(f"{d}/{x}") for x in [f"null_{k}.json" for k in range(N_PERM)] +
               [f"inject_{b}.json" for b in range(4)] + ["real.json", "known.json"])


def V_of(fV):
    return -2.5 * np.log10(np.maximum(fV, 1e-12) / 3.636e6)


def posthoc_summary(done, G):
    """POST HOC sensitivity variant: detection statistic S/N_min (see docs/protocols/E17.md, 15:04)."""
    s, n = [], 0
    for nm in done:
        for k in range(N_PERM):
            d = json.load(open(f"{ROOT}/{nm}/null_{k}.json"))
            if not d.get("min_done"):
                return None
            n += 1
            s += [c["snr_min"] for c in d["candidates"] if c.get("snr_min") is not None]
    s = np.sort(s)[::-1]
    kmax = int(np.floor(FALSE_PER_FIELD * n + 1e-9))
    T = float(s[kmax])
    inj = []
    for nm in done:
        for b in range(4):
            inj += json.load(open(f"{ROOT}/{nm}/inject_{b}.json"))["movers"]
    V = np.array([m["V"] for m in inj])
    rec = np.array([(m.get("rec_snr_min") or -1) > T for m in inj])
    trk = np.array([m.get("rec_n_exp", 0) > 0 for m in inj])
    out = dict(statistic="S/N_min = min over visits of the stack S/N without that visit (post hoc)", threshold=T,
               top_null=[round(float(x), 3) for x in s[:12]], n_null_searches=n, injections_recovered=int(rec.sum()),
               fit_all=logistic_fit(V, rec), fit_trackable=logistic_fit(V[trk], rec[trk]), known=[], candidates=[])
    for nm in done:
        ks = json.load(open(f"{ROOT}/{nm}/known.json"))["objects"]
        real = json.load(open(f"{ROOT}/{nm}/real.json"))
        above = [c for c in real["candidates"] if (c.get("snr_min") or -1) > T]
        for c in above:
            kn = matches_known(c, ks, G)
            out["candidates"].append(dict(field=nm, x=c["x"], y=c["y"], speed=c["speed"], angle_deg=c["angle_deg"],
                                          snr=c["snr"], snr_min=c["snr_min"], known=kn))
        for k in ks:
            if k.get("trackable"):
                out["known"].append(dict(field=nm, name=k["name"], V=k["V"],
                                         recovered=any(c["known"] == k["name"] for c in out["candidates"] if c["field"] == nm)))
    out["known_recovered"] = sum(r["recovered"] for r in out["known"])
    out["unknown_above_T"] = [c for c in out["candidates"] if c["known"] is None]
    return out


def cmd_final():
    from orrery.change import deproject
    allf = [f["field"] for f in draw()["fields"]]
    done = [nm for nm in allf if complete(nm)]
    T, nullstat = threshold(done)
    G = rate_grid()
    # injections
    inj = []
    for nm in done:
        for b in range(4):
            for m in json.load(open(f"{ROOT}/{nm}/inject_{b}.json"))["movers"]:
                m["field"] = nm
                m["recovered"] = bool(m.get("rec_snr_logo") is not None and np.isfinite(m["rec_snr_logo"]) and m["rec_snr_logo"] > T)
                inj.append(m)
    V = np.array([m["V"] for m in inj]); rec = np.array([m["recovered"] for m in inj])
    sp = np.array([m["speed"] for m in inj])
    trk = np.array([m.get("rec_n_exp", 0) > 0 for m in inj])
    fit_all = logistic_fit(V, rec)
    fit_trk = logistic_fit(V[trk], rec[trk])
    mod = np.array([m.get("modulated", False) for m in inj])
    fit_const = logistic_fit(V[trk & ~mod], rec[trk & ~mod])
    fit_mod = logistic_fit(V[trk & mod], rec[trk & mod])
    hi = np.array([m.get("rec_snr", 0) > 10 for m in inj])
    ratio = np.array([m.get("rec_flux_V_mJy", np.nan) / (3.636e6 * 10 ** (-0.4 * m["V"])) for m in inj])
    fscale = float(np.nanmedian(ratio[hi])) if hi.sum() >= 5 else 1.0
    fscale_err = float(1.4826 * np.nanmedian(np.abs(ratio[hi] - fscale)) / np.sqrt(hi.sum())) if hi.sum() >= 5 else None
    sbins = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
    by_speed = []
    for a, b in zip(sbins[:-1], sbins[1:]):
        s_ = (sp >= a) & (sp < b)
        bright = s_ & (V < 18.0)
        by_speed.append(dict(speed=[a, b], n=int(s_.sum()), trackable=int((s_ & trk).sum()),
                             recovered_all_V=int((s_ & rec).sum()), n_V_lt_18=int(bright.sum()),
                             recovered_V_lt_18=int((bright & rec).sum())))
    vb = np.arange(16.5, 21.51, 0.5)
    curve = [dict(V=[float(a), float(b)], n=int(((V >= a) & (V < b)).sum()), rec=int((rec & (V >= a) & (V < b)).sum()),
                  n_trackable=int(((V >= a) & (V < b) & trk).sum()), rec_trackable=int((rec & trk & (V >= a) & (V < b)).sum()),
                  n_trk_const=int(((V >= a) & (V < b) & trk & ~mod).sum()), rec_trk_const=int((rec & trk & ~mod & (V >= a) & (V < b)).sum()),
                  n_trk_mod=int(((V >= a) & (V < b) & trk & mod).sum()), rec_trk_mod=int((rec & trk & mod & (V >= a) & (V < b)).sum()))
             for a, b in zip(vb[:-1], vb[1:])]
    mjy10 = lambda v: 3.636e6 * 10 ** (-0.4 * v) * 1.2969          # mJy at 1.0 um, solar colour
    # known objects and real candidates
    known_rows, cands_out = [], []
    for nm in done:
        fi, f = field(nm)
        ks = json.load(open(f"{ROOT}/{nm}/known.json"))["objects"]
        real = json.load(open(f"{ROOT}/{nm}/real.json"))
        above = [c for c in real["candidates"] if c.get("snr_logo") is not None and c["snr_logo"] > T]
        for c in above:
            c["known"] = matches_known(c, ks, G)
        P = X = gaia = None
        for k in ks:
            if not k.get("trackable"):
                continue
            m = [c for c in above if c["known"] == k["name"]]
            row = dict(field=nm, name=k["name"], number=k["number"], cls=k["cls"], V=k["V"], speed=round(k["speed"], 4),
                       forced_snr=round(k["forced"]["snr"], 2) if k["forced"] and np.isfinite(k["forced"]["snr"]) else None,
                       forced_flux_V_mJy=k["forced"].get("flux_V_mJy") if k["forced"] else None,
                       forced_flux_err=k["forced"].get("flux_err") if k["forced"] else None,
                       forced_snr_logo=k["forced"].get("snr_logo") if k["forced"] else None,
                       recovered=bool(m), snr=m[0]["snr"] if m else None, snr_logo=m[0]["snr_logo"] if m else None)
            known_rows.append(row)
        unk = [c for c in above if c["known"] is None]
        print(f"  final {nm}: {len(above)} above T, {len(unk)} not matching a known object", flush=True)
        if unk or any(r["recovered"] for r in known_rows if r["field"] == nm):
            _, P = load_pics(nm)
            X = exposures(P)
            gaia = gaia_bright(nm, f)
        for row in known_rows:
            if row["field"] == nm and row["recovered"]:
                c = [c for c in above if c["known"] == row["name"]][0]
                row["filters"] = checks(nm, f, P, X, c, G, gaia)
        for c in unk:
            ch = checks(nm, f, P, X, c, G, gaia)
            c0, s = (S - 1) / 2, SCALE / 3600
            ra, dec = deproject((c0 - (MARGIN + c["x"])) * s, (c0 - (MARGIN + c["y"])) * s, f["ra"], f["dec"])
            EX, t, grp = ex_arrays(X)
            fr = forced(EX, t, f["t_ref"], grp, G, c["x"], c["y"], c["speed"], np.radians(c["angle_deg"]), rad=0.5)
            fV = fr.get("flux_V_mJy", np.nan) / fscale
            fe = fr.get("flux_err", np.nan) / fscale
            # sky motion direction: grid x grows to the west, y grows to the south
            a = np.radians(c["angle_deg"])
            pa = float(np.degrees(np.arctan2(-np.cos(a), -np.sin(a))) % 360)        # east of north
            cands_out.append(dict(field=nm, **c, ra_tref=float(ra), dec_tref=float(dec), t_ref_mjd=f["t_ref"],
                                  position_angle_deg=round(pa, 1), flux_V_mJy=fV, flux_err=fe,
                                  V=float(V_of(fV)) if fV > 0 else None, checks=ch))
    kr = [r for r in known_rows]
    nrec = sum(r["recovered"] for r in kr)
    vlim = None
    for vcut in np.arange(15.0, 21.51, 0.5):
        sub = [r for r in kr if r["V"] < vcut]
        if sub and sum(r["recovered"] for r in sub) / len(sub) >= 0.5:
            vlim = float(vcut)
    surv = [c for c in cands_out if c["checks"]["pass_all"]]
    out = dict(protocol=PROTO, protocol_commit=git_committed(PROTO), code="scripts/run_e17.py",
               fields_drawn=allf, fields_complete=done,
               fields=[{k: v for k, v in field(nm)[1].items() if k != "keys"} | json.load(open(f"{ROOT}/{nm}/ex_meta.json"))
                       for nm in done],
               grid=dict(scale_arcsec=SCALE, patch_px=PATCH, stamp_px=S, dmax_deg=DMAX_DEG, half_span_days=HALF_SPAN_D,
                         speeds=[VMIN, VMAX], n_speeds=G["nsp"], n_directions=G["nan"], n_rates=len(G["vx"]),
                         relative_step=STEP, trail_error_px=TRAIL, matched_filter_fwhm_px=FWHM_PX, clip=CLIP),
               threshold=dict(snr=T, rule="S/N strictly above the (k+1)-th highest pooled null candidate, k = 0.1 x "
                                         "number of null searches", **nullstat),
               injections=dict(n=len(inj), n_trackable=int(trk.sum()), recovered=int(rec.sum()),
                               fit_all=fit_all, fit_trackable=fit_trk, fit_trackable_constant=fit_const,
                               fit_trackable_modulated=fit_mod,
                               V50_all_mJy_1um=mjy10(fit_all["V50"]), V50_trackable_mJy_1um=mjy10(fit_trk["V50"]),
                               curve=curve, by_speed=by_speed, flux_scale_measured_over_true=fscale,
                               flux_scale_err=fscale_err,
                               note="trackable = at least 2 groups within 0.15 deg of motion of t_ref at the mover's rate; "
                                    "mJy at 1.0 um for a solar-coloured source (= 1.297 x F_V)",
                               movers=inj),
               known=dict(service="IMCCE SkyBoT conesearch (LOC=500), queried during this run", n_searchable=len(kr),
                          n_recovered=nrec, V_limit_50pct=vlim, rows=kr),
               candidates_above_T=cands_out, unknown_survivors=surv,
               expected_false_candidates=FALSE_PER_FIELD * len(done),
               look_elsewhere=dict(distinct_shift_patterns={nm: json.load(open(f"{ROOT}/{nm}/real.json"))["n_rates_distinct"] for nm in done},
                                   positions_per_pattern=PATCH * PATCH,
                                   trajectories_total=int(sum(json.load(open(f"{ROOT}/{nm}/real.json"))["n_rates_distinct"] for nm in done) * PATCH * PATCH),
                                   note="trajectories are strongly correlated (neighbouring rates and pixels share pictures); the "
                                        "false-candidate rate is calibrated empirically on the time-permuted nulls, not from this count"))
    out["posthoc_snr_min"] = posthoc_summary(done, G)
    json.dump(out, open(OUT, "w"), indent=1, default=float)
    print(f"T = {T:.2f}; injections V50 = {fit_all['V50']:.2f} ± {fit_all['V50_err']:.2f} (trackable {fit_trk['V50']:.2f}); "
          f"known {nrec}/{len(kr)}; unknown above T {len(cands_out)}, survivors {len(surv)}; fscale {fscale:.3f}")


# ---------------------------------------------------------------- figures
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 15, "axes.titlesize": 16, "axes.labelsize": 16, "figure.facecolor": "white",
                         "axes.facecolor": "white", "savefig.facecolor": "white"})
    return plt


def fig_recovery(R):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(10, 6.2))
    cv = R["injections"]["curve"]
    xc = [np.mean(c["V"]) for c in cv]
    for key_n, key_r, lab, col, mk in (("n", "rec", "planted test objects (all speeds)", "#1f77b4", "o"),
                                         ("n_trackable", "rec_trackable", "planted, seen in ≥ 2 visits", "#2ca02c", "s")):
        n = np.array([c[key_n] for c in cv], float); k = np.array([c[key_r] for c in cv], float)
        fr = np.where(n > 0, k / np.maximum(n, 1), np.nan)
        err = np.sqrt(np.maximum(fr * (1 - fr), 0.25 / np.maximum(n, 1)) / np.maximum(n, 1))
        ax.errorbar(xc, fr, yerr=err, marker=mk, color=col, lw=2, ms=8, capsize=3, label=lab)
    kr = R["known"]["rows"]
    if kr:
        kv = np.array([r["V"] for r in kr]); kk = np.array([r["recovered"] for r in kr])
        bins = np.arange(np.floor(kv.min()), np.ceil(kv.max()) + 1.01, 1.0)
        for a, b in zip(bins[:-1], bins[1:]):
            s = (kv >= a) & (kv < b)
            if s.sum():
                ax.plot((a + b) / 2, kk[s].mean(), "D", color="#d62728", ms=11, zorder=5)
                ax.annotate(f"{int(kk[s].sum())}/{int(s.sum())}", ((a + b) / 2, kk[s].mean()), textcoords="offset points",
                            xytext=(8, 6), color="#d62728", fontsize=13)
        ax.plot([], [], "D", color="#d62728", ms=11, label="catalogued asteroids (found / present)")
    ax.axhline(0.5, color="grey", ls=":", lw=1)
    ax.set_xlabel("brightness (V magnitude, fainter →)")
    ax.set_ylabel("fraction found by the blind search")
    ax.set_ylim(-0.03, 1.08)
    T = R["threshold"]["snr"]
    ax.set_title(f"Blind moving-object search in {len(R['fields_complete'])} SPHEREx sky patches near the ecliptic",
                 pad=12)
    sec = ax.secondary_xaxis("top", functions=(lambda v: 3.636e6 * 10 ** (-0.4 * np.asarray(v)) * 1.297,
                                                lambda m: -2.5 * np.log10(np.maximum(np.asarray(m), 1e-9) / (3.636e6 * 1.297))))
    sec.set_xticks([1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005])
    sec.set_xticklabels(["1", "0.5", "0.2", "0.1", "0.05", "0.02", "0.01", "0.005"])
    sec.set_xlabel("brightness at 1 µm (mJy, sun-coloured)")
    ax.text(0.98, 0.62, f"threshold chosen on shuffled-time\ncontrols: ≤ 0.1 false detection\nper patch "
                        f"({R['threshold']['n_null_searches']} control searches)", transform=ax.transAxes, ha="right",
            fontsize=12, color="#444")
    ax.legend(loc="lower left", fontsize=13, frameon=False)
    fig.tight_layout()
    fig.savefig("site/data/figs/e17_recovery.png", dpi=130)
    plt.close(fig)


def stamps_along(nm, c, G, half=10):
    """Per exposure S/N stamps centred on the track, the shift-and-stack stamp, and the zero-motion stack."""
    fi, f = field(nm)
    _, P, X = ex_cached(nm)
    EX, t, grp = ex_arrays(X)
    use, gx, gy = track_pixels(EX, t, f["t_ref"], G, c)
    singles = []
    for e in np.where(use)[0]:
        sl = (slice(gy[e] - half, gy[e] + half + 1), slice(gx[e] - half, gx[e] + half + 1))
        w = EX[e]["w"][sl]
        singles.append((t[e] - f["t_ref"], np.where(w > 0, EX[e]["Fw"][sl] / np.sqrt(np.where(w > 0, w, 1)), np.nan)))
    num = sum(EX[e]["Fw"][gy[e] - half:gy[e] + half + 1, gx[e] - half:gx[e] + half + 1] for e in np.where(use)[0])
    den = sum(EX[e]["w"][gy[e] - half:gy[e] + half + 1, gx[e] - half:gx[e] + half + 1] for e in np.where(use)[0])
    stack = np.where(den > 0, num / np.sqrt(np.where(den > 0, den, 1)), np.nan)
    y0, x0 = MARGIN + c["y"], MARGIN + c["x"]
    num0 = sum(EX[e]["Fw"][y0 - half:y0 + half + 1, x0 - half:x0 + half + 1] for e in np.where(use)[0])
    den0 = sum(EX[e]["w"][y0 - half:y0 + half + 1, x0 - half:x0 + half + 1] for e in np.where(use)[0])
    static = np.where(den0 > 0, num0 / np.sqrt(np.where(den0 > 0, den0, 1)), np.nan)
    return singles, stack, static, (gx[use], gy[use])


def fig_strip(nm, c, G, path, title, nmax=12):
    plt = _plt()
    singles, stack, static, (tx, ty) = stamps_along(nm, c, G)
    singles = singles[:nmax]
    n = len(singles)
    fig, axs = plt.subplots(1, n + 2, figsize=(1.9 * (n + 2), 3.4))
    kw = dict(cmap="gray_r", vmin=-3, vmax=6, origin="upper", interpolation="nearest")
    for a, (dt, im) in zip(axs, singles):
        a.imshow(im, **kw)
        a.set_title(f"{dt * 24:+.1f} h", fontsize=12)
        a.set_xticks([]); a.set_yticks([])
        a.add_patch(plt.Circle((10, 10), 3, fill=False, color="#d62728", lw=1.2))
    axs[n].imshow(static, **kw); axs[n].set_title("added, no motion", fontsize=12)
    axs[n + 1].imshow(stack, **kw); axs[n + 1].set_title("added along track", fontsize=12, color="#d62728")
    for a in axs[n:]:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_example(R):
    """The faintest recovered known asteroid with a clear stack: single pictures vs shift-and-stack, with its track."""
    plt = _plt()
    G = rate_grid()
    rec = [r for r in R["known"]["rows"] if r["recovered"]]
    if not rec:
        return None
    rec.sort(key=lambda r: (r["snr"] < 8, -r["V"]))          # faintest with S/N >= 8 preferred
    r = rec[0]
    nm = r["field"]
    fi, f = field(nm)
    real = json.load(open(f"{ROOT}/{nm}/real.json"))
    T = R["threshold"]["snr"]
    c = [c for c in real["candidates"] if (c.get("snr_logo") or 0) > T and matches_known(c, json.load(open(f"{ROOT}/{nm}/known.json"))["objects"], G) == r["name"]][0]
    singles, stack, static, (tx, ty) = stamps_along(nm, c, G)
    _, P, X = ex_cached(nm)
    EX, t, grp = ex_arrays(X)
    use, gx, gy = track_pixels(EX, t, f["t_ref"], G, c)
    e_mid = np.where(use)[0][np.argmin(np.abs(t[use] - f["t_ref"]))]
    cx, cy = MARGIN + c["x"], MARGIN + c["y"]
    h = int(max(25, np.max(np.abs(tx - cx)), np.max(np.abs(ty - cy))) + 8)
    sl = (slice(cy - h, cy + h + 1), slice(cx - h, cx + h + 1))
    w = EX[e_mid]["w"][sl]
    big = np.where(w > 0, EX[e_mid]["Fw"][sl] / np.sqrt(np.where(w > 0, w, 1)), np.nan)
    pick = np.linspace(0, len(singles) - 1, min(6, len(singles))).round().astype(int)
    fig = plt.figure(figsize=(15, 6.4))
    gs = fig.add_gridspec(2, 7, width_ratios=[2.6, 1, 1, 1, 0.15, 1.4, 1.4])
    a0 = fig.add_subplot(gs[:, 0])
    a0.imshow(big, cmap="gray_r", vmin=-3, vmax=6, origin="upper", interpolation="nearest")
    a0.plot(tx - (cx - h), ty - (cy - h), "-", color="#d62728", lw=1.5, alpha=0.8)
    a0.plot(tx - (cx - h), ty - (cy - h), "o", mfc="none", color="#d62728", ms=9)
    a0.set_title(f"one picture ({2 * h + 1}×{2 * h + 1} px of 6.2″)\nred: predicted track over "
                 f"{(t[use].max() - t[use].min()) * 24:.0f} h", fontsize=13)
    a0.set_xticks([]); a0.set_yticks([])
    kw = dict(cmap="gray_r", vmin=-3, vmax=6, origin="upper", interpolation="nearest")
    for k, i in enumerate(pick):
        a = fig.add_subplot(gs[k // 3, 1 + k % 3])
        a.imshow(singles[i][1], **kw)
        a.set_title(f"{singles[i][0] * 24:+.1f} h", fontsize=12)
        a.add_patch(plt.Circle((10, 10), 3, fill=False, color="#d62728", lw=1.2))
        a.set_xticks([]); a.set_yticks([])
    a5 = fig.add_subplot(gs[:, 5]); a5.imshow(static, **kw)
    a5.set_title("all pictures added\nwithout following it", fontsize=13); a5.set_xticks([]); a5.set_yticks([])
    a6 = fig.add_subplot(gs[:, 6]); a6.imshow(stack, **kw)
    a6.set_title(f"added along its motion\nS/N {c['snr']:.1f}", fontsize=13, color="#d62728")
    a6.set_xticks([]); a6.set_yticks([])
    fig.suptitle(f"Catalogued asteroid {r['name']} (predicted V = {r['V']:.1f}, {r['speed']:.2f}°/day) found blind in "
                 f"{int(use.sum())} SPHEREx exposures", fontsize=16)
    fig.tight_layout()
    fig.savefig("site/data/figs/e17_example.png", dpi=120)
    plt.close(fig)
    return r["name"]


# By-eye inspection of the per-exposure strips (written after looking; not part of any automatic decision)
INSPECTION = {
    ("e17_03", 152, 219): "rejected by filter (4); strip: the -8.2 h visit is the catalogued asteroid (3993) Sorm "
                          "(SkyBoT V = 18.0, 3.7 arcsec from the track at that time, queried 2026-11-15), linked at a wrong rate "
                          "(0.41 vs ~0.25 deg/day) to the bright detector-edge artefact of the +4.8 h visit; not a new object",
    ("e17_03", 157, 223): "same as the e17_03 (152, 219) candidate: Sorm's -8.2 h visit + detector-edge artefact at +4.8 h",
}
POSTHOC_INSPECTION = ("the 5 extra e17_00 trajectories above the post-hoc S/N_min threshold all pass through the "
                      "catalogued asteroid 1888 Zu Chong-Zhi (V = 16.0) in its -8.2 h visit (SkyBoT position matches "
                      "grid pixel 277,129) and link it to other single-visit features near the G = 5.9 star "
                      "(Gaia, 58.293 +17.327); all fail filter (6) bloom/crosstalk and/or (4)")


def cmd_figs():
    R = json.load(open(OUT))
    fig_recovery(R)
    ex = fig_example(R)
    G = rate_grid()
    surv = R["unknown_survivors"]
    for c in R["candidates_above_T"]:
        c["inspection"] = INSPECTION.get((c["field"], c["x"], c["y"]))
    if R.get("posthoc_snr_min"):
        R["posthoc_snr_min"]["inspection"] = POSTHOC_INSPECTION
    show = surv or R["candidates_above_T"]
    if show:
        plt = _plt()
        import matplotlib.image as mpimg
        paths = []
        for i, c in enumerate(show[:6]):
            p = f"{ROOT}/cand_{i}.png"
            tag = "" if c["checks"]["pass_all"] else " — REJECTED by the frozen filters"
            if c.get("inspection"):
                tag += ": catalogued asteroid Sorm (−8 h) + detector-edge artefact (+5 h)"
            fig_strip(c["field"], c, G, p, f"{c['field']} unmatched trajectory {i}: stack S/N {c['snr']:.1f}, "
                                           f"{c['speed']:.2f}°/day{tag}")
            paths.append(p)
        fig, axs = plt.subplots(len(paths), 1, figsize=(16, 3.6 * len(paths)))
        for a, p in zip(np.atleast_1d(axs), paths):
            a.imshow(mpimg.imread(p)); a.axis("off")
        fig.tight_layout(); fig.savefig("site/data/figs/e17_candidates.png", dpi=110); plt.close(fig)
    kr = R["known"]["rows"]
    nb = [r for r in kr if r["V"] < 19.5]
    R["headline"] = (f"The blind search finds {sum(r['recovered'] for r in nb)} of {len(nb)} catalogued asteroids brighter than "
                     f"V = 19.5 in the {len(R['fields_complete'])} patches (and 0 of {len(kr) - len(nb)} fainter ones, as "
                     f"expected from the measured 50% depth V = {R['injections']['fit_all']['V50']:.2f} ± "
                     f"{R['injections']['fit_all']['V50_err']:.2f}), with {R['expected_false_candidates']:.1f} false "
                     f"candidates expected; {len(R['unknown_survivors'])} moving-object candidates not matching any known "
                     f"object in IMCCE SkyBoT (queried 2026-11-15) survive the frozen filters.")
    R["deviations"] = "see docs/protocols/E17.md, section Deviations (pre-search: 14:02-14:11 PST; post hoc: 15:04 and 15:12 PST)"
    R["figures"] = dict(recovery="site/data/figs/e17_recovery.png", example="site/data/figs/e17_example.png",
                        example_object=ex, candidates="site/data/figs/e17_candidates.png" if show else None)
    json.dump(R, open(OUT, "w"), indent=1, default=float)
    print("figures done", ex)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    step = sys.argv[1]
    if step == "draw":
        cmd_draw()
    elif step == "cache":
        for n in sys.argv[2:]:
            cmd_cache(n)
    elif step == "null":
        cmd_null(sys.argv[2], [int(k) for k in sys.argv[3:]] or range(N_PERM))
    elif step == "inject":
        cmd_inject(sys.argv[2], [int(k) for k in sys.argv[3:]] or range(4))
    elif step == "known":
        for n in sys.argv[2:]:
            cmd_known(n)
    elif step == "real":
        for n in sys.argv[2:]:
            cmd_real(n)
    elif step == "robust":
        for n in sys.argv[2:]:
            cmd_robust(n)
    elif step == "posthoc":
        for n in sys.argv[2:]:
            cmd_posthoc(n)
    elif step == "final":
        cmd_final()
    elif step == "figs":
        cmd_figs()
