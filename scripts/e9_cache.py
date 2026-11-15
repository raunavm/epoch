"""E9 comparison-star stamps (docs/protocols/E9.md). One file per star: data/phase2/e9/stamps/<name>.npz with
image [N, S, S] float32 (MJy/sr; VARIANCE is not read, to halve the transfer), flags int32, per-picture mjd, detector, x0, y0, star position xs, ys
(full-frame 0-based pixels, Gaia DR3 position propagated to the picture epoch through the frame WCS), wavelength
lam/band at the star (um), omega (arcsec^2), key. S = 2 * HALF + 1; pictures whose
cutout is clipped by the detector edge are dropped. Frame list = scripts/build_field.py:candidates rule, capped at
250 by the data/efields rule (MJD-sorted, np.linspace). Resumable (existing star files are skipped).

  OMP_NUM_THREADS=1 .venv/bin/python scripts/e9_cache.py data/phase2/e9/download_<k>.json
"""
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

OUT = "data/phase2/e9"
HALF, CAP, THREADS, PROCS = 16, 250, 8, 6
_df = None


def lean_stamp(key, ra0, dec0, half=HALF):
    """scripts/e2_cache.py:fast_stamp without the VARIANCE read (same IMAGE / FLAGS / WCS / wavelength code path).
    Returns an orrery.data.Stamp with variance = 1, or None if the centre is off the detector."""
    import e2_cache as e
    from astropy.io.fits.hdu.compressed._codecs import Rice1
    from astropy.wcs import WCS
    from orrery.data import BUCKET, Frame, Stamp, _calib, release_of
    url = f"{BUCKET}/{key}"
    p, o = e._header(url, 0)
    h, img_off = e._header(url, o)
    assert h["EXTNAME"] == "IMAGE" and h["BITPIX"] == -32 and h["NAXIS1"] == 2040 and "BZERO" not in h
    fr = Frame.__new__(Frame)
    fr.key, fr.url, fr.header, fr.wcs = key, url, h, WCS(h)
    fr.mjd, fr.obs_id, fr.detector, fr.omega = float(h["MJD-AVG"]), h["OBSID"], int(h["DETECTOR"]), float(h["OMEGA_MEDIAN"])
    c = _calib(release_of(key), fr.detector)
    fr.epsf_tab = {"XCENTER": c["xc"], "YCENTER": c["yc"], "EPSF": c["epsf"]}
    fr.epsf_ovs, fr.wx, fr.wy, fr.wv = int(c["ovs"]), c["wx"], c["wy"], c["wv"]
    x, y = fr.world2pix(ra0, dec0)
    x, y = float(x[0]), float(y[0])
    if not (0 <= x <= 2039 and 0 <= y <= 2039):
        return None
    ix, iy = int(round(x)), int(round(y))
    x0, y0, x1, y1 = max(0, ix - half), max(0, iy - half), min(2040, ix + half + 1), min(2040, iy + half + 1)
    img = np.frombuffer(e._get(url, img_off + y0 * 8160, img_off + y1 * 8160), ">f4").reshape(-1, 2040)
    fh, fl_off = e._header(url, img_off + e._pad(2040 * 2040 * 4))
    assert fh["EXTNAME"] == "FLAGS" and fh["ZCMPTYPE"] == "RICE_1" and fh["ZTILE1"] == 2040 and fh["ZTILE2"] == 1
    assert fh["NAXIS1"] == 8 and fh["NAXIS2"] == 2040 and fh.get("THEAP", 8 * 2040) == 8 * 2040
    desc = np.frombuffer(e._get(url, fl_off + 8 * y0, fl_off + 8 * y1), ">i4").reshape(-1, 2)
    heap = fl_off + 8 * 2040
    lo, hi = int(desc[:, 1].min()), int((desc[:, 1] + desc[:, 0]).max())
    buf = e._get(url, heap + lo, heap + hi)
    codec = Rice1(blocksize=fh.get("ZVAL1", 32), bytepix=4, tilesize=2040)
    flg = np.stack([codec.decode(np.frombuffer(buf[o_ - lo:o_ - lo + n], np.uint8)) for n, o_ in desc]).astype(np.int32)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    cw, cb = fr.wave_at(xx.astype(float), yy.astype(float))
    im = np.ascontiguousarray(img[:, x0:x1], np.float32)
    return Stamp(obs_id=fr.obs_id, detector=fr.detector, release=release_of(key), url=url, mjd=fr.mjd, x0=x0, y0=y0,
                 image=im, variance=np.ones_like(im), zodi=np.zeros_like(im), flags=np.ascontiguousarray(flg[:, x0:x1]),
                 cwave=cw.astype(np.float32), cband=cb.astype(np.float32), omega_arcsec2=fr.omega,
                 epsf=fr.epsf_at(ix, iy), epsf_oversample=fr.epsf_ovs, wcs_header=fr.wcs.to_header(relax=True).tostring())


def star_pos(st, ra, dec, pmra, pmdec, mjd):
    """(xs, ys) full-frame pixel of a Gaia DR3 (J2016.0) position propagated linearly to mjd."""
    from astropy.io import fits
    from astropy.wcs import WCS
    yr = (mjd - 57388.5) / 365.25
    r = ra + (pmra or 0.0) * yr / 3.6e6 / np.cos(np.radians(dec))
    d = dec + (pmdec or 0.0) * yr / 3.6e6
    x, y = WCS(fits.Header.fromstring(st.wcs_header)).all_world2pix([r], [d], 0)
    return float(x[0]), float(y[0])


def pack(stamps, star):
    """List of orrery.data.Stamp (full-size, centred on the star's nearest pixel) -> dict of stacked arrays."""
    from scipy.ndimage import map_coordinates
    d = dict(image=[], flags=[], mjd=[], detector=[], x0=[], y0=[], xs=[], ys=[], lam=[], band=[],
             omega=[], key=[])
    for key, st in stamps:
        xs, ys = star_pos(st, star["ra"], star["dec"], star.get("pmra"), star.get("pmdec"), st.mjd)
        c = [[ys - st.y0], [xs - st.x0]]
        d["image"].append(st.image), d["flags"].append(st.flags)
        d["mjd"].append(st.mjd), d["detector"].append(st.detector), d["x0"].append(st.x0), d["y0"].append(st.y0)
        d["xs"].append(xs), d["ys"].append(ys), d["omega"].append(st.omega_arcsec2)
        d["lam"].append(float(map_coordinates(st.cwave.astype(float), c, order=1, mode="nearest")[0]))
        d["band"].append(float(map_coordinates(st.cband.astype(float), c, order=1, mode="nearest")[0]))
        d["key"].append(key)
    return {k: np.array(v) for k, v in d.items()}


def one(star):
    global _df
    import e2_cache
    from e9_select import n_frames
    path = f"{OUT}/stamps/{star['name']}.npz"
    if os.path.exists(path):
        return star["name"], -1, 0.0
    if _df is None:
        _df = e2_cache.load_index()[0]
    t = time.time()
    fr = n_frames(_df, star["ra"], star["dec"]).sort_values("mjd", kind="stable").reset_index(drop=True)
    if len(fr) > CAP:
        fr = fr.iloc[np.linspace(0, len(fr) - 1, CAP).round().astype(int)].reset_index(drop=True)
    yr = 9.9                                           # cutout centre: position at ~2025.9 (PM over a year << 1 px)
    ra = star["ra"] + (star.get("pmra") or 0.0) * yr / 3.6e6 / np.cos(np.radians(star["dec"]))
    dec = star["dec"] + (star.get("pmdec") or 0.0) * yr / 3.6e6

    def get(key):
        for i in range(3):
            try:
                return key, lean_stamp(key, ra, dec, half=HALF)
            except Exception:
                time.sleep(3 * (i + 1))
        return key, "error"

    with ThreadPoolExecutor(THREADS) as ex:
        got = list(ex.map(get, fr.key.values))
    S = 2 * HALF + 1
    ok = [(k, s) for k, s in got if s is not None and not isinstance(s, str) and s.image.shape == (S, S)]
    d = pack(ok, star)
    d["n_selected"], d["n_error"] = len(fr), sum(isinstance(s, str) for _, s in got)
    np.savez_compressed(path[:-4] + ".tmp.npz", **d)
    os.replace(path[:-4] + ".tmp.npz", path)
    return star["name"], len(ok), time.time() - t


def main():
    """argv[1] = JSON file with the ordered list of stars to fetch (written by scripts/e9_select.py)."""
    os.makedirs(f"{OUT}/stamps", exist_ok=True)
    stars = json.load(open(sys.argv[1]))
    t = time.time()
    with Pool(PROCS) as p:
        for name, k, dt in p.imap_unordered(one, stars):
            print(f"{name}: {k} pictures, {dt:.0f}s (elapsed {time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
