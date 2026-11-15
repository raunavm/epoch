"""E2 field draw + shared raw-cutout cache (docs/PROTOCOLS.md, "Common" and "E2"). Resumable: rerun to continue.

  OMP_NUM_THREADS=1 .venv/bin/python scripts/e2_cache.py

Order: calibration field `calib` (RA 7.79, Dec -65.37) first, then e2_00 ... e2_19 in draw order.
Per field: data/efields/<field>/frames/<obs_id>_D<det>.npz (orrery.data.save_stamp format, native pixels,
half = HALF, clipped at the detector edge), gaia.parquet, field.json, READY. Format: data/efields/README.md.
"""
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from astropy.coordinates import SkyCoord  # noqa: E402
import threading  # noqa: E402
import requests  # noqa: E402
from astropy.io import fits  # noqa: E402
from astropy.io.fits.hdu.compressed._codecs import Rice1  # noqa: E402
from astropy.wcs import WCS  # noqa: E402
from build_field import candidates, load_index  # noqa: E402
from orrery.data import BUCKET, Frame, Stamp, _calib, release_of, save_stamp  # noqa: E402

ROOT = "data/efields"
INDEX = "data/index/frames_dr1.parquet"
SIZE, HALF, CAP, MIN_FRAMES, WORKERS = 48, 38, 250, 40, 4
SEED = 20261115
QUOTA = {"high": 10, "mid": 5, "low": 5}          # |b| > 30, 15 <= |b| <= 30, |b| < 15
CAP_RULE = ("fields with > 250 indexed frames use 250: index rows (dqa Pass, footprint polygon contains the centre) "
            "sorted by MJD (stable), rows np.linspace(0, n-1, 250).round()")
GAIA_R_DEG = 0.1                                  # covers the 77 x 77 native cutout (half-diagonal ~ 0.085 deg)


def band(absb):
    return "high" if absb > 30 else ("mid" if absb >= 15 else "low")


def draw_fields(df):
    path = f"{ROOT}/e2_fields.json"
    if os.path.exists(path):
        return json.load(open(path))
    rng = np.random.default_rng(SEED)
    left, acc, rej = dict(QUOTA), [], []
    n = 0
    while sum(left.values()):
        ra = float(rng.uniform(0, 360))
        dec = float(np.degrees(np.arcsin(rng.uniform(-1, 1))))
        n += 1
        b = float(SkyCoord(ra, dec, unit="deg").galactic.b.deg)
        bd = band(abs(b))
        rec = dict(draw=n, ra=ra, dec=dec, b=b, band=bd)
        if left[bd] == 0:
            rej.append(dict(rec, reason="band quota full"))
            continue
        nf = len(candidates(df, ra, dec))
        rec["n_indexed"] = nf
        if nf < MIN_FRAMES:
            rej.append(dict(rec, reason=f"< {MIN_FRAMES} indexed frames"))
            continue
        left[bd] -= 1
        acc.append(dict(rec, field=f"e2_{len(acc):02d}"))
    out = dict(protocol="docs/PROTOCOLS.md E2", seed=SEED,
               rule="rng = default_rng(20261115); per draw ra = rng.uniform(0, 360) then dec = degrees(arcsin("
                    "rng.uniform(-1, 1))); |b| (astropy ICRS->Galactic) band: high |b|>30, mid 15<=|b|<=30, low |b|<15; "
                    "accepted if that band has remaining quota (10/5/5) and >= 40 indexed frames cover the centre",
               index=INDEX, n_index_frames=int(len(df)), n_draws=n, quota=QUOTA, fields=acc, rejected=rej,
               n_rejected_low_frames=sum(r["reason"].startswith("<") for r in rej),
               n_rejected_quota_full=sum(r["reason"] == "band quota full" for r in rej))
    os.makedirs(ROOT, exist_ok=True)
    json.dump(out, open(path, "w"), indent=1)
    return out


_tls = threading.local()


def _get(url, a, b):
    """Byte range [a, b) over a per-thread keep-alive session (TLS handshake once per worker)."""
    for i in range(4):
        try:
            if not hasattr(_tls, "s"):
                _tls.s = requests.Session()
            r = _tls.s.get(url, headers={"Range": f"bytes={a}-{b - 1}"}, timeout=60)
            r.raise_for_status()
            assert len(r.content) == b - a
            return r.content
        except Exception:
            _tls.__dict__.pop("s", None)
            if i == 3:
                raise
            time.sleep(2 * (i + 1))


def _header(url, off, nbytes=34560):
    """Parse the FITS header starting at byte `off`; returns (header, data offset)."""
    buf = _get(url, off, off + nbytes)
    for k in range(0, len(buf), 80):
        if buf[k:k + 8] == b"END     ":
            end = (k // 2880 + 1) * 2880
            return fits.Header.fromstring(buf[:k + 80].decode("ascii")), off + end
    return _header(url, off, 4 * nbytes)


def _pad(n):
    return (n + 2879) // 2880 * 2880


def fast_stamp(key, ra0, dec0, half=HALF):
    """Same content as orrery.data.Frame(key).stamp(x, y, half, with_zodi=False) (verified bit-identical), read with
    6 keep-alive range requests, and clipped (not None) at the detector edge. None if the field centre is off the
    frame. Layout (DR1 Level-2): PRIMARY, IMAGE f4 2040^2, FLAGS (RICE_1 tiles = rows), VARIANCE f4 2040^2, ..."""
    url = f"{BUCKET}/{key}"
    p, o = _header(url, 0)
    h, img_off = _header(url, o)
    assert h["EXTNAME"] == "IMAGE" and h["BITPIX"] == -32 and h["NAXIS1"] == 2040 and "BZERO" not in h
    fr = Frame.__new__(Frame)                    # header-only Frame: reuse its epsf_at / wave_at
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
    img = np.frombuffer(_get(url, img_off + y0 * 8160, img_off + y1 * 8160), ">f4").reshape(-1, 2040)
    fh, fl_off = _header(url, img_off + _pad(2040 * 2040 * 4))
    assert fh["EXTNAME"] == "FLAGS" and fh["ZCMPTYPE"] == "RICE_1" and fh["ZTILE1"] == 2040 and fh["ZTILE2"] == 1
    assert fh["NAXIS1"] == 8 and fh["NAXIS2"] == 2040 and fh.get("THEAP", 8 * 2040) == 8 * 2040
    desc = np.frombuffer(_get(url, fl_off + 8 * y0, fl_off + 8 * y1), ">i4").reshape(-1, 2)
    heap = fl_off + 8 * 2040
    lo, hi = int(desc[:, 1].min()), int((desc[:, 1] + desc[:, 0]).max())
    buf = _get(url, heap + lo, heap + hi)
    codec = Rice1(blocksize=fh.get("ZVAL1", 32), bytepix=4, tilesize=2040)
    flg = np.stack([codec.decode(np.frombuffer(buf[o_ - lo:o_ - lo + n], np.uint8)) for n, o_ in desc]).astype(np.int32)
    vh, var_off = _header(url, fl_off + _pad(8 * 2040 + fh["PCOUNT"]))
    assert vh["EXTNAME"] == "VARIANCE" and vh["BITPIX"] == -32 and "BZERO" not in vh and "BSCALE" not in vh
    var = np.frombuffer(_get(url, var_off + y0 * 8160, var_off + y1 * 8160), ">f4").reshape(-1, 2040)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    cw, cb = fr.wave_at(xx.astype(float), yy.astype(float))
    return Stamp(obs_id=fr.obs_id, detector=fr.detector, release=release_of(key), url=url, mjd=fr.mjd,
                 x0=x0, y0=y0, image=np.ascontiguousarray(img[:, x0:x1], np.float32),
                 variance=np.ascontiguousarray(var[:, x0:x1], np.float32),
                 zodi=np.zeros((y1 - y0, x1 - x0), np.float32), flags=np.ascontiguousarray(flg[:, x0:x1]),
                 cwave=cw.astype(np.float32), cband=cb.astype(np.float32), omega_arcsec2=fr.omega,
                 epsf=fr.epsf_at(ix, iy), epsf_oversample=fr.epsf_ovs,
                 wcs_header=fr.wcs.to_header(relax=True).tostring())


def fname(row):
    return f"{row.obs_id}_D{int(row.detector)}.npz"


def fetch(row, fdir, ra0, dec0):
    p = f"{fdir}/{fname(row)}"
    if os.path.exists(p):
        return "ok"
    st = fast_stamp(row.key, ra0, dec0)
    if st is None:
        return "off_frame"
    tmp = p[:-4] + ".tmp.npz"
    save_stamp(st, tmp)
    os.replace(tmp, p)
    return "ok"


def gaia(path, ra0, dec0):
    if os.path.exists(path):
        return True
    import pyvo
    q = (f"SELECT source_id, ra, dec, pmra, pmdec, parallax, phot_g_mean_mag, ref_epoch FROM gaiadr3.gaia_source "
         f"WHERE 1=CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra0}, {dec0}, {GAIA_R_DEG})) "
         f"AND phot_g_mean_mag < 19")
    for i in range(4):
        try:
            t = pyvo.dal.TAPService("https://gaia.aip.de/tap").run_sync(q, maxrec=100000).to_table().to_pandas()
            t.to_parquet(path + ".tmp", index=False)
            os.replace(path + ".tmp", path)
            return True
        except Exception as e:
            print(f"  gaia attempt {i + 1} failed: {e!r}", flush=True)
            time.sleep(10 * (i + 1))
    return False


def build(name, ra0, dec0, df, ex, extra):
    fdir = f"{ROOT}/{name}/frames"
    if os.path.exists(f"{ROOT}/{name}/READY"):
        print(f"{name}: READY", flush=True)
        return
    os.makedirs(fdir, exist_ok=True)
    cand = candidates(df, ra0, dec0).sort_values("mjd", kind="stable").reset_index(drop=True)
    n_idx = len(cand)
    if n_idx > CAP:
        cand = cand.iloc[np.linspace(0, n_idx - 1, CAP).round().astype(int)].reset_index(drop=True)
    t = time.time()
    status, todo = {}, list(cand.itertuples())
    for rnd in range(4):
        futs = {ex.submit(fetch, r, fdir, ra0, dec0): r for r in todo}
        todo = []
        for fu, r in futs.items():
            try:
                status[r.key] = fu.result()
            except Exception as e:
                status[r.key] = f"error: {e!r}"[:300]
                todo.append(r)
        print(f"{name}: round {rnd}: {sum(v == 'ok' for v in status.values())}/{len(cand)} ok, {len(todo)} errors, "
              f"{time.time() - t:.0f}s", flush=True)
        if not todo:
            break
        time.sleep(15)
    g_ok = gaia(f"{ROOT}/{name}/gaia.parquet", ra0, dec0)
    ok = cand[[status[k] == "ok" for k in cand.key]]
    fl = [dict(file=f"frames/{fname(r)}", key=r.key, obs_id=r.obs_id, detector=int(r.detector), mjd=float(r.mjd))
          for r in ok.itertuples()]
    rec = dict(field=name, ra=ra0, dec=dec0, size_px=SIZE, scale_arcsec=6.15, stamp_half=HALF,
               n_frames=len(fl), n_indexed=n_idx, n_selected=len(cand), frame_list=fl, index=INDEX,
               n_index_frames=int(len(df)), cap_rule=CAP_RULE,
               not_saved=[dict(key=k, status=s) for k, s in status.items() if s != "ok"],
               gaia="gaia.parquet" if g_ok else None,
               gaia_query=f"gaia.aip.de/tap gaiadr3.gaia_source, G < 19, cone {GAIA_R_DEG} deg, epoch J2016.0", **extra)
    json.dump(rec, open(f"{ROOT}/{name}/field.json", "w"), indent=1)
    if g_ok and not todo:
        open(f"{ROOT}/{name}/READY", "w").write(time.strftime("%Y-%m-%dT%H:%M:%S%z\n"))
        print(f"{name}: READY ({len(fl)} frames)", flush=True)
    else:
        print(f"{name}: NOT READY (gaia {g_ok}, {len(todo)} frame errors); rerun to resume", flush=True)


def main():
    df, src = load_index()
    assert src == INDEX, src
    fields = draw_fields(df)
    print(f"draw: {fields['n_draws']} draws, {len(fields['fields'])} fields", flush=True)
    with ThreadPoolExecutor(WORKERS) as ex:
        build("calib", 7.79, -65.37, df, ex, dict(role="calibration field (tuning only)"))
        for f in fields["fields"]:
            build(f["field"], f["ra"], f["dec"], df, ex, dict(role="E2 test field", b=f["b"], band=f["band"],
                                                                draw=f["draw"]))


if __name__ == "__main__":
    main()
