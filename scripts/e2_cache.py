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
from build_field import _get, candidates, load_index, read_rows  # noqa: E402
from orrery.data import Frame, Stamp, release_of, save_stamp  # noqa: E402

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


def fast_stamp(key, ra0, dec0, half=HALF):
    """Same content as orrery.data.Frame.stamp(x, y, half, with_zodi=False) but with 3 full-width row range reads
    and clipped (not None) at the detector edge. None if the field centre is off the frame."""
    fr = Frame(key)
    try:
        x, y = fr.world2pix(ra0, dec0)
        x, y = float(x[0]), float(y[0])
        if not (0 <= x <= 2039 and 0 <= y <= 2039):
            return None
        ix, iy = int(round(x)), int(round(y))
        x0, y0, x1, y1 = max(0, ix - half), max(0, iy - half), min(2040, ix + half + 1), min(2040, iy + half + 1)
        img, flg = read_rows(fr, y0, y1)
        vh = fr.hdul["VARIANCE"]
        assert vh.header["BITPIX"] == -32 and "BZERO" not in vh.header and "BSCALE" not in vh.header
        var = np.frombuffer(_get(fr.url, vh._data_offset + y0 * 8160, vh._data_offset + y1 * 8160), ">f4").reshape(-1, 2040)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        cw, cb = fr.wave_at(xx.astype(float), yy.astype(float))
        return Stamp(obs_id=fr.obs_id, detector=fr.detector, release=release_of(key), url=fr.url, mjd=fr.mjd,
                     x0=x0, y0=y0, image=np.ascontiguousarray(img[:, x0:x1]),
                     variance=np.ascontiguousarray(var[:, x0:x1], np.float32),
                     zodi=np.zeros((y1 - y0, x1 - x0), np.float32), flags=np.ascontiguousarray(flg[:, x0:x1]),
                     cwave=cw.astype(np.float32), cband=cb.astype(np.float32), omega_arcsec2=fr.omega,
                     epsf=fr.epsf_at(ix, iy), epsf_oversample=fr.epsf_ovs,
                     wcs_header=fr.wcs.to_header(relax=True).tostring())
    finally:
        fr.close()


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
