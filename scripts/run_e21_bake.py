"""E21 step 2: bake the DR1 North Ecliptic Pole stack of detector-1 stamps.

Adapted from Tracklet's scripts/bake_field.py (MIT, Team Tracklet): same frame rule
(`gap_series`: farthest first in time, >= 2 d apart, `keeps_sight` to 2,000 AU, at most 30 visits)
and the same per-epoch processing (`bake_epoch`: row-ramp removal, north-up bilinear reprojection,
defect masking). Pixels come from DR1 via orrery.data.Frame (range reads); stamps are stored as
float32 instead of 16-bit asinh PNG (docs/protocols/E21.md, change 1).

    .venv/bin/python scripts/run_e21_bake.py [nep|sep]
"""
import json
import math
import pathlib
import statistics
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from orrery.data import Frame  # noqa: E402
from tracklet.frames.stretch import finite_values, flatten_rows, reproject, robust_sigma  # noqa: E402
from tracklet.frames.wcs import Wcs  # noqa: E402
from tracklet.physics.ephemeris import apparent_place  # noqa: E402
from tracklet.search.link import sep_arcsec  # noqa: E402

FIELDS = {"nep": (270.0, 66.56, "North Ecliptic Pole"), "sep": (90.0, -66.56, "South Ecliptic Pole")}
NAME = sys.argv[1] if len(sys.argv) > 1 else "nep"
RA, DEC, TITLE = FIELDS[NAME]
HALF, EPOCHS, MIN_GAP, SEE_TO_AU, DEFECT_SIGMA = 96, 30, 2.0, 2000.0, 50.0
OUT = ROOT / "data" / "phase2" / "e21" / NAME


def log(m):
    print(f"[bake {time.strftime('%H:%M:%S')}] {m}", flush=True)


def candidates():
    """DR1 D1 frames whose footprint polygon contains the field (tangent-plane test)."""
    d = pd.read_parquet(ROOT / "data" / "index" / "frames_dr1.parquet")
    d = d[d.detector == 1]
    r = np.radians
    cs = np.sin(r(d.dec)) * np.sin(r(DEC)) + np.cos(r(d.dec)) * np.cos(r(DEC)) * np.cos(r(d.ra - RA))
    d = d[np.degrees(np.arccos(np.clip(cs, -1, 1))) < 2.6]
    keep = []
    for row in d.itertuples():
        ra = np.radians(np.asarray(row.c_ra, float)); de = np.radians(np.asarray(row.c_dec, float))
        a0, d0 = math.radians(RA), math.radians(DEC)
        c = np.sin(d0) * np.sin(de) + np.cos(d0) * np.cos(de) * np.cos(ra - a0)
        xi = np.cos(de) * np.sin(ra - a0) / c
        eta = (np.cos(d0) * np.sin(de) - np.sin(d0) * np.cos(de) * np.cos(ra - a0)) / c
        # origin inside the convex quadrilateral: all cross products share a sign
        cr = [xi[k] * eta[(k + 1) % 4] - xi[(k + 1) % 4] * eta[k] for k in range(4)]
        if all(v > 0 for v in cr) or all(v < 0 for v in cr):
            keep.append(row)
    keep.sort(key=lambda f: f.mjd)
    return keep


def keeps_sight(mjds, scale=6.15):
    if len(mjds) < 3:
        return True
    places = [apparent_place(RA, DEC, SEE_TO_AU, m) for m in mjds]
    for a in places:
        if sum(1 for p in places if sep_arcsec(a.ra_deg, a.dec_deg, p.ra_deg, p.dec_deg) <= scale) >= 3:
            return False
    return True


def bake_epoch(row, index):
    fr = Frame(row.key)
    cards = {k: str(v) for k, v in fr.header.items() if k}
    wcs = Wcs.from_cards(cards)
    x, y = wcs.world_to_pixel(RA, DEC)
    if not wcs.contains(x, y, margin=HALF + 2):
        log(f"  {row.obs_id}: target at ({x:.0f},{y:.0f}) too near the edge, skipped")
        fr.close()
        return None
    xi, yi = round(x), round(y)
    wide = int(math.ceil(HALF * math.sqrt(2.0)))
    y0, y1 = yi - HALF, yi + HALF + 1                 # 1-based rows
    wx0, wx1 = max(1, xi - wide), min(2040, xi + wide)  # 1-based cols, inclusive
    t0 = time.time()
    band = np.asarray(fr.hdul["IMAGE"].section[y0 - 1:y1 - 1, wx0 - 1:wx1], np.float64)
    vband = np.asarray(fr.hdul["VARIANCE"].section[yi - 1 - 8:yi - 1 + 8, xi - 1 - HALF:xi + HALF], np.float64)
    dt = time.time() - t0
    vv = np.sort(vband[np.isfinite(vband) & (vband > 0)])
    sigma_var = float(math.sqrt(vv[len(vv) // 2])) if len(vv) else None
    rows = [[float(v) for v in r] for r in band]
    rows, ramp = flatten_rows(rows)
    size = 2 * HALF + 1
    stamp_wcs = Wcs.north_up(RA, DEC, wcs.scale_arcsec, size)
    stamp, masked_frac, var_shrink = reproject(rows, wcs.world_to_pixel, stamp_wcs.pixel_to_world,
                                               size=size, col_origin=wx0, row_origin=y0)
    flat = finite_values(stamp)
    if not flat:
        fr.close()
        return None
    sigma_mad = robust_sigma(flat)
    floor = statistics.median(flat) - DEFECT_SIGMA * sigma_mad
    arr = np.array(stamp, np.float32)
    arr[np.isfinite(arr) & (arr < floor)] = np.nan
    fn = f"e{index:02d}.f32"
    arr.astype("<f4").tofile(OUT / fn)
    mjd = float(fr.header["MJD-AVG"])
    log(f"  e{index:02d} {row.obs_id:24s} mjd={mjd:.4f} read {dt:.1f}s sigma_mad={sigma_mad:.4f} "
        f"shrink={var_shrink:.3f} masked={masked_frac:.3f}")
    fr.close()
    return {"i": index, "obs_id": row.obs_id, "key": row.key, "detector": 1, "mjd": mjd,
            "wcs": wcs.to_dict(),
            "stamp": {"file": fn, "w": size, "h": size, "wcs": stamp_wcs.to_dict(), "center_px": [x, y]},
            "stats": {"sigma_variance": sigma_var, "sigma_mad": sigma_mad, "row_ramp": ramp,
                      "north_angle_deg": wcs.local_north_angle_deg(x, y),
                      "masked_fraction": masked_frac, "resample_var_shrink": var_shrink},
            "scale_arcsec": wcs.scale_arcsec}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frames = candidates()
    log(f"{len(frames)} DR1 D1 frames contain the field, MJD {frames[0].mjd:.2f}-{frames[-1].mjd:.2f}")
    entries, tried = [], set()
    while len(entries) < EPOCHS:
        kept = [e["mjd"] for e in entries]
        best, best_d = None, -1.0
        for k, f in enumerate(frames):
            if k in tried:
                continue
            d = min((abs(f.mjd - m) for m in kept), default=math.inf)
            if d < MIN_GAP:
                tried.add(k)
                continue
            if not kept:
                d = math.inf if k in (0, len(frames) - 1) else 0.0
            if d > best_d:
                best, best_d = k, d
        if best is None:
            break
        tried.add(best)
        f = frames[best]
        if not keeps_sight([*kept, f.mjd]):
            continue
        try:
            e = bake_epoch(f, len(entries))
        except Exception as exc:  # one bad frame must not end the bake
            log(f"  {f.obs_id}: FAILED {type(exc).__name__}: {exc}")
            continue
        if e:
            entries.append(e)
    # time order, renaming stamps to match
    order = sorted(range(len(entries)), key=lambda k: entries[k]["mjd"])
    for k in range(len(entries)):
        (OUT / f"e{k:02d}.f32").rename(OUT / f"tmp{k:02d}.f32")
    eps = []
    for new, old in enumerate(order):
        e = entries[old]
        (OUT / f"tmp{old:02d}.f32").rename(OUT / f"e{new:02d}.f32")
        e["i"], e["stamp"]["file"] = new, f"e{new:02d}.f32"
        eps.append(e)
    mjds = [e["mjd"] for e in eps]
    man = {"name": f"{NAME}_dr1", "title": f"{TITLE} (SPHEREx DR1, D1)", "ra": RA, "dec": DEC,
           "band": "D1", "half": HALF, "release": "dr1", "baked_local": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
           "selection": {"rule": "Tracklet gap_series", "min_gap_days": MIN_GAP, "epochs": EPOCHS,
                         "see_to_au": SEE_TO_AU, "n_candidate_frames": len(frames)},
           "n_epochs": len(eps), "time_span_days": max(mjds) - min(mjds), "epochs": eps}
    (OUT.parent / f"{NAME}_dr1.json").write_text(json.dumps(man, indent=1))
    log(f"wrote {len(eps)} epochs over {max(mjds) - min(mjds):.1f} d")


if __name__ == "__main__":
    main()
