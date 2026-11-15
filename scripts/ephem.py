"""Daily ephemerides as seen from SPHEREx (Horizons observer 500@-163182) over the DR1 span.

Usage: .venv/bin/python scripts/ephem.py --ids 24,433,... | --sample N
Writes data/ephem/<pdes>.parquet (idempotent). Quantities: RA/Dec (astrometric ICRF),
V, r, delta, phase angle, rates. Horizons times are UT.
"""
import argparse, os, sys, time
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from astroquery.jplhorizons import Horizons

OBS = "500@-163182"  # SPHEREx spacecraft (Horizons; concatenated TLEs), docs/notes/competition.md #8
SPAN = {"start": "2025-04-25", "stop": "2026-05-20", "step": "1d"}
OUT = "data/ephem"


def fetch(pdes, epochs=SPAN, out=OUT):
    path = f"{out}/{pdes.replace(' ', '_')}.parquet"
    if os.path.exists(path):
        return path
    for attempt in range(4):
        try:
            e = Horizons(id=pdes, id_type="smallbody", location=OBS, epochs=epochs).ephemerides(
                quantities="1,3,9,19,20,24", extra_precision=True)
            cols = [c for c in ["datetime_jd", "RA", "DEC", "RA_rate", "DEC_rate", "V", "r", "delta", "alpha"] if c in e.colnames]
            df = e[cols].to_pandas()
            df["mjd"] = df["datetime_jd"] - 2400000.5
            df.to_parquet(path)
            return path
        except Exception as ex:
            err = ex
            time.sleep(3 * (attempt + 1))
    print("FAIL", pdes, err, file=sys.stderr)
    return None


def fetch_epochs(pdes, mjds):
    """Precise positions at a list of MJD (UTC) epochs; batched to keep URLs short."""
    rows = []
    for i in range(0, len(mjds), 40):
        jd = [m + 2400000.5 for m in mjds[i:i + 40]]
        for attempt in range(8):
            try:
                e = Horizons(id=pdes, id_type="smallbody", location=OBS, epochs=jd).ephemerides(
                    quantities="1,3,9,19,20,24", extra_precision=True)
                break
            except Exception:
                time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError(f"Horizons failed for {pdes}")
        cols = [c for c in ["datetime_jd", "RA", "DEC", "RA_rate", "DEC_rate", "V", "r", "delta", "alpha"] if c in e.colnames]
        rows.append(e[cols].to_pandas())
    df = pd.concat(rows, ignore_index=True)
    df["mjd"] = df["datetime_jd"] - 2400000.5
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="")
    ap.add_argument("--file", default="")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    ids = [x for x in a.ids.split(",") if x]
    if a.file:
        ids += [l.strip() for l in open(a.file) if l.strip()]
    with ThreadPoolExecutor(a.workers) as ex:
        for p in ex.map(fetch, ids):
            pass
    print("done", len(ids))
