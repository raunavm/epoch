"""Forced-mode photometry of known objects in DR1.

Usage: .venv/bin/python scripts/run_forced.py --ids 24,433 [--vmax 17] [--workers 8] [--stamps]
Reads data/index (full parquet if present, else parts), data/ephem/<id>.parquet (scripts/ephem.py).
Writes data/forced/<id>.parquet (one row per frame) and, with --stamps, data/stamps/<id>/.
Idempotent per object.
"""
import argparse, os, subprocess, sys, warnings
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings("ignore")
from astropy import log
log.setLevel("ERROR")
from orrery.forced import candidate_frames, measure_frame
from ephem import fetch, fetch_epochs


def load_index(release="dr1"):
    p = f"data/index/frames_{release}.parquet"
    if os.path.exists(p):
        df = pd.read_parquet(p)
    else:
        d = f"data/index/parts_{release}"
        df = pd.concat([pd.read_parquet(f"{d}/{f}") for f in sorted(os.listdir(d))], ignore_index=True)
    if "error" in df:
        df = df[df["error"].isna()]
    return df[df["dqa"] == "Pass"].reset_index(drop=True)


def code_version():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "uncommitted"


def _job(a):
    key, eph, sd = a
    try:
        r = measure_frame(key, eph, stamp_dir=sd)
    except Exception as e:
        return dict(key=key, error=repr(e))
    if r:
        for c in ["V", "r", "delta", "alpha", "RA", "DEC", "RA_rate", "DEC_rate"]:
            if c in eph:
                r[c] = float(eph[c])
    return r


def run(pdes, index, workers, stamps, vmax, force=False, cap=0):
    out = f"data/forced/{pdes.replace(' ', '_')}.parquet"
    if os.path.exists(out) and not force:
        return pd.read_parquet(out)
    p = fetch(pdes)
    if p is None:
        return None
    eph = pd.read_parquet(p)
    cand = candidate_frames(index, eph)
    if len(cand) == 0:
        pd.DataFrame().to_parquet(out)
        return None
    # skip epochs when the object is fainter than vmax (V from Horizons)
    v = np.interp(cand["mjd"].values, eph["mjd"].values, eph["V"].values)
    cand = cand[v < vmax]
    if len(cand) == 0:
        pd.DataFrame().to_parquet(out)
        return None
    if cap and len(cand) > cap:
        # validation sample: keep detectors 1 (Gaia overlap), 4 (2.4–3.8 µm), 5, 6 (thermal); subsample 2 and 3
        keep = cand[cand.detector.isin([1, 4, 5, 6])]
        rest = [g.sample(min(len(g), 15), random_state=0) for _, g in cand[cand.detector.isin([2, 3])].groupby("detector")]
        cand = pd.concat([keep] + rest)
    cand = cand.sort_values("mjd")
    pe = fetch_epochs(pdes, list(cand["mjd"].values))
    # Horizons returns rows in time order; match each frame to its epoch explicitly (no positional zip)
    j = np.searchsorted(pe["mjd"].values, cand["mjd"].values).clip(0, len(pe) - 1)
    jm = (j - 1).clip(0)
    j = np.where(np.abs(pe["mjd"].values[jm] - cand["mjd"].values) < np.abs(pe["mjd"].values[j] - cand["mjd"].values), jm, j)
    assert np.all(np.abs(pe["mjd"].values[j] - cand["mjd"].values) < 1e-4), "ephemeris/frame epoch mismatch"
    pe = pe.iloc[j].set_index(cand.index)
    rows = []
    sd = f"data/stamps/{pdes.replace(' ', '_')}" if stamps else None
    args = [(cand.loc[i, "key"], pe.loc[i].to_dict(), sd) for i in cand.index]
    with ProcessPoolExecutor(workers) as ex:
        for r in ex.map(_job, args, chunksize=2):
            if r:
                rows.append(r)
    df = pd.DataFrame(rows)
    df["object"] = pdes
    df["release"] = "dr1"
    df["code_version"] = code_version()
    df["rerun"] = f".venv/bin/python scripts/run_forced.py --ids {pdes} --stamps --redo"
    os.makedirs("data/forced", exist_ok=True)
    df.to_parquet(out)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="")
    ap.add_argument("--file", default="")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--vmax", type=float, default=22)
    ap.add_argument("--stamps", action="store_true")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--cap", type=int, default=0)
    a = ap.parse_args()
    ids = [x for x in a.ids.split(",") if x] + ([l.strip() for l in open(a.file) if l.strip()] if a.file else [])
    index = load_index()
    for pdes in ids:
        df = run(pdes, index, a.workers, a.stamps, a.vmax, a.redo, a.cap)
        n = 0 if df is None else len(df)
        good = 0 if df is None or "snr" not in df else int((df["snr"] > 5).sum())
        print(f"{pdes}: frames={n} snr>5={good}", flush=True)
