"""Prefetch SkyBoT cone searches for E3 fields into data/cache/skybot/ (same files build_field.known_asteroids reads).
Frame times per field: the protocol's capped frame list (index candidates sorted by MJD, 250 evenly spread).
E2 fields from data/efields/e2_fields.json (drawn by scripts/e2_cache.py). Usage: .venv/bin/python scripts/e3_skybot_prefetch.py e2 [threads<=3]
"""
import json, sys, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, "scripts")
import build_field as bf  # noqa: E402


def draw_e2(df):
    """E2 fields as drawn by scripts/e2_cache.py (data/efields/e2_fields.json), with their index candidates."""
    fs = json.load(open("data/efields/e2_fields.json"))["fields"]
    return [dict(f, cand=bf.candidates(df, f["ra"], f["dec"])) for f in fs], None


def groups(mjd, tol=0.05):
    order, G, start = np.argsort(mjd), [], None
    for e in order:
        if start is None or mjd[e] - start > tol:
            G.append([]); start = mjd[e]
        G[-1].append(e)
    return [float(np.mean(mjd[g])) for g in G]


if __name__ == "__main__":
    threads = min(3, int(sys.argv[2]) if len(sys.argv) > 2 else 3)
    df, _ = bf.load_index()
    fields, _ = draw_e2(df)
    jobs = []
    for i, f in enumerate(fields):
        c = f["cand"].sort_values("mjd", kind="stable")
        if len(c) > 250:
            c = c.iloc[np.linspace(0, len(c) - 1, 250).round().astype(int)]
        f["n"] = len(c)
        jobs += [(t + 2400000.5, f["ra"], f["dec"]) for t in groups(c.mjd.values)]
        print(i, round(f["ra"], 5), round(f["dec"], 5), f["band"], len(f["cand"]), flush=True)
    if "reverse" in sys.argv:                 # a second process can work from the end (cached files are skipped)
        jobs = jobs[::-1]
    print(len(jobs), "queries", flush=True)
    t = time.time()
    with ThreadPoolExecutor(threads) as ex:
        def go(j):
            try:
                return len(bf._skybot_query(*j))
            except Exception as e:
                print("FAIL", j, repr(e), flush=True)
        for i, r in enumerate(ex.map(go, jobs)):
            if (i + 1) % 25 == 0:
                print(i + 1, f"{time.time() - t:.0f}s", flush=True)
    print("done", time.time() - t)
