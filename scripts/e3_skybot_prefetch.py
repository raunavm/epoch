"""Prefetch SkyBoT cone searches for E3 fields into data/cache/skybot/ (same files build_field.known_asteroids reads).
Frame times per field: the protocol's capped frame list (index candidates sorted by MJD, 250 evenly spread).
E2 fields are re-drawn exactly as in docs/PROTOCOLS.md E2 (only to know where to ask; the E2 cutouts come from
data/efields/). Usage: .venv/bin/python scripts/e3_skybot_prefetch.py e2 [threads<=3]
"""
import json, sys, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

sys.path.insert(0, "scripts")
import build_field as bf  # noqa: E402


def draw_e2(df):
    rng = np.random.default_rng(20261115)
    quota = {">30": 10, "15-30": 5, "<15": 5}
    out, n_rej = [], 0
    while sum(quota.values()):
        ra, dec = rng.uniform(0, 360), np.degrees(np.arcsin(rng.uniform(-1, 1)))
        b = abs(SkyCoord(ra * u.deg, dec * u.deg).galactic.b.deg)
        band = ">30" if b > 30 else ("15-30" if b >= 15 else "<15")
        if not quota[band]:
            continue
        c = bf.candidates(df, ra, dec)
        if len(c) < 40:
            n_rej += 1
            continue
        quota[band] -= 1
        out.append(dict(ra=ra, dec=dec, b=b, band=band, cand=c))
    return out, n_rej


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
    fields, n_rej = draw_e2(df)
    jobs = []
    for i, f in enumerate(fields):
        c = f["cand"].sort_values("mjd", kind="stable")
        if len(c) > 250:
            c = c.iloc[np.linspace(0, len(c) - 1, 250).round().astype(int)]
        f["n"] = len(c)
        jobs += [(t + 2400000.5, f["ra"], f["dec"]) for t in groups(c.mjd.values)]
        print(i, round(f["ra"], 5), round(f["dec"], 5), f["band"], len(f["cand"]), flush=True)
    print(len(jobs), "queries;", n_rej, "draws rejected (< 40 frames)", flush=True)
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
