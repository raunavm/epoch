"""E34: seeded draw of fresh fields that contain a real asteroid of V 13-17.5 (docs/protocols/E34_fields.md), and their cache."""
import json, os, sys, time
import numpy as np, requests
sys.path.insert(0, os.path.dirname(__file__))
import e2_cache, build_field as bf
from astropy.coordinates import SkyCoord, BarycentricMeanEcliptic, Angle
import astropy.units as u

BASE, SEED = "data/phase2/e34", 2026111560
ROOT, FIELDS = f"{BASE}/efields", f"{BASE}/fields.json"


def sep(ra1, dec1, ra2, dec2):
    return SkyCoord(ra1 * u.deg, dec1 * u.deg).separation(SkyCoord(ra2 * u.deg, dec2 * u.deg)).deg


def skybot(jd, ra, dec, sr=1.4):
    cp = f"{BASE}/skybot/{jd:.6f}_{ra:.5f}_{dec:+.5f}.json"
    if os.path.exists(cp):
        return json.load(open(cp))
    r = _skybot(jd, ra, dec, sr)
    if r is not None:
        os.makedirs(os.path.dirname(cp), exist_ok=True)
        json.dump(r, open(cp, "w"))
    return r


def _skybot(jd, ra, dec, sr=1.4):
    p = {"EPOCH": f"{jd:.6f}", "RA": ra, "DEC": dec, "SR": sr, "LOC": "500", "-mime": "json", "-output": "basic"}
    for i in range(4):
        try:
            q = requests.get(bf.SKYBOT, params=p, timeout=120)
        except requests.RequestException:
            time.sleep(3 * (i + 1)); continue
        if q.status_code == 200:
            return q.json()
        if q.status_code == 204:
            return []
        time.sleep(3 * (i + 1))
    return None


def draw():
    """Rule v2 (docs/protocols/E34_fields.md, deviation 1): seeded numbered asteroids, Horizons ephemerides as seen from SPHEREx."""
    if os.path.exists(FIELDS):
        return json.load(open(FIELDS))
    import run_e30, ephem
    from concurrent.futures import ThreadPoolExecutor
    df, src = e2_cache.load_index()
    df = df[df.dqa == "Pass"].reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    nums = [str(int(n)) for n in rng.permutation(np.arange(1, 4001))[:90]]
    os.makedirs(f"{BASE}/ephem", exist_ok=True)
    with ThreadPoolExecutor(8) as ex:
        paths = list(ex.map(lambda n: ephem.fetch(n, out=f"{BASE}/ephem"), nums))
    avoid = run_e30.used_centres() + [(x["ra"], x["dec"]) for g in ("calib", "test", "ecliptic", "train_extra")
                                      for x in run_e30.draw()[g]["fields"]]
    quota, out = {"bright": 8, "faint": 8}, []
    fm, fra, fdec = df.mjd.values, np.radians(df.ra.values), np.radians(df.dec.values)
    import pandas as pd
    for n, p in zip(nums, paths):
        if all(v == 0 for v in quota.values()):
            break
        if p is None:
            continue
        e = pd.read_parquet(p).sort_values("mjd")
        ok = (fm > e.mjd.min()) & (fm < e.mjd.max())
        ra = np.radians(np.interp(fm, e.mjd, np.degrees(np.unwrap(np.radians(e.RA.values)))))
        dec = np.radians(np.interp(fm, e.mjd, e.DEC.values))
        v = np.interp(fm, e.mjd, e.V.values)
        cs = np.sin(dec) * np.sin(fdec) + np.cos(dec) * np.cos(fdec) * np.cos(ra - fra)
        near = ok & (np.degrees(np.arccos(np.clip(cs, -1, 1))) < 1.2)
        strata = [s for s in sorted(quota, key=lambda s: -quota[s]) if quota[s] > 0]
        pick = None
        for s in strata:
            m = near & ((v >= 13) & (v < 16) if s == "bright" else (v >= 16) & (v <= 17.5))
            if m.any():
                pick = (s, int(rng.choice(np.flatnonzero(m))))
                break
        if pick is None:
            continue
        s, i = pick
        r = df.iloc[i]
        q = ephem.fetch_epochs(n, [float(r.mjd)]).iloc[0]
        dra, ddec = rng.uniform(-60, 60, 2) / 3600
        ra0, dec0 = float(q.RA) + dra / np.cos(np.radians(float(q.DEC))), float(q.DEC) + ddec
        if any(sep(ra0, dec0, a, b) < 0.5 for a, b in avoid):
            continue
        out.append(dict(field=f"e34a{len(out):02d}", ra=float(ra0), dec=float(dec0), stratum=s, selected_by=n,
                        vmag=float(q.V), frame_key=r.key, frame_mjd=float(r.mjd)))
        avoid.append((ra0, dec0)); quota[s] -= 1
        print(out[-1], flush=True)
    res = dict(protocol="docs/protocols/E34_fields.md", seed=SEED, index=src, numbers_tried=nums, fields=out)
    os.makedirs(BASE, exist_ok=True)
    json.dump(res, open(FIELDS, "w"), indent=1)
    return res


def fetch(part, nparts):
    import shutil
    from concurrent.futures import ThreadPoolExecutor
    fl = draw()["fields"][part::nparts]
    e2_cache.ROOT = ROOT
    df, _ = e2_cache.load_index()
    with ThreadPoolExecutor(8) as ex:
        for x in fl:
            if shutil.disk_usage(".").free / 1e9 < 20:
                sys.exit("less than 20 GB free; refusing")
            d = f"{ROOT}/{x['field']}"; os.makedirs(d, exist_ok=True)
            for i in range(5):
                if e2_cache.gaia(f"{d}/gaia.parquet", x["ra"], x["dec"]):
                    break
                time.sleep(20 * (i + 1))
            e2_cache.build(x["field"], x["ra"], x["dec"], df, ex, dict(role="E34 real-asteroid field", stratum=x["stratum"]))


if __name__ == "__main__":
    if sys.argv[1] == "draw":
        print(len(draw()["fields"]), "fields")
    else:
        fetch(int(sys.argv[2]), int(sys.argv[3]))
