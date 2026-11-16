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
    if os.path.exists(FIELDS):
        return json.load(open(FIELDS))
    import run_e30
    df, src = e2_cache.load_index()
    df = df[df.dqa == "Pass"]
    lat = SkyCoord(df.ra.values * u.deg, df.dec.values * u.deg).transform_to(BarycentricMeanEcliptic()).lat.deg
    df = df[np.abs(lat) < 6].reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    avoid = run_e30.used_centres() + [(x["ra"], x["dec"]) for g in ("calib", "test", "ecliptic", "train_extra")
                                      for x in run_e30.draw()[g]["fields"]]
    quota, out, nq = {"bright": 8, "faint": 8}, [], 0
    for i in rng.permutation(len(df)):
        if all(v == 0 for v in quota.values()):
            break
        r = df.iloc[int(i)]
        objs = skybot(r.mjd + 2400000.5, float(r.ra), float(r.dec)); nq += 1
        if not objs:
            continue
        el = {"bright": [], "faint": []}
        for o in objs:
            v = o["VMag (mag)"]
            ra, dec = Angle(o["RA (hms)"], "hourangle").deg, Angle(o["DEC (dms)"], "deg").deg
            if 13.0 <= v <= 17.5 and sep(ra, dec, r.ra, r.dec) < 1.2:
                el["bright" if v < 16 else "faint"].append((o["Name"], float(v), float(ra), float(dec)))
        strata = [s for s in sorted(quota, key=lambda s: -quota[s]) if quota[s] > 0 and el[s]]
        if not strata:
            continue
        s = strata[0]
        name, v, ra, dec = el[s][int(rng.integers(len(el[s])))]
        dra, ddec = rng.uniform(-60, 60, 2) / 3600
        ra0, dec0 = ra + dra / np.cos(np.radians(dec)), dec + ddec
        if any(sep(ra0, dec0, a, b) < 0.5 for a, b in avoid):
            continue
        out.append(dict(field=f"e34a{len(out):02d}", ra=float(ra0), dec=float(dec0), stratum=s, selected_by=name, vmag=v,
                        frame_key=r.key, frame_mjd=float(r.mjd)))
        avoid.append((ra0, dec0)); quota[s] -= 1
        print(out[-1], flush=True)
    res = dict(protocol="docs/protocols/E34_fields.md", seed=SEED, index=src, skybot_queries=nq, fields=out)
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
