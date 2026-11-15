"""E9 comparison-star draw (docs/PROTOCOLS.md "E9"; choices recorded in docs/protocols/E9.md).

No SPHEREx pixel of either target is read here. Catalogues only:
  Gaia DR3 (gaia.aip.de TAP)        -> candidate positions, proper motions, phot_variable_flag
  2MASS (VizieR II/246, CfA mirror) -> K magnitude
  VSX (VizieR B/vsx/vsx, CfA mirror; AAVSO VSX API as a second check) -> known variables within 5"
Output: data/phase2/e9/comparison_draw.json (ordered list per class; the download takes them in this order).

  OMP_NUM_THREADS=1 .venv/bin/python scripts/e9_select.py
"""
import json, os, sys, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from orrery.change import project  # noqa: E402
from orrery.e7 import windows  # noqa: E402

OUT = "data/phase2/e9"
SEED = 2026111509
VIZ = "https://vizier.cfa.harvard.edu/viz-bin/asu-tsv"
TARGETS = dict(rwaur=dict(ra=76.956589, dec=30.401293, K=7.020, n_indexed=226),
               rdra=dict(ra=248.167235, dec=66.755023, K=2.438, n_indexed=1637))
CLASSES = dict(
    rwaur=dict(centre=(76.956589, 30.401293), radius=2.5, K=(6.02, 8.02), g_max=12.5, bp_rp_min=None,
               n_frames=(113, 452), n_windows=2, n_draw=60),
    rdra=dict(centre=(270.0, 66.5607), radius=15.0, K=(1.44, 3.44), g_max=9.0, bp_rp_min=1.2,
              n_frames=(800, 10 ** 9), n_windows=None, n_draw=45))


def viz(source, ra, dec, rs, out, extra=""):
    """VizieR cone (arcsec) -> list of rows (list of str); None if the service failed after retries."""
    url = f"{VIZ}?-source={source}&-c={ra:.6f}{dec:+.6f}&-c.rs={rs}&-out={out}&-out.max=50{extra}"
    for i in range(5):
        try:
            t = requests.get(url, timeout=30).text
            if "QUERY_STATUS=ERROR" in t or "-- no connection" in t:
                raise RuntimeError("vizier error")
            rows = [ln.split("\t") for ln in t.splitlines() if ln and not ln.startswith("#")]
            return rows[3:] if len(rows) >= 3 else []
        except Exception:
            time.sleep(1.5 * (i + 1))
    return None


def vsx_api(ra, dec, rs=5.0):
    """AAVSO VSX API cone; number of objects (None on failure)."""
    for i in range(3):
        try:
            j = requests.get("https://vsx.aavso.org/index.php", timeout=30,
                             params=dict(view="api.list", ra=f"{ra:.6f}", dec=f"{dec:.6f}", radius=f"{rs / 3600:.6f}",
                                         format="json")).json()
            return len(j.get("VSXObjects", {}).get("VSXObject", [])) if j.get("VSXObjects") else 0
        except Exception:
            time.sleep(1.5 * (i + 1))
    return None


def n_frames(df, ra, dec):
    """Vectorised scripts/build_field.py:candidates -> sorted MJDs and detectors of frames whose footprint has (ra, dec)."""
    a1, d1, a2, d2 = map(np.radians, (df.ra.values, df.dec.values, ra, dec))
    sep = np.degrees(2 * np.arcsin(np.sqrt(np.sin((d1 - d2) / 2) ** 2 + np.cos(d1) * np.cos(d2) * np.sin((a1 - a2) / 2) ** 2)))
    sub = df[sep < 2.8]
    xi, eta = project(np.stack(sub.c_ra.values), np.stack(sub.c_dec.values), ra, dec)
    ang = np.argsort(np.arctan2(eta - eta.mean(1, keepdims=True), xi - xi.mean(1, keepdims=True)), 1)
    px, py = np.take_along_axis(xi, ang, 1), np.take_along_axis(eta, ang, 1)
    cr = (np.roll(px, -1, 1) - px) * (0 - py) - (np.roll(py, -1, 1) - py) * (0 - px)
    return sub[(cr > 0).all(1) | (cr < 0).all(1)]


def check(row, krange=None):
    """Catalogue checks for one Gaia candidate -> dict (K, vsx counts); K None if no 2MASS match.
    With krange, the VSX look-ups are skipped for stars whose K is outside it (they are rejected on K anyway)."""
    ra, dec = row["ra"], row["dec"]
    # 2MASS epoch ~1999: undo ~17 yr of proper motion for the match position
    r2 = ra - 16.5 * (row["pmra"] or 0) / 3.6e6 / np.cos(np.radians(dec))
    d2 = dec - 16.5 * (row["pmdec"] or 0) / 3.6e6
    tm = viz("II/246/out", r2, d2, 4, "RAJ2000,DEJ2000,Jmag,Hmag,Kmag,Qflg")
    out = dict(K=None, J=None, vsx_vizier=None, vsx_api=None, tmass_ok=tm is not None)
    if tm:
        ks = [(float(r[4]), float(r[2]) if r[2].strip() else np.nan) for r in tm if r[4].strip()]
        if ks:
            out["K"], out["J"] = min(ks)
    if krange is not None and (out["K"] is None or not (krange[0] <= out["K"] <= krange[1])):
        return out
    v1 = viz("B/vsx/vsx", r2, d2, 5, "OID,Name,Type")
    v2 = viz("B/vsx/vsx", ra, dec, 5, "OID,Name,Type")
    out["vsx_vizier"] = None if (v1 is None or v2 is None) else len(v1) + len(v2)
    out["vsx_api"] = vsx_api(r2, d2)
    return out


def main():
    import pyvo
    from build_field import load_index
    os.makedirs(OUT, exist_ok=True)
    df, src = load_index()
    gaia = pyvo.dal.TAPService("https://gaia.aip.de/tap")
    res = dict(seed=SEED, targets=TARGETS, classes={}, index=src, made=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    for ci, (cname, c) in enumerate(CLASSES.items()):
        ra0, dec0 = c["centre"]
        q = ("SELECT source_id, ra, dec, pmra, pmdec, phot_g_mean_mag, bp_rp, phot_variable_flag FROM gaiadr3.gaia_source "
             f"WHERE 1=CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',{ra0},{dec0},{c['radius']})) "
             f"AND phot_g_mean_mag<{c['g_max']}" + (f" AND bp_rp>{c['bp_rp_min']}" if c["bp_rp_min"] else ""))
        g = gaia.run_async(q, maxrec=500000).to_table().to_pandas().sort_values("source_id").reset_index(drop=True)
        n_gaia = len(g)
        # colour-predicted K (Gaia DR2 documentation relation G-Ks vs BP-RP) only to prune the 2MASS look-ups
        col = g.bp_rp.fillna(1.0)
        kpred = g.phot_g_mean_mag - (-0.1885 + 2.092 * col - 0.1345 * col ** 2)
        g = g[(kpred > c["K"][0] - 1.0) & (kpred < c["K"][1] + 1.0)]
        t = TARGETS[cname]
        sep = np.hypot((g.ra - t["ra"]) * np.cos(np.radians(t["dec"])), g.dec - t["dec"]) * 3600
        g = g[sep > 60]
        n_pruned = len(g)
        g = g[g.phot_variable_flag != "VARIABLE"].reset_index(drop=True)
        rng = np.random.default_rng(SEED + ci)
        order = rng.permutation(len(g))
        acc, rej = [], []
        pos = 0
        with ThreadPoolExecutor(6) as ex:
            while len(acc) < c["n_draw"] and pos < len(order):
                batch = [g.iloc[i].to_dict() for i in order[pos:pos + 24]]
                pos += 24
                for row, chk in zip(batch, ex.map(check, batch)):
                    rec = dict(source_id=str(int(row["source_id"])), ra=row["ra"], dec=row["dec"], pmra=row["pmra"],
                               pmdec=row["pmdec"], G=row["phot_g_mean_mag"], bp_rp=row["bp_rp"], **chk)
                    if chk["K"] is None or not (c["K"][0] <= chk["K"] <= c["K"][1]):
                        rej.append(dict(rec, reason="K outside range or no 2MASS match"))
                        continue
                    if chk["vsx_vizier"] is None and chk["vsx_api"] is None:
                        rej.append(dict(rec, reason="VSX unreachable"))
                        continue
                    if (chk["vsx_vizier"] or 0) + (chk["vsx_api"] or 0) > 0:
                        rej.append(dict(rec, reason="in VSX within 5 arcsec"))
                        continue
                    fr = n_frames(df, row["ra"], row["dec"])
                    nw = int(windows(fr.mjd.values).max() + 1) if len(fr) else 0
                    rec.update(n_indexed=int(len(fr)), n_windows=nw)
                    if not (c["n_frames"][0] <= len(fr) <= c["n_frames"][1]):
                        rej.append(dict(rec, reason="number of indexed pictures outside range"))
                        continue
                    if c["n_windows"] and nw != c["n_windows"]:
                        rej.append(dict(rec, reason="number of windows differs"))
                        continue
                    if len(acc) < c["n_draw"]:
                        acc.append(dict(rec, name=f"{cname}_c{len(acc):02d}"))
                print(cname, "examined", pos, "accepted", len(acc), "rejected", len(rej), flush=True)
        res["classes"][cname] = dict(rule=c, n_gaia=n_gaia, n_after_colour_prune=n_pruned, n_not_gaia_variable=len(g),
                                     n_examined=min(pos, len(order)), accepted=acc, rejected=rej)
        json.dump(res, open(f"{OUT}/comparison_draw.json", "w"), indent=1)
    print({k: len(v["accepted"]) for k, v in res["classes"].items()})


def gaia_pool(gaia, c, cname, centre=None):
    """Same candidate pool as in main(): Gaia query, colour prune, target exclusion, not Gaia-variable, sorted by source_id."""
    ra0, dec0 = centre or c["centre"]
    q = ("SELECT source_id, ra, dec, pmra, pmdec, phot_g_mean_mag, bp_rp, phot_variable_flag FROM gaiadr3.gaia_source "
         f"WHERE 1=CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',{ra0},{dec0},{c['radius']})) "
         f"AND phot_g_mean_mag<{c['g_max']}" + (f" AND bp_rp>{c['bp_rp_min']}" if c["bp_rp_min"] else ""))
    g = gaia.run_async(q, maxrec=500000).to_table().to_pandas().sort_values("source_id").reset_index(drop=True)
    col = g.bp_rp.fillna(1.0)
    kpred = g.phot_g_mean_mag - (-0.1885 + 2.092 * col - 0.1345 * col ** 2)
    g = g[(kpred > c["K"][0] - 1.0) & (kpred < c["K"][1] + 1.0)]
    t = TARGETS[cname]
    sep = np.hypot((g.ra - t["ra"]) * np.cos(np.radians(t["dec"])), g.dec - t["dec"]) * 3600
    g = g[sep > 60]
    return g[g.phot_variable_flag != "VARIABLE"].reset_index(drop=True)


def examine(df, g, order, pos, n_want, krange, nfr, nwin, cname, start_idx, kpred_max=None):
    acc, rej = [], []
    with ThreadPoolExecutor(12) as ex:
        while len(acc) < n_want and pos < len(order):
            batch = [g.iloc[i].to_dict() for i in order[pos:pos + 24]]
            pos += 24
            for row, chk in zip(batch, ex.map(lambda r: check(r, krange), batch)):
                rec = dict(source_id=str(int(row["source_id"])), ra=row["ra"], dec=row["dec"], pmra=row["pmra"],
                           pmdec=row["pmdec"], G=row["phot_g_mean_mag"], bp_rp=row["bp_rp"], **chk)
                if chk["K"] is None or not (krange[0] <= chk["K"] <= krange[1]):
                    rej.append(dict(rec, reason="K outside range or no 2MASS match"))
                    continue
                if chk["vsx_vizier"] is None and chk["vsx_api"] is None:
                    rej.append(dict(rec, reason="VSX unreachable"))
                    continue
                if (chk["vsx_vizier"] or 0) + (chk["vsx_api"] or 0) > 0:
                    rej.append(dict(rec, reason="in VSX within 5 arcsec"))
                    continue
                fr = n_frames(df, row["ra"], row["dec"])
                nw = int(windows(fr.mjd.values).max() + 1) if len(fr) else 0
                rec.update(n_indexed=int(len(fr)), n_windows=nw)
                if not (nfr[0] <= len(fr) <= nfr[1]):
                    rej.append(dict(rec, reason="number of indexed pictures outside range"))
                    continue
                if nwin and nw != nwin:
                    rej.append(dict(rec, reason="number of windows differs"))
                    continue
                if len(acc) < n_want:
                    acc.append(dict(rec, name=f"{cname}_c{start_idx + len(acc):02d}"))
            print(cname, "examined", pos, "accepted", len(acc), "rejected", len(rej), flush=True)
    return acc, rej, pos


def extend(which):
    """Second draw (recorded in docs/protocols/E9.md): made after the first comparison stars showed that
    (i) only RW-class stars at the bright end of the K range reach the target's flagged-core fraction, and
    (ii) the first rule left only 12 R-class stars. No target pixel was read."""
    import pyvo
    from build_field import load_index
    df, src = load_index()
    gaia = pyvo.dal.TAPService("https://gaia.aip.de/tap")
    d1 = json.load(open(f"{OUT}/comparison_draw.json"))["classes"]
    res = dict(seed=SEED, made=time.strftime("%Y-%m-%dT%H:%M:%S%z"), classes={})
    if which == "rdra":
        return extend_rdra(df, gaia, d1, res)
    # RW class: same pool and permutation, continued after the first draw's last examined position; K 6.02-6.90
    c = CLASSES["rwaur"]
    g = gaia_pool(gaia, c, "rwaur")
    assert len(g) == d1["rwaur"]["n_not_gaia_variable"], (len(g), d1["rwaur"]["n_not_gaia_variable"])
    order = np.random.default_rng(SEED + 0).permutation(len(g))
    acc, rej, pos = examine(df, g, order, d1["rwaur"]["n_examined"], 45, (6.02, 6.90), c["n_frames"], 2, "rwaur", 60)
    res["classes"]["rwaur"] = dict(rule="pool and permutation of the first draw continued from its last position; "
                                        "K 6.02-6.90; other rules unchanged", K=[6.02, 6.90], n_examined_to=pos,
                                   accepted=acc, rejected=rej)
    json.dump(res, open(f"{OUT}/comparison_draw2_rwaur.json", "w"), indent=1)
    json.dump(acc, open(f"{OUT}/download_2_rwaur.json", "w"), indent=1)
    print("rwaur extension", len(acc))


def extend_rdra(df, gaia, d1, res):
    # R class: (i) first-draw stars rejected only for < 800 indexed pictures, now accepted if >= 300;
    # (ii) the same rule around the south ecliptic pole (seed + 2)
    c = CLASSES["rdra"]
    acc1 = [dict({k: v for k, v in r.items() if k != "reason"}, name=f"rdra_c{12 + i:02d}") for i, r in enumerate(
        [r for r in d1["rdra"]["rejected"] if r["reason"].startswith("number") and r["n_indexed"] >= 300])]
    g = gaia_pool(gaia, c, "rdra", centre=(90.0, -66.5607))
    order = np.random.default_rng(SEED + 2).permutation(len(g))
    acc2, rej2, pos = examine(df, g, order, 0, 60, c["K"], (300, 10 ** 9), None, "rdra", 12 + len(acc1))
    res["classes"]["rdra"] = dict(rule=">= 300 indexed pictures (was 800); north-pole stars of the first draw that failed only "
                                       "that rule, then a 15 deg cone about the south ecliptic pole (RA 90, Dec -66.5607), "
                                       "permutation seed + 2", accepted=acc1 + acc2, rejected=rej2, n_pool_south=int(len(g)),
                                  n_examined_south=pos)
    json.dump(res, open(f"{OUT}/comparison_draw2_rdra.json", "w"), indent=1)
    json.dump(acc1 + acc2, open(f"{OUT}/download_2_rdra.json", "w"), indent=1)
    print("rdra extension", len(acc1), len(acc2))


if __name__ == "__main__":
    extend(sys.argv[2]) if sys.argv[1:2] == ["extend"] else main()
