"""E3 - recall of real movers (docs/PROTOCOLS.md, E3).

Truth: IMCCE SkyBoT known asteroids (geocentric, LOC=500: SPHEREx's low-Earth-orbit view differs by <~1" for
main-belt objects), V < 19, inside the 48 px field >= 3 px from every edge, per frame (build_field.known_asteroids,
cached queries in data/cache/skybot/; truth is computed on all cached frames, then the frame cap is applied).
Frame cap: fields with > 250 frames use 250 frames spread evenly in time (frames sorted by MJD,
np.linspace(0, n-1, 250).round()). Maps: M4 = change model revision 2 corrected map, M2 = consecutive blink of the
same detector (the asteroid frame is the later frame of the pair); both from orrery.change.change_model with the
frozen f. Detected: a change component (8-connected, >= 2 px, |map| > k_method) with any pixel within 2 px of the
predicted position. Pairs without an M2 map (first frame of a detector) are excluded for both methods so the two
methods are scored on identical pairs (count recorded).
Usage: .venv/bin/python scripts/e3_recall.py            (fields from data/findings/e3_fields.json + 6 existing + E2)
"""
import json, os, sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from orrery.change import change_model, _EIGHT  # noqa: E402
import build_field as bf  # noqa: E402

S, EDGE, VMAX, R_MATCH, CAP = 48, 3, 19.0, 2.0, 250
EXISTING = ["barnard", "wise0855", "epsindib", "rwaur", "rdra", "ecliptic_control"]


def thresholds():
    p = "data/findings/e1_thresholds.json"
    if os.path.exists(p):
        t = json.load(open(p))
        get = lambda m: t[m]["k"] if isinstance(t.get(m), dict) else t[m]
        return dict(M2=float(get("M2")), M4=float(get("M4"))), p
    return dict(M2=5.0, M4=5.0), "e1_thresholds.json not available: k = 5 for both (rev 2 frozen value)"


def wilson(k, n, z):
    if n == 0:
        return [None, None]
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(float(c - h), 4), round(float(c + h), 4)]


def components(m, k):
    if m is None:
        return None
    lab, n = ndimage.label(np.abs(np.nan_to_num(np.asarray(m, float))) > k, structure=_EIGHT)
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        jj, ii = np.nonzero(lab[sl] == i)
        if len(jj) >= 2:
            out.append(np.stack([ii + sl[1].start, jj + sl[0].start], 1))
    return out


def hit(comps, x, y):
    return any(np.hypot(c[:, 0] - x, c[:, 1] - y).min() <= R_MATCH for c in comps)


def load_field(name):
    """(d, ra0, dec0, source) for a field cached by build_field.py (data/cache/fields/<name>_S48.npz)."""
    d = dict(np.load(f"data/cache/fields/{name}_S48.npz", allow_pickle=True))
    f1 = json.load(open(f"data/findings/f1_{name}.json"))
    return d, f1["ra"], f1["dec"], f"data/cache/fields/{name}_S48.npz"


def run_field(name, d, ra0, dec0, src, k, f):
    mjd_all = d["mjd"].astype(float)
    ast_all, n_q = bf.known_asteroids(mjd_all, S, ra0, dec0)
    N0 = len(mjd_all)
    sel = np.argsort(mjd_all, kind="stable")
    if N0 > CAP:
        sel = sel[np.linspace(0, N0 - 1, CAP).round().astype(int)]
    sel = np.sort(sel)
    v, lam, det, mjd = d["v"][sel], d["lam"][sel], d["detector"][sel], mjd_all[sel]
    r = change_model(v, lam, det, mjd, f=f)
    rows = []
    n_no_m2 = 0
    for e, g in enumerate(sel):
        truth = [o for o in ast_all[g] if o["vmag"] < VMAX and EDGE <= o["x"] <= S - 1 - EDGE
                 and EDGE <= o["y"] <= S - 1 - EDGE]
        if not truth:
            continue
        if r["naive_det"][e] is None:
            n_no_m2 += len(truth)
            continue
        c4, c2 = components(r["corr"][e], k["M4"]), components(r["naive_det"][e], k["M2"])
        for o in truth:
            ix, iy = int(round(o["x"])), int(round(o["y"]))
            rows.append(dict(field=name, frame=int(e), obs_id=str(d["obs_id"][g]), detector=int(det[e]),
                             mjd=round(float(mjd[e]), 6), name=o["name"], vmag=o["vmag"], orbit_class=o["cls"],
                             x=round(o["x"], 2), y=round(o["y"], 2),
                             wl_um=round(float(lam[e][iy, ix]), 3) if np.isfinite(lam[e][iy, ix]) else None,
                             centre_has_data=bool(np.isfinite(v[e][iy, ix])),
                             M4=hit(c4, o["x"], o["y"]), M2=hit(c2, o["x"], o["y"])))
    rec = dict(field=name, source=src, n_frames_cached=N0, n_frames_used=len(sel), skybot_queries=n_q,
               n_pairs=len(rows), n_pairs_excluded_no_m2_map=n_no_m2,
               n_asteroids=len({q["name"] for q in rows}),
               M4_hits=sum(q["M4"] for q in rows), M2_hits=sum(q["M2"] for q in rows))
    return rec, rows, (r, v, d["v_raw"][sel], sel)


def summary(rows):
    out = dict(n_pairs=len(rows))
    for m in ("M4", "M2"):
        kk = sum(q[m] for q in rows)
        out[m] = dict(hits=kk, recall=round(kk / len(rows), 4) if rows else None,
                      wilson68=wilson(kk, len(rows), 1.0), wilson95=wilson(kk, len(rows), 1.96))
    ast = {}
    for q in rows:
        a = ast.setdefault((q["field"], q["name"]), dict(M4=False, M2=False, vmag=q["vmag"], n=0))
        a["M4"] |= q["M4"]; a["M2"] |= q["M2"]; a["n"] += 1
    out["per_asteroid"] = dict(n_asteroids=len(ast))
    for m in ("M4", "M2"):
        kk = sum(a[m] for a in ast.values())
        out["per_asteroid"][m] = dict(detected_in_ge1_frame=kk, recall=round(kk / len(ast), 4) if ast else None,
                                      wilson68=wilson(kk, len(ast), 1.0), wilson95=wilson(kk, len(ast), 1.96))
    out["by_vmag"] = []
    for lo, hi in ((0, 16), (16, 17), (17, 18), (18, 19)):
        sub = [q for q in rows if lo <= q["vmag"] < hi]
        out["by_vmag"].append(dict(vmag=[lo, hi], n=len(sub), **{m: sum(q[m] for q in sub) for m in ("M4", "M2")}))
    return out


def gallery(misses, cache, path, n=24):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    misses = sorted(misses, key=lambda q: q["vmag"])[:n]
    if not misses:
        return 0
    fig, ax = plt.subplots(2, len(misses), figsize=(1.3 * len(misses), 3.0), squeeze=False)
    for k_, q in enumerate(misses):
        r, v, v_raw, _ = cache[q["field"]]
        e = q["frame"]
        raw = (v_raw[e] - r["bg"][e]) / r["sigma"][e]
        ax[0, k_].imshow(np.arcsinh(raw / 3), cmap="gray", vmin=0, vmax=np.arcsinh(max(np.nanpercentile(raw, 99.5), 10) / 3))
        cm = plt.get_cmap("PuOr_r").copy(); cm.set_bad("#203050")
        ax[1, k_].imshow(r["corr"][e], cmap=cm, vmin=-8, vmax=8)
        for a in ax[:, k_]:
            a.plot(q["x"], q["y"], "o", mfc="none", mec="cyan", ms=9, mew=0.7)
            a.set_xlim(q["x"] - 8, q["x"] + 8); a.set_ylim(q["y"] + 8, q["y"] - 8)
            a.set_xticks([]); a.set_yticks([])
        ax[0, k_].set_title(f"{q['name']}\nV{q['vmag']} {q['wl_um']}um", fontsize=5)
        ax[1, k_].set_title(f"{q['field']} f{e}", fontsize=4.5)
    fig.suptitle("E3 misses (M4, brightest first): raw (top, unmasked) and rev-2 corrected map +-8 sigma (bottom; "
                 "navy = masked); cyan = SkyBoT position, 16 px boxes", fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return len(misses)


def main():
    k, k_src = thresholds()
    f = json.load(open("data/findings/f1_calibration.json"))["f"]
    e3f = json.load(open("data/findings/e3_fields.json"))["fields"]
    groups = dict(existing=EXISTING, ecliptic=[x["field"] for x in e3f])
    # E2 fields (data/efields/e2_fields.json); a field without data/efields/<field>/READY is listed as missing
    groups["e2"] = [x["field"] for x in json.load(open("data/efields/e2_fields.json"))["fields"]]
    recs, rows, cache, missing = [], [], {}, []
    for gname, names in groups.items():
        for nm in names:
            try:
                d, ra0, dec0, src = load_e2(nm) if gname == "e2" else load_field(nm)
            except FileNotFoundError as ex:
                missing.append(dict(field=nm, group=gname, reason=str(ex)))
                continue
            rec, rr, c = run_field(nm, d, ra0, dec0, src, k, f)
            rec["group"] = gname
            recs.append(rec); rows += rr; cache[nm] = c
            print(rec, flush=True)
    misses = [q for q in rows if not q["M4"]]
    n_gal = gallery(misses, cache, "data/findings/e3_misses.png")
    out = dict(
        experiment="E3 recall of real movers", protocol="docs/PROTOCOLS.md#e3", method=" ".join(__doc__.split("Usage:")[0].split()),
        thresholds=k, thresholds_source=k_src, noise_floor_f=f, vmag_max=VMAX, edge_px=EDGE, match_radius_px=R_MATCH,
        frame_cap=CAP, index="data/index/frames_dr1.parquet",
        frame_cap_note="ecliptic fields: cap applied to index rows before reading (build_field.py --max-frames 250); "
                       "existing fields: cap applied to the cached frames sorted by MJD (cache = all index candidates "
                       "whose grid centre is on the frame, 0 read failures; rdra 1635 of 1637)",
        truth_source="IMCCE SkyBoT conesearch, geocentric (LOC=500); approximation: SPHEREx is in low Earth orbit, "
                     "parallax offset <~1 arcsec (0.16 px) for main-belt asteroids",
        all=summary(rows), by_group={g: summary([q for q in rows if q["field"] in set(n)]) for g, n in groups.items()},
        per_field=recs, fields_missing=missing, gallery=dict(path="data/findings/e3_misses.png", n_shown=n_gal,
                                                            n_misses_M4=len(misses)),
        rows=rows)
    json.dump(out, open("data/findings/e3.json", "w"), indent=1)
    print(json.dumps({k_: v_ for k_, v_ in out.items() if k_ not in ("rows", "per_field", "method")}, indent=1))


def load_e2(name):
    """E2 field from data/efields/<name> (README there): native cutouts resampled with scripts/e2_counts.resample
    (= build_field.cut); v_raw is the same bilinear resample without the flag mask. Frames sorted (mjd, detector)."""
    from astropy.io import fits
    from astropy.wcs import WCS
    from orrery.change import bilinear, grid_world
    from orrery.data import load_stamp
    import e2_counts
    root = f"data/efields/{name}"
    if not os.path.exists(f"{root}/READY"):
        raise FileNotFoundError(f"{root}/READY")
    fj = json.load(open(f"{root}/field.json"))
    gra, gdec = grid_world(S, fj["ra"], fj["dec"])
    rows = []
    for fl in fj["frame_list"]:
        st = load_stamp(f"{root}/{fl['file']}")
        rr = e2_counts.resample(st, gra, gdec)
        if rr is None:
            continue
        x, y = WCS(fits.Header.fromstring(st.wcs_header)).all_world2pix(gra.ravel(), gdec.ravel(), 0)
        v_raw = bilinear(st.image, x.reshape(S, S) - st.x0, y.reshape(S, S) - st.y0).astype(np.float32)
        rows.append((st.mjd, st.detector, st.obs_id, rr[0], rr[1], v_raw))
    rows.sort(key=lambda t: (t[0], t[1]))
    d = dict(mjd=np.array([t[0] for t in rows]), detector=np.array([t[1] for t in rows]),
             obs_id=np.array([t[2] for t in rows]), v=np.array([t[3] for t in rows]),
             lam=np.array([t[4] for t in rows]), v_raw=np.array([t[5] for t in rows]))
    return d, fj["ra"], fj["dec"], f"{root} (scripts/e2_counts.resample)"


if __name__ == "__main__":
    main()
