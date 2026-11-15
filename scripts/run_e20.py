"""E20: hydrogen-line emission excess (Pa-alpha 1.875, Br-alpha 4.052, Br-gamma 2.166 um) in SPHEREx DR1 spectra of
Gaia DR3 ESP-ELS Herbig Ae/Be and Be candidates, against brightness/colour-matched control stars measured in the
same pictures. Frozen choices: docs/protocols/E20.md.

  OMP_NUM_THREADS=1 .venv/bin/python scripts/run_e20.py select    # seeded draw + Gaia fields -> data/phase2/e20/sample.json
  OMP_NUM_THREADS=1 .venv/bin/python scripts/run_e20.py fetch     # D3 + D5 cutouts (resumable) -> data/phase2/e20/fields/<id>/
  OMP_NUM_THREADS=1 .venv/bin/python scripts/run_e20.py phot      # forced photometry per picture -> data/phase2/e20/phot/<id>.json
  OMP_NUM_THREADS=1 .venv/bin/python scripts/run_e20.py analyze   # frozen statistic + rule -> data/findings/e20.json
  OMP_NUM_THREADS=1 .venv/bin/python scripts/run_e20.py fig       # site/data/figs/e20_lines.png
"""
import json, os, sys, time, warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
warnings.filterwarnings("ignore")

P2 = "data/phase2/e20"
OUT = "data/findings/e20.json"
FIG = "site/data/figs/e20_lines.png"
SEED = 2026111520
TAP = "https://gaia.aip.de/tap"
N_PER_CLASS = 30
CLASSES = {"HerbigStar": ("classprob_espels_herbigstar", 0.70), "beStar": ("classprob_espels_bestar", 0.90)}
G_RANGE = (9.5, 14.0)
GAIA_R_DEG = 0.1
CTRL_SEP = (20.0, 200.0)                       # arcsec from the candidate (inside the 77 px cutout)
CTRL_MATCH = [(1.0, 0.5), (1.5, 1.0)]          # (|dG|, |d(BP-RP)|), second only if the first gives < 2
N_CTRL, MIN_CTRL = 3, 2
DETS = (3, 5)
CAP = 160                                      # D3 + D5 pictures per field (E2 linspace rule)
LINES = {"Paa": (1.8751, 3), "Brg": (2.1655, 3), "Bra": (4.0523, 5)}
OTHER_EXCL = [1.8751, 2.1655, 4.0523, 2.0587]  # um: target lines + He I 2.0587, kept out of every continuum
CONT_MAX = 0.20                                # |lambda/lambda_L - 1| <= 0.20
LINE_HALF, CONT_GAP = 0.5, 1.0                 # in units of the picture's bandwidth at the star
MIN_SIDE = 3
SUB = 12                                       # photometry sub-cutout half size (px)
MIN_PIX = 18
SAT_R = 3.0
Z_DET, Z_WIN, N_WIN = 5.0, 2.0, 2


def tap(q, tries=4):
    import pyvo
    for i in range(tries):
        try:
            return pyvo.dal.TAPService(TAP).run_sync(q, maxrec=200000).to_table().to_pandas()
        except Exception as e:
            print(f"  TAP attempt {i + 1}: {e!r}"[:200], flush=True)
            time.sleep(5 * (i + 1))
    raise RuntimeError("TAP failed")


def sep_arcsec(ra1, dec1, ra2, dec2):
    a1, d1, a2, d2 = map(np.radians, (ra1, dec1, ra2, dec2))
    return np.degrees(2 * np.arcsin(np.sqrt(np.sin((d1 - d2) / 2) ** 2 + np.cos(d1) * np.cos(d2) * np.sin((a1 - a2) / 2) ** 2))) * 3600


def field_gaia(sid, ra, dec):
    p = f"{P2}/fields/{sid}/gaia.parquet"
    if os.path.exists(p):
        return pd.read_parquet(p)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    g = tap(f"SELECT source_id, ra, dec, pmra, pmdec, parallax, phot_g_mean_mag, bp_rp, ref_epoch FROM gaiadr3.gaia_source "
            f"WHERE 1=CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, {GAIA_R_DEG})) AND phot_g_mean_mag < 19")
    g.to_parquet(p + ".tmp", index=False)
    os.replace(p + ".tmp", p)
    return g


def isolated(g, i):
    """Pre-registered isolation rule for any measured star (row i of the field table)."""
    s = sep_arcsec(g.ra.values, g.dec.values, g.ra.values[i], g.dec.values[i])
    G, Gs = g.phot_g_mean_mag.values[i], g.phot_g_mean_mag.values
    other = (np.arange(len(g)) != i) & (s >= 3.0)          # Gaia sources within 3 arcsec are blend members
    if np.any(other & (s < 30) & (Gs < G)):
        return False, "brighter Gaia star within 30 arcsec"
    if np.any(other & (s < 15) & (Gs < G + 2.5)):
        return False, "Gaia star within 2.5 mag within 15 arcsec"
    if np.any(other & (s < 90) & (Gs < 8)):
        return False, "G < 8 star within 90 arcsec"
    if np.sum(other & (s < 30) & (Gs < 18)) > 10:
        return False, "crowded: > 10 Gaia G < 18 within 30 arcsec"
    return True, ""


def stage_select():
    from astropy.table import Table
    rng = np.random.default_rng(SEED)
    pools = {}
    for c, (pcol, pmin) in CLASSES.items():
        t = Table.read(f"{P2}/espels_{c}.ecsv").to_pandas()
        flag = t.classlabel_espels_flag.astype(str).astype(int)
        s = t[(t[pcol] >= pmin) & (flag < 10) & t.phot_g_mean_mag.between(*G_RANGE) & t.bp_rp.notna()]
        s = s.sort_values("source_id").reset_index(drop=True)
        pools[c] = s.iloc[rng.permutation(len(s))].reset_index(drop=True)
        print(c, "eligible", len(s), flush=True)
    pre = [r for c in CLASSES for r in pools[c].head(150).itertuples()]
    with ThreadPoolExecutor(4) as ex:                     # prefetch only (cache); the draw below stays sequential
        list(ex.map(lambda r: field_gaia(str(int(r.source_id)), float(r.ra), float(r.dec)), pre))
    labelled = set(int(x) for c in CLASSES for x in Table.read(f"{P2}/espels_{c}.ecsv")["source_id"])
    sample, rejected = [], []
    ptr = {c: 0 for c in CLASSES}
    while any(sum(x["cls"] == c for x in sample) < N_PER_CLASS and ptr[c] < len(pools[c]) for c in CLASSES):
        for c in CLASSES:
            if sum(x["cls"] == c for x in sample) >= N_PER_CLASS or ptr[c] >= len(pools[c]):
                continue
            r = pools[c].iloc[ptr[c]]
            ptr[c] += 1
            sid = str(int(r.source_id))
            g = field_gaia(sid, float(r.ra), float(r.dec)).reset_index(drop=True)
            ic = np.flatnonzero(g.source_id.values == int(sid))
            if len(ic) != 1:
                rejected.append(dict(source_id=sid, cls=c, reason="not found in field table")); continue
            ic = int(ic[0])
            ok, why = isolated(g, ic)
            if not ok:
                rejected.append(dict(source_id=sid, cls=c, reason=why)); continue
            s = sep_arcsec(g.ra.values, g.dec.values, float(r.ra), float(r.dec))
            ctrl, rule = [], None
            for dG, dC in CTRL_MATCH:
                m = np.flatnonzero((s >= CTRL_SEP[0]) & (s <= CTRL_SEP[1]) & (np.abs(g.phot_g_mean_mag.values - r.phot_g_mean_mag) <= dG)
                                   & (np.abs(g.bp_rp.values - r.bp_rp) <= dC) & ~np.isin(g.source_id.values, list(labelled)))
                m = [int(k) for k in m if isolated(g, int(k))[0]]
                if len(m) >= MIN_CTRL:
                    ctrl, rule = [m[k] for k in rng.permutation(len(m))], (dG, dC)
                    break
            if len(ctrl) < MIN_CTRL:
                rejected.append(dict(source_id=sid, cls=c, reason="fewer than 2 matched isolated controls")); continue
            # controls must carry no ESP-ELS label of any class
            ids = ",".join(str(int(g.source_id[k])) for k in ctrl[:8])
            lab = tap(f"SELECT source_id, classlabel_espels FROM gaiadr3.astrophysical_parameters WHERE source_id IN ({ids})")
            bad = set(int(x) for x, l in zip(lab.source_id, lab.classlabel_espels) if isinstance(l, str) and l.strip())
            ctrl = [k for k in ctrl[:8] if int(g.source_id[k]) not in bad][:N_CTRL]
            if len(ctrl) < MIN_CTRL:
                rejected.append(dict(source_id=sid, cls=c, reason="controls carry ESP-ELS labels")); continue
            sample.append(dict(source_id=sid, cls=c, ra=float(r.ra), dec=float(r.dec), l=float(r.l), b=float(r.b),
                               G=float(r.phot_g_mean_mag), bp_rp=float(r.bp_rp), prob=float(r[CLASSES[c][0]]),
                               flag=int(r.classlabel_espels_flag), ew_halpha_nm=float(r.ew_espels_halpha),
                               ctrl_match=list(rule), draw_rank=int(ptr[c] - 1),
                               controls=[dict(source_id=str(int(g.source_id[k])), ra=float(g.ra[k]), dec=float(g.dec[k]),
                                              G=float(g.phot_g_mean_mag[k]), bp_rp=float(g.bp_rp[k]), sep_arcsec=float(s[k]))
                                         for k in ctrl]))
            print(f"  {c} {sid} G={r.phot_g_mean_mag:.2f} controls={len(ctrl)} ({len(sample)} accepted)", flush=True)
    out = dict(seed=SEED, classes={c: dict(prob_col=p, prob_min=m) for c, (p, m) in CLASSES.items()}, g_range=G_RANGE,
               n_per_class=N_PER_CLASS, eligible={c: len(pools[c]) for c in CLASSES}, drawn=ptr,
               sample=sample, rejected=rejected)
    json.dump(out, open(f"{P2}/sample.json", "w"), indent=1)
    print("accepted", len(sample), "rejected", len(rejected), "controls", sum(len(x["controls"]) for x in sample))


def frame_lists():
    from build_field import candidates
    p = f"{P2}/frame_lists.json"
    if os.path.exists(p):
        return json.load(open(p))
    df = pd.read_parquet("data/index/frames_dr1.parquet")
    df = df[(df.dqa == "Pass") & df.detector.isin(DETS)].reset_index(drop=True)
    out = {}
    for s in json.load(open(f"{P2}/sample.json"))["sample"]:
        c = candidates(df, s["ra"], s["dec"]).sort_values("mjd", kind="stable").reset_index(drop=True)
        if len(c) > CAP:
            c = c.iloc[np.linspace(0, len(c) - 1, CAP).round().astype(int)]
        out[s["source_id"]] = [dict(key=r.key, obs_id=r.obs_id, detector=int(r.detector), mjd=float(r.mjd)) for r in c.itertuples()]
    json.dump(out, open(p, "w"))
    return out


def stage_fetch(threads=int(os.environ.get("E20_THREADS", "6"))):
    import e2_cache
    from orrery.data import save_stamp
    fl = frame_lists()
    sample = json.load(open(f"{P2}/sample.json"))["sample"]
    t0 = time.time()

    def one(s, f):
        p = f"{P2}/fields/{s['source_id']}/frames/{f['obs_id']}_D{f['detector']}.npz"
        if os.path.exists(p) or os.path.exists(p + ".off"):
            return "ok"
        for i in range(3):
            try:
                st = e2_cache.fast_stamp(f["key"], s["ra"], s["dec"])
                break
            except Exception as e:
                if i == 2:
                    return f"error {e!r}"[:200]
                time.sleep(5)
        if st is None:
            open(p + ".off", "w").close()
            return "off"
        save_stamp(st, p[:-4] + ".tmp.npz")
        os.replace(p[:-4] + ".tmp.npz", p)
        return "ok"

    with ThreadPoolExecutor(threads) as ex:
        for n, s in enumerate(sample):
            os.makedirs(f"{P2}/fields/{s['source_id']}/frames", exist_ok=True)
            res = list(ex.map(lambda f: one(s, f), fl[s["source_id"]]))
            err = [r for r in res if r.startswith("error")]
            if not err:
                open(f"{P2}/fields/{s['source_id']}/READY", "w").close()
            print(f"{n + 1}/{len(sample)} {s['cls']} {s['source_id']}: {len(res)} frames, {len(err)} errors, "
                  f"{time.time() - t0:.0f}s", flush=True)


def measured(s):
    return [dict(source_id=s["source_id"], role="candidate")] + [dict(source_id=c["source_id"], role="control") for c in s["controls"]]


def phot_one(sid):
    from astropy.io import fits
    from astropy.wcs import WCS
    from orrery import e6, e7
    from orrery.data import load_stamp
    from orrery.e1 import wave_map
    from orrery.fit import flux_mJy
    p = f"{P2}/phot/{sid}.json"
    d = f"{P2}/fields/{sid}"
    if os.path.exists(p) or not os.path.exists(f"{d}/READY"):
        return sid
    s = next(x for x in json.load(open(f"{P2}/sample.json"))["sample"] if x["source_id"] == sid)
    g = pd.read_parquet(f"{d}/gaia.parquet")
    g = g[g.phot_g_mean_mag < e6.GMAX].reset_index(drop=True)
    stars = measured(s)
    for m in stars:
        i0 = int(np.flatnonzero(g.source_id.values == int(m["source_id"]))[0])
        sep = sep_arcsec(g.ra.values, g.dec.values, g.ra.values[i0], g.dec.values[i0])
        m["members"] = [i0] + [int(k) for k in np.flatnonzero(sep < e7.BLEND_ARCSEC) if k != i0]
    rec = {m["source_id"]: dict(role=m["role"], obs_id=[], det=[], mjd=[], lam=[], bw=[], flux=[], err=[], npix=[], sat=[])
           for m in stars}
    files = sorted(f for f in os.listdir(f"{d}/frames") if f.endswith(".npz"))
    for fn in files:
        st = load_stamp(f"{d}/frames/{fn}")
        w = WCS(fits.Header.fromstring(st.wcs_header))
        xy, _ = e6.star_xy_native(g, st.mjd, w, st.x0, st.y0)
        H, W = st.image.shape
        for m in stars:
            x, y = xy[m["members"][0]]
            if not (4 <= x <= W - 5 and 4 <= y <= H - 5):
                continue
            ix, iy = int(round(x)), int(round(y))
            sx0, sy0 = max(0, ix - SUB), max(0, iy - SUB)
            sx1, sy1 = min(W, ix + SUB + 1), min(H, iy + SUB + 1)
            near = np.flatnonzero((xy[:, 0] >= sx0 - 3) & (xy[:, 0] <= sx1 + 2) & (xy[:, 1] >= sy0 - 3) & (xy[:, 1] <= sy1 + 2))
            loc = {int(k): j for j, k in enumerate(near)}
            sub_xy = xy[near] - [sx0, sy0]
            shape = (sy1 - sy0, sx1 - sx0)
            cols = e6.star_columns(sub_xy, st.detector, st.x0 + sx0, st.y0 + sy0, shape)
            grp = [loc[k] for k in m["members"] if k in loc]
            sl = (slice(sy0, sy1), slice(sx0, sx1))
            a, e, npix = e7.forced_photometry(st.image[sl], st.variance[sl], st.flags[sl], cols, [grp])[0]
            yy, xx = np.indices((H, W))
            core = np.hypot(xx - x, yy - y) <= SAT_R
            r = rec[m["source_id"]]
            r["obs_id"].append(st.obs_id), r["det"].append(int(st.detector)), r["mjd"].append(float(st.mjd))
            r["lam"].append(float(wave_map(st.detector, x + st.x0, y + st.y0)))
            r["bw"].append(float(st.cband[iy, ix]))
            r["flux"].append(flux_mJy(a, st.omega_arcsec2) if np.isfinite(a) else None)
            r["err"].append(flux_mJy(e, st.omega_arcsec2) if np.isfinite(e) else None)
            r["npix"].append(int(npix)), r["sat"].append(bool((st.flags[core] & e7.SAT_BITS).any()))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(dict(source_id=sid, n_files=len(files), stars=rec), open(p + ".tmp", "w"), separators=(",", ":"))
    os.replace(p + ".tmp", p)
    return sid


def stage_phot(procs=int(os.environ.get("E20_PROCS", "2"))):
    from multiprocessing import Pool
    os.makedirs(f"{P2}/phot", exist_ok=True)
    sample = json.load(open(f"{P2}/sample.json"))["sample"]
    todo = [s["source_id"] for s in sample if os.path.exists(f"{P2}/fields/{s['source_id']}/READY")
            and not os.path.exists(f"{P2}/phot/{s['source_id']}.json")]
    with Pool(procs) as p:
        for sid in p.imap_unordered(phot_one, todo):
            print("phot", sid, flush=True)


# ------------------------------------------------------------------ frozen statistic
def anomaly_set():
    a = pd.read_csv("data/ref/spherex_exposure_anomaly_log_20250810.csv")
    return {(o, d) for o, row in zip(a.OBSID, a.itertuples()) for d in range(1, 7) if getattr(row, f"BAND_{d}_AFFECTED") > 0}


def star_arrays(r, s_d, anom):
    n = len(r["flux"])
    f = np.array([np.nan if v is None else v for v in r["flux"]], float)
    e = np.array([np.nan if v is None else v for v in r["err"]], float)
    det = np.array(r["det"], int)
    e = e * np.array([s_d[str(d)] for d in det]) if n else e
    bad_anom = np.array([(o, d) in anom for o, d in zip(r["obs_id"], det)], bool)
    ok = np.isfinite(f) & np.isfinite(e) & (e > 0) & (np.array(r["npix"]) >= MIN_PIX) & ~np.array(r["sat"], bool) & ~bad_anom
    a = {k: np.array(r[k])[ok] for k in ("lam", "bw", "mjd", "obs_id")}
    a.update(flux=f[ok], err=e[ok], det=det[ok])
    return a, dict(n_pictures=n, n_used=int(ok.sum()), n_sat=int(np.sum(r["sat"])), n_anomaly=int(bad_anom.sum()))


def window_excess(lam, bw, f, e, lamL):
    """One visit window, one line: continuum line fit on both sides, fractional excess of the line pictures."""
    x = lam / lamL - 1
    line = np.abs(lam - lamL) <= LINE_HALF * bw
    cont = (np.abs(lam - lamL) > CONT_GAP * bw) & (np.abs(x) <= CONT_MAX)
    for lo in OTHER_EXCL:
        if lo != lamL:
            cont &= np.abs(lam - lo) > LINE_HALF * bw
    if line.sum() < 1:
        return None, "no line picture"
    keep = cont.copy()
    for it in range(2):
        if (keep & (lam < lamL)).sum() < MIN_SIDE or (keep & (lam > lamL)).sum() < MIN_SIDE:
            return None, "continuum: < 3 pictures on a side"
        A = np.stack([np.ones(keep.sum()), lam[keep] - lamL], 1)
        Aw, bw_ = A / e[keep, None], f[keep] / e[keep]
        c, *_ = np.linalg.lstsq(Aw, bw_, rcond=None)
        res = f[keep] - A @ c
        if it == 0:
            mad = 1.4826 * np.median(np.abs(res - np.median(res)))
            drop = np.abs(res) > 5 * np.maximum(e[keep], mad)
            idx = np.flatnonzero(keep)
            keep[idx[drop]] = False
    dof = keep.sum() - 2
    chi2r = float(np.sum((res / e[keep]) ** 2) / dof) if dof > 0 else 1.0
    cov = np.linalg.inv(Aw.T @ Aw) * max(1.0, chi2r)
    L = np.stack([np.ones(line.sum()), lam[line] - lamL], 1)
    C = L @ c
    sC = np.sqrt(np.einsum("ij,jk,ik->i", L, cov, L))
    if np.any(C <= 0):
        return None, "continuum <= 0"
    r = f[line] / C - 1
    sr = np.sqrt(e[line] ** 2 + (f[line] / C) ** 2 * sC ** 2) / C
    w = 1 / sr ** 2
    E, sE = float(np.sum(w * r) / w.sum()), float(1 / np.sqrt(w.sum()))
    return dict(E=E, sE=sE, z=E / sE, n_line=int(line.sum()), n_cont=int(keep.sum()), chi2r_cont=chi2r,
                cont_mJy=float(np.mean(C)), lam_line=lam[line].tolist(), r_line=r.tolist(), sr_line=sr.tolist()), "ok"


def star_lines(a, win_of):
    out = {}
    for L, (lamL, dL) in LINES.items():
        s = a["det"] == dL
        win = np.array([win_of[m] for m in a["mjd"][s]], int)
        wins = {}
        for w in np.unique(win):
            k = win == w
            res, why = window_excess(a["lam"][s][k], a["bw"][s][k], a["flux"][s][k], a["err"][s][k], lamL)
            wins[int(w)] = res if res else dict(skip=why)
        good = [v for v in wins.values() if "E" in v]
        o = dict(windows=wins, n_windows=len(good))
        if good:
            wt = np.array([1 / v["sE"] ** 2 for v in good])
            E = float(np.sum(wt * [v["E"] for v in good]) / wt.sum())
            sE = float(1 / np.sqrt(wt.sum()))
            o.update(E=E, sE=sE, z=E / sE)
        out[L] = o
    return out


def robust(z):
    z = np.asarray(z, float)
    m = float(np.median(z))
    return m, float(1.4826 * np.median(np.abs(z - m)))


def stage_analyze():
    from scipy.stats import beta as B
    from orrery.e7 import windows
    s_d = json.load(open("data/findings/e7_gate0.json"))["s_d"]
    anom = anomaly_set()
    sample = json.load(open(f"{P2}/sample.json"))["sample"]
    rows = []
    for s in sample:
        p = f"{P2}/phot/{s['source_id']}.json"
        if not os.path.exists(p):
            continue
        ph = json.load(open(p))["stars"]
        mj = sorted(set(m for r in ph.values() for m in r["mjd"]))
        win_of = dict(zip(mj, windows(np.array(mj)))) if mj else {}
        for sid, r in ph.items():
            a, q = star_arrays(r, s_d, anom)
            rows.append(dict(source_id=sid, role=r["role"], field=s["source_id"], cls=s["cls"],
                             G=s["G"] if r["role"] == "candidate" else next(c["G"] for c in s["controls"] if c["source_id"] == sid),
                             **q, lines=star_lines(a, win_of)))
    cal = {}
    for L in LINES:
        zc = [r["lines"][L]["z"] for r in rows if r["role"] == "control" and "z" in r["lines"][L]]
        zw = [v["z"] for r in rows if r["role"] == "control" for v in r["lines"][L]["windows"].values() if "z" in v]
        m, sc = robust(zc) if len(zc) >= 10 else (0.0, 1.0)
        mw, sw = robust(zw) if len(zw) >= 10 else (0.0, 1.0)
        cal[L] = dict(n_controls=len(zc), median_z=m, scale=sc, n_control_windows=len(zw), median_zw=mw, scale_w=sw)
    for r in rows:
        r["detected"] = {}
        for L in LINES:
            o, c = r["lines"][L], cal[L]
            if "z" in o:
                o["Z"] = (o["z"] - c["median_z"]) / c["scale"]
                for v in o["windows"].values():
                    if "z" in v:
                        v["Z"] = (v["z"] - c["median_zw"]) / c["scale_w"]
                nw = sum(v.get("Z", -99) >= Z_WIN for v in o["windows"].values())
                o["n_windows_Zw_ge2"] = int(nw)
            o["testable"] = o["n_windows"] >= N_WIN
            r["detected"][L] = bool(o["testable"] and o["Z"] >= Z_DET and o["n_windows_Zw_ge2"] >= N_WIN)
        r["testable_main"] = r["lines"]["Paa"]["testable"] or r["lines"]["Bra"]["testable"]
        r["excess_main"] = r["detected"]["Paa"] or r["detected"]["Bra"]

    def summary(sel):
        n = sum(1 for r in sel if r["testable_main"])
        k = sum(1 for r in sel if r["excess_main"])
        per = {L: dict(testable=sum(r["lines"][L]["testable"] for r in sel), detected=sum(r["detected"][L] for r in sel)) for L in LINES}
        return n, k, per
    ctrl = [r for r in rows if r["role"] == "control"]
    nc, kc, perc = summary(ctrl)
    p_fp = kc / nc if nc else None
    up95 = float(B.ppf(0.95, kc + 1, nc - kc)) if nc else None
    FP_EXPECT = 0.02
    out = dict(experiment="E20", protocol="docs/protocols/E20.md", seed=SEED, lines_um={k: v[0] for k, v in LINES.items()},
               rule=dict(z_det=Z_DET, z_window=Z_WIN, n_windows=N_WIN, line_half_bw=LINE_HALF, cont_gap_bw=CONT_GAP,
                         cont_max_frac=CONT_MAX, min_side=MIN_SIDE, fp_expectation=FP_EXPECT),
               calibration_from_controls=cal,
               n_fields_measured=len({r["field"] for r in rows}), n_fields_selected=len(sample),
               controls=dict(n=len(ctrl), n_testable=nc, n_pass=kc, fp_rate=p_fp, fp_rate_upper95=up95, per_line=perc))
    out["method_works"] = bool(nc > 0 and p_fp <= FP_EXPECT)
    for c in CLASSES:
        cand = [r for r in rows if r["role"] == "candidate" and r["cls"] == c]
        n, k, per = summary(cand)
        out[f"cand_{c}"] = dict(n_measured=len(cand), M_testable=n, N_excess=k, per_line=per,
                                expected_fp=(p_fp * n if p_fp is not None else None),
                                expected_fp_upper95=(up95 * n if up95 is not None else None),
                                detected_ids=[r["source_id"] for r in cand if r["excess_main"]],
                                detected_brg_ids=[r["source_id"] for r in cand if r["detected"]["Brg"]])
    wise = json.load(open(f"{P2}/allwise.json")) if os.path.exists(f"{P2}/allwise.json") else {}
    simb = pd.read_csv(f"{P2}/simbad_ids.csv") if os.path.exists(f"{P2}/simbad_ids.csv") else None
    names = {} if simb is None else {i.replace("Gaia DR3 ", ""): (m, o, int(n)) for i, m, o, n in zip(simb.id, simb.main_id, simb.otype, simb.nbref)}
    for r in rows:
        if r["role"] == "candidate":
            w = wise.get(r["source_id"])
            r["allwise"] = w
            r["midir_excess_flag"] = None if not w else w["midir_excess_flag"]
            if r["source_id"] in names:
                r["simbad_main_id"], r["simbad_otype"], r["simbad_nbref"] = names[r["source_id"]]
    det = [r for r in rows if r["role"] == "candidate" and r["excess_main"]]
    out["detections_midir_flag"] = dict(n=len(det), n_flag_true=sum(bool(r.get("midir_excess_flag")) for r in det),
                                        n_flag_unknown=sum(r.get("midir_excess_flag") is None for r in det),
                                        note="AllWISE W1-W3 >= 1.0 mag (W3 S/N >= 5); reported only, not used for stellar vs nebular origin")
    out["protocol_freeze"] = json.load(open(f"{P2}/protocol_freeze.json"))
    cand = [r for r in rows if r["role"] == "candidate"]
    for L in LINES:                               # descriptive: size of the excess (fraction of the channel's continuum)
        e = [r["lines"][L]["E"] for r in cand if r["detected"][L]]
        out[f"median_frac_excess_detected_{L}"] = float(np.median(e)) if e else None
        out[f"range_frac_excess_detected_{L}"] = [float(min(e)), float(max(e))] if e else None
    out["n_candidates_not_testable"] = sum(not r["testable_main"] for r in cand)
    out["n_not_testable_mostly_saturated"] = sum(not r["testable_main"] and r["n_sat"] > 0.5 * r["n_pictures"] for r in cand)
    out["control_passes"] = [dict(source_id=r["source_id"], field=r["field"], G=r["G"],
                                  Z={L: r["lines"][L].get("Z") for L in LINES}) for r in ctrl if r["excess_main"]]
    known = sorted([r for r in cand if r.get("simbad_nbref", 0) >= 50], key=lambda r: -r["simbad_nbref"])
    out["well_known_in_sample"] = [dict(source_id=r["source_id"], name=r["simbad_main_id"], otype=r["simbad_otype"],
                                        simbad_nbref=r["simbad_nbref"], cls=r["cls"], testable=r["testable_main"],
                                        excess=r["excess_main"], n_sat=r["n_sat"], n_pictures=r["n_pictures"],
                                        Z={L: r["lines"][L].get("Z") for L in LINES}) for r in known]
    tot_M = sum(out[f"cand_{c}"]["M_testable"] for c in CLASSES)
    tot_N = sum(out[f"cand_{c}"]["N_excess"] for c in CLASSES)
    out.update(N_total=tot_N, M_total=tot_M, n_candidates=len(cand), n_controls=len(ctrl),
               N_herbig=out["cand_HerbigStar"]["N_excess"], M_herbig=out["cand_HerbigStar"]["M_testable"],
               N_be=out["cand_beStar"]["N_excess"], M_be=out["cand_beStar"]["M_testable"],
               control_fp=kc, control_testable=nc, control_fp_rate=p_fp, control_fp_rate_upper95=up95,
               expected_fp_total=(p_fp * tot_M if p_fp is not None else None),
               verdict="method works (control false-positive rate <= 0.02)" if out["method_works"] else "FAILED: control false-positive rate > 0.02")
    if os.path.exists(f"{P2}/offline_diagnostic.json"):
        out["posthoc_offline_diagnostic"] = json.load(open(f"{P2}/offline_diagnostic.json"))
    out["sentence_herbig"] = (f"SPHEREx shows Paα/Brα excess consistent with emission-line stars for {out['N_herbig']} of {out['M_herbig']} "
                              "Gaia ESP-ELS Herbig Ae/Be candidates; low resolution cannot separate stellar from nebular emission.")
    out["sentence_be"] = (f"SPHEREx shows Paα/Brα excess consistent with emission-line stars for {out['N_be']} of {out['M_be']} "
                          "Gaia ESP-ELS Be candidates; low resolution cannot separate stellar from nebular emission.")
    out["summary_plain"] = (
        f"We picked {len(cand)} stars that Gaia's software flags as likely emission-line stars (30 young Herbig Ae/Be stars, "
        f"30 Be stars), plus {len(ctrl)} ordinary stars of similar brightness and colour in the same SPHEREx pictures. "
        "For each star we compared its brightness in the SPHEREx channel that contains a hydrogen line (Paschen-alpha at "
        "1.875 µm or Brackett-alpha at 4.05 µm) with the neighbouring channels, using rules written down before measuring. "
        f"{out['N_total']} of the {out['M_total']} measurable candidates show a clear bump at a hydrogen line in at least two "
        f"separate visits ({out['N_herbig']} of {out['M_herbig']} Herbig, {out['N_be']} of {out['M_be']} Be); only "
        f"{kc} of {nc} ordinary stars do (both in the bright nebula around Trumpler 16), so fewer than one of the candidate "
        f"results is expected to be a fluke. {out['n_not_testable_mostly_saturated']} candidates were too bright to measure. "
        "SPHEREx's colour resolution is coarse, so it cannot tell whether the hydrogen glow comes from the star or from gas "
        "around it.")
    out["stars"] = rows
    json.dump(out, open(OUT + ".tmp", "w"), indent=1)
    os.replace(OUT + ".tmp", OUT)
    print(json.dumps({k: v for k, v in out.items() if k != "stars"}, indent=1))


def stage_wise():
    """AllWISE (VizieR II/328, CfA mirror) nearest source within 3 arcsec of each candidate: W1-W3, W3-W4 and the
    reported-only mid-IR excess flag (W1 - W3 >= 1.0 mag with W3 S/N >= 5)."""
    import io
    import requests
    p = f"{P2}/allwise.json"
    out = json.load(open(p)) if os.path.exists(p) else {}
    for s in json.load(open(f"{P2}/sample.json"))["sample"]:
        if s["source_id"] in out:
            continue
        url = ("https://vizier.cfa.harvard.edu/viz-bin/asu-tsv?-source=II/328/allwise&-c=%.6f%+.6f&-c.rs=3&-sort=_r"
               "&-out=AllWISE,W1mag,e_W1mag,W2mag,e_W2mag,W3mag,e_W3mag,W4mag,e_W4mag,ccf,ex&-out.add=_r" % (s["ra"], s["dec"]))
        try:
            t = requests.get(url, timeout=60).text
        except Exception as e:
            print("allwise", s["source_id"], repr(e)[:100]); continue
        lines = [l for l in t.splitlines() if l and not l.startswith("#")]
        if "#Table\tII_328_allwise" not in t:
            print("allwise: unexpected reply for", s["source_id"]); continue
        if len(lines) < 4:                       # header, units, dashes, rows
            out[s["source_id"]] = None; continue
        hdr = lines[0].split("\t")
        row = dict(zip(hdr, lines[3].split("\t")))
        def val(k):
            try:
                return float(row[k])
            except (KeyError, ValueError):
                return None
        d = {k: val(k) for k in ("W1mag", "e_W1mag", "W2mag", "W3mag", "e_W3mag", "W4mag", "e_W4mag", "_r")}
        d.update(AllWISE=row.get("AllWISE"), ccf=row.get("ccf"), ex=row.get("ex"))
        d["W1_W3"] = d["W1mag"] - d["W3mag"] if d["W1mag"] is not None and d["W3mag"] is not None else None
        d["W3_W4"] = d["W3mag"] - d["W4mag"] if d["W3mag"] is not None and d["W4mag"] is not None else None
        snr3 = 1.0857 / d["e_W3mag"] if d["e_W3mag"] else 0.0
        d["midir_excess_flag"] = bool(d["W1_W3"] is not None and d["W1_W3"] >= 1.0 and snr3 >= 5)
        out[s["source_id"]] = d
        json.dump(out, open(p, "w"), indent=1)
    print("allwise", sum(v is not None for v in out.values()), "of", len(out))


CLS_NAME = {"HerbigStar": "Herbig Ae/Be", "beStar": "Be"}
BLUE, ORANGE, INK, MUTED, RED ="#2a78d6", "#eb6834", "#1a1a19", "#6b6a63", "#c23b22"


def stage_fig(example=os.environ.get("E20_EXAMPLE"), FIG=os.environ.get("E20_FIG", FIG)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    o = json.load(open(OUT))
    s_d = json.load(open("data/findings/e7_gate0.json"))["s_d"]
    anom = anomaly_set()
    rows = o["stars"]
    cand = [r for r in rows if r["role"] == "candidate"]
    key = lambda r: max(r["lines"][L].get("Z", -99) for L in ("Paa", "Bra"))
    ex = next(r for r in cand if r["source_id"] == example) if example else max(cand, key=key)
    ctl = next(r for r in rows if r["field"] == ex["field"] and r["role"] == "control")
    ph = json.load(open(f"{P2}/phot/{ex['field']}.json"))["stars"]
    plt.rcParams.update({"font.size": 13, "font.family": "DejaVu Sans", "axes.edgecolor": MUTED, "text.color": INK})
    fig = plt.figure(figsize=(13, 9.5), dpi=140)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.42, wspace=0.22)
    for j, (L, lo, hi, nm) in enumerate((("Paa", 1.62, 2.30, "Paschen-α 1.875 µm"), ("Bra", 3.80, 4.42, "Brackett-α 4.05 µm"))):
        ax = fig.add_subplot(gs[0, j])
        lamL, dL = LINES[L]
        for r, col, lab in ((ex, ORANGE, f"Gaia ESP-ELS {CLS_NAME[ex['cls']]} candidate"), (ctl, BLUE, "control star, same pictures")):
            a, _ = star_arrays(ph[r["source_id"]], s_d, anom)
            s = (a["det"] == dL) & (a["lam"] >= lo) & (a["lam"] <= hi)
            cont = r["lines"][L].get("windows", {})
            norm = np.nanmedian([v["cont_mJy"] for v in cont.values() if "cont_mJy" in v]) if any("cont_mJy" in v for v in cont.values()) else np.nanmedian(a["flux"][s])
            ax.errorbar(a["lam"][s], a["flux"][s] / norm, yerr=a["err"][s] / norm, fmt="o", ms=5, color=col, alpha=0.85, label=lab, elinewidth=1)
            if r is ex:
                q = np.percentile(a["flux"][s] / norm, [2, 98])
                ax.set_ylim(q[0] - 0.25 * (q[1] - q[0]) - 0.05, q[1] + 0.35 * (q[1] - q[0]) + 0.05)
        for l2, n2 in ((1.8751, "Paα"), (2.1655, "Brγ"), (4.0523, "Brα")):
            if lo < l2 < hi:
                ax.axvline(l2, color=RED, lw=1.2, ls="--")
                ax.text(l2, ax.get_ylim()[1], " " + n2, color=RED, va="top", fontsize=12)
        Z = ex["lines"][L].get("Z")
        ax.set_title(f"{nm}: candidate Z = {Z:.1f}" if Z is not None else f"{nm}: not testable", loc="left", fontsize=13)
        ax.set_xlabel("wavelength (µm)"), ax.set_ylabel("brightness / local continuum")
        ax.spines[["top", "right"]].set_visible(False)
        if j == 0:
            ax.legend(fontsize=10.5, frameon=False, loc="lower left")
    ax = fig.add_subplot(gs[1, :])
    bins = np.linspace(-8, 12, 41)
    for role, col in (("control", BLUE), ("candidate", ORANGE)):
        z = [max(r["lines"][L].get("Z", -99) for L in ("Paa", "Bra")) for r in rows if r["role"] == role and r["testable_main"]]
        z = np.clip(z, bins[0], bins[-1] - 1e-6)
        ax.hist(z, bins=bins, histtype="stepfilled" if role == "control" else "step", color=col, alpha=0.45 if role == "control" else 1,
                lw=2.2, weights=np.full(len(z), 1 / len(z)), label=f"{role}s (n = {len(z)})")
    ax.axvline(Z_DET, color=RED, lw=1.5, ls="--")
    ax.text(Z_DET + 0.2, ax.get_ylim()[1] * 0.9, f"threshold Z = {Z_DET:g}\n(+ ≥ 2 visits)", color=RED, fontsize=11, va="top")
    ax.set_xlabel("line excess Z (larger of Paα, Brα; calibrated on controls; clipped to the axis)")
    ax.set_ylabel("fraction of stars")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    ac = [o[f"cand_{c}"] for c in CLASSES]
    if os.environ.get("E20_TITLE"):
        fig.suptitle(os.environ["E20_TITLE"], x=0.06, ha="left", fontsize=15, fontweight="bold")
    else:
        fig.suptitle(f"SPHEREx hydrogen-line excess: {sum(x['N_excess'] for x in ac)} of {sum(x['M_testable'] for x in ac)} Gaia ESP-ELS candidates; "
                     f"{o['controls']['n_pass']} of {o['controls']['n_testable']} controls", x=0.06, ha="left", fontsize=15, fontweight="bold")
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, facecolor="white", bbox_inches="tight")
    print("figure", FIG, "example", ex["source_id"])


if __name__ == "__main__":
    os.makedirs(P2, exist_ok=True)
    {"select": stage_select, "fetch": stage_fetch, "phot": stage_phot, "analyze": stage_analyze, "fig": stage_fig, "wise": stage_wise}[sys.argv[1]]()
