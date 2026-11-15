"""E15 (docs/protocols/E15.md): can SPHEREx's 0.75-5 um colours tell methane-bearing (T/Y) brown dwarfs from L dwarfs?

Stages (resumable; partial results on disk under data/phase2/e15/):
  calib      draw the calibration set from Kirkpatrick et al. 2021 (J/ApJS/253/7, table A1) by a fixed rule
  cands      list candidates (no spectroscopic type in K21) by the fixed rule
  forced N   per-picture forced ePSF fits for list N (calib|cands) -> data/phase2/e15/forced/<obj>/<obs>_D<d>.json
  analyze    bins, index, threshold, leave-one-out accuracy, candidate classification, figures, data/findings/e15.json
Usage: OMP_NUM_THREADS=1 .venv/bin/python scripts/run_e15.py calib|cands|forced calib|forced cands|analyze [--workers 5]
"""
import argparse, glob, json, os, sys, time, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
warnings.filterwarnings("ignore")
from astropy import log
log.setLevel("ERROR")
from astropy.time import Time

P = "data/phase2/e15"
K21 = f"{P}/k21_tablea1.tsv"
INDEX = "data/index/frames_dr1.parquet"
SEED = 15
EP_C2 = 57170.0                      # CatWISE2020 position epoch in K21 table A1 (MJD)
EXCL_ARCSEC = 3.0                    # Gaia sources this close to the target are the target itself (or an unresolvable blend)
GAIA_R = 0.07                        # deg; one Gaia DR3 query per object covers every stamp along the path
# Frozen bins (um) and index definitions (docs/protocols/E15.md)
BINS = {"m33": (3.2, 3.4), "c30": (2.9, 3.1), "c40": (3.9, 4.1), "r45": (4.4, 5.0), "j": (1.0, 1.3)}
NSIG = 3.0
ACC_MIN = 0.80


def load_k21():
    d = pd.read_csv(K21, sep="\t", comment="#", dtype=str).iloc[2:].reset_index(drop=True)
    d = d.apply(lambda c: c.str.strip())
    for c in ["SpTO", "SpTIR", "SpAd", "W1magC2", "W2magC2", "W1mag", "W2mag", "pmRA", "pmDE", "pmRAC2", "pmDEC2", "RACdeg", "DECdeg"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["w2"] = d.W2magC2.fillna(d.W2mag)
    d["w1"] = d.W1magC2.fillna(d.W1mag)
    d["spt"] = d.SpTIR.fillna(d.SpTO)                      # IR type preferred (K21 codes: L0 = 0, T0 = 10, Y0 = 20)
    d["pmra_mas"] = d.pmRA.fillna(d.pmRAC2 * 1000)
    d["pmdec_mas"] = d.pmDE.fillna(d.pmDEC2 * 1000)
    from astropy.coordinates import SkyCoord
    d["glat"] = SkyCoord(d.RACdeg.values, d.DECdeg.values, unit="deg").galactic.b.deg
    d["obj"] = d.Name.str.replace(" ", "_").str.replace("+", "p")
    return d


def spt_str(s):
    if not np.isfinite(s):
        return "none"
    letter = "M" if s < 0 else "L" if s < 10 else "T" if s < 20 else "Y"
    v = s + 10 if s < 0 else s % 10
    return f"{letter}{v:g}"


CLASSES = [("L0-L4.5", 0, 5), ("L5-L9.5", 5, 10), ("T0-T4.5", 10, 15), ("T5-T9.5", 15, 20), ("Y", 20, 30)]


def stage_calib():
    d = load_k21()
    c = d[(d["T"] == "T") & d.spt.notna() & (d.N == "") & (d.m_PName.fillna("") == "") &
          (d.w2 > 10) & (d.w2 < 15) & (d.glat.abs() > 10) & d.pmra_mas.notna()].copy()
    rng = np.random.default_rng(SEED)
    out = []
    for name, lo, hi in CLASSES:
        g = c[(c.spt >= lo) & (c.spt < hi)].sort_values("Name")
        out.append(g.iloc[rng.permutation(len(g))[:5]].assign(cls=name))
    s = pd.concat(out)
    s.to_csv(f"{P}/calib.csv", index=False)
    print(s[["Name", "cls", "SpTO", "SpTIR", "spt", "w2", "glat"]].to_string())


def stage_cands():
    d = load_k21()
    cal = set(pd.read_csv(f"{P}/calib.csv").Name)
    c = d[d.SpTO.isna() & d.SpTIR.isna() & (d.w2 < 15) & ((d.w1 - d.w2) > 1.5) & ~d.Name.isin(cal) & d.pmra_mas.notna()].copy()
    c.to_csv(f"{P}/cands_k21.csv", index=False)
    print(len(c))
    print(c[["Name", "FName", "SpAd", "w2", "w1", "T", "N", "New"]].to_string())


SIMBAD_TAP = "https://simbad.harvard.edu/simbad/sim-tap/sync"   # CDS SIMBAD TAP mirror (Strasbourg returned 404)


def stage_lit():
    """Prior-spectrum check for every K21 candidate: (1) Kirkpatrick et al. 2024 (J/ApJS/271/55, table 4) spectral
    types by designation; (2) SIMBAD sp_type + its bibcode and all post-2020 references; (3) arXiv API search of the
    short designation. Raw answers saved to data/phase2/e15/lit/."""
    import requests, urllib.parse
    os.makedirs(f"{P}/lit", exist_ok=True)
    c = pd.read_csv(f"{P}/cands_k21.csv")
    k24 = pd.read_csv(f"{P}/k24_table4.tsv", sep="\t", comment="#", dtype=str).iloc[2:].apply(lambda x: x.str.strip())
    out = []
    for _, row in c.iterrows():
        rec = dict(name=row["Name"], designation=row["FName"])
        m = k24[k24.Name == row["FName"]]
        rec["k24_found"] = bool(len(m))
        rec["k24_SpTOpt"] = m.SpTOpt.iloc[0] if len(m) else None
        rec["k24_SpTNIR"] = m.SpTNIR.iloc[0] if len(m) else None
        ra, dec = position(row, 51544.5)                         # SIMBAD coordinates are epoch J2000
        q = ("SELECT b.oid, b.main_id, b.sp_type, b.sp_bibcode, b.ra, b.dec FROM basic AS b WHERE "
             f"CONTAINS(POINT('ICRS', b.ra, b.dec), CIRCLE('ICRS', {ra}, {dec}, {10 / 3600})) = 1")
        rs = requests.get(SIMBAD_TAP, params=dict(request="doQuery", lang="adql", format="json", query=q), timeout=60).json()
        rows = rs.get("data", [])
        rec["simbad"] = [dict(main_id=r[1], sp_type=r[2], sp_bibcode=r[3]) for r in rows]
        refs = []
        for r in rows:
            q2 = ("SELECT r.bibcode, r.title FROM has_ref AS h JOIN ref AS r ON h.oidbibref = r.oidbib "
                  f"WHERE h.oidref = {r[0]} AND r.\"year\" >= 2021")
            rr = requests.get(SIMBAD_TAP, params=dict(request="doQuery", lang="adql", format="json", query=q2), timeout=60).json()
            refs += [dict(bibcode=x[0], title=x[1]) for x in rr.get("data", [])]
        rec["simbad_refs_2021on"] = refs
        tail = row["FName"].split()[1]
        short = tail[:5] + tail[tail.index("+") if "+" in tail else tail.index("-"):][:5]   # e.g. J0805+5153
        aq = f'abs:"{short}" OR ti:"{short}"'
        ax = requests.get("https://export.arxiv.org/api/query", params=dict(search_query=aq, max_results=20), timeout=60).text
        import re
        rec["arxiv_query"] = aq
        rec["arxiv_hits"] = [dict(id=i, title=" ".join(t.split())) for i, t in
                             zip(re.findall(r"<id>http://arxiv.org/abs/([^<]+)</id>", ax), re.findall(r"<title>([^<]+)</title>", ax)[1:])]
        out.append(rec)
        print(rec["name"], rec["k24_found"], [(s["main_id"], s["sp_type"], s["sp_bibcode"]) for s in rec["simbad"]],
              len(refs), len(rec["arxiv_hits"]), flush=True)
        time.sleep(1)
    json.dump(out, open(f"{P}/lit/lit_check.json", "w"), indent=1)


# Prior-spectrum decisions from data/phase2/e15/lit/lit_check.json (abstracts read via the SIMBAD ref table)
DROP = {"WISE 0855-0714": "JWST/NIRSpec spectrum (Luhman et al. 2024, AJ 167, 5; 2024AJ....167....5L) and many others",
        "CWISE 1055+5443": "Keck/NIRES spectrum, Y0 pec (2023ApJ...958...94R)",
        "CWISE 0411+4714": "near-IR spectrum, d/sdT7.5 (Burgasser et al. 2025, ApJ 982, 79)",
        "CWISE 1138+7212": "near-IR spectrum, d/sdT7 (Burgasser et al. 2025, ApJ 982, 79)",
        "CWISE 1308-0321": "near-IR spectrum, d/sdT8 (Burgasser et al. 2025, ApJ 982, 79)"}


PRIOR = f"{P}/lit/prior"   # SPIFF (Gagne et al. 2026, zenodo 19051216) tables, SUDA (Tu et al. 2026, zenodo 20659257), paper texts


def prior_spherex(row):
    """Position match (< 5 arcsec at epochs 2000.0, 2015.4 and 2025.6) to the SPIFF autotype table (all four datasets)
    and to the SUDA catalogue; grep of the designation in the Rustamkulov, Kiman and Tu paper texts."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    sp = pd.read_csv(f"{PRIOR}/spiff/spiff_tables/spherex_autotype_datasets_fulltable.csv")
    su = pd.read_csv(f"{PRIOR}/suda/catalog/suda_catalog.csv")
    spc = SkyCoord(sp.ra_j2000, sp.dec_j2000, unit=(u.hourangle, u.deg))
    suc = SkyCoord(su.RA.values, su.Dec.values, unit="deg")
    pos = [SkyCoord(*position(row, m), unit="deg") for m in (51544.5, EP_C2, 60900.0)]
    d1 = np.min([spc.separation(q).arcsec for q in pos], axis=0)
    d2 = np.min([suc.separation(q).arcsec for q in pos], axis=0)
    k1, k2 = int(np.argmin(d1)), int(np.argmin(d2))
    out = dict(spiff=None, suda=None, paper_text_hits=[])
    if d1[k1] < 5:
        r = sp.iloc[k1]
        out["spiff"] = dict(designation=r.designation, dataset=r.dataset, spiff_autotype=r.spiff_autotype,
                            literature_spectral_type=None if pd.isna(r.literature_spectral_type) else r.literature_spectral_type,
                            sep_arcsec=round(float(d1[k1]), 2))
    if d2[k2] < 5:
        r = su.iloc[k2]
        out["suda"] = dict(name=r.Name, SpT_lit=None if pd.isna(r.SpT_lit) else r.SpT_lit, SpT_spherex=None if pd.isna(r.SpT) else r.SpT,
                           sep_arcsec=round(float(d2[k2]), 2))
    tail = row["FName"].split()[1]
    for f in ("2607.00543", "2608.08908", "2604.26771", "spiff"):
        t = open(f"{PRIOR}/{f}.txt").read()
        if tail[:7] in t or (tail[:5] + tail[10:15]) in t:
            out["paper_text_hits"].append(f)
    return out


def stage_cands_final():
    c = pd.read_csv(f"{P}/cands_k21.csv")
    lit = {r["name"]: r for r in json.load(open(f"{P}/lit/lit_check.json"))}
    idx = pd.read_parquet(INDEX, columns=["key", "obs_id", "detector", "mjd", "ra", "dec", "c_ra", "c_dec", "dqa"])
    keep, summ = [], []
    for _, row in c.iterrows():
        o = row["obj"]
        fd = f"{P}/forced/{o}"
        os.makedirs(fd, exist_ok=True)
        cp = f"{fd}/_frames.parquet"
        if not os.path.exists(cp):
            frames_for(row, idx)[["key", "obs_id", "detector", "mjd", "pred_ra", "pred_dec"]].to_parquet(cp)
        n456 = int(pd.read_parquet(cp).detector.isin([4, 5, 6]).sum())
        L = lit[row["Name"]]
        rec = dict(name=row["Name"], designation=row["FName"], k21_photometric_type=spt_str(row["SpAd"]) if row["SpAd"] > -9 else "none",
                   W2=float(row["w2"]), W1_minus_W2=round(float(row["w1"] - row["w2"]), 3), n_listed_pictures_d456=n456,
                   k24_listed=L["k24_found"], k24_spectral_type=(L["k24_SpTOpt"] or L["k24_SpTNIR"] or None),
                   simbad=L["simbad"], simbad_refs_2021on=[x["bibcode"] for x in L["simbad_refs_2021on"]],
                   arxiv_query=L["arxiv_query"], arxiv_hits=[x["id"] for x in L["arxiv_hits"]],
                   prior_spherex=prior_spherex(row))
        ps = rec["prior_spherex"]
        why = DROP.get(row["Name"])
        if not why and ps["spiff"] and ps["spiff"]["dataset"] == "Known":
            why = f"spectroscopic literature type {ps['spiff']['literature_spectral_type']} (MOCAdb, via SPIFF 'Known'); SPHEREx spectrum already in SPIFF"
        if not why and (ps["spiff"] or ps["suda"] or ps["paper_text_hits"]):
            why = "SPHEREx spectrum already published (" + ", ".join(
                ([f"SPIFF {ps['spiff']['dataset']}, autotype {ps['spiff']['spiff_autotype']}"] if ps["spiff"] else []) +
                ([f"SUDA, SpT {ps['suda']['SpT_spherex']}"] if ps["suda"] else []) + ps["paper_text_hits"]) + ")"
        rec["dropped"] = why or (f"coverage {n456} < 20 pictures on detectors 4-6" if n456 < 20 else None)
        summ.append(rec)
        if rec["dropped"] is None:
            keep.append(row)
        print(row["Name"], n456, rec["dropped"], flush=True)
    pd.DataFrame(keep).to_csv(f"{P}/cands_final.csv", index=False)
    json.dump(summ, open(f"{P}/cands_lit_summary.json", "w"), indent=1)
    print(len(keep), "candidates kept")


# ---------------------------------------------------------------- forced photometry
def position(row, mjd):
    dt = (np.asarray(mjd, float) - EP_C2) / 365.25
    dec = row["DECdeg"] + row["pmdec_mas"] * dt / 3.6e6
    ra = row["RACdeg"] + row["pmra_mas"] * dt / 3.6e6 / np.cos(np.radians(row["DECdeg"]))
    return ra, dec


def frames_for(row, idx):
    from run_e4 import in_quad
    r0, d0 = position(row, 60900.0)
    cosd = np.sin(np.radians(d0)) * np.sin(np.radians(idx.dec.values)) + np.cos(np.radians(d0)) * \
        np.cos(np.radians(idx.dec.values)) * np.cos(np.radians(idx.ra.values - r0))
    c = idx[np.degrees(np.arccos(np.clip(cosd, -1, 1))) < 2.6].copy()
    c["pred_ra"], c["pred_dec"] = position(row, c.mjd.values)
    c = c[[in_quad(r.c_ra, r.c_dec, r.pred_ra, r.pred_dec) for r in c.itertuples()]]
    return c[c.dqa == "Pass"]


def gaia_field(obj, ra, dec):
    path = f"{P}/gaia/{obj}.parquet"
    if os.path.exists(path):
        return path
    import pyvo
    from orrery.forced import GAIA_TAP
    os.makedirs(f"{P}/gaia", exist_ok=True)
    q = ("SELECT ra, dec, pmra, pmdec, phot_g_mean_mag FROM gaiadr3.gaia_source WHERE "
         f"1=CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',{ra},{dec},{GAIA_R})) AND phot_g_mean_mag<19.5")
    for k in range(4):
        try:
            pyvo.dal.TAPService(GAIA_TAP).run_sync(q).to_table().to_pandas().to_parquet(path)
            return path
        except Exception:
            time.sleep(3 * (k + 1))
    raise RuntimeError("Gaia query failed")


def _stars(gpath, ra_, dec_, mjd, radius_deg=(15 + 4) * 6.2 / 3600 * 1.42):
    """Gaia DR3 neighbours propagated to the picture epoch (as orrery.forced.gaia_stars), minus the target itself."""
    df = pd.read_parquet(gpath)
    dt = Time(mjd, format="mjd").jyear - 2016.0
    sd = df.dec.values + df.pmdec.fillna(0).values * dt / 3.6e6
    sr = df.ra.values + df.pmra.fillna(0).values * dt / 3.6e6 / np.cos(np.radians(df.dec.values))
    sep = np.hypot((sr - ra_) * np.cos(np.radians(dec_)), sd - dec_) * 3600
    k = (sep < radius_deg * 3600) & (sep > EXCL_ARCSEC)
    return sr[k], sd[k]


def measure(key, ra, dec, gpath, sdir, half=15, edge=18, save=True):
    """orrery.forced.measure_frame (same edge cut, star selection, mask and fit), with the stamp read by
    e2_cache.fast_stamp (6 keep-alive range requests, verified bit-identical to Frame.stamp apart from ZODI)."""
    from e2_cache import fast_stamp
    from astropy.io import fits
    from astropy.wcs import WCS
    from orrery.data import BAD_BITS, save_stamp
    from orrery.fit import fit_stamp, flux_mJy
    st = fast_stamp(key, ra, dec, half=half)
    if st is None:
        return None
    w = WCS(fits.Header.fromstring(st.wcs_header))
    x, y = w.all_world2pix([ra], [dec], 0)
    x, y = float(x[0]), float(y[0])
    if not (edge <= x <= 2039 - edge and edge <= y <= 2039 - edge) or st.image.shape != (2 * half + 1, 2 * half + 1):
        return None
    xs, ys = x - st.x0, y - st.y0
    sra, sdec = _stars(gpath, ra, dec, st.mjd)
    star_xy = []
    if len(sra):
        gx, gy = w.all_world2pix(sra, sdec, 0)
        H, W = st.image.shape
        star_xy = [(float(a - st.x0), float(b - st.y0)) for a, b in zip(gx, gy)
                   if -3 <= a - st.x0 <= W + 2 and -3 <= b - st.y0 <= H + 2]
    star_xy = np.array(star_xy).reshape(-1, 2)
    r0 = fit_stamp(st.image, st.variance, st.flags, st.epsf, xs, ys, star_xy, BAD_BITS, st.epsf_oversample)
    if save:
        os.makedirs(sdir, exist_ok=True)
        save_stamp(st, f"{sdir}/{st.obs_id}_D{st.detector}.npz")
    return dict(key=key, obs_id=st.obs_id, detector=st.detector, mjd=st.mjd, x=x, y=y,
                wl_um=float(st.cwave[half, half]), bw_um=float(st.cband[half, half]),
                flux_mJy=flux_mJy(r0.A, st.omega_arcsec2), flux_err_mJy=flux_mJy(r0.A_err, st.omega_arcsec2),
                snr=r0.A / r0.A_err if r0.A_err > 0 else np.nan, chi2=r0.chi2, dof=r0.dof,
                chi2_local=r0.chi2_local, dof_local=r0.dof_local, n_stars=len(star_xy),
                flags_src=int(np.bitwise_or.reduce(st.flags[half - 1:half + 2, half - 1:half + 2].ravel())),
                n_bad=int((~r0.mask).sum()), omega=st.omega_arcsec2)


def _job(args):
    key, ra, dec, out, gpath = args
    try:
        r = measure(key, ra, dec, gpath, os.path.dirname(out).replace("/forced/", "/stamps/"), save="/heldout/" not in out)
        r = dict(key=key, on_frame=False) if r is None else dict(r, on_frame=True)
    except Exception as e:
        return dict(key=key, error=repr(e))
    r.update(pred_ra=ra, pred_dec=dec)
    json.dump({k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()}, open(out, "w"))
    return r


def stage_forced(which, workers):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    lst = pd.read_csv(f"{P}/{dict(calib='calib', cands='cands_final', heldout='heldout')[which]}.csv")
    idx = pd.read_parquet(INDEX, columns=["key", "obs_id", "detector", "mjd", "ra", "dec", "c_ra", "c_dec", "dqa"])
    todo = []
    for _, row in lst.iterrows():
        o = row["obj"]
        fd = f"{P}/forced/{o}"
        os.makedirs(fd, exist_ok=True)
        cp = f"{fd}/_frames.parquet"
        if which == "heldout":              # held-out objects: fit JSONs under forced/heldout/, no stamps kept (disk budget)
            fd = f"{P}/forced/heldout/{o}"
            os.makedirs(fd, exist_ok=True)
        if os.path.exists(cp):
            c = pd.read_parquet(cp)
        else:
            c = frames_for(row, idx)[["key", "obs_id", "detector", "mjd", "pred_ra", "pred_dec"]]
            c.to_parquet(cp)
        r0, d0 = position(row, 60900.0)
        gp = gaia_field(o, r0, d0)
        t = [(r.key, float(r.pred_ra), float(r.pred_dec), f"{fd}/{r.obs_id}_D{r.detector}.json", gp) for r in c.itertuples()]
        todo += [x for x in t if not os.path.exists(x[3])]
        print(o, len(c), "frames", flush=True)
    print("to do", len(todo), time.strftime("%H:%M:%S"), flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:      # I/O bound: threads in one process
        futs = [ex.submit(_job, t) for t in todo]
        for i, f in enumerate(as_completed(futs)):
            r = f.result()
            if "error" in r or i % 200 == 0:
                print(i, f"{time.time() - t0:.0f}s", r.get("key", "")[-40:], r.get("error", ""), flush=True)
    print("done", time.strftime("%H:%M:%S"), flush=True)


# ---------------------------------------------------------------- analysis
def load_obj(o):
    rows = [json.load(open(f)) for f in glob.glob(f"{P}/forced/{o}/*.json") if not os.path.basename(f).startswith("_")]
    df = pd.DataFrame(rows)
    if not len(df) or "flux_mJy" not in df:
        return pd.DataFrame()
    df = df[df.on_frame & np.isfinite(df.flux_mJy) & np.isfinite(df.flux_err_mJy) & (df.flux_err_mJy > 0)]
    return df.reset_index(drop=True)


def bin_fluxes(df):
    """Frozen bin estimator (protocol (c)). Returns {bin: (F, err, n)} and the error scale s."""
    lab = pd.Series(index=df.index, dtype=object)
    for b, (lo, hi) in BINS.items():
        lab[(df.wl_um >= lo) & (df.wl_um < hi)] = b
    d = df.assign(bin=lab).dropna(subset=["bin"])
    keep = np.ones(len(d), bool)

    def solve(keep):
        F, z = {}, np.full(len(d), np.nan)
        for b, g in d[keep].groupby("bin"):
            w = 1 / g.flux_err_mJy.values ** 2
            F[b] = float(np.sum(w * g.flux_mJy.values) / w.sum())
        for i, (b, f, e) in enumerate(zip(d["bin"], d.flux_mJy, d.flux_err_mJy)):
            if b in F:
                z[i] = (f - F[b]) / e
        n_b = d[keep].groupby("bin").size()
        ok = keep & d["bin"].map(lambda b: n_b.get(b, 0) >= 2).values
        zz = z[ok]
        s = max(1.0, 1.4826 * float(np.median(np.abs(zz - np.median(zz))))) if len(zz) >= 3 else 1.0
        return F, z, s
    F, z, s = solve(keep)
    keep = ~(np.abs(z) > 5 * s)
    F, z, s = solve(keep)
    out = {}
    for b, g in d[keep].groupby("bin"):
        out[b] = (F[b], float(s / np.sqrt(np.sum(1 / g.flux_err_mJy.values ** 2))), int(len(g)))
    return out, s, int((~keep).sum())


def index_of(bf):
    if not all(k in bf for k in ("m33", "c30", "c40")):
        return np.nan, np.nan, "missing bin"
    n, en = bf["m33"][:2]
    d = 0.5 * (bf["c30"][0] + bf["c40"][0])
    ed = 0.5 * np.hypot(bf["c30"][1], bf["c40"][1])
    if d < 2 * ed:
        return np.nan, np.nan, "denominator < 2 sigma"
    return float(n / d), float(np.sqrt(en ** 2 / d ** 2 + n ** 2 * ed ** 2 / d ** 4)), ""


def colour_of(bf):
    if "r45" not in bf or "j" not in bf or bf["j"][0] <= 0:
        return np.nan, np.nan
    r, er = bf["r45"][:2]
    j, ej = bf["j"][:2]
    return float(r / j), float(np.hypot(er / j, r * ej / j ** 2))


def threshold(I, E, isT):
    """Protocol (d): midpoint of the lowest interval of cuts minimising misclassifications (training objects with E < 0.2)."""
    m = np.isfinite(I) & (E < 0.2)
    x, t = np.asarray(I)[m], np.asarray(isT)[m]
    xs = np.sort(x)
    cuts = np.concatenate([[xs[0] - 1], 0.5 * (xs[1:] + xs[:-1]), [xs[-1] + 1]])
    err = np.array([np.sum((x < c) != t) for c in cuts])
    best = np.flatnonzero(err == err.min())
    run = [best[0]]                       # lowest contiguous run of optimal cuts
    for k in best[1:]:
        if k == run[-1] + 1:
            run.append(k)
        else:
            break
    lo, hi = (xs[run[0] - 1] if run[0] > 0 else xs[0]), (xs[run[-1]] if run[-1] < len(xs) else xs[-1])
    return float(0.5 * (lo + hi)), int(err.min()), int(m.sum())


def decide(I, E, th):
    if not np.isfinite(I):
        return "undetermined"
    z = (th - I) / E
    return "T or later" if z >= NSIG else ("L-like" if z <= -NSIG else "undetermined")


def measure_list(lst, sub=""):
    res = []
    for _, row in lst.iterrows():
        df = load_obj(sub + row["obj"])
        fr = pd.read_parquet(f"{P}/forced/{row['obj']}/_frames.parquet")
        if not len(df):
            res.append(dict(obj=row["obj"], name=row["Name"], n_pictures=0))
            continue
        bf, s, nclip = bin_fluxes(df)
        I, E, why = index_of(bf)
        C, CE = colour_of(bf)
        res.append(dict(obj=row["obj"], name=row["Name"], designation=row["FName"], W2=float(row["w2"]), W1=float(row["w1"]),
                        n_pictures=int(len(df)), n_pictures_listed=int(len(fr)), n_pictures_d456=int(df.detector.isin([4, 5, 6]).sum()),
                        n_listed_d456=int(fr.detector.isin([4, 5, 6]).sum()),
                        error_scale=round(s, 3), n_clipped=nclip, bins={k: dict(F_mJy=round(v[0], 5), err_mJy=round(v[1], 5), n=v[2]) for k, v in bf.items()},
                        ch4_index=I, ch4_err=E, index_note=why, colour_r45_over_j=C, colour_err=CE))
    return res


def stage_analyze():
    cal = pd.read_csv(f"{P}/calib.csv")
    R = measure_list(cal)
    for r, (_, row) in zip(R, cal.iterrows()):
        r.update(spt_code=float(row["spt"]), spt=spt_str(row["spt"]), cls=row["cls"], is_T=bool(row["spt"] >= 10))
    I = np.array([r.get("ch4_index", np.nan) for r in R], float)
    E = np.array([r.get("ch4_err", np.nan) for r in R], float)
    T = np.array([r["is_T"] for r in R])
    th, nerr, ntrain = threshold(I, E, T)
    for k, r in enumerate(R):
        m = np.arange(len(R)) != k
        th_k, _, _ = threshold(I[m], E[m], T[m])
        r["loo_threshold"] = th_k
        r["loo_decision"] = decide(I[k], E[k], th_k)
        r["full_decision"] = decide(I[k], E[k], th)
        r["loo_correct"] = None if r["loo_decision"] == "undetermined" else (r["loo_decision"] == "T or later") == r["is_T"]
    dec = [r for r in R if r["loo_correct"] is not None]
    acc = sum(r["loo_correct"] for r in dec) / max(len(dec), 1)
    verdict = "FAILED" if (len(dec) and acc < ACC_MIN) else ("INCONCLUSIVE" if len(dec) < 10 else "PASSED")
    out = dict(experiment="E15", protocol="docs/protocols/E15.md", threshold=th, threshold_train_errors=nerr, threshold_n_train=ntrain,
               n_calibration=len(R), n_decisive_loo=len(dec), loo_accuracy=acc, loo_completeness=len(dec) / len(R),
               calibration_verdict=verdict, calibration=R)
    print(f"threshold {th:.3f} (train errors {nerr}/{ntrain}); LOO decisive {len(dec)}/{len(R)}, accuracy {acc:.3f} -> {verdict}")
    for r in R:
        print(f"{r['name']:18s} {r['spt']:6s} I={r.get('ch4_index', np.nan):7.3f}±{r.get('ch4_err', np.nan):6.3f} "
              f"C={r.get('colour_r45_over_j', np.nan):7.2f}±{r.get('colour_err', np.nan):6.2f} s={r.get('error_scale', 0):.2f} "
              f"n={r['n_pictures']} {r['loo_decision']} {r.get('index_note', '')}")
    if os.path.exists(f"{P}/cands_final.csv"):
        cl = pd.read_csv(f"{P}/cands_final.csv")
        C = measure_list(cl)
        for c, (_, row) in zip(C, cl.iterrows()):
            c["k21_photometric_type"] = spt_str(row["SpAd"]) if row["SpAd"] > -9 else "none"
            c["decision"] = decide(c.get("ch4_index", np.nan), c.get("ch4_err", np.nan), th) if verdict != "FAILED" else "not classified (calibration FAILED)"
            c["z_from_threshold"] = (th - c["ch4_index"]) / c["ch4_err"] if np.isfinite(c.get("ch4_index", np.nan)) else None
            print(f"{c['name']:18s} {c['k21_photometric_type']:6s} I={c.get('ch4_index', np.nan):7.3f}±{c.get('ch4_err', np.nan):6.3f} "
                  f"n={c['n_pictures']} {c['decision']} {c.get('index_note', '')}")
        out["candidates"] = C
        out["candidate_lit_check"] = json.load(open(f"{P}/cands_lit_summary.json"))
    if os.path.isdir(f"{P}/forced/heldout"):
        hl = pd.read_csv(f"{P}/heldout.csv")
        H = measure_list(hl, "heldout/")
        for h, (_, row) in zip(H, hl.iterrows()):
            h["drop_reason"] = row["drop_reason"]
            h["decision"] = decide(h.get("ch4_index", np.nan), h.get("ch4_err", np.nan), th)
            print(f"HELD {h['name']:18s} I={h.get('ch4_index', np.nan):7.3f}±{h.get('ch4_err', np.nan):6.3f} n={h['n_pictures']} {h['decision']}")
        dd = [h for h in H if h["decision"] != "undetermined"]
        out["heldout"] = dict(note="objects dropped from the candidate list because a spectroscopic or SPHEREx type exists (all T or later); "
                                   "measured after the candidates with the frozen pipeline; secondary recall check",
                              n=len(H), n_decisive=len(dd), n_T_or_later=sum(h["decision"] == "T or later" for h in H), objects=H)
    json.dump(out, open(f"{P}/analysis.json", "w"), indent=1, default=float)
    return out


# ---------------------------------------------------------------- figures
COLS = {"L0-L4.5": ("#2a78d6", "o"), "L5-L9.5": ("#1baf7a", "s"), "T0-T4.5": ("#eda100", "^"),
        "T5-T9.5": ("#eb6834", "D"), "Y": ("#e87ba4", "v")}


def fig_methane(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 15})
    fig, ax = plt.subplots(figsize=(11, 7.5), facecolor="white")
    th = out["threshold"]
    ax.axvline(th, color="#555", lw=2, ls="--")
    ax.text(th + 0.02, 0.03, f"dividing line {th:.2f}", transform=ax.get_xaxis_transform(), va="bottom", fontsize=13, color="#333")
    n_off = 0
    for cls, (c, m) in COLS.items():
        rr = [r for r in out["calibration"] if r["cls"] == cls and np.isfinite(r.get("ch4_index", np.nan)) and np.isfinite(r.get("colour_r45_over_j", np.nan))]
        n_off += sum(1 for r in out["calibration"] if r["cls"] == cls) - len(rr)
        if rr:
            ax.errorbar([r["ch4_index"] for r in rr], [r["colour_r45_over_j"] for r in rr], xerr=[r["ch4_err"] for r in rr],
                        yerr=[r["colour_err"] for r in rr], fmt=m, ms=10, color=c, mec="white", mew=1.2, elinewidth=1.5,
                        capsize=0, label=f"known {cls.replace('-', '–')} (published type)")
    cc = [r for r in out.get("candidates", []) if np.isfinite(r.get("ch4_index", np.nan)) and np.isfinite(r.get("colour_r45_over_j", np.nan))]
    if cc:
        ax.errorbar([r["ch4_index"] for r in cc], [r["colour_r45_over_j"] for r in cc], xerr=[r["ch4_err"] for r in cc],
                    yerr=[r["colour_err"] for r in cc], fmt="*", ms=16, color="black", elinewidth=1.2, capsize=0,
                    label="candidates (no published spectrum found)")
    ax.set_yscale("log")
    ax.set_xlim(-1.0, 2.0)
    ax.set_ylim(0.4, 600)
    ax.set_xlabel("Methane absorption at 3.3 µm: flux at 3.2–3.4 µm ÷ neighbouring flux\n(deeper absorption ←)")
    ax.set_ylabel("Flux at 4.4–5.0 µm ÷ flux at 1.0–1.3 µm\n(colder ↑)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=12, loc="upper right", framealpha=0.95)
    n_c_off = len(out.get("candidates", [])) - len(cc)
    ax.set_title("SPHEREx: L dwarfs vs methane-bearing T/Y dwarfs", fontsize=17)
    note = f"Error bars: 1σ from the picture-to-picture scatter. Not shown (index or J-band colour undefined: too faint): {n_off} known, {n_c_off} candidates."
    fig.text(0.01, 0.005, note, fontsize=10.5, color="#444")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig("site/data/figs/e15_methane.png", dpi=130, facecolor="white")


def fig_examples(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 14})
    picks = []
    for lab, lo, hi in (("L dwarf", 0, 10), ("T dwarf", 10, 20), ("Y dwarf", 20, 30)):
        rr = [r for r in out["calibration"] if lo <= r["spt_code"] < hi and r["n_pictures"] > 0]
        r = min(rr, key=lambda r: r["W2"])                         # brightest in W2 (rule fixed in the protocol)
        picks.append((f"{lab}: {r['name']} ({r['spt']}, published)", r))
    dec = [r for r in out.get("candidates", []) if r.get("decision") == "T or later"]
    if dec:
        r = max(dec, key=lambda r: r["z_from_threshold"])
        picks.append((f"candidate: {r['name']} (no published spectrum found)", r))
    fig, axs = plt.subplots(len(picks), 1, figsize=(11, 3.3 * len(picks)), facecolor="white", sharex=True)
    edges = np.arange(0.75, 5.0001, 0.1)
    for ax, (title, r) in zip(axs, picks):
        df = load_obj(r["obj"])
        s = r["error_scale"]
        ax.plot(df.wl_um, df.flux_mJy, ".", color="#bbb", ms=4, label="single pictures")
        x, y, e = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            g = df[(df.wl_um >= lo) & (df.wl_um < hi)]
            if len(g):
                w = 1 / g.flux_err_mJy.values ** 2
                x.append(0.5 * (lo + hi)); y.append(np.sum(w * g.flux_mJy) / w.sum()); e.append(s / np.sqrt(w.sum()))
        ax.errorbar(x, y, yerr=e, fmt="o", color="#2a78d6", ms=6, elinewidth=1.5, capsize=0, label="0.1 µm bins")
        for b in ("c30", "c40"):
            ax.axvspan(*BINS[b], color="#1baf7a", alpha=0.15, lw=0)
        ax.axvspan(*BINS["m33"], color="#eb6834", alpha=0.25, lw=0)
        ax.axhline(0, color="#888", lw=0.8)
        top = np.nanmax(y) if len(y) else 1
        ax.set_ylim(-0.25 * top, 1.35 * top)
        I, E = r.get("ch4_index", np.nan), r.get("ch4_err", np.nan)
        ax.set_title(f"{title}   methane index {I:.2f} ± {E:.2f}", fontsize=13.5, loc="left")
        ax.set_ylabel("flux (mJy)")
        ax.grid(alpha=0.25)
    axs[0].legend(fontsize=11, loc="upper left")
    axs[0].text(3.3, 0.97, "methane\n3.3 µm", transform=axs[0].get_xaxis_transform(), ha="center", va="top", fontsize=11, color="#b04010")
    axs[-1].set_xlabel("wavelength (µm)   — orange band: methane window; green bands: neighbouring reference windows")
    fig.tight_layout()
    fig.savefig("site/data/figs/e15_examples.png", dpi=120, facecolor="white")


NULL_OFFS = [(6 * np.cos(np.radians(30 * k)), 6 * np.sin(np.radians(30 * k))) for k in range(12)]   # E4-style empty positions (px)


def _null_obj(args):
    """Same fit as measure() on the cached stamps, at 12 empty positions 6 px from the target (target added as a
    fitted neighbour). Returns per-picture rows (wl, offset k, flux, err)."""
    o, gpath = args
    from astropy.io import fits
    from astropy.wcs import WCS
    from orrery.data import BAD_BITS, load_stamp
    from orrery.fit import fit_stamp, flux_mJy
    rows = []
    for f in glob.glob(f"{P}/forced/{o}/*.json"):
        if os.path.basename(f).startswith("_"):
            continue
        r = json.load(open(f))
        if not r.get("on_frame"):
            continue
        wl = r["wl_um"]
        if not any(lo <= wl < hi for lo, hi in BINS.values()):
            continue
        st = load_stamp(f"{P}/stamps/{o}/{r['obs_id']}_D{r['detector']}.npz")
        w = WCS(fits.Header.fromstring(st.wcs_header))
        xs, ys = r["x"] - st.x0, r["y"] - st.y0
        sra, sdec = _stars(gpath, r["pred_ra"], r["pred_dec"], st.mjd)
        sxy = []
        if len(sra):
            gx, gy = w.all_world2pix(sra, sdec, 0)
            sxy = [(a - st.x0, b - st.y0) for a, b in zip(gx, gy) if -3 <= a - st.x0 <= 33 and -3 <= b - st.y0 <= 33]
        for k, (dx, dy) in enumerate(NULL_OFFS):
            sx = np.array(sxy + [(xs, ys)], float).reshape(-1, 2)
            q = fit_stamp(st.image, st.variance, st.flags, st.epsf, xs + dx, ys + dy, sx, BAD_BITS, st.epsf_oversample)
            rows.append(dict(obj=o, k=k, wl_um=wl, flux_mJy=flux_mJy(q.A, st.omega_arcsec2), flux_err_mJy=flux_mJy(q.A_err, st.omega_arcsec2)))
    return rows


def stage_nulls():
    """Diagnostic added after the candidate results were seen (logged as a deviation): empty-position bin fluxes."""
    from concurrent.futures import ProcessPoolExecutor
    objs = list(pd.read_csv(f"{P}/cands_final.csv").obj) + list(pd.read_csv(f"{P}/calib.csv").obj)
    jobs = [(o, f"{P}/gaia/{o}.parquet") for o in objs]
    with ProcessPoolExecutor(5) as ex:
        rows = [x for rr in ex.map(_null_obj, jobs) for x in rr]
    df = pd.DataFrame(rows)
    df.to_parquet(f"{P}/nulls.parquet")
    out = []
    for o, g in df.groupby("obj"):
        for k, gk in g.groupby("k"):
            gk = gk[np.isfinite(gk.flux_mJy) & (gk.flux_err_mJy > 0)]
            bf, sc, _ = bin_fluxes(gk)
            out.append(dict(obj=o, k=k, **{f"{b}_F": v[0] for b, v in bf.items()}, **{f"{b}_E": v[1] for b, v in bf.items()}))
    pd.DataFrame(out).to_csv(f"{P}/nulls_bins.csv", index=False)
    print(pd.DataFrame(out).groupby("obj")[["m33_F", "m33_E", "c30_F", "c40_F"]].agg(["mean", "std"]).round(4).to_string())


def stage_nullcheck():
    """Post-hoc sensitivity (deviation, logged): bin errors replaced by max(quoted, std over the 12 empty positions of the
    same object and bin); index error re-propagated; same frozen threshold and 3-sigma rule."""
    o = json.load(open(f"{P}/analysis.json"))
    nb = pd.read_csv(f"{P}/nulls_bins.csv")
    th = o["threshold"]
    def redo(r):
        g = nb[nb.obj == r["obj"]]
        bf = {}
        for k, v in r["bins"].items():
            sd = float(g[f"{k}_F"].std()) if f"{k}_F" in g and g[f"{k}_F"].notna().sum() >= 6 else np.nan
            bf[k] = (v["F_mJy"], max(v["err_mJy"], sd) if np.isfinite(sd) else v["err_mJy"], v["n"], sd)
        I, E, why = index_of(bf)
        return dict(ch4_index=I, ch4_err_null=E, decision_null=decide(I, E, th),
                    null_std_mJy={k: v[3] for k, v in bf.items()}, error_ratio_m33=bf["m33"][3] / r["bins"]["m33"]["err_mJy"] if "m33" in bf else None)
    res = dict(rule="bin error = max(quoted scatter-based error, std of the bin flux over 12 empty positions 6 px from the object, same pictures)",
               added="2026-11-15 13:21 PST, after the candidate classifications were seen (deviation)", threshold=th)
    for key in ("calibration", "candidates", "heldout"):
        lst = o.get(key, [])
        if key == "heldout":
            lst = (lst or {}).get("objects", [])
        res[key] = []
        for r in lst:
            if "bins" not in r or not len(nb[nb.obj == r["obj"]]):
                continue
            x = redo(r)
            x.update(name=r["name"], obj=r["obj"], decision_frozen=r.get("decision", r.get("full_decision")))
            if key == "calibration":
                x["is_T"] = r["is_T"]
            res[key].append(x)
            print(f"{key[:5]} {r['name']:17s} I={x['ch4_index']:.3f} err {r['ch4_err']:.3f} -> {x['ch4_err_null']:.3f}  {x['decision_frozen']} -> {x['decision_null']}  m33 err ratio {x['error_ratio_m33']}")
    cal = [x for x in res["calibration"] if x["decision_null"] != "undetermined"]
    res["calibration_null_errors"] = dict(n_decisive=len(cal), n_correct=sum((x["decision_null"] == "T or later") == x["is_T"] for x in cal),
                                          note="full-set threshold, not leave-one-out")
    rat = [x["error_ratio_m33"] for x in res["calibration"] + res["candidates"] if x["error_ratio_m33"]]
    res["median_m33_error_ratio"] = float(np.median(rat))
    print(res["calibration_null_errors"], "median m33 null/quoted error ratio", res["median_m33_error_ratio"])
    json.dump(res, open(f"{P}/nullcheck.json", "w"), indent=1, default=float)


def stage_findings():
    o = json.load(open(f"{P}/analysis.json"))
    C = o.get("candidates", [])
    n_t = sum(c["decision"] == "T or later" for c in C)
    n_l = sum(c["decision"] == "L-like" for c in C)
    n_u = sum(c["decision"] == "undetermined" for c in C)
    f = dict(experiment="E15", question="Can SPHEREx's 0.75-5 um colours tell methane-bearing (T/Y) brown dwarfs from L dwarfs, "
                                        "and what do they say about candidates with no published spectrum?",
             protocol="docs/protocols/E15.md", script="scripts/run_e15.py", generated=time.strftime("%Y-%m-%d %H:%M %Z"),
             method="Per-picture forced ePSF fit (Gaia DR3 neighbours fitted, hard-bad/NONLINEAR masked) at the CatWISE2020 position moved "
                    "with the Kirkpatrick et al. 2021 proper motion; inverse-variance bin means with errors rescaled by the per-object "
                    "picture-to-picture scatter; methane index CH4 = F(3.2-3.4)/mean(F(2.9-3.1), F(3.9-4.1)); colour F(4.4-5.0)/F(1.0-1.3); "
                    "rule 'T or later' if CH4 is >= 3 sigma below the threshold, 'L-like' if >= 3 sigma above, else undetermined.",
             prior_work=["Gagne, Faherty et al. 2026, ApJ, SPIFF (arXiv:2604.22012; Zenodo 10.5281/zenodo.19051216): per-image PSF-fit SPHEREx "
                         "QR2 spectra and automated types for 6003 L0-Y1 dwarfs",
                         "Tu et al. 2026, ApJ, SUDA (arXiv:2604.26771; Zenodo 20659257): 1675 field ultracool dwarfs",
                         "Kiman et al. 2026 (arXiv:2608.08908): young L/T dwarfs",
                         "Rustamkulov et al. 2026, ApJ (arXiv:2607.00543): 37 L0-Y4 dwarfs incl. WISE 0855"],
             contribution="Not SPHEREx brown-dwarf spectra as such (published above). Added here: per-picture forced fit at the "
                          "motion-propagated position with an empirical error, and a frozen methane rule calibrated on 25 objects "
                          "with leave-one-out accuracy, applied to candidates absent from SPIFF/SUDA.",
             calibration=dict(n=o["n_calibration"], threshold=o["threshold"], loo_accuracy=o["loo_accuracy"], n_decisive=o["n_decisive_loo"],
                              loo_completeness=o["loo_completeness"], verdict=o["calibration_verdict"], objects=o["calibration"]),
             candidates=dict(n=len(C), n_T_or_later=n_t, n_L_like=n_l, n_undetermined=n_u, objects=C,
                             selection_and_prior_spectrum_check=o.get("candidate_lit_check")),
             heldout=o.get("heldout"),
             empty_position_check=json.load(open(f"{P}/nullcheck.json")) if os.path.exists(f"{P}/nullcheck.json") else None,
             wording="A candidate meeting the rule: 'SPHEREx colours indicate a methane-bearing (T-type or later) brown dwarf; no published "
                     "spectrum found in the sources we checked.' Not a discovery.",
             figures=["site/data/figs/e15_methane.png", "site/data/figs/e15_examples.png"])
    json.dump(f, open("data/findings/e15.json", "w"), indent=1, default=float)
    print("candidates:", n_t, "T or later,", n_l, "L-like,", n_u, "undetermined")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("which", nargs="?")
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()
    if a.stage == "calib":
        stage_calib()
    elif a.stage == "cands":
        stage_cands()
    elif a.stage == "cands_final":
        stage_cands_final()
    elif a.stage == "lit":
        stage_lit()
    elif a.stage == "forced":
        stage_forced(a.which, a.workers)
    elif a.stage == "analyze":
        stage_analyze()
    elif a.stage == "nulls":
        stage_nulls()
    elif a.stage == "nullcheck":
        stage_nullcheck()
    elif a.stage == "findings":
        stage_findings()
    elif a.stage == "figs":
        o = json.load(open(f"{P}/analysis.json"))
        fig_methane(o)
        fig_examples(o)
