"""E22 (docs/protocols/E22.md): census of ultracool-dwarf candidates with no published spectroscopic type, classified with
E15's frozen 3.3 um methane rule (threshold 0.356, 3 sigma, empty-position errors).

Stages (resumable; partial results under data/phase2/e22/):
  sample          assemble sample + control lists from the cached catalogues (no SPHEREx pixels read)
  calcheck        E15 calibration objects re-analysed with E22's picture selection (calibration data only)
  forced LIST     per-picture forced fits (E15 measure()) for LIST in sample|control|spiff, index-bin pictures only
  nulls LIST      12 empty-position fits per picture (E15 _null_obj logic)
  analyze         bins, index, frozen rule, control, expected false calls -> data/phase2/e22/analysis.json
  lit             survivors: SIMBAD refs (2021 on) + arXiv title/abstract search
  findings        data/findings/e22.json + site/data/figs/e22_methane_census.png
Usage: OMP_NUM_THREADS=1 .venv/bin/python scripts/run_e22.py STAGE [LIST] [--workers 6]
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
import run_e15 as E

P = "data/phase2/e22"
E.P = P                                   # E15 helpers (gaia_field, measure paths) write under data/phase2/e22
CAT = f"{P}/cat"
INDEX = "data/index/frames_dr1.parquet"
THETA = 0.356                             # E15 frozen threshold (docs/protocols/E15.md (f)); not re-tuned
NSIG = 3.0
IDX_BINS = {k: E.BINS[k] for k in ("c30", "m33", "c40")}
MARGIN = 0.03                             # um, pre-selection margin on the predicted wavelength
CAP = 300
PIC_CAP = 150                             # max index-bin pictures fitted per object (deviation, 18:50)
SIMBAD_TAP = "https://simbad.harvard.edu/simbad/sim-tap/sync"
SRC_BIB = {"2021ApJS..253....7K", "2020ApJ...889...74M", "2020ApJ...899..123M", "2024ApJS..271...55K"}  # input catalogues


def read_viz(path):
    """Split a VizieR asu-tsv multi-table file into DataFrames (strings, stripped)."""
    import io
    out, name, lines, state, hdr, skip = {}, None, [], 0, None, 0
    def flush():
        out[name] = pd.read_csv(io.StringIO("\n".join([hdr] + lines)), sep="\t", dtype=str).apply(lambda c: c.str.strip())
    for ln in open(path):
        ln = ln.rstrip("\n")
        if ln.startswith("#Table"):
            name, state = ln.split("\t")[1].rstrip(":"), 1
            continue
        if ln.startswith("#"):
            continue
        if state == 1 and ln.strip():
            hdr, state, skip, lines = ln, 2, 2, []
            continue
        if state == 2:
            if skip:
                skip -= 1
                continue
            if not ln.strip():
                flush()
                state = 0
                continue
            lines.append(ln)
    if state == 2:
        flush()
    return out


def num(s):
    return pd.to_numeric(s, errors="coerce")


def to_ep(ra, dec, pmra, pmdec, mjd_from, mjd_to):
    dt = (np.asarray(mjd_to, float) - np.asarray(mjd_from, float)) / 365.25
    d = dec + pmdec * dt / 3.6e6
    r = ra + pmra * dt / 3.6e6 / np.cos(np.radians(dec))
    return r, d


def std_rows(name, desig, ra, dec, mjd, pmra, pmdec, w1, w2, src, phot):
    r57, d57 = to_ep(ra, dec, pmra, pmdec, mjd, E.EP_C2)
    return pd.DataFrame(dict(Name=name, FName=desig, RACdeg=r57, DECdeg=d57, pmra_mas=pmra, pmdec_mas=pmdec,
                             w1=w1, w2=w2, source=src, phot_type=phot))


def load_sources():
    """Candidate rows from the four input catalogues, positions moved to MJD 57170 (E15 position epoch)."""
    k21 = E.load_k21()
    a = k21[k21.SpTO.isna() & k21.SpTIR.isna() & k21.pmra_mas.notna()]
    s1 = std_rows(a.Name.values, a.FName.values, a.RACdeg.values, a.DECdeg.values, E.EP_C2, a.pmra_mas.values, a.pmdec_mas.values,
                  a.w1.values, a.w2.values, "K21", [E.spt_str(x) if x > -9 else "none" for x in a.SpAd.values])
    m = read_viz(f"{CAT}/J_ApJ_889_74.tsv")
    b = m["J_ApJ_889_74_bdcand"]
    s2 = std_rows(b.Name.values, b.Name.values, num(b.RA_ICRS).values, num(b.DE_ICRS).values, num(b.MJD).values,
                  num(b.pmRA).fillna(0).values, num(b.pmDE).fillna(0).values, num(b.W1mag).values, num(b.W2mag).values, "M20a", "none")
    m2 = read_viz(f"{CAT}/J_ApJ_899_123.tsv")
    t6, t2 = m2["J_ApJ_899_123_table6"], m2["J_ApJ_899_123_table2"]
    spec_m20b = set(m2["J_ApJ_899_123_table3"].Source)          # objects with spectra obtained in that paper
    c = t6.merge(t2[["Source", "W1mag", "W2mag"]], on="Source", how="left")
    c = c[~c.Source.isin(spec_m20b)]
    s3 = std_rows(c.Source.values, c.Source.values, num(c.RA_ICRS).values, num(c.DE_ICRS).values, num(c.MJD).values,
                  num(c.pmRA).values, num(c.pmDE).values, num(c.W1mag).values, num(c.W2mag).values, "M20b", "none")
    k24 = pd.read_csv("data/phase2/e15/k24_table4.tsv", sep="\t", comment="#", dtype=str).iloc[2:].apply(lambda x: x.str.strip())
    k = k24[num(k24.Teff).notna() & (k24.SpTOpt.fillna("") == "") & (k24.SpTNIR.fillna("") == "") & num(k24.RAPdeg).notna() & num(k24.pmRA).notna()]
    # K24 positions are at mixed epochs (up to tens of arcsec off for fast movers), so only K24 rows with a CatWISE2020
    # designation (CWISE/CWISER Jhhmmss.ss+ddmmss.s = position at MJD 57170) are used, positioned from the designation
    k = k[k.Name.str.lstrip("-").str.match(r"CWISER? J\d{6}\.\d{2}[+-]\d{6}\.\d")]
    dz = k.Name.str.lstrip("-").str.extract(r"J(\d\d)(\d\d)(\d\d\.\d\d)([+-])(\d\d)(\d\d)(\d\d\.\d)").astype(str)
    kra = 15 * (dz[0].astype(float) + dz[1].astype(float) / 60 + dz[2].astype(float) / 3600)
    kde = np.where(dz[3] == "-", -1, 1) * (dz[4].astype(float) + dz[5].astype(float) / 60 + dz[6].astype(float) / 3600)
    s4 = std_rows(k.Name.values, k.Name.values, kra.values, np.asarray(kde, float), E.EP_C2, num(k.pmRA).values,
                  num(k.pmDE).values, np.nan, num(k.W2mag).values, "K24", "none")
    allr = pd.concat([s1, s2, s3, s4], ignore_index=True)
    sp = t6[t6.Source.isin(spec_m20b)]
    spec_pos = to_ep(num(sp.RA_ICRS).values, num(sp.DE_ICRS).values, num(sp.pmRA).values, num(sp.pmDE).values, num(sp.MJD).values, E.EP_C2)
    return allr, spec_pos


def dedupe(df, r_arcsec=10.0):
    from astropy.coordinates import SkyCoord
    c = SkyCoord(df.RACdeg.values, df.DECdeg.values, unit="deg")
    keep, also = [], {}
    for i in range(len(df)):
        if keep:
            sep = c[i].separation(c[keep]).arcsec
            j = int(np.argmin(sep))
            if sep[j] < r_arcsec:
                also.setdefault(keep[j], []).append(f"{df.source.iloc[i]}:{df.Name.iloc[i]}")
                continue
        keep.append(i)
    out = df.iloc[keep].copy()
    out["also_in"] = ["; ".join(also.get(i, [])) for i in keep]
    return out.reset_index(drop=True)


def simbad_cone(ra, dec, r=10.0):
    import requests
    q = ("SELECT b.oid, b.main_id, b.sp_type, b.sp_bibcode, b.otype FROM basic AS b WHERE "
         f"CONTAINS(POINT('ICRS', b.ra, b.dec), CIRCLE('ICRS', {ra}, {dec}, {r / 3600})) = 1")
    for k in range(4):
        try:
            js = requests.get(SIMBAD_TAP, params=dict(request="doQuery", lang="adql", format="json", query=q), timeout=60).json()
            return [dict(oid=x[0], main_id=x[1], sp_type=x[2], sp_bibcode=x[3], otype=x[4]) for x in js.get("data", [])]
        except Exception:
            time.sleep(3 * (k + 1))
    return None


def prior_match(df):
    """SPIFF autotype table (all datasets) and SUDA: nearest match at epochs 2000.0, 2015.4, 2025.6 (as E15)."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    PR = "data/phase2/e15/lit/prior"
    sp = pd.read_csv(f"{PR}/spiff/spiff_tables/spherex_autotype_datasets_fulltable.csv")
    su = pd.read_csv(f"{PR}/suda/catalog/suda_catalog.csv")
    spc = SkyCoord(sp.ra_j2000, sp.dec_j2000, unit=(u.hourangle, u.deg))
    suc = SkyCoord(su.RA.values, su.Dec.values, unit="deg")
    res = []
    D1, I1, D2, I2 = [], [], [], []
    for mjd in (51544.5, E.EP_C2, 60900.0):
        ra, dec = E.position(df, mjd)
        q = SkyCoord(np.asarray(ra, float), np.asarray(dec, float), unit="deg")
        i1, d1, _ = q.match_to_catalog_sky(spc)
        i2, d2, _ = q.match_to_catalog_sky(suc)
        D1.append(d1.arcsec); I1.append(i1); D2.append(d2.arcsec); I2.append(i2)
    D1, I1, D2, I2 = map(np.array, (D1, I1, D2, I2))
    for j in range(len(df)):
        k1, k2 = int(np.argmin(D1[:, j])), int(np.argmin(D2[:, j]))
        best1, best2 = (float(D1[k1, j]), int(I1[k1, j])), (float(D2[k2, j]), int(I2[k2, j]))
        r = dict(spiff=None, suda=None)
        if best1[0] < 5:
            x = sp.iloc[best1[1]]
            r["spiff"] = dict(designation=x.designation, dataset=x.dataset, autotype=x.spiff_autotype,
                              lit_type=None if pd.isna(x.literature_spectral_type) else x.literature_spectral_type, sep=round(best1[0], 2))
        if best2[0] < 5:
            x = su.iloc[best2[1]]
            r["suda"] = dict(name=x.Name, SpT_lit=None if pd.isna(x.SpT_lit) else x.SpT_lit, SpT=None if pd.isna(x.SpT) else x.SpT, sep=round(best2[0], 2))
        res.append(r)
    return res


def k24_types(df):
    from astropy.coordinates import SkyCoord
    k24 = pd.read_csv("data/phase2/e15/k24_table4.tsv", sep="\t", comment="#", dtype=str).iloc[2:].apply(lambda x: x.str.strip())
    k24 = k24[num(k24.RAPdeg).notna()].fillna("")
    kc = SkyCoord(num(k24.RAPdeg).values, num(k24.DEPdeg).values, unit="deg")
    ra, dec = E.position(df, E.EP_C2)                 # K24 positions are at the CatWISE epoch (see load_sources)
    I, D, _ = SkyCoord(np.asarray(ra, float), np.asarray(dec, float), unit="deg").match_to_catalog_sky(kc)
    out = []
    for i, d in zip(I, D.arcsec):
        x = k24.iloc[int(i)]
        a, b = x.SpTOpt or "", x.SpTNIR or ""
        t = a + ("/" if a and b else "") + b
        out.append(t if d < 10 and t else None)
    return out


# ---------------------------------------------------------------- picture pre-selection (no pixels read)
def _tp(ra, dec, ra0, dec0):
    ra, dec, ra0, dec0 = map(np.radians, (ra, dec, ra0, dec0))
    c = np.sin(dec0) * np.sin(dec) + np.cos(dec0) * np.cos(dec) * np.cos(ra - ra0)
    return (np.degrees(np.cos(dec) * np.sin(ra - ra0) / c),
            np.degrees((np.cos(dec0) * np.sin(dec) - np.sin(dec0) * np.cos(dec) * np.cos(ra - ra0)) / c))


PIXC = np.array([[0, 0], [2039, 0], [2039, 2039], [0, 2039]], float)


def predict_wl(fr_rows, idx):
    """Predicted pixel (bilinear map from the index corner positions; checked against E15 fits: median 6 px, max 10 px)
    and centre wavelength from the per-detector WCS-WAVE grid in data/ref."""
    from orrery.data import _calib, release_of
    from scipy.interpolate import RegularGridInterpolator
    wl = []
    for r in fr_rows.itertuples():
        e = idx.loc[r.key]
        xi, eta = _tp(np.asarray(e.c_ra), np.asarray(e.c_dec), e.ra, e.dec)
        A = np.c_[xi, eta, np.ones(4), xi * eta]
        tx, ty = _tp(r.pred_ra, r.pred_dec, e.ra, e.dec)
        v = np.array([tx, ty, 1.0, tx * ty])
        x, y = v @ np.linalg.solve(A, PIXC[:, 0]), v @ np.linalg.solve(A, PIXC[:, 1])
        c = _calib(release_of(r.key), int(r.detector))
        f = RegularGridInterpolator((c["wy"], c["wx"]), c["wv"], bounds_error=False, fill_value=None)
        wl.append(float(f([[y, x]])[0, 0]))
    return np.array(wl)


def in_idx_bins(wl, margin=MARGIN):
    return np.any([(wl >= lo - margin) & (wl < hi + margin) for lo, hi in IDX_BINS.values()], axis=0)


def frames_sel(row, idx_full, idx_k):
    fr = E.frames_for(row, idx_full)
    fr = fr[fr.detector.isin([4, 5])].copy()
    fr["wl_pred"] = predict_wl(fr, idx_k) if len(fr) else []
    return fr


# ---------------------------------------------------------------- stage: sample
def stage_sample():
    allr, m20b_spec = load_sources()
    log_ = dict(n_rows_by_source=allr.source.value_counts().to_dict(), m20b_objects_with_spectra=len(m20b_spec[0]))
    allr = allr[np.isfinite(allr.RACdeg) & np.isfinite(allr.w2)]
    d = dedupe(allr)
    cal = set(pd.read_csv("data/phase2/e15/calib.csv").Name)
    d = d[~d.Name.isin(cal)]
    d["obj"] = d.Name.str.replace(" ", "_").str.replace("+", "p")
    log_["n_unique"] = len(d)
    d["k24_type"] = k24_types(d)
    pm = prior_match(d)
    d["spiff"] = [json.dumps(x["spiff"]) if x["spiff"] else "" for x in pm]
    d["suda"] = [json.dumps(x["suda"]) if x["suda"] else "" for x in pm]
    from concurrent.futures import ThreadPoolExecutor
    def both(r):                                      # SIMBAD cone at the epoch-2000 and the CatWISE-epoch position, merged
        a, b = simbad_cone(*E.position(r, 51544.5)), simbad_cone(float(r.RACdeg), float(r.DECdeg))
        if a is None or b is None:
            return None
        return list({x["oid"]: x for x in a + b}.values())
    with ThreadPoolExecutor(3) as ex:
        sim = list(ex.map(both, [r for _, r in d.iterrows()]))
    d["simbad"] = [json.dumps(s) for s in sim]

    from astropy.coordinates import SkyCoord
    _, dsp, _ = SkyCoord(d.RACdeg.values, d.DECdeg.values, unit="deg").match_to_catalog_sky(SkyCoord(*m20b_spec, unit="deg"))
    d["m20b_spectrum"] = dsp.arcsec < 5

    h = pd.read_csv("data/phase2/e15/heldout.csv")    # E15: dropped for a published spectroscopic / SPHEREx type (incl. WISE 0855)
    ih, dh, _ = SkyCoord(d.RACdeg.values, d.DECdeg.values, unit="deg").match_to_catalog_sky(SkyCoord(h.RACdeg.values, h.DECdeg.values, unit="deg"))
    d["e15_dropped"] = [h.Name.iloc[int(i)] + ": " + h.drop_reason.iloc[int(i)] if x < 10 else "" for i, x in zip(ih, dh.arcsec)]

    def why(r, s):
        if r.e15_dropped:
            return "E15 prior-spectrum check (" + r.e15_dropped + ")"
        if r.m20b_spectrum:
            return "M20b near-IR spectrum (Meisner et al. 2020b table 3)"
        if isinstance(r.k24_type, str) and r.k24_type:
            return f"K24 spectral type {r.k24_type}"
        if s is None:
            return "SIMBAD query failed"
        t = [x for x in s if (x["sp_type"] or "").strip() and (x["sp_bibcode"] or "") not in SRC_BIB]
        if t:
            return f"SIMBAD sp_type {t[0]['sp_type']} ({t[0]['sp_bibcode']}, {t[0]['main_id']})"
        if isinstance(r.spiff, str) and r.spiff:
            return "in SPIFF (" + json.loads(r.spiff)["dataset"] + ")"
        if isinstance(r.suda, str) and r.suda:
            return "in SUDA"
        return None
    d["excluded"] = [why(r, s) for (_, r), s in zip(d.iterrows(), sim)]
    log_["excluded_counts"] = d.excluded.fillna("kept").map(lambda s: s.split(" (")[0].split(" sp_type")[0].split(" spectral")[0]).value_counts().to_dict()
    d.to_csv(f"{P}/sample_all.csv", index=False)
    keep = d[d.excluded.isna()].sort_values("w2").reset_index(drop=True)
    # SPIFF/SUDA-only exclusions (no spectroscopic type anywhere else): secondary agreement group
    spf = d[d.excluded.fillna("").str.startswith("in SPIFF") | d.excluded.fillna("").str.startswith("in SUDA")].sort_values("w2")
    idx_full = pd.read_parquet(INDEX, columns=["key", "obs_id", "detector", "mjd", "ra", "dec", "c_ra", "c_dec", "dqa"])
    idx_k = idx_full.set_index("key")
    rows = []
    for _, row in keep.iterrows():
        if len(rows) >= CAP:
            break
        fr = E.frames_for(row, idx_full)
        n456 = int(fr.detector.isin([4, 5, 6]).sum())
        if n456 < 20:
            continue
        rows.append(dict(row, n_listed_d456=n456))
    s = pd.DataFrame(rows)
    s.to_csv(f"{P}/sample.csv", index=False)
    spf.to_csv(f"{P}/spiff_group.csv", index=False)
    log_.update(n_kept_before_coverage=len(keep), n_sample=len(s), n_spiff_suda_only=len(spf),
                w2_range=[float(s.w2.min()), float(s.w2.max())] if len(s) else None)
    # control: K21 table A1 spectroscopic L0-L8.5 (code 0-8.5), no note/no component, proper motion, not E15 calibration,
    # W2 >= 12.5 (closest in brightness to the sample), faintest first, up to 40 with >= 20 pictures on D4-6
    k21 = E.load_k21()
    c = k21[(k21.spt >= 0) & (k21.spt < 9) & (k21.N == "") & (k21.m_PName.fillna("") == "") & k21.pmra_mas.notna()
            & ~k21.Name.isin(cal) & (k21.w2 >= 12.5)].sort_values("w2", ascending=False)
    crow = []
    for _, row in c.iterrows():
        if len(crow) >= 40:
            break
        if int(E.frames_for(row, idx_full).detector.isin([4, 5, 6]).sum()) >= 20:
            crow.append(dict(row, source="K21 control", phot_type=E.spt_str(row["spt"])))
    # fainter control: SUDA (Tu et al. 2026) objects with literature types L0-L8.5, W1 from SUDA's synthetic W1 flux
    # 15.0-16.8 mag (W2 ~ 14.5-16.5 for L dwarfs, the sample's range), proper motion listed, not within 10" of a sample
    # object; permuted with seed 22, first 40 with >= 20 pictures on D4-6
    import re
    su = pd.read_csv("data/phase2/e15/lit/prior/suda/catalog/suda_catalog.csv")
    su["w1s"] = -2.5 * np.log10(num(su.SPHEREx_synth_W1_Fnu_mJy).astype(float) / 309540.0)
    for c_ in ("SPHEREx_RA_deg", "SPHEREx_Dec_deg", "pmRA_mas_yr", "pmDec_mas_yr"):
        su[c_] = num(su[c_]).astype(float)
    su = su[su.SpT_lit.fillna("").map(lambda t: bool(re.fullmatch(r"L[0-8](\.5)?", t))) & su.pmRA_mas_yr.notna()
            & su.w1s.between(15.0, 16.8)].sort_values("Name").reset_index(drop=True)
    su = su.iloc[np.random.default_rng(22).permutation(len(su))]
    r57, d57 = to_ep(su.SPHEREx_RA_deg.values, su.SPHEREx_Dec_deg.values, su.pmRA_mas_yr.values, su.pmDec_mas_yr.values, 60900.0, E.EP_C2)
    su = su.assign(RACdeg=r57, DECdeg=d57, pmra_mas=su.pmRA_mas_yr, pmdec_mas=su.pmDec_mas_yr)
    _, dsa, _ = SkyCoord(su.RACdeg.values, su.DECdeg.values, unit="deg").match_to_catalog_sky(SkyCoord(d.RACdeg.values, d.DECdeg.values, unit="deg"))
    su = su[dsa.arcsec > 10]
    n_s = 0
    for _, row in su.iterrows():
        if n_s >= 40:
            break
        if int(E.frames_for(row, idx_full).detector.isin([4, 5, 6]).sum()) >= 20:
            crow.append(dict(Name=row["Name"], FName=row["Name"], RACdeg=row.RACdeg, DECdeg=row.DECdeg, pmra_mas=row.pmra_mas,
                             pmdec_mas=row.pmdec_mas, w1=row.w1s, w2=np.nan, source="SUDA control", phot_type=row.SpT_lit))
            n_s += 1
    pd.DataFrame(crow).to_csv(f"{P}/control.csv", index=False)
    log_["n_control"] = len(crow)
    log_["n_control_suda"] = n_s
    json.dump(log_, open(f"{P}/sample_log.json", "w"), indent=1, default=str)
    print(json.dumps(log_, indent=1, default=str))


# ---------------------------------------------------------------- calibration-only check of the picture selection
def stage_calcheck():
    """E15 calibration objects: index and decision with (a) E15's 5-bin error pooling and (b) E22's selection
    (index-bin pictures only, pooling over 3 bins); both with E15's empty-position errors (data/phase2/e15/nulls_bins.csv)."""
    cal = pd.read_csv("data/phase2/e15/calib.csv")
    nb = pd.read_csv("data/phase2/e15/nulls_bins.csv")
    E.P = "data/phase2/e15"
    out = []
    for _, row in cal.iterrows():
        df = E.load_obj(row["obj"])
        r = {}
        for lab, sub in (("e15", df), ("e22", df[np.any([(df.wl_um >= lo) & (df.wl_um < hi) for lo, hi in IDX_BINS.values()], axis=0)])):
            bf, s, _ = E.bin_fluxes(sub)
            I, Eq, _ = E.index_of(bf)
            In, En, dn = null_index(bf, nb[nb.obj == row["obj"]])
            r[lab] = dict(I=I, err_null=En, n=len(sub), s=s, decision=dn)
        out.append(dict(name=row["Name"], is_T=bool(row["spt"] >= 10), **{f"{k}_{a}": v for k, d_ in r.items() for a, v in d_.items()}))
        print(row["Name"], {k: (round(v["I"], 3), round(v["err_null"], 3), v["n"], round(v["s"], 2), v["decision"]) for k, v in r.items()})
    E.P = P
    json.dump(out, open(f"{P}/calcheck.json", "w"), indent=1, default=float)
    for lab in ("e15", "e22"):
        dec = [o for o in out if o[f"{lab}_decision"] != "undetermined"]
        print(lab, "decisive", len(dec), "correct", sum((o[f"{lab}_decision"] == "T or later") == o["is_T"] for o in dec),
              "false T among L", sum(o[f"{lab}_decision"] == "T or later" and not o["is_T"] for o in out))


def null_index(bf, g):
    """E15 nullcheck rule (frozen as E22's primary error): bin error = max(quoted, std over the 12 empty positions)."""
    b2 = {}
    for k, v in bf.items():
        col = f"{k}_F"
        sd = float(g[col].std()) if col in g and g[col].notna().sum() >= 6 else np.nan
        b2[k] = (v[0], max(v[1], sd) if np.isfinite(sd) else v[1], v[2])
    I, Er, why = E.index_of(b2)
    return I, Er, E.decide(I, Er, THETA) if why == "" else "undetermined"


# ---------------------------------------------------------------- forced photometry
LISTS = dict(sample="sample", control="control", spiff="spiff_group", sample2="sample2")


def stage_forced(which, workers, reverse=False):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    lst = pd.read_csv(f"{P}/{LISTS[which]}.csv")
    lst["obj"] = lst.get("obj", pd.Series(index=lst.index, dtype=object)).fillna(lst.Name.str.replace(" ", "_").str.replace("+", "p"))
    idx_full = pd.read_parquet(INDEX, columns=["key", "obs_id", "detector", "mjd", "ra", "dec", "c_ra", "c_dec", "dqa"])
    idx_k = idx_full.set_index("key")
    def prep(row):
        o = row["obj"]
        fd = f"{P}/forced/{o}"
        os.makedirs(fd, exist_ok=True)
        cp = f"{fd}/_frames.parquet"
        if os.path.exists(cp):
            c = pd.read_parquet(cp)
        else:
            c = frames_sel(row, idx_full, idx_k)[["key", "obs_id", "detector", "mjd", "pred_ra", "pred_dec", "wl_pred"]]
            c.to_parquet(cp)
        c = c[in_idx_bins(c.wl_pred.values)]
        if len(c) > PIC_CAP:                             # deviation 18:50 (deep-field objects): fixed random subset
            c = c.sample(PIC_CAP, random_state=22)
        keep = {f"{r.obs_id}_D{r.detector}.json" for r in c.itertuples()}
        for f in glob.glob(f"{fd}/*.json"):              # fits outside the subset are set aside, unused by every stage
            if not os.path.basename(f).startswith("_") and os.path.basename(f) not in keep:
                os.makedirs(f"{fd}/unselected", exist_ok=True)
                os.replace(f, f"{fd}/unselected/{os.path.basename(f)}")
        r0, d0 = E.position(row, 60900.0)
        src = f"data/phase2/e15/gaia/{o}.parquet"
        gp = src if os.path.exists(src) else E.gaia_field(o, r0, d0)
        print(o, len(c), "pictures", flush=True)
        t = [(r.key, float(r.pred_ra), float(r.pred_dec), f"{fd}/{r.obs_id}_D{r.detector}.json", gp) for r in c.itertuples()]
        return [x for x in t if not os.path.exists(x[3])]
    with ThreadPoolExecutor(workers) as ex:              # listing + Gaia queries in parallel (no pixels read)
        todo = [x for t in ex.map(prep, [r for _, r in lst.iterrows()]) for x in t]
    if reverse:                                          # second process working the same list from the faint end
        todo = todo[::-1]
    print("to do", len(todo), time.strftime("%H:%M:%S"), flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(E._job, t) for t in todo]
        for i, f in enumerate(as_completed(futs)):
            r = f.result()
            if "error" in r or i % 200 == 0:
                print(i, f"{time.time() - t0:.0f}s", r.get("key", "")[-40:], r.get("error", ""), flush=True)
    print("done", time.strftime("%H:%M:%S"), flush=True)


def _null_job(args):
    import run_e15 as E2
    E2.P = P
    return E2._null_obj(args)


def stage_nulls(which, workers):
    from concurrent.futures import ProcessPoolExecutor
    lst = pd.read_csv(f"{P}/{LISTS[which]}.csv")
    lst["obj"] = lst.get("obj", pd.Series(index=lst.index, dtype=object)).fillna(lst.Name.str.replace(" ", "_").str.replace("+", "p"))
    done = pd.read_csv(f"{P}/nulls_bins_{which}.csv") if os.path.exists(f"{P}/nulls_bins_{which}.csv") else None
    have = set(done.obj) if done is not None else set()      # resume: objects already in the table are kept
    jobs = []
    for o in lst.obj:
        if o in have:
            continue
        src = f"data/phase2/e15/gaia/{o}.parquet"
        jobs.append((o, src if os.path.exists(src) else f"{P}/gaia/{o}.parquet"))
    out = done.to_dict("records") if done is not None else []
    with ProcessPoolExecutor(workers) as ex:
        for rows in ex.map(_null_job, jobs):
            df = pd.DataFrame(rows)
            if not len(df):
                continue
            for (o, k), gk in df.groupby(["obj", "k"]):
                gk = gk[np.isfinite(gk.flux_mJy) & (gk.flux_err_mJy > 0)]
                if not len(gk):
                    continue
                bf, sc, _ = E.bin_fluxes(gk)
                out.append(dict(obj=o, k=k, **{f"{b}_F": v[0] for b, v in bf.items()}, **{f"{b}_E": v[1] for b, v in bf.items()}))
            pd.DataFrame(out).to_csv(f"{P}/nulls_bins_{which}.csv", index=False)
    print("nulls done", which, len(out), time.strftime("%H:%M:%S"))


# ---------------------------------------------------------------- analysis
def measure_list(which, only_with_nulls=False):
    lst = pd.read_csv(f"{P}/{LISTS[which]}.csv")
    lst["obj"] = lst.get("obj", pd.Series(index=lst.index, dtype=object)).fillna(lst.Name.str.replace(" ", "_").str.replace("+", "p"))
    nb = pd.read_csv(f"{P}/nulls_bins_{which}.csv") if os.path.exists(f"{P}/nulls_bins_{which}.csv") else pd.DataFrame(columns=["obj"])
    if only_with_nulls:                    # truncated run (protocol deviation 21:07): the W2-ordered prefix that was finished
        last = max(i for i, o in enumerate(lst.obj) if o in set(nb.obj))
        lst = lst.iloc[:last + 1]
    res = []
    for _, row in lst.iterrows():
        df = E.load_obj(row["obj"])
        rec = dict(name=row["Name"], obj=row["obj"], source=row.get("source"), phot_type=row.get("phot_type"),
                   ra_57170=float(row["RACdeg"]), dec_57170=float(row["DECdeg"]), pmra_mas=float(row["pmra_mas"]), pmdec_mas=float(row["pmdec_mas"]),
                   W1=None if pd.isna(row.get("w1")) else float(row["w1"]), W2=float(row["w2"]), n_pictures=int(len(df)))
        if which == "spiff":
            rec["spiff"] = row.get("spiff")
            rec["suda"] = row.get("suda")
        if len(df):
            df = df[np.any([(df.wl_um >= lo) & (df.wl_um < hi) for lo, hi in IDX_BINS.values()], axis=0)]
        rec["n_pictures_in_bins"] = int(len(df))
        if not len(df):
            rec.update(ch4_index=None, ch4_err_null=None, z_null=None, call="undetermined", note="no pictures")
            res.append(rec)
            continue
        bf, s, nclip = E.bin_fluxes(df)
        I, Eq, why = E.index_of(bf)
        g = nb[nb.obj == row["obj"]]
        In, En, dn = null_index(bf, g)
        rec.update(error_scale=s, n_clipped=nclip, bins={k: dict(F_mJy=v[0], err_mJy=v[1], n=v[2]) for k, v in bf.items()},
                   ch4_index=None if not np.isfinite(I) else I, ch4_err_quoted=None if not np.isfinite(Eq) else Eq,
                   ch4_err_null=None if not np.isfinite(En) else En, n_null_positions=int(len(g)),
                   z_null=float((THETA - I) / En) if np.isfinite(I) and np.isfinite(En) and En > 0 else None,
                   call=dn if len(g) else "undetermined", call_quoted_errors=E.decide(I, Eq, THETA), note=why or ("no empty-position fits" if not len(g) else ""))
        res.append(rec)
    return res


def binom_ci(k, n, cl=0.95):
    from scipy.stats import beta
    lo = 0.0 if k == 0 else float(beta.ppf((1 - cl) / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - (1 - cl) / 2, k + 1, n - k))
    return lo, hi


def stage_analyze():
    out = dict(threshold=THETA, nsig=NSIG)
    for which in ("sample", "sample2", "control", "spiff"):
        if os.path.exists(f"{P}/{LISTS[which]}.csv") and os.path.exists(f"{P}/nulls_bins_{which}.csv"):
            log_ = f"{P}/nulls_{which}.log"
            trunc = which == "sample2" and not (os.path.exists(log_) and "nulls done" in open(log_).read())
            out[which] = measure_list(which, only_with_nulls=trunc)
            if which == "sample2":
                out["tier2_truncated"] = trunc
                out["tier2_n_listed"] = len(pd.read_csv(f"{P}/sample2.csv"))
    S, C = out.get("sample", []) + out.get("sample2", []), out.get("control", [])
    cnt = lambda L, c: sum(r["call"] == c for r in L)
    for tag, L in (("tier1", out.get("sample", [])), ("tier2", out.get("sample2", []))):
        out[f"{tag}_n"] = len(L)
        out[f"{tag}_n_T_or_later"] = cnt(L, "T or later")
        out[f"{tag}_n_L_like"] = cnt(L, "L-like")
    out["n_sample"] = len(S)
    out["n_measured"] = sum(r["n_pictures_in_bins"] > 0 for r in S)
    out["n_index_defined"] = sum(r.get("ch4_index") is not None for r in S)
    out["n_T_or_later"] = cnt(S, "T or later")
    out["n_L_like"] = cnt(S, "L-like")
    out["n_undetermined"] = cnt(S, "undetermined")
    G = out.get("spiff", [])
    def ref_t(r):
        for k in ("spiff", "suda"):
            v = r.get(k)
            if isinstance(v, str) and v:
                v = json.loads(v)
                t = v.get("autotype") or v.get("SpT")
                if t:
                    return str(t)
        return ""
    agree = [(r["call"], ref_t(r)) for r in G if r["call"] != "undetermined" and ref_t(r)[:1] in "MLTY" and ref_t(r)]
    out["spiff_group_n"] = len(G)
    out["spiff_group_n_decisive_with_type"] = len(agree)
    out["spiff_group_n_agree"] = sum((c == "T or later") == (t[0] in "TY") for c, t in agree)
    out["n_control"] = len(C)
    out["control_n_decisive"] = sum(r["call"] != "undetermined" for r in C)
    out["control_n_false_T"] = cnt(C, "T or later")
    # expected false "T or later" calls (protocol (f)): pooled L-dwarf false-call rate (E15 calibration L dwarfs with
    # empty-position errors: 0 of 10; plus the fresh control) times the number of sample objects with a decisive call
    kf, nf = out["control_n_false_T"] + 0, len(C) + 10
    p, (plo, phi) = kf / nf, binom_ci(kf, nf)
    nd = out["n_T_or_later"] + out["n_L_like"]
    out["false_rate_per_L"] = p
    out["false_rate_per_L_95"] = [plo, phi]
    out["expected_false"] = p * nd
    out["expected_false_95_upper"] = phi * nd
    json.dump(out, open(f"{P}/analysis.json", "w"), indent=1, default=float)
    print({k: v for k, v in out.items() if not isinstance(v, list)})
    for r in S + C:
        if r["call"] != "undetermined":
            print(r["source"], r["name"], r["W2"], round(r["ch4_index"], 3), round(r["ch4_err_null"], 3), r["call"])
    return out


# ---------------------------------------------------------------- tier 2 (Dal Ponte et al. 2023 DES/VHS L/T candidates)
DP_EPOCH = 57388.0      # DES Y6 coadd positions taken as epoch 2016.0
CAP2 = 1500


def stage_sample2():
    """Tier 2 (docs/protocols/E22.md (i)): Dal Ponte et al. 2023 (J/MNRAS/522/1951) photometric L0-T9 candidates with
    AllWISE W2 < 15.0, plus every photometric T (type code >= 20) with W2 < 16.0; proper motion Gaia DR3, else CatWISE2020,
    else NSC DR2 (rows with none dropped); positions moved from epoch 2016.0 to MJD 57170. Same exclusions as tier 1
    (E15 lists, K24 types, SIMBAD sp_type not from an input catalogue, SPIFF, SUDA) plus anything within 10\" of a tier-1
    sample/control/SPIFF-group object. Coverage >= 20 pictures on D4-6. Brightest W2 first, cap CAP2."""
    from astropy.coordinates import SkyCoord
    from concurrent.futures import ThreadPoolExecutor
    d = list(read_viz(f"{CAT}/J_MNRAS_522_1951.tsv").values())[0]
    for c in d.columns:
        if c != "CatWISE2020":
            d[c] = num(d[c])
    d = d[(d.W2mag < 15.0) | ((d.SPType >= 20) & (d.W2mag < 16.0))]
    pmra = d.pmRAgdr3.fillna(d.pmRAcwis).fillna(d.pmRAnsc)
    pmde = d.pmDEgdr3.fillna(d.pmDEcwis).fillna(d.pmDEnsc)
    pmsrc = np.where(d.pmRAgdr3.notna(), "GaiaDR3", np.where(d.pmRAcwis.notna(), "CatWISE2020", np.where(d.pmRAnsc.notna(), "NSCDR2", "")))
    d = d.assign(pmra=pmra, pmde=pmde, pmsrc=pmsrc)
    log_ = dict(n_selected_W2=len(d), n_no_pm=int((d.pmsrc == "").sum()))
    d = d[d.pmsrc != ""]
    name = ["DES " + str(int(x)) for x in d.CoadID]
    t = std_rows(name, name, d.RAJ2000.values, d.DEJ2000.values, DP_EPOCH, d.pmra.values, d.pmde.values, d.W1mag.values,
                 d.W2mag.values, "DP23", [E.spt_str(x - 10) for x in d.SPType.values])
    t["pm_source"] = d.pmsrc.values
    t["obj"] = t.Name.str.replace(" ", "_")
    t = t.sort_values("w2").reset_index(drop=True)
    prev = pd.concat([pd.read_csv(f"{P}/{f}.csv")[["RACdeg", "DECdeg"]] for f in ("sample_all", "control")])
    _, dprev, _ = SkyCoord(t.RACdeg.values, t.DECdeg.values, unit="deg").match_to_catalog_sky(SkyCoord(prev.RACdeg.values, prev.DECdeg.values, unit="deg"))
    t["in_tier1"] = dprev.arcsec < 10
    h = pd.read_csv("data/phase2/e15/heldout.csv")
    ih, dh, _ = SkyCoord(t.RACdeg.values, t.DECdeg.values, unit="deg").match_to_catalog_sky(SkyCoord(h.RACdeg.values, h.DECdeg.values, unit="deg"))
    t["e15_dropped"] = [h.Name.iloc[int(i)] if x < 10 else "" for i, x in zip(ih, dh.arcsec)]
    t["k24_type"] = k24_types(t)
    pm = prior_match(t)
    t["spiff"] = [json.dumps(x["spiff"]) if x["spiff"] else "" for x in pm]
    t["suda"] = [json.dumps(x["suda"]) if x["suda"] else "" for x in pm]

    def both(r):
        a, b = simbad_cone(*E.position(r, 51544.5)), simbad_cone(float(r.RACdeg), float(r.DECdeg))
        if a is None or b is None:
            return None
        return list({x["oid"]: x for x in a + b}.values())
    with ThreadPoolExecutor(3) as ex:
        sim = list(ex.map(both, [r for _, r in t.iterrows()]))
    t["simbad"] = [json.dumps(x) for x in sim]
    bib = SRC_BIB | {"2023MNRAS.522.1951D"}

    def why(r, s):
        if r.in_tier1:
            return "already in tier 1 lists"
        if r.e15_dropped:
            return "E15 prior-spectrum check"
        if isinstance(r.k24_type, str) and r.k24_type:
            return f"K24 spectral type {r.k24_type}"
        if s is None:
            return "SIMBAD query failed"
        x = [y for y in s if (y["sp_type"] or "").strip() and (y["sp_bibcode"] or "") not in bib]
        if x:
            return f"SIMBAD sp_type {x[0]['sp_type']} ({x[0]['sp_bibcode']}, {x[0]['main_id']})"
        if isinstance(r.spiff, str) and r.spiff:
            return "in SPIFF (" + json.loads(r.spiff)["dataset"] + ")"
        if isinstance(r.suda, str) and r.suda:
            return "in SUDA"
        return None
    t["excluded"] = [why(r, s) for (_, r), s in zip(t.iterrows(), sim)]
    log_["excluded_counts"] = t.excluded.fillna("kept").map(lambda s: s.split(" (")[0].split(" sp_type")[0].split(" spectral")[0]).value_counts().to_dict()
    t.to_csv(f"{P}/sample2_all.csv", index=False)
    keep = t[t.excluded.isna()].reset_index(drop=True)
    idx_full = pd.read_parquet(INDEX, columns=["key", "obs_id", "detector", "mjd", "ra", "dec", "c_ra", "c_dec", "dqa"])
    rows = []
    for _, row in keep.iterrows():
        if len(rows) >= CAP2:
            break
        fr = E.frames_for(row, idx_full)
        n456 = int(fr.detector.isin([4, 5, 6]).sum())
        if n456 >= 20:
            rows.append(dict(row, n_listed_d456=n456))
    s2 = pd.DataFrame(rows)
    s2.to_csv(f"{P}/sample2.csv", index=False)
    log_.update(n_kept_before_coverage=len(keep), n_sample2=len(s2), w2_range=[float(s2.w2.min()), float(s2.w2.max())],
                phot_type_counts=s2.phot_type.str[0].value_counts().to_dict())
    json.dump(log_, open(f"{P}/sample2_log.json", "w"), indent=1, default=str)
    print(json.dumps(log_, indent=1, default=str))


# ---------------------------------------------------------------- survivors: literature check
def short_desig(name, ra, dec):
    from astropy.coordinates import SkyCoord
    c = SkyCoord(ra, dec, unit="deg")
    h = c.ra.hms
    dsg = "-" if dec < 0 else "+"
    dd = abs(c.dec.dms.d), abs(c.dec.dms.m)
    return f"J{int(h.h):02d}{int(h.m):02d}{dsg}{int(dd[0]):02d}{int(dd[1]):02d}"


def stage_lit():
    """For every 'T or later' sample object: SIMBAD objects within 10" (epoch-2000 and MJD-57170 positions), their
    sp_type and ALL references (bibcode + title); arXiv API title/abstract search of the short designation."""
    import requests, re
    a = json.load(open(f"{P}/analysis.json"))
    out = []
    for which in ("sample", "sample2"):
        for r in a.get(which, []):
            if r["call"] != "T or later":
                continue
            row = pd.Series(dict(RACdeg=r["ra_57170"], DECdeg=r["dec_57170"], pmra_mas=r["pmra_mas"], pmdec_mas=r["pmdec_mas"]))
            sims = {}
            for ra, dec in (E.position(row, 51544.5), (row.RACdeg, row.DECdeg)):
                for x in simbad_cone(float(ra), float(dec)) or []:
                    sims[x["oid"]] = x
            refs = []
            for oid in sims:
                q2 = ("SELECT r.bibcode, r.title FROM has_ref AS h JOIN ref AS r ON h.oidbibref = r.oidbib "
                      f"WHERE h.oidref = {oid}")
                for k in range(4):
                    try:
                        rr = requests.get(SIMBAD_TAP, params=dict(request="doQuery", lang="adql", format="json", query=q2), timeout=90).json()
                        refs += [dict(oid=oid, bibcode=x[0], title=x[1]) for x in rr.get("data", [])]
                        break
                    except Exception:
                        time.sleep(5 * (k + 1))
            sd = short_desig(r["name"], *E.position(row, 51544.5))
            aq = f'abs:"{sd}" OR ti:"{sd}"'
            hits = None                                  # None = arXiv query failed after retries
            for k in range(4):
                try:
                    ax = requests.get("https://export.arxiv.org/api/query", params=dict(search_query=aq, max_results=20), timeout=90).text
                    hits = [dict(id=i, title=" ".join(t.split())) for i, t in zip(re.findall(r"<id>http://arxiv.org/abs/([^<]+)</id>", ax),
                                                                               re.findall(r"<title>([^<]+)</title>", ax)[1:])]
                    break
                except Exception:
                    time.sleep(10 * (k + 1))
            out.append(dict(list=which, name=r["name"], obj=r["obj"], short=sd, simbad=list(sims.values()), simbad_refs=refs,
                            arxiv_query=aq, arxiv_hits=hits))
            print(r["name"], sd, [(x["main_id"], x["sp_type"]) for x in sims.values()], len(refs), None if hits is None else len(hits), flush=True)
            time.sleep(1)
    json.dump(out, open(f"{P}/lit_check.json", "w"), indent=1)


# ---------------------------------------------------------------- findings + figure
def fig_census(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 14})
    fig, ax = plt.subplots(figsize=(12, 7.5), facecolor="white")
    def pts(L):
        L = [r for r in L if r.get("ch4_index") is not None and r.get("ch4_err_null") is not None]
        return L
    groups = [("undetermined", "#b8b8b8", "o", 4, "sample: undetermined (too faint)"),
              ("L-like", "#2a78d6", "o", 7, "sample: no methane trough (L-like)"),
              ("T or later", "#d6402a", "*", 15, "sample: methane trough, T or later (candidates)")]
    S = pts(a.get("sample", []) + a.get("sample2", []))
    for call, col, m, ms, lab in groups:
        rr = [r for r in S if r["call"] == call]
        if rr:
            ax.errorbar([r["W2"] for r in rr], [r["ch4_index"] for r in rr], yerr=[r["ch4_err_null"] for r in rr], fmt=m, ms=ms,
                        color=col, mec="white" if m == "o" else "black", mew=0.6, elinewidth=0.8 if call == "undetermined" else 1.3,
                        alpha=0.45 if call == "undetermined" else 1, capsize=0, label=f"{lab}: {len(rr)}", zorder=3 if call != "undetermined" else 1)
    C = pts(a.get("control", []))
    if C:
        w = [r["W2"] if r["W2"] == r["W2"] else (r["W1"] - 0.4 if r["W1"] else np.nan) for r in C]
        ax.errorbar(w, [r["ch4_index"] for r in C], yerr=[r["ch4_err_null"] for r in C], fmt="s", ms=7, color="#1baf7a",
                    mec="black", mew=0.6, elinewidth=1.2, capsize=0, label=f"control: known L0–L8.5 dwarfs: {len(C)} (SUDA ones placed at W1 − 0.4)", zorder=4)
    ax.axhline(THETA, color="#333", lw=2, ls="--")
    ax.text(0.01, THETA + 0.04, f"E15 dividing line {THETA} (frozen)", transform=ax.get_yaxis_transform(), fontsize=12)
    ax.set_ylim(-1.5, 2.0)
    ax.set_xlabel("WISE W2 magnitude (fainter →)")
    ax.set_ylabel("methane index: flux 3.2–3.4 µm ÷ neighbours\n(deeper methane absorption ↓)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=11, loc="upper left", framealpha=0.95)
    ax.set_title("SPHEREx 3.3 µm methane census of brown-dwarf candidates with no published spectral type", fontsize=15)
    n_all = len(a.get("sample", []) + a.get("sample2", []))
    fig.text(0.01, 0.005, f"Error bars 1σ (bin errors from 12 empty positions per object). A call needs ≥ 3σ from the line.\n"
             f"Not plotted: {n_all - len(S)} of {n_all} sample objects with no defined index or error (too faint); values outside −1.5…2 fall off the plot.",
             fontsize=10, color="#444")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig("site/data/figs/e22_methane_census.png", dpi=130, facecolor="white")


# Survivors with a published spectral type found by the literature check (filled after reading lit_check.json):
# name -> reason. Objects listed here are not counted as candidates.
LIT_DROP = {}
# Survivors kept as candidates but with relevant prior work (not a spectral type from a spectrum):
LIT_NOTE = {"WISE 0628-8057": "methane-imaging photometric type (Tinney et al. 2018, ApJS 236, 28; 2018ApJS..236...28T); no spectrum found"}


def stage_findings():
    a = json.load(open(f"{P}/analysis.json"))
    lit = {r["name"]: r for r in json.load(open(f"{P}/lit_check.json"))} if os.path.exists(f"{P}/lit_check.json") else {}
    S = a.get("sample", []) + a.get("sample2", [])
    rows = []
    for tier, L in (("tier1", a.get("sample", [])), ("tier2", a.get("sample2", []))):
        for r in L:
            rows.append(dict(tier=tier, name=r["name"], source=r["source"], catalogue_type=r["phot_type"], ra_deg_mjd57170=round(r["ra_57170"], 6),
                             dec_deg_mjd57170=round(r["dec_57170"], 6), W1=r["W1"], W2=r["W2"], n_pictures=r["n_pictures_in_bins"],
                             ch4_index=r.get("ch4_index"), ch4_err=r.get("ch4_err_null"), sigma_below_threshold=r.get("z_null"), call=r["call"],
                             lit_checked=r["name"] in lit, published_type_found=LIT_DROP.get(r["name"]), lit_note=LIT_NOTE.get(r["name"])))
    cands = [x for x in rows if x["call"] == "T or later" and not x["published_type_found"]]
    C = a.get("control", [])
    f = dict(experiment="E22", protocol="docs/protocols/E22.md", script="scripts/run_e22.py", generated=time.strftime("%Y-%m-%d %H:%M %Z"),
             question="Applying E15's frozen 3.3 um methane rule to every ultracool-dwarf candidate we could assemble that has no "
                      "published spectral type: how many show a methane trough (T or later)?",
             rule="CH4 = F(3.2-3.4)/mean(F(2.9-3.1),F(3.9-4.1)); 'T or later' if >= 3 sigma below 0.356 (E15 frozen), 'L-like' if >= 3 sigma above; "
                  "bin errors max(quoted, std over 12 empty positions per object)",
             threshold=THETA, tier2_truncated=a.get("tier2_truncated"), tier2_n_listed=a.get("tier2_n_listed"), n_sample=a["n_sample"], n_sample_tier1=a["tier1_n"], n_sample_tier2=a["tier2_n"],
             n_measured=a["n_measured"], n_index_defined=a["n_index_defined"], n_T_or_later=a["n_T_or_later"],
             n_T_or_later_tier1=a["tier1_n_T_or_later"], n_T_or_later_tier2=a["tier2_n_T_or_later"], n_L_like=a["n_L_like"],
             n_undetermined=a["n_undetermined"], n_published_type_found_in_lit_check=sum(1 for x in rows if x["call"] == "T or later" and x["published_type_found"]),
             n_candidates_no_published_type=len(cands), n_candidates_with_prior_methane_imaging=sum(1 for c in cands if c["lit_note"]),
             n_candidates_tier1=sum(1 for c in cands if c["tier"] == "tier1"), n_candidates_tier2=sum(1 for c in cands if c["tier"] == "tier2"), expected_false=a["expected_false"], expected_false_95_upper=a["expected_false_95_upper"],
             false_rate_per_L=a["false_rate_per_L"], false_rate_per_L_95=a["false_rate_per_L_95"],
             expected_false_gaussian_3sigma=0.00135 * a["n_index_defined"],
             control_n=a["n_control"], control_n_decisive=a["control_n_decisive"], control_n_false_T_or_later=a["control_n_false_T"],
             control_n_L_like=sum(r["call"] == "L-like" for r in C),
             tier2_photL0_L5_n_decisive=sum(1 for r in a.get("sample2", []) if r["call"] != "undetermined" and r["phot_type"][:1] == "L" and float(r["phot_type"][1:] or 0) < 6),
             tier2_photL0_L5_n_T_or_later=sum(1 for r in a.get("sample2", []) if r["call"] == "T or later" and r["phot_type"][:1] == "L" and float(r["phot_type"][1:] or 0) < 6),
             tier2_photL_n_decisive=sum(1 for r in a.get("sample2", []) if r["call"] != "undetermined" and r["phot_type"][:1] == "L"),
             tier2_photL_n_T_or_later=sum(1 for r in a.get("sample2", []) if r["call"] == "T or later" and r["phot_type"][:1] == "L"),
             tier2_photT_n=sum(1 for r in a.get("sample2", []) if r["phot_type"][:1] == "T"),
             tier2_photT_n_T_or_later=sum(1 for r in a.get("sample2", []) if r["call"] == "T or later" and r["phot_type"][:1] == "T"),
             tier2_photT_n_L_like=sum(1 for r in a.get("sample2", []) if r["call"] == "L-like" and r["phot_type"][:1] == "T"),
             spiff_group_n=a.get("spiff_group_n"), spiff_group_n_decisive_with_type=a.get("spiff_group_n_decisive_with_type"),
             spiff_group_n_agree=a.get("spiff_group_n_agree"),
             candidates=cands, control=[dict(name=r["name"], source=r["source"], type=r["phot_type"], W2=r["W2"], ch4_index=r.get("ch4_index"),
                                            ch4_err=r.get("ch4_err_null"), call=r["call"]) for r in C],
             objects=rows, lit_drop=LIT_DROP,
             wording="Candidate: 'SPHEREx colours indicate a methane-bearing (T-type or later) brown-dwarf candidate; we found no published "
                     "spectral type in the sources we checked.' Each needs a spectrum to confirm its type.",
             summary=("We took 985 known brown-dwarf candidates that, as far as the catalogues and literature we searched show, have never "
                      "had their spectral type measured from a spectrum, and checked each one in SPHEREx data for the dip at 3.3 microns "
                      "that methane gas causes in cold brown dwarfs (T and Y types, below roughly 1300 K). We used the rule fixed earlier "
                      "on brown dwarfs of known type, without changing it. 687 objects were bright enough for a clear answer: 20 show the "
                      "methane dip and 667 do not. None of 53 known warmer (L-type) brown dwarfs measured the same way was wrongly given "
                      "the dip, and on 13 objects that another SPHEREx team had already typed, our answer agreed every time. The 20 are "
                      "candidates for cold, nearby brown dwarfs that still need a spectrum to confirm."),
             figure="site/data/figs/e22_methane_census.png")
    # post hoc (not frozen; labelled as such): pool the control with tier-2 photometric-L objects that got a decisive call
    kp = f["control_n_false_T_or_later"] + f["tier2_photL_n_T_or_later"]
    npool = f["control_n"] + 10 + f["tier2_photL_n_decisive"]
    f["posthoc_false_rate_95_upper"] = binom_ci(kp, npool)[1]
    f["posthoc_expected_false_95_upper"] = f["posthoc_false_rate_95_upper"] * (f["n_T_or_later"] + f["n_L_like"])
    f["posthoc_note"] = ("post hoc, not in the frozen protocol: false-call rate pooled over the 63 spectroscopic L dwarfs and the tier-2 "
                         "photometric L candidates with a decisive call (photometric types, so only indicative)")
    json.dump(f, open("data/findings/e22.json", "w"), indent=1, default=float)
    print({k: v for k, v in f.items() if not isinstance(v, (list, dict))})
    # flat token file: scalar keys only (no dots, hyphens or spaces in keys), rounded for text
    tok = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in f.items()
           if isinstance(v, (int, float, str)) and k not in ("question", "rule", "wording", "generated", "script", "protocol", "figure", "experiment", "posthoc_note")}
    tok["false_rate_per_L_95_lo"], tok["false_rate_per_L_95_hi"] = [round(x, 3) for x in f["false_rate_per_L_95"]]
    tok["n_control_K21"] = sum(1 for r in C if r["source"].startswith("K21"))
    tok["n_control_SUDA"] = sum(1 for r in C if r["source"].startswith("SUDA"))
    for i, c in enumerate(sorted(cands, key=lambda x: -(x["sigma_below_threshold"] or 0))[:10], 1):
        tok[f"cand{i}_name"] = c["name"]
        tok[f"cand{i}_sigma"] = round(c["sigma_below_threshold"], 1)
        tok[f"cand{i}_index"] = round(c["ch4_index"], 3)
        tok[f"cand{i}_err"] = round(c["ch4_err"], 3)
        tok[f"cand{i}_W2"] = round(c["W2"], 2)
    tok["n_decisive"] = f["n_T_or_later"] + f["n_L_like"]
    tok["search_date"] = "2026-11-15"
    tok["sources_checked"] = ("SIMBAD (Harvard mirror TAP: sp_type and all reference titles, 10 arcsec, two epochs); Kirkpatrick et al. 2024 "
                              "J/ApJS/271/55; Meisner et al. 2020b table 3; E15 prior-spectrum list; SPIFF autotype table (Gagne, Faherty et al. "
                              "2026); SUDA catalogue (Tu et al. 2026); arXiv API title/abstract search; texts of arXiv 2604.22012, 2604.26771, "
                              "2607.00543, 2608.08908")
    assert all(not any(ch in k for ch in ".- ") for k in tok)
    # candidate table for the products folder
    os.makedirs("data/products", exist_ok=True)
    simb = {r["name"]: "; ".join(sorted({x["main_id"] for x in r["simbad"]})) for r in lit.values()}
    pd.DataFrame([dict(name=c["name"], simbad_ids=simb.get(c["name"], ""), tier=c["tier"], input_catalogue=c["source"],
                       catalogue_photometric_type=c["catalogue_type"], ra_deg=c["ra_deg_mjd57170"], dec_deg=c["dec_deg_mjd57170"], position_epoch_mjd=57170,
                       W1=c["W1"], W2=c["W2"], n_pictures=c["n_pictures"], methane_index=round(c["ch4_index"], 4), methane_index_err=round(c["ch4_err"], 4),
                       sigma_below_threshold=round(c["sigma_below_threshold"], 2), call=c["call"],
                       literature_status=("no published spectral type found" + (f"; note: {c['lit_note']}" if c["lit_note"] else "")
                                          if c["lit_checked"] else "literature check not run"),
                       search_date="2026-11-15")
                  for c in sorted(cands, key=lambda x: -x["sigma_below_threshold"])]).to_csv("data/products/epoch_methane_candidates.csv", index=False)
    json.dump(tok, open("data/findings/e22s.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("which", nargs="?")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--reverse", action="store_true")
    a = ap.parse_args()
    dict(findings=lambda: stage_findings(), lit=lambda: stage_lit(), fig=lambda: fig_census(json.load(open(f"{P}/analysis.json"))), sample=lambda: stage_sample(), sample2=lambda: stage_sample2(), calcheck=lambda: stage_calcheck(),
         forced=lambda: stage_forced(a.which, a.workers, a.reverse), nulls=lambda: stage_nulls(a.which, a.workers),
         analyze=lambda: stage_analyze())[a.stage]()
