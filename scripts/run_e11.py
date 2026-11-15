"""E11 (docs/protocols/E11.md): binned spectra of WISE 0855-0714, Epsilon Indi B (Ba+Bb) and Barnard's Star,
compared with published values. Stages (each writes to disk):
  w0855    per-picture fluxes on the E4 path and on the 24 E4 empty paths -> data/phase2/e11/wise0855_frames.json
  barnard  E7-style forced photometry on data/efields/e3x_barnard        -> data/phase2/e11/barnard_phot.json
  bin      binned spectra, band tests, comparison table                  -> data/findings/e11.json
  fig      site/data/figs/e11_<target>.png
Usage: OMP_NUM_THREADS=1 .venv/bin/python scripts/run_e11.py w0855|barnard|bin|fig
"""
import json, os, sys, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
warnings.filterwarnings("ignore")

P2 = "data/phase2/e11"
LIT = f"{P2}/lit"
OUT = "data/findings/e11.json"
FIGS = "site/data/figs"
W_EDGES = [round(3.8 + 0.1 * k, 1) for k in range(13)]
W_CTRL = [0.75, 1.10, 1.64, 2.42, 3.10, 3.80]
LOG_EDGES = (0.75 * (5.0 / 0.75) ** (np.arange(39) / 38)).tolist()
FLOOR = 0.03
MIN_PIX = 18
SAT_R = 1.5
BARNARD_ID = "4472832130942575872"

# (label, band window, reference windows) per docs/protocols/E11.md
BANDS = {
    "epsindib": [("H2O+CH4 1.15", (1.10, 1.20), [(1.04, 1.09), (1.24, 1.31)]),
                 ("H2O 1.4", (1.34, 1.45), [(1.24, 1.31), (1.55, 1.60)]),
                 ("CH4 1.6", (1.63, 1.74), [(1.55, 1.60)]),
                 ("H2O 1.9", (1.80, 1.98), [(1.55, 1.60), (2.06, 2.12)]),
                 ("CH4 2.2", (2.20, 2.35), [(2.06, 2.12)]),
                 ("CH4 3.3", (3.15, 3.45), [(3.90, 4.15)]),
                 ("CO 4.5-4.8", (4.50, 4.80), [(3.90, 4.15)])],
    "barnard": [("H2O 1.4", (1.34, 1.45), [(1.24, 1.31), (1.62, 1.72)]),
                ("H2O 1.9", (1.80, 1.98), [(1.62, 1.72), (2.10, 2.25)]),
                ("H2O 2.7", (2.45, 2.95), [(2.10, 2.25), (3.50, 3.90)])],
    "wise0855": [("CO 4.5-4.9", (4.5, 4.9), [(4.3, 4.5)])],
}


# ------------------------------------------------------------------ per-picture stages
def stage_w0855():
    import run_e4 as E4
    from orrery.stack import scene_residual
    from orrery.fit import flux_mJy
    df, _ = E4.load_rows()
    rows = []
    for f in E4.load_frames(df, restore=True):
        r, st = f["row"], f["st"]
        fl = []
        for ox, oy in [(0.0, 0.0)] + E4.NULLS:
            fit = scene_residual(st, f["xs"] + ox, f["ys"] + oy, f["star_xy"])[2]
            ok = np.isfinite(fit.A) and fit.A_err > 0
            fl.append((flux_mJy(fit.A, st.omega_arcsec2), flux_mJy(fit.A_err, st.omega_arcsec2)) if ok else (None, None))
        rows.append(dict(obs_id=r.obs_id, det=int(r.detector), mjd=float(r.mjd), wl=float(r.wl_um), key=r.key,
                         chi2_local_dof=float(r.chi2_local / max(r.dof_local, 1)), n_restored=int(f["n_restored"]),
                         flux=fl[0][0], err=fl[0][1], null_flux=[x[0] for x in fl[1:]], null_err=[x[1] for x in fl[1:]]))
    json.dump(rows, open(f"{P2}/wise0855_frames.json", "w"))
    print("w0855 pictures:", len(rows), "with chi2<=3:", sum(r["chi2_local_dof"] <= 3 for r in rows))


def stage_barnard():
    """Same photometry as scripts/run_e7.py:phot_field, on e3x_barnard; flagged-core test within SAT_R px."""
    import pandas as pd
    from astropy.io import fits
    from astropy.wcs import WCS
    import run_e7 as R7
    from orrery import e6, e7
    from orrery.data import load_stamp
    from orrery.e1 import wave_map
    from orrery.fit import flux_mJy
    d = "data/efields/e3x_barnard"
    meta = json.load(open(f"{d}/field.json"))
    gaia = pd.read_parquet(f"{d}/gaia.parquet")
    src, g19 = R7.sources("e3x_barnard", meta, gaia, float(np.mean([fl["mjd"] for fl in meta["frame_list"]])))
    assert any(s["id"] == BARNARD_ID for s in src), "Barnard's Star not among the reported sources"
    rec = {s["id"]: dict(G=s["G"], mjd=[], det=[], lam=[], flux=[], err=[], npix=[], sat=[], key=[]) for s in src}
    for fl in meta["frame_list"]:
        st = load_stamp(f"{d}/{fl['file']}")
        w = WCS(fits.Header.fromstring(st.wcs_header))
        xy, _ = e6.star_xy_native(gaia, st.mjd, w, st.x0, st.y0)
        cols = e6.star_columns(xy, st.detector, st.x0, st.y0, st.image.shape)
        res = e7.forced_photometry(st.image, st.variance, st.flags, cols, [s["members"] for s in src])
        H, W = st.image.shape
        yy, xx = np.indices((H, W))
        for s, (a, e, npix) in zip(src, res):
            x, y = xy[s["members"][0]]
            if not (0 <= x < W and 0 <= y < H):
                continue
            core = np.hypot(xx - x, yy - y) <= SAT_R
            r = rec[s["id"]]
            r["mjd"].append(st.mjd), r["det"].append(st.detector), r["key"].append(fl["key"])
            r["lam"].append(float(wave_map(st.detector, x + st.x0, y + st.y0)))
            r["flux"].append(flux_mJy(a, st.omega_arcsec2) if np.isfinite(a) else None)
            r["err"].append(flux_mJy(e, st.omega_arcsec2) if np.isfinite(e) else None)
            r["npix"].append(npix), r["sat"].append(bool((st.flags[core] & e7.SAT_BITS).any()))
    json.dump(dict(field="e3x_barnard", sat_radius_px=SAT_R, phot=rec), open(f"{P2}/barnard_phot.json", "w"))
    b = rec[BARNARD_ID]
    print("barnard pictures:", len(b["mjd"]), "unflagged core:", int(np.sum(~np.array(b["sat"]))), "sources:", len(src))


# ------------------------------------------------------------------ binning
def wmean(f, e):
    w = 1.0 / e ** 2
    return float(np.sum(w * f) / w.sum()), float(1 / np.sqrt(w.sum()))


def bin_spectrum(lam, flux, err, edges, clip=True):
    """Protocol bin rule for the two bright targets: outlier rule, weighted mean, error rescaled by the scatter
    about a straight line within the bin. Returns one dict per bin with >= 1 picture."""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (lam >= lo) & (lam < hi)
        l, f, e = lam[s], flux[s], err[s]
        n0 = len(f)
        if n0 == 0:
            continue
        if clip and n0 >= 4:
            med = np.median(f)
            keep = np.abs(f - med) <= 5 * np.maximum(1.4826 * np.median(np.abs(f - med)), e)
            l, f, e = l[keep], f[keep], e[keep]
        n = len(f)
        m, sm = wmean(f, e)
        chi2r = None
        if n >= 4:
            A = np.stack([np.ones(n), l - l.mean()], 1)
            c, *_ = np.linalg.lstsq(A / e[:, None], f / e, rcond=None)
            chi2r = float(np.sum(((f - A @ c) / e) ** 2) / (n - 2))
        elif n >= 2:
            chi2r = float(np.sum(((f - m) / e) ** 2) / (n - 1))
        stat = sm * max(1.0, np.sqrt(chi2r)) if chi2r is not None else sm
        out.append(dict(lo=float(lo), hi=float(hi), centre=float(np.sum(l / e ** 2) / np.sum(1 / e ** 2)), flux=m,
                        stat_err=float(stat), formal_err=sm, chi2_red=chi2r, n=int(n), n_dropped=int(n0 - n)))
    return out


def star_arrays(r, s_d, need_unsat=True):
    f = np.array([np.nan if v is None else v for v in r["flux"]], float)
    e = np.array([np.nan if v is None else v for v in r["err"]], float)
    det = np.array(r["det"])
    e = e * np.array([s_d[str(int(d))] for d in det])
    ok = np.isfinite(f) & np.isfinite(e) & (e > 0) & (np.array(r["npix"]) >= MIN_PIX)
    if need_unsat:
        ok &= ~np.array(r["sat"], bool)
    return dict(lam=np.array(r["lam"])[ok], flux=f[ok], err=e[ok], mjd=np.array(r["mjd"])[ok], det=det[ok]), int(ok.sum()), len(f)


def spectra_by_window(a, win_of, edges):
    win = np.array([win_of[m] for m in a["mjd"]])
    allw = bin_spectrum(a["lam"], a["flux"], a["err"], edges)
    per = {int(w): bin_spectrum(a["lam"][win == w], a["flux"][win == w], a["err"][win == w], edges) for w in np.unique(win)}
    return allw, per


def f_cmp(stars, win_of, edges):
    """Comparison-star check: robust spread of per-window bin flux / all-window bin flux - 1."""
    d = []
    for a in stars:
        if len(a["flux"]) < 20:
            continue
        allw, per = spectra_by_window(a, win_of, edges)
        ref = {b["lo"]: b["flux"] for b in allw}
        for bins in per.values():
            d += [b["flux"] / ref[b["lo"]] - 1 for b in bins if b["flux"] > 10 * b["stat_err"] and ref[b["lo"]] > 0]
    d = np.array(d)
    return (float(1.4826 * np.median(np.abs(d - np.median(d)))), int(len(d))) if len(d) else (None, 0)


def add_total(bins, f):
    for b in bins:
        b["err"] = float(np.hypot(b["stat_err"], f * abs(b["flux"])))
    return bins


def window_table(mjd, win_of):
    from astropy.time import Time
    win = np.array([win_of[m] for m in mjd])
    return [dict(window=int(w), start=Time(mjd[win == w].min(), format="mjd").iso[:10],
                 end=Time(mjd[win == w].max(), format="mjd").iso[:10], n_pictures=int((win == w).sum())) for w in np.unique(win)]


def band_tests(bins, bands):
    c = np.array([b["centre"] for b in bins]); f = np.array([b["flux"] for b in bins]); e = np.array([b["err"] for b in bins])
    out = []
    for label, bw, refs in bands:
        def grp(w):
            s = (c >= w[0]) & (c < w[1]) & (e > 0)
            if not s.any():
                return None
            m, sm = wmean(f[s], e[s])
            return m, sm, float(np.sum(c[s] / e[s] ** 2) / np.sum(1 / e[s] ** 2)), int(s.sum())
        B, R = grp(bw), [grp(w) for w in refs]
        r = dict(label=label, band_um=list(bw), reference_um=[list(w) for w in refs], named=False)
        if B is None or any(x is None for x in R):
            r["status"] = "not testable: a window has no bin"
        elif len(R) == 2 and (R[0][0] <= 0 or R[1][0] <= 0):
            r["status"] = "not testable: non-positive reference"
        else:
            if len(R) == 1:
                ref, sref = R[0][0], R[0][1]
            else:
                t = (np.log(B[2]) - np.log(R[0][2])) / (np.log(R[1][2]) - np.log(R[0][2]))
                ref = float(np.exp((1 - t) * np.log(R[0][0]) + t * np.log(R[1][0])))
                sref = float(ref * np.hypot((1 - t) * R[0][1] / R[0][0], t * R[1][1] / R[1][0]))
            D, sD = ref - B[0], float(np.hypot(sref, B[1]))
            r.update(status="tested", band_flux_mJy=B[0], band_err_mJy=B[1], n_band_bins=B[3], reference_mJy=ref,
                     reference_err_mJy=sref, dip_mJy=D, dip_err_mJy=sD, dip_over_err=D / sD,
                     depth_fraction=D / ref, named=bool(D > 3 * sD))
        out.append(r)
    return out


# ------------------------------------------------------------------ synthetic photometry
FILT = f"{LIT}/filters"
# zero-magnitude flux (Jy) and isophotal wavelength (um)
ZP = {"2MASS J": (1594.0, 1.235), "2MASS H": (1024.0, 1.662), "2MASS Ks": (666.7, 2.159),      # Cohen et al. 2003 (IPAC 2MASS doc VI.4a Table 1)
      "WISE W1": (306.681, 3.3526), "WISE W2": (170.663, 4.6028),                              # Wright et al. 2010 (arXiv:1008.0031)
      "IRAC [3.6]": (280.9, 3.550), "IRAC [4.5]": (179.7, 4.493)}                              # Reach et al. 2005 (as cited in scripts/run_e4.py)


def responses():
    """Per-photon response curves R(lambda [um])."""
    R = {}
    for b, f in (("2MASS J", "twomass-J.ecsv"), ("2MASS H", "twomass-H.ecsv"), ("2MASS Ks", "twomass-Ks.ecsv")):
        d = np.array([l.split() for l in open(f"{FILT}/{f}") if l[0].isdigit()], float)
        R[b] = (d[:, 0] / 1e4, d[:, 1] / d[:, 0])        # Cohen RSRs are for integration over F_lambda: per-photon = RSR / lambda
    for b, f in (("WISE W1", "wise_RSR-W1.txt"), ("WISE W2", "wise_RSR-W2.txt"), ("IRAC [3.6]", "irsa_irac_ch1.txt"), ("IRAC [4.5]", "irsa_irac_ch2.txt")):
        d = np.loadtxt(f"{FILT}/{f}", comments="#")
        R[b] = (d[:, 0], d[:, 1])
    return R


def synth(bins, f_floor, R, n_mc=400, seed=20261115):
    """Vega-relative synthetic band flux F = ZP * int(F_nu R dl/l) / int(ZP (l_iso/l)^2 R dl/l), spectrum linear between
    bin centres (flat to the outer bin edges). Returns {band: dict} for bands passing the coverage rule."""
    c = np.array([b["centre"] for b in bins]); f = np.array([b["flux"] for b in bins]); se = np.array([b["stat_err"] for b in bins])
    lo, hi = bins[0]["lo"], bins[-1]["hi"]
    gaps = [(bins[i]["hi"], bins[i + 1]["lo"]) for i in range(len(bins) - 1) if bins[i + 1]["lo"] - bins[i]["hi"] > 1e-9]
    rng = np.random.default_rng(seed)
    out = {}
    for b, (l, r) in R.items():
        zp, liso = ZP[b]
        tot = np.trapezoid(r, l)
        s = (l >= lo) & (l <= hi)
        cov = float(np.trapezoid(r[s], l[s]) / tot) if s.sum() > 1 else 0.0
        sig = l[r > 0.01 * r.max()]
        big_gap = any((g1 - g0) > 2 * 0.05 * g0 * 1.01 and g1 > sig.min() and g0 < sig.max() for g0, g1 in gaps)
        if cov < 0.90 or big_gap:
            out[b] = dict(tabulated=False, covered_fraction=round(cov, 3), gap_inside=bool(big_gap))
            continue
        den = np.trapezoid(zp * (liso / l[s]) ** 2 * r[s] / l[s], l[s])
        val = lambda ff: zp * np.trapezoid(np.interp(l[s], c, ff) * r[s] / l[s], l[s]) / den      # ff in mJy -> F in mJy
        v = float(val(f))
        mc = np.array([val(f + se * rng.standard_normal(len(f))) for _ in range(n_mc)])
        out[b] = dict(tabulated=True, covered_fraction=round(cov, 3), flux_mJy=v, stat_err_mJy=float(mc.std(ddof=1)),
                      err_mJy=float(np.hypot(mc.std(ddof=1), f_floor * abs(v))))
    return out


def as_points(a):
    """Per-picture fluxes as a 'spectrum' at the pictures' own wavelengths (secondary, unbinned synthetic photometry)."""
    o = np.argsort(a["lam"])
    return [dict(lo=float(a["lam"][i]), hi=float(a["lam"][i]), centre=float(a["lam"][i]), flux=float(a["flux"][i]),
                 stat_err=float(a["err"][i])) for i in o]


def mag_to_mJy(band, m, em):
    f = ZP[band][0] * 1e3 * 10 ** (-0.4 * m)
    return f, f * 0.4 * np.log(10) * em


def compare(syn, published):
    rows = []
    for band, m, em, src, note in published:
        fp, efp = mag_to_mJy(band, m, em)
        s = syn.get(band, {})
        row = dict(band=band, published_mag=m, published_mag_err=em, published_mJy=fp, published_err_mJy=efp, source=src, note=note,
                   covered_fraction=s.get("covered_fraction"))
        if s.get("tabulated"):
            ratio = s["flux_mJy"] / fp
            row.update(spherex_mJy=s["flux_mJy"], spherex_err_mJy=s["err_mJy"], ratio=ratio,
                       ratio_err=float(abs(ratio) * np.hypot(s["err_mJy"] / s["flux_mJy"], efp / fp)) if s["flux_mJy"] != 0 else None)
        else:
            row.update(spherex_mJy=None, ratio=None, reason="band not covered (protocol coverage rule)")
        rows.append(row)
    return rows


def sum_mags(m1, e1, m2, e2):
    f1, f2 = 10 ** (-0.4 * m1), 10 ** (-0.4 * m2)
    return float(-2.5 * np.log10(f1 + f2)), float(np.hypot(f1 * e1, f2 * e2) / (f1 + f2))


def rebin_table(l, f, edges_lo, edges_hi):
    """Mean F_nu of a published spectrum inside each bin (None if < 3 samples or coverage gap)."""
    out = []
    for lo, hi in zip(edges_lo, edges_hi):
        s = (l >= lo) & (l < hi) & np.isfinite(f)
        ok = s.sum() >= 3 and (l[s].max() - l[s].min()) > 0.6 * (hi - lo)
        out.append(float(np.mean(f[s])) if ok else None)
    return out


# ------------------------------------------------------------------ targets
def do_wise0855(R):
    from orrery.e7 import windows
    rows = [r for r in json.load(open(f"{P2}/wise0855_frames.json")) if r["flux"] is not None]
    n_all = len(rows)
    rows = [r for r in rows if r["chi2_local_dof"] <= 3]
    wl = np.array([r["wl"] for r in rows]); f = np.array([r["flux"] for r in rows]); e = np.array([r["err"] for r in rows])
    mjd = np.array([r["mjd"] for r in rows])
    nf = np.array([[np.nan if v is None else v for v in r["null_flux"]] for r in rows], float)
    win = windows(mjd)

    def one(sel, lo, hi):
        s = sel & (wl >= lo) & (wl < hi)
        if not s.any():
            return None
        w = 1 / e[s] ** 2
        F, sF = float(np.sum(w * f[s]) / w.sum()), float(1 / np.sqrt(w.sum()))
        nk = np.array([np.nansum(w * nf[s, k]) / np.sum(w[np.isfinite(nf[s, k])]) for k in range(nf.shape[1])])
        sd = float(np.std(nk, ddof=1))
        err = max(sF, sd)
        b = dict(lo=lo, hi=hi, centre=float(np.sum(w * wl[s]) / w.sum()), flux=F, formal_err=sF, empty_path_std=sd,
                 empty_path_mean=float(nk.mean()), empty_path_robust_std=float(1.4826 * np.median(np.abs(nk - np.median(nk)))),
                 err=err, stat_err=err, n=int(s.sum()), detected_3sigma=bool(F > 3 * err),
                 upper_limit_95=float(max(F, 0) + 1.645 * err), detectors=sorted(set(int(rows[i]["det"]) for i in np.flatnonzero(s))))
        return b
    allsel = np.ones(len(rows), bool)
    bins = [b for b in (one(allsel, lo, hi) for lo, hi in zip(W_EDGES[:-1], W_EDGES[1:])) if b]
    ctrl = [b for b in (one(allsel, lo, hi) for lo, hi in zip(W_CTRL[:-1], W_CTRL[1:])) if b]
    wide = {}
    for lo, hi in ((3.8, 4.4), (4.4, 5.0)):
        wide[f"{lo}-{hi}"] = dict(all_windows=one(allsel, lo, hi), per_window={int(w): one(win == w, lo, hi) for w in np.unique(win)})
    # JWST (Luhman et al. 2024, data behind Figs 3 and 4; F_nu in Jy)
    jw = {}
    for name, p in (("PRISM", f"{LIT}/wise0855/luhman2024_fig3_prism/dbf3.txt"), ("G395M", f"{LIT}/wise0855/luhman2024_fig4_g395m/dbf4.txt")):
        d = [l.split() for l in open(p).read().splitlines()[13:]]
        d = np.array([[float(x[0]), float(x[1])] for x in d if len(x) >= 2])
        jw[name] = dict(binned_mJy=rebin_table(d[:, 0], d[:, 1] * 1e3, [b["lo"] for b in bins], [b["hi"] for b in bins]),
                        ctrl_binned_mJy=rebin_table(d[:, 0], d[:, 1] * 1e3, [b["lo"] for b in ctrl], [b["hi"] for b in ctrl]),
                        wl=d[:, 0].tolist(), mJy=(d[:, 1] * 1e3).tolist())
    comp = {}
    for name in jw:
        j = np.array([np.nan if v is None else v for v in jw[name]["binned_mJy"]])
        F = np.array([b["flux"] for b in bins]); E = np.array([b["err"] for b in bins])
        det = np.array([b["detected_3sigma"] for b in bins]) & np.isfinite(j)
        ok = np.isfinite(j)
        ratio, er = F[det] / j[det], E[det] / j[det]
        wm, swm = wmean(ratio, er)
        comp[name] = dict(n_bins_detected=int(det.sum()), mean_ratio_spherex_over_jwst=wm, mean_ratio_err=swm,
                          chi2_detected_bins=float(np.sum(((F[det] - j[det]) / E[det]) ** 2)),
                          chi2_all_bins=float(np.sum(((F[ok] - j[ok]) / E[ok]) ** 2)), n_bins_all=int(ok.sum()),
                          per_bin_ratio=[None if not np.isfinite(x) else float(x) for x in F / j],
                          per_bin_ratio_err=[None if not np.isfinite(x) else float(x) for x in E / j])
    spec = sorted(ctrl + bins, key=lambda b: b["lo"])
    syn = synth(spec, FLOOR, R)
    pub = [("IRAC [4.5]", 13.87, 0.08, "Luhman et al. 2024, AJ 167, 5 (arXiv:2311.17316) Sec. 4.1: adopted from nine Spitzer epochs 2013-2015 (Luhman & Esplin 2016 Table 3)", "midpoint and range of the epochs"),
           ("IRAC [4.5]", 13.923, 0.016, "Kirkpatrick et al. 2021, ApJS 253, 7 (arXiv:2011.11616) Table A1 ch2", ""),
           ("WISE W2", 13.90, 0.16, "Luhman et al. 2024 Sec. 4.1: adopted from 18 NEOWISE epochs 2014-2022", ""),
           ("WISE W2", 14.016, 0.048, "Wright et al. 2014, AJ 148, 82 (arXiv:1405.7350) Table 2 (NEOWISE May 2014)", ""),
           ("IRAC [3.6]", 17.34, 0.02, "Luhman & Esplin 2016, AJ 152, 78 (arXiv:1605.06655) Table 3 (2015 Mar 10 monitoring mean; epochs range 17.28-17.44)", ""),
           ("IRAC [3.6]", 17.470, 0.066, "Kirkpatrick et al. 2021 Table A1 ch1", ""),
           ("WISE W1", 17.819, 0.327, "Wright et al. 2014 Table 2", "")]
    return dict(name="WISE 0855-0714", source="E4 per-picture cache; neighbours restored; local chi2/dof <= 3",
                n_pictures_measured=n_all, n_pictures_used=len(rows), windows=window_table(mjd, dict(zip(mjd, win))),
                bins=bins, control_bins=ctrl, wide_bands=wide,
                jwst=dict(source="Luhman et al. 2024, AJ 167, 5 (doi:10.3847/1538-3881/ad0b72), data behind Figs. 3 (PRISM) and 4 (G395M); F_nu in Jy; observed 2023 Apr 17; absolute scale tied by the authors to Spitzer [4.5] = 13.87 +- 0.08",
                          **{k: {q: v[q] for q in ("binned_mJy", "ctrl_binned_mJy")} for k, v in jw.items()}, comparison=comp),
                synthetic=syn, comparison=compare(syn, pub), band_tests=band_tests(bins, BANDS["wise0855"])), jw


def do_bright(key, name, phot, target_id, R, s_d, published, extra=None):
    from orrery.e7 import windows
    all_mjd = sorted(set(m for r in phot.values() for m in r["mjd"]))
    win_of = dict(zip(all_mjd, windows(np.array(all_mjd))))
    a, n_used, n_all = star_arrays(phot[target_id], s_d)
    cmp_stars = [star_arrays(r, s_d)[0] for k, r in phot.items() if k != target_id]
    fc, n_fc = f_cmp(cmp_stars, win_of, LOG_EDGES)
    f = max(FLOOR, fc or 0.0)
    res = dict(name=name, n_pictures=n_all, n_pictures_used=n_used, f_cmp=fc, f_cmp_n=n_fc, n_comparison_stars=len(cmp_stars),
               fractional_floor_used=f, bin_edges_um=LOG_EDGES)
    if n_used:
        allw, per = spectra_by_window(a, win_of, LOG_EDGES)
        add_total(allw, f)
        res.update(windows=window_table(a["mjd"], win_of), wavelength_range_um=[float(a["lam"].min()), float(a["lam"].max())],
                   pictures_per_detector={int(d): int((a["det"] == d).sum()) for d in np.unique(a["det"])},
                   bins=allw, per_window={w: add_total(b, f) for w, b in per.items()})
        syn = synth(allw, f, R)
        res.update(synthetic=syn, comparison=compare(syn, published), band_tests=band_tests(allw, BANDS[key]))
        res["secondary_unbinned_comparison"] = dict(
            rule="secondary, added after the first table (see Deviations): same synthetic photometry on the per-picture fluxes at their own wavelengths, no binning",
            rows=compare(synth(as_points(a), f, R), published))
    cmp_ = {k: star_arrays(r, s_d)[0] for k, r in phot.items() if k != target_id}
    return res, a, win_of, f, cmp_


def bin_():
    R = responses()
    s_d = json.load(open("data/findings/e7_gate0.json"))["s_d"]
    out = dict(experiment="E11", protocol="docs/protocols/E11.md", protocol_freeze=json.load(open(f"{P2}/protocol_freeze.json")),
               units=dict(wavelength="um", flux="mJy"),
               synthetic_photometry="Vega-relative: F = ZP * int(F_nu R dl/l) / int(ZP (l_iso/l)^2 R dl/l), R per photon; curves: 2MASS RSRs of Cohen et al. 2003 "
                                    "(speclite copy), WISE RSRs of Wright et al. 2010 (WISE docs), IRAC response (IRSA 080924ch?trans_full); zero points in ZP",
               zero_points_Jy_and_iso_um=ZP, targets={})
    out["targets"]["wise0855"], _ = do_wise0855(R)
    json.dump(out, open(OUT + ".tmp", "w"), indent=1)

    e = json.load(open("data/findings/e7_runs/e7_epsindib.json"))["phot"]
    J, eJ = sum_mags(12.29, 0.03, 13.23, 0.03); H, eH = sum_mags(11.50, 0.03, 13.19, 0.03); K, eK = sum_mags(11.38, 0.03, 13.49, 0.03)
    k10 = "King et al. 2010, A&A 510, A99 (arXiv:0911.3143) Table 4, synthetic 2MASS-system magnitudes of Ba and Bb, flux-summed here"
    pub = [("2MASS J", 11.908, 0.022, "2MASS PSC (II/246) 2MASS J22041052-5646577, epoch 1999-11-08", "unresolved pair"),
           ("2MASS H", 11.306, 0.024, "2MASS PSC", "unresolved pair"), ("2MASS Ks", 11.208, 0.021, "2MASS PSC", "unresolved pair"),
           ("2MASS J", J, eJ, k10, "sum of components"), ("2MASS H", H, eH, k10, "sum of components"), ("2MASS Ks", K, eK, k10, "sum of components"),
           ("WISE W1", 10.612, 0.024, "WISE All-Sky (II/311) J220415.65-564723.6", ""), ("WISE W1", 10.718, 0.023, "AllWISE (II/328) J220415.75-564724.4", "profile-fit chi2 = 21.5 (poor)"),
           ("WISE W2", 9.426, 0.020, "WISE All-Sky (II/311)", ""), ("WISE W2", 9.492, 0.020, "AllWISE (II/328)", "profile-fit chi2 = 19.7 (poor)"),
           ("IRAC [3.6]", 9.97, 0.01, "Patten et al. 2006, ApJ 651, 502 (VizieR J/ApJ/651/502, 'eps Ind BC')", "unresolved pair"),
           ("IRAC [4.5]", 9.44, 0.02, "Patten et al. 2006", "unresolved pair")]
    out["targets"]["epsindib"], _, _, f_e, cmp_e = do_bright("epsindib", "Epsilon Indi B (Ba+Bb, unresolved)", e, "target", R, s_d, pub)
    # comparison stars of the same pictures against their own 2MASS magnitudes (photometric-scale check; see Deviations)
    tm = {"6412596458823395328": ("2MASS J22043113-5646287", 11.746, 0.028, 11.351, 0.023, 11.277, 0.021),
          "6412596183945494144": ("2MASS J22041041-5647578", 14.081, 0.023, 13.497, 0.038, 13.361, 0.039),
          "6412595973491218944": ("2MASS J22041480-5648001", 14.375, 0.029, 13.839, 0.030, 13.778, 0.059)}
    chk = []
    for sid, (nm, J_, eJ_, H_, eH_, K_, eK_) in tm.items():
        pb = [("2MASS J", J_, eJ_, f"2MASS PSC (II/246) {nm}", ""), ("2MASS H", H_, eH_, "2MASS PSC", ""), ("2MASS Ks", K_, eK_, "2MASS PSC", "")]
        bins = add_total(bin_spectrum(cmp_e[sid]["lam"], cmp_e[sid]["flux"], cmp_e[sid]["err"], LOG_EDGES), f_e)
        chk.append(dict(gaia_dr3=sid, twomass=nm, binned=compare(synth(bins, f_e, R), pb), unbinned=compare(synth(as_points(cmp_e[sid]), f_e, R), pb),
                        bins=bins))
    out["targets"]["epsindib"]["comparison_star_check"] = dict(
        note="Gaia G < 16 stars measured in the same pictures with the same code, compared with their own 2MASS magnitudes (VizieR II/246 via the CfA mirror; files in data/phase2/e11/lit/epsindib/cmp_2mass/)",
        stars=chk)
    out["targets"]["epsindib"]["source"] = "data/findings/e7_runs/e7_epsindib.json (E7 forced photometry, 250 pictures)"
    json.dump(out, open(OUT + ".tmp", "w"), indent=1)

    b = json.load(open(f"{P2}/barnard_phot.json"))["phot"]
    pubb = [("2MASS J", 5.244, 0.020, "2MASS PSC (II/246) 2MASS J17574849+0441405, epoch 2000-05-28, Qflg AAA, Rflg 111", ""),
            ("2MASS H", 4.834, 0.034, "2MASS PSC", ""), ("2MASS Ks", 4.524, 0.020, "2MASS PSC", ""),
            ("2MASS J", 5.296, 0.017, "Mann et al. 2015, ApJ 804, 64 (VizieR J/ApJ/804/64), synthetic from their spectrum", ""),
            ("2MASS H", 4.801, 0.017, "Mann et al. 2015", ""), ("2MASS Ks", 4.533, 0.020, "Mann et al. 2015", ""),
            ("WISE W1", 4.386, 0.073, "WISE All-Sky (II/311) J175747.97+044321.9", "catalogue value saturated (w1sat = 0.178)"),
            ("WISE W2", 3.600, 0.062, "WISE All-Sky (II/311)", "catalogue value saturated (w2sat = 0.179)"),
            ("WISE W2", 3.658, 0.232, "AllWISE (II/328) J175747.94+044323.8", "catalogue value saturated (w2sat = 0.157)")]
    out["targets"]["barnard"], *_ = do_bright("barnard", "Barnard's Star", b, BARNARD_ID, R, s_d, pubb)
    tb = out["targets"]["barnard"]
    tb["source"] = "E7 photometry code on data/efields/e3x_barnard (239 pictures); frozen rule: no OVERFLOW/NONLINEAR pixel within 1.5 px"
    sat = np.array(b[BARNARD_ID]["sat"]); det = np.array(b[BARNARD_ID]["det"]); npx = np.array(b[BARNARD_ID]["npix"])
    tb["flagged_core_pictures_per_detector"] = {int(d): [int(sat[det == d].sum()), int((det == d).sum())] for d in np.unique(det)}
    tb["median_usable_pixels_in_psf_box"] = float(np.median(npx))
    # template: IRTF Spectral Library M4 V Gl 213 scaled to Barnard's 2MASS Ks
    t = np.loadtxt(f"{LIT}/barnard/irtf_library_m_dwarfs/M4V_Gl213.txt", comments="#")
    l, fl = t[:, 0], np.where(t[:, 1] > -900, t[:, 1], np.nan)
    fnu = fl * l ** 2 / 2.99792458e14 * 1e26 * 1e3 * 10 ** (-0.4 * (4.524 - 6.389))        # W m-2 um-1 -> mJy, scaled by Ks
    tb["template"] = dict(source="IRTF Spectral Library, M4 V Gl 213 (Rayner, Cushing & Vacca 2009, ApJS 185, 289), scaled by 10^(-0.4 (4.524 - 6.389)) to Barnard's 2MASS Ks; a different star",
                          bin_lo=LOG_EDGES[:-1], bin_hi=LOG_EDGES[1:], binned_mJy=rebin_table(l, fnu, LOG_EDGES[:-1], LOG_EDGES[1:]))
    # labelled diagnostic (NOT the frozen selection; see Deviations): flagged-core pictures with enough unmasked wing pixels
    diag = {}
    for tag, minpix in (("min18pix", 18), ("min4pix", 4)):
        global MIN_PIX
        keep, MIN_PIX = MIN_PIX, minpix
        a, n_used, _ = star_arrays(b[BARNARD_ID], s_d, need_unsat=False)
        MIN_PIX = keep
        bins = add_total(bin_spectrum(a["lam"], a["flux"], a["err"], LOG_EDGES), tb["fractional_floor_used"])
        syn = synth(bins, tb["fractional_floor_used"], R)
        diag[tag] = dict(rule=f"NOT per protocol: flagged-core pictures allowed, >= {minpix} unmasked pixels in the ePSF box (fit on the unmasked wing)",
                         n_pictures_used=n_used, pictures_per_detector={int(d): int((a["det"] == d).sum()) for d in np.unique(a["det"])},
                         wavelength_range_um=[float(a["lam"].min()), float(a["lam"].max())], bins=bins, synthetic=syn,
                         comparison=compare(syn, pubb))
    tb["diagnostic_flagged_core"] = diag
    for k, t in out["targets"].items():
        named = [r["label"] for r in t.get("band_tests", []) if r["named"]]
        t["bands_named"] = named
    w = out["targets"]["wise0855"]
    w["status"] = "descriptive: %d of %d bins above 3 x error; no band named" % (sum(b["detected_3sigma"] for b in w["bins"]), len(w["bins"]))
    out["targets"]["epsindib"]["status"] = "descriptive: spectrum 0.75-5.0 um from %d pictures" % out["targets"]["epsindib"]["n_pictures_used"]
    tb["status"] = ("FAILED: %d of %d pictures pass the frozen no-flagged-core rule; no wavelength range survives as a spectrum"
                    % (tb["n_pictures_used"], tb["n_pictures"]))
    json.dump(out, open(OUT + ".tmp", "w"), indent=1)
    os.replace(OUT + ".tmp", OUT)
    summarize(out)


def summarize(out):
    for k, t in out["targets"].items():
        print("==", k, "pictures used", t["n_pictures_used"])
        for st in t.get("comparison_star_check", {}).get("stars", []):
            for kind in ("binned", "unbinned"):
                print("  cmp star", st["gaia_dr3"][-6:], kind, ["%s %.3f+-%.3f" % (r["band"], r["ratio"], r["ratio_err"]) for r in st[kind] if r["ratio"] is not None])
        for r in t.get("secondary_unbinned_comparison", {}).get("rows", []):
            if r["ratio"] is not None:
                print("  unbinned %-11s %9.3f +- %6.3f ratio %.3f +- %.3f [%s]" % (r["band"], r["spherex_mJy"], r["spherex_err_mJy"], r["ratio"], r["ratio_err"], r["source"][:25]))
        for b in t.get("bins", []):
            print("  %.3f-%.3f  c=%.3f  F=%9.4f +- %7.4f  n=%d" % (b["lo"], b["hi"], b["centre"], b["flux"], b["err"], b["n"]))
        for r in t.get("comparison", []):
            if r["ratio"] is not None:
                print("  %-11s SPHEREx %9.4f +- %7.4f  published %9.4f +- %7.4f  ratio %.3f +- %.3f  [%s]" % (
                    r["band"], r["spherex_mJy"], r["spherex_err_mJy"], r["published_mJy"], r["published_err_mJy"], r["ratio"], r["ratio_err"], r["source"][:40]))
            else:
                print("  %-11s not covered (%.2f)  published %9.4f mJy" % (r["band"], r["covered_fraction"] or 0, r["published_mJy"]))
        for r in t.get("band_tests", []):
            print("  band", r["label"], r["status"], "D/sigma = %.2f depth %.2f named=%s" % (r["dip_over_err"], r["depth_fraction"], r["named"]) if r["status"] == "tested" else "")


# ------------------------------------------------------------------ figures
BLUE, ORANGE, AQUA, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#1a1a19", "#6b6a63"
BAND_HALF = {"2MASS J": 0.081, "2MASS H": 0.1255, "2MASS Ks": 0.131, "WISE W1": 0.33, "WISE W2": 0.52, "IRAC [3.6]": 0.375, "IRAC [4.5]": 0.5}
PLAIN = {"H2O": "water", "CH4": "methane", "CO": "carbon monoxide"}


def _axes(title, sub):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 15, "font.family": "DejaVu Sans", "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": INK, "text.color": INK})
    fig, ax = plt.subplots(figsize=(11, 6.6), dpi=150)
    fig.patch.set_facecolor("white"), ax.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color="#e6e5df", lw=0.8), ax.set_axisbelow(True)
    ax.set_xlabel("Colour of light (wavelength, µm)", fontsize=17)
    ax.set_ylabel("Brightness (mJy)", fontsize=17)
    fig.suptitle(title, x=0.07, ha="left", fontsize=19, fontweight="bold", y=0.975)
    ax.set_title(sub, loc="left", fontsize=13, color=MUTED, pad=10)
    return fig, ax, plt


def _published_points(ax, rows, pick, show_spherex=True):
    """Published filter brightness (aqua diamonds, bar = filter width) and SPHEREx through the same filter (open blue)."""
    first = True
    for band, src_key in pick:
        r = next(x for x in rows if x["band"] == band and src_key in x["source"])
        lam = ZP[band][1]
        ax.errorbar(lam, r["published_mJy"], yerr=r["published_err_mJy"], xerr=BAND_HALF[band], fmt="D", ms=8, color=AQUA, mec="white", mew=1.2,
                    ecolor=AQUA, elinewidth=2, capsize=0, zorder=7, label="Published brightness through a filter (bar = filter width)" if first else None)
        if show_spherex and r["ratio"] is not None:
            ax.errorbar(lam, r["spherex_mJy"], yerr=r["spherex_err_mJy"], fmt="D", ms=15, mfc="white", mec=BLUE, mew=2.2, ecolor=BLUE,
                        elinewidth=2, zorder=6, label="SPHEREx through the same filter" if first else None)
        first = False


def fig():
    o = json.load(open(OUT))["targets"]
    os.makedirs(FIGS, exist_ok=True)

    # ---- WISE 0855
    t = o["wise0855"]
    fig_, ax, plt = _axes("WISE 0855−0714: SPHEREx spectrum next to JWST",
                          "Coldest known brown dwarf; %d pictures combined along its path" % sum(b["n"] for b in t["bins"]))
    d = [l.split() for l in open(f"{LIT}/wise0855/luhman2024_fig3_prism/dbf3.txt").read().splitlines()[13:]]
    d = np.array([[float(x[0]), float(x[1]) * 1e3] for x in d if len(x) >= 2])
    s = (d[:, 0] > 3.7) & (d[:, 0] < 5.1)
    ax.plot(d[s, 0], d[s, 1], color=ORANGE, lw=1.2, alpha=0.45, zorder=2)
    c = [b["centre"] for b in t["bins"]]
    mid = [(b["lo"] + b["hi"]) / 2 for b in t["bins"]]
    ax.plot(mid, t["jwst"]["PRISM"]["binned_mJy"], "s-", color=ORANGE, ms=9, mec="white", mew=1.5, lw=2, zorder=3,
            label="JWST (Luhman et al. 2024), averaged in the same 0.1 µm steps")
    det = [b for b in t["bins"] if b["detected_3sigma"]]
    und = [b for b in t["bins"] if not b["detected_3sigma"]]
    ax.errorbar([b["centre"] for b in det], [b["flux"] for b in det], yerr=[b["err"] for b in det], fmt="o", ms=11, color=BLUE, mec="white",
                mew=1.5, ecolor=BLUE, elinewidth=2, capsize=4, zorder=6, label="SPHEREx (this work); bars = error from 24 empty paths")
    if und:
        ax.errorbar([b["centre"] for b in und], [b["flux"] for b in und], yerr=[b["err"] for b in und], fmt="o", ms=11, mfc="white", mec=BLUE,
                    mew=2.2, ecolor=BLUE, elinewidth=2, capsize=4, zorder=6, label="SPHEREx, below 3× its error")
    r = next(x for x in t["comparison"] if x["band"] == "IRAC [4.5]" and "Luhman et al. 2024" in x["source"])
    ax.errorbar(4.493, r["published_mJy"], yerr=r["published_err_mJy"], xerr=0.5, fmt="D", ms=10, color=AQUA, mec="white", mew=1.5, ecolor=AQUA,
                elinewidth=2, zorder=4, label="Spitzer 4.5 µm filter, 2013–2015 (bar = filter width)")
    ax.set_xlim(3.72, 5.06), ax.set_ylim(-0.03, 0.93)
    ax.legend(loc="upper left", fontsize=12.5, frameon=False)
    fig_.tight_layout(rect=(0, 0, 1, 0.96))
    fig_.savefig(f"{FIGS}/e11_wise0855.png", facecolor="white"), plt.close(fig_)

    # ---- Epsilon Indi B
    t = o["epsindib"]
    fig_, ax, plt = _axes("Epsilon Indi B (two brown dwarfs seen as one)",
                          "%d pictures, April 2025 – May 2026. Grey strips: absorption bands that passed our test" % t["n_pictures_used"])
    top = 78
    for bt in t["band_tests"]:
        if bt["named"]:
            lo, hi = bt["band_um"]
            ax.axvspan(lo, hi, color="#ecebe6", zorder=0)
            ax.text((lo + hi) / 2, top * 0.985, PLAIN[bt["label"].split()[0].split("+")[0]], rotation=90, ha="center", va="top", fontsize=12.5, color=MUTED)
    b = t["bins"]
    ax.plot([x["centre"] for x in b], [x["flux"] for x in b], "-", color=BLUE, lw=1.6, alpha=0.6, zorder=3)
    ax.errorbar([x["centre"] for x in b], [x["flux"] for x in b], yerr=[x["err"] for x in b], fmt="o", ms=7.5, color=BLUE, mec="white", mew=1.2,
                ecolor=BLUE, elinewidth=1.6, zorder=4, label="SPHEREx (this work)")
    _published_points(ax, t["comparison"], [("2MASS J", "II/246"), ("2MASS H", "2MASS PSC"), ("2MASS Ks", "2MASS PSC"), ("WISE W1", "All-Sky"),
                                            ("IRAC [3.6]", "Patten"), ("IRAC [4.5]", "Patten")])
    ax.set_xlim(0.7, 5.08), ax.set_ylim(-3, top)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=12.5, frameon=False, columnspacing=1.5)
    fig_.tight_layout(rect=(0, 0, 1, 0.96))
    fig_.savefig(f"{FIGS}/e11_epsindib.png", facecolor="white"), plt.close(fig_)

    # ---- Barnard's Star
    t = o["barnard"]
    fig_, ax, plt = _axes("Barnard's Star: too bright — no valid SPHEREx spectrum",
                          "The star overloads the detector in %d of %d pictures.\nGrey points use only the faint outskirts of its image and are not reliable."
                          % (sum(v[0] for v in t["flagged_core_pictures_per_detector"].values()), t["n_pictures"]))
    tl = np.array(t["template"]["bin_lo"]); th = np.array(t["template"]["bin_hi"])
    tv = np.array([np.nan if v is None else v for v in t["template"]["binned_mJy"]])
    ax.plot(np.sqrt(tl * th), tv, "s-", color=ORANGE, ms=7, mec="white", mew=1.2, lw=2, zorder=3,
            label="Published spectrum of a similar star (Gl 213, IRTF library), scaled to Barnard's Star")
    dg = t["diagnostic_flagged_core"]["min4pix"]["bins"]
    ax.errorbar([x["centre"] for x in dg], [x["flux"] for x in dg], yerr=[x["err"] for x in dg], fmt="o", ms=8, mfc="white", mec=MUTED, mew=1.8,
                ecolor=MUTED, elinewidth=1.4, zorder=4, label="SPHEREx, overloaded pictures (not valid)")
    for x in t.get("bins", []):
        ax.errorbar(x["centre"], x["flux"], yerr=x["err"], fmt="o", ms=12, color=BLUE, mec="white", mew=1.5, ecolor=BLUE, elinewidth=2, zorder=6,
                    label="SPHEREx, the one picture that passed our rule")
    _published_points(ax, t["comparison"], [("2MASS J", "II/246"), ("2MASS H", "2MASS PSC"), ("2MASS Ks", "2MASS PSC")], show_spherex=False)
    ax.set_xlim(0.7, 5.08), ax.set_ylim(0, 21000)
    ax.legend(loc="upper right", fontsize=12, frameon=True, facecolor="white", edgecolor="none", framealpha=0.9)
    fig_.tight_layout(rect=(0, 0, 1, 0.96))
    fig_.savefig(f"{FIGS}/e11_barnard.png", facecolor="white"), plt.close(fig_)
    print("figures written")


if __name__ == "__main__":
    stage = sys.argv[1]
    os.makedirs(P2, exist_ok=True)
    {"w0855": stage_w0855, "barnard": stage_barnard, "bin": bin_, "fig": fig}[stage]()
