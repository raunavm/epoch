"""E18-A: label-free, injection-validated, conformally calibrated anomaly search on real SPHEREx NEP spectra.

Run with the framework Python that has scikit-learn / torch / umap (the repo .venv has none of them):
  PY=/Library/Frameworks/Python.framework/Versions/3.13/bin/python3
  $PY scripts/run_e18a.py build      # per-source 102-channel mean spectra -> data/phase2/e18a/spectra.npz
  $PY scripts/run_e18a.py pilot      # injection-only recall (no real-source ranking is printed or saved)
  $PY scripts/run_e18a.py run        # models, conformal p-values, injections, held-out classes -> scores.parquet
  $PY scripts/run_e18a.py triage     # artefact vetoes + catalogue cross-match of flags
  $PY scripts/run_e18a.py figs       # figures + data/findings/e18a.json
Choices are frozen in docs/protocols/E18.md, section "E18-A frozen choices".
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "1"
import sys, glob, json, time
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
E18 = "data/phase2/e18"
OUT = "data/phase2/e18a"
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- frozen constants (see protocol)
DET_RANGES = [(0.74, 1.12), (1.10, 1.65), (1.64, 2.42), (2.42, 3.82), (3.81, 4.42), (4.41, 5.00)]
EDGES = [np.geomspace(lo, hi, 18) for lo, hi in DET_RANGES]
CEN = np.concatenate([np.sqrt(e[:-1] * e[1:]) for e in EDGES])          # 102 channel centres
WID = np.concatenate([np.diff(e) for e in EDGES])
NCH = 102
BAD_BITS = sum(1 << b for b in (2, 6, 9, 10, 11, 15, 17, 26, 27, 28, 29))   # = orrery.data.BAD_BITS
DROP_BITS = BAD_BITS | (1 << 1) | (1 << 0) | (1 << 20) | (1 << 22) | (1 << 23) | (1 << 24) | (1 << 25)
SYS_FRAC = 0.03          # ES v2.1 Eq. 19 irreducible per-measurement scatter
CH_FLOOR = 0.01          # channel-level systematic floor used in the AE chi2 score
MAX_NMASK, MAX_CHI2, MIN_EDGE = 4, 20.0, 5.0
CLIP = 5.0
MIN_N = 3
MIN_VALID_FRAC = 0.70
CHAN_COVER = 0.50        # a channel is kept if valid in >= 50% of reported in-region sources
SNR_FLOOR = 5.0          # model-inclusion floor (median per-channel SNR)
ALPHA = 0.005            # conformal level on the rank-averaged score
AE_TOPK = 5              # AE score = mean of the 5 largest per-channel chi2
ALPHA_M = 0.02           # per-model level for the >= 2-of-3 agreement rule
SEED = 20261115
SNR_BINS = [5, 10, 20, 50, 100, 300, 1e9]
SNR_LAB = ["5-10", "10-20", "20-50", "50-100", "100-300", ">300"]
PY_NOTE = "framework Python 3.13 (scikit-learn 1.7.2, torch 2.9.1, umap 0.5.12)"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def chan_index(det, wl):
    c = np.full(len(wl), -1, np.int16)
    for d in range(1, 7):
        m = det == d
        i = np.searchsorted(EDGES[d - 1], wl[m]) - 1
        ok = (i >= 0) & (i < 17)
        c[np.flatnonzero(m)[ok]] = (d - 1) * 17 + i[ok]
    return c


def load_clean(srcs=None):
    """All per-picture rows passing the frozen per-picture cuts. Returns a DataFrame of compact columns."""
    cols = ["src", "flux_mJy", "flux_err_mJy", "wl_um", "bw_um", "detector", "obs_id", "mjd", "x", "y",
            "flags_core", "n_masked_box", "chi2dof_local", "edge_px"]
    anom = pd.read_csv("data/ref/spherex_exposure_anomaly_log_20250810.csv")
    bad_pairs = set()
    for d in range(1, 7):
        for o in anom.OBSID[anom[f"BAND_{d}_AFFECTED"] > 0]:
            bad_pairs.add((o, d))
    parts = []
    files = sorted(glob.glob(f"{E18}/phot/part-*.parquet"))
    for f in files:
        t = pd.read_parquet(f, columns=cols)
        if srcs is not None:
            t = t[t.src.isin(srcs)]
        ok = np.isfinite(t.flux_mJy.values) & np.isfinite(t.flux_err_mJy.values) & (t.flux_err_mJy.values > 0)
        fc = t.flags_core.values
        ok &= (fc >= 0) & ((fc & DROP_BITS) == 0)
        ok &= (t.n_masked_box.values <= MAX_NMASK) & (t.chi2dof_local.fillna(1e9).values <= MAX_CHI2)
        ok &= t.edge_px.values >= MIN_EDGE
        if bad_pairs:
            ok &= ~np.array([(o, d) in bad_pairs for o, d in zip(t.obs_id.values, t.detector.values)]) \
                if t.obs_id.isin(anom.OBSID).any() else True
        t = t[ok]
        ch = chan_index(t.detector.values, t.wl_um.values)
        t = t.assign(ch=ch)[ch >= 0]
        parts.append(pd.DataFrame(dict(
            src=t.src.values.astype(np.int32), ch=t.ch.values.astype(np.int16), f=t.flux_mJy.values.astype(np.float32),
            e=t.flux_err_mJy.values.astype(np.float32), wl=t.wl_um.values.astype(np.float32),
            bw=t.bw_um.values.astype(np.float32), det=t.detector.values.astype(np.int8),
            half=(np.floor((t.mjd.values - 60789) / 10).astype(int) % 2).astype(np.int8),
            x=t.x.values.astype(np.float32), y=t.y.values.astype(np.float32),
            obs=(t.obs_id.values if srcs is not None else
                 (len(parts) * 100000 + pd.factorize(t.obs_id.values)[0]).astype(np.int64)),
            fl=t.flags_core.values.astype(np.int64))))
    d = pd.concat(parts, ignore_index=True)
    if srcs is not None:
        d["obs"] = d.obs.astype("category")
    return d, len(files)


def aggregate(src, ch, f, e, nsrc):
    """Clipped inverse-variance channel means. Returns F, E, N arrays (nsrc x NCH).
    err_i^2 = formal^2 + (SYS_FRAC * |channel median|)^2; clip |f - med| > CLIP * 1.4826 MAD (n >= 5);
    error of the mean inflated by sqrt(reduced chi2) when > 1 (error from scatter)."""
    key = src.astype(np.int64) * NCH + ch
    g = pd.DataFrame(dict(k=key, f=f.astype(np.float64)))
    med = g.groupby("k").f.median()
    m = med.reindex(key).values
    ad = np.abs(f - m)
    mad = pd.Series(ad).groupby(key).median()
    cnt = pd.Series(ad).groupby(key).size()
    s = 1.4826 * mad.reindex(key).values
    n0 = cnt.reindex(key).values
    keep = ~((n0 >= 5) & (s > 0) & (ad > CLIP * s))
    et2 = e.astype(np.float64) ** 2 + (SYS_FRAC * np.abs(m)) ** 2
    w = np.where(keep, 1.0 / et2, 0.0)
    L = nsrc * NCH
    sw = np.bincount(key, w, L)
    swf = np.bincount(key, w * f, L)
    n = np.bincount(key, keep.astype(float), L)
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = swf / sw
        r2 = np.bincount(key, w * (f - mu[key]) ** 2, L)
        chi2r = r2 / np.maximum(n - 1, 1)
        err = np.sqrt(1 / sw) * np.sqrt(np.maximum(chi2r, 1.0))
    mu[n < MIN_N] = np.nan
    err[n < MIN_N] = np.nan
    return mu.reshape(nsrc, NCH), err.reshape(nsrc, NCH), n.reshape(nsrc, NCH)


def features(F, E, N, keep):
    """Normalised spectra. X: spectrum / broadband median (masked channels interpolated in log-lambda); S: errors;
    M: validity; Z: tree-model features = [18 coarse log-continuum values (3 per detector),
    high-pass residuals (X - running median of 7 channels within the detector) / S, clipped to +-30]."""
    valid = (N >= MIN_N) & np.isfinite(F) & np.isfinite(E) & keep[None, :]
    Fv = np.where(valid, F, np.nan)
    with np.errstate(all="ignore"):
        med = np.nanmedian(Fv, 1)
        snr = np.nanmedian(np.where(valid, F / E, np.nan), 1)
    frac = valid[:, keep].sum(1) / keep.sum()
    with np.errstate(all="ignore"):
        X = F / med[:, None]
        S = E / np.abs(med)[:, None]
    lx = np.log(CEN)
    Xi = np.zeros_like(X)
    for i in range(len(X)):
        v = valid[i]
        if v.sum() >= 2:
            Xi[i] = np.interp(lx, lx[v], X[i, v])
    sm = np.zeros_like(Xi)
    C = np.zeros((len(X), 18))
    for dd in range(6):
        blk = Xi[:, dd * 17:(dd + 1) * 17]
        for j in range(17):
            sm[:, dd * 17 + j] = np.median(blk[:, max(0, j - 3):j + 4], 1)
        for g, (a, b) in enumerate(((0, 6), (6, 12), (12, 17))):
            C[:, dd * 3 + g] = np.log10(np.clip(blk[:, a:b].mean(1), 0.01, None))
    with np.errstate(all="ignore"):
        R = np.where(valid, np.clip((Xi - sm) / S, -30, 30), 0.0)
    Z = np.hstack([C, R[:, keep]])
    return dict(X=Xi[:, keep], S=np.where(valid, S, np.nan)[:, keep], M=valid[:, keep], Z=Z, med=med, snr=snr, frac=frac)


# ---------------------------------------------------------------- models
class Models:
    """The three scorers, fitted once on a training fold; .score(F) scores any feature dict."""

    def __init__(self, Ftr, seed):
        import torch
        from sklearn.ensemble import RandomForestClassifier, IsolationForest
        torch.set_num_threads(3)
        Xtr, Mtr, Ztr = Ftr["X"], Ftr["M"], Ftr["Z"]
        rng = np.random.default_rng(seed)
        # (i) Baron & Poznanski 2017 unsupervised RF: real vs column-shuffled synthetic
        syn = np.column_stack([rng.permutation(Ztr[:, j]) for j in range(Ztr.shape[1])])
        self.rf = RandomForestClassifier(n_estimators=500, max_features="sqrt", min_samples_leaf=1, n_jobs=4,
                                         random_state=seed).fit(np.vstack([Ztr, syn]),
                                                                np.r_[np.ones(len(Ztr)), np.zeros(len(syn))])
        Ltr = self.rf.apply(Ztr)
        self.rf_real, self.rf_cnt = [], []
        for t, est in enumerate(self.rf.estimators_):
            v = est.tree_.value[:, 0, :]
            self.rf_real.append(v[:, 1] > v[:, 0])
            self.rf_cnt.append(np.bincount(Ltr[:, t], minlength=est.tree_.node_count))
        self.ntr = len(Xtr)
        # (iii) isolation forest
        self.iso = IsolationForest(n_estimators=500, random_state=seed, n_jobs=4).fit(Ztr)
        # (ii) error-weighted autoencoder (masked MSE in standardised units for training; chi2 score)
        torch.manual_seed(seed)
        self.mu, self.sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        p = Xtr.shape[1]
        nn = torch.nn
        self.net = nn.Sequential(nn.Linear(p, 64), nn.ELU(), nn.Linear(64, 32), nn.ELU(), nn.Linear(32, 6),
                                 nn.Linear(6, 32), nn.ELU(), nn.Linear(32, 64), nn.ELU(), nn.Linear(64, p))
        z = torch.tensor((Xtr - self.mu) / self.sd, dtype=torch.float32)
        mk = torch.tensor(Mtr, dtype=torch.float32)
        opt = torch.optim.Adam(self.net.parameters(), lr=3e-3, weight_decay=1e-5)
        for ep in range(800):
            opt.zero_grad()
            loss = ((self.net(z) - z) ** 2 * mk).sum() / mk.sum()
            loss.backward()
            opt.step()

    def score(self, F):
        import torch
        X, Sg, M, Zf = F["X"], F["S"], F["M"], F["Z"]
        L = self.rf.apply(Zf)
        w = np.zeros(len(X))
        for t in range(L.shape[1]):
            lt = L[:, t]
            w += np.where(self.rf_real[t][lt], self.rf_cnt[t][lt], 0)
        s = {"rf": 1.0 - w / (L.shape[1] * self.ntr), "if": -self.iso.score_samples(Zf)}
        with torch.no_grad():
            zh = self.net(torch.tensor((X - self.mu) / self.sd, dtype=torch.float32)).numpy()
        xh = zh * self.sd + self.mu
        sig2 = np.where(M, Sg, 1.0) ** 2 + (CH_FLOOR * np.abs(xh)) ** 2
        r2 = np.where(M, (X - xh) ** 2 / sig2, 0.0)
        s["ae"] = np.sort(r2, 1)[:, -AE_TOPK:].mean(1)          # mean of the AE_TOPK largest channel chi2
        return s, xh


MODELS = ("rf", "ae", "if")


def pvals(s_ref, s_new, exclude_self=True):
    """Conformal p of each s_new against reference scores s_ref (same fold-bin).
    exclude_self: s_new are the reference points themselves (leave-one-out)."""
    srt = np.sort(s_ref)
    ge = len(srt) - np.searchsorted(srt, s_new, side="left")
    if exclude_self:
        return ge / len(srt)                     # (1 + #{j != i: s_j >= s_i}) / n
    return (1 + ge) / (len(srt) + 1)


def mondrian_bins(snr, edges):
    return np.searchsorted(edges, snr, side="right") - 1


# ---------------------------------------------------------------- injection templates
def lines_profile(lam, bw, W_um, centres=(1.282, 1.875, 2.166, 4.052)):
    """Fractional excess (relative to continuum) seen by a Gaussian channel response of FWHM bw: unresolved lines."""
    sig = bw / 2.3548
    out = np.zeros_like(lam)
    for c in centres:
        out += W_um / (sig * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((lam - c) / sig) ** 2)
    return out


def absorb_profile(lam, bw, bands):
    """bands: list of (centre, FWHM, depth); Gaussian absorbed fraction convolved with the channel response."""
    sr = bw / 2.3548
    out = np.zeros_like(lam)
    for c, fw, d in bands:
        sl = fw / 2.3548
        st = np.sqrt(sl ** 2 + sr ** 2)
        out -= d * sl / st * np.exp(-0.5 * ((lam - c) / st) ** 2)
    return out


def bb_nu(lam_um, T):
    x = 14387.77 / (lam_um * T)
    return lam_um ** -3 / np.expm1(x)


def carbon_template(lam):
    bands = [(1.10, 0.06, 0.25), (1.42, 0.08, 0.25), (1.77, 0.06, 0.30), (3.10, 0.30, 0.45), (3.80, 0.15, 0.20)]
    return bb_nu(lam, 2600.0) * (1 + absorb_profile(lam, np.full_like(lam, 1e-4), bands))


def redqso_template(lam):
    return lam ** 2.0


def tdwarf_template():
    """Channel-binned SPHEREx spectra of two real T dwarfs from E15 (per-picture forced photometry), coarse-binned
    (3 channels) and interpolated in log-lambda; returns function lam -> F_nu (arbitrary units)."""
    out = {}
    for obj in ("2MASS_0559-1404", "SDSS_0741p2351"):
        rows = [json.load(open(p)) for p in glob.glob(f"data/phase2/e15/forced/{obj}/*.json")]
        t = pd.DataFrame(rows)
        t = t[t.on_frame & np.isfinite(t.flux_mJy) & (t.flux_err_mJy > 0)]
        t = t[(t.flags_src.fillna(0).astype(np.int64) & BAD_BITS) == 0]
        ch = chan_index(t.detector.values, t.wl_um.values)
        t = t[ch >= 0].assign(c3=ch[ch >= 0] // 3)
        g = t.groupby("c3").apply(lambda q: pd.Series(dict(
            f=np.average(q.flux_mJy, weights=q.flux_err_mJy ** -2), lw=np.log(q.wl_um).mean(), n=len(q))),
            include_groups=False)
        g = g[g.n >= 2].sort_values("lw")
        out[obj] = (g.lw.values, np.clip(g.f.values, 0.02 * g.f.max(), None))
    return out


def make_tdwarf_fn(tpl):
    lw, f = tpl
    return lambda lam: np.exp(np.interp(np.log(lam), lw, np.log(f)))


INJ_CLASSES = {
    # name: (kind, {amp_label: amp}, plain label)
    "hlines": ("lines", {"weak": 30e-4, "medium": 100e-4, "strong": 300e-4},
               "H emission lines (Pa-beta, Pa-alpha, Br-gamma, Br-alpha)"),
    "ices": ("absorb", {"weak": 0.3, "medium": 1.0, "strong": 3.0}, "water (3.05 um) + CO2 (4.27 um) ice absorption"),
    "tdwarf": ("shape", {"weak": 0.1, "medium": 0.3, "strong": 1.0}, "T dwarf (methane) spectrum, real SPHEREx template"),
    "redqso": ("shape", {"weak": 0.1, "medium": 0.3, "strong": 1.0}, "red / obscured quasar continuum F_nu ~ lambda^2"),
    "carbon": ("shape", {"weak": 0.1, "medium": 0.3, "strong": 1.0}, "carbon star (2600 K + CN, C2, HCN/C2H2 bands)"),
}
SHORT = {"hlines": "hydrogen emission lines", "ices": "water + CO2 ice absorption", "tdwarf": "T dwarf (methane)",
         "redqso": "red / obscured quasar", "carbon": "carbon star"}
AMP_TXT = {"weak": "weak fakes: lines 30 A, ice tau 0.3, 10% template blend",
           "medium": "medium fakes: lines 100 A, ice tau 1, 30% blend",
           "strong": "strong fakes: lines 300 A, ice tau 3, full template"}
REPORT_RECALL = 0.25     # reporting floor: lowest host-SNR bin edge with class-averaged medium recall >= 0.25


def reporting_floor(inj):
    """Lowest edge of SNR_BINS from which the class-averaged 'medium' recall is >= REPORT_RECALL in that bin and
    every brighter bin with >= 5 hosts. Returns inf if never reached (then no flag is reportable)."""
    m = inj[(inj.amp_label == "medium")].assign(sb=pd.cut(inj.host_snr, SNR_BINS, labels=False))
    r = m.groupby(["sb", "cls"]).flag.mean().groupby("sb").mean()
    n = m.groupby("sb").size() / m.cls.nunique()
    good = [b for b in r.index if n[b] >= 5]
    for b in sorted(good):
        if all(r[c] >= REPORT_RECALL for c in good if c >= b):
            return float(SNR_BINS[b])
    return float("inf")


def inject_rows(rows, Fhost, kind, amp, shape_fn=None):
    """rows: per-picture rows of ONE host; Fhost: host channel spectrum (NCH) for the smooth continuum.
    Returns modified fluxes. Host noise, masks, epochs, errors are kept (additive modification)."""
    lam, bw = rows.wl.values.astype(float), rows.bw.values.astype(float)
    v = np.isfinite(Fhost)
    cont = np.interp(np.log(lam), np.log(CEN[v]), Fhost[v])
    f = rows.f.values.astype(float)
    if kind == "lines":
        return f + cont * lines_profile(lam, bw, amp)
    if kind == "absorb":
        bands = [(3.05, 0.35, 1 - np.exp(-amp)), (4.27, 0.03, 1 - np.exp(-2 * amp))]
        return f + cont * absorb_profile(lam, bw, bands)
    # shape replacement: blend host continuum with template scaled to the host's broadband median
    tc = shape_fn(CEN[v])
    k = np.median(Fhost[v]) / np.median(tc)
    new = (1 - amp) * cont + amp * k * shape_fn(lam)
    return f + (new - cont)


# ---------------------------------------------------------------- catalogue labels (held-out known rare classes)
RARE_SIMBAD = {"QSO", "AGN", "Sy1", "Sy2", "SyG", "Sy*", "BLL", "Bla", "Q?", "AG?", "PN", "WD*", "WD?", "Pe*", "EB*",
               "RR*", "BY*", "C*", "Em*", "Be*", "YSO", "Y*O", "Or*", "TT*", "BD*", "LP*", "Mi*", "AB*", "S*", "CV*",
               "No*", "WR*", "EmG", "Ce*", "dS*", "RS*", "El*", "HB*", "sdB", "Er*", "Ae*", "Y*?", "Ir*", "s*b"}


def sky_match(ra1, de1, ra2, de2, r_arcsec):
    """index into set 2 of nearest match within r for each of set 1 (-1 if none); small field, flat approx."""
    x1, y1 = (ra1 - 270.0) * np.cos(np.radians(66.56)) * 3600, (de1 - 66.56) * 3600
    x2, y2 = (ra2 - 270.0) * np.cos(np.radians(66.56)) * 3600, (de2 - 66.56) * 3600
    from scipy.spatial import cKDTree
    d, i = cKDTree(np.c_[x2, y2]).query(np.c_[x1, y1], distance_upper_bound=r_arcsec)
    return np.where(np.isfinite(d), i, -1), d


def labels(S):
    """Per-source known-class labels from the catalogues (no SPHEREx data used)."""
    L = pd.DataFrame(index=S.index)
    sb = pd.read_parquet(f"{OUT}/simbad_raw.parquet")
    i, d = sky_match(S.ra.values, S.dec.values, sb.ra.values, sb.dec.values, 3.0)
    L["simbad_id"] = np.where(i >= 0, sb.main_id.values[np.clip(i, 0, None)], None)
    L["simbad_type"] = np.where(i >= 0, sb.otype.values[np.clip(i, 0, None)], None)
    L["simbad_sp"] = np.where(i >= 0, sb.sp_type.astype(str).values[np.clip(i, 0, None)], None)
    vs = pd.read_parquet(f"{OUT}/vsx_raw.parquet")
    i, d = sky_match(S.ra.values, S.dec.values, vs.RAJ2000.values.astype(float), vs.DEJ2000.values.astype(float), 3.0)
    L["vsx_name"] = np.where(i >= 0, vs.Name.values[np.clip(i, 0, None)], None)
    L["vsx_type"] = np.where(i >= 0, vs.Type.values[np.clip(i, 0, None)], None)
    cls = pd.Series(None, index=S.index, dtype=object)

    def put(mask, name):
        m = mask & cls.isna()
        cls[m] = name
    st = L.simbad_type.fillna("")
    put(S.milliquas_name.notna().values | st.isin({"QSO", "AGN", "Sy1", "Sy2", "SyG", "BLL", "Bla", "Q?", "AG?"}).values
        | (S.classprob_quasar.fillna(0) > 0.5).values, "quasar / AGN")
    put(st.isin({"PN"}).values, "planetary nebula")
    put(st.isin({"WD*", "WD?"}).values, "white dwarf")
    put(st.isin({"C*", "Em*", "Be*", "YSO", "Y*O", "Or*", "TT*", "BD*", "LP*", "Mi*", "AB*", "S*", "CV*", "No*", "WR*",
                 "Pe*", "Ae*", "Y*?", "HB*", "sdB"}).values, "peculiar / emission-line star")
    var = (S.vari_class.notna() & (S.vari_class != "SOLAR_LIKE")).values | S.neowise_var_kim2026.fillna(False).values \
        | st.isin({"EB*", "RR*", "BY*", "dS*", "RS*", "El*", "Ce*", "Ir*"}).values \
        | (L.vsx_type.notna() & (L.vsx_type.fillna("") != "ROT")).values
    put(var, "known variable star")
    L["known_class"] = cls
    return L


def population(spec, S, L):
    feat = features(spec["F"], spec["E"], spec["N"], spec["keep"])
    rep = (S.in_region & S.is_primary).values
    usable = rep & (feat["frac"] >= MIN_VALID_FRAC) & (feat["med"] > 0) & np.isfinite(feat["snr"])
    held = usable & L.known_class.notna().values & (feat["snr"] >= SNR_FLOOR)
    pop = usable & (feat["snr"] >= SNR_FLOOR) & ~held
    fold = np.random.default_rng(SEED).integers(0, 2, len(S))
    n = pop.sum()
    nb = 3 if n / 6 >= 300 else 2
    edges = np.r_[SNR_FLOOR, np.quantile(feat["snr"][pop], np.arange(1, nb) / nb), np.inf]
    return feat, dict(rep=rep, usable=usable, held=held, pop=pop, fold=fold, edges=edges)


def build():
    S = pd.read_parquet(f"{E18}/sources.parquet")
    d, nparts = load_clean()
    log("clean rows", len(d), "parts", nparts)
    F, E, N = aggregate(d.src.values, d.ch.values, d.f.values, d.e.values, len(S))
    rep = (S.in_region & S.is_primary).values
    cover = ((N[rep] >= MIN_N) & np.isfinite(F[rep])).mean(0)
    keep = cover >= CHAN_COVER
    np.savez_compressed(f"{OUT}/spectra.npz", F=F, E=E, N=N, keep=keep, cover=cover, nparts=nparts, nrows=len(d),
                        npics=d.obs.nunique())
    log("kept channels", keep.sum(), "dropped", np.round(CEN[~keep], 3).tolist())
    return d


def load_spec():
    z = np.load(f"{OUT}/spectra.npz")
    return {k: z[k] for k in z.files}


def rank_combine(P):
    """Rank-average over the two most anomalous models: S = 1 - mean of the two smallest per-model conformal
    p-values (a calibrated form of '>= 2 of 3 models agree'; S is then itself conformally calibrated)."""
    A = np.sort(np.vstack([P[m] for m in MODELS]), 0)
    return 1 - A[:2].mean(0)


def run(pilot=False, n_hosts=120):
    S = pd.read_parquet(f"{E18}/sources.parquet")
    L = labels(S)
    d = build()
    spec = load_spec()
    feat, pp = population(spec, S, L)
    keep = spec["keep"]
    edges = pp["edges"]
    log("usable", pp["usable"].sum(), "pop", pp["pop"].sum(), "held-out", pp["held"].sum(), "bin edges", edges)
    sub = lambda m: {k: feat[k][m] for k in ("X", "S", "M", "Z")}
    res = pd.DataFrame(index=S.index)
    res["id"], res["snr"], res["frac_valid"], res["fold"] = S.id.values, feat["snr"], feat["frac"], pp["fold"]
    res["set"] = np.where(pp["pop"], "population", np.where(pp["held"], "held_out", "not_scored"))
    res["bin"] = mondrian_bins(feat["snr"], edges)
    for m in MODELS:
        res[f"s_{m}"] = np.nan
        res[f"p_{m}"] = np.nan
    res["S"], res["p_comb"] = np.nan, np.nan
    XH = np.full(feat["X"].shape, np.nan)
    # host per-picture rows indexed by src
    order = np.argsort(d.src.values, kind="stable")
    d = d.iloc[order].reset_index(drop=True)
    st = np.searchsorted(d.src.values, np.arange(len(S) + 1))
    tdw = tdwarf_template()
    shape_fns = {"tdwarf": make_tdwarf_fn(tdw["2MASS_0559-1404"]), "redqso": redqso_template, "carbon": carbon_template}
    inj = []
    rng = np.random.default_rng(SEED + 1)
    for k in (0, 1):
        tr = pp["pop"] & (pp["fold"] == k)
        te = pp["pop"] & (pp["fold"] == 1 - k)
        ho = pp["held"] & (pp["fold"] == 1 - k)
        log(f"fold {k}: train {tr.sum()}  test {te.sum()}  held-out {ho.sum()}")
        mdl = Models(sub(tr), SEED + k)
        log(f"fold {k}: models fitted")
        tgt = te | ho
        s, xh = mdl.score(sub(tgt))
        XH[tgt] = xh
        idx_t = np.flatnonzero(tgt)
        for m in MODELS:
            res.loc[idx_t, f"s_{m}"] = s[m]
        # per-model conformal p within (fold, bin), reference = population of the test fold
        for b in range(len(edges) - 1):
            ref = te & (res.bin.values == b)
            q = tgt & (res.bin.values == b)
            for m in MODELS:
                sr = res[f"s_{m}"].values[ref]
                res.loc[ref, f"p_{m}"] = pvals(sr, sr, True)
                hq = q & ~ref
                if hq.any():
                    res.loc[hq, f"p_{m}"] = pvals(sr, res[f"s_{m}"].values[hq], False)
            Sref = rank_combine({m: res[f"p_{m}"].values[ref] for m in MODELS})
            res.loc[ref, "S"] = Sref
            res.loc[ref, "p_comb"] = pvals(Sref, Sref, True)
            hq = q & ~ref
            if hq.any():
                Sh = rank_combine({m: res[f"p_{m}"].values[hq] for m in MODELS})
                res.loc[hq, "S"] = Sh
                res.loc[hq, "p_comb"] = pvals(Sref, Sh, False)
        # ---- injections on hosts from the test fold (same models, same calibration set)
        cand = np.flatnonzero(te)
        ls = np.log10(feat["snr"][cand])
        h, be = np.histogram(ls, 12)
        wgt = 1.0 / h[np.clip(np.searchsorted(be, ls, side="right") - 1, 0, 11)]
        hosts = rng.choice(cand, size=min(n_hosts, len(cand)), replace=False, p=wgt / wgt.sum())
        classes = [("null", "none", "null", 0.0)] + [(c, kind, a, v) for c, (kind, amps, _) in INJ_CLASSES.items()
                                                     for a, v in amps.items()]
        for cname, kind, alab, amp in classes:
            fs, srcs, chs, es = [], [], [], []
            for j, hsrc in enumerate(hosts):
                r = d.iloc[st[hsrc]:st[hsrc + 1]]
                fnew = r.f.values if kind == "none" else inject_rows(r, spec["F"][hsrc], kind, amp, shape_fns.get(cname))
                fs.append(fnew); srcs.append(np.full(len(r), j)); chs.append(r.ch.values); es.append(r.e.values)
            Fi, Ei, Ni = aggregate(np.concatenate(srcs), np.concatenate(chs), np.concatenate(fs).astype(np.float32),
                                   np.concatenate(es), len(hosts))
            fi = features(Fi, Ei, Ni, keep)
            ok = (fi["frac"] >= MIN_VALID_FRAC) & (fi["med"] > 0) & (fi["snr"] >= SNR_FLOOR)
            si, _ = mdl.score({kk: fi[kk] for kk in ("X", "S", "M", "Z")})
            bins = mondrian_bins(fi["snr"], edges)
            for j, hsrc in enumerate(hosts):
                rec = dict(fold=1 - k, host=int(hsrc), host_snr=float(feat["snr"][hsrc]), snr=float(fi["snr"][j]),
                           cls=cname, amp_label=alab, amp=amp, ok=bool(ok[j]))
                if ok[j] and 0 <= bins[j] < len(edges) - 1:
                    ref = te & (res.bin.values == bins[j])
                    ref[hsrc] = False
                    pm = {}
                    for m in MODELS:
                        pm[m] = float(pvals(res[f"s_{m}"].values[ref], np.array([si[m][j]]), False)[0])
                        rec[f"p_{m}"] = pm[m]
                    Sj = rank_combine(pm)
                    rec["p_comb"] = float(pvals(res.S.values[ref], np.array([Sj]), False)[0])
                inj.append(rec)
            log(f"  fold {k} inj {cname}/{alab} done")
    inj = pd.DataFrame(inj)
    for t in (res, inj):
        n2 = sum((t[f"p_{m}"] <= ALPHA_M).astype(int) for m in MODELS)
        t["n_models"] = n2
        t["flag"] = (t.p_comb <= ALPHA) & (n2 >= 2)
    inj.to_parquet(f"{OUT}/injections{'_pilot' if pilot else ''}.parquet", index=False)
    summarize_inj(inj, edges)
    rfloor = reporting_floor(inj)
    log("reporting floor (from injections only):", rfloor)
    res["flag_raw"] = res.flag
    res["flag"] = res.flag & (res.snr >= rfloor)
    json.dump(dict(reporting_floor=rfloor), open(f"{OUT}/reporting_floor{'_pilot' if pilot else ''}.json", "w"))
    if not pilot:
        res = pd.concat([res, L], axis=1)
        res.to_parquet(f"{OUT}/scores.parquet")
        np.savez_compressed(f"{OUT}/features.npz", X=feat["X"], S=feat["S"], M=feat["M"], XH=XH, med=feat["med"],
                            keep=keep, edges=edges)
    else:
        # calibration-only summary (no ids, no ranking)
        nfb = res[res.set == "population"].groupby(["fold", "bin"]).size()
        log("pilot fold-bin sizes", nfb.to_dict())
    return res, inj


def summarize_inj(inj, edges):
    sb = [5, 10, 20, 50, 100, 300, 1e9]
    inj = inj.assign(sbin=pd.cut(inj.host_snr, sb))
    t = inj.groupby(["cls", "amp_label", "sbin"], observed=True).flag.agg(["mean", "size"]).unstack("sbin")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(t.round(2))
    g = inj.groupby(["cls", "amp_label"])
    t2 = pd.DataFrame({"flag": g.flag.mean(), "p_comb<=a": g.p_comb.apply(lambda q: (q <= ALPHA).mean()),
                       **{f"{m}<=am": g[f"p_{m}"].apply(lambda q: (q <= ALPHA_M).mean()) for m in MODELS}})
    print(t2.round(3))



# ---------------------------------------------------------------- triage
FEATS = [(0.78, "O I airglow 0.78"), (0.84, "O I / He I-leak airglow 0.84"), (1.083, "He I 1.083 airglow"),
         (1.13, "O I airglow 1.13"), (1.25, "He I leak 1.25"), (1.49, "He I leak 1.49")]
PHYS = [(1.282, "Pa-beta 1.282 um"), (1.875, "Pa-alpha 1.875 um"), (2.166, "Br-gamma 2.166 um"), (4.052, "Br-alpha 4.052 um"),
        (2.29, "CO 2.29 um bandhead"), (3.05, "3.05 um H2O ice/absorption"), (3.3, "3.3 um CH4 / PAH"),
        (4.27, "4.27 um CO2"), (4.6, "4.6 um CO fundamental"), (1.4, "1.4 um H2O"), (1.77, "1.77 um C2")]
AIRGLOW = [w for w, _ in FEATS]


def neighbours(Z, ordinary, i, k=8):
    d = np.sqrt(((Z[ordinary] - Z[i]) ** 2).sum(1))
    return np.flatnonzero(ordinary)[np.argsort(d)[:k]]


def deviation(x, s, m, ref):
    rmed = np.median(ref, 0)
    rsc = 1.4826 * np.median(np.abs(ref - rmed), 0)
    sig = np.sqrt(np.where(m, s, np.inf) ** 2 + (CH_FLOOR * np.abs(rmed)) ** 2 + rsc ** 2)
    return (x - rmed) / sig, rmed, rsc


def segments(dev, lam, thr=4.0):
    out = []
    sgn = np.sign(dev) * (np.abs(dev) > thr)
    i = 0
    while i < len(dev):
        if sgn[i] != 0:
            j = i
            while j + 1 < len(dev) and sgn[j + 1] == sgn[i]:
                j += 1
            out.append(dict(lo=float(lam[i]), hi=float(lam[j]), n=j - i + 1, sign="excess" if sgn[i] > 0 else "deficit",
                            peak_sigma=float(np.max(np.abs(dev[i:j + 1]))), chi2=float((dev[i:j + 1] ** 2).sum())))
            i = j + 1
        else:
            i += 1
    return sorted(out, key=lambda q: -q["chi2"])


def describe(segs, lam, dev):
    txt = []
    for g in segs[:3]:
        near = [nm for w, nm in PHYS if g["lo"] - 0.05 <= w <= g["hi"] + 0.05] if g["n"] <= 4 else []
        if g["n"] > 4:
            g = dict(g, sign="broad " + g["sign"])
        rng = f"{g['lo']:.2f}-{g['hi']:.2f} um" if g["n"] > 1 else f"{g['lo']:.2f} um"
        t = f"{g['sign']} at {rng} ({g['n']} ch, peak {g['peak_sigma']:.0f} sigma)"
        if near:
            t += " near " + ", ".join(near)
        txt.append(t)
    blue, red = dev[lam < 1.5], dev[lam > 3.0]
    if len(blue) and len(red):
        sl = np.median(red) - np.median(blue)
        if abs(sl) > 3:
            txt.append(("redder" if sl > 0 else "bluer") + f" continuum than its nearest normal neighbours ({sl:+.0f} sigma red-minus-blue)")
    return "; ".join(txt) if txt else "diffuse low-level deviations"


def triage():
    S = pd.read_parquet(f"{E18}/sources.parquet")
    res = pd.read_parquet(f"{OUT}/scores.parquet")
    z = np.load(f"{OUT}/features.npz")
    X, Sg, M, keep = z["X"], z["S"], z["M"], z["keep"]
    lam = CEN[keep]
    det_of = (np.arange(NCH) // 17 + 1)[keep]
    scored = res.set.isin(["population", "held_out"]).values
    pop = (res.set == "population").values
    ordinary = pop & (res.p_comb.values > 0.2)
    mu, sd = X[pop].mean(0), X[pop].std(0) + 1e-9
    Z = (X - mu) / sd
    fl = np.flatnonzero(res.flag.values & scored)
    fl = fl[np.lexsort((-res.S.values[fl], res.p_comb.values[fl]))]
    log("flags", len(fl))
    # per-picture rows of flagged sources (clean) for half-split / crosstalk-excluded spectra
    d, _ = load_clean(set(fl.tolist()))
    # raw rows of bright sources (for crosstalk and overflow neighbours)
    bright = np.flatnonzero(((S.G.fillna(99) < 12) | (S.W1mag.fillna(99) < 10)).values)
    raw = pd.concat([pd.read_parquet(f, columns=["src", "obs_id", "x", "y", "flux_mJy", "flags_core"]).query("src in @bs",
                     local_dict=dict(bs=set(bright.tolist()))) for f in sorted(glob.glob(f"{E18}/phot/part-*.parquet"))])
    ovf = raw.groupby("src").flags_core.apply(lambda q: float(((q.values >= 0) & ((q.values >> 1) & 1 == 1)).mean()))
    rawg = {o: g for o, g in raw.groupby("obs_id", observed=True)}
    rows = []
    dom_ch = {}
    for i in fl:
        ref_idx = neighbours(Z, ordinary, i)
        dev, rmed, rsc = deviation(X[i], Sg[i], M[i], X[ref_idx])
        segs = segments(dev, lam)
        r = d[d.src == i]
        # half split (10-day-bin parity) spectra, normalised by the full-spectrum median
        hs = []
        for h in (0, 1):
            q = r[r.half == h]
            Fh, Eh, Nh = aggregate(np.zeros(len(q), int), q.ch.values, q.f.values, q.e.values, 1)
            Xh = Fh[0, keep] / z["med"][i]
            Sh = Eh[0, keep] / z["med"][i]
            Mh = (Nh[0, keep] >= MIN_N) & np.isfinite(Xh)
            dh, _, _ = deviation(np.where(Mh, Xh, X[i]), np.where(Mh, Sh, np.inf), Mh, X[ref_idx])
            hs.append(np.where(Mh, dh, np.nan))
        v = np.isfinite(hs[0]) & np.isfinite(hs[1]) & (np.abs(dev) > 3)
        corr = float(np.corrcoef(hs[0][v], hs[1][v])[0, 1]) if v.sum() >= 3 else np.nan
        rms = lambda a, w: float(np.sqrt(np.nanmean(a[w] ** 2))) if w.any() else np.nan
        big = np.abs(dev) > 3
        ratioA, ratioB = rms(hs[0], big) / max(rms(dev, big), 1e-9), rms(hs[1], big) / max(rms(dev, big), 1e-9)
        # crosstalk geometry vs bright catalogue sources in the same picture (32 px slow-read, 64i px fast-read)
        nxt, nimg = 0, 0
        xt_obs = set()
        fi = np.nanmedian(r.f.values) if len(r) else np.nan
        for o, q in r.groupby("obs", observed=True):
            nimg += 1
            g = rawg.get(o)
            if g is None:
                continue
            g = g[(g.src != i) & (g.flux_mJy > 30 * max(fi, 1e-3))]
            if not len(g):
                continue
            dx, dy = g.x.values - q.x.values[0], g.y.values - q.y.values[0]
            t1 = (np.abs(dx) <= 2) & (np.abs(np.abs(dy) - 32) <= 2)
            m64 = np.abs(np.abs(dx) - 64 * np.maximum(np.round(np.abs(dx) / 64), 1)) <= 2
            t2 = (np.abs(dy) <= 2) & m64 & (np.abs(dx) > 30)
            if (t1 | t2).any():
                nxt += 1
                xt_obs.add(o)
        xt_frac = nxt / max(nimg, 1)
        xt_drop = np.nan
        if nxt:
            q = r[~r.obs.isin(xt_obs)]
            Fq, Eq, Nq = aggregate(np.zeros(len(q), int), q.ch.values, q.f.values, q.e.values, 1)
            Xq = Fq[0, keep] / z["med"][i]
            Mq = (Nq[0, keep] >= MIN_N) & np.isfinite(Xq)
            dq, _, _ = deviation(np.where(Mq, Xq, X[i]), np.where(Mq, Eq[0, keep] / z["med"][i], np.inf), Mq, X[ref_idx])
            xt_drop = float(np.nansum(np.where(Mq, dq, 0) ** 2) / max(np.sum(dev ** 2), 1e-9))
        # overflowing bright neighbour within 25 px (154") ; brighter neighbour within 12" (2 beams)
        sep = np.hypot((S.ra.values[bright] - S.ra.values[i]) * np.cos(np.radians(S.dec.values[i])),
                       S.dec.values[bright] - S.dec.values[i]) * 3600
        nb = bright[(sep < 154) & (bright != i)]
        ovf_nb = [int(b) for b in nb if ovf.get(b, 0) >= 0.10]
        conf = bool(np.nan_to_num(S.d_brighter_arcsec.values[i], nan=99) < 12)
        # detector-level offset test: piecewise-constant per detector explains the deviation?
        dd = np.where(M[i], dev, 0)
        pc = np.zeros_like(dd)
        for k in range(1, 7):
            w = det_of == k
            if w.any():
                pc[w] = np.median(dd[w])
        tot = (dd ** 2).sum()
        det_share = {k: float((dd[det_of == k] ** 2).sum() / max(tot, 1e-9)) for k in range(1, 7)}
        det_off = bool(1 - ((dd - pc) ** 2).sum() / max(tot, 1e-9) > 0.8 and max(det_share.values()) > 0.8)
        top = segs[0] if segs else None
        if top:
            dom_ch[i] = round((top["lo"] + top["hi"]) / 2, 2)
        narrow = bool(top and top["n"] <= 2)
        airglow = bool(top and narrow and any(top["lo"] - 0.6 * WID[keep][np.argmin(np.abs(lam - top["lo"]))] <= a <=
                                               top["hi"] + 0.6 * WID[keep][np.argmin(np.abs(lam - top["hi"]))] for a in AIRGLOW))
        dbs = bool(top and top["lo"] >= 2.3 and top["hi"] <= 2.5)
        rows.append(dict(idx=int(i), id=res.id.values[i], set=res.set.values[i], snr=float(res.snr.values[i]),
                         p_comb=float(res.p_comb.values[i]), p_rf=float(res.p_rf.values[i]), p_ae=float(res.p_ae.values[i]),
                         p_if=float(res.p_if.values[i]), n_models=int(res.n_models.values[i]),
                         neighbours=[int(q) for q in ref_idx], description=describe(segs, lam, dev),
                         segments=segs[:4], half_corr=corr, half_ratio=[ratioA, ratioB], n_pictures=int(r.obs.nunique()),
                         crosstalk_frac=xt_frac, crosstalk_chi2_ratio=xt_drop, overflow_neighbours=ovf_nb,
                         confusion=conf, det_share=det_share, detector_offset=det_off, airglow=airglow, dbs_2p4=dbs))
    T = pd.DataFrame(rows)
    # catalogue context
    cat_cols = ["kind", "G", "bp_rp", "parallax", "parallax_error", "ruwe", "phot_variable_flag", "classprob_quasar",
                "classprob_galaxy", "classprob_star", "vari_class", "allwise", "W1mag", "W2mag", "W3mag", "Jmag", "Kmag",
                "blend_n", "d_brighter_arcsec", "neowise_var_kim2026", "kim2026_name", "milliquas_name", "milliquas_type",
                "milliquas_z", "ra", "dec"]
    C = S.iloc[T.idx.values][cat_cols].reset_index(drop=True)
    T = pd.concat([T, C, res.iloc[T.idx.values][["simbad_id", "simbad_type", "simbad_sp", "vsx_name", "vsx_type",
                                                 "known_class"]].reset_index(drop=True)], axis=1)
    if os.path.exists(f"{OUT}/desi_dr1_raw.parquet"):        # post-freeze addition (logged): DESI DR1 context
        de = pd.read_parquet(f"{OUT}/desi_dr1_raw.parquet")
        j, _ = sky_match(T.ra.values, T.dec.values, de.mean_fiber_ra.values, de.mean_fiber_dec.values, 1.5)
        T["desi_spectype"] = np.where(j >= 0, de.spectype.values[np.clip(j, 0, None)], None)
        T["desi_z"] = np.where(j >= 0, de.z.values[np.clip(j, 0, None)], np.nan)
        T["desi_zwarn"] = np.where(j >= 0, de.zwarn.values[np.clip(j, 0, None)], -1)
    rep = pd.Series(list(dom_ch.values())).value_counts()
    T["repeat_channel"] = [bool(dom_ch.get(i) is not None and rep.get(dom_ch[i], 0) >= 3) for i in T.idx]
    ver, why = [], []
    for _, t in T.iterrows():
        art = []
        if t.confusion: art.append("brighter neighbour within 12 arcsec (confusion)")
        if t.overflow_neighbours: art.append("overflowing bright star within 25 px (bloom)")
        if t.crosstalk_frac > 0.10 and (np.isnan(t.crosstalk_chi2_ratio) or t.crosstalk_chi2_ratio < 0.5):
            art.append(f"crosstalk geometry in {t.crosstalk_frac:.0%} of pictures; deviation drops without them")
        if t.detector_offset: art.append("deviation is a single-detector offset")
        if t.airglow: art.append("narrow feature in an airglow channel")
        if not (np.isfinite(t.half_corr) and t.half_corr > 0.5 and min(t.half_ratio) > 0.33):
            art.append("not reproduced in both independent halves of the pictures")
        if t.repeat_channel: art.append("same dominant channel in >= 3 flags (systematic)")
        known = []
        if isinstance(t.known_class, str): known.append(t.known_class)
        if isinstance(t.simbad_type, str) and t.simbad_type not in ("*", "?", "IR", "Opt", "X", "Rad", "UV"):
            known.append(f"SIMBAD {t.simbad_type} {t.simbad_id}")
        if isinstance(t.vsx_type, str): known.append(f"VSX {t.vsx_type} {t.vsx_name}")
        if t.phot_variable_flag == "VARIABLE": known.append("Gaia DR3 photometric variable")
        if isinstance(t.vari_class, str): known.append(f"Gaia vari {t.vari_class}")
        if isinstance(t.milliquas_name, str): known.append(f"Milliquas {t.milliquas_type} {t.milliquas_name}")
        if t.neowise_var_kim2026: known.append(f"NEOWISE NEP variable {t.kim2026_name}")
        if t.get("desi_spectype") == "QSO": known.append(f"DESI DR1 QSO z={t.desi_z:.3f}")
        if t.kind == "gaia" and t.classprob_quasar > 0.5: known.append("Gaia DSC quasar")
        if t.kind == "gaia" and t.classprob_galaxy > 0.5: known.append("Gaia DSC galaxy")
        if t.kind == "allwise_only" and np.nan_to_num(t.W1mag - t.W2mag) > 0.8: known.append("AllWISE W1-W2>0.8 (AGN colour)")
        if art:
            ver.append("artefact veto"); why.append("; ".join(art))
        elif known:
            ver.append("explained / known class"); why.append("; ".join(known))
        else:
            ver.append("survivor"); why.append("unusual SPHEREx spectrum not explained by the catalogues we checked")
    T["verdict"], T["verdict_reason"] = ver, why
    T.to_json(f"{OUT}/triage.json", orient="records", indent=1, default_handler=str)
    T.drop(columns=["segments", "neighbours", "det_share"]).to_parquet(f"{OUT}/triage.parquet", index=False)
    log(T.verdict.value_counts().to_dict())
    return T



# ---------------------------------------------------------------- figures + findings
COL = {"hlines": "#c0392b", "ices": "#2471a3", "tdwarf": "#7d3c98", "redqso": "#d35400", "carbon": "#1e8449", "null": "#7f8c8d"}


def recall_table(inj):
    inj = inj.assign(sb=pd.cut(inj.host_snr, SNR_BINS, labels=SNR_LAB))
    out = {}
    for (c, a), g in inj.groupby(["cls", "amp_label"]):
        t = g.groupby("sb", observed=False).flag.agg(["sum", "size"])
        out[f"{c}/{a}"] = dict(snr_bins=SNR_LAB, n=[int(v) for v in t["size"]], n_flagged=[int(v) for v in t["sum"]],
                               recall=[(float(s / n) if n else None) for s, n in zip(t["sum"], t["size"])],
                               overall=float(g.flag.mean()), n_total=int(len(g)))
    return out


def wilson(k, n, z=1.0):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (c - h, c + h)


def figs():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 15, "axes.titlesize": 16, "axes.labelsize": 16, "legend.fontsize": 12,
                         "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"})
    S = pd.read_parquet(f"{E18}/sources.parquet")
    res = pd.read_parquet(f"{OUT}/scores.parquet")
    inj = pd.read_parquet(f"{OUT}/injections.parquet")
    T = pd.read_json(f"{OUT}/triage.json")
    z = np.load(f"{OUT}/features.npz")
    X, Sg, M, keep = z["X"], z["S"], z["M"], z["keep"]
    lam = CEN[keep]
    os.makedirs("site/data/figs", exist_ok=True)
    # ---- 1. injection recall vs SNR
    rt = recall_table(inj)
    fig, axs = plt.subplots(1, 3, figsize=(21, 7.5), sharey=True)
    nonempty = [i for i in range(len(SNR_LAB)) if rt["null/null"]["n"][i] > 0]
    xs = np.arange(len(nonempty))
    for ax, al in zip(axs, ("weak", "medium", "strong")):
        for c, (kind, amps, lab) in INJ_CLASSES.items():
            r = rt[f"{c}/{al}"]
            y = np.array([np.nan if r["recall"][i] is None else r["recall"][i] for i in nonempty])
            lo_hi = np.array([wilson(r["n_flagged"][i], r["n"][i]) for i in nonempty])
            ax.errorbar(xs, y, yerr=[np.clip(y - lo_hi[:, 0], 0, None), np.clip(lo_hi[:, 1] - y, 0, None)], color=COL[c], marker="o", lw=2.5, ms=7,
                        capsize=3, label=SHORT[c])
        r = rt["null/null"]
        ax.plot(xs, [np.nan if r["recall"][i] is None else r["recall"][i] for i in nonempty], color=COL["null"], marker="s", lw=2, ls=":",
                label="no injection (false-flag check)")
        ax.set_xticks(xs, [SNR_LAB[i] for i in nonempty])
        ax.set_xlabel("host signal-to-noise per channel")
        ax.set_title(AMP_TXT[al], fontsize=13)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("fraction of fake unusual spectra flagged")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, framealpha=0.95, fontsize=13)
    fig.suptitle("Can the anomaly search find fake unusual spectra planted in real SPHEREx data?", fontsize=17)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig("site/data/figs/e18a_injections.png", dpi=120)
    plt.close(fig)
    # ---- 2. UMAP
    import umap
    sc = res.set.isin(["population", "held_out"]).values
    mu, sd = X[res.set.values == "population"].mean(0), X[res.set.values == "population"].std(0) + 1e-9
    emb = umap.UMAP(n_neighbors=30, min_dist=0.2, random_state=SEED).fit_transform(((X - mu) / sd)[sc])
    idx = np.flatnonzero(sc)
    np.save(f"{OUT}/umap.npy", np.c_[idx, emb])
    fig, ax = plt.subplots(figsize=(11, 9))
    kind = S.kind.values[idx]
    ax.scatter(*emb[kind == "gaia"].T, s=8, c="#b3b6b7", label="stars (Gaia)", lw=0)
    ax.scatter(*emb[kind != "gaia"].T, s=8, c="#85c1e9", label="galaxies / AGN (infrared-only sources)", lw=0)
    kc = res.known_class.values[idx]
    mk = {"quasar / AGN": ("D", "#e67e22"), "planetary nebula": ("P", "#c0392b"), "white dwarf": ("v", "#2e86c1"),
          "peculiar / emission-line star": ("*", "#8e44ad"), "known variable star": ("^", "#27ae60")}
    for c, (m, col) in mk.items():
        w = kc == c
        if w.any():
            ax.scatter(*emb[w].T, s=70, marker=m, c=col, edgecolor="k", lw=0.5, label=f"{c} (catalogue, {w.sum()})")
    flg = res.flag.values[idx]
    ax.scatter(*emb[flg].T, s=160, facecolor="none", edgecolor="k", lw=1.8, label=f"flagged as unusual ({flg.sum()})")
    surv = set(T.idx[T.verdict == "survivor"]) if len(T) else set()
    ws = np.isin(idx, list(surv))
    if ws.any():
        ax.scatter(*emb[ws].T, s=320, marker="*", c="#f1c40f", edgecolor="k", lw=1.2,
                   label=f"unusual, not explained by catalogues ({ws.sum()})")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Map of ~%d SPHEREx spectra near the north ecliptic pole\n(similar spectra sit close together)" % sc.sum())
    ax.legend(loc="best", fontsize=11, framealpha=0.95, markerscale=1.2)
    fig.tight_layout()
    fig.savefig("site/data/figs/e18a_umap.png", dpi=120)
    plt.close(fig)
    # ---- 3. flag spectra
    if len(T):
        order = np.r_[np.flatnonzero(T.verdict.values == "survivor"), np.flatnonzero(T.verdict.values != "survivor")][:12]
        n = len(order)
        nc = 3 if n > 4 else max(n, 1)
        nr = int(np.ceil(n / nc))
        fig, axs = plt.subplots(nr, nc, figsize=(6.2 * nc, 4.3 * nr), squeeze=False)
        for a, j in zip(axs.ravel(), order):
            t = T.iloc[j]
            i = int(t.idx)
            ref = X[np.array(t.neighbours, int)]
            rmed = np.median(ref, 0)
            for q in ref:
                a.plot(lam, q, color="#aab7b8", lw=0.8, alpha=0.7)
            a.plot(lam, rmed, color="#566573", lw=2, label="most similar normal sources")
            m = M[i]
            a.errorbar(lam[m], X[i][m], yerr=Sg[i][m], fmt="o", ms=3.5, color="#c0392b", ecolor="#e6b0aa", label="this source")
            yy = np.r_[X[i][m], rmed]
            a.set_ylim(min(0, np.nanmin(yy)) - 0.05 * np.nanmax(yy), 1.15 * np.nanmax(yy))
            a.set_xscale("log")
            a.set_xticks([0.8, 1, 1.5, 2, 3, 4, 5], ["0.8", "1", "1.5", "2", "3", "4", "5"])
            ttl = (t.simbad_id if isinstance(t.simbad_id, str) else str(t.id))[:28]
            vt = {"survivor": "not explained by catalogues", "artefact veto": "artefact veto",
                  "explained / known class": "known in catalogues"}[t.verdict]
            a.set_title(f"{ttl}\n[{vt}]", fontsize=12)
            a.set_xlabel("wavelength (micron)", fontsize=13)
            a.set_ylabel("brightness / typical", fontsize=13)
            a.text(0.02, 0.97, "\n".join(textwrap_short(t.description)), transform=a.transAxes, va="top", fontsize=9.5,
                   bbox=dict(fc="white", ec="0.8", alpha=0.9))
        for a in axs.ravel()[n:]:
            a.axis("off")
        axs[0, 0].legend(loc="lower left", fontsize=10)
        fig.suptitle("Spectra of flagged sources (red) against their most similar ordinary neighbours (grey)", fontsize=17)
        fig.tight_layout()
        fig.savefig("site/data/figs/e18a_flags.png", dpi=110)
        plt.close(fig)
    findings(S, res, inj, T, rt)


def textwrap_short(s, w=48, n=5):
    import textwrap
    return textwrap.wrap(s, w)[:n]


def findings(S, res, inj, T, rt):
    z = np.load(f"{OUT}/features.npz")
    spec = np.load(f"{OUT}/spectra.npz")
    pop = res.set == "population"
    nfb = res[pop].groupby(["fold", "bin"]).size()
    rfloor = json.load(open(f"{OUT}/reporting_floor.json"))["reporting_floor"]
    n_above = int((pop & (res.snr >= rfloor)).sum())
    exp_false = ALPHA * n_above
    held = res[res.set == "held_out"]
    kc = {}
    for c, g in held.groupby("known_class"):
        k, n = int(g.flag.sum()), len(g)
        kc[c] = dict(n=n, n_flagged=k, recall=k / n, recall_1sigma=list(wilson(k, n)),
                     flagged_ids=g.id[g.flag].tolist())
    sc = res.set.isin(["population", "held_out"])
    red = sc & (S.kind.values == "allwise_only") & ((S.W1mag - S.W2mag).fillna(0).values > 0.8)
    kc["(not held out) infrared-only sources with AllWISE W1-W2 > 0.8 (AGN colour)"] = dict(
        n=int(red.sum()), n_flagged=int(res.flag[red].sum()), recall=float(res.flag[red].mean()) if red.any() else None,
        note="descriptive flag rate; these are part of the training population, not held out")
    if os.path.exists(f"{OUT}/desi_dr1_raw.parquet"):
        de = pd.read_parquet(f"{OUT}/desi_dr1_raw.parquet")
        j, _ = sky_match(S.ra.values, S.dec.values, de.mean_fiber_ra.values, de.mean_fiber_dec.values, 1.5)
        st_ = np.where(j >= 0, de.spectype.values[np.clip(j, 0, None)], "")
        for typ in ("QSO", "GALAXY", "STAR"):
            w = sc.values & (st_ == typ)
            kc[f"(post-freeze, descriptive) DESI DR1 {typ}"] = dict(
                n=int(w.sum()), n_flagged=int(res.flag[w].sum()), recall=float(res.flag[w].mean()) if w.any() else None,
                n_held_out=int((w & (res.set == "held_out").values).sum()))
    for c, g in held.groupby("known_class"):
        a = g[g.snr >= rfloor]
        kc[c]["n_above_reporting_floor"] = int(len(a))
        kc[c]["n_flagged_above_floor"] = int(a.flag.sum())
    clean = pop & (res.known_class.isna()) & (res.simbad_type.isna() | res.simbad_type.isin(["*"]))
    calib = {f"frac_p_le_{a}": dict(observed=float((res.p_comb[pop] <= a).mean()), nominal_max=a) for a in (0.005, 0.02, 0.05, 0.2)}
    calib["note"] = ("p-values are leave-one-out ranks within (fold, SNR bin) of a fixed score function trained on the other "
                     "fold, so the population p-values are uniform by construction; the informative checks are the null "
                     "injections (hosts re-aggregated unchanged) and the flag rate of catalogue-clean ordinary stars")
    calib["null_injection_flag_rate"] = rt["null/null"]["overall"]
    calib["null_injection_n"] = rt["null/null"]["n_total"]
    calib["clean_star_flag_rate"] = float(res.flag[clean].mean())
    calib["clean_star_n"] = int(clean.sum())
    flags = []
    for _, t in T.iterrows():
        flags.append({k: (None if (isinstance(v, float) and not np.isfinite(v)) else v) for k, v in t.items()
                      if k not in ("neighbours",)})
    out = dict(
        experiment="E18-A", title="Label-free anomaly search on real SPHEREx NEP spectra",
        protocol="docs/protocols/E18.md (section 'E18-A frozen choices')", script="scripts/run_e18a.py", python=PY_NOTE,
        generated=time.strftime("%Y-%m-%d %H:%M %Z"),
        data=dict(pictures_used=int(spec["npics"]), phot_parts=int(spec["nparts"]), clean_rows=int(spec["nrows"]),
                  channels_kept=int(z["keep"].sum()), channels_dropped_um=np.round(CEN[~z["keep"]], 3).tolist(),
                  n_reported=int((S.in_region & S.is_primary).sum()), n_usable=int(res.set.ne("not_scored").sum()),
                  n_population=int(pop.sum()), n_held_out=int((res.set == "held_out").sum()),
                  snr_bin_edges=[float(v) for v in z["edges"]], fold_bin_sizes={f"{a}/{b}": int(v) for (a, b), v in nfb.items()}),
        design=dict(features="102-channel (17 per detector, log-spaced in the six band ranges) clipped inverse-variance mean "
                             "spectrum, per-picture error^2 = formal^2 + (0.03 F)^2, error of mean inflated by sqrt(chi2_r); "
                             "normalised by the median over valid channels; masked channels interpolated in log-lambda",
                    models=["Baron & Poznanski 2017 unsupervised RF (500 trees, real vs column-shuffled; weirdness = 1 - mean "
                            "shared-real-leaf similarity to the training fold)",
                            "autoencoder 98-64-32-6-32-64-98 (masked MSE training; score = reduced chi2 with 1% channel floor)",
                            "isolation forest (500 trees)"],
                    combination="rank-average of per-model conformal p-values; flag = p_comb <= alpha AND >= 2 of 3 models with p <= 0.02",
                    conformal="Mondrian (SNR bins) leave-one-out rank p-values within the test fold; models trained on the other fold"),
        threshold=dict(alpha=ALPHA, alpha_per_model=ALPHA_M, snr_floor=SNR_FLOOR, reporting_floor_snr=rfloor,
                       n_population_above_floor=n_above, n_flags_all_snr=int(res.flag_raw.sum()),
                       expected_false_flags=exp_false,
                       expected_false_flags_note="alpha x population sources above the reporting floor (marginal conformal guarantee; upper bound, conservative if the calibration set contains true anomalies)"),
        calibration_checks=calib, injection_recall=rt, known_class_recall=kc,
        n_flags=int(len(T)), verdict_counts=T.verdict.value_counts().to_dict() if len(T) else {},
        flags=flags,
        figures=["site/data/figs/e18a_injections.png", "site/data/figs/e18a_umap.png", "site/data/figs/e18a_flags.png"],
        prior_work=["Baron & Poznanski 2017 (MNRAS 465, 4530) unsupervised RF", "Reis et al. 2018 (APOGEE RF outliers)",
                    "Sasli et al. 2026 ASTRANet (Mondrian conformal anomaly p-values)",
                    "Nguyen et al. 2026, arXiv:2607.20797 (supervised contrastive star-galaxy separation on real SPHEREx data; nearest ML prior work)",
                    "Ren et al. 2026, arXiv:2609.02167 (NMF templates of NEP galaxies/AGN)",
                    "Crowther & Smith 2026 (proposed ML selection of SPHEREx WR candidates)"],
        wording="Survivors are sources with unusual SPHEREx spectra not explained by the catalogues we checked; "
                "expected false flags given above. No independent confirmation.",
        deviations=json.load(open(f"{OUT}/deviations.json")) if os.path.exists(f"{OUT}/deviations.json") else [])
    json.dump(out, open("data/findings/e18a.json", "w"), indent=1, default=str)
    log("wrote data/findings/e18a.json")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        build()
    elif cmd == "pilot":
        run(pilot=True, n_hosts=int(sys.argv[2]) if len(sys.argv) > 2 else 60)
    elif cmd == "run":
        run(pilot=False)
    elif cmd == "triage":
        triage()
    elif cmd == "figs":
        figs()
