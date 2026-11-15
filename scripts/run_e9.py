"""E9 — RW Aurigae and R Draconis: did they change between SPHEREx visits, and at which colours?
Protocol: docs/PROTOCOLS.md "E9"; frozen choices: docs/protocols/E9.md.

Stages (run in this order; `targets` must not be run before docs/protocols/E9.md is frozen):
  OMP_NUM_THREADS=1 .venv/bin/python scripts/run_e9.py phot      # comparison-star photometry (all variants)
  OMP_NUM_THREADS=1 .venv/bin/python scripts/run_e9.py null      # variant table, sigma_null -> data/phase2/e9/null_scan.json
  OMP_NUM_THREADS=1 .venv/bin/python scripts/run_e9.py freeze    # frozen choice -> data/phase2/e9/frozen.json
  OMP_NUM_THREADS=1 .venv/bin/python scripts/run_e9.py targets   # targets -> data/findings/e9.json, e9_spectra.json, figures
"""
import glob, hashlib, json, os, sys, time
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e9_lib as L  # noqa: E402

OUT = "data/phase2/e9"
CLASS = dict(rwaur=dict(mode="gap", core_frac=0.88, name="RW Aurigae"),
             rdra=dict(mode="bin", core_frac=1.00, name="R Draconis"))
S_D = json.load(open("data/findings/e7.json"))["s_d"]
SIGMA_MAX = 0.15               # a detector whose comparison-star sigma_null exceeds this is not used (PROTOCOLS E9)
MIN_PIC = 2                    # pictures needed in a (detector, window) cell
MIN_STAR_PIC = 6               # pictures needed per detector to fit a star's static spectrum


def _phot_one(path):
    out = f"{OUT}/phot/{os.path.basename(path)[:-4]}.parquet"
    if not os.path.exists(out):
        df = L.measure(L.load_star(path), S_D)
        df.to_parquet(out + ".tmp")
        os.replace(out + ".tmp", out)
    return out


def stage_phot():
    os.makedirs(f"{OUT}/phot", exist_ok=True)
    files = sorted(glob.glob(f"{OUT}/stamps/*_c[0-9][0-9].npz"))
    with Pool(6) as p:
        for i, o in enumerate(p.imap_unordered(_phot_one, files)):
            print(i + 1, len(files), o, flush=True)


def load_class(cname):
    """{star: DataFrame} for the comparison stars of one class that pass the flagged-core-fraction match."""
    stars, info = {}, []
    lo, hi = CLASS[cname]["core_frac"] - 0.2, CLASS[cname]["core_frac"] + 0.2
    for f in sorted(glob.glob(f"{OUT}/phot/{cname}_c*.parquet")):
        df = pd.read_parquet(f)
        if len(df) == 0:
            continue
        name = os.path.basename(f)[:-8]
        cf = float(df.core_flag.mean())
        w = L.window_index(df.mjd.values, "gap")
        rec = dict(name=name, n_pictures=int(len(df)), core_frac=cf, n_gap_windows=int(w.max() + 1),
                   used=bool(lo <= cf <= hi))
        info.append(rec)
        if rec["used"]:
            stars[name] = df
    return stars, info


def star_a(df, col, mode, method, deg=2, use_x=False):
    """a_{d,w} table for one star: rows (det, win, a, n). Flux column `col` must be > 0."""
    win = L.window_index(df.mjd.values, mode)
    rows, resid = [], np.full(len(df), np.nan)
    kept = np.zeros(len(df), bool)
    for d in range(1, 7):
        m = (df.det.values == d) & np.isfinite(df[col].values) & (df[col].values > 0)
        if m.sum() < MIN_STAR_PIC:
            continue
        r = L.fit_detector(np.log(df[col].values[m]), df.lam.values[m], win[m], method, deg,
                           df.x.values[m] if use_x else None)
        if r is None:
            continue
        aw, s, res, keep = r
        resid[np.flatnonzero(m)] = res
        kept[np.flatnonzero(m)] = keep
        # within-window picture scatter of this star in this detector (windows with >= 2 kept pictures)
        rr, ww = res[keep], win[m][keep]
        dev = np.concatenate([rr[ww == u] - np.median(rr[ww == u]) for u in np.unique(ww) if (ww == u).sum() >= 2] or [[]])
        sw = mad(dev) if len(dev) >= 4 else np.nan
        for w, (a, n) in aw.items():
            rows.append(dict(det=d, win=w, a=a, n=n, s=s, sw=sw))
    return pd.DataFrame(rows), resid, kept


def class_table(stars, col, mode, method, deg=2, use_x=False, inject=None):
    """All comparison stars -> one table of a_{d,w}. inject = (window rank, factor) multiplies the flux of every
    picture in the star's k-th window (by time) to test how much of a real change the model returns."""
    tabs = []
    for name, df in stars.items():
        if inject is not None:
            df = df.copy()
            win = L.window_index(df.mjd.values, mode)
            ws = np.unique(win)
            wsel = ws[min(inject[0], len(ws) - 1)] if inject[0] >= 0 else ws[len(ws) // 2]
            df.loc[win == wsel, col] = df.loc[win == wsel, col] * inject[1]
        t, _, _ = star_a(df, col, mode, method, deg, use_x)
        if len(t):
            t["star"] = name
            if inject is not None:
                t["inj"] = t.win == wsel
            tabs.append(t)
    return pd.concat(tabs, ignore_index=True) if tabs else pd.DataFrame()


def mad(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) else np.nan


def sigma_null(tab):
    """Robust spread of a_{d,w} per detector over cells with >= MIN_PIC pictures."""
    t = tab[tab.n >= MIN_PIC]
    return {d: mad(t.a[t.det == d]) for d in range(1, 7) if (t.det == d).sum() >= 10}


def add_inflation(tab, swmed=None):
    """Column f = max(1, star's within-window picture scatter / class median for that detector): a star whose
    pictures scatter more than usual gets proportionally larger errors. swmed is fitted here if not given."""
    tab = tab.copy()
    if swmed is None:
        u = tab.drop_duplicates(["star", "det"])
        swmed = {int(d): float(np.nanmedian(q.sw)) for d, q in u.groupby("det") if np.isfinite(q.sw).sum() >= 5}
    ref = tab.det.map(swmed)
    tab["f"] = np.where(np.isfinite(tab.sw) & np.isfinite(ref), np.maximum(1.0, tab.sw / ref), 1.0)
    return tab, swmed


def sigma_model(tab, dets):
    """sigma(d, n)^2 = S_d^2 + P_d^2 / n for a cell with n pictures, fitted on the comparison stars:
    MAD of a in three groups of n, non-negative least squares of MAD^2 on 1/n, then rescaled so that
    MAD(a / sigma(d, n)) = 1. Returns {d: (S, P)}."""
    from scipy.optimize import nnls
    out = {}
    for d in dets:
        t = tab[(tab.det == d) & (tab.n >= MIN_PIC)]
        t = t.assign(a=t.a / t.f)
        S, P = mad(t.a), 0.0
        if len(t) >= 45 and t.n.nunique() >= 6:
            q = np.quantile(t.n, [1 / 3, 2 / 3])
            grp = np.digitize(t.n, q, right=True)
            A, y = [], []
            for g in range(3):
                m = grp == g
                if m.sum() >= 10:
                    A.append([1.0, float(np.mean(1.0 / t.n[m]))]), y.append(mad(t.a[m]) ** 2)
            if len(A) >= 2:
                c, _ = nnls(np.array(A), np.array(y))
                if c.sum() > 0:
                    S, P = float(np.sqrt(c[0])), float(np.sqrt(c[1]))
        k = mad(t.a / np.sqrt(S ** 2 + P ** 2 / t.n))
        out[int(d)] = (S * k, P * k)
    return out


def cell_sigma(model, det, n):
    S = np.array([model[int(d)][0] for d in det])
    P = np.array([model[int(d)][1] for d in det])
    return np.sqrt(S ** 2 + P ** 2 / np.asarray(n, float))


def all_det(tab, model):
    """All-detector factor per (star, window): inverse-variance mean of a_{d,w} with sigma(d, n); `err` is its
    formal error (to be multiplied by the comparison-star calibration k_all)."""
    t = tab[(tab.n >= MIN_PIC) & tab.det.isin(list(model))].copy()
    t["w"] = 1.0 / (cell_sigma(model, t.det.values, t.n.values) * t.f.values) ** 2
    t["aw"] = t.a * t.w
    g = t.groupby(["star", "win"]).agg(aw=("aw", "sum"), w=("w", "sum"), ndet=("det", "count")).reset_index()
    g["a"] = g.aw / g.w
    g["err"] = 1.0 / np.sqrt(g.w)
    return g[["star", "win", "a", "err", "ndet"]]


def exceed(tab, model, k_all, k=5.0):
    """Fraction of comparison-star windows that would be called 'changed' by the frozen rule."""
    t = tab[(tab.n >= MIN_PIC) & tab.det.isin(list(model))].copy()
    t["big"] = np.abs(t.a) > k * cell_sigma(model, t.det.values, t.n.values) * t.f.values
    nbig = t.groupby(["star", "win"]).big.sum().rename("nbig").reset_index()
    g = all_det(tab, model).merge(nbig, on=["star", "win"])
    g["z"] = g.a / (g.err * k_all)
    g["hit"] = (np.abs(g.z) > k) | (g.nbig >= 2)
    return float(g.hit.mean()), int(g.hit.sum()), int(len(g)), g


def null_summary(tab):
    """Usable detectors, sigma model, all-detector calibration and exceedance for one table of comparison cells."""
    sig_every = sigma_null(tab)
    sig = {d: v for d, v in sig_every.items() if v <= SIGMA_MAX}
    if len(sig) < 2:
        return None
    tab, swmed = add_inflation(tab)
    model = sigma_model(tab, list(sig))
    ad = all_det(tab, model)
    k_all = mad(ad.a / ad.err)
    fr, nh, nt, g = exceed(tab, model, k_all)
    return dict(tab=tab, swmed=swmed, sig_every=sig_every, sig=sig, model=model, k_all=k_all, sigma_all_typical=float(np.median(ad.err) * k_all),
                frac_exceed=fr, n_exceed=nh, n_windows=nt, table=g)


def split_half(names, seed=2026111509):
    """Seeded split of the comparison stars: selection half (variant choice) and held-out half."""
    names = sorted(names)
    perm = np.random.default_rng(seed).permutation(len(names))
    sel = {names[i] for i in perm[: (len(names) + 1) // 2]}
    return sel, set(names) - sel


_STARS, _MODE = None, None


def _variant(args):
    est, ri, ro, method, use_x = args
    stars, mode = _STARS, _MODE
    col = f"{est}_{ri}_{ro}"
    tab = class_table(stars, col, mode, method, 2, use_x)
    if not len(tab):
        return None
    ns = null_summary(tab)
    if ns is None:
        return None
    sig = ns["sig"]
    inj = class_table(stars, col, mode, method, 2, use_x, inject=(-1, 0.80))
    # returned contrast of the injected window against the star's other windows (truth: -0.20)
    q = inj[inj.n >= MIN_PIC]
    oth = q[~q.inj].groupby(["star", "det"]).a.median().rename("a_other").reset_index()
    ii = q[q.inj].merge(oth, on=["star", "det"])
    ii["c"] = (1 + ii.a) / (1 + ii.a_other) - 1
    rec_d = {d: float(np.median(ii.c[ii.det == d])) for d in sig if (ii.det == d).sum() >= 5}
    mf = float(np.nanmedian([df[("mfb" if est.endswith("b") else "mf") + f"_{ri}_{ro}"].median() for df in stars.values()]))
    nbad = float(np.mean([df[col].isna().mean() for df in stars.values()]))
    return dict(est=est, r_in=ri, r_out=ro, method=method, use_x=use_x, sigma=sig, sigma_every_detector=ns["sig_every"],
                n_usable=len(sig), sigma_all=ns["sigma_all_typical"], k_all=ns["k_all"],
                median_sigma=float(np.median(list(sig.values()))),
                max_sigma=float(max(sig.values())), frac_exceed=ns["frac_exceed"], n_exceed=ns["n_exceed"],
                n_windows=ns["n_windows"], recovered=rec_d,
                median_recovered=float(np.median(list(rec_d.values()))) if rec_d else None, masked_frac=mf,
                frac_unmeasured=nbad, n_cells=int((tab.n >= MIN_PIC).sum()))


def stage_null():
    global _STARS, _MODE
    import warnings
    warnings.filterwarnings("ignore")
    res = {}
    for cname, c in CLASS.items():
        stars, info = load_class(cname)
        print(cname, "stars used", len(stars), "of", len(info), flush=True)
        sel, held = split_half(stars)
        _STARS, _MODE = {k: v for k, v in stars.items() if k in sel}, c["mode"]     # variants are ranked on the selection half
        grid = [(est, ri, ro, method, use_x) for est in ("fill", "med", "fillb", "medb") for ri in L.R_IN
                for ro in L.R_OUT if ro > ri for method in ("static", "joint") for use_x in (False, True)]
        with Pool(6) as p:                       # fork: workers inherit _STARS
            rows = [r for r in p.map(_variant, grid, chunksize=4) if r is not None]
        for r in sorted([r for r in rows if r["frac_unmeasured"] <= 0.05 and r["median_recovered"]],
                        key=lambda r: (r["frac_exceed"] >= 0.01, -abs(r["median_recovered"]) / r["sigma_all"]))[:12]:
            print(cname, r["est"], r["r_in"], r["r_out"], r["method"], "x" if r["use_x"] else "-",
                  "sig", {k: round(v, 3) for k, v in r["sigma_every_detector"].items()}, "all", round(r["sigma_all"], 4),
                  "exc", round(r["frac_exceed"], 4), "rec", r["median_recovered"] and round(r["median_recovered"], 3),
                  "mf", round(r["masked_frac"], 2), "unmeas", round(r["frac_unmeasured"], 3), flush=True)
        res[cname] = dict(stars=info, n_used=len(stars), selection_half=sorted(sel), held_out_half=sorted(held),
                          n_variants=len(grid), variants=rows)
        json.dump(res, open(f"{OUT}/null_scan.json", "w"), indent=1)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("fork")
    stage = sys.argv[1]
    if stage == "phot":
        stage_phot()
    elif stage == "null":
        stage_null()
    else:
        import e9_targets
        getattr(e9_targets, f"stage_{stage}")()
