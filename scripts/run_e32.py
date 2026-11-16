"""E32 (docs/protocols/E32.md): WISE 0855-0714 SPHEREx/JWST ratio with JWST passed through each picture's spectral
response (centre CWAVE, bandwidth CBAND), combined with the SPHEREx weights, versus E11's 0.1 um box average.
Usage: OMP_NUM_THREADS=1 .venv/bin/python scripts/run_e32.py [fig]
"""
import json, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_e11 as E11

OUT = "data/findings/e32.json"
OUTS = "data/findings/e32s.json"
FIG = "data/findings/e32_wise0855"
SQ2PI = np.sqrt(2 * np.pi)
R_OLD = {"PRISM": (0.9657805520748421, 0.031029244468284443), "G395M": (0.9564636641352257, 0.030685832708893912)}

# name -> (shape, equivalent-width factor relative to CBAND, centre shift um)
MODELS = {"M0": ("gauss", 1.0, 0.0), "M1": ("tophat", 1.0, 0.0), "M2": ("gauss", 1.0644670, 0.0),   # FWHM = CBAND -> EW = 1.0645 CBAND
          "M3": ("gauss", 0.95, 0.0), "M4": ("gauss", 1.05, 0.0), "Cminus": ("gauss", 1.0, -0.010),
          "Cplus": ("gauss", 1.0, 0.010), "M5": ("gauss", 2.0, 0.0)}
PLAUSIBLE = ["M1", "M2", "M3", "M4", "Cminus", "Cplus"]


def load_jwst(name):
    p = {"PRISM": f"{E11.LIT}/wise0855/luhman2024_fig3_prism/dbf3.txt", "G395M": f"{E11.LIT}/wise0855/luhman2024_fig4_g395m/dbf4.txt"}[name]
    d = [l.split() for l in open(p).read().splitlines()[13:]]
    d = np.array([[float(x[0]), float(x[1])] for x in d if len(x) >= 2])
    return d[:, 0], d[:, 1] * 1e3


def through(lj, fj, cw, cb, model):
    shape, k, dc = MODELS[model]
    c, ew = cw + dc, k * cb
    if shape == "gauss":
        s = ew / SQ2PI
        lo, hi = c - 4 * s, c + 4 * s
    else:
        lo, hi = c - ew / 2, c + ew / 2
    if lo < lj.min() or hi > lj.max():
        return np.nan
    g = np.arange(lo, hi + 2.5e-4, 5e-4)
    T = np.exp(-0.5 * ((g - c) / s) ** 2) if shape == "gauss" else np.ones_like(g)
    w = T / g
    return float(np.trapezoid(np.interp(g, lj, fj) * w, g) / np.trapezoid(w, g))


def stat(F, E, P, det):
    ok = np.isfinite(P)
    d = det & ok
    r, er = F[d] / P[d], E[d] / P[d]
    wm, swm = E11.wmean(r, er)
    return dict(n_bins_detected=int(d.sum()), mean_ratio=wm, mean_ratio_err=swm,
                chi2_detected_bins=float(np.sum(((F[d] - P[d]) / E[d]) ** 2)), chi2_all_bins=float(np.sum(((F[ok] - P[ok]) / E[ok]) ** 2)),
                n_bins_all=int(ok.sum()), per_bin_ratio=[float(x) if np.isfinite(x) else None for x in F / P],
                per_bin_ratio_err=[float(x) if np.isfinite(x) else None for x in E / P])


def run():
    e11 = json.load(open(E11.OUT))["targets"]["wise0855"]
    bins = e11["bins"]
    rows = [r for r in json.load(open(f"{E11.P2}/wise0855_frames.json")) if r["flux"] is not None and r["chi2_local_dof"] <= 3]
    cb, dwl = [], []
    for r in rows:
        q = json.load(open(f"data/forced/wise0855_e4/{r['obs_id']}_D{r['det']}.json"))
        cb.append(q["bw_um"]), dwl.append(abs(q["wl_um"] - r["wl"]))
    wl = np.array([r["wl"] for r in rows]); e = np.array([r["err"] for r in rows]); f = np.array([r["flux"] for r in rows])
    cb = np.array(cb)
    F = np.array([b["flux"] for b in bins]); E = np.array([b["err"] for b in bins]); det = np.array([b["detected_3sigma"] for b in bins])
    members = [np.flatnonzero((wl >= b["lo"]) & (wl < b["hi"])) for b in bins]
    # SPHEREx step values must be reproduced from the same pictures and weights
    f_chk = max(abs(np.sum(f[m] / e[m] ** 2) / np.sum(1 / e[m] ** 2) - b["flux"]) for m, b in zip(members, bins))
    out = dict(experiment="E32", protocol="docs/protocols/E32.md", target="WISE 0855-0714", units=dict(wavelength="um", flux="mJy"),
               n_pictures=len(rows), max_abs_wl_difference_e11_vs_e4_um=float(max(dwl)), max_abs_spherex_step_reproduction_mJy=float(f_chk),
               cband_um=dict(min=float(cb.min()), median=float(np.median(cb)), max=float(cb.max())),
               resolving_power=dict(min=float((wl / cb).min()), median=float(np.median(wl / cb)), max=float((wl / cb).max())),
               models={k: dict(shape=v[0], equivalent_width_over_cband=v[1], centre_shift_um=v[2]) for k, v in MODELS.items()},
               steps=[dict(lo=b["lo"], hi=b["hi"], centre=b["centre"], spherex_mJy=b["flux"], spherex_err_mJy=b["err"], n=b["n"],
                           detected_3sigma=b["detected_3sigma"], cband_median_um=float(np.median(cb[m]))) for b, m in zip(bins, members)],
               spectra={})
    for name in ("PRISM", "G395M"):
        lj, fj = load_jwst(name)
        box = np.array([np.nan if v is None else v for v in E11.rebin_table(lj, fj, [b["lo"] for b in bins], [b["hi"] for b in bins])])
        res = dict(box_average=dict(binned_mJy=box.tolist(), **stat(F, E, box, det)), models={})
        for mname in MODELS:
            Pi = np.array([through(lj, fj, w_, c_, mname) for w_, c_ in zip(wl, cb)])
            Pb = np.array([np.sum(Pi[m] / e[m] ** 2) / np.sum(1 / e[m] ** 2) if np.isfinite(Pi[m]).all() else np.nan for m in members])
            s = stat(F, E, Pb, det)
            s.update(binned_mJy=Pb.tolist(), shift_vs_box=s["mean_ratio"] - res["box_average"]["mean_ratio"],
                     within_1sigma_of_e11=bool(abs(s["mean_ratio"] - R_OLD[name][0]) <= s["mean_ratio_err"]))
            res["models"][mname] = s
        shifts = [res["models"][k]["shift_vs_box"] for k in ["M0"] + PLAUSIBLE]
        res["shift_spread_plausible"] = dict(min=float(min(shifts)), max=float(max(shifts)))
        res["e11_reproduced"] = bool(abs(res["box_average"]["mean_ratio"] - R_OLD[name][0]) < 1e-9 and abs(res["box_average"]["mean_ratio_err"] - R_OLD[name][1]) < 1e-9)
        out["spectra"][name] = res
    P = out["spectra"]["PRISM"]
    m0 = P["models"]["M0"]
    stands = m0["within_1sigma_of_e11"]
    flips = [k for k in PLAUSIBLE if P["models"][k]["within_1sigma_of_e11"] != stands]
    out["verdict"] = dict(rule="stands if |r_new - r_old| <= sigma_new (M0, PRISM); response-model dependent if any of M1-M4, C-, C+ differs",
                          r_old=R_OLD["PRISM"][0], r_old_err=R_OLD["PRISM"][1], r_new=m0["mean_ratio"], r_new_err=m0["mean_ratio_err"],
                          difference=m0["mean_ratio"] - R_OLD["PRISM"][0], stands=stands, model_dependent=bool(flips), flipping_models=flips,
                          text=("the public figure stands" if stands else "the public figure does not stand") + ("; response-model dependent" if flips else ""))
    out["epsindib"] = "skipped: E11 has no JWST or other spectrum comparison for Epsilon Indi B (filter photometry only)"
    json.dump(out, open(OUT, "w"), indent=1)
    flat = dict(source=OUT, r_old=R_OLD["PRISM"][0], r_old_err=R_OLD["PRISM"][1], r_new=m0["mean_ratio"], r_new_err=m0["mean_ratio_err"],
                r_diff=out["verdict"]["difference"], stands=stands, model_dependent=bool(flips), n_steps_detected=m0["n_bins_detected"],
                n_pictures=len(rows), cband_median_um=out["cband_um"]["median"], R_median=out["resolving_power"]["median"],
                shift_min_plausible=P["shift_spread_plausible"]["min"], shift_max_plausible=P["shift_spread_plausible"]["max"])
    for name in ("PRISM", "G395M"):
        for k, s in out["spectra"][name]["models"].items():
            flat[f"{name}_{k}_ratio"], flat[f"{name}_{k}_err"] = s["mean_ratio"], s["mean_ratio_err"]
        flat[f"{name}_box_ratio"] = out["spectra"][name]["box_average"]["mean_ratio"]
        flat[f"{name}_box_err"] = out["spectra"][name]["box_average"]["mean_ratio_err"]
    json.dump(flat, open(OUTS, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("max_abs_wl_difference_e11_vs_e4_um", "max_abs_spherex_step_reproduction_mJy", "cband_um", "resolving_power", "verdict")}, indent=1))
    for name in ("PRISM", "G395M"):
        r = out["spectra"][name]
        print(name, "reproduced", r["e11_reproduced"], "box %.4f +- %.4f" % (r["box_average"]["mean_ratio"], r["box_average"]["mean_ratio_err"]))
        for k, s in r["models"].items():
            print("  %-6s %.4f +- %.4f  shift %+.4f  chi2det %.2f  within %s" % (k, s["mean_ratio"], s["mean_ratio_err"], s["shift_vs_box"], s["chi2_detected_bins"], s["within_1sigma_of_e11"]))
    for i, st in enumerate(out["steps"]):
        print("  %.1f-%.1f det=%d F=%.4f+-%.4f box=%.4f M0=%.4f  r_box=%.3f r_M0=%.3f+-%.3f" % (
            st["lo"], st["hi"], st["detected_3sigma"], st["spherex_mJy"], st["spherex_err_mJy"], P["box_average"]["binned_mJy"][i],
            m0["binned_mJy"][i], P["box_average"]["per_bin_ratio"][i], m0["per_bin_ratio"][i], m0["per_bin_ratio_err"][i]))


def fig():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    o = json.load(open(OUT))
    P = o["spectra"]["PRISM"]
    st = o["steps"]
    c = np.array([s["centre"] for s in st]); mid = np.array([(s["lo"] + s["hi"]) / 2 for s in st])
    F = np.array([s["spherex_mJy"] for s in st]); E = np.array([s["spherex_err_mJy"] for s in st]); det = np.array([s["detected_3sigma"] for s in st])
    box = np.array(P["box_average"]["binned_mJy"]); m0 = np.array(P["models"]["M0"]["binned_mJy"])
    lj, fj = load_jwst("PRISM")
    BLUE, ORANGE, VIOLET, INK, MUTED, GRID = E11.BLUE, E11.ORANGE, "#7b4fc9", E11.INK, E11.MUTED, "#e6e5df"
    plt.rcParams.update({"font.size": 12, "font.family": "DejaVu Sans", "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": INK, "text.color": INK})
    fg, (a, b) = plt.subplots(2, 1, figsize=(8, 7), dpi=150, sharex=True, gridspec_kw=dict(height_ratios=[3, 1.4], hspace=0.06))
    for ax in (a, b):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(True, color=GRID, lw=0.7), ax.set_axisbelow(True)
    s = (lj > 3.7) & (lj < 5.1)
    a.plot(lj[s], fj[s], color=MUTED, lw=0.8, alpha=0.6, zorder=1, label="JWST NIRSpec PRISM (Luhman et al. 2024)")
    a.plot(mid, box, "s", color=ORANGE, ms=7, mec="white", mew=1.0, zorder=3, label="JWST, 0.1 µm box average")
    a.plot(c, m0, "D", color=VIOLET, ms=7, mec="white", mew=1.0, zorder=4, label="JWST through SPHEREx response (M0)")
    a.errorbar(c[det], F[det], yerr=E[det], fmt="o", ms=7, color=BLUE, mec="white", mew=1.0, ecolor=BLUE, elinewidth=1.5, capsize=0, zorder=5,
               label="SPHEREx")
    a.errorbar(c[~det], F[~det], yerr=E[~det], fmt="o", ms=7, mfc="white", mec=BLUE, mew=1.5, ecolor=BLUE, elinewidth=1.5, capsize=0, zorder=5,
               label="SPHEREx, < 3σ")
    a.set_ylabel("F$_\\nu$ (mJy)")
    a.set_ylim(-0.03, 0.85)
    a.legend(loc="upper left", fontsize=10, frameon=False)
    rb = np.array([np.nan if v is None else v for v in P["box_average"]["per_bin_ratio"]]); eb = np.array([np.nan if v is None else v for v in P["box_average"]["per_bin_ratio_err"]])
    rm = np.array(P["models"]["M0"]["per_bin_ratio"], float); em = np.array(P["models"]["M0"]["per_bin_ratio_err"], float)
    b.axhline(1.0, color=MUTED, lw=1.0, zorder=1)
    for x, r, er, col, mk, dx in ((mid, rb, eb, ORANGE, "s", -0.012), (c, rm, em, VIOLET, "D", 0.012)):
        b.errorbar((x + dx)[det], r[det], yerr=er[det], fmt=mk, ms=6, color=col, mec="white", mew=0.8, ecolor=col, elinewidth=1.3, zorder=3)
        b.errorbar((x + dx)[~det], r[~det], yerr=er[~det], fmt=mk, ms=6, mfc="white", mec=col, mew=1.3, ecolor=col, elinewidth=1.3, zorder=3)
    b.set_ylabel("SPHEREx / JWST")
    b.set_xlabel("Wavelength (µm)")
    b.set_ylim(0.2, 1.8)
    b.set_xlim(3.75, 5.05)
    fg.subplots_adjust(left=0.11, right=0.98, top=0.98, bottom=0.09)
    fg.savefig(FIG + ".png", facecolor="white"), fg.savefig(FIG + ".svg", facecolor="white")
    print("figure written", FIG)


if __name__ == "__main__":
    fig() if sys.argv[1:] == ["fig"] else run()
