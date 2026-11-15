"""E7 runner (docs/PROTOCOLS.md E7, frozen 2026-11-15 11:05 PST). Resumable.

  export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  .venv/bin/python scripts/run_e7.py phot       # per-picture photometry, 6 fields -> data/findings/e7_runs/<field>.json
  .venv/bin/python scripts/run_e7.py gate0      # s_d on the 3 control fields only -> data/findings/e7_gate0.json (frozen)
  .venv/bin/python scripts/run_e7.py analyze    # T, p, a_w -> data/findings/e7.json, site/data/e7/*.png
"""
import json, os, sys, time
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings  # noqa: E402
from astropy.io import fits  # noqa: E402
from astropy.wcs import WCS, FITSFixedWarning  # noqa: E402
from build_field import J2000, position  # noqa: E402
from orrery import e6, e7  # noqa: E402
from orrery.change import world_to_grid  # noqa: E402
from orrery.data import load_stamp  # noqa: E402
from orrery.e1 import GAIA_EPOCH, wave_map  # noqa: E402
from orrery.fit import flux_mJy  # noqa: E402

warnings.simplefilter("ignore", FITSFixedWarning)
ROOT = "data/efields"
RUNS = "data/findings/e7_runs"
GATE = "data/findings/e7_gate0.json"
OUT = "data/findings/e7.json"
FIGS = "site/data/e7"
TARGETS = {"e7_rwaur": "rwaur", "e7_rdra": "rdra", "e7_epsindib": "epsindib"}
CONTROLS = ["e2_00", "e2_04", "e2_12"]
FIELDS = list(TARGETS) + CONTROLS
S, GMAX_REP, MIN_PICS = 48, 16.0, 20


def sha(p):
    import hashlib
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def sources(name, meta, gaia, mjd_mean):
    """Reported sources: Gaia G < 16 inside the central 40x40 grid px (mean epoch), blends merged, plus the target.
    Each: dict(id, G, target, members = Gaia row indices (G < 19 list) within BLEND_ARCSEC at J2016, or extra)."""
    g19 = gaia[gaia.phot_g_mean_mag < e6.GMAX].reset_index(drop=True)
    cosd = np.cos(np.radians(g19.dec.values))

    def members(ra, dec):
        d = np.hypot((g19.ra.values - ra) * cosd, g19.dec.values - dec) * 3600
        return sorted((int(i) for i in np.flatnonzero(d < e7.BLEND_ARCSEC)), key=lambda q: g19.phot_g_mean_mag[q])

    out, used = [], set()
    tgt = None
    if name in TARGETS:
        t = json.load(open(f"site/data/timemachine/{TARGETS[name]}.json"))["target"]
        ra16, de16 = position(t["ra_j2000"], t["dec_j2000"], t["pmra"], t["pmdec"], 0, J2000, GAIA_EPOCH)
        mem = members(float(np.ravel(ra16)[0]), float(np.ravel(de16)[0]))
        tgt = dict(id="target", name=t["name"], G=float(g19.phot_g_mean_mag[mem].min()) if mem else None, target=True,
                   members=mem, extra=None if mem else t,
                   gaia_ids=[int(g19.source_id[i]) for i in mem])
        used |= set(mem)
    ra, dec, G = e6.star_radec(gaia, mjd_mean, GMAX_REP)
    i, j = world_to_grid(ra, dec, S, meta["ra"], meta["dec"])
    g16 = gaia[gaia.phot_g_mean_mag < GMAX_REP].reset_index(drop=True)
    inside = (i >= 3.5) & (i <= 43.5) & (j >= 3.5) & (j <= 43.5)
    for k in np.argsort(G):
        if not inside[k]:
            continue
        mem = members(g16.ra[k], g16.dec[k])
        if set(mem) & used:                     # part of the target's or a brighter star's blend
            continue
        used |= set(mem)
        out.append(dict(id=str(int(g16.source_id[k])), G=float(G[k]), target=False, members=mem, extra=None,
                        gaia_ids=[int(g19.source_id[q]) for q in mem]))
    return ([tgt] if tgt else []) + out, g19


def phot_field(name):
    p = f"{RUNS}/{name}.json"
    if os.path.exists(p):
        return name
    t0 = time.time()
    d = f"{ROOT}/{name}"
    meta = json.load(open(f"{d}/field.json"))
    gaia = pd.read_parquet(f"{d}/gaia.parquet")
    mjds = [fl["mjd"] for fl in meta["frame_list"]]
    src, g19 = sources(name, meta, gaia, float(np.mean(mjds)))
    rec = {s["id"]: dict(mjd=[], det=[], lam=[], flux=[], err=[], npix=[], sat=[], key=[]) for s in src}
    for fl in meta["frame_list"]:
        st = load_stamp(f"{d}/{fl['file']}")
        w = WCS(fits.Header.fromstring(st.wcs_header))
        xy, _ = e6.star_xy_native(gaia, st.mjd, w, st.x0, st.y0)      # same order as g19 (G < 19 filter)
        cols = e6.star_columns(xy, st.detector, st.x0, st.y0, st.image.shape)
        groups, pos = [], []
        for s in src:
            if s["extra"] is not None:                                     # target not in Gaia G < 19
                t = s["extra"]
                ra, de = position(t["ra_j2000"], t["dec_j2000"], t["pmra"], t["pmdec"], t["plx"], J2000, st.mjd)
                x, y = w.all_world2pix(np.ravel(ra), np.ravel(de), 0)
                txy = (float(x[0]) - st.x0, float(y[0]) - st.y0)
                cols = cols + e6.star_columns([txy], st.detector, st.x0, st.y0, st.image.shape)
                groups.append([len(cols) - 1])
                pos.append(txy)
            else:
                groups.append(s["members"])
                pos.append(tuple(xy[s["members"][0]]))
        res = e7.forced_photometry(st.image, st.variance, st.flags, cols, groups)
        H, W = st.image.shape
        yy, xx = np.indices((H, W))
        for s, (a, e, npix), (x, y) in zip(src, res, pos):
            if not (0 <= x < W and 0 <= y < H):
                continue
            core = np.hypot(xx - x, yy - y) <= 1.0
            r = rec[s["id"]]
            r["mjd"].append(st.mjd), r["det"].append(st.detector), r["key"].append(fl["key"])
            r["lam"].append(float(wave_map(st.detector, x + st.x0, y + st.y0)))
            r["flux"].append(flux_mJy(a, st.omega_arcsec2) if np.isfinite(a) else None)
            r["err"].append(flux_mJy(e, st.omega_arcsec2) if np.isfinite(e) else None)
            r["npix"].append(npix), r["sat"].append(bool((st.flags[core] & e7.SAT_BITS).any()))
    out = dict(field=name, ra=meta["ra"], dec=meta["dec"], n_frames=len(meta["frame_list"]), sources=src,
               phot=rec, mask_bits=int(e7.PHOT_BITS), blend_arcsec=e7.BLEND_ARCSEC,
               runtime_s=round(time.time() - t0, 1))
    json.dump(out, open(p + ".tmp", "w"), separators=(",", ":"))
    os.replace(p + ".tmp", p)
    print(f"{name}: {len(src)} sources, {out['n_frames']} frames ({out['runtime_s']}s)", flush=True)
    return name


def phot():
    os.makedirs(RUNS, exist_ok=True)
    todo = [f for f in FIELDS if os.path.exists(f"{ROOT}/{f}/READY") and not os.path.exists(f"{RUNS}/{f}.json")]
    missing = [f for f in FIELDS if not os.path.exists(f"{ROOT}/{f}/READY")]
    if missing:
        print("not READY:", missing, flush=True)
    with Pool(min(3, max(1, len(todo)))) as p:
        list(p.imap_unordered(phot_field, todo))


# ---------------------------------------------------------------- per-star arrays
def star_data(r, scale=None):
    """Usable pictures of one source: finite flux and >= MIN_USABLE_PIX unmasked pixels under its ePSF box."""
    f = np.array([np.nan if v is None else v for v in r["flux"]], float)
    e = np.array([np.nan if v is None else v for v in r["err"]], float)
    ok = np.isfinite(f) & np.isfinite(e) & (e > 0) & (np.array(r["npix"]) >= e7.MIN_USABLE_PIX)
    det = np.array(r["det"])[ok]
    e = e[ok]
    if scale is not None:
        e = e * np.array([scale[str(d)] for d in det])
    return dict(flux=f[ok], err=e, lam=np.array(r["lam"])[ok], det=det, mjd=np.array(r["mjd"])[ok],
                n_all=len(f), n_usable=int(ok.sum()), sat_frac=float(np.mean(r["sat"])) if r["sat"] else None)


def load_runs():
    return {f: json.load(open(f"{RUNS}/{f}.json")) for f in FIELDS if os.path.exists(f"{RUNS}/{f}.json")}


def pull_summary(p):
    p = np.asarray(p, float)
    return dict(n=int(p.size), std=round(float(np.std(p)), 4), sigma_mad=round(float(1.4826 * np.median(np.abs(p - np.median(p)))), 4),
                median=round(float(np.median(p)), 4), frac_abs_gt3=round(float(np.mean(np.abs(p) > 3)), 4),
                frac_abs_gt5=round(float(np.mean(np.abs(p) > 5)), 4),
                q=[round(float(x), 3) for x in np.percentile(p, [0.5, 2.5, 16, 50, 84, 97.5, 99.5])])


def gate0():
    if os.path.exists(GATE):
        sys.exit(f"{GATE} exists (frozen)")
    R = load_runs()
    assert all(f in R for f in CONTROLS)
    red = {}
    pulls_before = {}
    stars = []
    for f in CONTROLS:
        for s in R[f]["sources"]:
            if s["target"]:
                continue
            dd = star_data(R[f]["phot"][s["id"]])
            if dd["n_usable"] < MIN_PICS:
                continue
            stars.append((f, s["id"], dd["n_usable"]))
            model, per, _, _ = e7.static_fit(dd["flux"], dd["err"], dd["lam"], dd["det"])
            for d_, (c2, n) in per.items():
                if n > 3:
                    red.setdefault(str(d_), []).append(c2 / (n - 3))
            pr = (dd["flux"] - model) / dd["err"]
            for d_ in np.unique(dd["det"]):
                pulls_before.setdefault(str(d_), []).extend(pr[dd["det"] == d_].tolist())
    s_d = {d_: float(np.sqrt(np.median(v))) for d_, v in sorted(red.items())}
    allb = np.concatenate([pulls_before[d_] for d_ in s_d])
    alla = np.concatenate([np.array(pulls_before[d_]) / s_d[d_] for d_ in s_d])
    out = dict(protocol="docs/PROTOCOLS.md E7 Gate 0 (frozen 2026-11-15 11:05 PST)", control_fields=CONTROLS,
               rule="per detector d: s_d = sqrt(median over control G < 16 stars of the static model's reduced chi2 "
                    "on that detector, chi2_d / (n_d - 3), stars with n_d > 3); errors multiplied by s_d; frozen and "
                    "applied to every field",
               s_d=s_d, n_star_detector_terms={d_: len(v) for d_, v in red.items()},
               stars=[dict(field=f, id=i, n_usable=n) for f, i, n in stars], n_stars=len(stars),
               min_usable_pictures=MIN_PICS,
               pulls_before=dict(all=pull_summary(allb), per_detector={d_: pull_summary(pulls_before[d_]) for d_ in s_d}),
               pulls_after=dict(all=pull_summary(alla),
                                per_detector={d_: pull_summary(np.array(pulls_before[d_]) / s_d[d_]) for d_ in s_d}),
               frozen_local=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               code_sha256={p: sha(p) for p in ("src/orrery/e7.py", "scripts/run_e7.py", "src/orrery/e6.py")})
    json.dump(out, open(GATE, "w"), indent=1)
    pull_plot(allb, alla)
    print(json.dumps({k: out[k] for k in ("s_d", "n_stars", "n_star_detector_terms")}, indent=1))
    print("before", out["pulls_before"]["all"])
    print("after ", out["pulls_after"]["all"])


def pull_plot(b, a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    bins = np.linspace(-8, 8, 81)
    ax.hist(np.clip(b, -8, 8), bins, histtype="step", color="#999999", lw=1.4, density=True, label="before s_d")
    ax.hist(np.clip(a, -8, 8), bins, histtype="step", color="#0072B2", lw=1.6, density=True, label="after s_d")
    x = np.linspace(-8, 8, 400)
    ax.plot(x, np.exp(-x * x / 2) / np.sqrt(2 * np.pi), color="k", lw=0.8, ls=":", label="unit normal")
    ax.set_xlabel("pull (flux - static model) / error, clipped at ±8")
    ax.set_ylabel("density")
    ax.set_title("E7 Gate 0: control-field G < 16 stars", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    os.makedirs(FIGS, exist_ok=True)
    fig.savefig(f"{FIGS}/gate0_pulls.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- analysis
def analyze_star(r, s_d, win_of):
    dd = star_data(r, s_d)
    res = dict(n_pictures=dd["n_all"], n_usable=dd["n_usable"], saturation_fraction=dd["sat_frac"])
    if dd["n_usable"] < MIN_PICS:
        res["status"] = f"FAILED: {dd['n_usable']} usable pictures (< {MIN_PICS})"
        return res, None
    win = np.array([win_of[m] for m in dd["mjd"]])
    model, per, chi_s, npar = e7.static_fit(dd["flux"], dd["err"], dd["lam"], dd["det"])
    c = e7.change_fit(dd["flux"], dd["err"], dd["lam"], dd["det"], win)
    sig = np.where(c["a_err"] > 0, np.abs(c["a"]) / np.where(c["a_err"] > 0, c["a_err"], 1), 0)
    res.update(status="ok", chi2_static=chi_s, dof_static=dd["n_usable"] - npar, chi2_changing=c["chi2"],
               T=chi_s - c["chi2"], windows=[int(w) for w in c["windows"]],
               a_w=[float(x) for x in c["a"]], a_err=[float(x) for x in c["a_err"]],
               max_abs_a_over_err=float(sig[1:].max()) if len(sig) > 1 else 0.0)
    return res, dict(dd=dd, win=win, static=model, change=c)


def pval(T, null):
    null = np.asarray(null, float)
    return float((1 + np.sum(null >= T)) / (1 + len(null))), int(len(null))


def analyze():
    gate = json.load(open(GATE))
    s_d = gate["s_d"]
    R = load_runs()
    out_fields, fits_ = {}, {}
    for f, r in R.items():
        all_mjd = np.array([fl for s in r["phot"].values() for fl in s["mjd"]])
        frames = sorted(set(all_mjd.tolist()))
        wf = e7.windows(frames)
        win_of = dict(zip(frames, wf))
        wins = [dict(window=int(w), mjd_min=float(np.min(np.array(frames)[wf == w])),
                     mjd_max=float(np.max(np.array(frames)[wf == w])), n_mjd=int((wf == w).sum())) for w in np.unique(wf)]
        stars = {}
        for s in r["sources"]:
            res, fit = analyze_star(r["phot"][s["id"]], s_d, win_of)
            stars[s["id"]] = dict(G=s["G"], target=s["target"], gaia_ids=s["gaia_ids"],
                                  name=s.get("name"), **res)
            fits_[(f, s["id"])] = fit
        out_fields[f] = dict(n_frames=r["n_frames"], windows=wins, stars=stars)
    nullc = [((f, sid), st["T"]) for f in CONTROLS for sid, st in out_fields[f]["stars"].items()
             if not st["target"] and st["status"] == "ok"]
    null_ctrl = [t for _, t in nullc]
    for f in out_fields:
        own = [((f, sid), st["T"]) for sid, st in out_fields[f]["stars"].items()
               if not st["target"] and st["status"] == "ok"]
        for sid, st in out_fields[f]["stars"].items():
            if st["status"] != "ok":
                continue
            st["p_control"], st["n_null_control"] = pval(st["T"], [t for k, t in nullc if k != (f, sid)])
            o = [t for k, t in own if k != (f, sid)]
            st["p_own_field"], st["n_null_own_field"] = pval(st["T"], o) if o else (None, 0)
            st["changed_between_visits"] = bool(st["p_control"] < 0.01 and st["max_abs_a_over_err"] > 3)
    targets = {}
    for f, tm in TARGETS.items():
        if f not in out_fields:
            targets[tm] = dict(status="FAILED: field not cached")
            continue
        st = out_fields[f]["stars"].get("target")
        targets[tm] = st
    min_p = 1 / (1 + len(null_ctrl))
    out = dict(experiment="E7", protocol="docs/PROTOCOLS.md E7 (frozen 2026-11-15 11:05 PST)", index="data/index/frames_dr1.parquet",
               raw_cache=ROOT, fields=list(out_fields), control_fields=CONTROLS, gate0_file=GATE, s_d=s_d,
               photometry=dict(mask="orrery.data.BAD_BITS | OVERFLOW (1) | NONLINEAR (15)", stars="Gaia DR3 G < 19 "
                               "(proper motion to picture epoch) + target, zone ePSF per star, background plane, WLS with "
                               "pipeline variance (orrery.e6 / orrery.fit machinery)", flux="mJy = A * Omega (orrery.fit.flux_mJy)",
                               wavelength="WCS-WAVE central wavelength at the star (orrery.e1.wave_map); CWAVE/CBAND "
                                          "per-pixel product not used (~1 nm approximation, stated limitation)",
                               blends=f"a reported source = sum of Gaia G < 19 amplitudes within {e7.BLEND_ARCSEC} arcsec "
                                      "(J2016), error sqrt(1'Cov1); a G < 16 star inside a brighter source's blend is not "
                                      "reported separately",
                               usable=f"finite amplitude and >= {e7.MIN_USABLE_PIX} unmasked pixels under the group's "
                                      f"ePSF box; a star needs >= {MIN_PICS} usable pictures, else FAILED",
                               saturation_fraction="fraction of pictures with an OVERFLOW or NONLINEAR pixel within 1 px "
                                                   "of the source"),
               null=dict(n_control_stars=len(null_ctrl), T_control=sorted(null_ctrl),
                         smallest_attainable_p=min_p,
                         note="p = (1 + #null T >= T) / (1 + n_null); with n_null control stars the smallest attainable "
                              f"p is {min_p:.3f}" + (" — p < 0.01 is unreachable" if min_p >= 0.01 else "")),
               verdict_rule="changed between visits iff p_control < 0.01 AND max |a_w| / err > 3",
               targets=targets, per_field=out_fields,
               code_sha256={p: sha(p) for p in ("src/orrery/e7.py", "scripts/run_e7.py", "src/orrery/e6.py")},
               generated_local=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    json.dump(out, open(OUT, "w"), indent=1, default=float)
    # plots: targets + 2 typical constant control stars (T nearest the control-null median)
    med = np.median(null_ctrl) if null_ctrl else 0
    ctrl = sorted([(abs(st["T"] - med), f, sid) for f in CONTROLS for sid, st in out_fields[f]["stars"].items()
                   if st["status"] == "ok"])[:2]
    plots = [(f, "target", tm) for f, tm in TARGETS.items() if f in out_fields] + \
            [(f, sid, f"const_{f}_{sid}") for _, f, sid in ctrl]
    for f, sid, tag in plots:
        st = out_fields[f]["stars"][sid]
        title = (st.get("name") or f"constant star Gaia DR3 {sid}, G = {st['G']:.1f} ({f})")
        lt_plot(fits_.get((f, sid)), st, title, f"{FIGS}/{tag}_lt.png", out_fields[f]["windows"])
    print(json.dumps({k: {q: v.get(q) for q in ("status", "n_usable", "n_pictures", "saturation_fraction", "T",
                                                 "p_control", "p_own_field", "max_abs_a_over_err",
                                                 "changed_between_visits", "a_w", "a_err")} if v else None
                      for k, v in targets.items()}, indent=1, default=float))
    print("null:", json.dumps(out["null"], default=float))


def lt_plot(fit, st, title, path, wins):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.6, 3.9), gridspec_kw=dict(width_ratios=[2.2, 1]))
    if fit is None:
        ax.text(0.5, 0.5, f"fit failed: {st['status']}\nsaturation fraction = {st['saturation_fraction']}",
                ha="center", va="center", transform=ax.transAxes, fontsize=9)
        bx.axis("off")
    else:
        dd, c = fit["dd"], fit["change"]
        sc = ax.scatter(dd["lam"], dd["flux"], c=dd["mjd"], s=9, cmap="viridis", zorder=3, lw=0)
        ax.errorbar(dd["lam"], dd["flux"], dd["err"], fmt="none", ecolor="0.75", lw=0.6, zorder=2)
        A, dets = e7._design(dd["lam"], dd["det"])
        coef, *_ = np.linalg.lstsq(A / dd["err"][:, None], dd["flux"] / dd["err"], rcond=None)
        for j, d in enumerate(dets):
            s = dd["det"] == d
            lam = np.linspace(dd["lam"][s].min(), dd["lam"][s].max(), 50)
            x = lam - dd["lam"][s].mean()
            ax.plot(lam, coef[3 * j] + coef[3 * j + 1] * x + coef[3 * j + 2] * x * x, color="#D55E00", lw=1.3,
                    zorder=4, label="static model" if j == 0 else None)
        cb = fig.colorbar(sc, ax=ax, pad=0.01)
        cb.set_label("MJD", fontsize=8)
        ax.legend(fontsize=7, frameon=False)
        wm = {w["window"]: 0.5 * (w["mjd_min"] + w["mjd_max"]) for w in wins}
        x = [wm[int(w)] for w in c["windows"]]
        bx.errorbar(x, c["a"], np.where(np.isfinite(c["a_err"]), c["a_err"], 0), fmt="o", color="#0072B2", ms=4, capsize=2)
        bx.axhline(0, color="0.6", lw=0.8)
        bx.set_xlabel("window mid MJD")
        bx.set_ylabel("brightness factor a_w (first = 0)")
        bx.set_title(f"T = {st['T']:.1f}, p_control = {st.get('p_control', float('nan')):.3f}", fontsize=8)
        bx.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("wavelength at the star (µm, WCS-WAVE)")
    ax.set_ylabel("flux (mJy)")
    ax.set_title(f"{title}\n{st['n_usable']} usable of {st['n_pictures']} pictures; saturation fraction {st['saturation_fraction']:.2f}", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    os.makedirs(FIGS, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    {"phot": phot, "gate0": gate0, "analyze": analyze}[sys.argv[1]]()
