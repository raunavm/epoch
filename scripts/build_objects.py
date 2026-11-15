"""Object pages + finding 3 from forced photometry.

For each data/forced/<id>.parquet: reflectance (λ < λ_max where modelled thermal < 1%), NEATM size
(orrery.analysis.fit_size), Gaia DR3 shape check, 3 µm band depth, plain-language summary, per-frame
thumbnails (data / model / residual) and provenance. Writes site/data/objects.json,
site/data/objects/<id>.json, site/data/objects/<id>/*.png and data/findings/f3.json.
Usage: .venv/bin/python scripts/build_objects.py [--ids a,b]
"""
import argparse, glob, json, os, subprocess, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")
from orrery.analysis import fit_size, reflectance, frame_errors, visit_groups
from orrery.validate import gaia_compare, band_depth_3um
from orrery.physics import neatm_mJy, reflected_mJy
from orrery.truth import sbdb, gaia_reflectance
from orrery.data import load_stamp

FAMILIAR = [(2.7, "the Golden Gate Bridge (2.7 km long)"), (8.8, "Mount Everest's height above sea level (8.8 km)"),
            (21.6, "the length of Manhattan (21.6 km)"), (33, "the Strait of Dover (33 km wide)"),
            (73, "the length of Lake Geneva (73 km)"), (446, "the length of the Grand Canyon (446 km)")]
CLASS_PLAIN = {"MBA": "main asteroid belt, between Mars and Jupiter", "OMB": "outer main belt", "IMB": "inner main belt",
               "MCA": "Mars-crosser", "TJN": "Jupiter Trojan, sharing Jupiter's orbit", "AMO": "near-Earth (Amor)",
               "APO": "near-Earth (Apollo, crosses Earth's orbit)", "ATE": "near-Earth (Aten)", "CEN": "Centaur, between Jupiter and Neptune"}


def familiar(D):
    best = min(FAMILIAR, key=lambda f: abs(np.log(D / f[0])))
    return dict(name=best[1], km=best[0], ratio=round(D / best[0], 2))


def plain_composition(pV, bd, slope):
    words = []
    if pV is not None:
        if pV < 0.10:
            words.append("Very dark surface (reflects under 10% of sunlight), typical of carbon-rich, primitive material")
        elif pV < 0.30:
            words.append("Moderately bright surface, typical of stony (rock and metal) asteroids")
        else:
            words.append("Unusually bright surface, reflecting over 30% of sunlight")
    if bd is not None and bd["err"] > 0:
        if bd["depth"] / bd["err"] > 3:
            words.append(f"a dip near 3 µm ({100*bd['depth']:.0f} ± {100*bd['err']:.0f}%) of the kind caused by water-bearing minerals")
        else:
            words.append("no significant water-mineral dip near 3 µm at this precision")
    return "; ".join(words) + "." if words else ""


def thumb(arr, path, vmin, vmax, cmap):
    fig = plt.figure(figsize=(1.0, 1.0), dpi=64)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(arr, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_axis_off()
    fig.savefig(path)
    plt.close(fig)


def code_version():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "uncommitted"


def build(pdes, ver):
    df = pd.read_parquet(f"data/forced/{pdes}.parquet")
    if len(df) == 0 or "snr" not in df:
        return None
    df = df[np.isfinite(df.flux_mJy)].sort_values("mjd")
    good = df[df.snr > 5]
    if len(good) < 8:
        return None
    t = sbdb(pdes)
    H, G = t["H"], (t["G"] if t["G"] is not None else 0.15)
    # wavelength limit for reflectance: modelled thermal < 1% of reflected (uses SBDB D, pV, η=1 as a MODEL check)
    lam_max = 3.15
    if t.get("diameter") and t.get("albedo"):
        rr, dd, aa = good.r.median(), good.delta.median(), good.alpha.median()
        for w in np.arange(2.0, 3.2, 0.05):
            th = neatm_mJy(w, t["diameter"], t["albedo"], 1.0, rr, dd, aa, G=G, n_grid=24)
            rf = reflected_mJy(w, H, G, rr, dd, aa)
            if th > 0.01 * rf:
                lam_max = float(w)
                break
    rsel = good[good.wl_um < lam_max]
    R, Re, _ = reflectance(rsel, H, G)
    size = fit_size(df, H, G)
    gaia = None
    try:
        num = int(pdes)
        g = gaia_reflectance(num)
        if g is not None and len(g):
            gaia = g
    except Exception:
        pass
    gc = gaia_compare(rsel.wl_um.values, R, Re, gaia, visit_groups(rsel.mjd.values)) if gaia is not None else None
    bd = band_depth_3um(rsel.wl_um.values, R, Re) if lam_max >= 3.15 else None
    outdir = f"site/data/objects/{pdes}"
    os.makedirs(outdir, exist_ok=True)
    # thumbnails for up to 40 frames spread over the track
    sel = set(np.linspace(0, len(df) - 1, min(40, len(df))).astype(int))
    frames = []
    for k, (_, row) in enumerate(df.iterrows()):
        fr = dict(obs_id=row.obs_id, detector=int(row.detector), mjd=round(row.mjd, 6), wl_um=round(row.wl_um, 4),
                  bw_um=round(row.bw_um, 4), flux_mJy=round(row.flux_mJy, 4), flux_err_mJy=round(float(np.hypot(row.flux_err_mJy, 0.03 * row.flux_mJy)), 4),
                  snr=round(row.snr, 1), chi2_dof_local=round(row.chi2_local / max(row.dof_local, 1), 2),
                  s3_url=f"https://nasa-irsa-spherex.s3.amazonaws.com/{row.key}")
        f = f"data/stamps/{pdes}/{row.obs_id}_D{row.detector}.npz"
        if k in sel and os.path.exists(f):
            st = load_stamp(f)
            z = np.load(f.replace(".npz", "_fit.npz"))
            sig = np.sqrt(np.maximum(st.variance, 1e-12))
            med = np.median(st.image)
            pk = max(np.percentile(st.image - med, 99.5), 1e-6)
            base = f"{row.obs_id}_D{row.detector}"
            thumb(np.arcsinh((st.image - med) / sig), f"{outdir}/{base}_d.png", -2, np.arcsinh(pk / np.median(sig)), "magma")
            thumb(np.arcsinh((z["model"] - med) / sig), f"{outdir}/{base}_m.png", -2, np.arcsinh(pk / np.median(sig)), "magma")
            thumb(np.where(z["mask"], z["resid"] / sig, 0), f"{outdir}/{base}_r.png", -5, 5, "RdBu_r")
            fr.update(stamp_png=f"objects/{pdes}/{base}_d.png", model_png=f"objects/{pdes}/{base}_m.png", resid_png=f"objects/{pdes}/{base}_r.png")
        frames.append(fr)
    eph = pd.read_parquet(f"data/ephem/{pdes}.parquet")
    obs_mjd = df.mjd.values
    near = eph[np.min(np.abs(eph.mjd.values[:, None] - obs_mjd[None, :]), axis=1) < 3]
    D_truth = t.get("diameter")
    rec = dict(id=str(pdes), name=t.get("fullname", str(pdes)).strip(), kind=t.get("orbit_class"),
               class_plain=CLASS_PLAIN.get(t.get("orbit_class"), t.get("orbit_class_name", "")),
               release="dr1", code_version=ver, rerun_command=f".venv/bin/python scripts/run_forced.py --ids {pdes} --stamps --redo && .venv/bin/python scripts/build_objects.py --ids {pdes}",
               track=[dict(mjd=round(r.mjd, 3), ra=round(r.RA, 5), dec=round(r.DEC, 5)) for r in near.itertuples()],
               frames=frames,
               spectrum=dict(wl_um=[round(x, 4) for x in rsel.wl_um], refl=[round(x, 4) for x in R], refl_err=[round(x, 4) for x in Re],
                             lam_max_um=round(lam_max, 2), normalisation="R = F_obs / F_sun-reflected(H, G, geometry); 1 = V-band level",
                             gaia=None if gaia is None else dict(wl_um=gaia.wavelength_um.round(4).tolist(), refl=gaia.reflectance.round(4).tolist(),
                                                                err=gaia.err.round(4).tolist(), scale_to_ours=None if gc is None else round(gc["k"], 4)),
                             gaia_check=gc, band_3um=bd),
               size=None if size is None else dict(D_km=round(size["D_km"], 2), D_lo_km=round(size["D_lo"], 2), D_hi_km=round(size["D_hi"], 2),
                                                   pV=round(size["pV"], 3), pV_p16=round(size["pV_p16"], 3), pV_p84=round(size["pV_p84"], 3),
                                                   eta=round(size["eta"], 2), n_thermal=size["n_thermal"], chi2_dof=round(size["chi2_dof"], 2),
                                                   truth=dict(D_km=D_truth, D_sigma_km=t.get("diameter_sigma"), pV=t.get("albedo"), source=t.get("diameter_ref")),
                                                   familiar=familiar(size["D_km"])),
               composition_plain=plain_composition(size["pV"] if size else t.get("albedo"), bd, None),
               labels={"D_km": "derived", "pV": "derived", "eta": "derived", "refl": "derived", "flux_mJy": "data"})
    json.dump(rec, open(f"site/data/objects/{pdes}.json", "w"), separators=(",", ":"))
    return dict(id=str(pdes), name=rec["name"], number=str(pdes), kind=rec["kind"], n_frames=len(df),
                D_km=None if size is None else round(size["D_km"], 2),
                D_err_km=None if size is None else round((size["D_hi"] - size["D_lo"]) / 2, 2),
                sigma_lnD=None if size is None else round(size["sigma_lnD"], 3),
                D_truth_km=D_truth, D_truth_source=t.get("diameter_ref"), pV=None if size is None else round(size["pV"], 3),
                class_plain=rec["class_plain"], page=f"object.html?id={pdes}",
                gaia_chi2_dof=None if gc is None else round(gc["chi2"] / max(gc["dof"], 1), 2))


def main(ids):
    ver = code_version()
    rows = []
    for p in ids:
        try:
            r = build(p, ver)
        except Exception as e:
            print("FAIL", p, repr(e), flush=True)
            continue
        if r:
            rows.append(r)
            print(r, flush=True)
    path = "site/data/objects.json"
    old = {o["id"]: o for o in json.load(open(path))} if os.path.exists(path) else {}
    for r in rows:
        old[r["id"]] = r
    json.dump(list(old.values()), open(path, "w"), indent=1)
    o = pd.DataFrame(list(old.values()))
    v = o.dropna(subset=["D_km", "D_truth_km"])
    if len(v):
        lr = np.log(v.D_km / v.D_truth_km)
        mad = 1.4826 * np.median(np.abs(lr - np.median(lr)))
        pull = lr / np.sqrt(v.sigma_lnD.astype(float) ** 2 + (v.D_truth_km * 0 + 0.10) ** 2)  # NEOWISE ~10% (Mainzer et al. 2011)
        f3 = dict(n=int(len(v)), pull_std=round(float(np.std(pull)), 2), median_ratio=round(float(np.exp(np.median(lr))), 3),
                  scatter_frac_robust=round(float(mad), 3), scatter_frac_std=round(float(np.std(lr)), 3),
                  within_20pct=round(float(np.mean(np.abs(lr) < np.log(1.2))), 3),
                  gaia_n=int(o.gaia_chi2_dof.notna().sum()),
                  gaia_chi2dof_median=None if o.gaia_chi2_dof.notna().sum() == 0 else round(float(o.gaia_chi2_dof.median()), 2))
        os.makedirs("data/findings", exist_ok=True)
        json.dump(f3, open("data/findings/f3.json", "w"), indent=1)
        print(json.dumps(f3, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default="")
    a = ap.parse_args()
    ids = [x for x in a.ids.split(",") if x] or [os.path.basename(f)[:-8] for f in sorted(glob.glob("data/forced/*.parquet"))]
    main(ids)
