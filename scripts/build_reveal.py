"""Finding 2: one real known asteroid invisible in every single frame, revealed by stacking along its orbit.

Steps: forced photometry (scripts/run_forced.py output + stamps) for candidate faint asteroids ->
choose the object with max single-frame S/N < 5 (the standard 5σ detection threshold) and the highest stack S/N -> cumulative stacks
(n = 1..N in time order) -> PNGs + site/data/reveal.json + data/findings/f2.json.
Null test: the same frames stacked on tracks displaced by fixed offsets (off-orbit) must give
S/N consistent with zero.
Usage: .venv/bin/python scripts/build_reveal.py --ids-file data/faint_ids.txt [--pick <pdes>]
"""
import argparse, glob, json, os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")
from orrery.data import load_stamp
from orrery.stack import scene_residual, combine, amp_from_mJy, centre
from orrery.physics import reflected_mJy, neatm_mJy
from orrery.truth import sbdb

HALF = 15


def expected_mJy(row, H, G, D, pV):
    f = reflected_mJy(row.wl_um, H, G, row.r, row.delta, row.alpha)
    if row.wl_um > 2.8 and D and pV:
        f += neatm_mJy(row.wl_um, D, pV, 1.0, row.r, row.delta, row.alpha, G=G, n_grid=32)
    return float(f)


def items_for(pdes, df, offset=(0.0, 0.0)):
    t = sbdb(pdes)
    H, G = t["H"], t["G"] or 0.15
    D, pV = t.get("diameter"), t.get("albedo")
    items = []
    # data-quality rule, applied identically to single pictures and the stack: drop frames where the
    # fitted scene is poor near the object (local chi2/dof > 3: an uncatalogued source, cosmic ray or artefact).
    df = df[(df.chi2_local / df.dof_local.clip(lower=1)) <= 3]
    for _, row in df.sort_values("mjd").iterrows():
        f = f"data/stamps/{pdes}/{row.obs_id}_D{row.detector}.npz"
        if not os.path.exists(f):
            continue
        st = load_stamp(f)
        z = np.load(f.replace(".npz", "_fit.npz"))
        xs, ys = float(z["xs"]) + offset[0], float(z["ys"]) + offset[1]
        res, w, r = scene_residual(st, xs, ys, z["star_xy"])
        p = amp_from_mJy(expected_mJy(row, H, G, D, pV), st.omega_arcsec2)
        items.append(dict(res=res, w=w, xs=xs, ys=ys, p=p, A=r.A, A_err=r.A_err, mjd=row.mjd, wl=row.wl_um,
                          obs_id=row.obs_id, det=int(row.detector), img=st.image, key=row.key,
                          f_mJy=row.flux_mJy, f_err=row.flux_err_mJy, snr=r.A / r.A_err))
    return items, dict(H=H, G=G, D=D, pV=pV, D_ref=t.get("diameter_ref"), name=(t.get("fullname") or str(pdes)).strip())


def null_calibration(pdes, df):
    """E5 (docs/PROTOCOLS.md): 24 off-path tracks; scale estimated on the 12 inner tracks (6.5 px), verified on
    the 12 outer tracks (8.5 px): rescaled outer spread must be 0.9–1.1, else the larger scale is used.
    Returns (stack_mean, stack_scale, single_std, all_null_snrs, check dict)."""
    inner, outer, singles = [], [], []
    for rad, dst in ((6.5, inner), (8.5, outer)):
        for a in np.arange(12) * np.pi / 6 + (0 if rad == 6.5 else np.pi / 12):
            it, _ = items_for(pdes, df, offset=(rad * np.cos(a), rad * np.sin(a)))
            _, _, A, E = combine(it, HALF)
            dst.append(A / E)
            singles += [i["snr"] for i in it if np.isfinite(i["snr"])]
    m_in, s_in = float(np.mean(inner)), float(np.std(inner, ddof=1))
    s_out_rescaled = float(np.std(outer, ddof=1) / s_in)
    scale = s_in if 0.9 <= s_out_rescaled <= 1.1 else max(s_in, float(np.std(outer, ddof=1)))
    check = dict(scale_inner=round(s_in, 3), outer_spread_after_rescale=round(s_out_rescaled, 3),
                 passed=bool(0.9 <= s_out_rescaled <= 1.1), scale_used=round(scale, 3))
    return m_in, scale, float(np.std(singles)), inner + outer, check


def detection_image(arr, half=HALF):
    """PSF-scale smoothing (Gaussian σ = 0.7 px ≈ SPHEREx PSF core), normalised by the robust rms of the
    outer region (r > 8 px from centre) of the same image — identical treatment for singles and stacks."""
    from scipy.ndimage import gaussian_filter
    sm = gaussian_filter(np.nan_to_num(arr), 0.7)
    yy, xx = np.mgrid[:sm.shape[0], :sm.shape[1]]
    out = np.hypot(xx - half, yy - half) > 8
    v = sm[out]
    rms = 1.4826 * np.median(np.abs(v - np.median(v)))
    return (sm - np.median(v)) / (rms if rms > 0 else 1)


def png(arr, path, vmin, vmax, cmap="magma", mark=None):
    fig = plt.figure(figsize=(2.4, 2.4), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(arr, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    if mark is not None:
        ax.add_patch(plt.Circle(mark, 3.2, fill=False, ec="#56B4E9", lw=1.2, ls="--"))
    ax.set_axis_off()
    fig.savefig(path)
    plt.close(fig)


def main(ids, pick):
    stats = []
    for pdes in ids:
        f = f"data/forced/{pdes}.parquet"
        if not os.path.exists(f):
            continue
        df = pd.read_parquet(f)
        if len(df) < 20 or "snr" not in df:
            continue
        items, meta = items_for(pdes, df)
        if len(items) < 20:
            continue
        _, _, a, ae = combine(items, HALF)
        if a / ae < 5:  # cannot reach a calibrated 5σ (null spread ≥ 1)
            stats.append(dict(pdes=pdes, n=len(items), max_single=float(np.nanmax([i["snr"] for i in items])),
                              med_single=float(np.nanmedian([i["snr"] for i in items])), stack=float(a / ae), V=float(df.V.median())))
            print(stats[-1], flush=True)
            continue
        m0, s0, s1, _, _ = null_calibration(pdes, df)
        stats.append(dict(pdes=pdes, n=len(items), max_single=float(np.nanmax([i["snr"] for i in items]) / s1),
                          med_single=float(np.nanmedian([i["snr"] for i in items]) / s1), stack=float((a / ae - m0) / s0),
                          stack_nominal=float(a / ae), null_std=s0, single_std=s1, V=float(df.V.median())))
        print(stats[-1], flush=True)
    st = pd.DataFrame(stats)
    os.makedirs("data/findings", exist_ok=True)
    st.to_csv("data/findings/f2_candidates.csv", index=False)
    if pick is None:
        ok = st[(st["max_single"] < 5) & (st["stack"] >= 5)]
        if len(ok) == 0:
            print("NO CANDIDATE satisfies max single-frame S/N < 5 (the standard 5σ detection threshold); finding 2 not met with this sample")
            return
        pick = ok.sort_values("stack", ascending=False)["pdes"].iloc[0]
    df = pd.read_parquet(f"data/forced/{pick}.parquet")
    items, meta = items_for(pick, df)
    # combination order: by EXPECTED per-picture signal-to-noise (brightness model / noise; never the measured flux)
    items = sorted(items, key=lambda i: -i["p"] / i["A_err"])
    N = len(items)
    out = "site/data/reveal"
    os.makedirs(out, exist_ok=True)
    # display ranges: singles in σ-units of each frame; stack in fraction-of-expected units
    frames, stacks, spec = [], [], []
    ns = sorted(set([1, 2, 3, 5, 8, 12, 18, 25, 35, 50, 70, 100, 140, 200, N]))
    ns = [n for n in ns if n <= N]
    for k, it in enumerate(items):
        sig = np.sqrt(1 / np.maximum(it["w"], 1e-30))
        z = np.where(it["w"] > 0, it["res"], 0)  # stars removed, same as the stack
        z = detection_image(centre(z, it["xs"], it["ys"], HALF))
        name = f"single_{k:03d}.png"
        png(z, f"{out}/{name}", -3, 6, cmap="gray", mark=(HALF, HALF))
        frames.append(dict(obs_id=it["obs_id"], detector=it["det"], mjd=it["mjd"], wl_um=round(it["wl"], 4),
                           single_png=f"reveal/{name}", snr=round(float(it["snr"]), 2), s3_key=it["key"]))
    for n in ns:
        img, snr_map, a, ae = combine(items[:n], HALF)
        name = f"stack_{n:03d}.png"
        png(detection_image(img), f"{out}/{name}", -3, 6, cmap="gray", mark=(HALF, HALF))
        stacks.append(dict(n=n, png=f"reveal/{name}", snr=round(float(a / ae), 2), a_hat=round(float(a), 3), a_err=round(float(ae), 3)))
        # spectrum: weighted mean flux in wavelength bins using frames so far
        sub = items[:n]
        edges = np.array([0.7, 1.0, 1.3, 1.65, 2.0, 2.45, 2.9, 3.4, 3.9, 4.45, 5.05])
        wl, fl, er = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            s = [i for i in sub if lo <= i["wl"] < hi]
            if not s:
                continue
            w_ = np.array([1 / i["f_err"] ** 2 for i in s])
            wl.append(round(float(np.sum(w_ * [i["wl"] for i in s]) / w_.sum()), 3))
            fl.append(round(float(np.sum(w_ * [i["f_mJy"] for i in s]) / w_.sum()), 5))
            er.append(round(float(1 / np.sqrt(w_.sum())), 5))
        spec.append(dict(n_frames_used=n, wl_um=wl, flux_mJy=fl, err=er))
    # null test + noise calibration on 24 off-orbit tracks (static scene refit there)
    m0, s0, s1, nulls, e5 = null_calibration(pick, df)
    for stk in stacks:
        stk["snr_nominal"] = stk["snr"]
        stk["snr"] = round((stk["snr"] - m0) / s0, 2)  # calibrated significance
    for fr in frames:
        fr["snr"] = round(fr["snr"] / s1, 2)
    single = np.array([i["snr"] for i in items]) / s1
    stack_snr = stacks[-1]["snr"]
    # depth gain: typical single-frame 1σ error vs stack 1σ error, both in units of the expected brightness
    s_single = np.array([i["A_err"] / i["p"] for i in items])
    gain = 2.5 * np.log10(np.median(s_single) * s1 / (stacks[-1]["a_err"] * s0))  # with empirical noise calibration
    ideal = 2.5 * np.log10(np.median(s_single) * np.sqrt(np.sum(1 / s_single ** 2)))
    n_all = len(pd.read_parquet(f"data/forced/{pick}.parquet"))
    rev = dict(object=meta["name"], designation=f"{pick}", n_frames_before_quality_cut=n_all, meta=meta, frames=frames, stack_pngs=stacks, spectrum_points=spec,
               depth_gain_mag=None if gain is None else round(float(gain), 2), ideal_gain_mag=round(float(ideal), 2),
               noise_calibration=dict(stack_null_mean=round(m0, 2), stack_null_std=round(s0, 2), single_std=round(s1, 2), n_null_tracks=24, e5_check=e5),
               single_frame_snr_max=round(float(np.nanmax(single)), 2), single_frame_snr_median=round(float(np.median(single)), 2),
               stack_snr=stack_snr, null_snrs=[round(x, 2) for x in nulls], n_frames=N,
               order="pictures are combined in order of their expected signal-to-noise (brightness model and noise only; measured fluxes are not used to choose the order)",
               units="singles and stacks: stars removed, smoothed at the PSF scale, in units of each image's own background noise (same treatment for both)",
               rerun="python scripts/run_forced.py --ids %s --stamps && python scripts/build_reveal.py --pick %s" % (pick, pick))
    json.dump(rev, open("site/data/reveal.json", "w"), indent=1)
    f2 = dict(object=meta["name"], designation=pick, n_frames=N, n_frames_dropped_quality=n_all - N, single_frame_snr_max=rev["single_frame_snr_max"],
              single_frame_snr_median=rev["single_frame_snr_median"], stack_snr=stack_snr,
              null_snr_mean=round(float((np.mean(nulls) - m0) / s0), 2), null_snr_std=1.0, null_nominal_std=round(s0, 2),
              single_std=round(s1, 2), e5_check=e5, significance_note="scale from 12 inner empty tracks, verified on 12 outer tracks (docs/PROTOCOLS.md E5)",
              depth_gain_mag=rev["depth_gain_mag"], ideal_gain_mag=rev["ideal_gain_mag"])
    json.dump(f2, open("data/findings/f2.json", "w"), indent=1)
    print(json.dumps(f2, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids-file", default="data/faint_ids.txt")
    ap.add_argument("--pick", default=None)
    a = ap.parse_args()
    ids = [l.strip() for l in open(a.ids_file) if l.strip()]
    main(ids, a.pick)
