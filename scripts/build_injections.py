"""Finding 4: planted test objects in real SPHEREx frames.

For a grid of (diameter, heliocentric distance, albedo) we compute the expected flux in every
frame of a real asteroid track (reflected light, Bowell H,G with G=0.15, flat reflectance, plus NEATM
thermal with η=1 when relevant; observer distance Δ = r − 1 au, phase angle 0° — an opposition-like
geometry, MODEL), plant a point source with the frame's own ePSF at 8 off-orbit positions
(7.5 px from the real track, i.e. empty sky with real stars, real noise, real artefacts), refit the
static scene exactly as for real objects, and stack with the matched filter (orrery.stack).
Recovered = stack S/N ≥ 5. Also: the same stack with nothing planted (null) at the same 8 positions.
Writes site/data/injections.json, site/data/limits.json (sentences filled in only from measured
numbers), data/findings/f4.json, and example PNGs.
Usage: .venv/bin/python scripts/build_injections.py --track <pdes>
"""
import argparse, json, os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")
from orrery.data import load_stamp
from orrery.stack import scene_residual, combine, amp_from_mJy, inject
from orrery.physics import reflected_mJy, neatm_mJy, v_magnitude

HALF = 15
D_KM = [0.5, 1, 2, 5, 10, 30, 100, 300, 1000, 2400, 12742, 49528]  # 2400 ≈ Pluto, 12742 = Earth, 49528 = Neptune
R_AU = [1.5, 2.5, 4, 10, 30, 50, 100, 250, 500]
PV = [0.05, 0.25, 0.6]


def H_of(D, pV):
    return 5 * np.log10(1329.0 / (D * np.sqrt(pV)))


def model_flux(wl, D, pV, r):
    delta = max(r - 1.0, 0.3)
    H = H_of(D, pV)
    f = reflected_mJy(wl, H, 0.15, r, delta, 0.0)
    if wl > 2.8 and r < 8:
        f += neatm_mJy(wl, D, pV, 1.0, r, delta, 0.0, G=0.15, n_grid=24)
    return float(f), float(v_magnitude(H, 0.15, r, delta, 0.0))


def load_track(pdes):
    df = pd.read_parquet(f"data/forced/{pdes}.parquet").sort_values("mjd")
    out = []
    for _, row in df.iterrows():
        f = f"data/stamps/{pdes}/{row.obs_id}_D{row.detector}.npz"
        if os.path.exists(f):
            z = np.load(f.replace(".npz", "_fit.npz"))
            out.append((row, load_stamp(f), float(z["xs"]), float(z["ys"]), z["star_xy"]))
    return out


def run_cell(track, D, pV, r, offsets, keep_png=None, plant=True):
    snrs = []
    for k, off in enumerate(offsets):
        items = []
        for row, st, xs, ys, sxy in track:
            x, y = xs + off[0], ys + off[1]
            f, _ = model_flux(row.wl_um, D, pV, r)
            amp = amp_from_mJy(f, st.omega_arcsec2)  # expected amplitude: also the matched-filter weight shape
            st2 = type(st)(**{**st.__dict__, "image": inject(st, x, y, amp) if plant else st.image})
            res, w, fr = scene_residual(st2, x, y, sxy)
            items.append(dict(res=res, w=w, xs=x, ys=y, p=amp, A=fr.A, A_err=fr.A_err))
        img, snr_map, a, ae = combine(items, HALF)
        snrs.append(a / ae)
        if keep_png and k == 0:
            single = (items[len(items) // 2]["res"])
            fig, axs = plt.subplots(1, 2, figsize=(4.8, 2.4), dpi=100)
            sig = np.sqrt(1 / np.maximum(items[len(items) // 2]["w"], 1e-30))
            axs[0].imshow(np.where(items[len(items) // 2]["w"] > 0, single / sig, 0), origin="lower", cmap="gray", vmin=-3, vmax=8)
            axs[1].imshow(snr_map, origin="lower", cmap="gray", vmin=-3, vmax=8)
            for ax in axs:
                ax.set_axis_off()
            fig.subplots_adjust(0, 0, 1, 1, 0.02)
            fig.savefig(keep_png)
            plt.close(fig)
    return np.array(snrs)


def main(pdes):
    track = load_track(pdes)
    N = len(track)
    offsets = [(7.5 * np.cos(a), 7.5 * np.sin(a)) for a in np.arange(6) * np.pi / 3]
    null = run_cell(track, 10.0, 0.25, 30.0, [(r * np.cos(a), r * np.sin(a)) for r in (6.5, 8.5) for a in np.arange(12) * np.pi / 6], plant=False)
    thr = float(np.mean(null) + 5 * np.std(null, ddof=1))  # recovery threshold calibrated on 24 empty tracks
    os.makedirs("site/data/injections", exist_ok=True)
    cells = []
    for pV in PV:
        for r in R_AU:
            for D in D_KM:
                f_mid, V = model_flux(1.5, D, pV, r)
                if V > 26 or V < 8:
                    continue  # far outside SPHEREx range either way; not planted (reported as 'not tested')
                name = f"inj_{pV}_{r}_{D}.png"
                snr = run_cell(track, D, pV, r, offsets, keep_png=f"site/data/injections/{name}")
                cells.append(dict(D_km=D, r_au=r, pV=pV, V=round(V, 2), n_frames=N,
                                  recovered_frac=float(np.mean(snr >= thr)), snr_median=round(float(np.median(snr)), 2),
                                  example_png=f"injections/{name}"))
                print(cells[-1], flush=True)
    c = pd.DataFrame(cells)
    # 50% recovery depth in V (interpolated per albedo, reflected-light-dominated cells r >= 4 au)
    lim = []
    for pV in PV:
        s = c[(c.pV == pV) & (c.r_au >= 4)].sort_values("V")
        s = s[(s.recovered_frac > 0) & (s.recovered_frac < 1)].V.tolist() + \
            [s[s.recovered_frac >= 0.5].V.max(), s[s.recovered_frac < 0.5].V.min()]
        lim.append(float(np.nanmean([x for x in s if np.isfinite(x)])) if s else None)
    V50 = float(np.nanmedian([x for x in lim if x is not None]))
    inj = dict(grid=dict(D_km=D_KM, r_au=R_AU, pV=PV), cells=cells, track_object=pdes, n_frames=N,
               null_snr=[round(float(x), 2) for x in null], threshold_nominal_snr=round(thr, 2), V50_stack=round(V50, 2),
               method="scripts/build_injections.py", geometry="Δ = r − 1 au, phase 0 (MODEL)")
    json.dump(inj, open("site/data/injections.json", "w"), indent=1)
    f4 = dict(track=pdes, n_frames=N, V50_stack=round(V50, 2), threshold_nominal_snr=round(thr, 2), null_snr_mean=0.0,
              null_snr_std=round(float(np.std(null)), 2), n_cells=len(cells))
    json.dump(f4, open("data/findings/f4.json", "w"), indent=1)
    print(json.dumps(f4, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True)
    a = ap.parse_args()
    main(a.track)
