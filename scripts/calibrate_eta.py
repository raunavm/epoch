"""Empirical η calibration for 3–5 µm NEATM fits, with a declared train/test split.

Why: with the literature main-belt prior (η = 0.95 ± 0.20, Masiero et al. 2021, calibrated on NEOWISE
W3/W4 at 12–22 µm), our SPHEREx 3–5 µm diameters came out ~15% larger than NEOWISE for the first objects,
with data/model rising with wavelength — the Wien side of the thermal spectrum at phase 20–30° is where NEATM
is least accurate. NEOWISE faced the same issue for W1/W2-only fits and used an empirically calibrated η.
Protocol (declared before looking at the test set):
  calibration set = objects with an independent diameter and an EVEN number; test set = ODD numbers.
  For each calibration object, fit η with D fixed to the independent value (pV from H) -> median and
  robust spread of η -> frozen prior written to data/findings/eta_calibration.json (separately for
  near-Earth objects if ≥ 3 are available, otherwise one prior for all).
Then scripts/build_objects.py uses that prior, and data/findings/f3.json is computed on the TEST set only.
Usage: .venv/bin/python scripts/calibrate_eta.py
"""
import glob, json, os, sys, warnings
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
warnings.filterwarnings("ignore")
from orrery.analysis import frame_errors, reflectance, visit_groups
from orrery.physics import neatm_mJy, reflected_mJy
from orrery.truth import sbdb


def eta_at_fixed_D(df, H, G, D, kg_sigma=0.15):
    df = df[(df.snr > 5) & np.isfinite(df.flux_mJy)].copy()
    th = df[df.wl_um >= 3.0]
    rf = df[(df.wl_um >= 2.2) & (df.wl_um < 2.6)]
    if len(th) < 8 or len(rf) < 1:
        return None
    pV = (1329.0 / D) ** 2 * 10 ** (-0.4 * H)
    R25 = np.median(reflectance(rf, H, G)[0])
    frefl = np.array([reflected_mJy(w, H, G, r, d, a) for w, r, d, a in zip(th.wl_um, th.r, th.delta, th.alpha)]) * R25 * (1 + 0.05 * (th.wl_um.values - 2.4))
    y, e = th.flux_mJy.values, frame_errors(th)
    g = visit_groups(th.mjd.values)
    etas = np.linspace(0.5, 3.0, 101)
    chi = []
    for eta in etas:
        fth = np.array([neatm_mJy(w, D, pV, eta, r, d, a, G=G, n_grid=32) for w, r, d, a in zip(th.wl_um, th.r, th.delta, th.alpha)])
        mu = frefl + fth
        c2 = 0
        for gg in np.unique(g):
            s = g == gg
            k = (np.sum(y[s] * mu[s] / e[s] ** 2) + 1 / kg_sigma ** 2) / (np.sum(mu[s] ** 2 / e[s] ** 2) + 1 / kg_sigma ** 2)
            c2 += np.sum((y[s] - k * mu[s]) ** 2 / e[s] ** 2) + (k - 1) ** 2 / kg_sigma ** 2
        chi.append(c2)
    return float(etas[int(np.argmin(chi))])


def main():
    rows = []
    for f in sorted(glob.glob("data/forced/*.parquet")):
        pid = os.path.basename(f)[:-8]
        if not pid.isdigit() or int(pid) % 2:
            continue  # calibration set: even numbers only
        df = pd.read_parquet(f)
        if len(df) == 0 or "snr" not in df:
            continue
        t = sbdb(pid)
        if not t.get("diameter"):
            continue
        eta = eta_at_fixed_D(df, t["H"], t["G"] or 0.15, t["diameter"])
        if eta is not None:
            rows.append(dict(id=pid, cls=t.get("orbit_class"), eta=eta, r=float(df.r.median()), alpha=float(df.alpha.median())))
            print(rows[-1], flush=True)
    c = pd.DataFrame(rows)
    out = {"protocol": "calibration = even-numbered objects with independent diameters; test = odd-numbered (scripts/calibrate_eta.py)",
           "n_cal": int(len(c)), "objects": rows}
    for name, sel in [("all", c), ("neo", c[c.cls.isin(["AMO", "APO", "ATE"])])]:
        if len(sel) >= 3:
            med = float(np.median(sel.eta))
            mad = float(1.4826 * np.median(np.abs(sel.eta - med)))
            out[name] = {"eta_median": round(med, 3), "eta_spread": round(max(mad, 0.1), 3), "n": int(len(sel))}
    os.makedirs("data/findings", exist_ok=True)
    json.dump(out, open("data/findings/eta_calibration.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "objects"}, indent=1))


if __name__ == "__main__":
    main()
