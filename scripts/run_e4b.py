"""E4b (docs/protocols/E4b.md): WISE 0855-0714 pass significance against 200 empty paths.
Same pictures, estimator (orrery.stack.combine, flat 1 mJy template) and path model as scripts/run_e4.py;
only the number and placement of the empty paths change (E13b geometry: radii 5-9 px x 40 angles).
Usage: OMP_NUM_THREADS=1 .venv/bin/python scripts/run_e4b.py geom|run
"""
import json, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from run_e4 import load_rows, passes_of, load_frames, stat  # noqa: E402

RADII = (5, 6, 7, 8, 9)
OFFS = [(r, a) for j, r in enumerate(RADII) for a in np.arange(40) * 2 * np.pi / 40 + (np.pi / 40 if j % 2 else 0)]
CLEAR_PX = 2.0     # secondary subset: path stays >= 2 px from every scene star in every picture
E4_SIG = {2: 16.473, 3: 10.602}            # public figures (e4.json passes[1].significance, passes[2].neighbour_restored.significance)
E4_NULL = {2: (0.042, 1.069), 3: (0.323, 1.284)}
E4_SNR = {2: 17.646, 3: 13.931}


def pass_frames(k):
    df, _ = load_rows()
    prim = df[(df.wl_um >= 4.4) & (df.wl_um <= 5.0)].sort_values("mjd").reset_index(drop=True)
    prim["pass"] = passes_of(prim.mjd.values) + 1
    return load_frames(prim[prim["pass"] == k], restore=(k == 3))   # pass 3: the quoted (neighbour-restored) variant


def clear_mask(frames):
    ok = []
    for r, a in OFFS:
        dx, dy = r * np.cos(a), r * np.sin(a)
        d = min((np.hypot(np.asarray(f["star_xy"], float).reshape(-1, 2)[:, 0] - f["xs"] - dx,
                          np.asarray(f["star_xy"], float).reshape(-1, 2)[:, 1] - f["ys"] - dy).min()
                 if len(np.asarray(f["star_xy"]).reshape(-1, 2)) else np.inf) for f in frames)
        ok.append(bool(d >= CLEAR_PX))
    return np.array(ok)


def summarize(real, nulls):
    nn = nulls[np.isfinite(nulls)]
    m, sd = float(nn.mean()), float(nn.std(ddof=1))
    n_ge = int((nn >= real).sum())
    return dict(n_nulls=int(len(nn)), null_mean=round(m, 3), null_std=round(sd, 3), null_max=round(float(nn.max()), 3),
                z=round((real - m) / sd, 3), n_nulls_ge_real=n_ge, empirical_p=round((1 + n_ge) / (1 + len(nn)), 5),
                empirical_p_is_upper_bound=n_ge == 0)


def run_pass(k):
    fr = pass_frames(k)
    real = stat(fr, lambda wl: 1.0)["snr"]
    nulls = np.array([stat(fr, lambda wl: 1.0, (r * np.cos(a), r * np.sin(a)))["snr"] for r, a in OFFS])
    clear = clear_mask(fr)
    allp = summarize(real, nulls)
    sig = allp["z"]
    lo, hi = 0.9 * E4_SIG[k], 1.1 * E4_SIG[k]
    if allp["n_nulls_ge_real"] == 0 and lo <= sig <= hi:
        verdict = "stands"
    elif allp["n_nulls_ge_real"] == 0 and sig > hi:
        verdict = "stands (public figure is conservative)"
    else:
        verdict = "does not stand"
    rings = {f"r{r}px": summarize(real, nulls[[i for i, (rr, _) in enumerate(OFFS) if rr == r]]) for r in RADII}
    out = dict(pass_id=k, n_pictures=len(fr), snr_nominal=round(float(real), 3), reproduces_e4_snr=abs(real - E4_SNR[k]) < 0.01,
               e4_public_significance=E4_SIG[k], e4_24path_null_mean=E4_NULL[k][0], e4_24path_null_std=E4_NULL[k][1],
               null200=allp, significance_200=sig, ratio_to_public=round(sig / E4_SIG[k], 3), stands_window=[round(lo, 3), round(hi, 3)],
               verdict=verdict,
               star_clear_subset=dict(rule=f">= {CLEAR_PX} px from every scene (Gaia) star in every picture", **summarize(real, nulls[clear])),
               by_radius=rings, null_snrs=[round(float(x), 3) for x in nulls], null_clear=[bool(c) for c in clear])
    print({x: out[x] for x in ("pass_id", "n_pictures", "snr_nominal", "significance_200", "verdict")}, out["null200"], flush=True)
    return out


if __name__ == "__main__":
    if sys.argv[1] == "geom":   # placement only: no photometry
        for k in (2, 3):
            print(k, int(clear_mask(pass_frames(k)).sum()), "of", len(OFFS), "paths clear of stars")
    else:
        res = [run_pass(2), run_pass(3)]
        json.dump(dict(protocol="docs/protocols/E4b.md", source="data/findings/e4.json", results=res), open("data/findings/e4b.json", "w"), indent=1)
