"""E21 step 4: findings JSON and figures from the outputs of scripts/e21_search.mjs.

    node scripts/e21_search.mjs summary [--field sep] [--grid supp]
    .venv/bin/python scripts/run_e21_report.py [nep|sep]
"""
import glob
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent

FIELD = sys.argv[1] if len(sys.argv) > 1 else "nep"
E = ROOT / "data" / "phase2" / "e21"
R = E if FIELD == "nep" else E / f"{FIELD}_res"
FIG = ROOT / "site" / "data" / "figs"
SUF = "" if FIELD == "nep" else f"_{FIELD}"
LABEL = {"nep": "NEP", "sep": "SEP"}[FIELD]
load = lambda p: json.loads(pathlib.Path(p).read_text())  # noqa: E731


def upper95(k, n):
    """One-sided 95% Clopper-Pearson upper limit (as scramble_nulls.mjs)."""
    if k >= n:
        return 1.0
    from scipy.stats import beta
    return float(beta.ppf(0.95, k + 1, n - k))


man = load(E / f"{FIELD}_dr1.json")
real = load(R / "real.json")
chk = load(R / "check500.json")
summ = load(R / "summary_inj.json")
cells = [load(p) for p in sorted(glob.glob(str(R / "inj_cell*.json")))]
supp = load(R / "summary_supp.json") if (R / "summary_supp.json").exists() else None
null_rows, levels = [], None
for p in sorted(glob.glob(str(R / "nulls_*.json"))):
    d = load(p)
    levels = d["levels"]
    null_rows += d["rows"]
null_rows = list({r["k"]: r for r in null_rows}.values())
n_null = len(null_rows)
null_by_level = {}
for j, lv in enumerate(levels or []):
    c = [r["counts"][j] for r in null_rows]
    k = sum(1 for v in c if v > 0)
    null_by_level[str(lv)] = {"runs_with_false": k, "false_candidates_total": int(sum(c)),
                              "mean_false_per_search": float(np.mean(c)) if c else None,
                              "rate_upper95": upper95(k, n_null) if n_null else None}
shipped = null_by_level.get(str(real["required"]), {})

# distance recovery over all recovered injected bodies
fits = [(b["trueAu"], b["fitAu"], b["fitErrAu"]) for c in cells for b in c["bodies"]
        if b["recovered"] and b["fitAu"]]
by_au = {}
for t, f, e in fits:
    by_au.setdefault(int(t), []).append((f, e))
dist_rec = {str(a): {"n": len(v), "median_fit_over_true": float(np.median([f / a for f, _ in v])),
                     "rms_pull": float(np.sqrt(np.mean([((f - a) / e) ** 2 for f, e in v if e])))}
            for a, v in sorted(by_au.items())}

n_inj = sum(c["n"] for c in cells)
n_rec = sum(c["k"] for c in cells)
ex = summ["exclusion"]
best = summ["best_cell"]
best_cells = [(c["snr"], c["au"], c["k"], c["n"]) for c in cells if best and c["k"] == best["k"] and c["n"] == best["n"]]
pass500 = bool(chk["recovered"] and chk["fitAu"] and abs(chk["fitAu"] - 500) <= 3 * chk["fitErrAu"])

TRACKLET = {
    "nep": {"n_visits": 30, "baseline_days": 493, "n_detections": 8747, "n_static": 7974, "n_moving": 773,
            "n_candidates": 0, "check500_fit_au": 501, "check500_err_au": 0.9, "null_false": "0/400",
            "best_cell": "46/48", "exclusion_95": False},
    "sep": {"n_visits": 25, "baseline_days": 93, "n_detections": 12105, "n_moving": 716, "n_candidates": 0,
            "null_false": "0/400", "best_cell": "21/48 (17 sigma, 1000 AU)", "exclusion_95": False},
}
NOTES = {
    "nep": ("Tracklet's frozen search re-run on SPHEREx DR1 Level-2, detector 1, North Ecliptic Pole. "
        "No candidate (also none when 14 or 13 of 30 visits are required). Time-permutation nulls give the "
        "false-alarm rate per search. The frozen 95% rule is formally met in one cell only (17 sigma per "
        "visit at 1,200 AU, 47/48 recovered, recovery chance 0.96; one fewer would fail it); in reflected "
        "light at albedo 0.30 that cell corresponds to bodies of about 90 Earth radii or more, far larger "
        "than any planet, so no planet-sized body (Earth to Jupiter) is excluded at 90% or 95% anywhere in "
        "200-1,200 AU. Half of planted bodies are recovered from about 14 Earth radii at 500 AU. Below "
        "about 400 AU the parallax loop (radius 206265/d arcsec, 168 px at 200 AU) is larger than the "
        "193-px stamp, so bodies leave the stamp in most visits and completeness is near zero: a stamp-size "
        "limit, not a sensitivity limit. The field is 0.044 square degrees. Candidates, if any, would be "
        "reported only as moving with parallax in SPHEREx data, not identified with known objects, not "
        "confirmed."),
    "sep": "",
}

out = {
    "experiment": "E21",
    "protocol": "docs/protocols/E21.md",
    "field": f"{man['title']}, RA {man['ra']} Dec {man['dec']:+}, 193 px stamps",
    "n_candidate_frames": man["selection"]["n_candidate_frames"],
    "n_visits": real["n_epochs"],
    "baseline_days": round(real["baseline_days"], 1),
    "required_visits": real["required"],
    "n_detections": real["n_detections"],
    "n_static": real["n_static"],
    "n_moving": real["n_moving"],
    "n_linked": real["n_linked"],
    "n_candidates": len(real["candidates"]),
    "n_candidates_required_minus_1": len(real["looser"][0]["candidates"]),
    "n_candidates_required_minus_2": len(real["looser"][1]["candidates"]),
    "n_null_runs": n_null,
    "n_null_false": shipped.get("runs_with_false"),
    "null_false_rate_upper95": shipped.get("rate_upper95"),
    "expected_false_count": shipped.get("mean_false_per_search"),
    "nulls_by_visit_requirement": null_by_level,
    "n_injected": n_inj,
    "n_recovered": n_rec,
    "injection_grid_snr": summ["snr"],
    "injection_grid_au": summ["au"],
    "injection_per_cell": cells[0]["n"] if cells else None,
    "n_cells_measured": len(cells),
    "completeness_k": summ["k"],
    "completeness_n": summ["n"],
    "planted_ab_mag_by_snr": [round(a, 2) for a in summ["ab"]],
    "check500_snr": chk["snr"],
    "check500_recovered": chk["recovered"],
    "check500_visits_matched": chk["matched"],
    "check500_fit_au": chk["fitAu"],
    "check500_fit_err_au": chk["fitErrAu"],
    "check500_chi2_nu": chk["chi2"],
    "check500_pass": pass500,
    "distance_recovery_by_au": dist_rec,
    "best_cell_k": best["k"] if best else None,
    "best_cell_n": best["n"] if best else None,
    "best_cell_recovery_chance": summ["best_chance"],
    "best_cells_snr_au": best_cells,
    "needed_of_48_for_90": summ["needed90"],
    "needed_of_48_for_95": summ["needed95"],
    "exclusion_reached_90": bool(ex["0.9"]["reached"]) and not real["candidates"],
    "exclusion_reached_95": bool(ex["0.95"]["reached"]) and not real["candidates"],
    "exclusion_90_bodies": ex["0.9"]["bodies"],
    "exclusion_95_bodies": ex["0.95"]["bodies"],
    "smallest_radius_half_recovered_rearth": ex["0.5"]["smallest"],
    "area_deg2": summ["area_deg2"],
    "tracklet_qr_reference": TRACKLET[FIELD],
    "smallest_radius_90_rearth": ex["0.9"]["smallest"],
    "smallest_radius_95_rearth": ex["0.95"]["smallest"],
    "notes": NOTES[FIELD],
}
if supp:
    ks, ns = np.array(supp["k"]), np.array(supp["n"])
    pooled = {}
    for si, sn in enumerate(summ["snr"]):
        for di, au in enumerate(summ["au"]):
            a, b = supp["snr"].index(sn), supp["au"].index(au)
            pooled[f"{sn}sigma_{au}au"] = [int(summ["k"][si][di] + ks[a, b]), int(summ["n"][si][di] + ns[a, b])]
    sx = supp["exclusion"]
    out.update({
        "supp_grid_snr": supp["snr"], "supp_grid_au": supp["au"], "supp_seed": 8,
        "supp_completeness_k": supp["k"], "supp_completeness_n": supp["n"],
        "supp_n_injected": int(ns.sum()), "supp_n_recovered": int(ks.sum()),
        "supp_best_cell_k": supp["best_cell"]["k"] if supp["best_cell"] else None,
        "supp_best_cell_n": supp["best_cell"]["n"] if supp["best_cell"] else None,
        "supp_reached_90": bool(sx["0.9"]["reached"]), "supp_reached_95": bool(sx["0.95"]["reached"]),
        "supp_smallest_radius_50_rearth": sx["0.5"]["smallest"],
        "supp_smallest_radius_90_rearth": sx["0.9"]["smallest"],
        "supp_smallest_radius_95_rearth": sx["0.95"]["smallest"],
        "pooled_frozen_cells_k_n": pooled,
    })
json.dump(out, open(ROOT / "data" / "findings" / f"e21{SUF}.json", "w"), indent=1)

# ---- figure 1: parallax path and the recovered 500 AU body ------------------------------------
stamps = [np.fromfile(E / FIELD / e["stamp"]["file"], "<f4").reshape(193, 193) for e in man["epochs"]]
med = np.nanmedian(np.stack(stamps), 0)
lo, hi = np.nanpercentile(med, [5, 99.5])
fig, ax = plt.subplots(figsize=(6.4, 6.0), dpi=150)
ax.imshow(med, origin="lower", cmap="gray_r", vmin=lo, vmax=hi, interpolation="nearest")
mj = np.array([e["mjd"] for e in man["epochs"]])
pl = [p for p in chk["planted"] if p["xy"]]
px = np.array([p["xy"][0] for p in pl]); py = np.array([p["xy"][1] for p in pl])
# a body at rest at 500 AU through one year, through the stamp's own TAN grid
cand = None
if chk["candidates"]:
    cand = max(chk["candidates"], key=lambda c: c["members"])
tr = np.array(chk["track"])
ax.plot(tr[:, 1], tr[:, 2], color="#3b6fb6", lw=1.2,
        label="true apparent path: parallax loop + orbital drift")
sc = ax.scatter(px, py, c=mj[[p["i"] for p in pl]] - mj.min(), cmap="viridis", s=34, marker="o", zorder=3,
                edgecolors="white", linewidths=0.6, label=f"planted, 12σ/visit, 500 AU ({len(pl)} visits)")
if cand:
    ax.scatter([m[1] for m in cand["members_xy"]], [m[2] for m in cand["members_xy"]], marker="x", s=22, zorder=4,
               color="#c0392b", lw=1.2, label=f"search candidate: {cand['members']} members, {chk['matched']} on the planted body")
ax.set_xlim(0, 192); ax.set_ylim(0, 192)
ax.set_xlabel("stamp x (px, north up, east left; 6.15″/px)"); ax.set_ylabel("stamp y (px)")
fit_txt = (f"fitted d = {chk['fitAu']:.1f} ± {chk['fitErrAu']:.1f} AU (planted 500), χ²/ν = {chk['chi2']:.2f}"
           if chk["fitAu"] else "not recovered")
ax.set_title(f"E21 · {LABEL}, SPHEREx DR1 D1, {real['n_epochs']} visits / {real['baseline_days']:.0f} d\n{fit_txt}",
             fontsize=9)
cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03); cb.set_label("days since first visit")
ax.legend(loc="lower left", fontsize=7, framealpha=0.9)
fig.tight_layout(); fig.savefig(FIG / f"e21{SUF}_parallax.png"); plt.close(fig)

# ---- figure 2: completeness maps ---------------------------------------------------------------
def completeness_fig(sm, path, what):
    k = np.array(sm["k"], float); n = np.array(sm["n"], float)
    frac = np.where(n > 0, k / np.maximum(n, 1), np.nan)
    fig, ax = plt.subplots(figsize=(1.6 + 0.85 * frac.shape[1], 1.4 + 0.42 * frac.shape[0]), dpi=150)
    im = ax.imshow(frac, origin="lower", cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for si in range(frac.shape[0]):
        for di in range(frac.shape[1]):
            txt = f"{int(k[si, di])}/{int(n[si, di])}" if n[si, di] > 0 else "—"
            ax.text(di, si, txt, ha="center", va="center", fontsize=7,
                    color="white" if (n[si, di] > 0 and frac[si, di] > 0.6) else "#222")
    ax.set_xticks(range(len(sm["au"]))); ax.set_xticklabels(sm["au"])
    ax.set_yticks(range(len(sm["snr"])))
    ax.set_yticklabels([f"{s_}σ  (AB {a:.1f})" for s_, a in zip(sm["snr"], sm["ab"])])
    ax.set_xlabel("distance (AU)"); ax.set_ylabel("per-visit S/N in the median visit (planted AB mag)")
    ax.set_title(f"E21 completeness, {LABEL} DR1 D1, {what}: recovered / planted "
                 f"(95% needs {sm['needed95']}/48)", fontsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.set_label("fraction recovered")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


completeness_fig(summ, FIG / f"e21{SUF}_completeness.png", "frozen grid (seed 7)")
if supp:
    completeness_fig(supp, FIG / f"e21{SUF}_completeness_supp.png", "supplementary grid (seed 8)")
print(json.dumps({k_: out[k_] for k_ in ("n_visits", "baseline_days", "n_detections", "n_candidates", "n_null_runs",
      "n_null_false", "n_injected", "n_recovered", "check500_fit_au", "check500_fit_err_au", "best_cell_k",
      "best_cell_n", "exclusion_reached_90", "exclusion_reached_95")}, indent=0))
