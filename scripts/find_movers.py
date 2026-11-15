"""Unexplained-mover CANDIDATE search in the Time Machine fields (not discoveries).

Input: data/cache/fields/<field>_S48.npz and site/data/timemachine/<field>.json as written by build_field.py
(run this AFTER build_field.py: it adds tracklet tags to that JSON's "unexplained" labels).
Detections: corrected-map changes (frozen revision-2 model, f from data/findings/f1_calibration.json) that
build_field.py left "unexplained" (not within 2 px of the target, a Gaia high-proper-motion star or a SkyBoT
asteroid V < 21), positive (a mover adds light), and with no flagged pixel (hard-bad or OUTLIER, i.e. masked)
within 1 px of the change. Linking, within one survey pass: pairs of detections from different visits
(gap > 0.5 h) and <= 48 h apart with rate 0.5-60 "/h define a line; detections within 1.5 px of it join.
A tracklet needs >= 3 distinct exposure times, >= 2 visits, fitted displacement >= 2 px over its span, linear-fit
rms <= 1 px and fitted rate 0.5-60 "/h. Greedy, largest first, no shared detections.
Null: frame times shuffled within each pass (detections stay with their frame), same linking, N_NULL trials.
Usage: .venv/bin/python scripts/find_movers.py barnard wise0855 epsindib rwaur rdra ecliptic_control
"""
import json, os, sys

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from orrery.change import SCALE_ARCSEC, change_model, find_changes, THRESH, _EIGHT  # noqa: E402

TOL_PX, RMS_PX, MIN_DISP_PX = 1.5, 1.0, 2.0
RATE = (0.5, 60.0)               # arcsec / hour
VISIT_GAP_H, MAX_SPAN_H, EPOCH_H = 0.5, 48.0, 1 / 60
N_NULL = 50


def passes(mjd, gap=30):
    o = np.argsort(mjd)
    cuts = np.flatnonzero(np.diff(mjd[o]) > gap) + 1
    return np.split(o, cuts)


def detections(d, r, field):
    """Unexplained, positive, unflagged changes: list of dicts with frame index e."""
    flagged = ~np.isfinite(d["v"]) & np.isfinite(d["v_raw"])
    out, near, n_all, n_neg, n_flag = [], [], 0, 0, 0
    for e, fr in enumerate(field["frames"]):
        un = {(l_["x"], l_["y"]) for l_ in fr["labels"] if l_["label"] == "unexplained"}
        if not un:
            continue
        corr = r["corr"][e]
        lab, _ = ndimage.label(np.abs(np.nan_to_num(corr)) > THRESH, structure=_EIGHT)
        for c in find_changes(corr):
            if (c["x"], c["y"]) not in un:
                continue
            n_all += 1
            if c["peak"] <= 0:
                n_neg += 1
                continue
            comp = ndimage.binary_dilation(lab == lab[c["y"], c["x"]], structure=_EIGHT)
            q = dict(e=e, x=float(c["x"]), y=float(c["y"]), peak=c["peak"], npix=c["npix"])
            if (comp & flagged[e]).any():
                n_flag += 1
                near.append(q)
                continue
            out.append(q)
    return out, near, dict(n_unexplained=n_all, n_negative=n_neg, n_near_flagged=n_flag, n_used=len(out))


def link(det, t_h, P):
    """det: detections; t_h: per-frame time (hours). Returns list of tracklets (lists of detection indices)."""
    xy = np.array([(q["x"], q["y"]) for q in det]).reshape(-1, 2)
    fe = np.array([q["e"] for q in det], int)
    t = t_h[fe] if len(det) else np.zeros(0)
    pass_of = np.empty(len(t_h), int)
    for k, p in enumerate(P):
        pass_of[p] = k
    cands = []
    for k in range(len(P)):
        I = np.flatnonzero(pass_of[fe] == k) if len(det) else []
        for a in I:
            dts = t[I] - t[a]
            ok = (dts > VISIT_GAP_H) & (dts <= MAX_SPAN_H)
            with np.errstate(divide="ignore", invalid="ignore"):
                rt = np.hypot(*(xy[I] - xy[a]).T) / dts * SCALE_ARCSEC
            for b in I[ok & (rt >= RATE[0]) & (rt <= RATE[1])]:
                dt = t[b] - t[a]
                v = (xy[b] - xy[a]) / dt
                pred = xy[a] + v * (t[I] - t[a])[:, None]
                m = I[(np.hypot(*(xy[I] - pred).T) <= TOL_PX) & (np.abs(t[I] - t[a] - dt / 2) <= MAX_SPAN_H / 2)]
                tm = t[m]
                if len(np.unique(np.round(tm / EPOCH_H))) < 3:
                    continue
                A = np.stack([np.ones(len(m)), tm - tm.mean()], 1)
                cx, *_ = np.linalg.lstsq(A, xy[m, 0], rcond=None)
                cy, *_ = np.linalg.lstsq(A, xy[m, 1], rcond=None)
                res = np.hypot(xy[m, 0] - A @ cx, xy[m, 1] - A @ cy)
                rate = np.hypot(cx[1], cy[1]) * SCALE_ARCSEC
                disp = np.hypot(cx[1], cy[1]) * (tm.max() - tm.min())
                ts = np.sort(tm)
                n_visits = 1 + int((np.diff(ts) > VISIT_GAP_H).sum())
                if (np.sqrt(np.mean(res ** 2)) <= RMS_PX and RATE[0] <= rate <= RATE[1] and disp >= MIN_DISP_PX
                        and n_visits >= 2):
                    cands.append((len(m), -np.sqrt(np.mean(res ** 2)), tuple(sorted(m.tolist()))))
    used, out = set(), []
    for _, _, m in sorted(set(cands), reverse=True):
        if used.isdisjoint(m):
            out.append(list(m))
            used |= set(m)
    return out


def describe(tr, det, d, mjd, field_name, k):
    q = [det[i] for i in tr]
    e = np.array([x["e"] for x in q])
    th = (mjd[e] - mjd[e].min()) * 24
    xy = np.array([(x["x"], x["y"]) for x in q])
    A = np.stack([np.ones(len(q)), th], 1)
    cx, *_ = np.linalg.lstsq(A, xy[:, 0], rcond=None)
    cy, *_ = np.linalg.lstsq(A, xy[:, 1], rcond=None)
    res = np.hypot(xy[:, 0] - A @ cx, xy[:, 1] - A @ cy)
    ts = np.sort(th)
    # grid: +x = west (east left), +y = south (north up)
    return dict(id=f"{field_name}-T{k}", field=field_name, n_detections=len(q),
                n_epochs=int(len(np.unique(np.round(th / EPOCH_H)))),
                n_visits=1 + int((np.diff(ts) > VISIT_GAP_H).sum()),
                frames=[dict(frame=int(x["e"]), obs_id=str(d["obs_id"][x["e"]]), detector=int(d["detector"][x["e"]]),
                             mjd=round(float(mjd[x["e"]]), 6), x=x["x"], y=x["y"],
                             wl_um=round(float(d["lam"][x["e"]][int(x["y"]), int(x["x"])]), 4),
                             peak_sigma=round(x["peak"], 2), npix=x["npix"]) for x in q],
                peak_significance=round(max(x["peak"] for x in q), 2),
                motion=dict(x0_px=round(float(cx[0]), 2), y0_px=round(float(cy[0]), 2),
                            mjd0=round(float(mjd[e].min()), 6), vx_px_per_h=round(float(cx[1]), 4),
                            vy_px_per_h=round(float(cy[1]), 4),
                            rate_arcsec_per_h=round(float(np.hypot(cx[1], cy[1]) * SCALE_ARCSEC), 3),
                            position_angle_deg=round(float(np.degrees(np.arctan2(-cx[1], -cy[1])) % 360), 1),
                            span_h=round(float(th.max()), 3), rms_px=round(float(np.sqrt(np.mean(res ** 2))), 3)))


def run(name, f, rng):
    d = dict(np.load(f"data/cache/fields/{name}_S48.npz", allow_pickle=True))
    field = json.load(open(f"site/data/timemachine/{name}.json"))
    assert len(field["frames"]) == len(d["mjd"]), "field JSON and cache disagree: rerun build_field.py"
    r = change_model(d["v"], d["lam"], d["detector"], d["mjd"], f=f)
    det, near, stats = detections(d, r, field)
    mjd = d["mjd"].astype(float)
    P = passes(mjd)
    trs = link(det, mjd * 24, P)
    null = []
    for _ in range(N_NULL):
        ts = mjd.copy()
        for p in P:
            ts[p] = mjd[rng.permutation(p)]
        null.append([len(x) for x in link(det, ts * 24, P)])
    # sensitivity only (not labelled): also allow changes next to flagged pixels, since OUTLIER can fire on movers
    allp = det + near
    sens_n = len(link(allp, mjd * 24, P))
    sens_null = []
    for _ in range(N_NULL):
        ts = mjd.copy()
        for p in P:
            ts[p] = mjd[rng.permutation(p)]
        sens_null.append(len(link(allp, ts * 24, P)))
    tracklets = [describe(tr, det, d, mjd, name, k) for k, tr in enumerate(trs)]
    n_null = np.array([len(x) for x in null])
    for T in tracklets:
        # field-level false-alarm probability: fraction of null trials with a tracklet at least this long
        pfa = float(np.mean([any(n >= T["n_detections"] for n in x) for x in null]))
        T["null_fraction_with_tracklet_this_long"] = pfa
        T["confidence"] = round(1 - pfa, 3)
    # tag the field JSON's unexplained labels that belong to a tracklet
    for fr in field["frames"]:
        for l_ in fr["labels"]:
            l_.pop("tracklet", None)
    for T in tracklets:
        for q in T["frames"]:
            for l_ in field["frames"][q["frame"]]["labels"]:
                if l_["label"] == "unexplained" and (l_["x"], l_["y"]) == (q["x"], q["y"]):
                    l_.update(tracklet=T["id"], confidence=T["confidence"],
                              note="candidate mover (linked unexplained changes); not a discovery")
    field["mover_candidates"] = [dict(id=T["id"], n_detections=T["n_detections"], confidence=T["confidence"],
                                      motion=T["motion"]) for T in tracklets]
    json.dump(field, open(f"site/data/timemachine/{name}.json", "w"), separators=(",", ":"))
    return dict(field=name, detections=stats, n_tracklets=len(tracklets),
                null=dict(n_trials=N_NULL, mean=float(n_null.mean()), std=float(n_null.std()),
                          max=int(n_null.max()), frac_trials_ge_observed=float(np.mean(n_null >= len(tracklets)))),
                sensitivity_including_near_flagged=dict(n_tracklets=sens_n, null_mean=float(np.mean(sens_null)),
                                                        null_std=float(np.std(sens_null))),
                tracklets=tracklets)


def main(fields):
    f = json.load(open("data/findings/f1_calibration.json"))["f"]
    rng = np.random.default_rng(0)
    per = [run(n, f, rng) for n in fields]
    out = dict(
        status="CANDIDATES ONLY - linked unexplained changes, not discoveries; none verified",
        method=" ".join(__doc__.split("Usage:")[0].split()),
        params=dict(tol_px=TOL_PX, rms_px=RMS_PX, min_displacement_px=MIN_DISP_PX, rate_arcsec_per_h=RATE,
                    visit_gap_h=VISIT_GAP_H, max_span_h=MAX_SPAN_H, min_epochs=3, min_visits=2, sign="positive",
                    n_null=N_NULL, f=f),
        grid="pixel x = column (east left), y = row (north up), 6.15 arcsec/px, same grid as the field PNGs",
        confidence_rule="1 - fraction of time-shuffled null trials in that field with a tracklet of at least as "
                        "many detections",
        n_tracklets=sum(p["n_tracklets"] for p in per),
        null_mean_total=round(sum(p["null"]["mean"] for p in per), 2),
        per_field=per)
    json.dump(out, open("data/findings/mover_candidates.json", "w"), indent=1)
    for p in per:
        print(p["field"], p["detections"], "tracklets", p["n_tracklets"], "null", p["null"])
    print("total", out["n_tracklets"], "null mean total", out["null_mean_total"])


if __name__ == "__main__":
    main(sys.argv[1:])
