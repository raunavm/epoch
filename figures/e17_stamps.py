"""Stored-data helpers shared by plot_e17_example.py and plot_e17_candidates.py.

Copied from scripts/run_e17.py (constants, rate_grid, shifts, track_pixels, stamps_along, matches_known). The
per-exposure maps are read from the stored cache data/phase2/e17/<field>/ex.npz; nothing is searched or refitted.
"""
import json
import os

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data/phase2/e17")
SCALE = 6.2
DMAX = 0.15 * 3600 / SCALE
MARGIN = int(np.ceil(DMAX)) + 4
HALF_SPAN_D = 3.0
VMIN, VMAX = 0.05, 1.0
STEP = 0.7 * np.sqrt(2) / DMAX
PATCH = 232
MATCH_PX, MATCH_RATE = 15 / SCALE, 0.05


def field(name):
    return next(f for f in json.load(open(f"{ROOT}/fields.json"))["fields"] if f["field"] == name)


def rate_grid():
    nsp = int(np.ceil(np.log(VMAX / VMIN) / np.log(1 + STEP))) + 1
    sp = VMIN * (1 + STEP) ** np.arange(nsp)
    nan = int(np.ceil(2 * np.pi / STEP))
    th = np.arange(nan) * 2 * np.pi / nan
    v = np.repeat(sp, nan)
    a = np.tile(th, nsp)
    pxd = v * 3600 / SCALE
    return dict(speed=v, angle=a, vx=pxd * np.cos(a), vy=pxd * np.sin(a))


def shifts(G, t, t_ref, grp, sel):
    vx, vy = G["vx"][sel], G["vy"][sel]
    dt = (np.asarray(t) - t_ref)[None]
    sp = np.hypot(vx, vy)[:, None]
    use = (sp * np.abs(dt) <= DMAX) & (np.abs(dt) <= HALF_SPAN_D)
    sx = np.rint(vx[:, None] * dt).astype(np.int16)
    sy = np.rint(vy[:, None] * dt).astype(np.int16)
    gmat = np.where(use, np.asarray(grp)[None], -1)
    ngr = np.array([len(set(r[r >= 0])) for r in gmat])
    return sx, sy, use & (ngr >= 2)[:, None], ngr


def load_ex(name):
    z = np.load(f"{ROOT}/{name}/ex.npz")
    EX = [dict(Fw=z["Fw"][e], w=z["w"][e]) for e in range(len(z["t"]))]
    return EX, np.array(z["t"], float), np.array(z["group"], int)


def track_pixels(EX, t, grp, t_ref, G, c):
    sx, sy, use, _ = shifts(G, t, t_ref, grp, np.array([c["rate"]]))
    return use[0], MARGIN + c["x"] + sx[0], MARGIN + c["y"] + sy[0]


def stamps_along(EX, t, grp, t_ref, c, G, half=10):
    """Per-exposure S/N stamps centred on the track, the shift-and-stack stamp, and the zero-motion stack."""
    use, gx, gy = track_pixels(EX, t, grp, t_ref, G, c)
    singles = []
    for e in np.where(use)[0]:
        sl = (slice(gy[e] - half, gy[e] + half + 1), slice(gx[e] - half, gx[e] + half + 1))
        w = EX[e]["w"][sl]
        singles.append((t[e] - t_ref, np.where(w > 0, EX[e]["Fw"][sl] / np.sqrt(np.where(w > 0, w, 1)), np.nan)))
    num = sum(EX[e]["Fw"][gy[e] - half:gy[e] + half + 1, gx[e] - half:gx[e] + half + 1] for e in np.where(use)[0])
    den = sum(EX[e]["w"][gy[e] - half:gy[e] + half + 1, gx[e] - half:gx[e] + half + 1] for e in np.where(use)[0])
    stack = np.where(den > 0, num / np.sqrt(np.where(den > 0, den, 1)), np.nan)
    y0, x0 = MARGIN + c["y"], MARGIN + c["x"]
    num0 = sum(EX[e]["Fw"][y0 - half:y0 + half + 1, x0 - half:x0 + half + 1] for e in np.where(use)[0])
    den0 = sum(EX[e]["w"][y0 - half:y0 + half + 1, x0 - half:x0 + half + 1] for e in np.where(use)[0])
    static = np.where(den0 > 0, num0 / np.sqrt(np.where(den0 > 0, den0, 1)), np.nan)
    return singles, stack, static, (use, gx, gy)


def matches_known(c, ks):
    for k in ks:
        if not k["in_patch"] and not (-5 <= k["x"] <= PATCH + 5 and -5 <= k["y"] <= PATCH + 5):
            continue
        d = np.hypot(c["x"] - k["x"], c["y"] - k["y"])
        a, b = np.radians(c["angle_deg"]), k["angle"]
        dv = np.hypot(c["speed"] * np.cos(a) - k["speed"] * np.cos(b), c["speed"] * np.sin(a) - k["speed"] * np.sin(b))
        if d <= MATCH_PX and dv <= MATCH_RATE:
            return k["name"]
    return None
