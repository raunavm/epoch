"""Object-level analysis of forced photometry: reflectance spectrum and thermal (NEATM) size.

Model for every frame e of one object (one apparition):
    F_e = k_g(e) * [ F_refl(λ_e; H, G, r_e, Δ_e, α_e) * R(λ_e) + F_th(λ_e; D, pV(D,H), η, r_e, Δ_e, α_e) ]
- F_refl, F_th: orrery.physics (Bowell et al. 1989 H,G; Harris 1998 NEATM).
- pV is tied to D and H by D = 1329 km / sqrt(pV) * 10^(-H/5) (Fowler & Chillemi 1992).
- R(λ): relative reflectance (=1 at 0.55 µm by construction of H). Below 3.0 µm it is measured
  frame by frame (no thermal there for main-belt geometry: thermal < 1% at λ < 3 µm, checked per object);
  in the thermal channels (λ ≥ 3.0 µm) R is the median R(2.2–2.6 µm) times (1 + s (λ − 2.4)), with s marginalised
  over a flat prior [−0.1, 0.2] per µm, following Masiero et al. 2021 (IR/visible albedo ratio uncertainty).
- k_g: rotational/epoch scale per survey visit group g (frames within 0.5 d), prior N(1, 0.15); detectors
  see reflected (D1–3) and thermal (D4–6) light simultaneously in pairs, which constrains k_g.
- η prior N(0.95, 0.20) for main-belt asteroids (Masiero et al. 2021 §2); H uncertainty 0.1 mag MC.
Flux errors include the Explanatory Supplement 3% systematic in quadrature (Eq. 19).
Fit: grid over (D, η) with k_g and the reflectance scale profiled analytically; uncertainties by
Monte Carlo over H, G and s. Returns D, pV, η with 16/50/84 percentiles.
"""
import numpy as np
import pandas as pd

from .physics import reflected_mJy, neatm_mJy

SYS_FRAC = 0.03  # Explanatory Supplement v2.1 Eq. 19


def frame_errors(df):
    return np.sqrt(df["flux_err_mJy"].values ** 2 + (SYS_FRAC * np.abs(df["flux_mJy"].values)) ** 2)


def reflectance(df, H, G):
    """Per-frame relative reflectance R = F_obs / F_refl(R=1). Valid where thermal is negligible."""
    fr = np.array([reflected_mJy(w, H, G, r, d, a) for w, r, d, a in
                   zip(df.wl_um, df.r, df.delta, df.alpha)])
    R = df["flux_mJy"].values / fr
    Re = frame_errors(df) / fr
    return R, Re, fr


def visit_groups(mjd, gap=0.5):
    order = np.argsort(mjd)
    g = np.zeros(len(mjd), int)
    gid = 0
    for i in range(1, len(order)):
        if mjd[order[i]] - mjd[order[i - 1]] > gap:
            gid += 1
        g[order[i]] = gid
    return g


def fit_size(df, H, G=0.15, eta_prior=(0.95, 0.20), H_err=0.1, n_mc=30, seed=0,
             wl_thermal=3.0, kg_sigma=0.15):
    """NEATM diameter from thermal channels + reflected continuum. Returns dict with uncertainties.

    Speed: at fixed (η, pV) NEATM flux ∝ D², so the thermal model is tabulated for D = 1 km on an
    (η, log pV) grid per visit group (median geometry of the group; geometry changes < 1% within 0.5 d),
    and each trial D uses pV(D, H) by interpolation in log pV.
    """
    df = df[(df["snr"] > 3) & np.isfinite(df["flux_mJy"])].copy()
    df["g"] = visit_groups(df["mjd"].values)
    th = df[df.wl_um >= wl_thermal]
    rf = df[(df.wl_um >= 2.2) & (df.wl_um < 2.6)]
    if len(th) < 4 or len(rf) < 1:
        return None
    rng = np.random.default_rng(seed)
    eta_grid = np.linspace(0.5, 2.2, 35)
    lpv_grid = np.linspace(np.log10(0.01), np.log10(1.0), 25)
    D_grid = np.geomspace(0.3, 1200, 500)
    groups = th["g"].values
    ug = np.unique(groups)
    # tabulate unit-diameter NEATM: tab[eta, pv, frame]
    tab = np.zeros((len(eta_grid), len(lpv_grid), len(th)))
    for gg in ug:
        sel = groups == gg
        r, d, a = np.median(th.r[sel]), np.median(th.delta[sel]), np.median(th.alpha[sel])
        wl = th.wl_um.values[sel]
        for i, eta in enumerate(eta_grid):
            for k, lp in enumerate(lpv_grid):
                tab[i, k, sel] = neatm_mJy(wl, 1.0, 10 ** lp, eta, r, d, a, G=G, n_grid=32)
    y = th.flux_mJy.values
    sig = frame_errors(th)
    res = []
    chi0 = None
    for m in range(n_mc):
        Hm = H + (rng.normal(0, H_err) if m else 0.0)
        s = rng.uniform(-0.1, 0.2) if m else 0.05
        R25 = np.median(reflectance(rf, Hm, G)[0])
        frefl = np.array([reflected_mJy(w, Hm, G, r, d, a) for w, r, d, a in zip(th.wl_um, th.r, th.delta, th.alpha)])
        frefl = frefl * R25 * (1 + s * (th.wl_um.values - 2.4))
        lpv = np.log10((1329.0 / D_grid) ** 2 * 10 ** (-0.4 * Hm))
        okD = (lpv >= lpv_grid[0]) & (lpv <= lpv_grid[-1])
        chi = np.full((len(eta_grid), len(D_grid)), np.inf)
        for i in range(len(eta_grid)):
            # interpolate unit-D thermal flux at each trial pV, scale by D^2
            T = np.array([np.interp(lpv[okD], lpv_grid, tab[i, :, f]) for f in range(len(th))])  # [frame, D]
            mod = frefl[:, None] + T * D_grid[okD][None, :] ** 2
            c2 = np.zeros(okD.sum())
            for gg in ug:
                sel = groups == gg
                yy, mu, e = y[sel, None], mod[sel], sig[sel, None]
                kg = (np.sum(yy * mu / e ** 2, 0) + 1 / kg_sigma ** 2) / (np.sum(mu ** 2 / e ** 2, 0) + 1 / kg_sigma ** 2)
                c2 += np.sum((yy - kg * mu) ** 2 / e ** 2, 0) + (kg - 1) ** 2 / kg_sigma ** 2
            c2 += ((eta_grid[i] - eta_prior[0]) / eta_prior[1]) ** 2
            chi[i, okD] = c2
        i, j = np.unravel_index(np.argmin(chi), chi.shape)
        D = D_grid[j]
        res.append((D, (1329.0 / D) ** 2 * 10 ** (-0.4 * Hm), eta_grid[i], chi[i, j]))
        if m == 0:
            chi0 = chi
    o = np.array(res)
    best = o[0]
    prof = chi0.min(axis=0)
    # Δχ² threshold inflated by the reduced χ² when the model under-fits (PDG scale-factor convention)
    scale = max(1.0, best[3] / max(1, len(th) - 2))
    ok = prof <= prof.min() + scale
    D_lo_stat, D_hi_stat = D_grid[ok].min(), D_grid[ok].max()
    # MC spread (H, G, reflectance-slope systematics) as a symmetric log-width about the best fit
    s_mc = np.std(np.log(o[:, 0]))
    lo = best[0] * (1 - np.exp(-np.hypot(np.log(best[0] / D_lo_stat), s_mc)))
    hi = best[0] * (np.exp(np.hypot(np.log(D_hi_stat / best[0]), s_mc)) - 1)
    return dict(D_km=float(best[0]), D_lo=float(best[0] - lo), D_hi=float(best[0] + hi), sigma_lnD=float(0.5 * np.log((best[0] + hi) / (best[0] - lo))),
                pV=float(best[1]), pV_p16=float(np.percentile(o[:, 1], 16)), pV_p84=float(np.percentile(o[:, 1], 84)),
                eta=float(best[2]), eta_p16=float(np.percentile(o[:, 2], 16)), eta_p84=float(np.percentile(o[:, 2], 84)),
                chi2=float(best[3]), n_thermal=int(len(th)), chi2_dof=float(best[3] / max(1, len(th) - 2)),
                n_groups=int(len(ug)), n_mc=n_mc)
