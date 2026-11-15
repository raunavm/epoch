"""Independent checks: SPHEREx reflectance vs Gaia DR3 (overlap 0.74–1.034 µm) and 3 µm band depth.

Gaia DR3 reflectance spectra (Gaia Collaboration, Galluccio et al. 2023) are normalised to 1 at
0.55 µm and cover 0.374–1.034 µm in 16 bands. Our reflectance is tied to V via H, which carries
H-magnitude and rotation systematics (~0.1–0.3 mag), so only the *shape* is compared: one free
scale k is fitted, and we report χ²/dof and the spectral slope in the overlap for both.
"""
import numpy as np


def gaia_compare(wl, R, Re, gaia, groups=None):
    """Shape comparison with one free scale per visit group (rotation between visits is not modelled,
    so brightness differences between visits must not be read as colour). Groups with < 2 points in
    the overlap carry no shape information and are dropped."""
    g = gaia[(gaia["flag"] == 0)]
    if groups is None:
        groups = np.zeros(len(wl), int)
    m = (wl >= g.wavelength_um.min()) & (wl <= g.wavelength_um.max())
    if m.sum() < 3 or len(g) < 4:
        return None
    gi = np.interp(wl, g.wavelength_um.values, g.reflectance.values)
    ge = np.interp(wl, g.wavelength_um.values, g.err.values)
    chi2, dof, n, ks, used = 0.0, 0, 0, [], np.zeros(len(wl), bool)
    resid = []
    for gg in np.unique(groups[m]):
        s = m & (groups == gg)
        if s.sum() < 2:
            continue
        y, e, gv = R[s], Re[s], gi[s]
        k = np.sum(y * gv / e ** 2) / np.sum(gv ** 2 / e ** 2)
        tot = np.sqrt(e ** 2 + (k * ge[s]) ** 2)
        chi2 += float(np.sum((y - k * gv) ** 2 / tot ** 2))
        dof += int(s.sum() - 1)
        n += int(s.sum())
        ks.append(float(k))
        used |= s
        resid += list((y / k - gv))
    if dof < 1:
        return None
    return dict(n=n, dof=dof, chi2=chi2, k=float(np.mean(ks)), k_groups=ks, n_groups=len(ks),
                rms_shape=float(np.sqrt(np.mean(np.square(resid)))),
                wl_overlap=[float(wl[used].min()), float(wl[used].max())],
                note="one free scale per visit group; shape compared within visits only")


def band_depth_3um(wl, R, Re, cont=(2.2, 2.6), band=(2.7, 3.0)):
    """Hydration band depth = 1 − R_band / C(λ_band), with C a weighted linear continuum fitted over `cont`
    and extrapolated to the band's weighted mean wavelength (band window per the 2.7–2.9 µm phyllosilicate
    feature seen by AKARI, Usui et al. 2019). Valid only after thermal emission is removed (caller)."""
    c = (wl >= cont[0]) & (wl < cont[1])
    b = (wl >= band[0]) & (wl < band[1])
    if c.sum() < 3 or b.sum() < 1:
        return None
    W = 1 / Re[c] ** 2
    A = np.vstack([np.ones(c.sum()), wl[c] - 2.4]).T
    cov = np.linalg.inv(A.T @ (A * W[:, None]))
    coef = cov @ (A.T @ (W * R[c]))
    wb = 1 / Re[b] ** 2
    lb = np.sum(wb * wl[b]) / wb.sum()
    Rb, Rbe = np.sum(wb * R[b]) / wb.sum(), 1 / np.sqrt(wb.sum())
    x = np.array([1, lb - 2.4])
    Cb, Cbe = x @ coef, np.sqrt(x @ cov @ x)
    d = 1 - Rb / Cb
    de = (Rb / Cb) * np.hypot(Rbe / Rb, Cbe / Cb)
    # robustness: flat continuum (weighted mean) as the alternative analysis choice; the difference is a systematic
    Cf = np.sum(W * R[c]) / W.sum()
    d_flat = 1 - Rb / Cf
    sys_cont = abs(d - d_flat)
    return dict(depth=float(d), err=float(np.hypot(de, sys_cont)), err_stat=float(de), err_continuum=float(sys_cont),
                depth_flat_continuum=float(d_flat), n_band=int(b.sum()), n_cont=int(c.sum()), lam_band=float(lb),
                cont_slope_per_um=float(coef[1] / coef[0]))
