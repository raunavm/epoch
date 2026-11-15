"""Forced mode: per-frame pixel-level fit of a known moving object (brief §10, 'Forced').

For each frame that contains the object's predicted position (Horizons, observer = SPHEREx):
stamp -> static stars from Gaia DR3 (propagated to the frame epoch) -> joint linear fit of
object + stars + background plane (orrery.fit) -> flux density in mJy with its error.
"""
import os
import time
import numpy as np
import pandas as pd
import pyvo
from astropy.time import Time

from .data import Frame, BAD_BITS, save_stamp
from .fit import fit_stamp, refine_position, flux_mJy

GAIA_TAP = "https://gaia.aip.de/tap"  # AIP mirror of Gaia DR3 (ESA archive unreachable 2026-11-15)
_gaia = None


def gaia_stars(ra, dec, radius_deg, mjd, gmax=19.5, cache_dir="data/cache/gaia"):
    """Gaia DR3 sources within radius, positions propagated from J2016.0 to the frame epoch."""
    global _gaia
    os.makedirs(cache_dir, exist_ok=True)
    path = f"{cache_dir}/{ra:.4f}_{dec:.4f}_{radius_deg:.3f}.parquet"
    if os.path.exists(path):
        df = pd.read_parquet(path)
    else:
        if _gaia is None:
            _gaia = pyvo.dal.TAPService(GAIA_TAP)
        q = ("SELECT ra, dec, pmra, pmdec, phot_g_mean_mag FROM gaiadr3.gaia_source WHERE "
             f"1=CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',{ra},{dec},{radius_deg})) AND phot_g_mean_mag<{gmax}")
        for attempt in range(4):
            try:
                df = _gaia.run_sync(q).to_table().to_pandas()
                break
            except Exception:
                time.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError("Gaia query failed")
        df.to_parquet(path)
    # Linear proper-motion propagation (Gaia DR3 reference epoch J2016.0); pm in mas/yr, pmra includes cos(dec).
    dt = (Time(mjd, format="mjd").jyear - 2016.0)
    pmra = df["pmra"].fillna(0).values
    pmdec = df["pmdec"].fillna(0).values
    d_dec = pmdec * dt / 3.6e6
    d_ra = pmra * dt / 3.6e6 / np.cos(np.radians(df["dec"].values))
    return df["ra"].values + d_ra, df["dec"].values + d_dec, df["phot_g_mean_mag"].values


def candidate_frames(index, eph_daily, max_sep_deg=2.6):
    """Index rows whose centre lies within max_sep of the object's (daily-interpolated) position."""
    m = index["mjd"].values
    ok = (m >= eph_daily["mjd"].min()) & (m <= eph_daily["mjd"].max())
    idx = index[ok]
    ra = np.interp(idx["mjd"].values, eph_daily["mjd"].values, np.unwrap(np.radians(eph_daily["RA"].values)))
    dec = np.radians(np.interp(idx["mjd"].values, eph_daily["mjd"].values, eph_daily["DEC"].values))
    fra, fdec = np.radians(idx["ra"].values), np.radians(idx["dec"].values)
    cosd = np.sin(dec) * np.sin(fdec) + np.cos(dec) * np.cos(fdec) * np.cos(ra - fra)
    sep = np.degrees(np.arccos(np.clip(cosd, -1, 1)))
    return idx[sep < max_sep_deg].copy()


def measure_frame(key, eph_row, half=15, edge=18, refine=True, stamp_dir=None, gmax=19.5):
    """Fit one frame. eph_row has RA, DEC (deg) at the frame mid-time. Returns a dict or None."""
    f = Frame(key)
    try:
        x, y = f.world2pix(eph_row["RA"], eph_row["DEC"])
        x, y = float(x[0]), float(y[0])
        if not (edge <= x <= 2039 - edge and edge <= y <= 2039 - edge):
            return None
        st = f.stamp(x, y, half=half)
        if st is None:
            return None
        xs, ys = x - st.x0, y - st.y0
        sra, sdec, sg = gaia_stars(float(eph_row["RA"]), float(eph_row["DEC"]), (half + 4) * 6.2 / 3600 * 1.42, f.mjd, gmax)
        star_xy = []
        if len(sra):
            gx, gy = f.world2pix(sra, sdec)
            gx, gy = gx - st.x0, gy - st.y0
            H, W = st.image.shape
            for a, b in zip(gx, gy):
                # keep every Gaia star whose ePSF box touches the stamp. Moving targets (asteroids, brown dwarfs not in
                # Gaia) are never Gaia sources, so a star near the target is a real neighbour and must be modelled;
                # an exactly coincident star shows up as a large, honest flux error (degenerate columns), not as leaked light.
                if -3 <= a <= W + 2 and -3 <= b <= H + 2:
                    star_xy.append((float(a), float(b)))
        star_xy = np.array(star_xy).reshape(-1, 2)
        r0 = fit_stamp(st.image, st.variance, st.flags, st.epsf, xs, ys, star_xy, BAD_BITS, st.epsf_oversample)
        out = dict(key=key, obs_id=st.obs_id, detector=st.detector, mjd=st.mjd, x=x, y=y,
                   wl_um=float(st.cwave[half, half]), bw_um=float(st.cband[half, half]),
                   flux_mJy=flux_mJy(r0.A, st.omega_arcsec2), flux_err_mJy=flux_mJy(r0.A_err, st.omega_arcsec2),
                   snr=r0.A / r0.A_err if r0.A_err > 0 else np.nan,
                   chi2=r0.chi2, dof=r0.dof, chi2_local=r0.chi2_local, dof_local=r0.dof_local,
                   n_stars=len(star_xy), flags_src=int(np.bitwise_or.reduce(st.flags[half - 1:half + 2, half - 1:half + 2].ravel())),
                   n_bad=int((~r0.mask).sum()), zodi=float(np.median(st.zodi)), omega=st.omega_arcsec2)
        if refine and out["snr"] > 8:
            xb, yb, surf, offs = refine_position(st.image, st.variance, st.flags, st.epsf, xs, ys, star_xy, BAD_BITS, st.epsf_oversample)
            out["dx_px"], out["dy_px"] = xb - xs, yb - ys
        if stamp_dir:
            os.makedirs(stamp_dir, exist_ok=True)
            save_stamp(st, f"{stamp_dir}/{st.obs_id}_D{st.detector}.npz")
            np.savez_compressed(f"{stamp_dir}/{st.obs_id}_D{st.detector}_fit.npz", model=r0.model, resid=r0.resid,
                                mask=r0.mask, xs=xs, ys=ys, star_xy=star_xy)
        return out
    finally:
        f.close()
