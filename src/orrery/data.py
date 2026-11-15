"""Stamp: the unit of data (docs/INTERFACES.md, "Stamp")."""
from dataclasses import dataclass

import numpy as np


@dataclass
class Stamp:
    obs_id: str
    detector: int
    release: str
    url: str
    mjd: float
    x0: int                 # full-frame pixel of stamp[0, 0]
    y0: int
    image: np.ndarray       # float32 [H, W], MJy/sr
    variance: np.ndarray    # float32 [H, W], (MJy/sr)^2
    zodi: np.ndarray        # float32 [H, W], MJy/sr
    flags: np.ndarray       # int32 [H, W], raw FLAGS bits
    cwave: np.ndarray       # float32 [H, W], um
    cband: np.ndarray       # float32 [H, W], um
    omega_arcsec2: float
    epsf: np.ndarray        # float64 [33, 33], epsf[iy, ix]
    epsf_oversample: int
    wcs_header: str


# --------------------------------------------------------------------------
# Pixel access from the public S3 bucket (HTTPS range reads; DECISIONS #3).
# Conventions: docs/INTERFACES.md; flag/EPSF/WCS-WAVE semantics:
# docs/notes/spherex_docs_digest.md (SPHEREx Explanatory Supplement v2.1).
import io
import warnings
from functools import lru_cache

from astropy.io import fits
from astropy.wcs import WCS, FITSFixedWarning

warnings.simplefilter("ignore", FITSFixedWarning)
BUCKET = "https://nasa-irsa-spherex.s3.amazonaws.com"

# Explanatory Supplement v2.1 §3.3 photometry mask (hard-bad bits):
# SUR_ERROR 2, NONFUNC 6, MISSING_DATA 9, HOT 10, COLD 11, NONLINEAR 15, PERSIST 17,
# BLOOM 26, SNOWBALL 27, HALO 28, SATELLITE_HALO 29. SOURCE/OUTLIER/TRANSIENT are kept.
BAD_BITS = sum(1 << b for b in (2, 6, 9, 10, 11, 15, 17, 26, 27, 28, 29))
FLAG_NAMES = {0: "TRANSIENT", 1: "OVERFLOW", 2: "SUR_ERROR", 6: "NONFUNC", 7: "DICHROIC",
              9: "MISSING_DATA", 10: "HOT", 11: "COLD", 12: "FULLSAMPLE", 14: "PHANMISS",
              15: "NONLINEAR", 17: "PERSIST", 19: "OUTLIER", 20: "CROSSTALK", 21: "SOURCE",
              22: "GHOST", 23: "GHOST_FPA", 24: "GHOST_EXT", 25: "STREAK", 26: "BLOOM",
              27: "SNOWBALL", 28: "HALO", 29: "SATELLITE_HALO"}


def release_of(key):
    return key.split("/")[0]


def _timeout():
    import aiohttp
    return aiohttp.ClientTimeout(total=90, sock_read=45)  # a stalled S3 read must not hang a worker forever


@lru_cache(None)
def _calib(release, det):
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    z = np.load(os.path.join(here, "..", "..", "data", "ref", f"epsf_{release}_D{det}.npz"))
    return {k: z[k] for k in z.files}


class Frame:
    """Lazily opened Level-2 file; headers and small tables are read once."""

    def __init__(self, key):
        self.key = key
        self.url = f"{BUCKET}/{key}"
        self.hdul = fits.open(self.url, use_fsspec=True, lazy_load_hdus=True,
                              fsspec_kwargs={"block_size": 2 ** 17, "cache_type": "bytes",
                                             "client_kwargs": {"timeout": _timeout()}})
        h = self.hdul["IMAGE"].header
        self.header = h
        self.wcs = WCS(h)
        self.mjd = float(h["MJD-AVG"])
        self.obs_id = h["OBSID"]
        self.detector = int(h["DETECTOR"])
        self.omega = float(h["OMEGA_MEDIAN"])
        # EPSF and WCS-WAVE are fixed per detector in DR1 (verified identical across frames,
        # calibration files epsf_<det>_20260507 / spectral_wcs_<det>_v2.2); cached in data/ref/.
        c = _calib(release_of(key), self.detector)
        self.epsf_tab = {"XCENTER": c["xc"], "YCENTER": c["yc"], "EPSF": c["epsf"]}
        self.epsf_ovs = int(c["ovs"])
        self.wx, self.wy, self.wv = c["wx"], c["wy"], c["wv"]

    def world2pix(self, ra, dec):
        x, y = self.wcs.all_world2pix(np.atleast_1d(ra), np.atleast_1d(dec), 0)
        return x, y

    def epsf_at(self, x, y):
        t = self.epsf_tab
        i = np.argmin((t["XCENTER"] - x) ** 2 + (t["YCENTER"] - y) ** 2)  # nearest zone (0-based centres)
        return np.asarray(t["EPSF"][i], float)

    def wave_at(self, x, y):
        """Bilinear interpolation of (cwave, cband) on the WCS-WAVE node grid."""
        from scipy.interpolate import RegularGridInterpolator
        f = RegularGridInterpolator((self.wy, self.wx), self.wv, bounds_error=False, fill_value=None)
        pts = np.stack([np.ravel(y), np.ravel(x)], -1)
        v = f(pts)
        return v[:, 0].reshape(np.shape(x)), v[:, 1].reshape(np.shape(x))

    def stamp(self, xc, yc, half=15, with_zodi=True):
        """Stamp of size (2*half+1)^2 centred on the nearest pixel to (xc, yc); None if off-frame."""
        ix, iy = int(round(float(xc))), int(round(float(yc)))
        x0, y0 = ix - half, iy - half
        x1, y1 = ix + half + 1, iy + half + 1
        if x0 < 0 or y0 < 0 or x1 > 2040 or y1 > 2040:
            return None
        sl = (slice(y0, y1), slice(x0, x1))
        img = np.asarray(self.hdul["IMAGE"].section[sl], np.float32)
        var = np.asarray(self.hdul["VARIANCE"].section[sl], np.float32)
        flg = np.asarray(self.hdul["FLAGS"].section[sl], np.int32)
        zod = np.asarray(self.hdul["ZODI"].section[sl], np.float32) if with_zodi else np.zeros_like(img)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        cw, cb = self.wave_at(xx.astype(float), yy.astype(float))
        return Stamp(obs_id=self.obs_id, detector=self.detector, release=release_of(self.key),
                     url=self.url, mjd=self.mjd, x0=x0, y0=y0, image=img, variance=var, zodi=zod,
                     flags=flg, cwave=cw.astype(np.float32), cband=cb.astype(np.float32),
                     omega_arcsec2=self.omega, epsf=self.epsf_at(ix, iy),
                     epsf_oversample=self.epsf_ovs, wcs_header=self.wcs.to_header(relax=True).tostring())

    def close(self):
        self.hdul.close()


def save_stamp(st, path):
    d = {k: getattr(st, k) for k in Stamp.__dataclass_fields__}
    np.savez_compressed(path, **d)


def load_stamp(path):
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    for k in ("obs_id", "release", "url", "wcs_header"):
        d[k] = str(d[k])
    for k in ("detector", "x0", "y0", "epsf_oversample"):
        d[k] = int(d[k])
    for k in ("mjd", "omega_arcsec2"):
        d[k] = float(d[k])
    return Stamp(**d)
