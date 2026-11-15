"""Truth data for validation: JPL SBDB physical parameters and Gaia DR3 reflectance spectra.

Results are cached as JSON under data/cache/truth/ (delete a file to force a re-query).

Sources
-------
* JPL Small-Body Database API, https://ssd-api.jpl.nasa.gov/sbdb.api (``phys-par=1``),
  documentation https://ssd-api.jpl.nasa.gov/doc/sbdb.html
* Gaia DR3 ``gaiadr3.sso_reflectance_spectrum`` (Gaia Collaboration, Galluccio et al. 2023,
  A&A 674, A35): mean reflectance spectra in 16 bands, 374-1034 nm, normalised to 1 at 550 nm.
  Queried at the ESA Gaia archive TAP; if that is unreachable, at the AIP Gaia mirror
  (https://gaia.aip.de/tap), which serves the same DR3 table.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyvo
import requests

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache" / "truth"
SBDB_URL = "https://ssd-api.jpl.nasa.gov/sbdb.api"
GAIA_TAP_URLS = ("https://gea.esac.esa.int/tap-server/tap", "https://gaia.aip.de/tap")
TIMEOUT_S = (10, 60)  # requests (connect, read) timeout in s


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _float(x):
    return None if x is None else float(x)


def sbdb(designation):
    """Physical parameters from JPL SBDB.

    Returns dict with H, G, diameter [km], diameter_sigma [km], albedo, albedo_sigma (floats or
    None when SBDB has no value), diameter_ref, diameter_notes, albedo_ref, albedo_notes, H_ref,
    G_ref, orbit_class (code, e.g. 'MBA'), orbit_class_name, fullname, spkid, and query metadata.
    """
    designation = str(designation)
    path = CACHE_DIR / f"sbdb_{designation.replace(' ', '_').replace('/', '_')}.json"
    if path.exists():
        raw = json.loads(path.read_text())
    else:
        resp = requests.get(SBDB_URL, params={"sstr": designation, "phys-par": 1}, timeout=TIMEOUT_S)
        resp.raise_for_status()
        raw = resp.json()
        if "object" not in raw:  # e.g. multiple matches (code 300) or not found
            raise LookupError(f"SBDB lookup failed for {designation!r}: {raw}")
        raw["_retrieved_utc"] = _now()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(raw, indent=1))

    pars = {p["name"]: p for p in raw.get("phys_par", [])}

    def get(name, field="value"):
        return pars.get(name, {}).get(field)

    obj = raw["object"]
    return {
        "designation": designation,
        "fullname": obj["fullname"],
        "spkid": obj["spkid"],
        "H": _float(get("H")),
        "H_ref": get("H", "ref"),
        "G": _float(get("G")),  # None if SBDB lists no G (callers usually assume 0.15)
        "G_ref": get("G", "ref"),
        "diameter": _float(get("diameter")),
        "diameter_sigma": _float(get("diameter", "sigma")),
        "diameter_ref": get("diameter", "ref"),
        "diameter_notes": get("diameter", "notes"),
        "albedo": _float(get("albedo")),
        "albedo_sigma": _float(get("albedo", "sigma")),
        "albedo_ref": get("albedo", "ref"),
        "albedo_notes": get("albedo", "notes"),
        "orbit_class": obj["orbit_class"]["code"],
        "orbit_class_name": obj["orbit_class"]["name"],
        "source": SBDB_URL,
        "retrieved_utc": raw.get("_retrieved_utc"),
    }


class _TimeoutSession(requests.Session):
    def request(self, *args, **kwargs):
        kwargs.setdefault("timeout", TIMEOUT_S)
        return super().request(*args, **kwargs)


def gaia_reflectance(number):
    """Gaia DR3 mean reflectance spectrum for numbered minor planet ``number``.

    Returns a DataFrame with columns wavelength_um, reflectance, err, flag (sorted by wavelength;
    empty if Gaia DR3 has no spectrum). ``df.attrs`` holds the TAP service used and retrieval time.
    flag is ``reflectance_spectrum_flag`` as published: 0 = no issue; for non-zero values see the
    Gaia DR3 datamodel (gaiadr3.sso_reflectance_spectrum).
    """
    number = int(number)
    path = CACHE_DIR / f"gaia_reflectance_{number}.json"
    if path.exists():
        cached = json.loads(path.read_text())
    else:
        query = (
            "SELECT wavelength, reflectance_spectrum, reflectance_spectrum_err, "
            "reflectance_spectrum_flag FROM gaiadr3.sso_reflectance_spectrum "
            f"WHERE number_mp = {number} ORDER BY wavelength"
        )
        errors = []
        for url in GAIA_TAP_URLS:
            try:
                tab = pyvo.dal.TAPService(url, session=_TimeoutSession()).run_sync(query).to_table()
                break
            except Exception as exc:  # unreachable service: try the next mirror
                errors.append(f"{url}: {exc}")
        else:
            raise ConnectionError("all Gaia TAP services failed:\n" + "\n".join(errors))
        assert str(tab["wavelength"].unit) == "nm", tab["wavelength"].unit
        cached = {
            "number": number,
            "service": url,
            "retrieved_utc": _now(),
            "wavelength_um": [float(w) / 1e3 for w in tab["wavelength"]],
            "reflectance": [float(v) for v in np.ma.filled(tab["reflectance_spectrum"], np.nan)],
            "err": [float(v) for v in np.ma.filled(tab["reflectance_spectrum_err"], np.nan)],
            "flag": [int(v) for v in tab["reflectance_spectrum_flag"]],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cached, indent=1))

    df = pd.DataFrame({k: cached[k] for k in ("wavelength_um", "reflectance", "err", "flag")})
    df.attrs = {"service": cached["service"], "retrieved_utc": cached["retrieved_utc"]}
    return df
