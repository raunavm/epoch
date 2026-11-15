import numpy as np
import pytest
import requests

from orrery import truth

OFFLINE = (requests.exceptions.ConnectionError, requests.exceptions.Timeout, ConnectionError)


def test_sbdb_themis():
    try:
        d = truth.sbdb("24")
    except OFFLINE as exc:
        pytest.skip(f"SBDB unreachable: {exc}")
    assert d["fullname"].startswith("24 Themis")
    assert d["orbit_class"] == "MBA"
    assert d["diameter"] == pytest.approx(200.0, rel=0.1)
    assert 0.03 < d["albedo"] < 0.15
    assert d["H"] == pytest.approx(7.2, abs=0.3)
    assert d["diameter_ref"] and d["albedo_ref"]


def test_gaia_reflectance_themis():
    try:
        df = truth.gaia_reflectance(24)
    except OFFLINE as exc:
        pytest.skip(f"Gaia TAP unreachable: {exc}")
    assert list(df.columns) == ["wavelength_um", "reflectance", "err", "flag"]
    # Gaia DR3 SSO reflectance spectra: 16 bands, 374-1034 nm (Galluccio et al. 2023)
    assert len(df) == 16
    assert df.wavelength_um.min() >= 0.37 and df.wavelength_um.max() <= 1.04
    assert np.all(np.diff(df.wavelength_um) > 0)
    # normalised to 1 at 550 nm
    assert df.loc[np.argmin(abs(df.wavelength_um - 0.55)), "reflectance"] == pytest.approx(1.0, abs=0.02)
    assert np.all(df.err[df.flag == 0] > 0)
