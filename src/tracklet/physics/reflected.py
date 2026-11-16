"""Reflected-light photometry for a distant, cold body.

The observable is sunlight bounced off a body that is hundreds of AU away. Flux
falls as 1/d^2 on the way out and 1/D^2 on the way back, and for the outer
solar system d ~= D, so the signal goes as **1/d^4**. Doubling the distance
costs four magnitudes. This is why the search is hard and why the exclusion is
a steep function of distance.

    F_obj / F_sun(1 AU) = p * Phi(alpha) * R_AU^2 / (d^2 * D^2)
    m = m_sun - 2.5 log10( p * Phi * R_AU^2 / (d^2 D^2) )

The near-infrared caveat matters for a giant planet. A cold methane-rich ice giant is *dark*
in the NIR: Neptune's geometric albedo is 0.442 in the visible (NASA's Neptune
fact sheet) and falls steeply beyond 0.8 um because CH4 absorbs. The NIR
values used below are this project's assumptions, not measurements. SPHEREx covers 0.75-5 um, so
optical V-band magnitudes from the Planet Nine literature overstate what it
would see -- except in the bluest channel, ~0.75-1.1 um, which sits before the
deepest methane bands. That channel is the highest-value place to search for one.

Albedo is therefore treated as an explicit axis of every result, never a
hidden constant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

AU_KM = 1.495978707e8
R_EARTH_KM = 6371.0
AB_ZP_JY = 3631.0

#: Sun's apparent AB magnitude at 1 AU, by band.
#:
#: Willmer (2018, ApJS 236, 47) absolute AB magnitudes -- V 4.81, SDSS z 4.50,
#: 2MASS J 4.54, H 4.66, Ks 5.08 -- shifted by the distance modulus of 1 AU,
#: ``5*log10(AU / 10 pc) = -31.5721``.
#:
#: These used to run the other way, getting steadily *brighter* from z to K,
#: which is the ordering of Vega magnitudes and is impossible in AB. AB is flat
#: in f_nu, and the Sun's f_nu peaks near 0.9 um and falls through the near
#: infrared, so the Sun is brightest in AB at z and faintest at K. J and H were
#: swapped outright. It mattered: ``max_distance`` goes as
#: ``10**(-m_sun/10)``, so the old K entry overstated the reach there by 18
#: percent and the old z entry understated the band this search actually uses
#: by 6.
#:
#: ``test_reflected.py`` pins the ordering as well as the values, because a
#: swapped pair is exactly the kind of error that leaves every number looking
#: plausible.
M_SUN_AB = {"z": -27.07, "V": -26.76, "J": -27.03, "H": -26.91, "K": -26.49}

#: Nominal band centres, microns, only used to state the ordering above as a
#: checkable fact rather than a claim.
BAND_MICRON = {"V": 0.55, "z": 0.90, "J": 1.25, "H": 1.65, "K": 2.19}

#: Assumed geometric albedo for a cold, methane-rich ice giant. ``V`` is
#: Neptune's measured visible value, 0.442 (NASA Neptune fact sheet), rounded.
#: The others are assumptions, not measurements of any such body: CH4 suppresses
#: the NIR, and ``z`` (~0.9 um, SPHEREx channel 1) is taken as less affected.
#: Nothing here is sourced for them, and the conclusions that depend on ``z``
#: move with it: docs/FEASIBILITY.md §5 shows how far.
ALBEDO_ICE_GIANT = {"z": 0.30, "V": 0.45, "J": 0.10, "H": 0.04, "K": 0.02}


def radius_au(radius_earth: float) -> float:
    return radius_earth * R_EARTH_KM / AU_KM


def apparent_mag(
    radius_earth: float,
    d_au: float,
    albedo: float,
    *,
    band: str = "J",
    geo_au: float | None = None,
    phase: float = 1.0,
) -> float:
    """Apparent AB magnitude of a reflecting sphere.

    ``geo_au`` defaults to ``d_au`` (opposition-ish geometry, adequate beyond
    100 AU where the 1 AU Earth offset is a sub-percent correction).
    """
    D = d_au if geo_au is None else geo_au
    r = radius_au(radius_earth)
    ratio = albedo * phase * r * r / (d_au * d_au * D * D)
    return M_SUN_AB[band] - 2.5 * math.log10(ratio)


def max_distance(
    radius_earth: float, albedo: float, m_limit: float, *, band: str = "J"
) -> float:
    """Heliocentric distance at which the body falls to ``m_limit``.

    Inverts the 1/d^4 law, so this is the outer edge of the excluded shell for
    that (radius, albedo) combination.
    """
    r = radius_au(radius_earth)
    d4 = albedo * r * r * 10 ** ((m_limit - M_SUN_AB[band]) / 2.5)
    return d4 ** 0.25


def min_radius(
    d_au: float, albedo: float, m_limit: float, *, band: str = "J"
) -> float:
    """Smallest detectable radius (Earth radii) at a given distance."""
    r_au = math.sqrt(
        d_au ** 4 / (albedo * 10 ** ((m_limit - M_SUN_AB[band]) / 2.5))
    )
    return r_au * AU_KM / R_EARTH_KM


def stacked_depth(single_frame_depth: float, n_frames: int, *, efficiency: float = 1.0) -> float:
    """Ideal sqrt(N) coadd depth, optionally derated.

    ``efficiency`` < 1 absorbs the real-world losses -- imperfect registration,
    residual zodi structure, masked pixels, trajectory mismatch. Calibrate it
    from injection/recovery rather than guessing; 1.0 is the optimistic bound,
    not a prediction.
    """
    if n_frames < 1:
        raise ValueError("n_frames must be >= 1")
    return single_frame_depth + 1.25 * math.log10(n_frames * efficiency)


def flux_to_ab(flux_jy: float) -> float:
    return -2.5 * math.log10(flux_jy / AB_ZP_JY)


def ab_to_flux(mag_ab: float) -> float:
    return AB_ZP_JY * 10 ** (-0.4 * mag_ab)


@dataclass(frozen=True)
class ReachCurve:
    """Excluded distance as a function of radius, at one depth and albedo."""

    band: str
    albedo: float
    depth_ab: float
    radii_earth: tuple[float, ...]
    d_max_au: tuple[float, ...]

    def as_rows(self) -> list[tuple[float, float]]:
        return list(zip(self.radii_earth, self.d_max_au))


def reach_curve(
    depth_ab: float,
    *,
    band: str = "J",
    albedo: float | None = None,
    radii_earth: tuple[float, ...] = (1.0, 2.0, 4.0, 6.0, 10.0),
) -> ReachCurve:
    """Build the exclusion reach curve for one depth."""
    p = ALBEDO_ICE_GIANT[band] if albedo is None else albedo
    return ReachCurve(
        band=band,
        albedo=p,
        depth_ab=depth_ab,
        radii_earth=radii_earth,
        d_max_au=tuple(max_distance(r, p, depth_ab, band=band) for r in radii_earth),
    )
