"""Apparent motion of a distant body, and the inverse problem: distance from motion.

This is the module that makes SPHEREx's six-month all-sky cadence into an
instrument rather than a coincidence.

The observable
--------------
A body at heliocentric distance ``d`` seen from Earth sits at

    rho = d * s_hat - r_E,        r_E = Earth's heliocentric position (AU)

so to first order in ``1/d`` its apparent direction is displaced from its
heliocentric direction by ``-r_E_perp / d``. The displacement is at most
``1/d`` radian, i.e.

    parallax amplitude  pi = 206265 / d   arcsec

and it *reverses* every six months, because ``r_E`` does. Meanwhile the body's
own orbital motion accumulates. For a circular orbit,

    orbital rate  mu = 1.296e6 / d^1.5   arcsec/yr

Their ratio over a six-month baseline is ``0.6366 * sqrt(d)``, so beyond
**2.5 AU parallax always wins**, and at 500 AU it wins by a factor of 14. A
SPHEREx pair six months apart is therefore very close to a pure parallax
measurement -- the cadence and the science are matched by construction.

Why the ecliptic latitude of the field is a first-class concern
--------------------------------------------------------------
The apparent path over a year is the projected Earth orbit: an ellipse with
semi-major axis ``pi`` along the ecliptic and semi-minor axis ``pi*sin|beta|``
perpendicular to it. So

* at the **ecliptic poles** (``beta = +/-90``) it is a *circle* of radius ``pi``
  -- two independent components, so parallax is separable from proper motion by
  geometry alone, from as few as three epochs however they are spaced;
* in the **ecliptic plane** (``beta = 0``) it degenerates to a *line* along the
  ecliptic, collinear with the orbital motion of anything else bound to the Sun,
  and that geometric separation is gone.

It is tempting to call the second case degenerate. Measured, it is not: parallax
*oscillates* while proper motion *accumulates*, so adequate sampling in time
separates them even when the sky geometry does not. Fitting seven epochs over
SPHEREx's real 479-day NEP baseline at 0.15 arcsec recovers a 500 AU distance to
0.071 AU at the pole and 0.100 AU in the plane -- a factor of 1.4 in precision,
not a wall. The honest statement is that the ecliptic plane costs you the
geometric handle and makes you pay for it with cadence, and that SPHEREx's
six-month all-sky repeat is exactly the cadence that buys it back.

``docs/FEASIBILITY.md`` flags the deep-field/ecliptic-plane mismatch as a depth
problem. This is the second half of it: the poles are where the signature is
cleanest, the plane is where it depends on sampling. Both belong in the
exclusion curve.

The inverse problem
-------------------
Because ``pi = 1/d`` enters the observation equations *linearly* through
factors that depend only on Earth's known position, recovering distance from a
set of positions is a **linear** least-squares problem -- the same five-parameter
fit Hipparcos and Gaia do for stars, applied to a solar-system body:

    dalpha*(t) = a0 + mu_a (t - t0) + pi * F_a*(t)
    ddelta (t) = d0 + mu_d (t - t0) + pi * F_d (t)

Five unknowns; three epochs give six equations. :func:`solve_astrometry` does
that fit with proper weights, a full covariance matrix, and a chi-square, so
the distance it returns comes with an honest error bar instead of a point value.

Accuracy budget, stated so it can be checked
--------------------------------------------
Earth's position comes from the Meeus (Ch. 25) solar series, good to ~0.01 deg
in longitude and ~1e-5 AU in radius. An error ``delta`` in Earth's position
propagates into the predicted direction of a body at ``d`` as ``delta/d``:

    0.01 deg of Earth longitude  ->  1.7e-4 AU transverse  ->  0.07 arcsec at 500 AU

which is **0.011 SPHEREx pixels**. Light-time (69 h at 500 AU) and the
UTC->TT offset are applied because they are one line each. Stellar aberration
is deliberately *not* applied: a SPHEREx WCS is fitted against catalogue
(barycentric) star positions, so the plate solution has already absorbed the
near-uniform aberration shift, and re-applying it would introduce the 20 arcsec
error it looks like it removes. See :func:`apparent_place`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

DEG = math.pi / 180.0
ARCSEC_PER_RAD = 648000.0 / math.pi  # 206264.80624709636

#: Days of light travel per AU. IAU 2012: c = 299792458 m/s, AU = 1.495978707e11 m.
LIGHT_DAYS_PER_AU = 1.495978707e11 / 299792458.0 / 86400.0  # 0.005775 d
#: Gaussian year: 2*pi/k, the period of a 1 AU circular orbit, in days.
YEAR_DAYS = 365.256363004
#: TT - UTC in seconds. TAI-UTC has been 37 s since 2017-01-01 and no leap
#: second is scheduled; TT = TAI + 32.184 s. Worth 0.006 arcsec at 500 AU, so
#: the constant is applied but its staleness is not a risk.
TT_MINUS_UTC_S = 69.184

#: Mean obliquity of the ecliptic at J2000, IAU 2006: 84381.406 arcsec.
OBLIQUITY_J2000_DEG = 84381.406 / 3600.0
#: General precession in longitude, IAU 2006: 5029.0966 arcsec/century.
PRECESSION_DEG_PER_CENTURY = 5029.0966 / 3600.0

MJD_TO_JD = 2400000.5
J2000_JD = 2451545.0


def mjd_to_jd_tt(mjd_utc: float) -> float:
    """MJD(UTC) -> JD(TT). SPHEREx header times (``MJD-AVG``) are UTC."""
    return mjd_utc + MJD_TO_JD + TT_MINUS_UTC_S / 86400.0


# --------------------------------------------------------------------------- #
# Earth
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EarthState:
    """Earth's heliocentric position, equatorial ICRS, in AU."""

    jd_tt: float
    x: float
    y: float
    z: float
    #: Heliocentric ecliptic longitude of Earth, J2000 frame, degrees.
    ecliptic_lon_deg: float
    #: Sun-Earth distance, AU.
    radius_au: float

    def as_tuple(self) -> tuple[float, float, float]:
        return self.x, self.y, self.z


def earth_state(jd_tt: float) -> EarthState:
    """Earth's heliocentric equatorial position from the Meeus solar series.

    Meeus, *Astronomical Algorithms* 2nd ed., Ch. 25: geometric mean longitude,
    equation of centre, and the radius vector from the true anomaly. The Sun's
    geocentric place is Earth's heliocentric place turned through 180 degrees.

    Longitudes are precessed from the mean equinox of date to J2000 with the
    first-order term in general precession, then rotated into equatorial
    coordinates by the J2000 mean obliquity -- so the result is in the same
    frame as a SPHEREx ``RADESYS = 'ICRS'`` WCS, to within the 23 mas frame
    bias, which is 1/260th of a pixel.
    """
    t = (jd_tt - J2000_JD) / 36525.0

    # Geometric mean longitude and mean anomaly, mean equinox of date (deg).
    l0 = 280.46646 + 36000.76983 * t + 0.0003032 * t * t
    m = 357.52911 + 35999.05029 * t - 0.0001537 * t * t
    e = 0.016708634 - 0.000042037 * t - 0.0000001267 * t * t

    mr = math.radians(m % 360.0)
    c = (
        (1.914602 - 0.004817 * t - 0.000014 * t * t) * math.sin(mr)
        + (0.019993 - 0.000101 * t) * math.sin(2 * mr)
        + 0.000289 * math.sin(3 * mr)
    )
    sun_true_lon = l0 + c  # deg, mean equinox of date
    nu = mr + math.radians(c)  # true anomaly
    radius = 1.000001018 * (1.0 - e * e) / (1.0 + e * math.cos(nu))  # AU

    # Precess longitude to the J2000 mean equinox.
    sun_lon_j2000 = sun_true_lon - PRECESSION_DEG_PER_CENTURY * t

    # Earth is opposite the Sun. The Sun's ecliptic latitude is under 1.2
    # arcsec, worth 0.002 arcsec at 500 AU, so Earth's is taken as zero.
    lon_e = (sun_lon_j2000 + 180.0) % 360.0
    lr = math.radians(lon_e)
    xe, ye = radius * math.cos(lr), radius * math.sin(lr)

    eps = math.radians(OBLIQUITY_J2000_DEG)
    return EarthState(
        jd_tt=jd_tt,
        x=xe,
        y=ye * math.cos(eps),
        z=ye * math.sin(eps),
        ecliptic_lon_deg=lon_e,
        radius_au=radius,
    )


def earth_velocity(jd_tt: float, *, h: float = 0.5) -> tuple[float, float, float]:
    """Earth's heliocentric velocity, AU/day, by central difference.

    Only used for diagnostics (the aberration magnitude the UI quotes); the
    astrometric solution never needs it. A half-day step keeps the truncation
    error near 1e-7 AU/day, which is 0.006 percent of Earth's speed.
    """
    a, b = earth_state(jd_tt - h), earth_state(jd_tt + h)
    return ((b.x - a.x) / (2 * h), (b.y - a.y) / (2 * h), (b.z - a.z) / (2 * h))


# --------------------------------------------------------------------------- #
# Closed-form scales
# --------------------------------------------------------------------------- #


def parallax_arcsec(d_au: float) -> float:
    """Annual parallax ``pi = 206265/d`` arcsec: the semi-major axis of the wobble."""
    if d_au <= 0.0:
        raise ValueError("d_au must be positive")
    return ARCSEC_PER_RAD / d_au


def distance_from_parallax(pi_arcsec: float) -> float:
    """Inverse of :func:`parallax_arcsec`."""
    if pi_arcsec <= 0.0:
        raise ValueError("parallax must be positive")
    return ARCSEC_PER_RAD / pi_arcsec


def orbital_rate_arcsec_per_year(d_au: float) -> float:
    """On-sky rate of a circular heliocentric orbit at ``d``, arcsec/yr.

    ``n = 2*pi/P`` with ``P = d^1.5`` years, converted to arcsec. This is the
    body's motion about the Sun, seen from roughly the Sun -- the piece that
    accumulates while parallax oscillates.
    """
    if d_au <= 0.0:
        raise ValueError("d_au must be positive")
    return 2.0 * math.pi * ARCSEC_PER_RAD / d_au**1.5


def drift_direction(
    ra_deg: float, dec_deg: float, pm_ra_arcsec_yr: float, pm_dec_arcsec_yr: float, years: float
) -> tuple[float, float]:
    """Heliocentric direction after ``years`` of uniform drift from ``(ra, dec)``.

    A straight line in the tangent plane at the starting point, which is a great
    circle on the sky and the motion :func:`solve_astrometry` fits. Adding
    ``pm_ra * t / cos(dec)`` to RA instead walks a small circle of constant
    declination. The two agree to 0.1 arcsec at 500 AU over the North Ecliptic
    Pole's 440 days and part by 109 arcsec at 50 AU, where a body planted one
    way and fitted the other came back at chi2/nu 3,700 against 0.1 arcsec
    errors.
    """
    from ..frames.wcs import tan_deproject

    return tan_deproject(
        pm_ra_arcsec_yr * years / 3600.0, pm_dec_arcsec_yr * years / 3600.0, ra_deg, dec_deg
    )


def parallax_dominance(d_au: float, baseline_days: float = YEAR_DAYS / 2) -> float:
    """Ratio of parallax swing to orbital drift over ``baseline_days``.

    At the default six-month baseline -- SPHEREx's all-sky repeat -- this is
    ``0.6366*sqrt(d)`` and crosses unity at 2.47 AU.
    """
    years = baseline_days / YEAR_DAYS
    swing = 2.0 * parallax_arcsec(d_au) * math.sin(math.pi * years)
    drift = orbital_rate_arcsec_per_year(d_au) * years
    return swing / drift if drift else math.inf


def parallactic_ellipse(d_au: float, ecliptic_lat_deg: float) -> tuple[float, float]:
    """``(semi_major, semi_minor)`` of the annual wobble, arcsec.

    Semi-major lies along the ecliptic and equals ``pi``; semi-minor is
    ``pi*sin|beta|`` and vanishes in the ecliptic plane. The **axis ratio is
    ``sin|beta|``**, independent of distance, which is why the shape of the path
    identifies where you are looking and its size identifies how far away the
    object is. Those two facts are separable, and the UI says so.
    """
    pi = parallax_arcsec(d_au)
    return pi, pi * abs(math.sin(math.radians(ecliptic_lat_deg)))


def ecliptic_latitude_deg(ra_deg: float, dec_deg: float) -> float:
    """Ecliptic latitude of an equatorial direction, J2000."""
    eps = math.radians(OBLIQUITY_J2000_DEG)
    a, d = math.radians(ra_deg), math.radians(dec_deg)
    return math.degrees(
        math.asin(
            max(
                -1.0,
                min(
                    1.0, math.sin(d) * math.cos(eps) - math.cos(d) * math.sin(eps) * math.sin(a)
                ),
            )
        )
    )


def ecliptic_longitude_deg(ra_deg: float, dec_deg: float) -> float:
    """Ecliptic longitude of an equatorial direction, J2000."""
    eps = math.radians(OBLIQUITY_J2000_DEG)
    a, d = math.radians(ra_deg), math.radians(dec_deg)
    y = math.sin(a) * math.cos(eps) + math.tan(d) * math.sin(eps)
    return math.degrees(math.atan2(y, math.cos(a))) % 360.0


def equatorial_from_ecliptic(lon_deg: float, lat_deg: float) -> tuple[float, float]:
    """``(ra_deg, dec_deg)`` for an ecliptic direction, J2000.

    The exact inverse of :func:`ecliptic_latitude_deg` and
    :func:`ecliptic_longitude_deg`, and the direction the *drawing* code needs:
    a globe or a diagram is given a latitude and has to find out where that
    lands on the sky, not the other way round.

    Solved rather than eyeballed, which is also how ``bake_field.py`` picked the
    mid-latitude field: ``sin(dec) = sin(b)cos(eps) + cos(b)sin(eps)sin(lon)``.
    """
    eps = math.radians(OBLIQUITY_J2000_DEG)
    b, lon = math.radians(lat_deg), math.radians(lon_deg)
    sd = math.sin(b) * math.cos(eps) + math.cos(b) * math.sin(eps) * math.sin(lon)
    dec = math.degrees(math.asin(max(-1.0, min(1.0, sd))))
    y = math.cos(b) * math.sin(lon) * math.cos(eps) - math.sin(b) * math.sin(eps)
    x = math.cos(b) * math.cos(lon)
    return math.degrees(math.atan2(y, x)) % 360.0, dec


# --------------------------------------------------------------------------- #
# Parallax factors and forward prediction
# --------------------------------------------------------------------------- #


def parallax_factors(
    ra_deg: float, dec_deg: float, earth_xyz: tuple[float, float, float]
) -> tuple[float, float]:
    """``(F_alpha*, F_delta)``: the parallax displacement per unit ``pi``.

    Derived, not recalled. With ``e_hat`` east and ``n_hat`` north at the
    source, the displacement is ``-r_E_perp/d``, whose components are

        F_a* = X sin(a) - Y cos(a)
        F_d  = X cos(a) sin(d) + Y sin(a) sin(d) - Z cos(d)

    so that ``dalpha*cos(dec) = pi * F_a*`` and ``ddelta = pi * F_d`` with
    ``pi`` in the same angular unit as the outputs. ``F_a*`` is already the
    great-circle component, so no ``1/cos(dec)`` appears here -- putting one in
    is the classic sign-and-scale bug near the pole, and SPHEREx's deepest
    fields are 23 degrees from it.
    """
    a, d = math.radians(ra_deg), math.radians(dec_deg)
    x, y, z = earth_xyz
    f_a = x * math.sin(a) - y * math.cos(a)
    f_d = x * math.cos(a) * math.sin(d) + y * math.sin(a) * math.sin(d) - z * math.cos(d)
    return f_a, f_d


def _unit_from_radec(ra_deg: float, dec_deg: float) -> tuple[float, float, float]:
    a, d = math.radians(ra_deg), math.radians(dec_deg)
    return math.cos(d) * math.cos(a), math.cos(d) * math.sin(a), math.sin(d)


def _radec_from_vec(v: tuple[float, float, float]) -> tuple[float, float]:
    x, y, z = v
    n = math.sqrt(x * x + y * y + z * z)
    return math.degrees(math.atan2(y, x)) % 360.0, math.degrees(
        math.asin(max(-1.0, min(1.0, z / n)))
    )


@dataclass(frozen=True)
class ApparentPlace:
    ra_deg: float
    dec_deg: float
    #: Geocentric distance, AU.
    range_au: float
    #: Light travel time applied, days.
    light_time_days: float
    #: Total offset from the body's heliocentric direction, arcsec.
    parallax_offset_arcsec: float


def apparent_place(
    ra_helio_deg: float,
    dec_helio_deg: float,
    d_au: float,
    mjd_utc: float,
    *,
    light_time: bool = True,
) -> ApparentPlace:
    """Where a body at heliocentric direction ``(ra, dec)`` and distance ``d`` appears.

    Exact in the geometry (no ``1/d`` expansion): the full vector difference is
    formed and renormalised.

    ``light_time`` controls only whether the light travel time is *reported*,
    and that is deliberate rather than a shortcut. The parameterisation here is
    a heliocentric *direction* plus a distance, both constant, so the body is in
    the same place whichever instant you evaluate it at and the retardation has
    no geometric effect to apply -- it is already absorbed into the direction.
    What light-time does bound is how stale the geometry is: 69 hours at 500 AU,
    during which a circular orbit there turns by 0.9 arcsec. That is the honest
    size of the effect the constant-direction model leaves out, it is 0.15
    pixels, and :attr:`ApparentPlace.light_time_days` is what lets a caller
    check it rather than take the claim on trust.

    Stellar aberration is not applied, on purpose. SPHEREx's per-frame WCS is
    solved against catalogue star positions, so the ~20 arcsec near-uniform
    aberration shift is already inside the plate solution; adding it here would
    put a 3-pixel error into every prediction. The differential part across the
    3.5 degree field is ~0.4 mas and is below everything else in the budget.
    """
    jd = mjd_to_jd_tt(mjd_utc)
    s = _unit_from_radec(ra_helio_deg, dec_helio_deg)
    e = earth_state(jd)

    rho = (d_au * s[0] - e.x, d_au * s[1] - e.y, d_au * s[2] - e.z)
    rng = math.sqrt(rho[0] ** 2 + rho[1] ** 2 + rho[2] ** 2)
    ra, dec = _radec_from_vec(rho)
    tau = rng * LIGHT_DAYS_PER_AU if light_time else 0.0

    # Separation from the heliocentric direction, for the readout.
    dot = (rho[0] * s[0] + rho[1] * s[1] + rho[2] * s[2]) / rng
    off = math.degrees(math.acos(max(-1.0, min(1.0, dot)))) * 3600.0
    return ApparentPlace(ra, dec, rng, tau, off)


def helio_direction(
    ra_app_deg: float, dec_app_deg: float, d_au: float, mjd_utc: float
) -> tuple[float, float]:
    """Exact inverse of :func:`apparent_place`: where the body actually points.

    Given where something *appears* and an assumed heliocentric distance, the
    heliocentric direction follows from geometry alone. With ``u`` the apparent
    unit vector, ``E`` Earth's position and ``rho`` the unknown geocentric
    range, the body is at ``E + rho*u`` and that has length ``d``:

        rho^2 + 2 rho (E.u) + |E|^2 - d^2 = 0

    so ``rho = -(E.u) + sqrt((E.u)^2 + d^2 - |E|^2)``, taking the root in front
    of the observer. Solved rather than iterated, and rather than subtracting a
    first-order parallax offset -- at 40 AU that approximation is off by 2.5
    percent, which is the same error that puts a reduced chi-squared of 10^6 on
    an otherwise plausible-looking fit.

    Raises :class:`ValueError` for ``d`` inside Earth's orbit along that line of
    sight, where no such body exists.
    """
    u = _unit_from_radec(ra_app_deg, dec_app_deg)
    e = earth_state(mjd_to_jd_tt(mjd_utc))
    ex, ey, ez = e.x, e.y, e.z
    b = ex * u[0] + ey * u[1] + ez * u[2]
    disc = b * b + d_au * d_au - (ex * ex + ey * ey + ez * ez)
    if disc < 0.0:
        raise ValueError(f"no body at {d_au} AU lies along that line of sight")
    rho = -b + math.sqrt(disc)
    if rho <= 0.0:
        raise ValueError(f"{d_au} AU is behind the observer along that line of sight")
    return _radec_from_vec((ex + rho * u[0], ey + rho * u[1], ez + rho * u[2]))


def apparent_track(
    ra_helio_deg: float,
    dec_helio_deg: float,
    d_au: float,
    mjds: list[float],
) -> list[tuple[float, float]]:
    """Apparent ``(ra, dec)`` at each epoch.

    Not the curve the UI overlays, though this used to say so. Every track the
    web tool draws calls :func:`apparent_place` in its own loop -- six places do
    -- because each wants a different shape out of it: arcsec offsets from the
    field centre, canvas coordinates, or a curve sampled far finer than the
    epoch list. Bare ``(ra, dec)`` pairs are the one thing none of them wanted.
    The JavaScript mirror of this had no callers at all and was removed.
    """
    return [
        (p.ra_deg, p.dec_deg)
        for p in (apparent_place(ra_helio_deg, dec_helio_deg, d_au, m) for m in mjds)
    ]


# --------------------------------------------------------------------------- #
# Linear algebra, just enough
# --------------------------------------------------------------------------- #


def _cholesky_solve_and_invert(
    a: list[list[float]], b: list[float]
) -> tuple[list[float], list[list[float]]]:
    """Solve ``A x = b`` and return ``(x, A^-1)`` for symmetric positive-definite A.

    Cholesky rather than Gaussian elimination because the normal matrix is SPD
    by construction and Cholesky fails loudly (a negative pivot) exactly when
    the design is degenerate -- which is the case worth detecting: two epochs
    cannot separate proper motion from parallax, and the caller must be told so
    rather than handed a plausible-looking number.
    """
    n = len(a)
    lo = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = a[i][j] - math.fsum(lo[i][k] * lo[j][k] for k in range(j))
            if i == j:
                if s <= 0.0:
                    raise ValueError("normal matrix is not positive definite")
                lo[i][j] = math.sqrt(s)
            else:
                lo[i][j] = s / lo[j][j]

    def fwd(rhs: list[float]) -> list[float]:
        y = [0.0] * n
        for i in range(n):
            y[i] = (rhs[i] - math.fsum(lo[i][k] * y[k] for k in range(i))) / lo[i][i]
        return y

    def back(y: list[float]) -> list[float]:
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            x[i] = (y[i] - math.fsum(lo[k][i] * x[k] for k in range(i + 1, n))) / lo[i][i]
        return x

    x = back(fwd(b))
    inv = [[0.0] * n for _ in range(n)]
    for j in range(n):
        e = [1.0 if i == j else 0.0 for i in range(n)]
        col = back(fwd(e))
        for i in range(n):
            inv[i][j] = col[i]
    return x, inv


# --------------------------------------------------------------------------- #
# The inverse problem
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Observation:
    """One measured position of a candidate, with its uncertainty."""

    mjd: float
    ra_deg: float
    dec_deg: float
    #: 1-sigma positional uncertainty, arcsec. A single SPHEREx centroid is
    #: ~0.15 arcsec for a high-SNR source; a marginal one is nearer a pixel.
    sigma_arcsec: float = 1.0


@dataclass(frozen=True)
class AstrometricSolution:
    """Five-parameter fit: reference position, proper motion, and parallax."""

    #: Reference epoch, MJD -- the weighted mean of the input epochs.
    t0_mjd: float
    ra0_deg: float
    dec0_deg: float
    #: Proper motion, arcsec/yr. ``pm_ra`` is the great-circle component.
    pm_ra_arcsec_yr: float
    pm_dec_arcsec_yr: float
    parallax_arcsec: float
    #: 1-sigma formal errors, same units as the parameters.
    err_ra_arcsec: float
    err_dec_arcsec: float
    err_pm_ra_arcsec_yr: float
    err_pm_dec_arcsec_yr: float
    err_parallax_arcsec: float
    chi2: float
    n_obs: int
    dof: int
    #: How much of the parallax could be proper motion instead: the multiple
    #: correlation between parallax and the two proper-motion components, in
    #: [0, 1]. Unsigned on purpose -- it is the largest correlation any linear
    #: combination of the components achieves, so it has a magnitude and no
    #: direction. Near 1 means these visit dates cannot tell the two apart.
    corr_parallax_pm: float
    #: Gauss-Newton iterations against the exact geometry. 1 means the linear
    #: solution was already at the optimum.
    n_iterations: int = 1
    #: The tangent point the fit was made in, and the reference position as an
    #: offset in that plane, arcsec east and north. The fitted motion is a
    #: straight line in this plane, so :meth:`place` needs all four to evaluate
    #: the model that was actually fitted.
    ra_ref_deg: float = math.nan
    dec_ref_deg: float = math.nan
    x0_arcsec: float = 0.0
    y0_arcsec: float = 0.0

    def place(self, mjd: float) -> tuple[float, float, float, float]:
        """Where this fit puts the body at ``mjd``, in degrees.

        Returns ``(ra_helio, dec_helio, ra, dec)``: the direction seen from the
        Sun and the apparent place seen from Earth. The linker's predictions,
        the console's residuals and its fitted track all come from here, so none
        of them can evaluate a model other than the one the fit minimised.
        """
        from ..frames.wcs import tan_deproject

        dt = (mjd - self.t0_mjd) / YEAR_DAYS
        ra_h, dec_h = tan_deproject(
            (self.x0_arcsec + self.pm_ra_arcsec_yr * dt) / 3600.0,
            (self.y0_arcsec + self.pm_dec_arcsec_yr * dt) / 3600.0,
            self.ra_ref_deg,
            self.dec_ref_deg,
        )
        ap = apparent_place(ra_h, dec_h, self.distance_au, mjd)
        return ra_h, dec_h, ap.ra_deg, ap.dec_deg

    @property
    def reduced_chi2(self) -> float:
        return self.chi2 / self.dof if self.dof > 0 else float("nan")

    @property
    def err_scale(self) -> float:
        """``max(1, sqrt(chi2/dof))`` -- the factor the formal errors are short by.

        The ``err_*`` fields come straight out of the covariance matrix and are
        therefore only as truthful as the input ``sigma_arcsec`` values and the
        model. When the fit does not actually describe the data, reduced
        chi-square exceeds one and the formal errors are optimistic by
        ``sqrt(chi2/dof)``. Everything user-facing -- :attr:`distance_err_au`,
        :attr:`distance_interval_au`, :attr:`significance` -- is scaled by this,
        so an under-quoted per-epoch sigma or a model that does not fit widens
        the distance bar instead of silently shrinking it. Never below 1: a fit
        that happens to land better than its noise does not earn tighter bars.
        """
        r = self.reduced_chi2
        return math.sqrt(r) if r == r and r > 1.0 else 1.0

    @property
    def err_parallax_quoted(self) -> float:
        """The parallax error to show a human: formal, scaled by :attr:`err_scale`."""
        return self.err_parallax_arcsec * self.err_scale

    @property
    def distance_au(self) -> float:
        return distance_from_parallax(self.parallax_arcsec)

    @property
    def distance_err_au(self) -> float:
        """Propagated from ``d = 206265/pi`` as ``sigma_d = d * sigma_pi/pi``.

        Symmetric only while the fractional parallax error is small. Above
        about 20 percent the distance distribution is badly skewed and a
        one-sigma bar is misleading; :attr:`distance_interval_au` is the honest
        form and the UI shows that instead.
        """
        p = self.parallax_arcsec
        return ARCSEC_PER_RAD * self.err_parallax_quoted / (p * p)

    @property
    def distance_interval_au(self) -> tuple[float, float]:
        """Distance range from the parallax 1-sigma interval, transformed exactly.

        Inverting the endpoints rather than the centre keeps the asymmetry that
        ``1/pi`` really has. Returns ``(inf, ...)`` upper bound when the lower
        parallax endpoint reaches zero -- an unbounded distance is the correct
        answer to a parallax consistent with zero, not an error.
        """
        e = self.err_parallax_quoted
        lo_pi = self.parallax_arcsec - e
        hi_pi = self.parallax_arcsec + e
        far = math.inf if lo_pi <= 0.0 else ARCSEC_PER_RAD / lo_pi
        near = ARCSEC_PER_RAD / hi_pi if hi_pi > 0.0 else math.inf
        return near, far

    @property
    def significance(self) -> float:
        """Parallax detection significance, ``pi/sigma_pi``."""
        e = self.err_parallax_quoted
        return self.parallax_arcsec / e if e else math.inf

    @property
    def separable(self) -> bool:
        """Whether the epoch sampling actually distinguishes parallax from drift."""
        return self.dof > 0 and abs(self.corr_parallax_pm) < 0.98


def solve_astrometry(obs: list[Observation], *, refine: bool = True) -> AstrometricSolution:
    """Fit position, proper motion, and parallax; five parameters, real error bars.

    Two stages, because one is not enough.

    **Stage 1 -- linear.** The parallax factors depend only on Earth's known
    position, so to first order in ``1/d`` the observation equations

        dalpha*(t) = a0 + mu_a (t - t0) + pi * F_a*(t)
        ddelta (t) = d0 + mu_d (t - t0) + pi * F_d (t)

    are linear in all five parameters. That gives a unique, non-iterative
    solution with a covariance matrix and no starting guess -- the same fit
    Hipparcos and Gaia do for stars.

    **Stage 2 -- exact.** Those equations are only first order. The true
    displacement is ``arctan(|r_E_perp| / (d - r_E_par))``, not ``|r_E_perp|/d``,
    so the linear model carries a fractional error of order ``r_E/d``: 0.2
    percent at 500 AU, but 2.5 percent at 40 AU. Against SPHEREx's 0.15 arcsec
    astrometry a 2.5 percent error on a 6875 arcsec parallax is a 170 arcsec
    residual -- reduced chi-square of order 1e6, and formal error bars that are
    nonsense while the fitted distance still looks plausible. That failure is
    silent, which is the dangerous kind.

    So the linear solution is used as a starting point for Gauss-Newton against
    :func:`apparent_place`, the exact forward model, reusing the linear design
    matrix as the Jacobian (correct to the same ``O(1/d)``, which is all
    Gauss-Newton needs to converge). Two or three iterations bring reduced
    chi-square to order unity at every distance, and the returned covariance
    then means what it says. Pass ``refine=False`` to see the linear-only result
    and the size of the effect.

    Requires at least three epochs; two leave the parallax/proper-motion
    degeneracy unbroken, so rather than return a plausible-looking number the
    function raises and points at :func:`distance_from_two_epochs`.
    """
    if len(obs) < 3:
        raise ValueError(
            f"need >= 3 epochs to separate parallax from proper motion, got {len(obs)}; "
            "use distance_from_two_epochs() for the constrained two-epoch case"
        )
    # Refused by name rather than carried through: a NaN position came back as
    # a solution of NaNs, and a zero uncertainty as a division by zero -- or in
    # the browser, as NaNs with an infinite significance.
    for k, o in enumerate(obs):
        if not (math.isfinite(o.mjd) and math.isfinite(o.ra_deg) and math.isfinite(o.dec_deg)):
            raise ValueError(f"observation {k} has a time or position that is not a number")
        if abs(o.dec_deg) > 90.0:
            raise ValueError(f"observation {k} has declination {o.dec_deg}, beyond a pole")
        if not (math.isfinite(o.sigma_arcsec) and o.sigma_arcsec > 0.0):
            raise ValueError(f"observation {k} has uncertainty {o.sigma_arcsec}; it must be "
                             "a positive number of arcsec")

    from ..frames.wcs import tan_deproject, tan_project

    w = [1.0 / (o.sigma_arcsec**2) for o in obs]
    wsum = math.fsum(w)
    t0 = math.fsum(wi * o.mjd for wi, o in zip(w, obs)) / wsum

    # Tangent point: the weighted mean direction, by vector sum so it stays well
    # behaved at the ecliptic poles -- where SPHEREx's deepest fields are.
    units = [_unit_from_radec(o.ra_deg, o.dec_deg) for o in obs]
    ra_ref, dec_ref = _radec_from_vec(
        tuple(math.fsum(wi * u[k] for wi, u in zip(w, units)) for k in range(3))  # type: ignore[arg-type]
    )

    # Observed offsets from the tangent point, arcsec east/north, and the
    # per-epoch parallax factors. Both are fixed; only the model moves.
    obs_xy: list[tuple[float, float]] = []
    design: list[list[float]] = []
    for o in obs:
        xw, yw = tan_project(o.ra_deg, o.dec_deg, ra_ref, dec_ref)
        obs_xy.append((xw * 3600.0, yw * 3600.0))
        dt = (o.mjd - t0) / YEAR_DAYS
        e = earth_state(mjd_to_jd_tt(o.mjd))
        fa, fd = parallax_factors(ra_ref, dec_ref, e.as_tuple())
        #              a0   d0   mu_a  mu_d  pi
        design.append([1.0, 0.0, dt, 0.0, fa])
        design.append([0.0, 1.0, 0.0, dt, fd])

    n = 5
    ata = [[0.0] * n for _ in range(n)]
    for k, row in enumerate(design):
        iw = 1.0 / (obs[k // 2].sigma_arcsec ** 2)
        for i in range(n):
            for j in range(n):
                ata[i][j] += iw * row[i] * row[j]

    def normal_solve(values: list[float]) -> tuple[list[float], list[list[float]]]:
        atb = [0.0] * n
        for k, row in enumerate(design):
            iw = 1.0 / (obs[k // 2].sigma_arcsec ** 2)
            for i in range(n):
                atb[i] += iw * row[i] * values[k]
        try:
            return _cholesky_solve_and_invert(ata, atb)
        except ValueError as exc:
            raise ValueError(
                "epoch sampling does not constrain all five parameters "
                "(collinear times or no parallax leverage)"
            ) from exc

    flat_obs = [c for xy in obs_xy for c in xy]
    params, cov = normal_solve(flat_obs)

    def model_offsets(p: list[float]) -> list[float]:
        """Exact predicted offsets, arcsec east/north of the tangent point."""
        a0, d0, mu_a, mu_d, pi = p
        if pi <= 0.0:
            raise ValueError("fit drove the parallax non-positive; check the epochs")
        d_au = ARCSEC_PER_RAD / pi
        out: list[float] = []
        for o in obs:
            dt = (o.mjd - t0) / YEAR_DAYS
            # Heliocentric direction at this epoch: reference plus linear drift.
            ra_h, dec_h = tan_deproject(
                (a0 + mu_a * dt) / 3600.0, (d0 + mu_d * dt) / 3600.0, ra_ref, dec_ref
            )
            ap = apparent_place(ra_h, dec_h, d_au, o.mjd)
            xw, yw = tan_project(ap.ra_deg, ap.dec_deg, ra_ref, dec_ref)
            out.append(xw * 3600.0)
            out.append(yw * 3600.0)
        return out

    iterations = 1
    if refine:
        for iterations in range(1, 13):
            pred = model_offsets(params)
            resid = [flat_obs[k] - pred[k] for k in range(len(flat_obs))]
            delta, cov = normal_solve(resid)
            params = [params[i] + delta[i] for i in range(n)]
            # Convergence on the parallax, the parameter everything hangs off:
            # 1e-9 arcsec is 12 orders below any real measurement error.
            if max(abs(delta[4]), abs(delta[2]), abs(delta[3])) < 1e-9:
                break

    a0, d0, mu_a, mu_d, pi = params
    final = (
        model_offsets(params)
        if refine
        else [math.fsum(row[i] * params[i] for i in range(n)) for row in design]
    )
    chi2 = math.fsum(
        ((final[k] - flat_obs[k]) / obs[k // 2].sigma_arcsec) ** 2 for k in range(len(flat_obs))
    )
    dof = len(flat_obs) - n

    # Correlation between parallax and proper motion, as a *multiple*
    # correlation rather than a projection onto the fitted direction.
    #
    # The obvious version -- project the covariance onto mu/|mu| -- is
    # undefined exactly where it matters most. A body with little or no fitted
    # drift has |mu| at the level of the noise, so mu/|mu| is a random
    # direction and the number that comes back is a random number. The
    # forecast helper plants a body with no drift at all, and Python and
    # JavaScript disagreed by 0.07 on it while every other covariance-derived
    # quantity in the same fit matched to the last bit: identical formulas fed
    # an arbitrary unit vector.
    #
    # R^2 = c' S^-1 c / var(pi), with c the parallax/proper-motion covariances
    # and S the 2x2 proper-motion block, is the largest correlation attainable
    # by *any* linear combination of the two components. Well defined however
    # mu came out, conservative by construction, and it is the quantity the
    # question actually asks: how much of what you think is parallax could the
    # proper motion be accounting for.
    a, b, d2 = cov[2][2], cov[2][3], cov[3][3]
    c0, c1 = cov[4][2], cov[4][3]
    det = a * d2 - b * b
    var_pi = cov[4][4]
    if det > 0.0 and var_pi > 0.0:
        quad = (d2 * c0 * c0 - 2.0 * b * c0 * c1 + a * c1 * c1) / det
        corr = math.sqrt(max(0.0, min(1.0, quad / var_pi)))
    else:
        corr = 0.0

    ra_fit, dec_fit = tan_deproject(a0 / 3600.0, d0 / 3600.0, ra_ref, dec_ref)
    return AstrometricSolution(
        t0_mjd=t0,
        ra0_deg=ra_fit,
        dec0_deg=dec_fit,
        pm_ra_arcsec_yr=mu_a,
        pm_dec_arcsec_yr=mu_d,
        parallax_arcsec=pi,
        err_ra_arcsec=math.sqrt(max(cov[0][0], 0.0)),
        err_dec_arcsec=math.sqrt(max(cov[1][1], 0.0)),
        err_pm_ra_arcsec_yr=math.sqrt(max(cov[2][2], 0.0)),
        err_pm_dec_arcsec_yr=math.sqrt(max(cov[3][3], 0.0)),
        err_parallax_arcsec=math.sqrt(max(cov[4][4], 0.0)),
        chi2=chi2,
        n_obs=len(obs),
        dof=dof,
        corr_parallax_pm=max(0.0, min(1.0, corr)),
        n_iterations=iterations,
        ra_ref_deg=ra_ref,
        dec_ref_deg=dec_ref,
        x0_arcsec=a0,
        y0_arcsec=d0,
    )


@dataclass(frozen=True)
class TwoEpochDistance:
    """What two epochs alone can say about distance: an interval, not a number."""

    #: Smallest and largest heliocentric distance consistent with the measured
    #: displacement, given a circular-orbit bound on the drift *magnitude* and
    #: no assumption at all about its direction. This is the honest answer.
    d_min_au: float
    d_max_au: float
    #: Point estimate assuming the parallax and orbital displacements are
    #: collinear and add. The conservative (largest) end of the assumption space.
    d_aligned_au: float
    #: Point estimate ignoring orbital motion entirely: ``pi(d) = displacement``.
    d_parallax_only_au: float
    #: Measured total on-sky displacement between the two epochs, arcsec.
    displacement_arcsec: float
    #: Parallax and orbital contributions evaluated at :attr:`d_aligned_au`.
    parallax_part_arcsec: float
    orbital_part_arcsec: float
    #: Change in parallax factor magnitude over the baseline, in AU. 2.0 is a
    #: perfect six-month baseline at the ecliptic pole; less means less leverage.
    parallax_leverage_au: float
    baseline_days: float
    #: Fractional distance spread contributed by the positional errors alone,
    #: separate from the spread contributed by not knowing the drift direction.
    frac_err_measurement: float

    @property
    def distance_au(self) -> float:
        """Geometric centre of the interval -- for display, never for a claim."""
        return math.sqrt(self.d_min_au * self.d_max_au)

    @property
    def interval_au(self) -> tuple[float, float]:
        """The interval widened by measurement error, which is what to quote."""
        f = min(0.9, self.frac_err_measurement)
        return self.d_min_au * (1.0 - f), self.d_max_au / max(1e-9, 1.0 - f)

    @property
    def is_degenerate(self) -> bool:
        """True when the drift ambiguity alone spans more than a factor of two."""
        return self.d_max_au / self.d_min_au > 2.0



def forecast_precision(
    ra_deg: float,
    dec_deg: float,
    mjds: list[float],
    *,
    d_au: float = 500.0,
    sigma_arcsec: float = 0.15,
) -> AstrometricSolution:
    """What this field's cadence and geometry could do, before any data.

    Plants a noiseless body at ``d_au``, samples it at the epochs this field
    actually has, and solves. The returned distance is exact by construction --
    the interesting numbers are :attr:`AstrometricSolution.distance_err_au` and
    :attr:`corr_parallax_pm`, which depend only on *where* and *when* the field
    was observed and not on anything in the pixels.

    That makes it a forecast rather than a result, and it is worth showing
    because the two things that set it are independent and easily confused. On
    the four baked fields, at 500 AU with 0.15 arcsec centroids:

        nep       beta +90.00   12 visits   440 d   +/- 0.054 AU
        sep       beta -90.00   11 visits    93 d   +/- 0.511 AU
        mid       beta +45.00    3 visits   365 d   +/- 0.115 AU
        ecliptic  beta  -0.52    4 visits   365 d   +/- 0.092 AU

    The two poles have identical geometry -- both perfect circles -- and differ
    by a factor of nine and a half, entirely because one was revisited over 440
    days and the other over 93. The mechanism shows up in the correlation:
    ``corr(pi, mu)`` is +0.16 at the north pole and +0.97 at the south, because
    a wobble sampled through a quarter of a turn has not yet reversed, and an
    un-reversed wobble is indistinguishable from drift. Both figures are pinned
    by ``FORECAST_BY_FIELD`` in ``tests/test_ephemeris.py``, because prose is
    not checked by anything and this pair was wrong here for a while.
    """
    obs = [
        Observation(m, p.ra_deg, p.dec_deg, sigma_arcsec)
        for m, p in ((m, apparent_place(ra_deg, dec_deg, d_au, m)) for m in mjds)
    ]
    return solve_astrometry(obs)

def distance_from_two_epochs(
    a: Observation,
    b: Observation,
    *,
    d_lo: float = 1.0,
    d_hi: float = 1.0e5,
    grid: int = 4000,
) -> TwoEpochDistance:
    """Bound the distance from exactly two positions, assuming only a bound orbit.

    Two epochs give four numbers; the free five-parameter model is
    underdetermined, so an assumption has to be bought. The one bought here is
    the weakest useful one: the body is on a **bound, roughly circular
    heliocentric orbit**, which fixes the *magnitude* of its drift at
    ``1.296e6/d^1.5`` arcsec/yr. Nothing is assumed about the drift's
    *direction*, so for any distance the observable displacement can be anywhere
    in

        | pi_disp(d) - orbital(d) |  <=  displacement  <=  pi_disp(d) + orbital(d)

    by the triangle inequality. The consistent set of distances is every ``d``
    satisfying that, and its endpoints are :attr:`TwoEpochDistance.d_min_au` and
    :attr:`~TwoEpochDistance.d_max_au`. That is a genuine bound rather than a
    point estimate dressed as one.

    The parallax displacement is not ``2*pi``: it is ``pi`` times the change in
    the parallax factor vector over the baseline, computed from Earth's actual
    position at both epochs. A six-month baseline at the ecliptic pole gives the
    full 2.0 AU of leverage; the same baseline in the ecliptic plane can give
    almost none, which is exactly the pole/plane asymmetry
    :func:`parallactic_ellipse` describes, and it appears here as a wider
    interval rather than as a wrong answer.

    A log grid is scanned rather than a monotone bisection, because
    ``|pi_disp - orbital|`` is *not* monotone in ``d`` -- it peaks near 5.5 AU
    for a six-month baseline -- and a bisection that assumes monotonicity
    returns a confident wrong endpoint in the main-belt regime.
    """
    from ..frames.wcs import angular_separation

    sep = angular_separation(a.ra_deg, a.dec_deg, b.ra_deg, b.dec_deg) * 3600.0
    dt_days = abs(b.mjd - a.mjd)
    if dt_days <= 0.0:
        raise ValueError("the two epochs must differ in time")
    if sep <= 0.0:
        raise ValueError("no measurable displacement between the two epochs")
    years = dt_days / YEAR_DAYS

    ea = earth_state(mjd_to_jd_tt(a.mjd))
    eb = earth_state(mjd_to_jd_tt(b.mjd))
    ra_m, dec_m = _radec_from_vec(
        tuple(
            x + y
            for x, y in zip(
                _unit_from_radec(a.ra_deg, a.dec_deg), _unit_from_radec(b.ra_deg, b.dec_deg)
            )
        )  # type: ignore[arg-type]
    )
    fa1, fd1 = parallax_factors(ra_m, dec_m, ea.as_tuple())
    fa2, fd2 = parallax_factors(ra_m, dec_m, eb.as_tuple())
    leverage = math.hypot(fa2 - fa1, fd2 - fd1)  # AU of baseline actually used
    if leverage <= 0.0:
        raise ValueError("this epoch pair has no parallax leverage at all")

    def parts(d: float) -> tuple[float, float]:
        return leverage * parallax_arcsec(d), orbital_rate_arcsec_per_year(d) * years

    def consistent(d: float) -> bool:
        par, orb = parts(d)
        return abs(par - orb) - 1e-12 <= sep <= par + orb + 1e-12

    # Scan in log d; the consistent set is an interval in every physical case,
    # but taking min and max of the hits costs nothing and survives if it is not.
    ratio = (d_hi / d_lo) ** (1.0 / (grid - 1))
    hits = [d_lo * ratio**i for i in range(grid) if consistent(d_lo * ratio**i)]
    if not hits:
        par_lo, orb_lo = parts(d_lo)
        raise ValueError(
            f"displacement {sep:.2f} arcsec is not reachable by a bound orbit between "
            f"{d_lo:g} and {d_hi:g} AU (at {d_lo:g} AU the most it could move is "
            f"{par_lo + orb_lo:.2f} arcsec) -- the two detections are probably not "
            "the same object"
        )

    def refine(inside: float, outside: float) -> float:
        """Bisect the consistency boundary between a hit and an adjacent miss."""
        for _ in range(60):
            mid = math.sqrt(inside * outside)
            if consistent(mid):
                inside = mid
            else:
                outside = mid
        return inside

    d_min, d_max = min(hits), max(hits)
    if d_min > d_lo * ratio:
        d_min = refine(d_min, d_min / ratio)
    if d_max < d_hi / ratio:
        d_max = refine(d_max, d_max * ratio)

    def solve_monotone(f) -> float:
        """Bisect a strictly decreasing f(d) = sep on [d_lo, d_hi]."""
        lo, hi = d_lo, d_hi
        if f(lo) < sep:
            return lo
        if f(hi) > sep:
            return hi
        for _ in range(200):
            mid = math.sqrt(lo * hi)
            if f(mid) > sep:
                lo = mid
            else:
                hi = mid
        return math.sqrt(lo * hi)

    d_aligned = solve_monotone(lambda d: sum(parts(d)))
    d_par_only = solve_monotone(lambda d: parts(d)[0])
    par, orb = parts(d_aligned)
    return TwoEpochDistance(
        d_min_au=d_min,
        d_max_au=d_max,
        d_aligned_au=d_aligned,
        d_parallax_only_au=d_par_only,
        displacement_arcsec=sep,
        parallax_part_arcsec=par,
        orbital_part_arcsec=orb,
        parallax_leverage_au=leverage,
        baseline_days=dt_days,
        frac_err_measurement=math.hypot(a.sigma_arcsec, b.sigma_arcsec) / sep,
    )
