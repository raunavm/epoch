"""Gnomonic (TAN) world coordinates with SIP distortion, exact in both directions.

Why this exists rather than a call into astropy: the browser side of Tracklet has
to run the identical transform on the identical bytes, and the only way to keep
two implementations honest is to have one authoritative version with golden
vectors generated from it (``scripts/emit_golden.py``). This module is that
authority. It is stdlib-only for the same reason the ``archive`` package is.

Conventions, stated because getting them wrong is silent
--------------------------------------------------------
* **Pixel coordinates are 1-based FITS**, pixel centres on integers, matching
  ``CRPIX``. Array indices are ``x - 1``; the conversion happens only at the
  point of array access, never here.
* ``CDELT``/``PC`` and ``CD`` both collapse to one matrix in deg/pixel via
  :func:`tracklet.archive.headers.matrix_from_cards`.
* Intermediate world coordinates are degrees, ``x`` east-positive and
  ``y`` north-positive, per WCS Paper II. The sign flip that makes right
  ascension run right-to-left across the detector lives in the CD matrix, not
  here -- SPHEREx ships ``PC1_1 < 0``.

The SIP trap
------------
SIP (Shupe et al. 2005) is

    u' = u + sum A_pq u^p v^q,    v' = v + sum B_pq u^p v^q,   p + q <= order

with ``u = x - CRPIX1``. Conventionally ``A_00``, ``A_10``, ``A_01`` are zero
because they are degenerate with ``CRPIX`` and ``CD`` -- and a good deal of
hand-rolled SIP code therefore skips them. **SPHEREx does not set them to
zero**: a real QR2 frame carries ``A_0_0 = 0.4829``, ``B_0_0 = -1.0932``.
Dropping them displaces every position by ~1.2 pixels, which is 7 arcsec, which
is 50x the mission's own astrometric error. Every coefficient present in the
header is applied. :func:`tests/test_wcs.py` guards this.

The inverse
-----------
``AP_``/``BP_`` are a *fitted* approximation to the inverse polynomial, not an
algebraic one, so using them alone leaves a residual -- on real SPHEREx
coefficients about 1e-3 pixel, which is 6 milliarcsec. That is small, but it is
not zero, and it is asymmetric in a way that biases a stack. :func:`Wcs.world_to_pixel`
therefore uses ``AP``/``BP`` only as a starting guess and then Newton-iterates
on the *forward* transform with its analytic Jacobian, converging to ~1e-11
pixel in two or three steps. Round-trip closure is then limited by float64, and
:func:`Wcs.roundtrip_residual` measures it rather than asserting it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..archive.headers import matrix_from_cards

DEG = math.pi / 180.0
ARCSEC = DEG / 3600.0

#: Newton iteration stops when the pixel correction falls below this, in pixels.
#: 1e-11 px is 6e-8 arcsec -- far below float64 noise in the forward transform,
#: so in practice this means "iterate until it stops improving".
INVERSE_TOL_PX = 1e-11
INVERSE_MAX_ITER = 24


class WcsError(ValueError):
    """The header does not describe a TAN projection we can evaluate."""


class NotConverged(WcsError):
    """Newton iteration on the SIP inverse failed to close."""


# --------------------------------------------------------------------------- #
# SIP
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Sip:
    """One SIP polynomial pair, stored as ``{(p, q): coefficient}``.

    ``a``/``b`` are the forward (detector -> intermediate) polynomials and are
    required. ``ap``/``bp`` are the fitted inverse and are optional -- absent,
    the Newton iteration in :meth:`Wcs.world_to_pixel` simply starts from the
    identity, costing an extra step or two.
    """

    a: dict[tuple[int, int], float] = field(default_factory=dict)
    b: dict[tuple[int, int], float] = field(default_factory=dict)
    ap: dict[tuple[int, int], float] = field(default_factory=dict)
    bp: dict[tuple[int, int], float] = field(default_factory=dict)

    @property
    def order(self) -> int:
        terms = tuple(self.a) + tuple(self.b)
        return max((p + q for p, q in terms), default=0)

    @property
    def has_inverse(self) -> bool:
        return bool(self.ap or self.bp)

    def __bool__(self) -> bool:
        return bool(self.a or self.b)

    # -- evaluation ------------------------------------------------------- #

    @staticmethod
    def _poly(coeffs: dict[tuple[int, int], float], u: float, v: float) -> float:
        """Evaluate ``sum c_pq u^p v^q``.

        Straight Horner is awkward for a sparse bivariate polynomial, so powers
        are memoised per call instead; SIP order is 3 for SPHEREx so this is a
        handful of multiplies either way.
        """
        if not coeffs:
            return 0.0
        pmax = max(p for p, _ in coeffs)
        qmax = max(q for _, q in coeffs)
        up = [1.0] * (pmax + 1)
        for i in range(1, pmax + 1):
            up[i] = up[i - 1] * u
        vq = [1.0] * (qmax + 1)
        for i in range(1, qmax + 1):
            vq[i] = vq[i - 1] * v
        return math.fsum(c * up[p] * vq[q] for (p, q), c in coeffs.items())

    @staticmethod
    def _dpoly(coeffs: dict[tuple[int, int], float], u: float, v: float) -> tuple[float, float]:
        """Partial derivatives ``(d/du, d/dv)`` of :meth:`_poly`."""
        du = math.fsum(c * p * u ** (p - 1) * v**q for (p, q), c in coeffs.items() if p >= 1)
        dv = math.fsum(c * q * u**p * v ** (q - 1) for (p, q), c in coeffs.items() if q >= 1)
        return du, dv

    def forward(self, u: float, v: float) -> tuple[float, float]:
        """Distorted (detector) offsets -> undistorted intermediate offsets."""
        return u + self._poly(self.a, u, v), v + self._poly(self.b, u, v)

    def jacobian(self, u: float, v: float) -> tuple[float, float, float, float]:
        """``(d u'/du, d u'/dv, d v'/du, d v'/dv)`` of :meth:`forward`."""
        a_du, a_dv = self._dpoly(self.a, u, v)
        b_du, b_dv = self._dpoly(self.b, u, v)
        return 1.0 + a_du, a_dv, b_du, 1.0 + b_dv

    def inverse_guess(self, up: float, vp: float) -> tuple[float, float]:
        """Undistorted -> distorted using the fitted ``AP``/``BP``, approximate."""
        if not self.has_inverse:
            return up, vp
        return up + self._poly(self.ap, up, vp), vp + self._poly(self.bp, up, vp)

    def inverse(self, up: float, vp: float) -> tuple[float, float]:
        """Undistorted -> distorted, Newton-refined to :data:`INVERSE_TOL_PX`.

        The Jacobian of a 3rd-order SIP map is far from singular anywhere on a
        real detector (it is the identity plus ~1e-3), so plain Newton converges
        quadratically and needs no damping. Non-convergence here means the
        coefficients are not a distortion map at all, hence the raise.
        """
        if not self:
            return up, vp
        u, v = self.inverse_guess(up, vp)
        for _ in range(INVERSE_MAX_ITER):
            fu, fv = self.forward(u, v)
            ru, rv = fu - up, fv - vp
            j11, j12, j21, j22 = self.jacobian(u, v)
            det = j11 * j22 - j12 * j21
            if det == 0.0:
                raise NotConverged("singular SIP Jacobian")
            du = (ru * j22 - rv * j12) / det
            dv = (rv * j11 - ru * j21) / det
            u -= du
            v -= dv
            if abs(du) < INVERSE_TOL_PX and abs(dv) < INVERSE_TOL_PX:
                return u, v
        raise NotConverged(f"SIP inverse did not converge in {INVERSE_MAX_ITER} iterations")


def sip_from_cards(cards: dict[str, str]) -> Sip:
    """Collect ``A_p_q``/``B_p_q``/``AP_p_q``/``BP_p_q`` out of a header.

    Order is taken from the ``*_ORDER`` cards when present, but coefficients are
    read by pattern regardless, so a header that carries a term above its stated
    order is still honoured rather than silently truncated.
    """
    out: dict[str, dict[tuple[int, int], float]] = {
        "a": {},
        "b": {},
        "ap": {},
        "bp": {},
    }
    prefixes = (("AP_", "ap"), ("BP_", "bp"), ("A_", "a"), ("B_", "b"))
    for key, raw in cards.items():
        for prefix, slot in prefixes:
            if not key.startswith(prefix):
                continue
            tail = key[len(prefix) :]
            if tail == "ORDER":
                break
            bits = tail.split("_")
            if len(bits) != 2 or not all(b.isdigit() for b in bits):
                break
            p, q = int(bits[0]), int(bits[1])
            try:
                value = float(raw)
            except (TypeError, ValueError):
                break
            if value != 0.0:
                out[slot][(p, q)] = value
            break
    return Sip(a=out["a"], b=out["b"], ap=out["ap"], bp=out["bp"])


# --------------------------------------------------------------------------- #
# Gnomonic projection, in vector form
# --------------------------------------------------------------------------- #


def _basis(
    ra0_deg: float, dec0_deg: float
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    """Orthonormal frame at the tangent point: ``(toward, east, north)``."""
    a, d = ra0_deg * DEG, dec0_deg * DEG
    ca, sa, cd, sd = math.cos(a), math.sin(a), math.cos(d), math.sin(d)
    toward = (cd * ca, cd * sa, sd)
    east = (-sa, ca, 0.0)
    north = (-sd * ca, -sd * sa, cd)
    return toward, east, north


def tan_deproject(
    x_deg: float, y_deg: float, ra0_deg: float, dec0_deg: float
) -> tuple[float, float]:
    """Intermediate world coords -> ``(ra, dec)`` in degrees.

    Vector form: a point at tangent-plane offset ``(x, y)`` radians lies along
    ``toward + x*east + y*north``. Normalising and reading off spherical angles
    is exactly the gnomonic deprojection, with no quadrant cases and no
    division that blows up away from the antipode -- unlike the textbook
    ``atan2`` form, which needs guarding when ``cos(dec0) - y sin(dec0)``
    approaches zero.
    """
    toward, east, north = _basis(ra0_deg, dec0_deg)
    xr, yr = x_deg * DEG, y_deg * DEG
    vx = toward[0] + xr * east[0] + yr * north[0]
    vy = toward[1] + xr * east[1] + yr * north[1]
    vz = toward[2] + xr * east[2] + yr * north[2]
    n = math.sqrt(vx * vx + vy * vy + vz * vz)
    ra = math.atan2(vy, vx) / DEG
    dec = math.asin(max(-1.0, min(1.0, vz / n))) / DEG
    return ra % 360.0, dec


def tan_project(
    ra_deg: float, dec_deg: float, ra0_deg: float, dec0_deg: float
) -> tuple[float, float]:
    """``(ra, dec)`` -> intermediate world coords in degrees.

    Raises :class:`WcsError` for directions on or behind the tangent plane's
    horizon, where the gnomonic projection genuinely does not exist. Callers
    plotting a trajectory that leaves the field want that as an error, not as a
    silently mirrored position on the far side of the sky.
    """
    toward, east, north = _basis(ra0_deg, dec0_deg)
    a, d = ra_deg * DEG, dec_deg * DEG
    ca, sa, cd, sd = math.cos(a), math.sin(a), math.cos(d), math.sin(d)
    r = (cd * ca, cd * sa, sd)
    w = r[0] * toward[0] + r[1] * toward[1] + r[2] * toward[2]
    if w <= 1e-12:
        raise WcsError("direction is >=90 deg from the tangent point")
    xe = r[0] * east[0] + r[1] * east[1] + r[2] * east[2]
    yn = r[0] * north[0] + r[1] * north[1] + r[2] * north[2]
    return (xe / w) / DEG, (yn / w) / DEG


def angular_separation(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """Great-circle separation in degrees, via the haversine form.

    The ``acos`` of a dot product loses all precision below ~1 arcsec, which is
    precisely the regime this project cares about (SPHEREx astrometry is 0.15
    arcsec), so haversine is not optional here.
    """
    d1, d2 = dec1 * DEG, dec2 * DEG
    dd, da = d2 - d1, (ra2 - ra1) * DEG
    h = math.sin(dd / 2) ** 2 + math.cos(d1) * math.cos(d2) * math.sin(da / 2) ** 2
    return 2.0 * math.asin(min(1.0, math.sqrt(h))) / DEG


# --------------------------------------------------------------------------- #
# The full transform
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Wcs:
    """A complete TAN-SIP solution for one frame.

    Construct from a header with :func:`from_cards`; the JSON form produced by
    :meth:`to_dict` is what the browser consumes, and is deliberately flat so
    the JavaScript side has no parsing to do.
    """

    crval1: float
    crval2: float
    crpix1: float
    crpix2: float
    cd: tuple[tuple[float, float], tuple[float, float]]
    sip: Sip = field(default_factory=Sip)
    naxis1: int = 0
    naxis2: int = 0

    # -- derived ---------------------------------------------------------- #

    @property
    def det(self) -> float:
        (a, b), (c, d) = self.cd
        return a * d - b * c

    @property
    def scale_arcsec(self) -> float:
        """Plate scale from ``sqrt(|det CD|)`` -- see ``headers.pixel_scale_arcsec``."""
        return math.sqrt(abs(self.det)) * 3600.0

    @property
    def pixel_area_sr(self) -> float:
        """Pixel solid angle in steradian, the conversion MJy/sr -> Jy needs."""
        return (self.scale_arcsec * ARCSEC) ** 2

    @property
    def flips_parity(self) -> bool:
        """True when the CD matrix mirrors the sky (``det > 0`` for east-left)."""
        return self.det > 0.0

    @property
    def position_angle_deg(self) -> float:
        """Rotation of detector +y from north, degrees east of north.

        Taken from the CD matrix columns rather than a single element, so it is
        correct under a parity flip.

        This is a property of the *frame*, evaluated at the reference pixel. It
        is the right number for a footprint or a header readout, and the wrong
        one for de-rotating a cutout taken far from ``CRPIX`` -- see
        :meth:`local_north_angle_deg`.
        """
        (_, b), (_, d) = self.cd
        return math.degrees(math.atan2(b, d)) % 360.0

    def local_north_angle_deg(self, x: float, y: float, step_arcsec: float = 5.0) -> float:
        """Rotation of detector +y from north *at this pixel*, degrees.

        :meth:`position_angle_deg` reads the CD matrix, which is the linear
        term at ``CRPIX``. Across a 2040-pixel detector with third-order SIP the
        sky orientation is not constant, and a stamp is rarely cut near the
        reference pixel: the North Ecliptic Pole's twelve cutouts sit between
        rows 188 and 1912, where the local angle departs from the header's by
        up to 5.6 degrees. At a stamp's corner, 96 pixels out, 5.6 degrees is a
        9.4-pixel error -- so de-rotating by the header angle leaves a visible
        smear that de-rotating by this one does not.

        Measured against the pixels rather than argued: rotating visit 1 of the
        NEP field to match visit 2, the cross-correlation peak sits at the
        angle this returns, not at the difference of the two header values.

        Computed from the Jacobian by finite difference -- step north on the
        sky, see which way that points in pixels -- so it carries the SIP
        distortion the CD matrix alone does not.
        """
        ra, dec = self.pixel_to_world(x, y)
        # Guard the pole: stepping north past +90 wraps the longitude.
        h = step_arcsec / 3600.0
        if dec + h > 90.0:
            h = -h
        xn, yn = self.world_to_pixel(ra, dec + h)
        dx, dy = (xn - x), (yn - y)
        if h < 0:
            dx, dy = -dx, -dy
        return math.degrees(math.atan2(dx, dy)) % 360.0

    # -- construction ----------------------------------------------------- #

    @classmethod
    def north_up(
        cls,
        ra_deg: float,
        dec_deg: float,
        scale_arcsec: float,
        size: int,
    ) -> Wcs:
        """A plain TAN grid, north up and east left, centred on a position.

        This is the frame a resampled cutout lives in. SPHEREx rolls between
        visits -- 31 to 81 degrees between consecutive North Ecliptic Pole
        visits -- so cutouts taken as detector-aligned pixel boxes arrive at
        wildly different orientations, and the same star sits in a different
        corner every time. Reprojecting each one onto a grid like this puts
        every visit of a field into one common frame.

        No SIP, and that is the point rather than an omission: the reprojection
        evaluates the source frame's distortion per output pixel, so what comes
        out is already undistorted. A cutout carrying this WCS can be read with
        plain linear arithmetic.

        East-left is the astronomical convention, and every SPHEREx frame these
        fields are built from has the same parity, so adopting it flips nothing.

        ``size`` is the side of the square cutout in pixels; the reference pixel
        lands at its centre, in the 1-based convention FITS and this class use.
        """
        centre = (size + 1) / 2.0
        s = scale_arcsec / 3600.0
        # Negative on the first axis: RA increases to the left.
        cd = ((-s, 0.0), (0.0, s))
        return cls(
            crval1=ra_deg,
            crval2=dec_deg,
            crpix1=centre,
            crpix2=centre,
            cd=cd,
            sip=None,
            naxis1=size,
            naxis2=size,
        )

    @classmethod
    def from_cards(cls, cards: dict[str, str]) -> Wcs:
        ctype1 = cards.get("CTYPE1", "")
        if "TAN" not in ctype1:
            raise WcsError(f"CTYPE1={ctype1!r} is not a TAN projection")
        return cls(
            crval1=float(cards["CRVAL1"]),
            crval2=float(cards["CRVAL2"]),
            crpix1=float(cards["CRPIX1"]),
            crpix2=float(cards["CRPIX2"]),
            cd=matrix_from_cards(cards),
            sip=sip_from_cards(cards) if "SIP" in ctype1 else Sip(),
            naxis1=int(cards.get("NAXIS1", 0) or 0),
            naxis2=int(cards.get("NAXIS2", 0) or 0),
        )

    def to_dict(self) -> dict:
        def pack(c: dict[tuple[int, int], float]) -> dict[str, float]:
            return {f"{p}_{q}": v for (p, q), v in sorted(c.items())}

        out: dict = {
            "crval": [self.crval1, self.crval2],
            "crpix": [self.crpix1, self.crpix2],
            "cd": [list(self.cd[0]), list(self.cd[1])],
            "naxis": [self.naxis1, self.naxis2],
        }
        if self.sip:
            out["sip"] = {
                "order": self.sip.order,
                "a": pack(self.sip.a),
                "b": pack(self.sip.b),
                "ap": pack(self.sip.ap),
                "bp": pack(self.sip.bp),
            }
        return out

    @classmethod
    def from_dict(cls, d: dict) -> Wcs:
        """Rebuild from :meth:`to_dict`, so a manifest round-trips.

        The SIP block is optional and absent by design on a reprojected
        cutout: the resampling evaluates the source distortion per pixel, so
        what it writes is already undistorted and linear.
        """
        def unpack(c: dict[str, float]) -> dict[tuple[int, int], float]:
            out: dict[tuple[int, int], float] = {}
            for k, v in (c or {}).items():
                p, q = k.split("_")
                out[(int(p), int(q))] = float(v)
            return out

        sip_d = d.get("sip")
        sip = (
            Sip(
                a=unpack(sip_d.get("a", {})),
                b=unpack(sip_d.get("b", {})),
                ap=unpack(sip_d.get("ap", {})),
                bp=unpack(sip_d.get("bp", {})),
            )
            if sip_d
            else None
        )
        cd = d["cd"]
        return cls(
            crval1=float(d["crval"][0]),
            crval2=float(d["crval"][1]),
            crpix1=float(d["crpix"][0]),
            crpix2=float(d["crpix"][1]),
            cd=((float(cd[0][0]), float(cd[0][1])), (float(cd[1][0]), float(cd[1][1]))),
            sip=sip,
            naxis1=int(d["naxis"][0]),
            naxis2=int(d["naxis"][1]),
        )

    # -- the transform ---------------------------------------------------- #

    def pixel_to_world(self, x: float, y: float) -> tuple[float, float]:
        """1-based FITS pixel -> ``(ra, dec)`` degrees."""
        u, v = x - self.crpix1, y - self.crpix2
        f, g = self.sip.forward(u, v) if self.sip else (u, v)
        (c11, c12), (c21, c22) = self.cd
        return tan_deproject(c11 * f + c12 * g, c21 * f + c22 * g, self.crval1, self.crval2)

    def world_to_pixel(self, ra: float, dec: float) -> tuple[float, float]:
        """``(ra, dec)`` degrees -> 1-based FITS pixel, exactly.

        The linear step is inverted algebraically; only the SIP step iterates.
        """
        xw, yw = tan_project(ra, dec, self.crval1, self.crval2)
        (c11, c12), (c21, c22) = self.cd
        det = c11 * c22 - c12 * c21
        if det == 0.0:
            raise WcsError("singular CD matrix")
        f = (xw * c22 - yw * c12) / det
        g = (yw * c11 - xw * c21) / det
        u, v = self.sip.inverse(f, g) if self.sip else (f, g)
        return u + self.crpix1, v + self.crpix2

    def contains(self, x: float, y: float, *, margin: float = 0.0) -> bool:
        """Is a 1-based pixel inside the array, allowing ``margin`` pixels in?"""
        if not (self.naxis1 and self.naxis2):
            return True
        return (
            0.5 + margin <= x <= self.naxis1 + 0.5 - margin
            and 0.5 + margin <= y <= self.naxis2 + 0.5 - margin
        )

    def footprint(self) -> list[tuple[float, float]]:
        """Corner sky positions, counter-clockwise from the (1, 1) corner."""
        n1, n2 = self.naxis1 or 1, self.naxis2 or 1
        corners = ((0.5, 0.5), (n1 + 0.5, 0.5), (n1 + 0.5, n2 + 0.5), (0.5, n2 + 0.5))
        return [self.pixel_to_world(x, y) for x, y in corners]

    # -- self-check ------------------------------------------------------- #

    def roundtrip_residual(self, step: int = 255) -> float:
        """Worst pixel->world->pixel closure error over the detector, in pixels.

        Reported rather than asserted: it is the honest measure of how well the
        SIP inverse is doing, and the UI shows it so a viewer can see that the
        astrometry is not the limiting factor.
        """
        worst = 0.0
        n1, n2 = self.naxis1 or 2040, self.naxis2 or 2040
        for x in range(1, n1 + 1, step):
            for y in range(1, n2 + 1, step):
                ra, dec = self.pixel_to_world(x, y)
                xx, yy = self.world_to_pixel(ra, dec)
                worst = max(worst, math.hypot(xx - x, yy - y))
        return worst

    def approx_inverse_residual(self, step: int = 255) -> float:
        """Same closure error if ``AP``/``BP`` were trusted without refinement.

        This is the number that justifies the Newton step; on SPHEREx frames it
        is ~1e-3 pixel while :meth:`roundtrip_residual` is ~1e-11.
        """
        if not self.sip.has_inverse:
            return float("nan")
        (c11, c12), (c21, c22) = self.cd
        det = c11 * c22 - c12 * c21
        worst = 0.0
        n1, n2 = self.naxis1 or 2040, self.naxis2 or 2040
        for x in range(1, n1 + 1, step):
            for y in range(1, n2 + 1, step):
                ra, dec = self.pixel_to_world(x, y)
                xw, yw = tan_project(ra, dec, self.crval1, self.crval2)
                f = (xw * c22 - yw * c12) / det
                g = (yw * c11 - xw * c21) / det
                u, v = self.sip.inverse_guess(f, g)
                worst = max(worst, math.hypot(u + self.crpix1 - x, v + self.crpix2 - y))
        return worst
