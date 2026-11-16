"""HEALPix NESTED indexing, enough of it to rectify a HiPS all-sky.

The mirror of ``web/js/healpix.js``, and it exists for one reason: the bake has
to know where a tile *is* on the sky. Levelling the background steps between
mosaic tiles means comparing each tile against its neighbours, and tiles
adjacent in an Allsky sheet are not adjacent on the sky -- the sheet is a grid
of 768 independent faces-of-faces, and treating grid neighbours as sky
neighbours smears the seam it was meant to remove.

Two implementations of one piece of mathematics is a liability unless something
keeps them honest, which is the argument ``docs/WEB-TOOL.md`` §5 makes for the
WCS existing twice. So this is emitted into the golden vectors by
``scripts/emit_golden.py`` and replayed through the JavaScript by
``tests/js/run-golden.mjs``, on the same footing as the WCS and the solver.

The failure mode here is unusually quiet. A wrong index does not raise; it
returns *a* sky position, just not the right one, and a background model built
on it looks entirely plausible while being wrong everywhere. Hence the
cross-check rather than a spot check.

Stdlib only, like the rest of ``tracklet``. The callers work over tens of
thousands of cells, not millions, so scalar Python is not the bottleneck --
the pixels are grouped by index arithmetic, which needs no trigonometry at all.
"""

from __future__ import annotations

import math

__all__ = ["ang2pix_nest", "pix2ang_nest", "vec2pix_nest"]

#: Which ring of faces each base face sits on, and where around it. The twelve
#: faces are four around the north cap, four around the equator, four around
#: the south; these two tables are the standard encoding of that layout.
_JRLL = (2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4)
_JPLL = (1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7)


def _spread(v: int) -> int:
    """Insert a zero bit between each of the low 16 bits. 0b1011 -> 0b01000101."""
    v &= 0xFFFF
    v = (v | (v << 8)) & 0x00FF00FF
    v = (v | (v << 4)) & 0x0F0F0F0F
    v = (v | (v << 2)) & 0x33333333
    v = (v | (v << 1)) & 0x55555555
    return v


def _compress(v: int) -> int:
    """Inverse of :func:`_spread`: keep the even bits and pack them down."""
    v &= 0x55555555
    v = (v | (v >> 1)) & 0x33333333
    v = (v | (v >> 2)) & 0x0F0F0F0F
    v = (v | (v >> 4)) & 0x00FF00FF
    v = (v | (v >> 8)) & 0x0000FFFF
    return v


def _face_xy(order: int, z: float, tt_raw: float) -> tuple[int, int, int]:
    """Face number and (x, y) within it, from ``z = sin(dec)`` and ``tt = ra/90``."""
    nside = 1 << order
    za = abs(z)
    tt = math.fmod(tt_raw, 4.0)
    if tt < 0:
        tt += 4.0

    if za <= 2 / 3:
        # Equatorial belt. The two diagonal families of face edges are straight
        # lines in (phi, z) here, so which face a point is on falls out of two
        # floors rather than a search.
        temp1 = nside * (0.5 + tt)
        temp2 = nside * z * 0.75
        jp = int(math.floor(temp1 - temp2))   # ascending edge
        jm = int(math.floor(temp1 + temp2))   # descending edge
        ifp = jp >> order
        ifm = jm >> order
        if ifp == ifm:
            face = (ifp & 3) + 4
        elif ifp < ifm:
            face = ifp & 3
        else:
            face = (ifm & 3) + 8
        ix = jm & (nside - 1)
        iy = nside - (jp & (nside - 1)) - 1
    else:
        # Polar caps, where the faces are triangles in (phi, z) and the edges
        # are parabolic. sqrt(3(1-|z|)) is the substitution that straightens
        # them.
        ntt = min(3, int(math.floor(tt)))
        tp = tt - ntt
        tmp = nside * math.sqrt(3 * (1 - za))
        jp = min(int(math.floor(tp * tmp)), nside - 1)
        jm = min(int(math.floor((1 - tp) * tmp)), nside - 1)
        if z >= 0:
            face = ntt
            ix = nside - jm - 1
            iy = nside - jp - 1
        else:
            face = ntt + 8
            ix = jp
            iy = jm
    return face, ix, iy


def _packed(order: int, z: float, tt: float) -> int:
    nside = 1 << order
    face, ix, iy = _face_xy(order, z, tt)
    return face * nside * nside + _spread(ix) + 2 * _spread(iy)


def ang2pix_nest(order: int, ra_deg: float, dec_deg: float) -> int:
    """NESTED pixel index at ``order`` for a sky position."""
    return _packed(order, math.sin(math.radians(dec_deg)), ra_deg / 90.0)


def vec2pix_nest(order: int, x: float, y: float, z: float) -> int:
    """NESTED pixel index for a unit vector in equatorial coordinates."""
    return _packed(order, z, math.atan2(y, x) / (math.pi / 2))


def pix2ang_nest(order: int, pix: int) -> tuple[float, float]:
    """``(ra_deg, dec_deg)`` of the centre of a NESTED pixel."""
    nside = 1 << order
    npface = nside * nside
    face = pix // npface
    p = pix - face * npface
    ix = _compress(p)
    iy = _compress(p >> 1)

    fact2 = 4.0 / (12 * nside * nside)
    fact1 = 2 * nside * fact2
    jr = _JRLL[face] * nside - ix - iy - 1

    if jr < nside:
        nr = jr
        z = 1 - (nr * nr) * fact2
    elif jr > 3 * nside:
        nr = nside * 4 - jr
        z = (nr * nr) * fact2 - 1
    else:
        nr = nside
        z = (2 * nside - jr) * fact1

    tmp = _JPLL[face] * nr + ix - iy
    if tmp < 0:
        tmp += 8 * nr
    phi = 0.0 if nr == 0 else (math.pi / 4) * tmp / nr

    return (
        math.degrees(phi) % 360.0,
        math.degrees(math.asin(max(-1.0, min(1.0, z)))),
    )
