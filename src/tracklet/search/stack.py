"""Add frames along a trajectory, so a source too faint to see in one appears.

This is the answer to the question :mod:`tracklet.search.link` leaves hanging.
``scripts/known_movers.py`` asks JPL what was really in the ecliptic field at
its four visit times and gets seventy-seven catalogued asteroids back, the
brightest at V = 19.9 against a single-frame depth of 18.54 AB. Nothing is
wrong with the search and the sky is not empty: those objects are simply below
the noise of any one exposure. The only way to reach them is to use more than
one exposure at a time.

Co-adding is trivial for something that does not move -- you add the frames.
For something that does, you have to add them *along the path it took*, which
means you have to assume a path first. That is the whole idea and the whole
cost: a distance hypothesis fixes where the body would be in every frame, the
frames are shifted so those positions coincide, and the sum is taken. Signal
adds as ``N`` and noise as ``sqrt(N)``, so

    depth gain = 2.5 * log10(sqrt(N))  magnitudes

which for the twelve NEP epochs is **1.35 mag** -- 18.54 AB becomes 19.89, and
a source that contributed 3 sigma to each frame contributes 10.4 to the stack.
Guess the wrong distance and the same arithmetic works against you: the source
smears over as many pixels as the error is wide and you lose it, which is why
the hypothesis grid exists and why a stack is evidence only when you say which
trajectory produced it.

The co-add is **inverse-variance weighted**. It was adopted because an
unweighted mean of the twelve NEP epochs appeared to make the noise *worse* --
0.0305 against a typical frame's 0.0180, a loss of 0.57 magnitudes -- with
epoch 0 measuring twenty times the pixel scatter of the other eleven.

That measurement was reading an artefact, and the honest version matters here
because it is the difference between a rule and a special case. Epoch 0's
cutout lands near the bottom of the detector, where the frame carries a
brightness ramp, and the MAD was counting the ramp as noise. ``flatten_rows``
in ``scripts/bake_field.py`` removes it at bake time, and epoch 0 is then an
ordinary frame at 2.3 times the typical scatter rather than twenty. Weighted,
the twelve now go from a typical 0.0135 to 0.0040, a gain of 1.31 mag against
``sqrt(12)``'s 1.35; unweighted they reach 0.0046, 1.17 mag. So the weighting
buys about 0.14 magnitudes here rather than rescuing the stack from a loss --
still the better estimator, and still the correct handling of a genuinely bad
frame, but no longer load-bearing. ``docs/WEB-TOOL.md`` §7.6 carries the full
correction and ``scripts/measure_stack.py`` re-measures every figure in this
paragraph.

So each frame is weighted by ``1/sigma**2``, with ``sigma`` measured from the
frame's own pixels by median absolute deviation. A frame twenty times noisier
than its neighbours then contributes four hundred times less instead of
equally, and no frame has to be excluded by hand. The weight is a per-frame
scalar rather than the Level-2 VARIANCE plane: the plane is the principled
choice and costs another row-band fetch per epoch, and a scalar recovers the
difference between a good exposure and a bad one, which is what was actually
wrong here.

Shifts here are **integer pixels**, and that is not free. Rounding scatters
each frame's contribution by up to half a pixel, and SPHEREx's PSF is only
0.6 px wide -- comparable to the error -- so the stacked profile is broadened
and some of the ``sqrt(N)`` is given back. Measured on twelve frames with a
3-sigma source planted in each: the ideal is 10.4 sigma, the stack delivers
**6.5**, so integer shifts cost **0.52 magnitudes**.

That is five times what a first guess would suggest, and it is still the right
trade. Sub-pixel resampling correlates the noise between neighbouring pixels,
and once the noise is correlated every significance the detector reports on the
result is wrong in a direction that flatters it. A stack that is half a
magnitude shallower and whose error bars mean what they say is worth more than
one that is deeper and cannot be quoted. The gain is real either way: 3 sigma
per frame is invisible and 6.5 sigma in the stack is a detection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from tracklet.frames.stretch import mad_sigma, median_sorted
from tracklet.physics.ephemeris import apparent_place, drift_direction

Array2D = list[list[float]]


def frame_sigma(rows: Array2D) -> float:
    """A frame's robust pixel scatter, by median absolute deviation.

    MAD rather than a standard deviation because a SPHEREx stamp is mostly sky
    with stars in it, and a few hundred bright pixels would set an RMS while
    saying nothing about the noise a faint source has to clear.

    Mirrored exactly by ``frameSigma`` in ``web/js/stack.js``; both take the
    same element of the same float64 values, so they agree to rounding.
    """
    vals = sorted(v for row in rows for v in row if v == v)
    if not vals:
        return float("nan")
    return mad_sigma(vals, median_sorted(vals))


def frame_weights(stamps: list[dict]) -> dict[int, float]:
    """Inverse-variance weight per epoch, from each frame's own scatter.

    A frame may supply ``sigma`` directly; otherwise it is measured from the
    pixels. Frames whose scatter cannot be measured -- empty, or flat -- get
    the median weight rather than being dropped or given infinite influence.
    """
    sigmas: dict[int, float] = {}
    for s in stamps:
        sig = s.get("sigma")
        if sig is None:
            sig = frame_sigma(s["rows"])
        sigmas[s["i"]] = float(sig)

    usable = sorted(v for v in sigmas.values() if v == v and v > 0.0)
    if not usable:
        return {i: 1.0 for i in sigmas}
    fallback = median_sorted(usable)
    return {
        i: 1.0 / (v * v) if (v == v and v > 0.0) else 1.0 / (fallback * fallback)
        for i, v in sigmas.items()
    }


def static_sky_template(
    stamps: list[dict],
    exclude: dict[int, tuple[float, float]] | None = None,
    radius_px: float = 2.5,
) -> Array2D:
    """The sky that does not move: a per-pixel median across frames on one grid.

    Stacking along a moving body's track holds the body still and drags every
    star along the track instead. At 500 AU on the North Ecliptic Pole that
    turns each star into a loop 134 pixels across, and on that crowded field the
    loops cover the stack: a 12-sigma body planted in all twelve frames and
    stacked along its own trajectory was not separable from them at all.
    Subtracting what every frame has in common before shifting removes the
    stars and leaves what moved.

    ``exclude`` maps an epoch to the ``(x, y)`` array position where the body
    is predicted in that frame, and pixels within ``radius_px`` of it are left
    out of that frame's vote. Without it a body that crosses its own path, or
    one that barely moves, would be part of the template and subtract itself.
    A pixel every frame excluded comes back NaN, which is the honest answer
    for a body that never left it: nothing distinguishes it from a star there.

    The frames must share one pixel grid. The baked stamps do: each is
    reprojected north-up about the field centre at the frame's own scale, and
    those scales agree to 0.01 px across the stamp.
    """
    if not stamps:
        return []
    exclude = exclude or {}
    ny = len(stamps[0]["rows"])
    nx = len(stamps[0]["rows"][0]) if ny else 0
    r2 = radius_px * radius_px
    frames = [(s["rows"], exclude.get(s["i"])) for s in stamps]
    out: Array2D = [[float("nan")] * nx for _ in range(ny)]
    for j in range(ny):
        band = [(rows[j], at) for rows, at in frames]
        orow = out[j]
        for i in range(nx):
            vals = sorted(
                row[i] for row, at in band
                if (at is None or (i - at[0]) ** 2 + (j - at[1]) ** 2 > r2) and row[i] == row[i]
            )
            if vals:
                orow[i] = median_sorted(vals)
    return out


def static_source_mask(template: Array2D, k: float = 5.0) -> list[list[bool]]:
    """Pixels a static source leaves unusable after subtraction, grown by one.

    Subtraction removes a star to about a percent, and a percent of a star at
    thousands of sigma is still tens of sigma. Measured on the NEP stack along
    500 AU, subtraction alone left 206 residuals above 6.5 sigma; masking where
    the template stands ``k`` sigma above its own median, and the ring of pixels
    around each, is what a moving-object search does with them too.
    """
    vals = sorted(v for row in template for v in row if v == v)
    ny = len(template)
    nx = len(template[0]) if ny else 0
    mask = [[False] * nx for _ in range(ny)]
    if not vals:
        return mask
    med = median_sorted(vals)
    sigma = mad_sigma(vals, med)
    if not sigma > 0.0:
        return mask
    cut = med + k * sigma
    for j in range(ny):
        for i in range(nx):
            v = template[j][i]
            if v == v and v > cut:
                for jj in range(max(0, j - 1), min(ny, j + 2)):
                    for ii in range(max(0, i - 1), min(nx, i + 2)):
                        mask[jj][ii] = True
    return mask


@dataclass
class Stack:
    """A co-added stamp, and everything needed to judge it."""

    #: Inverse-variance weighted mean of the contributing frames, in the
    #: input's units.
    rows: Array2D
    #: Per-pixel count of frames that actually contributed. Edge pixels get
    #: fewer, because a shifted frame does not cover the whole output.
    counts: list[list[int]]
    #: Frames that landed at all.
    n_frames: int
    #: The hypothesis this stack is evidence *for*. A stack without one is a
    #: picture, not a measurement.
    distance_au: float
    #: Integer (dx, dy) applied to each contributing epoch.
    shifts: list[tuple[int, int, int]] = field(default_factory=list)
    #: The inverse-variance weight each epoch was given.
    weights: dict[int, float] = field(default_factory=dict)

    @property
    def depth_gain_mag(self) -> float:
        """Magnitudes of depth bought, ``2.5*log10(sqrt(N))``.

        Quoted against the *median* per-pixel count rather than the frame
        count, because the corners of a stack are shallower than its middle and
        a number that ignores that is an advertisement.
        """
        n = self.median_count
        return 2.5 * math.log10(math.sqrt(n)) if n > 0 else 0.0

    @property
    def median_count(self) -> float:
        flat = sorted(c for row in self.counts for c in row if c > 0)
        if not flat:
            return 0.0
        mid = len(flat) // 2
        return float(flat[mid] if len(flat) % 2 else 0.5 * (flat[mid - 1] + flat[mid]))

    @property
    def effective_count(self) -> float:
        """``(sum w)^2 / sum w^2`` -- how many frames the weighting is *worth*.

        Twelve frames of which one is appreciably noisier are worth about
        eleven -- 11.2 on the NEP field, where one is 2.3 times the rest --
        and quoting twelve would be claiming depth the weighting deliberately
        declined to take. Equals the frame count exactly when the weights are
        equal.
        """
        ws = [w for w in self.weights.values() if w > 0]
        if not ws:
            return 0.0
        total = sum(ws)
        return (total * total) / sum(w * w for w in ws)


def trajectory_shifts(
    stamps: list[dict],
    ra_deg: float,
    dec_deg: float,
    d_au: float,
    *,
    ref: int = 0,
    pm_ra_arcsec_yr: float = 0.0,
    pm_dec_arcsec_yr: float = 0.0,
) -> list[tuple[int, int, int]]:
    """Integer pixel shift per epoch that puts the hypothesis on one spot.

    ``stamps`` are dicts with ``i``, ``mjd``, ``wcs`` (a :class:`Wcs`), ``x0``
    and ``y0`` -- the 1-based FITS pixel of the stamp's lower-left corner.
    Returns ``(epoch, dx, dy)``, the shift to *apply* to that epoch's array.
    """
    if not stamps:
        return []
    t0 = stamps[ref]["mjd"]
    places: list[tuple[int, float, float]] = []
    for s in stamps:
        yr = (s["mjd"] - t0) / 365.256363004
        ra_h, dec_h = drift_direction(ra_deg, dec_deg, pm_ra_arcsec_yr, pm_dec_arcsec_yr, yr)
        ap = apparent_place(ra_h, dec_h, d_au, s["mjd"])
        try:
            x, y = s["wcs"].world_to_pixel(ap.ra_deg, ap.dec_deg)
        except Exception:  # noqa: BLE001 - off this frame's projection entirely
            continue
        places.append((s["i"], x - s["x0"], y - s["y0"]))
    if not places:
        return []
    # Line the frames up on the *mean* predicted position, not on the reference
    # epoch's. The reference sits somewhere on the parallax ellipse rather than
    # at its centre, so aligning to it doubles every shift -- at 500 AU that is
    # +/-134 pixels instead of +/-67, and on a 193-pixel stamp the difference is
    # whether twelve frames overlap where the hypothesis points or eight do.
    bx = sum(p[1] for p in places) / len(places)
    by = sum(p[2] for p in places) / len(places)
    return [(i, int(round(bx - x)), int(round(by - y))) for i, x, y in places]


def shift_and_stack(
    stamps: list[dict],
    ra_deg: float,
    dec_deg: float,
    d_au: float,
    *,
    ref: int = 0,
    pm_ra_arcsec_yr: float = 0.0,
    pm_dec_arcsec_yr: float = 0.0,
) -> Stack:
    """Co-add every stamp along the trajectory of a body at ``d_au``.

    Each stamp dict also carries ``rows`` (its 2-D pixels, NaN for masked) and
    may carry ``sigma``; if it does not, the frame's scatter is measured from
    its own pixels. Output is the same size as the reference stamp. A pixel
    with no contributions comes back NaN rather than zero -- zero is a
    brightness and would be read as one.
    """
    if not stamps:
        return Stack([], [], 0, d_au)
    shifts = trajectory_shifts(
        stamps, ra_deg, dec_deg, d_au, ref=ref,
        pm_ra_arcsec_yr=pm_ra_arcsec_yr, pm_dec_arcsec_yr=pm_dec_arcsec_yr,
    )
    by_epoch = {s["i"]: s for s in stamps}
    weights = frame_weights(stamps)
    rows0 = stamps[ref]["rows"]
    ny = len(rows0)
    nx = len(rows0[0]) if ny else 0

    total = [[0.0] * nx for _ in range(ny)]
    wsum = [[0.0] * nx for _ in range(ny)]
    counts = [[0] * nx for _ in range(ny)]
    used = 0
    for epoch, dx, dy in shifts:
        s = by_epoch.get(epoch)
        if s is None:
            continue
        w = weights.get(epoch, 1.0)
        src = s["rows"]
        sny = len(src)
        snx = len(src[0]) if sny else 0
        landed = False
        # Only the rows and columns the shifted frame covers.
        for j in range(max(0, dy), min(ny, sny + dy)):
            srow = src[j - dy]
            trow = total[j]
            wrow = wsum[j]
            crow = counts[j]
            for i in range(max(0, dx), min(nx, snx + dx)):
                v = srow[i - dx]
                if v != v:
                    continue
                trow[i] += w * v
                wrow[i] += w
                crow[i] += 1
                landed = True
        if landed:
            used += 1

    # Divided by the weight that landed, not by the count: the point of the
    # weighting is that a noisy frame contributes less signal *and* less
    # denominator, so dividing by the count would scale the result by however
    # many bad frames happened to overlap this pixel.
    out = [[(total[j][i] / wsum[j][i]) if wsum[j][i] > 0 else float("nan")
            for i in range(nx)] for j in range(ny)]
    return Stack(out, counts, used, d_au, shifts, weights)
