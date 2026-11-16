"""Turning MJy/sr into pixels a human can read, without lying about the data.

Display stretch is the step where astronomical images are most often quietly
falsified: a percentile clip chosen to make a picture look good will erase a
faint moving source, or invent one out of read noise. Everything here is
therefore (a) an algorithm with a citable definition, (b) reversible -- the
mapping from display level back to surface brightness is exposed so the UI can
always answer "what is that pixel actually worth", and (c) computed from the
data rather than from a taste parameter.

Three pieces:

``zscale``
    The IRAF/DS9 algorithm (Tody 1986). Fits a line to the *sorted* sample
    values with iterative sigma rejection, then extrapolates that line's slope
    out to the ends of the sample. Because the fit is to the sorted
    distribution, it locks onto the sky background -- the long flat run in the
    middle -- and ignores both the cosmic rays at the top and the masked pixels
    at the bottom. On a background-limited SPHEREx frame (zodi/sigma ~ 12) this
    is exactly the right behaviour.

``normalize``
    Maps [z1, z2] into [0, 1] through a named transfer function. ``asinh`` is
    the default because it is linear in the noise and logarithmic in the
    sources (Lupton et al. 2004), so a faint tracklet and a bright star can
    share one frame without either clipping.

``Histogram``
    Real counts in real units, for the readout panel. The UI draws it; nothing
    about the stretch depends on it.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

Array2D = Sequence[Sequence[float]]

#: IRAF ``zscale`` defaults, unchanged. ``contrast=0.25`` is what DS9 ships.
ZSCALE_NSAMPLES = 1000
ZSCALE_CONTRAST = 0.25
ZSCALE_KREJ = 2.5
ZSCALE_MAX_ITER = 5
#: Below this many usable samples the line fit is meaningless and we fall back
#: to min/max. IRAF uses 5; the fit needs at least 3 points to have a residual.
ZSCALE_MIN_NPIXELS = 5


def finite_values(rows: Array2D) -> list[float]:
    """Flatten, dropping NaN and inf.

    SPHEREx L2 frames carry NaN in masked pixels -- detector edges, cosmic-ray
    hits, and the unilluminated corners. Left in, a single NaN poisons every
    comparison in the sort and silently wrecks the stretch.
    """
    return [v for row in rows for v in row if v == v and -math.inf < v < math.inf]


def _sample(rows: Array2D, nsamples: int) -> list[float]:
    """Subsample on a regular grid, the way IRAF does, then sort.

    A grid stride rather than a random draw, so the result is deterministic:
    two viewers looking at the same cutout must get the same stretch, or the
    blink comparison they are about to make is not a comparison.
    """
    ny = len(rows)
    if ny == 0:
        return []
    nx = len(rows[0])
    total = nx * ny
    stride = max(1, int(math.sqrt(total / max(1, nsamples))))
    out = [
        rows[y][x]
        for y in range(0, ny, stride)
        for x in range(0, nx, stride)
        if rows[y][x] == rows[y][x] and -math.inf < rows[y][x] < math.inf
    ]
    out.sort()
    return out


@dataclass(frozen=True)
class ZScale:
    """The result of a zscale solve, with the diagnostics that explain it."""

    z1: float
    z2: float
    median: float
    slope: float
    npoints: int
    nrejected: int
    iterations: int

    @property
    def span(self) -> float:
        return self.z2 - self.z1

    @property
    def degenerate(self) -> bool:
        """True when the solve collapsed and min/max was used instead."""
        return self.slope == 0.0


def zscale(
    rows: Array2D,
    *,
    nsamples: int = ZSCALE_NSAMPLES,
    contrast: float = ZSCALE_CONTRAST,
    krej: float = ZSCALE_KREJ,
    max_iterations: int = ZSCALE_MAX_ITER,
) -> ZScale:
    """IRAF ``zscale`` limits for a 2-D array.

    The fit is ``value[i] = median + slope * (i - midpoint)`` over the sorted
    samples, with points beyond ``krej`` sigma rejected and the fit repeated.
    Limits then come from extrapolating that slope, divided by ``contrast``:

        z1 = median - (midpoint - 1) * slope / contrast
        z2 = median + (npoints - midpoint) * slope / contrast

    clamped to the sample's own range -- extrapolation past the real data would
    show the viewer contrast that does not exist.
    """
    samples = _sample(rows, nsamples)
    n = len(samples)
    if n == 0:
        return ZScale(0.0, 1.0, 0.0, 0.0, 0, 0, 0)
    vmin, vmax = samples[0], samples[-1]
    median = samples[n // 2] if n % 2 else 0.5 * (samples[n // 2 - 1] + samples[n // 2])
    if n < ZSCALE_MIN_NPIXELS:
        return ZScale(vmin, vmax, median, 0.0, n, 0, 0)

    # Iterative sigma-clipped straight-line fit in (index, sorted value).
    # Index is centred so the intercept is directly the value at midpoint.
    xs = [float(i) - 0.5 * (n - 1) for i in range(n)]
    keep = [True] * n
    slope = 0.0
    intercept = median
    iterations = 0
    ngood = n

    for iterations in range(1, max_iterations + 1):
        sx = sy = sxx = sxy = 0.0
        ngood = 0
        for i in range(n):
            if not keep[i]:
                continue
            x, y = xs[i], samples[i]
            sx += x
            sy += y
            sxx += x * x
            sxy += x * y
            ngood += 1
        if ngood < 3:
            break
        denom = ngood * sxx - sx * sx
        if denom == 0.0:
            break
        slope = (ngood * sxy - sx * sy) / denom
        intercept = (sy - slope * sx) / ngood

        resid = [samples[i] - (intercept + slope * xs[i]) for i in range(n)]
        ss = math.fsum(resid[i] * resid[i] for i in range(n) if keep[i])
        sigma = math.sqrt(ss / ngood)
        if sigma == 0.0:
            break
        limit = krej * sigma
        newkeep = [keep[i] and abs(resid[i]) <= limit for i in range(n)]
        if newkeep == keep:
            break
        keep = newkeep

    nrejected = n - ngood
    if slope == 0.0:
        return ZScale(vmin, vmax, median, 0.0, n, nrejected, iterations)

    # IRAF applies contrast to the *slope*, i.e. it widens the displayed range
    # when contrast < 1. Guard against a zero contrast turning that into inf.
    c = contrast if contrast > 0.0 else 1.0
    midpoint = (n + 1) // 2
    z1 = max(vmin, intercept - (midpoint - 1) * slope / c)
    z2 = min(vmax, intercept + (n - midpoint) * slope / c)
    if z2 <= z1:  # pathological sample; fall back to the honest full range
        z1, z2 = vmin, vmax
    return ZScale(z1, z2, median, slope, n, nrejected, iterations)


def percentile_limits(rows: Array2D, lo: float = 0.5, hi: float = 99.5) -> tuple[float, float]:
    """Plain percentile clip, offered for comparison against :func:`zscale`.

    Linear interpolation between order statistics, so the result is continuous
    in the percentile rather than jumping between sample values.
    """
    vals = finite_values(rows)
    if not vals:
        return 0.0, 1.0
    vals.sort()
    n = len(vals)

    def at(p: float) -> float:
        if n == 1:
            return vals[0]
        k = max(0.0, min(1.0, p / 100.0)) * (n - 1)
        i = int(k)
        frac = k - i
        return vals[i] if i >= n - 1 else vals[i] * (1 - frac) + vals[i + 1] * frac

    a, b = at(lo), at(hi)
    return (a, b) if b > a else (vals[0], vals[-1] if vals[-1] > vals[0] else vals[0] + 1.0)


# --------------------------------------------------------------------------- #
# Transfer functions
# --------------------------------------------------------------------------- #

#: ``asinh`` softening. Small Q is nearly linear; large Q approaches log.
#: 8 keeps SPHEREx sky noise visibly textured while a 10th-magnitude star still
#: has structure rather than a white disc.
ASINH_Q_DEFAULT = 8.0


def _clamp01(t: float) -> float:
    return 0.0 if t < 0.0 else min(t, 1.0)


def transfer(
    name: str, *, q: float = ASINH_Q_DEFAULT, gamma: float = 2.2
) -> Callable[[float], float]:
    """A named [0,1] -> [0,1] transfer function.

    ``asinh`` is normalised so ``f(0)=0`` and ``f(1)=1`` exactly, which matters:
    an unnormalised ``asinh`` quietly rescales the white point and makes two
    stretches of the same data non-comparable.
    """
    if name == "linear":
        return _clamp01
    if name == "asinh":
        qq = max(1e-6, q)
        denom = math.asinh(qq)
        return lambda t: _clamp01(math.asinh(_clamp01(t) * qq) / denom)
    if name == "sinh":
        qq = max(1e-6, q)
        denom = math.sinh(qq)
        return lambda t: _clamp01(math.sinh(_clamp01(t) * qq) / denom)
    if name == "sqrt":
        return lambda t: math.sqrt(_clamp01(t))
    if name == "squared":
        return lambda t: _clamp01(t) ** 2
    if name == "log":
        # a = 1000 is the DS9 'log' default
        a = 1000.0
        return lambda t: _clamp01(math.log10(a * _clamp01(t) + 1.0) / math.log10(a + 1.0))
    if name == "power":
        g = max(1e-6, gamma)
        return lambda t: _clamp01(t) ** (1.0 / g)
    raise ValueError(f"unknown transfer function {name!r}")


TRANSFERS = ("linear", "asinh", "sqrt", "squared", "log", "sinh", "power")


def normalize(
    rows: Array2D,
    z1: float,
    z2: float,
    *,
    mode: str = "asinh",
    q: float = ASINH_Q_DEFAULT,
) -> list[list[float]]:
    """Map an array through ``[z1, z2]`` and a transfer function into ``[0, 1]``.

    NaN is preserved as NaN rather than becoming 0, so the renderer can paint
    masked pixels distinctly instead of as legitimate dark sky. Confusing a
    masked column for real dark sky is how a chip gap becomes a discovery.
    """
    f = transfer(mode, q=q)
    span = z2 - z1
    if span == 0.0:
        span = 1.0
    out: list[list[float]] = []
    for row in rows:
        acc: list[float] = []
        for v in row:
            if v != v:
                acc.append(float("nan"))
            else:
                acc.append(f((v - z1) / span))
        out.append(acc)
    return out


def invert_level(
    level: float, z1: float, z2: float, *, mode: str = "asinh", q: float = ASINH_Q_DEFAULT
) -> float:
    """Display level in ``[0, 1]`` back to data units.

    The readout panel needs this to answer "what surface brightness is that
    grey?" without re-reading the pixel. Analytic where possible; bisection
    otherwise, which is fine at 40 iterations for a scalar.
    """
    span = (z2 - z1) or 1.0
    t = _clamp01(level)
    if mode == "linear":
        return z1 + t * span
    if mode == "asinh":
        qq = max(1e-6, q)
        return z1 + span * math.sinh(t * math.asinh(qq)) / qq
    if mode == "sqrt":
        return z1 + span * t * t
    if mode == "squared":
        return z1 + span * math.sqrt(t)
    f = transfer(mode, q=q)
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if f(mid) < t:
            lo = mid
        else:
            hi = mid
    return z1 + span * 0.5 * (lo + hi)


# --------------------------------------------------------------------------- #
# Histogram
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Histogram:
    edges: tuple[float, ...]
    counts: tuple[int, ...]
    n_finite: int
    n_masked: int

    @property
    def peak(self) -> int:
        return max(self.counts, default=0)


def histogram(
    rows: Array2D, *, bins: int = 128, limits: tuple[float, float] | None = None
) -> Histogram:
    """Counts per bin over ``limits`` (default: the finite data range)."""
    vals = finite_values(rows)
    total = sum(len(r) for r in rows)
    if not vals:
        return Histogram((), (), 0, total)
    lo, hi = limits if limits is not None else (min(vals), max(vals))
    if hi <= lo:
        hi = lo + 1.0
    width = (hi - lo) / bins
    counts = [0] * bins
    for v in vals:
        k = int((v - lo) / width)
        counts[0 if k < 0 else (bins - 1 if k >= bins else k)] += 1
    edges = tuple(lo + i * width for i in range(bins + 1))
    return Histogram(edges, tuple(counts), len(vals), total - len(vals))


#: Scale from median absolute deviation to a Gaussian sigma, ``1/Phi^-1(0.75)``.
MAD_TO_SIGMA = 1.4826


def median_sorted(sorted_vals: Sequence[float]) -> float:
    """Median of an already-sorted sequence; 0.0 for an empty one."""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    mid = n // 2
    return sorted_vals[mid] if n % 2 else 0.5 * (sorted_vals[mid - 1] + sorted_vals[mid])


def kth_deviation(sorted_vals: Sequence[float], med: float, r: int) -> float:
    """The ``r``-th smallest (from 0) of ``abs(v - med)`` over a sorted sequence.

    Exactly ``sorted(abs(v - med) for v in sorted_vals)[r]``, without the sort.
    Measured from any value, a sorted sequence's deviations are two runs that
    are already in order -- falling towards ``med`` from the left, rising away
    from it on the right -- so the ``r``-th of them is where a bisection over
    how many to take from each run puts it: logarithmic in the length, where
    computing and sorting every deviation was ``n log n``, and the same float.
    """
    k = bisect.bisect_left(sorted_vals, med)
    m = len(sorted_vals) - k

    def left(t: int) -> float:   # t-th smallest deviation below med
        return abs(sorted_vals[k - 1 - t] - med)

    def right(t: int) -> float:  # t-th smallest at or above it
        return abs(sorted_vals[k + t] - med)

    # The `r + 1` smallest are the `a` smallest of the left run and the rest
    # from the right; bisect for that `a`.
    c = r + 1
    lo, hi = max(0, c - m), min(c, k)
    while lo < hi:
        a = (lo + hi) // 2
        if left(a) < right(c - a - 1):
            lo = a + 1
        else:
            hi = a
    return max(left(lo - 1) if lo else -math.inf, right(c - lo - 1) if c - lo else -math.inf)


def mad_sigma(sorted_vals: Sequence[float], med: float) -> float:
    """``MAD_TO_SIGMA`` times the median of ``abs(v - med)``, over a sorted sequence.

    Exactly ``median_sorted(sorted(abs(v - med) for v in sorted_vals))``, by way
    of :func:`kth_deviation`; 0.0 for an empty sequence.
    """
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    h = n // 2
    if n % 2:
        return MAD_TO_SIGMA * kth_deviation(sorted_vals, med, h)
    return MAD_TO_SIGMA * (0.5 * (kth_deviation(sorted_vals, med, h - 1)
                                  + kth_deviation(sorted_vals, med, h)))


def robust_sigma(values: Iterable[float]) -> float:
    """Noise from the median absolute deviation, scaled to a Gaussian sigma.

    1.4826 * MAD. Used instead of the sample standard deviation because a
    cutout containing an actual source -- which is the interesting case -- has
    a standard deviation set by the source, not by the noise we want to measure.
    """
    vals = sorted(v for v in values if v == v and -math.inf < v < math.inf)
    return mad_sigma(vals, median_sorted(vals))


def flatten_rows(rows: list[list[float]]) -> tuple[list[list[float]], float]:
    """Subtract each row's own median. Returns the rows and the ramp amplitude.

    Some cutouts land within about 300 pixels of the bottom edge of the
    detector, and those frames carry a strong brightness ramp along the
    detector's y axis. On the North Ecliptic Pole's first visit the background
    falls almost eightfold across the 193 pixels of the stamp -- a swing larger
    than the frame's own median level -- and the South Ecliptic Pole's first
    visit is worse, at nearly thirteen.

    The ramp is **very nearly** one-dimensional, which is what makes a row
    median an adequate description of it. Measured on that NEP frame, read back
    off S3: the row medians span 0.169 to 1.339, a ratio of 7.94, while the
    column medians span 1.013 to 1.153, a ratio of 1.14. Take each row's median
    out and the column medians that remain span -0.013 to +0.018 -- 0.031
    against the ramp's own 1.170, or 2.6% of it.

    This note used to say the gradient across x was *exactly* zero and every row
    uniform left to right, and neither is true: there is real sky in every row,
    and a couple of percent of the ramp does run across the frame. Two and a
    half percent of a correction is not worth a second axis, but it is not zero
    and the docstring should not promise that. Removing the row median leaves
    the sources on top of it alone either way -- a point source covers a few
    pixels of a 193-pixel row and cannot move its median.

    This is not cosmetic. ``sigma_mad`` below is taken over the whole stamp, so
    an uncorrected ramp is counted as noise: on that first NEP visit it came
    out twenty times the true per-pixel scatter, which blew the display window
    out by the same factor. ``detect`` builds its own background surface and so
    survives better, but not unharmed -- its sigma fell 1.67x once the ramp was
    gone (0.05645 to 0.03388), returning 35 sources at the 6.5 sigma the search
    ships with, 222 to 257, and 33 at 5 sigma. The frame is not shallow -- its
    per-pixel noise is about twice the others' -- it was being measured wrong.

    Applied to every epoch rather than to the ones that need it, because a
    single corrected frame beside eleven uncorrected ones is its own artefact.
    On a frame that is already flat the row medians are the sky level and the
    correction is a constant offset, which no downstream statistic can see.
    """
    out: list[list[float]] = []
    levels: list[float] = []
    for row in rows:
        finite = sorted(v for v in row if v == v and -math.inf < v < math.inf)
        if not finite:
            out.append(list(row))
            continue
        n = len(finite)
        med = finite[n // 2] if n % 2 else 0.5 * (finite[n // 2 - 1] + finite[n // 2])
        levels.append(med)
        out.append([v - med for v in row])
    amp = (max(levels) / min(levels)) if levels and min(levels) > 0 else float("nan")
    return out, amp


#: Tolerance when testing whether a resampled sample falls inside the source
#: band. Guards against round-off in the projection -- trigonometric round-off
#: at right angles -- and is far below the sub-pixel scale anything downstream
#: cares about.
_EDGE_EPS = 1e-9


def reproject(
    rows: Array2D,
    src_to_pixel: Callable[[float, float], tuple[float, float]],
    dst_to_world: Callable[[float, float], tuple[float, float]],
    *,
    size: int,
    col_origin: int,
    row_origin: int,
) -> tuple[list[list[float]], float, float]:
    """Resample a detector row band onto another WCS's grid.

    SPHEREx rolls between visits, so a cutout taken as a detector-aligned pixel
    box arrives in its own frame's orientation: across the twelve North
    Ecliptic Pole visits the position angle spans nearly the full circle, with
    31 to 81 degrees between consecutive ones. Blinked, the field pinwheels
    about its centre -- only the central pixel holds still, while a star at the
    corner swings up to 174 pixels between one visit and the next. That is
    camera roll, not motion on the sky.

    This maps every output pixel through the two projections rather than
    rotating the array, which matters for two reasons. The source frame's
    third-order SIP distortion is evaluated per pixel and so comes out
    *removed*, not merely turned; and the local sky orientation varies across a
    2040-pixel detector, so a single rotation angle is wrong away from the
    point it was measured at.

    ``rows`` is the band as read, ``row_origin``/``col_origin`` its lower-left
    corner in 1-based FITS pixels. Output pixels whose source falls outside the
    band become NaN, the sentinel the rest of the pipeline already treats as
    masked. Because a row band is read at the full detector width, only the row
    extent can run short, so the masked region is two slivers at opposite edges
    rather than four corners.

    Returns the resampled rows, the fraction that came out masked, and the
    mean bilinear variance-shrink factor ``mean(sum w^2)`` over the pixels that
    were filled. That last number is the honesty tax on resampling: an output
    pixel is a weighted mean of up to four inputs, so its variance is the
    input's times ``sum w^2 <= 1`` -- the shipped stamp's per-pixel scatter
    *understates* the underlying noise by ``sqrt`` of this factor, and
    neighbouring outputs share inputs, so a matched filter that assumes
    independent pixels overstates its significances by about the inverse.
    Measured on white noise pushed through this exact resampler, the SNR-map
    scatter comes out 1.28 where it should be 1.00. The factor is recorded so
    the detector can divide it back out; see ``search.detect``.
    """
    nrows = len(rows)
    if not nrows:
        return [], 0.0, 1.0
    ncols = len(rows[0])

    out: list[list[float]] = []
    masked = 0
    wsq_sum = 0.0
    filled = 0
    for j in range(size):
        acc: list[float] = []
        for i in range(size):
            ra, dec = dst_to_world(i + 1.0, j + 1.0)     # 1-based, as FITS counts
            try:
                sx, sy = src_to_pixel(ra, dec)
            except Exception:  # noqa: BLE001 -- off this frame's projection
                acc.append(float("nan"))
                masked += 1
                continue
            bx, by = sx - col_origin, sy - row_origin
            if bx < -_EDGE_EPS or by < -_EDGE_EPS:
                acc.append(float("nan"))
                masked += 1
                continue
            if bx > (ncols - 1) + _EDGE_EPS or by > (nrows - 1) + _EDGE_EPS:
                acc.append(float("nan"))
                masked += 1
                continue
            bx = min(max(bx, 0.0), float(ncols - 1))
            by = min(max(by, 0.0), float(nrows - 1))
            x0 = min(int(bx), ncols - 2)
            y0 = min(int(by), nrows - 2)
            tx, ty = bx - x0, by - y0
            p00, p01 = rows[y0][x0], rows[y0][x0 + 1]
            p10, p11 = rows[y0 + 1][x0], rows[y0 + 1][x0 + 1]
            if p00 != p00 or p01 != p01 or p10 != p10 or p11 != p11:
                acc.append(float("nan"))
                masked += 1
                continue
            top = p00 * (1.0 - tx) + p01 * tx
            bot = p10 * (1.0 - tx) + p11 * tx
            acc.append(top * (1.0 - ty) + bot * ty)
            # Separable bilinear weights, so sum(w^2) factorises exactly.
            wsq_sum += ((1.0 - tx) ** 2 + tx ** 2) * ((1.0 - ty) ** 2 + ty ** 2)
            filled += 1
        out.append(acc)
    shrink = wsq_sum / filled if filled else 1.0
    return out, masked / float(size * size), shrink


