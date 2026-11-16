"""Find point sources in one frame.

This is the half of the problem the console could not do. Everything else in
the tool assumes you have already *spotted* the thing that moved -- you blink,
you see it, you click it. The mission returns images of more than a billion
objects and nobody blinks their way through that.

The method is a matched filter, which is not a stylistic choice: for a point
source of known shape in Gaussian noise it is the optimal linear detector, and
"known shape" is exactly the case here because every SPHEREx frame ships its own
``EPSF``. A convolutional net would be the fashionable answer and it would be
the wrong one -- there is no labelled training set for *this* survey at *this*
depth, the objects of interest are by construction the ones no catalogue
contains, and a network's false-positive rate is not something you can state on
a plot. The whole project is built to report an exclusion limit, and you cannot
put a limit on a detector whose completeness you cannot write down. A matched
filter's completeness is an error function of the threshold.

Three details that are easy to get wrong and that the tests pin:

* **The background is a clipped median, not a mean.** A mean over a field with
  sources in it is pulled up by the sources, so the noise you then measure is
  inflated by the very thing you are trying to detect, and the threshold walks
  away from you.
* **NaN is a first-class case.** SPHEREx L2 frames carry NaN in masked pixels.
  A NaN that reaches a comparison does not raise -- it returns ``False`` and
  the pixel quietly drops below threshold; a NaN that reaches a sum poisons the
  whole footprint. Every accumulation here skips them explicitly and the
  normalisation counts only the pixels actually used.
* **The filter's noise is not the image's noise.** Convolving with a kernel
  ``k`` scales the signal by ``sum(k)`` and the noise by ``sqrt(sum(k^2))``.
  Thresholding the filtered image against the *unfiltered* sigma is the
  standard way to end up with a detector that is far more or far less sensitive
  than its stated number of sigma.
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from dataclasses import dataclass

from tracklet.frames.stretch import mad_sigma, median_sorted

Array2D = list[list[float]]

#: Gaussian sigma of the detection kernel, in pixels.
#:
#: SPHEREx's PSF is near-critically sampled: ``docs/FEASIBILITY.md`` uses an
#: effective aperture of ``n_eff = 4`` pixels, and for a Gaussian
#: ``n_eff = 4*pi*sigma^2``, which gives ``sigma = 0.56`` px. Rounded up a
#: little, because a filter slightly wider than the source loses a few percent
#: of signal-to-noise while a filter narrower than the source loses a lot.
PSF_SIGMA_PX = 0.6


@dataclass(frozen=True)
class Source:
    """One detection, in array coordinates of the stamp it was found in."""

    #: Flux-weighted centroid, fractional array indices (column, row).
    x: float
    y: float
    #: Highest matched-filter significance in the footprint, in sigma.
    snr: float
    #: Summed excess over the footprint, in the image's own units.
    flux: float
    #: Brightest single pixel's excess over the background.
    peak: float
    #: Pixels above threshold that are connected to this one.
    npix: int
    #: Fraction of the footprint's flux in its brightest pixel. A source that
    #: matches the PSF sits near 0.45; a cosmic ray or a hot pixel approaches
    #: 1.0, because it is sharper than the optics can produce.
    sharpness: float


@dataclass(frozen=True)
class Detection:
    """What one frame gave up, and the numbers the threshold was set from."""

    #: Median of the background *surface*. A single representative level, for
    #: display; the detection itself subtracts the surface pixel by pixel.
    background: float
    sigma: float
    threshold_sigma: float
    #: Finite pixels the background was measured from.
    npix_used: int
    #: Pixels rejected by the sigma clip -- sources, cosmic rays, edges.
    npix_clipped: int
    sources: list[Source]


def estimate_background(values: list[float], *, clip: float = 3.0, iters: int = 5):
    """Sigma-clipped median and MAD sigma.

    Returns ``(median, sigma, n_used, n_clipped)``. Iterates because one pass of
    clipping against a sigma that the sources themselves inflated still leaves
    the sources in; two or three passes converge on real sky.
    """
    keep = sorted(v for v in values if v == v and -math.inf < v < math.inf)
    if not keep:
        return 0.0, 0.0, 0, 0
    total = len(keep)
    med = sigma = 0.0
    for _ in range(iters):
        if not keep:
            break
        med = median_sorted(keep)
        sigma = mad_sigma(keep, med)
        if sigma <= 0.0:
            break
        # `keep` is sorted, so what survives the clip is one slice of it.
        a = bisect_left(keep, med - clip * sigma)
        b = bisect_right(keep, med + clip * sigma)
        if b - a == len(keep):
            break
        keep = keep[a:b]
    return med, sigma, len(keep), total - len(keep)


#: Side of the background mesh, in pixels.
#:
#: Must be large compared to a source and small compared to the gradient. A
#: SPHEREx stamp is dominated by zodiacal light, which ramps across the frame by
#: several times the pixel noise, so a single number for "the background" is
#: wrong in both directions at once: it invents sources in the bright half and
#: it deletes them from the faint half. 32 pixels is ~1000 samples per cell,
#: plenty for a clipped median, and fine enough to follow the ramp.
BACKGROUND_BOX_PX = 32


def background_map(rows: Array2D, *, box: int = BACKGROUND_BOX_PX):
    """A background *surface*, not a level, plus the noise about it.

    Clipped median per mesh cell, bilinearly interpolated back to full
    resolution. Returns ``(bg, sigma, n_used, n_clipped, level)`` where ``bg``
    is a full-size array and ``level`` is its median -- a single representative
    number, for display only.

    Why this matters more than it looks: the sharpness cut compares a source's
    peak to the flux in an aperture round it. Subtract a background that is too
    high and that aperture flux goes *negative*, the ratio is undefined, and a
    perfectly good source is discarded as a cosmic ray -- which is how a
    12-sigma planted source was being thrown away.

    Re-measured on the shipped NEP stamps, because this used to claim "0.44 in
    one corner, more than a sigma" and that matches none of them. Comparing this
    function's own ``level`` against its ``bg`` surface four pixels in from each
    corner, in units of the residual ``sigma`` it returns:

        epoch 0        15.9 sigma      a real gradient across the frame
        epochs 1-11    0.08-2.5 sigma  flat to within a few sigma

    So the effect is far larger than was claimed on the one frame that has a
    gradient, and smaller on the rest; quoting one number for all twelve was the
    mistake. Stated as a ratio deliberately -- the stamps ship quantised to their
    own display range, so an absolute flux figure here would describe the PNG
    encoding rather than the sky, and would not reproduce for anyone who
    dequantised against the physical range instead.
    """
    ny = len(rows)
    nx = len(rows[0]) if ny else 0
    if nx == 0 or ny == 0:
        return [], 0.0, 0, 0, 0.0
    gx = max(1, -(-nx // box))
    gy = max(1, -(-ny // box))

    # Cell medians. A cell that is entirely masked inherits the global level
    # rather than a zero, which would carve a hole in the surface.
    level, _, _, _ = estimate_background([v for row in rows for v in row])
    mesh = [[level] * gx for _ in range(gy)]
    for cy in range(gy):
        band = rows[cy * box:(cy + 1) * box]
        for cx in range(gx):
            vals = [v for row in band for v in row[cx * box:(cx + 1) * box] if v == v]
            if vals:
                mesh[cy][cx] = estimate_background(vals)[0]

    # Bilinear between cell centres, and **extrapolated** rather than clamped in
    # the half-cell border. Holding the surface flat out to the edge leaves a
    # pedestal of half a cell's worth of gradient under everything there, which
    # is enough to drive a real source's aperture flux negative and have the
    # sharpness cut delete it. The border is where a moving object spends most
    # of its time.
    #
    # Measured on the 128-pixel, 58-sigma ramp frame that
    # `test_a_source_survives_a_strong_background_gradient` builds, comparing
    # the surface against the known ramp over the 16-pixel border:
    #
    #     extrapolated (this code)    worst error 0.11 sigma
    #     clamped at the edge         worst error 7.03 sigma
    #
    # This comment used to put the clamped cost at 1.6 sigma, which undersold it
    # by a factor of four -- half a cell is 16 px and the ramp is 0.0071 per px,
    # so the pedestal is 0.11 in flux against a 0.0156 sigma.
    #
    # Interpolated along x once per mesh row, then between mesh rows per image
    # row: the same two products and sums per pixel as interpolating each pixel
    # from its four cells, so the same surface to the last bit.
    def cell(f: float, g: int) -> tuple[int, int, float]:
        c = 0 if g < 2 else min(g - 2, max(0, math.floor(f)))
        return c, min(g - 1, c + 1), 0.0 if g < 2 else f - c

    cols = [cell((i - 0.5 * box + 0.5) / box, gx) for i in range(nx)]
    across = [[m[ix] * (1 - tx) + m[ix1] * tx for ix, ix1, tx in cols] for m in mesh]
    bg = []
    for j in range(ny):
        jy, jy1, ty = cell((j - 0.5 * box + 0.5) / box, gy)
        bg.append([a * (1 - ty) + b * ty for a, b in zip(across[jy], across[jy1])])

    resid = [v - b for row, brow in zip(rows, bg) for v, b in zip(row, brow) if v == v]
    _, sigma, n_used, n_clipped = estimate_background(resid)
    return bg, sigma, n_used, n_clipped, level


def gaussian_kernel(sigma: float = PSF_SIGMA_PX) -> tuple[list[list[float]], int]:
    """A normalised 2-D Gaussian and its radius, for the matched filter.

    The radius rounds **half away from zero, written out**, because the two
    languages disagree otherwise. This was ``int(round(2.5 * sigma))`` against
    JavaScript's ``Math.round(2.5 * sigma)``, and Python's ``round`` is
    round-half-to-*even*: at ``sigma = 1.0`` the argument is exactly 2.5, so
    Python returned r = 2 where JavaScript returned 3 -- different kernels, and
    a matched-filter gain of 1.772390 against 1.772637.

    It diverged wherever ``2.5 * sigma`` fell on an odd half-integer, so
    sigma = 1.0, 1.8, 2.6 and upward. The shipped ``PSF_SIGMA_PX`` of 0.6 gives
    1.5, where both round to 2 -- the default was safe by luck rather than by
    design, and anyone who changed it would have had a browser detecting
    differently from the bake with nothing to say so. ``floor(x + 0.5)`` means
    the same thing in both languages and inherits neither's convention.
    """
    r = max(1, math.floor(2.5 * sigma + 0.5))
    k = [[math.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma)) for dx in range(-r, r + 1)]
         for dy in range(-r, r + 1)]
    s = sum(sum(row) for row in k)
    return [[v / s for v in row] for row in k], r


def resampled_filter_norm(psf_sigma: float = PSF_SIGMA_PX) -> float:
    """How much bilinear resampling widens the matched filter's noise.

    Resampling makes each output pixel a weighted mean of up to four inputs,
    which correlates neighbours; a matched filter that assumes independent
    pixels then understates its own noise. For bilinear weights with uniform
    sub-pixel phase and a near-unity scale ratio -- exactly what
    ``stretch.reproject`` produces -- the output correlations are
    ``rho(+-1,0) = rho(0,+-1) = 1/4`` and ``rho(+-1,+-1) = 1/16``
    (``E[t(1-t)] = 1/6`` against ``E[(1-t)^2 + t^2] = 2/3`` per axis), so the
    widening is ``sqrt(k'Ck / k'k)`` with that covariance. For the shipped
    0.6-pixel PSF kernel this evaluates to 1.22; white noise pushed through
    the real resampler at a 37-degree rotation measures 1.28, the difference
    being rotation mixing the axes. Either is a small honest correction where
    ignoring it flattered every significance by 28 to 50 percent.

    Not ``1/sqrt(shrink)``, deliberately: that per-pixel factor (1.50 here)
    overshoots, because the smoothing that correlates the noise also removes
    the high-frequency part the filter would otherwise integrate.
    """
    kernel, r = gaussian_kernel(psf_sigma)
    n = 2 * r + 1

    def at(j: int, i: int) -> float:
        return kernel[j][i] if 0 <= j < n and 0 <= i < n else 0.0

    rho = {(0, 0): 1.0}
    for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        rho[d] = 0.25
    for d in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        rho[d] = 1.0 / 16.0
    q = 0.0
    ktck = 0.0
    for j in range(n):
        for i in range(n):
            q += kernel[j][i] ** 2
            for (dj, di), p in rho.items():
                ktck += kernel[j][i] * at(j + dj, i + di) * p
    return math.sqrt(ktck / q) if q > 0.0 else 1.0


def matched_filter_gain(psf_sigma: float = PSF_SIGMA_PX) -> float:
    """Peak significance produced by a unit-amplitude point source.

    Invert it for the amplitude to plant at a stated signal-to-noise:
    ``A = target * sigma / matched_filter_gain(psf_sigma)``. Computed from the
    same kernel :func:`detect` uses rather than from the closed form, so an
    injection cannot claim a signal-to-noise the detector would not agree with
    -- which is the one way an injection-recovery test can quietly lie.
    """
    kernel, r = gaussian_kernel(psf_sigma)
    acc = q = 0.0
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            w = kernel[dy + r][dx + r]
            acc += w * math.exp(-(dx * dx + dy * dy) / (2.0 * psf_sigma * psf_sigma))
            q += w * w
    return acc / math.sqrt(q)


def psf_flux_integral(psf_sigma: float = PSF_SIGMA_PX) -> float:
    """Total flux of a unit-peak Gaussian PSF, in pixels.

    The factor that turns a **peak** pixel value into a flux: a source whose
    brightest pixel sits ``A`` above the background carries ``A *
    psf_flux_integral()`` in all of them together. ``2*pi*sigma**2`` in the
    continuum (2.2619 px at 0.6), and this is its discrete sum over the same
    radius an injection lays light down over, so planting a flux and measuring
    it back agree to the last digit rather than to 0.3%.

    Not ``n_eff``. The two are easy to confuse and they answer different
    questions: ``n_eff = (sum g)^2 / sum g^2 = 4.07`` px is the *noise*
    equivalent area, the pixel count whose summed noise matches the filter's,
    and it is what ``scripts/feasibility_gate.py`` multiplies a per-pixel sigma
    by. Using it on a peak value instead -- as this tool's cursor readout did --
    reports a point source 0.61 mag too bright, because the peak of a PSF that
    fits inside a pixel is nothing like its mean over four of them.
    """
    r = math.ceil(4.0 * psf_sigma)
    two_s2 = 2.0 * psf_sigma * psf_sigma
    return sum(
        math.exp(-((di * di + dj * dj) / two_s2))
        for dj in range(-r, r + 1)
        for di in range(-r, r + 1)
    )


def detect(
    rows: Array2D,
    *,
    threshold_sigma: float = 5.0,
    min_pixels: int = 1,
    psf_sigma: float = PSF_SIGMA_PX,
    max_sharpness: float = 0.92,
    sharpness_min_snr: float = 10.0,
    max_sources: int = 500,
    background_box: int = BACKGROUND_BOX_PX,
    pixel_noise_shrink: float = 1.0,
) -> Detection:
    """Matched-filter detection over one 2-D array.

    ``rows[j][i]`` is row ``j``, column ``i``, in whatever units the frame uses;
    NaN means masked. The returned centroids are fractional array indices in the
    same frame, so a caller converts them to sky with the stamp's own WCS and
    nothing here needs to know about astronomy.

    ``min_pixels`` defaults to **one**, which looks wrong and is not. An area
    cut is the right tool on a raw threshold map, where a real source covers
    several pixels and a noise spike covers one. It is the wrong tool *after*
    matched filtering, because the filter has already imposed exactly that
    requirement -- correlating with the PSF is what turns "several adjacent
    pixels agree" into a single number. The filtered response one pixel off a
    PSF-shaped source is half its peak, so a 6-sigma source has precisely one
    pixel above a 5-sigma cut, and requiring two would silently discard every
    marginal detection in the frame. What actually separates a cosmic ray from
    a star is not area but *shape*, and that is ``max_sharpness``.

    ``pixel_noise_shrink`` is ``mean(sum w^2)`` of whatever resampling produced
    these pixels -- ``stretch.reproject`` measures it exactly from its own
    bilinear weights, and the bake records it per epoch. Resampling makes a
    pixel a weighted mean of its neighbours, which shrinks the measured
    per-pixel scatter by ``sqrt`` of this factor while correlating adjacent
    pixels to match; a matched filter that then assumes independence overstates
    every significance by the inverse. When the factor says
    "resampled", the sigma in the filter's denominator is widened by
    ``resampled_filter_norm``, so a threshold stated in sigma keeps meaning
    sigma of *noise* whether or not the stamp was resampled. On white noise pushed through the real resampler the
    SNR-map scatter reads 1.28 uncorrected and 1.00 with this.
    """
    ny = len(rows)
    nx = len(rows[0]) if ny else 0
    if nx == 0 or ny == 0:
        return Detection(0.0, 0.0, threshold_sigma, 0, 0, [])

    bg_map, sigma, n_used, n_clipped, bg = background_map(rows, box=background_box)
    # `pixel_noise_shrink < 1` is the resampler's own statement that these
    # pixels are correlated. The sigma entering the matched-filter denominator
    # is widened by the *filter-level* factor; see `resampled_filter_norm` for
    # why that is smaller than the per-pixel `1/sqrt(shrink)`.
    if 0.0 < pixel_noise_shrink < 1.0:
        sigma *= resampled_filter_norm(psf_sigma)
    if sigma <= 0.0:
        return Detection(bg, sigma, threshold_sigma, n_used, n_clipped, [])

    kernel, r = gaussian_kernel(psf_sigma)

    # Matched-filter significance per pixel.
    #
    # The estimate of the source amplitude is sum(k*x)/sum(k) and its noise is
    # sigma*sqrt(sum(k^2))/sum(k), so the ratio -- which is what a threshold is
    # stated in -- is just sum(k*x)/(sigma*sqrt(sum(k^2))). Writing it that way
    # means partial kernels at the frame edge and around NaN are handled by
    # accumulating only over the pixels that exist, with no special case.
    # A masked pixel's excess is NaN, as the pixel is.
    excess = [[v - b for v, b in zip(row, brow)] for row, brow in zip(rows, bg_map)]
    snr = _matched_filter(excess, kernel, r, sigma, nx, ny)

    sources = _extract(snr, excess, threshold_sigma, min_pixels, nx, ny, r)
    # The shape cut only applies where shape is measured. Sharpness is
    # peak/flux, and flux is a sum over ~25 pixels, so its own noise is 5 sigma
    # of pixel noise -- about 28% on a 9-sigma source. Applied blindly, the cut
    # then deletes faint *real* sources at random, which is exactly the
    # completeness you cannot afford to have and cannot write down. It is also
    # unnecessary: a cosmic ray is bright by construction, so the discriminator
    # is only needed where it is well determined.
    sources = [s for s in sources
               if s.snr < sharpness_min_snr or s.sharpness <= max_sharpness]
    sources.sort(key=lambda s: -s.snr)
    return Detection(bg, sigma, threshold_sigma, n_used, n_clipped, sources[:max_sources])


def forced_snr(
    rows: Array2D,
    x: float,
    y: float,
    *,
    psf_sigma: float = PSF_SIGMA_PX,
    background_box: int = BACKGROUND_BOX_PX,
    pixel_noise_shrink: float = 1.0,
) -> float:
    """Matched-filter significance at a position given in advance.

    :func:`detect` answers "what is here?", and to answer it has to decide
    where one source ends and the next begins. At a position a fit predicted,
    that decision is the wrong one to make: on a stack along a trajectory the
    body sits among what subtraction left of the stars, a detection threshold
    joins them into one blob, and the blob's centroid lands five pixels from a
    body that stands 23 sigma above the noise in its own central pixel.

    So this is forced photometry: the PSF is centred on ``(x, y)`` itself, to
    the sub-pixel, and the same background, noise and resampling correction as
    :func:`detect` turn the fit into a significance. At a whole-pixel position
    it is exactly the value :func:`detect` thresholds there. Centring on the
    nearest pixel instead made the answer depend on rounding: two read-outs a
    quarter of a pixel apart, either side of a boundary, gave 16.8 and 20.6
    sigma for one body. NaN where the pixel under ``(x, y)`` is masked or off
    the array.
    """
    ny = len(rows)
    nx = len(rows[0]) if ny else 0
    i0, j0 = int(round(x)), int(round(y))
    if not (0 <= i0 < nx and 0 <= j0 < ny) or rows[j0][i0] != rows[j0][i0]:
        return float("nan")
    bg_map, sigma, _, _, _ = background_map(rows, box=background_box)
    if 0.0 < pixel_noise_shrink < 1.0:
        sigma *= resampled_filter_norm(psf_sigma)
    if sigma <= 0.0:
        return float("nan")
    # One pixel wider than the detection kernel, so the Gaussian shifted by up
    # to half a pixel is not truncated on its far side.
    _, r = gaussian_kernel(psf_sigma)
    r += 1
    two_s2 = 2.0 * psf_sigma * psf_sigma
    acc = 0.0
    q = 0.0
    for jj in range(max(0, j0 - r), min(ny, j0 + r + 1)):
        for ii in range(max(0, i0 - r), min(nx, i0 + r + 1)):
            v = rows[jj][ii]
            if v != v:
                continue
            w = math.exp(-((ii - x) ** 2 + (jj - y) ** 2) / two_s2)
            acc += w * (v - bg_map[jj][ii])
            q += w * w
    return acc / (sigma * math.sqrt(q)) if q > 0.0 else float("nan")


def forced_photometry(
    rows: Array2D,
    x: float,
    y: float,
    *,
    psf_sigma: float = PSF_SIGMA_PX,
    background_box: int = BACKGROUND_BOX_PX,
    pixel_noise_shrink: float = 1.0,
) -> tuple[float, float]:
    """PSF amplitude at a known position, and its 1-sigma error, frame units.

    The least-squares amplitude of a Gaussian PSF centred on ``(x, y)`` over
    the background the detector subtracts: ``sum(g*(v - bg)) / sum(g*g)``, with
    error ``sigma / sqrt(sum(g*g))``. The total flux is the amplitude times
    :func:`psf_flux_integral`, which is what turns a point in a light curve into
    a magnitude. ``(nan, nan)`` where the pixel under the position is masked or
    off the array.

    On resampled pixels (``pixel_noise_shrink`` below 1) the error is widened by
    :func:`resampled_filter_norm`, exactly as :func:`forced_snr` widens its
    noise. The fit is the same Gaussian-weighted sum as the matched filter, so
    it sums the same correlated neighbours; without the widening its error
    understated the scatter of repeated readings by that factor, and every
    light-curve error bar with it. So amplitude over error here is the same
    significance :func:`forced_snr` reads.
    """
    ny = len(rows)
    nx = len(rows[0]) if ny else 0
    i0, j0 = int(round(x)), int(round(y))
    if not (0 <= i0 < nx and 0 <= j0 < ny) or rows[j0][i0] != rows[j0][i0]:
        return float("nan"), float("nan")
    bg_map, sigma, _, _, _ = background_map(rows, box=background_box)
    if 0.0 < pixel_noise_shrink < 1.0:
        sigma *= resampled_filter_norm(psf_sigma)
    _, r = gaussian_kernel(psf_sigma)
    r += 1
    two_s2 = 2.0 * psf_sigma * psf_sigma
    acc = 0.0
    gg = 0.0
    for jj in range(max(0, j0 - r), min(ny, j0 + r + 1)):
        for ii in range(max(0, i0 - r), min(nx, i0 + r + 1)):
            v = rows[jj][ii]
            if v != v:
                continue
            g = math.exp(-((ii - x) ** 2 + (jj - y) ** 2) / two_s2)
            acc += g * (v - bg_map[jj][ii])
            gg += g * g
    if gg <= 0.0:
        return float("nan"), float("nan")
    return acc / gg, sigma / math.sqrt(gg)


def _matched_filter(
    excess: Array2D, kernel: list[list[float]], r: int, sigma: float, nx: int, ny: int
) -> Array2D:
    """``sum(k*x) / (sigma*sqrt(sum(k^2)))`` at every pixel, over the pixels that exist.

    Written pixel by pixel in ``exact``, which skips what is off the frame or
    masked. Where the whole kernel lands on finite pixels -- nearly everywhere --
    ``sum(k^2)`` is one number, and the numerator is built a kernel tap at a time
    across the row instead: the same products added in the same order from the
    same zero, so the same floats, for a fraction of the interpreter's work. The
    frame's edge and the neighbourhood of a masked pixel keep ``exact``.
    """
    n = 2 * r + 1
    q_full = 0.0
    for krow in kernel:
        for w in krow:
            q_full += w * w
    denom = sigma * math.sqrt(q_full)

    def exact(j: int, i: int) -> float:
        acc = 0.0
        q = 0.0
        for dj in range(max(0, r - j), min(n, ny + r - j)):
            krow = kernel[dj]
            erow = excess[j + dj - r]
            for di in range(max(0, r - i), min(n, nx + r - i)):
                e = erow[i + di - r]
                if e != e:
                    continue
                w = krow[di]
                acc += w * e
                q += w * w
        return acc / (sigma * math.sqrt(q)) if q > 0.0 else 0.0

    masked = [[i for i, e in enumerate(row) if e != e] for row in excess]
    m = nx - 2 * r
    snr: Array2D = []
    for j in range(ny):
        if m <= 0 or j < r or j >= ny - r:
            snr.append([exact(j, i) for i in range(nx)])
            continue
        acc = [0.0] * m
        for dj in range(n):
            erow = excess[j + dj - r]
            for di, w in enumerate(kernel[dj]):
                acc = [a + w * e for a, e in zip(acc, erow[di:di + m])]
        row = [0.0] * nx
        row[r:nx - r] = [a / denom for a in acc]
        redo = {i + d for jj in range(j - r, j + r + 1) for i in masked[jj]
                for d in range(-r, r + 1)}
        redo.update(range(r))
        redo.update(range(nx - r, nx))
        for i in redo:
            if 0 <= i < nx:
                row[i] = exact(j, i)
        snr.append(row)
    return snr


def _extract(
    snr: Array2D, excess: Array2D, thresh: float, min_pixels: int, nx: int, ny: int, r: int
) -> list[Source]:
    """Connected components above threshold, 8-connected, one pass.

    Iterative rather than recursive: a bright extended blob in a 2040-pixel
    frame is thousands of connected pixels, and a recursive flood fill on that
    is a stack overflow rather than a detection.
    """
    seen = [[False] * nx for _ in range(ny)]
    out: list[Source] = []
    for j0 in range(ny):
        for i0 in [i for i, v in enumerate(snr[j0]) if v >= thresh]:
            if seen[j0][i0]:
                continue
            stack = [(i0, j0)]
            seen[j0][i0] = True
            pix: list[tuple[int, int]] = []
            while stack:
                i, j = stack.pop()
                pix.append((i, j))
                for dj in (-1, 0, 1):
                    for di in (-1, 0, 1):
                        ii, jj = i + di, j + dj
                        if 0 <= ii < nx and 0 <= jj < ny and not seen[jj][ii] \
                                and snr[jj][ii] >= thresh:
                            seen[jj][ii] = True
                            stack.append((ii, jj))
            if len(pix) < min_pixels:
                continue
            out.append(_measure(pix, snr, excess, nx, ny, r))
    return out


def _measure(
    pix: list[tuple[int, int]], snr: Array2D, excess: Array2D, nx: int, ny: int, r: int
) -> Source:
    """Centroid and photometry for one footprint.

    Measured in a **fixed aperture** around the most significant pixel, not over
    the footprint. Over the footprint, a source detected on one pixel has its
    whole flux in that pixel by definition, so ``peak / flux`` is identically 1
    and the sharpness cut -- the thing that separates a star from a cosmic ray
    -- would throw away every marginal detection in the frame while keeping the
    ray. A box the width of the detection kernel is the same aperture for every
    source, so the ratio means the same thing for every source.

    The centroid is weighted by the *unfiltered* excess, not by the filtered
    significance: the filter is a smoothing kernel, and centroiding on a
    smoothed image throws away the sub-pixel information the smoothing spread
    out. Negative excess -- noise, at the aperture's edge -- is clipped to zero
    rather than allowed to pull the centroid the wrong way.

    For a blended or genuinely extended object the footprint is larger than the
    aperture and ``flux`` under-reports it. That is aperture photometry behaving
    as aperture photometry; ``npix`` is what says the footprint was bigger.

    The ``wsum <= 0`` centroid fallback is unreachable, and deliberately kept.
    The kernel radius and the aperture radius are the same ``r``, so the pixels
    feeding ``snr[bj][bi]`` are exactly the pixels this aperture sums. If none of
    them had positive excess then ``acc`` would be non-positive, the pixel's
    significance would be <= 0, and it could never have cleared the threshold to
    start a footprint. A detected source therefore always has ``wsum > 0``.

    Every other branch here is exercised by ``search/detect_branches`` in the
    golden vectors -- a masked pixel inside an aperture, a masked pixel *on* the
    peak, an aperture clipped by the frame edge, an aperture summing negative,
    and a raw sharpness above 1. Each was confirmed live by perturbing it and
    watching the vectors move; this is the one that cannot be, which is worth
    knowing before spending an afternoon trying to write the test.
    """
    best = -math.inf
    bi = bj = 0
    for i, j in pix:
        if snr[j][i] > best:
            best, bi, bj = snr[j][i], i, j

    wsum = xsum = ysum = flux = 0.0
    peak = excess[bj][bi]
    if peak != peak:
        peak = 0.0
    for j in range(max(0, bj - r), min(ny, bj + r + 1)):
        for i in range(max(0, bi - r), min(nx, bi + r + 1)):
            e = excess[j][i]
            if e != e:
                continue
            flux += e
            w = e if e > 0.0 else 0.0
            wsum += w
            xsum += w * i
            ysum += w * j
    cx, cy = (xsum / wsum, ysum / wsum) if wsum > 0.0 else (float(bi), float(bj))
    sharp = peak / flux if flux > 0.0 else 1.0
    return Source(cx, cy, best, flux, peak, len(pix), min(1.0, max(0.0, sharp)))
