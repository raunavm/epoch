"""Independent JAX reference implementation of the ORRERY-X stamp likelihood.

Written from docs/INTERFACES.md ("Conventions", "ePSF evaluation", "Stamp
likelihood") only, to cross-check the primary implementation.

Choices where the spec is ambiguous (see module report):
- ePSF interpolation: Keys cubic convolution (a=-0.5), taps beyond the
  33x33 grid read as 0; value is 0 if the sample point lies outside
  [0, n-1] on either axis. (Spec names scipy map_coordinates order=3, a
  prefiltered cubic B-spline; the two agree at grid nodes and differ at
  the 1e-3..1e-4 relative level in between for smooth PSFs.)
- Normalisation box: integer pixel centres (on the infinite native grid,
  not clipped to the stamp) with |dx|<=3 and |dy|<=3. The PSF is then
  evaluated on every stamp pixel (nonzero out to 3.2 px), divided by that sum.
- Background plane centre (xm, ym) = geometric stamp centre ((W-1)/2, (H-1)/2).
- Masked pixels get zero weight (rows of zeros in the whitened system);
  dof = n_good - n_params.
- model is computed on every pixel; resid = image - model on every pixel
  (NaN where the input image is NaN). mask is True where a pixel is used.
- Coefficient order: [A, B_1..B_K, c0, c1, c2].
"""
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.scipy.linalg import solve_triangular  # noqa: E402

ARCSEC_PER_RAD = 206264.806
BOX_HALF = 3


def _keys(t, a=-0.5):
    t = jnp.abs(t)
    w1 = (a + 2) * t**3 - (a + 3) * t**2 + 1
    w2 = a * t**3 - 5 * a * t**2 + 8 * a * t - 4 * a
    return jnp.where(t <= 1, w1, jnp.where(t < 2, w2, 0.0))


def epsf_eval(epsf, dx, dy, oversample=5):
    """ePSF value at native offsets (dx, dy) = pixel centre - source position.

    Keys bicubic convolution on the oversampled grid; zero outside the grid.
    dx, dy: arrays of identical shape.
    """
    epsf = jnp.asarray(epsf, dtype=jnp.float64)
    ny, nx = epsf.shape
    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
    u = cx + oversample * dx  # column coordinate
    v = cy + oversample * dy  # row coordinate
    inside = (u >= 0) & (u <= nx - 1) & (v >= 0) & (v <= ny - 1)
    pad = 2
    g = jnp.pad(epsf, pad)
    i0 = jnp.floor(u)
    j0 = jnp.floor(v)
    out = jnp.zeros_like(u)
    for di in range(-1, 3):
        wi = _keys(u - (i0 + di))
        ii = jnp.clip(i0 + di, -pad, nx - 1 + pad).astype(int) + pad
        for dj in range(-1, 3):
            wj = _keys(v - (j0 + dj))
            jj = jnp.clip(j0 + dj, -pad, ny - 1 + pad).astype(int) + pad
            out = out + wi * wj * g[jj, ii]
    return jnp.where(inside, out, 0.0)


def psf_image(epsf, x, y, shape, oversample=5):
    """Normalised native-pixel PSF image [H, W] for a source at (x, y)."""
    H, W = shape
    yy, xx = jnp.mgrid[0:H, 0:W]
    img = epsf_eval(epsf, xx - x, yy - y, oversample)
    # normalisation box on the infinite integer grid around the source
    k = jnp.arange(-BOX_HALF - 1, BOX_HALF + 2)
    bx = jnp.round(x) + k
    by = jnp.round(y) + k
    BX, BY = jnp.meshgrid(bx, by)
    DX, DY = BX - x, BY - y
    inbox = (jnp.abs(DX) <= BOX_HALF) & (jnp.abs(DY) <= BOX_HALF)
    norm = jnp.sum(jnp.where(inbox, epsf_eval(epsf, DX, DY, oversample), 0.0))
    return img / norm


def _design(epsf, xs, ys, star_xy, shape, oversample):
    H, W = shape
    yy, xx = jnp.mgrid[0:H, 0:W]
    xm, ym = (W - 1) / 2.0, (H - 1) / 2.0
    cols = [psf_image(epsf, xs, ys, shape, oversample)]
    for sx, sy in star_xy:
        cols.append(psf_image(epsf, sx, sy, shape, oversample))
    cols += [jnp.ones(shape), (xx - xm).astype(float), (yy - ym).astype(float)]
    return jnp.stack([c.ravel() for c in cols], axis=1)  # [H*W, P]


def _good_mask(variance, flags, bad_mask_bits):
    variance = jnp.asarray(variance, dtype=jnp.float64)
    return ((jnp.asarray(flags).astype(np.int64) & int(bad_mask_bits)) == 0) & jnp.isfinite(
        variance
    ) & (variance > 0)


def _solve(image, variance, good, X):
    y = jnp.asarray(image, dtype=jnp.float64).ravel()
    var = jnp.asarray(variance, dtype=jnp.float64).ravel()
    g = good.ravel()
    w = jnp.where(g, 1.0 / jnp.sqrt(jnp.where(g, var, 1.0)), 0.0)
    yw = jnp.where(g, y, 0.0) * w
    Xw = X * w[:, None]
    Q, R = jnp.linalg.qr(Xw, mode="reduced")
    coeffs = solve_triangular(R, Q.T @ yw, lower=False)
    r = yw - Xw @ coeffs
    chi2 = jnp.sum(r**2)
    return coeffs, R, chi2


def ref_fit(image, variance, flags, epsf, xs, ys, star_xy, bad_mask_bits, oversample=5):
    """Weighted linear fit of PSF(asteroid) + PSF(stars) + plane background."""
    image = np.asarray(image, dtype=np.float64)
    shape = image.shape
    star_xy = [tuple(map(float, s)) for s in star_xy]
    X = _design(epsf, float(xs), float(ys), star_xy, shape, oversample)
    good = _good_mask(variance, flags, bad_mask_bits)
    coeffs, R, chi2 = _solve(image, variance, good, X)
    Rinv = solve_triangular(R, jnp.eye(R.shape[0]), lower=False)
    cov = Rinv @ Rinv.T
    model = (X @ coeffs).reshape(shape)
    n_good = int(jnp.sum(good))
    return dict(
        A=float(coeffs[0]),
        A_err=float(jnp.sqrt(cov[0, 0])),
        coeffs=np.asarray(coeffs),
        cov=np.asarray(cov),
        chi2=float(chi2),
        dof=n_good - X.shape[1],
        model=np.asarray(model),
        resid=image - np.asarray(model),
        mask=np.asarray(good),
    )


def ref_loglike(xs, ys, image, variance, flags, epsf, star_xy, bad_mask_bits, oversample=5):
    """Profile log-likelihood -chi2_min/2 (linear params profiled out).

    Pure JAX in (xs, ys): use jax.grad(ref_loglike, argnums=(0, 1)).
    Constant terms (log variance) are dropped.
    """
    shape = np.shape(image)
    X = _design(epsf, xs, ys, [tuple(s) for s in star_xy], shape, oversample)
    good = _good_mask(variance, flags, bad_mask_bits)
    _, _, chi2 = _solve(image, variance, good, X)
    return -0.5 * chi2


def ref_flux_mJy(A, omega_arcsec2):
    """A (MJy/sr summed over pixels) -> flux density in mJy."""
    return A * omega_arcsec2 / ARCSEC_PER_RAD**2 * 1e9
