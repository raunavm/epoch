"""Central wavelength across each SPHEREx detector, with the positions of one constant star over a year of pictures.

Inputs: data/ref/epsf_dr1_D{1..6}.npz (wx, wy node grid; wv[..., 0] = central wavelength in µm, indexed [y, x]),
interpolated bilinearly as orrery.data.Frame.wave_at does; per-picture x, y of Gaia DR3 1633307862887569280 (a constant
north-ecliptic-pole star) from data/phase2/e18/phot.
Run from the repo root: .venv/bin/python figures/plot_wavelength_map.py
"""
import os
import sys

import numpy as np
import pyarrow.compute as pc
import pyarrow.dataset as ds
from scipy.interpolate import RegularGridInterpolator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SEQUENTIAL_CMAP, WIDTH, panel_label, plt, save, use_style

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
STAR = "1633307862887569280"
N = 2040

use_style()
star = ds.dataset(os.path.join(ROOT, "data/phase2/e18/phot"), format="parquet").to_table(
    columns=["x", "y", "detector"], filter=pc.field("id") == STAR).to_pandas()

fig, axs = plt.subplots(2, 3, figsize=(WIDTH, 4.6), sharex=True, sharey=True)
g = np.arange(0, N, 8.0)
for d, ax in zip(range(1, 7), axs.ravel()):
    z = np.load(os.path.join(ROOT, f"data/ref/epsf_dr1_D{d}.npz"))
    f = RegularGridInterpolator((z["wy"], z["wx"]), z["wv"][..., 0])
    yy, xx = np.meshgrid(g, g, indexing="ij")
    lam = f(np.stack([yy.ravel(), xx.ravel()], -1)).reshape(yy.shape)
    im = ax.imshow(lam, rasterized=True, origin="lower", extent=(0, N, 0, N), cmap=SEQUENTIAL_CMAP, interpolation="bilinear")
    s = star[star.detector == d]
    ax.plot(s.x, s.y, ls="none", marker="o", ms=1.2, mew=0, color="white", alpha=0.8, rasterized=True)
    cb = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.03, aspect=22)
    cb.set_ticks(np.round(np.linspace(lam.min(), lam.max(), 3), 2))
    cb.ax.tick_params(labelsize=8)
    cb.set_label("µm", fontsize=8, labelpad=2)
    panel_label(ax, "abcdef"[d - 1])
    ax.set_xticks([0, 1000, 2000])
    ax.set_yticks([0, 1000, 2000])
fig.supxlabel("x (pixel)", fontsize=10)
fig.supylabel("y (pixel)", fontsize=10)
save(fig, "wavelength_map")
