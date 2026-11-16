"""Shared style for redrawn Orrery figures: fixed palette roles, style loading, saving, panel letters."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "site", "data", "figs", "v2")

# Okabe-Ito colours (grey is the usual neutral companion of the set).
OKABE_ITO = dict(black="#000000", orange="#E69F00", sky_blue="#56B4E9", bluish_green="#009E73",
                 yellow="#F0E442", blue="#0072B2", vermillion="#D55E00", reddish_purple="#CC79A7", grey="#999999")

# Fixed meaning in every figure. Marker and line style keep roles apart in greyscale.
ROLES = {
    "spherex":   dict(color=OKABE_ITO["blue"],           marker="o", ls="-"),   # SPHEREx measurement
    "reference": dict(color=OKABE_ITO["orange"],         marker="s", ls="--"),  # JWST, Gaia, Spitzer, WISE, NEOWISE, literature
    "model":     dict(color=OKABE_ITO["bluish_green"],   marker="^", ls="-."),  # model / prediction
    "control":   dict(color=OKABE_ITO["grey"],           marker="x", ls=":"),   # control, null, empty paths
    "highlight": dict(color=OKABE_ITO["vermillion"],     marker="*", ls="-"),   # highlight / candidate
    "secondary": dict(color=OKABE_ITO["reddish_purple"], marker="v", ls=(0, (5, 2, 1, 2))),  # secondary category
}
INK = "#222222"
IMAGE_CMAP = "gray"         # sky pictures
SEQUENTIAL_CMAP = "viridis"  # sequential quantities
WIDTH = 6.8                  # inches; shown at ~680 px


def use_style():
    plt.style.use(os.path.join(HERE, "style.mplstyle"))


def save(fig, name):
    """Write <name>.svg, .pdf and .png (200 dpi, 2x the displayed width) into site/data/figs/v2/."""
    os.makedirs(OUT, exist_ok=True)
    for ext, kw in (("svg", {}), ("pdf", {}), ("png", {"dpi": 200})):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), **kw)
    plt.close(fig)


def panel_label(ax, letter):
    ax.text(-0.02, 1.02, letter, transform=ax.transAxes, ha="right", va="bottom", fontsize=11, fontweight="bold")
