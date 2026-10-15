"""House style for the matplotlib figures (plotting standard, section 5.2).

    import sys; sys.path.insert(0, "<results_analysis>/plot-style")
    import house_style as hs
    hs.use()                       # applies house.mplstyle
    fig, ax = plt.subplots(figsize=hs.canvas(183, aspect=0.6))
    hs.style_axes(ax)              # two spines, outward ticks, no grid
    hs.tag(fig, "a", x=0.01, y=0.99)

Colours and sizes match ResultsAnalysis/plot-style/house_style.R in the
Bayesian-ARGOS submodule. Never modifies package code.
"""
import os

import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
STYLE = os.path.join(HERE, "house.mplstyle")

K = 2.5                                     # canvas = K x final size
PT = {"tag": 8 * K, "text": 7 * K, "tick": 6.5 * K}

COLOURS = {"Bayesian-ARGOS": "#505ed5", "ARGOS": "#d74d3d", "SINDy": "#8eb63d"}
EXTRA = ["#FA5524", "#EF60AD", "#009E9E", "#8B5A2B"]
PRIMARY, HIGHLIGHT, GREY = "#505ed5", "#d74d3d", "#bfbfbf"       # grey75
GRU = "#6a51a3"                                                  # GRU latent trajectory
BAND_ALPHA = 0.25
FIELD_CMAP = "viridis"      # sequential map for the SST field (decision D4)
DIFF_CMAP = "RdBu_r"        # diverging map centred on zero for differences
MARKERS = {"Bayesian-ARGOS": "o", "ARGOS": "s", "SINDy": "^"}


def use():
    """Apply the house style globally."""
    plt.style.use(STYLE)


def canvas(width_mm=183, aspect=0.5):
    """Figure size in inches for a figure of final width width_mm (183 or 89)."""
    w = width_mm / 25.4 * K
    return (w, w * aspect)


def style_axes(ax, grid=False):
    """Two spines (left, bottom), outward ticks, no grid (standard 4.3)."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.25)
        ax.spines[side].set_color("black")
    ax.tick_params(direction="out", length=7.1, width=1.25, colors="black", labelsize=PT["tick"])
    ax.grid(grid)
    return ax


def image_axes(ax):
    """Image panel without ticks: thin neutral frame marking the extent (standard 4.3.1)."""
    ax.set_xticks([]); ax.set_yticks([])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.8)
        ax.spines[side].set_color("0.5")
    return ax


def tag(fig_or_ax, letter, x=0.0, y=1.0, **kw):
    """Lowercase bold panel tag, 8 pt at final size, top left (standard 4.8.1)."""
    opts = dict(fontsize=PT["tag"], fontweight="bold", ha="left", va="bottom")
    opts.update(kw)
    if isinstance(fig_or_ax, mpl.figure.Figure):
        return fig_or_ax.text(x, y, letter, **opts)
    return fig_or_ax.text(x, y, letter, transform=fig_or_ax.transAxes, **opts)


def save(fig, path, dpi=600, pad_inches=0.1):
    """Vector PDF, tight bounding box. Panels that are composed into one figure by
    TikZ pass a small pad (0.02 in) so that their widths add up to the canvas."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=pad_inches,
                facecolor="white", edgecolor="none")
    return path
