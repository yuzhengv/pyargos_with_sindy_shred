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
PT = {"tag": 8 * K, "text": 7 * K, "tick": 6.5 * K, "legend": 9 * K}

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


def save(fig, path, dpi=600, pad_inches=0.1, tight=True):
    """Vector PDF, tight bounding box. Panels that are composed into one figure by
    TikZ pass a small pad (0.02 in) so that their widths add up to the canvas;
    tight=False saves the whole canvas, so that panels drawn with the same
    canvas and the same axes box (Fig. 9a and 9b) align exactly in the composite."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight" if tight else None, pad_inches=pad_inches,
                facecolor="white", edgecolor="none")
    return path


LEGEND_GAP_IN = 0.3         # gap between the x-axis title and a legend row below it (3 mm final)


def hanging_legend(fig, handles, ncol=1, gap_in=LEGEND_GAP_IN, anchor_y=0.0, **kw):
    """A figure legend hung below the axes area: its top edge gap_in below the
    anchor height anchor_y (figure fraction; 0 = the canvas bottom, where
    constrained layout places the x-axis title), so that the gap between the
    axis title and the legend is explicit (Yuzheng, 2026-11-01: "enlarge the
    gap between axis name and legends"). Under constrained layout the legend is
    outside the canvas and a tight bounding box (save) includes it; with a fixed
    axes box (fig.subplots_adjust) anchor_y is the top of the reserved strip."""
    h = fig.get_size_inches()[1]
    opts = dict(loc="upper center", bbox_to_anchor=(0.5, anchor_y - gap_in / h), ncol=ncol, frameon=False,
                handlelength=2.2, columnspacing=1.6, labelspacing=0.6)
    opts.update(kw)
    return fig.legend(handles=handles, **opts)
