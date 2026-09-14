"""Map panels for Fig. 8c (observations / predictions / differences).

`plot_zoomed_comparison` is the function used for the submitted figure
(`results_analysis/analysis_script_ba.py`), copied verbatim except that the
inputs are already in degrees Celsius. `lon_range=(0, 180)` reproduces the
submitted Eastern-Hemisphere crop; `(0, 360)` gives the full globe used in the
revision. The RMSE printed on each panel is computed on whatever window is shown.
"""
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.transforms import Bbox


def plot_zoomed_comparison(
    real_data,
    reconstructed_data_via_forecaster,
    sst_locs,
    timesteps,
    lat_range,
    lon_range,
    diff_scale=1.0,
    *,
    cmap="coolwarm",
    diff_cmap="RdBu_r",  # diverging colormap
    figsize=None,
    fig_dpi=600,
    add_colorbar=True,
    annotate_stats=True,
    font_scale=1.0,
    background_value=np.nan,
    suptitle=None,
    robust=True,
    robust_pct=(1, 99),
    save=False,  # NEW: whether to save the figure
    save_dir=None,  # NEW: directory to save into
    filename="zoomed_comparison.pdf",  # NEW: filename
):
    """
    Publication-quality comparison of observed vs. model SST fields (zoomed) and their differences.
    Keeps original latitude orientation, removes geo ticks/labels.

    Args:
        real_data: The true SST data.
        reconstructed_data_via_forecaster: The reconstructed SST data from the forecaster.
        sst_locs: The locations of the SST data.
        timesteps: The timesteps to plot.
        lat_range: The latitude range to zoom in on.
        lon_range: The longitude range to zoom in on.
        diff_scale: Scaling factor for the difference plots.
        cmap: Colormap for the SST plots.
        diff_cmap: Colormap for the difference plots.
        figsize: Figure size.
        fig_dpi: Figure DPI.
        add_colorbar: Whether to add a colorbar to the plots.
        annotate_stats: Whether to annotate the plots with statistics.
        font_scale: Font size scale factor.
        background_value: Value for the background of the plots.
        suptitle: Super title for the figure.
        robust: Whether to use robust scaling for the color limits.
        robust_pct: Percentiles for robust scaling.
        save (bool): If True, save the generated figure.
        save_dir (str | None): Directory to save figure when save=True.
        filename (str): Output filename when save=True.
    Returns:
        dict: Global stats; includes 'saved_path' key if figure was saved.
    """
    # Inverse transform
    real_data_inv = real_data  # already in degC
    recon_data_inv = reconstructed_data_via_forecaster  # already in degC

    # Helper to rebuild the 2D field
    def build_field(vector):
        field = np.full(180 * 360, background_value, dtype=float)
        field[sst_locs] = vector
        field = field.reshape(180, 360)
        return field[lat_range[0] : lat_range[1], lon_range[0] : lon_range[1]]

    real_stack = np.stack(
        [build_field(real_data_inv[t, :]) for t in timesteps], axis=0
    )
    model_stack = np.stack(
        [build_field(recon_data_inv[t, :]) for t in timesteps], axis=0
    )
    diff_stack = real_stack - model_stack

    # Robust color limits for main data
    if robust:
        all_data = np.concatenate([real_stack.ravel(), model_stack.ravel()])
        data_min = np.nanpercentile(all_data, robust_pct[0])
        data_max = np.nanpercentile(all_data, robust_pct[1])
    else:
        data_min = np.nanmin([real_stack, model_stack])
        data_max = np.nanmax([real_stack, model_stack])

    # Symmetric diff limits
    if robust:
        m = np.nanpercentile(np.abs(diff_stack), robust_pct[1])
        diff_lim = float(m) * diff_scale
    else:
        diff_lim = np.nanmax(np.abs(diff_stack)) * diff_scale
    diff_norm = TwoSlopeNorm(vmin=-diff_lim, vcenter=0.0, vmax=diff_lim)

    # Figure layout
    n_cols = len(timesteps)
    if figsize is None:
        figsize = (max(6.5, 2.4 * n_cols), 8)

    fig, axes = plt.subplots(
        3, n_cols, figsize=figsize, dpi=fig_dpi, constrained_layout=True
    )
    if n_cols == 1:
        axes = axes.reshape(3, 1)

    row_titles = ["Observations", "Predictions", "Differences"]

    for j, t in enumerate(timesteps):
        im0 = axes[0, j].imshow(
            real_stack[j],
            cmap=cmap,
            norm=Normalize(vmin=data_min, vmax=data_max),
            aspect="equal",
            interpolation="nearest",
        )
        im1 = axes[1, j].imshow(
            model_stack[j],
            cmap=cmap,
            norm=Normalize(vmin=data_min, vmax=data_max),
            aspect="equal",
            interpolation="nearest",
        )
        im2 = axes[2, j].imshow(
            diff_stack[j],
            cmap=diff_cmap,
            norm=diff_norm,
            aspect="equal",
            interpolation="nearest",
        )

        # Titles & labels
        axes[0, j].set_title(f"t = {t}", fontsize=18 * font_scale, pad=6)
        for r in range(3):
            axes[r, j].set_xticks([])
            axes[r, j].set_yticks([])
            if j == 0:
                axes[r, j].set_ylabel(row_titles[r], fontsize=18 * font_scale)

        if annotate_stats:
            rmse = float(np.nanmean((diff_stack[j]) ** 2)) ** 0.5
            axes[2, j].text(
                0.02,
                0.02,
                f"RMSE={rmse:.3f}",
                fontsize=16 * font_scale,
                color="black",
                ha="left",
                va="bottom",
                transform=axes[2, j].transAxes,
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    alpha=0.7,
                    linewidth=0.5,
                ),
            )

    if add_colorbar:
        # Let tight_layout finish so we can read final axes positions
        # plt.tight_layout(w_pad=0.9, h_pad=0.9)
        fig.subplots_adjust(right=0.88)
        fig.canvas.draw()  # ensures positions are finalized

        # union of all axes in rows 0–1 (SST) and row 2 (diffs)
        row01_boxes = [ax.get_position(fig) for ax in axes[0, :]] + [
            ax.get_position(fig) for ax in axes[1, :]
        ]
        row2_boxes = [ax.get_position(fig) for ax in axes[2, :]]

        box_row01 = Bbox.union(row01_boxes)
        box_row2 = Bbox.union(row2_boxes)

        # spacing to the right of the panels (figure fraction)
        pad = 0.02
        width = 0.01

        # create colorbar axes that match the row heights
        shrink = 0.95
        shrink_2 = 0.9
        cax0 = fig.add_axes(
            [
                box_row01.x1 + pad,
                box_row01.y0 + (1 - shrink) * box_row01.height / 2,
                width,
                shrink * box_row01.height,
            ]
        )
        cax1 = fig.add_axes(
            [
                box_row2.x1 + pad,
                box_row2.y0 + (1 - shrink_2) * box_row2.height / 2,
                width,
                shrink_2 * box_row2.height,
            ]
        )

        # draw colorbars
        cb0 = fig.colorbar(im0, cax=cax0)
        cb1 = fig.colorbar(im2, cax=cax1)

        # styling
        cb0.ax.tick_params(labelsize=16 * font_scale)
        cb1.ax.tick_params(labelsize=16 * font_scale)
        cb0.set_label("SST (°C)", fontsize=18 * font_scale)
        cb1.set_label("Difference (°C)", fontsize=18 * font_scale)

    if suptitle:
        fig.suptitle(suptitle, fontsize=10 * font_scale, y=0.995)

    fig.text(
        -0.01, 1.03, "c", fontsize=28, fontweight="bold", ha="left", va="top"
    )
    # plt.tight_layout(w_pad=0.6, h_pad=0.6)

    saved_path = None
    if save:
        if save_dir is None:
            raise ValueError("save=True but save_dir is None.")
        os.makedirs(save_dir, exist_ok=True)
        saved_path = os.path.join(save_dir, filename)
        fig.savefig(saved_path, dpi=fig_dpi, bbox_inches="tight")

    plt.show()

    # Global stats
    global_rmse = float(np.nanmean(diff_stack**2) ** 0.5)
    global_mae = float(np.nanmean(np.abs(diff_stack)))
    result = {"RMSE": global_rmse, "MAE": global_mae}
    # if saved_path:
    #     result["saved_path"] = saved_path
    return result

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fields", default="results/fields_18.npz")
    p.add_argument("--lon", default="0,360", help="longitude columns, e.g. 0,180 for the submitted crop")
    p.add_argument("--figsize", default="12,5.6")
    p.add_argument("--font-scale", type=float, default=0.8)
    p.add_argument("--out", default="figures/global_comparison_18.pdf")
    a = p.parse_args()
    d = np.load(a.fields)
    ts = [int(t) for t in d["timesteps"]]
    # the npz stores only the plotted horizons; put them back at their row index
    real = np.full((max(ts) + 1, d["real"].shape[1]), np.nan); real[ts] = d["real"]
    pred = np.full((max(ts) + 1, d["pred"].shape[1]), np.nan); pred[ts] = d["pred"]
    lon = tuple(int(v) for v in a.lon.split(","))
    figsize = tuple(float(v) for v in a.figsize.split(","))
    plot_zoomed_comparison(
        real, pred, d["sst_locs"], ts, (0, 180), lon,
        diff_scale=1, figsize=figsize, font_scale=a.font_scale,
        save=True, save_dir=os.path.dirname(a.out) or ".", filename=os.path.basename(a.out),
    )
    print("saved", a.out)


if __name__ == "__main__":
    main()
