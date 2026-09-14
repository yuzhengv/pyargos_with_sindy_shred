"""Fig. S9: observed vs forecast trajectories at 18 held-out sensor locations.

`plot_multiple_sensor_predictions` is copied verbatim from
`results_analysis/analysis_script_ba.py`. Inputs are in the normalised
(min-max, [0, 1]) units the network works in, as in the submitted figure; the
RMSE printed per panel is in those units.
"""
import argparse

import numpy as np
from matplotlib.lines import Line2D


def plot_multiple_sensor_predictions(
    real_data,
    forecaster_data,
    sensor_locations,
    sensor_indices,
    num_train=52,
    num_pred=250,
    rows=3,
    cols=6,
    *,
    # Restored original colors
    train_color="#6a51a3",
    pred_color="#505ed5",
    train_label="Observation",
    pred_label="Forecast",
    figsize=None,
    dpi=600,
    lw=1.6,
    font_scale=1,
    robust=True,
    robust_pct=(2, 98),
    share_ylim=True,
    panel_labels=False,
    panel_label_style=dict(x=0.02, y=0.92, fontsize=9, weight="bold"),
    vertical_line_style=dict(color="gray", linestyle=":", linewidth=1.0),
    grid=True,
    grid_alpha=0.15,
    grid_linestyle="--",
    show_rmse=True,
    rmse_box_style=dict(
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.85", lw=0.6),
        fontsize=16,
        x=0.98,
        y=0.06,
        ha="right",
        va="bottom",
    ),
    xlabel="Time (weeks)",
    ylabel="Value",
    sci_yaxis=True,
    sci_limits=(-2, 3),
    tick_count_x=5,
    tick_count_y=5,
    tight=True,
    metadata_title="Observed vs Forecasted Trajectories",
    save_path=None,
):
    import os
    import string

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import AutoMinorLocator, MaxNLocator, ScalarFormatter

    # Style tweaks
    rc = mpl.rcParams
    rc["font.size"] = 16 * font_scale
    rc["axes.titlesize"] = 18 * font_scale
    rc["axes.labelsize"] = 16 * font_scale
    rc["xtick.labelsize"] = 16 * font_scale
    rc["ytick.labelsize"] = 16 * font_scale
    rc["axes.linewidth"] = 1
    rc["figure.facecolor"] = "white"
    rc["axes.facecolor"] = "white"
    rc["legend.frameon"] = True

    total_needed = len(sensor_indices)
    max_panels = rows * cols
    if total_needed > max_panels:
        sensor_indices = sensor_indices[:max_panels]

    if figsize is None:
        figsize = (cols * 4, rows * 4)

    T_window = num_train + num_pred
    rmse_per_sensor = {}
    fig, axes = plt.subplots(
        rows, cols, figsize=figsize, dpi=dpi, sharex=True, sharey=share_ylim
    )
    axes = np.atleast_1d(axes).reshape(rows, cols)
    y_collect = []

    y_formatter = ScalarFormatter(useMathText=True)
    if sci_yaxis:
        y_formatter.set_powerlimits(sci_limits)

    for plot_idx, sensor_idx in enumerate(sensor_indices):
        r, c = divmod(plot_idx, cols)
        ax = axes[r, c]
        sensor_col = int(sensor_locations[sensor_idx])
        obs_series = np.asarray(real_data[:T_window, sensor_col]).astype(float)
        model_series = np.asarray(
            forecaster_data[:T_window, sensor_col]
        ).astype(float)

        obs_hist = obs_series[:num_train]
        obs_fore = obs_series[num_train:]
        model_fore = model_series[num_train : num_train + num_pred]

        x_hist = np.arange(num_train)
        x_fore = np.arange(num_train, num_train + len(obs_fore))

        ax.plot(x_hist, obs_hist, color=train_color, lw=lw, alpha=0.9)
        ax.plot(x_fore, obs_fore, color=train_color, lw=lw, alpha=0.35)
        ax.plot(
            x_fore[: len(model_fore)],
            model_fore,
            color=pred_color,
            lw=lw,
            linestyle="--",
            alpha=0.95,
        )

        ax.axvline(x=num_train, **vertical_line_style)
        ax.set_title(f"Sensor {sensor_idx}", pad=3.5)

        if panel_labels:
            label = f"({string.ascii_lowercase[plot_idx]})"
            ax.text(
                panel_label_style.get("x", 0.02),
                panel_label_style.get("y", 0.92),
                label,
                transform=ax.transAxes,
                fontsize=panel_label_style.get("fontsize", 9) * font_scale,
                weight=panel_label_style.get("weight", "bold"),
                ha="left",
                va="center",
            )

        if grid:
            ax.grid(
                True, which="major", linestyle=grid_linestyle, alpha=grid_alpha
            )
        ax.xaxis.set_major_locator(
            MaxNLocator(nbins=tick_count_x, integer=True)
        )
        ax.yaxis.set_major_locator(MaxNLocator(nbins=tick_count_y))
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(axis="both", length=3.5, width=0.7, pad=2.5)
        ax.tick_params(which="minor", length=2.0, width=0.6)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        if sci_yaxis:
            ax.yaxis.set_major_formatter(y_formatter)

        if share_ylim:
            if len(obs_hist):
                y_collect.append(obs_hist[np.isfinite(obs_hist)])
            if len(obs_fore):
                y_collect.append(obs_fore[np.isfinite(obs_fore)])
            if len(model_fore):
                y_collect.append(model_fore[np.isfinite(model_fore)])

        min_len = min(len(model_fore), len(obs_fore))
        if min_len > 0:
            valid = np.isfinite(model_fore[:min_len]) & np.isfinite(
                obs_fore[:min_len]
            )
            if valid.any():
                rmse = float(
                    np.sqrt(
                        np.mean(
                            (
                                model_fore[:min_len][valid]
                                - obs_fore[:min_len][valid]
                            )
                            ** 2
                        )
                    )
                )
                rmse_per_sensor[sensor_idx] = rmse
                if show_rmse:
                    ax.text(
                        rmse_box_style["x"],
                        rmse_box_style["y"],
                        f"RMSE: {rmse:.3g}",
                        transform=ax.transAxes,
                        fontsize=rmse_box_style["fontsize"] * font_scale,
                        ha=rmse_box_style["ha"],
                        va=rmse_box_style["va"],
                        bbox=rmse_box_style["bbox"],
                    )
            else:
                rmse_per_sensor[sensor_idx] = float("nan")

    for leftover in range(len(sensor_indices), max_panels):
        r, c = divmod(leftover, cols)
        fig.delaxes(axes[r, c])

    if share_ylim and len(y_collect):
        all_vals = np.concatenate([arr for arr in y_collect if arr.size > 0])
        if all_vals.size > 0:
            if robust:
                vmin = np.nanpercentile(all_vals, robust_pct[0])
                vmax = np.nanpercentile(all_vals, robust_pct[1])
            else:
                vmin = np.nanmin(all_vals)
                vmax = np.nanmax(all_vals)
            if np.isfinite(vmin) and np.isfinite(vmax):
                pad = 0.04 * (
                    vmax - vmin if vmax > vmin else max(abs(vmax), 1.0)
                )
                for ax in fig.axes:
                    if ax.has_data():
                        ax.set_ylim(vmin - pad, vmax + pad)

    for r in range(rows):
        for c in range(cols):
            try:
                ax = axes[r, c]
            except Exception:
                continue
            if ax not in fig.axes or not ax.has_data():
                continue
            if r == rows - 1:
                ax.set_xlabel(xlabel)
            if c == 0:
                ax.set_ylabel(ylabel)

    # Legend at bottom
    legend_lines = [
        Line2D([0], [0], color=train_color, lw=2, label=train_label),
        Line2D(
            [0], [0], color=pred_color, lw=2, linestyle="--", label=pred_label
        ),
    ]
    fig.legend(
        handles=legend_lines,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=2,
        frameon=True,
        fontsize=20 * font_scale,
        handlelength=2.2,
        columnspacing=1.8,
    )

    if tight:
        fig.tight_layout(rect=[0, 0.03, 1, 1])  # extra bottom space for legend

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        ext = os.path.splitext(save_path)[1].lower()
        fig.savefig(
            save_path,
            dpi=dpi if ext in {".png", ".jpg", ".tif"} else None,
            bbox_inches="tight",
            metadata=dict(Title=metadata_title),
        )

    valid_vals = [v for v in rmse_per_sensor.values() if np.isfinite(v)]
    stats = {
        "per_sensor_rmse": rmse_per_sensor,
        "mean_rmse": float(np.mean(valid_vals)) if valid_vals else np.nan,
    }
    return fig, axes, stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trajectories", default="results/sensor_trajectories_18.npz")
    p.add_argument("--out", default="figures/multi_sensor_forecast_18.pdf")
    a = p.parse_args()
    d = np.load(a.trajectories)
    n = d["obs"].shape[1]
    _, _, stats = plot_multiple_sensor_predictions(
        d["obs"], d["fore"], np.arange(n), list(range(n)), rows=3, cols=6, save_path=a.out
    )
    print("saved", a.out, "mean RMSE", round(stats["mean_rmse"], 4))


if __name__ == "__main__":
    main()
