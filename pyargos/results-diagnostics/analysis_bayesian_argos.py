# %%
import os
import sys
from pathlib import Path

import adelie as ad
import arviz as az
import dill
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors

# sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import pandas as pd
import utils.argos_simulator as ags
import utils.argos_utils as agu
from src.argos_bayesian_argos import BayesianArgos, BayesianArgosAnalysis


# %%
# - Functions to adjust Adelie diagnostic plots
def adjust_adelie_diagnostic_plot_svg(
    diagnostic_obj,
    filename="figures/coefficients_initial.svg",
    style="seaborn-v0_8",
    remove_titles=True,
    show_axes=True,
    show_axis_edges=False,
    spine_linewidth=1.5,
    spine_color="#333333",
    show_ticks=False,  # NEW: control tick (marks + labels) visibility
):
    """
    Plot coefficients with styling and save as SVG.
    - Style (if provided) is applied first, then custom rcParams override it.
    - When show_axes is False, spines are hidden (but ticks may remain if show_ticks=True).
    - show_ticks toggles tick marks and tick labels independently of axis (spine) visibility.
    - spine_linewidth / spine_color ensure axis edges (spines) are visible when requested.
    """
    # Apply style first so our rcParams override it.
    if style:
        try:
            plt.style.use(style)
        except OSError:
            pass

    mpl.rcParams.update(
        {
            "figure.figsize": (12, 8),
            "font.size": 22,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "serif"],
            "axes.titlesize": 30,
            "axes.labelsize": 22,
            "axes.edgecolor": spine_color,  # updated
            "axes.linewidth": spine_linewidth,  # ensure base linewidth
            "axes.facecolor": "none",
            "figure.facecolor": "none",
            "grid.color": "#cccccc",
            "grid.alpha": 0.4,
            "savefig.format": "svg",
            "svg.fonttype": "none",
            "savefig.transparent": True,
            "figure.autolayout": False,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "figure.dpi": 300,
            "savefig.dpi": 300,
        }
    )
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(
        color=["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]
    )

    diagnostic_obj.plot_coefficients()
    fig = plt.gcf()
    fig.patch.set_alpha(0)

    for ax in fig.axes:
        ax.set_facecolor("none")
        if remove_titles:
            ax.set_title("")
        ax.grid(True)
        ax.tick_params(axis="both", labelsize=20)

        # Explicit label font sizes
        if ax.get_xlabel():
            ax.xaxis.label.set_size(22)
        if ax.get_ylabel():
            ax.yaxis.label.set_size(22)

        if not show_axes:
            # Hide all spines
            for spine in ax.spines.values():
                spine.set_visible(False)
            # Only remove ticks & labels if user disables ticks
            if not show_ticks:
                ax.tick_params(
                    left=False,
                    right=False,
                    top=False,
                    bottom=False,
                    labelleft=True,
                    labelright=False,
                    labeltop=False,
                    labelbottom=True,
                )
                # ax.set_xlabel("")
                # ax.set_ylabel("")
                ax.grid(False)
        else:
            # Explicit spine control
            for spine in ax.spines.values():
                if show_axis_edges:
                    spine.set_visible(True)
                    spine.set_linewidth(spine_linewidth)
                    spine.set_edgecolor(spine_color)
                else:
                    spine.set_visible(False)
            # If axes shown but ticks suppressed
            if not show_ticks:
                ax.tick_params(
                    left=False,
                    right=False,
                    top=False,
                    bottom=False,
                    labelleft=True,
                    labelright=False,
                    labeltop=False,
                    labelbottom=True,
                )
        # When both axes and their edges are shown, reinforce tick styling
        if show_axes and show_axis_edges and show_ticks:
            ax.tick_params(color=spine_color, width=spine_linewidth)
            for tick in ax.get_xticklines() + ax.get_yticklines():
                tick.set_color(spine_color)
                tick.set_markeredgewidth(spine_linewidth)
        # Final safeguard: if ticks disabled globally
        if not show_ticks:
            # Ensure no residual tick labels (e.g., from earlier operations)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
    if remove_titles:
        # Public API for clearing suptitle
        fig.suptitle(None)

    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(filename, format="svg", transparent=True)
    print(
        f"Saved coefficient plot to {filename} "
        f"(titles removed={remove_titles}, show_axes={show_axes}, show_axis_edges={show_axis_edges and show_axes})"
    )
    return fig


def plot_betas_vs_log_lambda(
    diagnostic_obj,
    sorted_feature_names=None,
    filename=None,
    figsize=(15, 8),
    output_format="svg",
    highlight_lambda=None,
    min_label_gap_fraction=0.175,
    min_gap_from_font_tune=0.1,
    draw_label_connectors=True,
    label_fontsize=40,
):
    """
    Plot coefficient paths from an Adelie diagnostic object against log(lambda).

    Notes:
    - betas coords are (i, j):
        i -> lambda index (diagnostic_obj.state.lmdas[i])
        j -> feature index (sorted_feature_names[j])
    - All active feature paths are drawn and each path is annotated by feature name.
    - output_format controls file export format when filename is provided ("svg" or "pdf").
    - highlight_lambda (optional) draws a dashed vertical line at log(highlight_lambda).
    - min_label_gap_fraction controls vertical spacing between annotation labels.
    - draw_label_connectors draws faint connector lines from labels to their true y-values.
    - label_fontsize controls annotation font size and adaptive spacing calculation.
    """
    betas_csr = diagnostic_obj.betas.tocsr()
    lmdas = np.asarray(diagnostic_obj.state.lmdas, dtype=float)

    if betas_csr.shape[0] != lmdas.shape[0]:
        raise ValueError(
            "Mismatch between betas rows and lambda count: "
            f"{betas_csr.shape[0]} vs {lmdas.shape[0]}"
        )

    x_values = np.log(lmdas)
    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    if sorted_feature_names is None:
        sorted_feature_names = np.array(
            [
                "$x_1$",
                "$x_2$",
                "$x_3$",
                "$x_1^2$",
                "$x_1 x_2$",
                "$x_1 x_3$",
                "$x_2^2$",
                "$x_2 x_3$",
                "$x_3^2$",
                "$x_1^3$",
                "$x_1^2 x_2$",
                "$x_1^2 x_3$",
                "$x_1 x_2^2$",
                "$x_1 x_2 x_3$",
                "$x_1 x_3^2$",
                "$x_2^3$",
                "$x_2^2 x_3$",
                "$x_2 x_3^2$",
                "$x_3^3$",
                "$x_1^4$",
                "$x_1^3 x_2$",
                "$x_1^3 x_3$",
                "$x_1^2 x_2^2$",
                "$x_1^2 x_2 x_3$",
                "$x_1^2 x_3^2$",
                "$x_1 x_2^3$",
                "$x_1 x_2^2 x_3$",
                "$x_1 x_2 x_3^2$",
                "$x_1 x_3^3$",
                "$x_2^4$",
                "$x_2^3 x_3$",
                "$x_2^2 x_3^2$",
                "$x_2 x_3^3$",
                "$x_3^4$",
                "$x_1^5$",
                "$x_1^4 x_2$",
                "$x_1^4 x_3$",
                "$x_1^3 x_2^2$",
                "$x_1^3 x_2 x_3$",
                "$x_1^3 x_3^2$",
                "$x_1^2 x_2^3$",
                "$x_1^2 x_2^2 x_3$",
                "$x_1^2 x_2 x_3^2$",
                "$x_1^2 x_3^3$",
                "$x_1 x_2^4$",
                "$x_1 x_2^3 x_3$",
                "$x_1 x_2^2 x_3^2$",
                "$x_1 x_2 x_3^3$",
                "$x_1 x_3^4$",
                "$x_2^5$",
                "$x_2^4 x_3$",
                "$x_2^3 x_3^2$",
                "$x_2^2 x_3^3$",
                "$x_2 x_3^4$",
                "$x_3^5$",
            ]
        )

    active_feature_indices = np.unique(betas_csr.tocoo().col)
    feature_count = max(len(sorted_feature_names), betas_csr.shape[1], 1)
    fixed_feature_colors = mpl.colormaps["tab20"](
        np.linspace(0, 1, feature_count)
    )

    if active_feature_indices.size == 0:
        ax.set_ylabel(r"$\beta$", fontsize=34)
        ax.set_xlabel(r"$\log(\lambda)$", fontsize=34)
        ax.tick_params(axis="both", labelsize=32, top=False, right=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(True)
        ax.spines["left"].set_visible(True)
        ax.grid(alpha=0.3)

        fig.tight_layout()
        if filename is not None:
            output_format = str(output_format).lower().strip()
            if output_format not in {"svg", "pdf"}:
                raise ValueError(
                    f"Unsupported output_format '{output_format}'. Use 'svg' or 'pdf'."
                )

            out_path = Path(filename)
            if out_path.suffix.lower() != f".{output_format}":
                out_path = out_path.with_suffix(f".{output_format}")

            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, bbox_inches="tight", format=output_format)
            print(f"Saved coefficient-path plot to {out_path}")

        return fig, ax

    x_min = float(np.min(x_values))
    x_max = float(np.max(x_values))
    x_span = max(x_max - x_min, 1e-9)
    label_anchor_x = x_min

    target_label_y = []
    plotted_feature_info = []

    for feature_index in active_feature_indices:
        coef_path = betas_csr[:, feature_index].toarray().ravel()
        feature_color = (
            fixed_feature_colors[feature_index]
            if feature_index < len(fixed_feature_colors)
            else "#666666"
        )
        ax.plot(
            x_values,
            coef_path,
            linewidth=2,
            alpha=0.9,
            color=feature_color,
        )

        nonzero_idx = np.flatnonzero(np.abs(coef_path) > 0)
        end_idx = (
            nonzero_idx[-1] if nonzero_idx.size > 0 else len(coef_path) - 1
        )

        feature_name = (
            str(sorted_feature_names[feature_index])
            if feature_index < len(sorted_feature_names)
            else f"feature_{feature_index}"
        )

        target_y = float(coef_path[end_idx])
        target_label_y.append(target_y)
        plotted_feature_info.append((feature_name, feature_color, target_y))

    target_label_y = np.asarray(target_label_y, dtype=float)
    y_min_data = float(np.min(target_label_y))
    y_max_data = float(np.max(target_label_y))
    y_span = max(y_max_data - y_min_data, 1e-9)
    n_labels = int(target_label_y.size)

    # Compute a spacing floor in data units from rendered pixel height.
    fig.canvas.draw()
    axes_height_px = max(float(ax.bbox.height), 1.0)
    data_per_px = y_span / axes_height_px
    pixel_gap = max(float(label_fontsize) * 1.15, 10.0)
    min_gap_from_font = pixel_gap * data_per_px + min_gap_from_font_tune
    min_gap_from_fraction = y_span * float(min_label_gap_fraction)
    min_gap = max(min_gap_from_font, min_gap_from_fraction, 1e-9) + 0.05

    # Build a dedicated label band that can expand beyond data range.
    required_span = min_gap * max(n_labels - 1, 1)
    natural_center = float(np.mean(target_label_y))
    band_span = max(required_span, y_span)
    band_lower = natural_center - 0.5 * band_span
    band_upper = natural_center + 0.5 * band_span

    # Keep some relation to data while allowing expansion for dense clusters.
    margin = 0.15 * y_span + 0.5 * min_gap
    band_lower = min(band_lower, y_min_data - margin)
    band_upper = max(band_upper, y_max_data + margin)
    if band_upper <= band_lower:
        band_upper = band_lower + max(min_gap, 1.0)

    if n_labels == 1:
        adjusted_label_y = np.array(
            [np.clip(target_label_y[0], band_lower, band_upper)],
            dtype=float,
        )
    else:
        sorted_indices = np.argsort(target_label_y)
        sorted_targets = target_label_y[sorted_indices]
        adjusted_sorted = np.empty_like(sorted_targets)

        # Forward pass: enforce minimum vertical separation.
        adjusted_sorted[0] = max(sorted_targets[0], band_lower)
        for idx in range(1, n_labels):
            adjusted_sorted[idx] = max(
                sorted_targets[idx], adjusted_sorted[idx - 1] + min_gap
            )

        # If top exceeds the band, shift all down once.
        overflow = adjusted_sorted[-1] - band_upper
        if overflow > 0:
            adjusted_sorted -= overflow

        # Backward pass: keep both gap and lower-bound validity.
        adjusted_sorted[0] = max(adjusted_sorted[0], band_lower)
        for idx in range(1, n_labels):
            adjusted_sorted[idx] = max(
                adjusted_sorted[idx], adjusted_sorted[idx - 1] + min_gap
            )

        adjusted_sorted = np.clip(adjusted_sorted, band_lower, band_upper)
        adjusted_label_y = np.empty_like(target_label_y)
        adjusted_label_y[sorted_indices] = adjusted_sorted

    label_band_low = float(np.min(adjusted_label_y))
    label_band_high = float(np.max(adjusted_label_y))
    display_span = max(y_max_data, label_band_high) - min(
        y_min_data, label_band_low
    )
    display_span = max(display_span, 1e-9)
    y_padding = max(0.08 * display_span, 0.6 * min_gap)
    ax.set_ylim(
        min(y_min_data, label_band_low) - y_padding,
        max(y_max_data, label_band_high) + y_padding,
    )

    # Increase left label margin when many features are active.
    label_offset = 0.15 + min(0.45, 0.015 * n_labels)
    label_x = x_min - label_offset

    annotation_objects = []
    connector_objects = []
    for idx, (feature_name, feature_color, target_y) in enumerate(
        plotted_feature_info
    ):
        label_y = float(adjusted_label_y[idx])

        text_obj = ax.annotate(
            feature_name,
            xy=(label_x, label_y),
            xycoords="data",
            color=feature_color,
            fontsize=label_fontsize,
            va="center",
            ha="right",
            annotation_clip=False,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.72,
                "pad": 0.0,
            },
        )
        annotation_objects.append(text_obj)

        if draw_label_connectors:
            (connector_line,) = ax.plot(
                [label_x + 0.015 * x_span, label_anchor_x],
                [label_y, target_y],
                color=feature_color,
                alpha=0.5,
                linewidth=1.0,
                linestyle=":",
                zorder=2,
            )
            connector_objects.append(connector_line)
        else:
            connector_objects.append(None)

    # Second-pass collision resolution in display coordinates.
    # This makes spacing robust for large label_fontsize values (e.g. 40).
    if len(annotation_objects) > 1:
        fig.canvas.draw()
        axes_bbox = ax.bbox
        min_gap_px = max(float(label_fontsize) * 1.28, 16.0)
        edge_margin_px = max(float(label_fontsize) * 0.32, 6.0)

        current_label_y = np.array(
            [ann.get_position()[1] for ann in annotation_objects]
        )
        label_y_px = np.array(
            [
                ax.transData.transform((label_x, y_val))[1]
                for y_val in current_label_y
            ],
            dtype=float,
        )

        order = np.argsort(label_y_px)
        sorted_px = label_y_px[order].copy()

        for idx in range(1, sorted_px.size):
            sorted_px[idx] = max(
                sorted_px[idx], sorted_px[idx - 1] + min_gap_px
            )

        min_allowed = float(axes_bbox.y0 + edge_margin_px)
        max_allowed = float(axes_bbox.y1 - edge_margin_px)
        if sorted_px[-1] > max_allowed:
            sorted_px -= sorted_px[-1] - max_allowed
        if sorted_px[0] < min_allowed:
            sorted_px += min_allowed - sorted_px[0]

        adjusted_px = np.empty_like(label_y_px)
        adjusted_px[order] = sorted_px
        adjusted_data_y = ax.transData.inverted().transform(
            np.column_stack([np.full_like(adjusted_px, label_x), adjusted_px])
        )[:, 1]

        for idx, ann in enumerate(annotation_objects):
            ann.set_position((label_x, float(adjusted_data_y[idx])))
            if draw_label_connectors and connector_objects[idx] is not None:
                connector_objects[idx].set_data(
                    [label_x + 0.015 * x_span, label_anchor_x],
                    [float(adjusted_data_y[idx]), plotted_feature_info[idx][2]],
                )

    x_left_margin = max(0.6575, label_offset + 0.16)
    ax.set_xlim(x_min - x_left_margin, x_max)

    ax.set_ylabel(r"$\beta$", fontsize=34)
    ax.set_xlabel(r"$\log(\lambda)$", fontsize=34)
    ax.tick_params(axis="both", labelsize=32, top=False, right=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)

    if highlight_lambda is not None:
        if highlight_lambda <= 0:
            raise ValueError("highlight_lambda must be positive.")
        x_highlight = np.log(highlight_lambda)
        ax.axvline(
            x_highlight,
            linestyle="--",
            color="#7570b3",
            linewidth=1.8,
            alpha=0.9,
        )
        ax.annotate(
            f"$\\lambda^*=${highlight_lambda:g}",
            xy=(x_highlight, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(6, -8),
            textcoords="offset points",
            color="#7570b3",
            fontsize=40,
            va="top",
            ha="left",
        )

    ax.grid(alpha=0.3)

    fig.tight_layout()
    if filename is not None:
        output_format = str(output_format).lower().strip()
        if output_format not in {"svg", "pdf"}:
            raise ValueError(
                f"Unsupported output_format '{output_format}'. Use 'svg' or 'pdf'."
            )

        out_path = Path(filename)
        if out_path.suffix.lower() != f".{output_format}":
            out_path = out_path.with_suffix(f".{output_format}")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", format=output_format)
        print(f"Saved coefficient-path plot to {out_path}")

    return fig, ax


# %%
# - Functions to adjust ArviZ plots
def plot_trace_with_spacing(
    results, figsize=(12, 36), hspace=0.65, wspace=0.25, **kwargs
):
    """
    Wrapper around az.plot_trace that enlarges vertical (hspace) and horizontal (wspace)
    gaps to prevent title overlap.
    """
    axes = az.plot_trace(results, backend_kwargs={"figsize": figsize}, **kwargs)
    fig = axes.flat[0].figure
    fig.subplots_adjust(hspace=hspace, wspace=wspace)
    return axes


def export_trace_density_plot(
    results,
    var_name,
    filename,
    var_name_latex=None,
    include_density=True,
    style="seaborn-v0_8",
    figsize=None,
    dpi=300,
    remove_titles=False,
    show_grid=True,
    show_edges_trace_plot=False,
    spine_color="#333333",
    spine_linewidth=1.5,
    bottom_edge_only=True,  # NEW: show only bottom axis line
    line_colors=["#7570b3"],  # NEW: custom colors for chain traces
    preserve_linestyles=True,  # NEW: keep original ArviZ line styles
    title_fontsize=30,
    tick_labelsize=20,
):
    """
    Export trace (and optionally density) for a single variable from an ArviZ InferenceData.
    Added:
        line_colors: list of color hex/rgb strings applied cyclically to chain trace lines
        preserve_linestyles: if True, retain original linestyle from ArviZ (else force solid)
    """
    # Defensive close
    plt.close("all")

    # Apply style first
    if style:
        try:
            plt.style.use(style)
        except OSError:
            pass

    # Core style alignment
    mpl.rcParams.update(
        {
            "figure.figsize": (12, 12) if figsize is None else figsize,
            "font.size": 22,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "serif"],
            "axes.titlesize": 30,
            "axes.labelsize": 22,
            "axes.edgecolor": spine_color,
            "axes.linewidth": spine_linewidth,
            "axes.facecolor": "none",
            "figure.facecolor": "none",
            "grid.color": "#cccccc",
            "grid.alpha": 0.4,
            "savefig.format": "svg",
            "svg.fonttype": "none",
            "savefig.transparent": True,
            "figure.dpi": dpi,
            "savefig.dpi": dpi,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
        }
    )

    # Ensure name exists (gives clearer error than deep ArviZ internals)
    try:
        _ = az.extract(results, var_names=[var_name])
    except Exception as e:
        raise ValueError(
            f"Variable '{var_name}' not found in results. Original error: {e}"
        )

    latex_title = (
        var_name_latex if var_name_latex and not remove_titles else None
    )  # NEW

    if include_density:
        axes = az.plot_trace(
            results,
            var_names=[var_name],
            backend_kwargs={"figsize": mpl.rcParams["figure.figsize"]},
        )
        fig = axes.flat[0].figure
        fig.patch.set_alpha(0)
        # --- NEW: recolor only chain traces if requested ---
        if line_colors:
            color_idx = 0
            for ax in fig.axes:
                for ln in ax.get_lines():
                    if ln.get_label().startswith("chain"):
                        ln.set_color(line_colors[color_idx % len(line_colors)])
                        color_idx += 1
        for ax in fig.axes:
            ax.set_facecolor("none")
            if show_grid:
                ax.grid(True, alpha=0.4)
            else:
                ax.grid(False)
            if remove_titles:
                ax.set_title("")
            else:
                if latex_title:
                    ax.set_title(
                        latex_title, fontsize=title_fontsize
                    )  # NEW override
                else:
                    ax.title.set_fontsize(title_fontsize)
            ax.tick_params(axis="both", labelsize=tick_labelsize)
            # Make spines subtle / configurable
            for spine in ax.spines.values():
                spine.set_linewidth(spine_linewidth)
                spine.set_edgecolor(spine_color)
            # --- NEW logic for bottom-only edge ---
            if bottom_edge_only:
                for pos, spine in ax.spines.items():
                    spine.set_visible(pos == "bottom")
                ax.tick_params(
                    axis="x", bottom=True, top=False, labelbottom=True
                )
                ax.tick_params(
                    axis="y", left=False, right=False, labelleft=False
                )
            else:
                if not show_edges_trace_plot:
                    # keep prior behavior (no full box) but retain ticks (unless already modified)
                    pass
        if remove_titles:
            fig.suptitle(None)
    else:
        # Still leverage ArviZ for consistency, then rebuild minimal figure
        raw_axes = az.plot_trace(
            results,
            var_names=[var_name],
            backend_kwargs={
                "figsize": (
                    (mpl.rcParams["figure.figsize"][0] / 2),
                    mpl.rcParams["figure.figsize"][1],
                )
            },
        )
        trace_ax = raw_axes[0, 0]
        fig, ax = plt.subplots(1, 1, figsize=mpl.rcParams["figure.figsize"])
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")

        # --- MODIFIED: preserve linestyle + optional custom colors ---
        line_idx = 0
        for line in trace_ax.get_lines():
            color_to_use = (
                line_colors[line_idx % len(line_colors)]
                if line_colors
                else line.get_color()
            )
            linestyle_to_use = (
                line.get_linestyle() if preserve_linestyles else "-"
            )
            ax.plot(
                line.get_xdata(),
                line.get_ydata(),
                color=color_to_use,
                alpha=line.get_alpha() or 1.0,
                linewidth=1.5 * line.get_linewidth(),
                linestyle=linestyle_to_use,  # NEW
            )
            line_idx += 1
        # Copy labels / title / limits
        if not remove_titles:
            if latex_title:
                ax.set_title(latex_title, fontsize=title_fontsize)  # NEW
            else:
                ax.set_title(trace_ax.get_title(), fontsize=title_fontsize)
        ax.tick_params(axis="both", labelsize=tick_labelsize)
        if show_grid:
            ax.grid(True, alpha=0.4)
        if not show_edges_trace_plot:
            ax.tick_params(
                left=False,
                right=False,
                top=False,
                bottom=True,
                labelleft=False,
                labelright=False,
                labeltop=False,
                labelbottom=True,
            )
        # ...existing code styling spines...
        # --- NEW logic for bottom-only edge (takes precedence) ---
        if bottom_edge_only:
            for pos, spine in ax.spines.items():
                spine.set_visible(pos == "bottom")
            ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True)
            ax.tick_params(axis="y", left=False, right=False, labelleft=False)

    # Output path handling
    out_path = Path(filename)
    if out_path.suffix.lower() != ".svg":
        out_path = out_path.with_suffix(".svg")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    fig.savefig(out_path, format="svg", transparent=True)
    print(
        f"Saved {'trace+density' if include_density else 'trace-only'} plot for '{var_name}' -> {out_path}"
    )
    return fig


def export_density_plot(
    results,
    var_name,
    filename,
    var_name_latex=None,
    style="seaborn-v0_8",
    figsize=(12, 12),
    shade=0.1,
    hdi_prob=0.90,
    remove_titles=False,
    show_grid=True,
    spine_color="#333333",
    spine_linewidth=1.5,
    bottom_edge_only=True,
    alpha_fill=0.6,
    linewidth=2.0,
):
    """
    Export a single posterior density (az.plot_density) for one variable to an SVG.
    No looping over multiple variables.
    """
    # Close prior figures
    plt.close("all")

    if style:
        try:
            plt.style.use(style)
        except OSError:
            pass

    mpl.rcParams.update(
        {
            "font.size": 22,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "serif"],
            "axes.titlesize": 28,
            "axes.labelsize": 22,
            "axes.edgecolor": spine_color,
            "axes.linewidth": spine_linewidth,
            "axes.facecolor": "none",
            "figure.facecolor": "none",
            "svg.fonttype": "none",
            "savefig.transparent": True,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
        }
    )

    posterior = getattr(results, "posterior", None)
    if posterior is None or var_name not in posterior.data_vars:
        raise ValueError(
            f"Variable '{var_name}' not found in results.posterior."
        )

    axes = az.plot_density(
        results,
        var_names=[var_name],
        shade=shade,
        hdi_prob=hdi_prob,
        backend_kwargs={"figsize": figsize},
    )
    fig = axes.ravel()[0].figure
    fig.patch.set_alpha(0)

    ax = fig.axes[0]
    ax.set_facecolor("none")
    if show_grid:
        ax.grid(True, alpha=0.4)
    else:
        ax.grid(False)

    # Title handling
    if remove_titles:
        ax.set_title("")
        fig.suptitle(None)
    else:
        if var_name_latex:
            ax.set_title(var_name_latex)

    # Re-style lines / fills
    for line in ax.get_lines():
        line.set_color("#7570b3")
        line.set_linewidth(linewidth)
    for coll in ax.collections:
        try:
            coll.set_alpha(alpha_fill)
            coll.set_facecolor("#6CB7FF")
            coll.set_edgecolor("#6CB7FF")
        except Exception:
            pass

    for spine in ax.spines.values():
        spine.set_linewidth(spine_linewidth)
        spine.set_edgecolor(spine_color)

    if bottom_edge_only:
        for pos, spine in ax.spines.items():
            spine.set_visible(pos == "bottom")
        ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True)
        ax.tick_params(axis="y", left=False, right=False, labelleft=False)

    out_path = Path(filename)
    if out_path.suffix.lower() != ".svg":
        out_path = out_path.with_suffix(".svg")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, format="svg", transparent=True)
    print(f"Saved density plot for '{var_name}' -> {out_path}")
    return fig


def export_posterior_plot(
    results,
    var_name,
    filename,
    var_name_latex=None,
    style="seaborn-v0_8",
    figsize=(12, 12),
    hdi_prob=0.90,
    remove_titles=False,
    show_grid=True,
    spine_color="#333333",
    spine_linewidth=1.5,
    bottom_edge_only=True,
    posterior_color="#7570b3",
    alpha_fill=0.6,
    linewidth=2.0,
    point_estimate="mean",
    ref_val=None,
    rope=None,
    title_fontsize=32,
    tick_labelsize=24,
    annotation_fontsize=24,
):
    """
    Export a posterior plot (az.plot_posterior) for a single variable with styling similar
    to export_density_plot.
    """
    plt.close("all")

    if style:
        try:
            plt.style.use(style)
        except OSError:
            pass

    mpl.rcParams.update(
        {
            "font.size": 24,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "serif"],
            "axes.titlesize": 28,
            "axes.labelsize": 24,
            "axes.edgecolor": spine_color,
            "axes.linewidth": spine_linewidth,
            "axes.facecolor": "none",
            "figure.facecolor": "none",
            "svg.fonttype": "none",
            "savefig.transparent": True,
            "xtick.labelsize": 22,
            "ytick.labelsize": 22,
        }
    )

    posterior = getattr(results, "posterior", None)
    if posterior is None or var_name not in posterior.data_vars:
        raise ValueError(
            f"Variable '{var_name}' not found in results.posterior."
        )

    axes = az.plot_posterior(
        results,
        var_names=[var_name],
        hdi_prob=hdi_prob,
        point_estimate=point_estimate,
        ref_val=ref_val,
        rope=rope,
        backend_kwargs={"figsize": figsize},
    )
    # --- NEW: normalize axes return (single Axes vs array-like) ---
    if isinstance(axes, mpl.axes.Axes):
        ax = axes
    else:
        try:
            ax = np.asarray(axes).ravel()[0]
        except Exception:
            if isinstance(axes, (list, tuple)) and axes:
                ax = axes[0]
            else:
                raise TypeError(
                    f"Unexpected return type from az.plot_posterior: {type(axes)}"
                )
    # --- END NEW ---
    for txt in ax.texts:
        label = txt.get_text()
        if point_estimate == "mean" and label.startswith("mean="):
            txt.set_text(label.replace("mean=", r"$\mu$=", 1))
            label = txt.get_text()
        if label.endswith(" HDI"):
            txt.set_text(label.replace(" HDI", ""))
        # Keep posterior annotation text (mu, HDI label, interval values) legible.
        txt.set_fontsize(annotation_fontsize)
    fig = ax.figure
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    if remove_titles:
        ax.set_title("")
        fig.suptitle(None)
    else:
        if var_name_latex:
            ax.set_title(var_name_latex, fontsize=title_fontsize)
    ax.tick_params(axis="both", labelsize=tick_labelsize)
    if show_grid:
        ax.grid(True, alpha=0.4)
    else:
        ax.grid(False)
    for line in ax.get_lines():
        if (
            line.get_linestyle() in ("-", "--", "-.", ":")
            and line.get_xdata().size > 5
        ):
            line.set_color(posterior_color)
            line.set_linewidth(linewidth)
    for coll in ax.collections:
        try:
            coll.set_alpha(alpha_fill)
            coll.set_facecolor(posterior_color)
            coll.set_edgecolor(posterior_color)
        except Exception:
            pass
    for spine in ax.spines.values():
        spine.set_linewidth(spine_linewidth)
        spine.set_edgecolor(spine_color)
    if bottom_edge_only:
        for pos, spine in ax.spines.items():
            spine.set_visible(pos == "bottom")
        ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True)
        ax.tick_params(axis="y", left=False, right=False, labelleft=False)

    out_path = Path(filename)
    if out_path.suffix.lower() != ".svg":
        out_path = out_path.with_suffix(".svg")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    fig.savefig(out_path, format="svg", transparent=True)
    print(f"Saved posterior plot for '{var_name}' -> {out_path}")
    return fig


def export_prior_plot(
    model_bayesian,
    var_name,
    filename,
    hdi_prob=0.90,
    var_name_latex=None,
    style="seaborn-v0_8",
    figsize=(12, 12),
    remove_titles=False,
    show_grid=True,
    spine_color="#333333",
    spine_linewidth=1.5,
    bottom_edge_only=True,
    prior_color="#51D7C1",
    alpha_fill=0.6,
    linewidth=1.5,
    title_fontsize=32,
    tick_labelsize=24,
):
    """
    Export a single prior density from model_bayesian.plot_priors similar to export_density_plot.

    Steps:
      1. Call model_bayesian.plot_priors(hdi_prob).
      2. Find axis whose title contains var_name (case-insensitive); fallback to first.
      3. Delete all other axes.
      4. Restyle target axis (color, fill, spines, grid, bottom edge only).
      5. Optionally replace title with var_name_latex.
      6. Save transparent SVG.

    Tolerant to different internal implementations of plot_priors.
    """
    plt.close("all")

    if style:
        try:
            plt.style.use(style)
        except OSError:
            pass

    mpl.rcParams.update(
        {
            "font.size": 24,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "serif"],
            "axes.titlesize": 28,
            "axes.labelsize": 24,
            "axes.edgecolor": spine_color,
            "axes.linewidth": spine_linewidth,
            "axes.facecolor": "none",
            "figure.facecolor": "none",
            "svg.fonttype": "none",
            "savefig.transparent": True,
            "xtick.labelsize": 22,
            "ytick.labelsize": 22,
        }
    )

    # --- Render all priors (full grid) and isolate the requested variable ---
    model_bayesian.plot_priors(hdi_prob=hdi_prob)
    full_fig = plt.gcf()
    axes = full_fig.axes
    if not axes:
        raise RuntimeError("No axes generated by model_bayesian.plot_priors().")

    # Locate target axis (case-insensitive match on its title)
    target_ax = None
    search = var_name.lower()
    for cand in axes:
        if search in (cand.get_title() or "").lower():
            target_ax = cand
            break
    if target_ax is None:
        target_ax = axes[0]  # fallback

    # Extract line data before discarding the original multi-axes figure
    density_lines = []  # (x, y)
    other_lines = []  # (x, y, ls, lw)
    for line in target_ax.get_lines():
        xdata = line.get_xdata()
        ydata = line.get_ydata()
        ls = line.get_linestyle()
        lw = line.get_linewidth()
        if xdata.size > 5 and ls in ("-", "--", "-.", ":"):
            density_lines.append((xdata.copy(), ydata.copy()))
        else:
            other_lines.append((xdata.copy(), ydata.copy(), ls, lw))

    # (Optional) axis limits to preserve original scaling
    xlim = target_ax.get_xlim()
    ylim = target_ax.get_ylim()

    # Close the original composite figure to avoid keeping the grid layout
    plt.close(full_fig)

    # --- Create a brand-new independent figure with a single axis ---
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    # Replot density lines
    for x, y in density_lines:
        ax.plot(x, y, color=prior_color, linewidth=linewidth)

    # Add a filled region under the first density line if requested
    if alpha_fill > 0 and density_lines:
        x0, y0 = density_lines[0]
        try:
            ax.fill_between(
                x0, y0, 0, color=prior_color, alpha=alpha_fill, linewidth=0
            )
        except (ValueError, TypeError, RuntimeError):
            # Safely ignore fill issues (e.g., malformed data)
            pass

    # Replot other (e.g., vertical) lines
    for x, y, ls, lw in other_lines:
        ax.plot(x, y, linestyle=ls, linewidth=lw, color=prior_color)

    # Grid / titles
    if show_grid:
        ax.grid(True, alpha=0.4)
    else:
        ax.grid(False)

    if remove_titles:
        ax.set_title("")
    else:
        if var_name_latex:
            ax.set_title(var_name_latex, fontsize=title_fontsize)
    ax.tick_params(axis="both", labelsize=tick_labelsize)

    # Restore limits (helpful if replot changed autoscale)
    try:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
    except (ValueError, RuntimeError):
        # Ignore if limits invalid (e.g., empty data)
        pass

    # Style spines
    for spine in ax.spines.values():
        spine.set_linewidth(spine_linewidth)
        spine.set_edgecolor(spine_color)

    if bottom_edge_only:
        for pos, spine in ax.spines.items():
            spine.set_visible(pos == "bottom")
        ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True)
        ax.tick_params(axis="y", left=False, right=False, labelleft=False)

    # Save independent SVG
    out_path = Path(filename)
    if out_path.suffix.lower() != ".svg":
        out_path = out_path.with_suffix(".svg")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig.tight_layout()
    fig.savefig(out_path, format="svg", transparent=True)
    print(
        f"Saved prior plot for '{var_name}' -> {out_path} (independent figure)"
    )
    return fig


# %%
# - New standalone utilities: align trace colors with plot_betas_vs_log_lambda
def _canonical_feature_name(name: str) -> str:
    """
    Normalize variable names so raw names like 'x1^3 x3' match LaTeX-ish
    labels like '$x_1^3 x_3$'.
    """
    normalized = str(name)
    for token in ["$", "\\", "{", "}", "_"]:
        normalized = normalized.replace(token, "")
    normalized = normalized.replace(" ", "").lower()
    return normalized


def _default_sorted_feature_names_from_betas_plot():
    """
    Returns the same default feature-label ordering used by plot_betas_vs_log_lambda.
    This preserves index -> color mapping consistency.
    """
    return np.array(
        [
            "$x_1$",
            "$x_2$",
            "$x_3$",
            "$x_1^2$",
            "$x_1 x_2$",
            "$x_1 x_3$",
            "$x_2^2$",
            "$x_2 x_3$",
            "$x_3^2$",
            "$x_1^3$",
            "$x_1^2 x_2$",
            "$x_1^2 x_3$",
            "$x_1 x_2^2$",
            "$x_1 x_2 x_3$",
            "$x_1 x_3^2$",
            "$x_2^3$",
            "$x_2^2 x_3$",
            "$x_2 x_3^2$",
            "$x_3^3$",
            "$x_1^4$",
            "$x_1^3 x_2$",
            "$x_1^3 x_3$",
            "$x_1^2 x_2^2$",
            "$x_1^2 x_2 x_3$",
            "$x_1^2 x_3^2$",
            "$x_1 x_2^3$",
            "$x_1 x_2^2 x_3$",
            "$x_1 x_2 x_3^2$",
            "$x_1 x_3^3$",
            "$x_2^4$",
            "$x_2^3 x_3$",
            "$x_2^2 x_3^2$",
            "$x_2 x_3^3$",
            "$x_3^4$",
            "$x_1^5$",
            "$x_1^4 x_2$",
            "$x_1^4 x_3$",
            "$x_1^3 x_2^2$",
            "$x_1^3 x_2 x_3$",
            "$x_1^3 x_3^2$",
            "$x_1^2 x_2^3$",
            "$x_1^2 x_2^2 x_3$",
            "$x_1^2 x_2 x_3^2$",
            "$x_1^2 x_3^3$",
            "$x_1 x_2^4$",
            "$x_1 x_2^3 x_3$",
            "$x_1 x_2^2 x_3^2$",
            "$x_1 x_2 x_3^3$",
            "$x_1 x_3^4$",
            "$x_2^5$",
            "$x_2^4 x_3$",
            "$x_2^3 x_3^2$",
            "$x_2^2 x_3^3$",
            "$x_2 x_3^4$",
            "$x_3^5$",
        ]
    )


def build_trace_line_color_map_from_betas_plot(
    diagnostic_obj,
    sorted_feature_names=None,
):
    """
    Build {raw_feature_name -> hex_color} using the exact color assignment logic
    from plot_betas_vs_log_lambda.

    Parameters
    ----------
    diagnostic_obj : adelie diagnostic object (e.g., dg_init / dg_final)
        Used to reproduce feature_count = max(len(names), betas_cols, 1).
    sorted_feature_names : sequence of str, optional
        Feature-name order used in plot_betas_vs_log_lambda. If None, uses
        the same default list as that function.
    """
    if sorted_feature_names is None:
        sorted_feature_names = _default_sorted_feature_names_from_betas_plot()

    betas_csr = diagnostic_obj.betas.tocsr()
    feature_count = max(len(sorted_feature_names), betas_csr.shape[1], 1)
    fixed_feature_colors = mpl.colormaps["tab20"](
        np.linspace(0, 1, feature_count)
    )

    color_map = {}
    for feature_index, feature_name in enumerate(sorted_feature_names):
        color_map[_canonical_feature_name(feature_name)] = mcolors.to_hex(
            fixed_feature_colors[feature_index]
        )

    return color_map


def export_trace_plots_with_betas_matched_colors(
    results,
    raw_var_names,
    var_names_to_plot,
    diagnostic_obj,
    output_dir="figures/diagnostics-aizawa/traces",
    sorted_feature_names=None,
    fallback_color="#7570b3",
):
    """
    Export trace-only plots where each variable color matches the color used in
    plot_betas_vs_log_lambda. Variables absent from that plot (e.g. sigma,
    Intercept) use fallback_color.
    """
    betas_color_map = build_trace_line_color_map_from_betas_plot(
        diagnostic_obj=diagnostic_obj,
        sorted_feature_names=sorted_feature_names,
    )

    for raw_name, latex_name in zip(raw_var_names, var_names_to_plot):
        if raw_name not in results.posterior.data_vars:
            continue

        key = _canonical_feature_name(raw_name)
        matched_color = betas_color_map.get(key, fallback_color)

        export_trace_density_plot(
            results,
            figsize=(8, 6),
            var_name=raw_name,
            filename=f"{output_dir}/{raw_name}.svg",
            var_name_latex=latex_name,
            include_density=False,
            line_colors=[matched_color],
        )


def export_density_plot_colored(
    results,
    var_name,
    filename,
    var_name_latex=None,
    style="seaborn-v0_8",
    figsize=(8, 6),
    shade=0.1,
    hdi_prob=0.90,
    remove_titles=False,
    show_grid=True,
    spine_color="#333333",
    spine_linewidth=1.5,
    bottom_edge_only=True,
    alpha_fill=0.6,
    linewidth=2.0,
    density_line_color="#7570b3",
    density_fill_color="#6CB7FF",
    title_fontsize=32,
    tick_labelsize=24,
):
    """
    New standalone variant of export_density_plot with explicit color controls.
    """
    plt.close("all")

    if style:
        try:
            plt.style.use(style)
        except OSError:
            pass

    mpl.rcParams.update(
        {
            "font.size": 24,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "serif"],
            "axes.titlesize": 28,
            "axes.labelsize": 24,
            "axes.edgecolor": spine_color,
            "axes.linewidth": spine_linewidth,
            "axes.facecolor": "none",
            "figure.facecolor": "none",
            "svg.fonttype": "none",
            "savefig.transparent": True,
            "xtick.labelsize": 22,
            "ytick.labelsize": 22,
        }
    )

    posterior = getattr(results, "posterior", None)
    if posterior is None or var_name not in posterior.data_vars:
        raise ValueError(
            f"Variable '{var_name}' not found in results.posterior."
        )

    axes = az.plot_density(
        results,
        var_names=[var_name],
        shade=shade,
        hdi_prob=hdi_prob,
        backend_kwargs={"figsize": figsize},
    )
    fig = axes.ravel()[0].figure
    fig.patch.set_alpha(0)

    ax = fig.axes[0]
    ax.set_facecolor("none")
    if show_grid:
        ax.grid(True, alpha=0.4)
    else:
        ax.grid(False)

    if remove_titles:
        ax.set_title("")
        fig.suptitle(None)
    else:
        if var_name_latex:
            ax.set_title(var_name_latex, fontsize=title_fontsize)
    ax.tick_params(axis="both", labelsize=tick_labelsize)

    for line in ax.get_lines():
        line.set_color(density_line_color)
        line.set_linewidth(linewidth)
    for coll in ax.collections:
        try:
            coll.set_alpha(alpha_fill)
            coll.set_facecolor(density_fill_color)
            coll.set_edgecolor(density_fill_color)
        except Exception:
            pass

    for spine in ax.spines.values():
        spine.set_linewidth(spine_linewidth)
        spine.set_edgecolor(spine_color)

    if bottom_edge_only:
        for pos, spine in ax.spines.items():
            spine.set_visible(pos == "bottom")
        ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True)
        ax.tick_params(axis="y", left=False, right=False, labelleft=False)

    out_path = Path(filename)
    if out_path.suffix.lower() != ".svg":
        out_path = out_path.with_suffix(".svg")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, format="svg", transparent=True)
    print(f"Saved colored density plot for '{var_name}' -> {out_path}")
    return fig


def export_density_posterior_prior_with_betas_matched_colors(
    results,
    model_bayesian,
    raw_var_names,
    var_names_to_plot,
    diagnostic_obj,
    density_output_dir="figures/diagnostics-aizawa/density",
    posterior_output_dir="figures/diagnostics-aizawa/posterior",
    prior_output_dir="figures/diagnostics-aizawa/priors",
    trace_output_dir=None,
    sorted_feature_names=None,
    fallback_color="#7570b3",
    title_fontsize=50,
    tick_labelsize=34,
    annotation_fontsize=36,
    shared_figsize=None,
    trace_figsize=(8, 6),
    trace_title_fontsize=None,
    trace_tick_labelsize=None,
    density_title_fontsize=None,
    density_tick_labelsize=None,
    posterior_title_fontsize=None,
    posterior_tick_labelsize=None,
    posterior_annotation_fontsize=None,
    prior_title_fontsize=None,
    prior_tick_labelsize=None,
):
    """
    Export trace/density/posterior/prior plots with per-variable colors matched to
    plot_betas_vs_log_lambda feature-path colors.
    Variables absent in beta paths (e.g. sigma, Intercept) use fallback_color.

    Font controls:
      - title_fontsize / tick_labelsize / annotation_fontsize are global defaults.
      - Use trace_* / density_* / posterior_* / prior_* kwargs to override per
        plot type.
      - shared_figsize (e.g. (8, 6)) applies one figure size to all four plot
        types (trace, density, posterior, prior).
      - Set trace_output_dir to a path to export trace plots from this function.
    """
    resolved_trace_figsize = (
        shared_figsize if shared_figsize is not None else trace_figsize
    )
    resolved_density_figsize = (
        shared_figsize if shared_figsize is not None else (8, 6)
    )
    resolved_posterior_figsize = (
        shared_figsize if shared_figsize is not None else (12, 12)
    )
    resolved_prior_figsize = (
        shared_figsize if shared_figsize is not None else (12, 12)
    )

    trace_title_fs = (
        trace_title_fontsize
        if trace_title_fontsize is not None
        else title_fontsize
    )
    trace_tick_fs = (
        trace_tick_labelsize
        if trace_tick_labelsize is not None
        else tick_labelsize
    )
    density_title_fs = (
        density_title_fontsize
        if density_title_fontsize is not None
        else title_fontsize
    )
    density_tick_fs = (
        density_tick_labelsize
        if density_tick_labelsize is not None
        else tick_labelsize
    )
    posterior_title_fs = (
        posterior_title_fontsize
        if posterior_title_fontsize is not None
        else title_fontsize
    )
    posterior_tick_fs = (
        posterior_tick_labelsize
        if posterior_tick_labelsize is not None
        else tick_labelsize
    )
    posterior_annot_fs = (
        posterior_annotation_fontsize
        if posterior_annotation_fontsize is not None
        else annotation_fontsize
    )
    prior_title_fs = (
        prior_title_fontsize
        if prior_title_fontsize is not None
        else title_fontsize
    )
    prior_tick_fs = (
        prior_tick_labelsize
        if prior_tick_labelsize is not None
        else tick_labelsize
    )

    betas_color_map = build_trace_line_color_map_from_betas_plot(
        diagnostic_obj=diagnostic_obj,
        sorted_feature_names=sorted_feature_names,
    )

    for raw_name, latex_name in zip(raw_var_names, var_names_to_plot):
        if raw_name not in results.posterior.data_vars:
            continue

        key = _canonical_feature_name(raw_name)
        matched_color = betas_color_map.get(key, fallback_color)

        if trace_output_dir:
            export_trace_density_plot(
                results,
                figsize=resolved_trace_figsize,
                var_name=raw_name,
                filename=f"{trace_output_dir}/{raw_name}.svg",
                var_name_latex=latex_name,
                include_density=False,
                line_colors=[matched_color],
                title_fontsize=trace_title_fs,
                tick_labelsize=trace_tick_fs,
            )

        export_density_plot_colored(
            results,
            var_name=raw_name,
            var_name_latex=latex_name,
            filename=f"{density_output_dir}/{raw_name}.svg",
            figsize=resolved_density_figsize,
            shade=0.1,
            hdi_prob=0.90,
            density_line_color=matched_color,
            density_fill_color=matched_color,
            alpha_fill=0.35,
            title_fontsize=density_title_fs,
            tick_labelsize=density_tick_fs,
        )

        export_posterior_plot(
            results,
            var_name=raw_name,
            var_name_latex=latex_name,
            filename=f"{posterior_output_dir}/{raw_name}.svg",
            figsize=resolved_posterior_figsize,
            hdi_prob=0.90,
            posterior_color=matched_color,
            title_fontsize=posterior_title_fs,
            tick_labelsize=posterior_tick_fs,
            annotation_fontsize=posterior_annot_fs,
        )

        try:
            export_prior_plot(
                model_bayesian,
                var_name=raw_name,
                var_name_latex=latex_name,
                filename=f"{prior_output_dir}/{raw_name}.svg",
                figsize=resolved_prior_figsize,
                hdi_prob=0.90,
                prior_color=matched_color,
                alpha_fill=0.35,
                title_fontsize=prior_title_fs,
                tick_labelsize=prior_tick_fs,
            )
        except Exception as e:
            print(f"Skipping prior export for '{raw_name}' with color map: {e}")


# %%
# ! -------------------- Lorenz --------------------
initial_value_df = ags.generate_initial_value_df(
    seed=100,
    num_init_samples=100,
    x1_range=(-15, 15),
    x2_range=(-15, 15),
    x3_range=(10, 40),
    num_columns=3,
)

x_t = ags.generate_noisy_dynamical_systems(
    variable_coeff=[[-10, 10], [28, -1, -1], [1, -8 / 3]],
    variable_names=[["x1", "x2"], ["x1", "x2", "x1x3"], ["x1x2", "x3"]],
    n=7000,
    dt=0.001,
    init_conditions=initial_value_df.iloc[95].values,
    snr=49,
)

ags.plot_3d_trajectory(
    x_t,
    "Lorenz",
    show_colorbar=False,
    save_figure=True,
    show_axes=False,
    output_format="svg",
)

design_matrix = agu.build_design_matrix(
    x_t=x_t,
    dt=0.001,
    library_degree=5,
    sg_poly_order=4,
    library_type="poly",
)

ags.plot_3d_trajectory(
    design_matrix["x_filtered"],
    "Lorenz_filtered",
    show_colorbar=False,
    save_figure=True,
    show_axes=False,
    output_format="svg",
)


model_system = BayesianArgosAnalysis(
    design_matrix=design_matrix, custom_prior=True, accelerator=True
)
model_system.run(mode="straight", parallel="yes", ncpus=None, ci_level=0.90)
model_system.get_frequentist_results()
model_system.expressions_for_simulation()
model_system.results["equation_3"]["model"].plot_priors()

simulated_data = model_system.simulate(
    n=5000, dt=0.001, init_conditions=initial_value_df.iloc[95].values
)

ags.plot_3d_trajectory(
    simulated_data,
    "Lorenz_Simulated",
    show_colorbar=False,
    save_figure=True,
    show_axes=False,
    output_format="svg",
)

# %%
# ! -------------------- Thomas --------------------
initial_value_df = ags.generate_initial_value_df(
    seed=100,
    num_init_samples=100,
    x1_range=(-1, 1),
    x2_range=(-1, 1),
    x3_range=(-1, 1),
    num_columns=3,
)

x_t = ags.generate_noisy_dynamical_systems(
    variable_coeff=[[1, -0.208186], [1, -0.208186], [1, -0.208186]],
    variable_names=[["sin(x2)", "x1"], ["sin(x3)", "x2"], ["sin(x1)", "x3"]],
    n=5000,
    dt=0.01,
    init_conditions=initial_value_df.iloc[95].values,
    snr=25,
)

ags.plot_3d_trajectory(
    x_t, "Thomas", show_colorbar=False, save_figure=True, show_axes=False
)

design_matrix = agu.build_design_matrix(
    x_t=x_t,
    dt=0.01,
    library_degree=5,
    sg_poly_order=4,
    library_type="poly_four",
)

model_system = BayesianArgosAnalysis(
    design_matrix=design_matrix, accelerator=True
)
model_system.run(
    mode="straight",
    parallel="yes",
    ncpus=None,
    ci_level=0.90,
)

simulated_data = model_system.simulate(
    n=10000, dt=0.01, init_conditions=initial_value_df.iloc[95].values
)

ags.plot_3d_trajectory(
    simulated_data,
    "Thomas_Simulated",
    show_colorbar=False,
    save_figure=True,
    show_axes=False,
)

# %%
# ! -------------------- Chenlee --------------------
initial_value_df = ags.generate_initial_value_df(
    seed=100,
    num_init_samples=100,
    x1_range=(5, 20),
    x2_range=(-20, 20),
    x3_range=(-20, 0),
    num_columns=3,
)

x_t = ags.generate_noisy_dynamical_systems(
    variable_coeff=[[5, -1], [-10, 1], [-3.8, 1 / 3]],
    variable_names=[["x1", "x2x3"], ["x2", "x1x3"], ["x3", "x1x2"]],
    n=5000,
    dt=0.01,
    init_conditions=initial_value_df.iloc[35].values,
    snr=40,
)

ags.plot_3d_trajectory(
    x_t,
    "Chen Lee",
    show_colorbar=True,
    save_figure=False,
    color_bar_ticks=1000,
)

design_matrix = agu.build_design_matrix(
    x_t=x_t,
    dt=0.01,
    library_degree=5,
    sg_poly_order=4,
    library_type="poly",
)

model_system = BayesianArgosAnalysis(
    design_matrix=design_matrix, accelerator=True
)

model_system.run(
    mode="straight",
    parallel="yes",
    ncpus=None,
    ci_level=0.90,
)

# %%
# ! -------------------- Dadras --------------------
initial_value_df = ags.generate_initial_value_df(
    seed=100,
    num_init_samples=100,
    x1_range=(-4, 4),
    x2_range=(-4, 4),
    x3_range=(-4, 4),
    num_columns=3,
)

x_t = ags.generate_noisy_dynamical_systems(
    variable_coeff=[
        [1, -3, 2.7],
        [1.7, -1, 1],
        [2, -9],
    ],
    variable_names=[
        ["x2", "x1", "x2x3"],
        ["x2", "x1x3", "x3"],
        ["x1x2", "x3"],
    ],
    n=5000,
    dt=0.01,
    init_conditions=initial_value_df.iloc[30].values,
    snr=11,
)

ags.plot_3d_trajectory(
    x_t,
    "Dadras",
    show_colorbar=True,
    save_figure=False,
    color_bar_ticks=1000,
)

design_matrix = agu.build_design_matrix(
    x_t=x_t,
    dt=0.01,
    library_degree=5,
    sg_poly_order=4,
    library_type="poly",
)

model_system = BayesianArgosAnalysis(
    design_matrix=design_matrix, accelerator=True
)

model_system.run(
    mode="straight",
    parallel="yes",
    ncpus=None,
    ci_level=0.90,
)

# %%
# ! -------------------- Sprott --------------------
initial_value_df = ags.generate_initial_value_df(
    seed=100,
    num_init_samples=100,
    x1_range=(-1, 1),
    x2_range=(-1, 1),
    x3_range=(-1, 1),
    num_columns=3,
)

x_t = ags.generate_noisy_dynamical_systems(
    variable_coeff=[[1, 2.07, 1], [1, -1.79, 1], [1, -1, -1]],
    variable_names=[
        ["x2", "x1x2", "x1x3"],
        ["", "x1^2", "x2x3"],
        ["x1", "x1^2", "x2^2"],
    ],
    n=5000,
    dt=0.01,
    init_conditions=initial_value_df.iloc[35].values,
    snr=49,
)

ags.plot_3d_trajectory(
    x_t,
    "Sprott",
    show_colorbar=True,
    save_figure=False,
    color_bar_ticks=1000,
)

design_matrix = agu.build_design_matrix(
    x_t=x_t,
    dt=0.01,
    library_degree=5,
    sg_poly_order=4,
    library_type="poly",
)

model_system = BayesianArgosAnalysis(
    design_matrix=design_matrix, accelerator=True
)

model_system.run(
    mode="straight",
    parallel="yes",
    ncpus=None,
    ci_level=0.90,
)

# %%
# ! -------------------- Halvorsen --------------------
initial_value_df = ags.generate_initial_value_df(
    seed=100,
    num_init_samples=100,
    x1_range=(-4, 4),
    x2_range=(-4, 4),
    x3_range=(-4, 4),
    num_columns=3,
)

x_t = ags.generate_noisy_dynamical_systems(
    variable_coeff=[
        [-1.89, -4, -4, -1],
        [-1.89, -4, -4, -1],
        [-1.89, -4, -4, -1],
    ],
    variable_names=[
        ["x1", "x2", "x3", "x2^2"],
        ["x2", "x3", "x1", "x3^2"],
        ["x3", "x1", "x2", "x1^2"],
    ],
    n=5000,
    dt=0.01,
    init_conditions=initial_value_df.iloc[95].values,
    snr=13,
)

ags.plot_3d_trajectory(
    x_t,
    "Halvorsen",
    show_colorbar=True,
    save_figure=False,
    color_bar_ticks=1000,
)

design_matrix = agu.build_design_matrix(
    x_t=x_t,
    dt=0.01,
    library_degree=5,
    sg_poly_order=4,
    library_type="poly",
)

model_system = BayesianArgosAnalysis(
    design_matrix=design_matrix, accelerator=True
)

model_system.run(
    mode="straight",
    parallel="yes",
    ncpus=None,
    ci_level=0.90,
)


# %%
# ! -------------------- Rossler --------------------
initial_value_df = ags.generate_initial_value_df(
    seed=100,
    num_init_samples=100,
    x1_range=(-10, 10),
    x2_range=(-10, 10),
    x3_range=(0, 20),
    num_columns=3,
)

x_t = ags.generate_noisy_dynamical_systems(
    variable_coeff=[[-1, -1], [1, 0.2], [0.2, 1, -5.7]],
    variable_names=[["x2", "x3"], ["x1", "x2"], ["", "x1x3", "x3"]],
    n=5000,
    dt=0.01,
    init_conditions=initial_value_df.iloc[95].values,
    snr=30,
)

ags.plot_3d_trajectory(
    x_t,
    "Rossler",
    show_colorbar=False,
    save_figure=True,
    show_axes=False,
    output_format="svg",
)

design_matrix = agu.build_design_matrix(
    x_t=x_t,
    dt=0.01,
    library_degree=5,
    sg_poly_order=4,
    library_type="poly",
)

ags.plot_3d_trajectory(
    design_matrix["x_filtered"],
    "Rossler_filtered",
    show_colorbar=False,
    save_figure=True,
    show_axes=False,
    output_format="svg",
)

model_system = BayesianArgosAnalysis(
    design_matrix=design_matrix, custom_prior=True, accelerator=True
)
model_system.run(
    mode="straight",
    parallel="yes",
    ncpus=None,
    ci_level=0.90,
)

# %%
model_system.get_frequentist_results()
model_system.get_identified_model_from_all_equations()
print(model_system.results["equation_3"]["model"])
model_system.results["equation_3"]["model"].plot_priors()

# %%
simulated_data = model_system.simulate(
    n=5000, dt=0.01, init_conditions=initial_value_df.iloc[95].values
)

ags.plot_3d_trajectory(
    simulated_data,
    "Rossler_Simulated",
    show_colorbar=False,
    save_figure=True,
    show_axes=False,
    output_format="svg",
)

# %%
# ! -------------------- Aizawa --------------------
initial_value_df = ags.generate_initial_value_df(
    seed=100,
    num_init_samples=100,
    x1_range=(-2, 2),
    x2_range=(-2, 2),
    x3_range=(-1, 2),
    num_columns=3,
)

x_t = ags.generate_noisy_dynamical_systems(
    variable_coeff=[
        [-3.5, -0.7, 1],
        [3.5, -0.7, 1],
        [0.95, 0.65, 0.1, -1 / 3, -0.25, -1, -0.25, -1],
    ],
    variable_names=[
        ["x2", "x1", "x1x3"],
        ["x1", "x2", "x2x3"],
        ["x3", "", "x1^3x3", "x3^3", "x1^2x3", "x1^2", "x2^2x3", "x2^2"],
    ],
    n=10**3.7,
    dt=0.01,
    init_conditions=np.array([-0.3, 0.2, 0.1]),
    snr=40,
)

# 35

ags.plot_3d_trajectory(
    x_t,
    "Aizawa",
    show_colorbar=False,
    save_figure=True,
    show_axes=False,
    output_format="svg",
)

design_matrix = agu.build_design_matrix(
    x_t=x_t,
    dt=0.01,
    library_degree=5,
    sg_poly_order=4,
    library_type="poly",
)

ags.plot_3d_trajectory(
    design_matrix["x_filtered"],
    "Aizawa_filtered",
    show_colorbar=False,
    save_figure=True,
    show_axes=False,
    output_format="svg",
)

model_system = BayesianArgosAnalysis(
    design_matrix=design_matrix, custom_prior=True, accelerator=True
)
model_system.run(
    mode="straight",
    parallel="yes",
    ncpus=None,
    ci_level=0.90,
)

model_system.get_frequentist_results()
model_system.get_identified_model_from_all_equations()

simulated_data = model_system.simulate(
    n=10**3.7, dt=0.01, init_conditions=np.array([-0.3, 0.2, 0.1])
)

ags.plot_3d_trajectory(
    simulated_data,
    "Aizawa_Simulated",
    show_colorbar=False,
    save_figure=True,
    show_axes=False,
    output_format="svg",
)

# %%
# The following code is only for the storage of BayesianArgos object, can not be used for BayesianArgosAnalysis.
# with open(
#     "../data/n-process-aizawa/aizawa_special_init.dill",
#     "wb",
# ) as f:
#     dill.dump(model_system, f)
# print("Saved aizawa_special_init.dill")


# %%
# ! -------------------- Process Analysis --------------------
# with open("../../../data/n-process-aizawa/aizawa_special_init.dill", "rb") as f:
#     model_system = dill.load(f)


# %%
# cv_grpnet_result_1 = model_system.results["equation_3"][
#     "init_alasso_second_model"
# ]
# cv_grpnet_result_2 = model_system.results["equation_3"][
#     "final_alasso_first_model"
# ]
# cv_grpnet_result_1.plot_loss()
# plt.show()
# model_system.results["equation_3"]
# %%
initial_lasso_result_grpnet = model_system.results["equation_3"][
    "init_alasso_model_analysis"
]
final_lasso_result_grpnet = model_system.results["equation_3"][
    "final_alasso_model_analysis"
]
dg_init = ad.diagnostic.diagnostic(initial_lasso_result_grpnet)
dg_final = ad.diagnostic.diagnostic(final_lasso_result_grpnet)

# print(initial_lasso_result_grpnet._lmda_path)
# list(initial_lasso_result_grpnet.__dict__.keys())
# print(dir(initial_lasso_result_grpnet))
# dg_init.state.__dict__.keys()

# %%
print(dg_init.betas)
dg_init.state.lmdas
# dg_init.plot_coefficients()
plot_betas_vs_log_lambda(
    dg_init,
    highlight_lambda=0.0060,
    label_fontsize=38,
    min_label_gap_fraction=0.155,
    filename="process-analysi/coefficients_initial_log_lambda_annotated.svg",
)
print(dg_final.betas)
dg_final.state.lmdas
# dg_final.plot_coefficients()
# %%
plot_betas_vs_log_lambda(
    dg_final,
    figsize=(15, 8.75),
    highlight_lambda=0.0060,
    label_fontsize=34,
    min_gap_from_font_tune=0.5,
    min_label_gap_fraction=0.1,
    filename="process-analysi/coefficients_final_log_lambda_annotated.svg",
)


# %%
# Use Adelie's built-in inspection
# dg_init = ad.diagnostic.diagnostic(final_lasso_result_grpnet)
# dg_init.plot_coefficients()
# # dg.plot_coefficients()
# adjust_adelie_diagnostic_plot_svg(
#     dg_init, filename="figures/coefficients_initial.svg"
# )
# dg_final = ad.diagnostic.diagnostic(final_lasso_result_grpnet)
# adjust_adelie_diagnostic_plot_svg(
#     dg_final, filename="figures/coefficients_final.svg"
# )

# - Show priors
model_bayesian = model_system.results["equation_3"]["model"]
# model_bayesian.plot_priors()
results = model_system.results["equation_3"]["results"]
# trace_axes = plot_trace_with_spacing(results, hspace=0.70)

raw_var_names = [
    "sigma",
    "x3",
    "Intercept",
    "x1^3 x3",
    "x3^3",
    "x1^2 x3",
    "x1^2",
    "x2^2 x3",
    "x2^2",
    "x2^2 x3^2",
    # "x1 x2^2 x3",
    # "x1 x3",
    # "x1^2 x2^2",
    # "x1^2 x2^2 x3",
    # "x2 x3^2",
    # "x3^4",
]
var_names_to_plot = [
    "$\\sigma$",
    "$x_3$",
    "$C$",
    "$x_1^3x_3$",
    "$x_3^3$",
    "$x_1^2x_3$",
    "$x_1^2$",
    "$x_2^2x_3$",
    "$x_2^2$",
    "$x_2^2x_3^2$",
    # "$x_1x_2^2x_3$",
    # "$x_1x_3$",
    # "$x_1^2x_2^2$",
    # "$x_1^2x_2^2x_3$",
    # "$x_2x_3^2$",
    # "$x_3^4$",
]

# %%
for raw_name, latex_name in zip(raw_var_names, var_names_to_plot):
    if raw_name in results.posterior.data_vars:
        export_trace_density_plot(
            results,
            figsize=(8, 6),
            var_name=raw_name,
            filename=f"process-analysis/diagnostics-aizawa/traces/{raw_name}.svg",
            var_name_latex=latex_name,
            include_density=False,
        )


# %%
# az.plot_density(results, shade=.1, hdi_prob=.90)
for raw_name, latex_name in zip(raw_var_names, var_names_to_plot):
    if raw_name in results.posterior.data_vars:
        export_density_plot(
            results,
            var_name=raw_name,
            var_name_latex=latex_name,
            filename=f"process-analysis/diagnostics-aizawa/density/{raw_name}.svg",
            shade=0.1,
            hdi_prob=0.90,
        )

# %%
# Replace direct posterior plotting with styled exports
# az.plot_posterior(results, hdi_prob=.90)
for raw_name, latex_name in zip(raw_var_names, var_names_to_plot):
    if raw_name in results.posterior.data_vars:
        export_posterior_plot(
            results,
            var_name=raw_name,
            var_name_latex=latex_name,
            filename=f"figures/diagnostics-aizawa/posterior/{raw_name}.svg",
            hdi_prob=0.90,
        )

# %%
# - Styled extraction of individual priors (example variables reused)
for raw_name, latex_name in zip(raw_var_names, var_names_to_plot):
    try:
        export_prior_plot(
            model_bayesian,
            var_name=raw_name,
            var_name_latex=latex_name,
            filename=f"figures/diagnostics-aizawa/priors/{raw_name}.svg",
            hdi_prob=0.90,
        )
    except Exception as e:
        print(f"Skipping prior export for '{raw_name}': {e}")

# %%
# - New code path: match trace colors to plot_betas_vs_log_lambda colors
#   (except variables not in beta paths, e.g., sigma / Intercept -> fallback_color)
export_trace_plots_with_betas_matched_colors(
    results=results,
    raw_var_names=raw_var_names,
    var_names_to_plot=var_names_to_plot,
    diagnostic_obj=dg_final,  # use dg_init if that is your reference beta-path plot
    output_dir="figures/diagnostics-aizawa/traces",
    fallback_color="#7570b3",
)
# %%
# - New code path: match colors for density + posterior + prior exports
export_density_posterior_prior_with_betas_matched_colors(
    results=results,
    model_bayesian=model_bayesian,
    raw_var_names=raw_var_names,
    var_names_to_plot=var_names_to_plot,
    diagnostic_obj=dg_final,  # use dg_init if that is your reference beta-path plot
    trace_output_dir="figures/diagnostics-aizawa/traces",
    density_output_dir="figures/diagnostics-aizawa/density",
    posterior_output_dir="figures/diagnostics-aizawa/posterior",
    prior_output_dir="figures/diagnostics-aizawa/priors",
    fallback_color="#7570b3",
    title_fontsize=76,
    tick_labelsize=52,
    posterior_annotation_fontsize=52,
    shared_figsize=(9, 7),
)

# %%
# ! -------------------- Residuals vs PP Mean Plot --------------------
model_bayesian.predict(results, kind="response")
y_obs = results.observed_data["target"].values
pp_mean = (
    results.posterior_predictive["target"].mean(dim=("chain", "draw")).values
)
residuals = y_obs - pp_mean

plt.rcParams.update(
    {
        "font.size": 24,
        "font.family": "serif",
        # "font.serif": ["Times New Roman", "Computer Modern Roman"],
        "font.serif": ["sans-serif", "DejaVu Serif", "serif"],
        "text.usetex": False,
        "axes.linewidth": 1.2,
        "axes.labelweight": "normal",
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.minor.width": 0.8,
        "ytick.minor.width": 0.8,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.fancybox": False,
        "legend.edgecolor": "black",
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.format": "pdf",
        "savefig.bbox": "tight",
        "savefig.transparent": False,
    }
)

# Create single plot
fig_residuals, ax = plt.subplots(1, 1, figsize=(10, 10))

ax.scatter(pp_mean, residuals, s=2, color="#116DA9", alpha=0.7)
z = np.polyfit(pp_mean, residuals, 3)
p = np.poly1d(z)
ax.plot(
    pp_mean,
    p(pp_mean),
    color="#634564",
    linestyle="-",
    alpha=0.8,
    linewidth=1.5,
)
ax.axhline(0, color="#9F0000", linestyle="--", alpha=0.6)
ax.tick_params(axis="both", labelsize=32)
ax.set_xlabel("Posterior Predictive Mean", fontsize=34)
ax.set_ylabel("Residuals", fontsize=34)
# ax.set_title(r"snr = $60$", fontsize=26)

fig_residuals.tight_layout()

# %%
fig_residuals.savefig(
    os.path.join(
        os.path.dirname(__file__),
        "figures",
        "diagnostics-aizawa",
        "aizawa_process_ppmean_vs_residuals.svg",
    ),
    dpi=300,
)


# ! -------------------- LOO Plot --------------------
# when deploy the LOO diagnostics, accelarator can not be used as its output lacks
# log_likeinbood output for the output plot.

# %%
model_system = BayesianArgosAnalysis(
    design_matrix=design_matrix, custom_prior=True, accelerator=False
)
model_system.run(
    mode="straight",
    parallel="yes",
    ncpus=None,
    ci_level=0.90,
)

model_system.get_identified_model_from_all_equations()

model_bayesian = model_system.results["equation_3"]["model"]
results = model_system.results["equation_3"]["results"]

# %%
loo = az.loo(results, pointwise=True)
khat = loo.pareto_k

# Create single LOO Pareto-k plot
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

khat_values = np.asarray(khat).ravel()
n_points = len(khat_values)
x_vals = range(n_points)

# Plot points < 0.7 in blue
ax.scatter(
    [i for i in x_vals if khat_values[i] < 0.7],
    [khat_values[i] for i in x_vals if khat_values[i] < 0.7],
    color="#116DA9",
    label="k < 0.7",
    s=20,
    marker="o",
    alpha=0.6,
)

# Plot points >= 0.7 in red
ax.scatter(
    [i for i in x_vals if khat_values[i] >= 0.7],
    [khat_values[i] for i in x_vals if khat_values[i] >= 0.7],
    color="#B03C2B",
    label="k ≥ 0.7",
    s=26,
    marker="^",
    alpha=0.9,
)

ax.axhline(y=0.7, color="#9F0000", linestyle="--", alpha=0.5)
ax.set_xlabel("Observations", fontsize=34)
ax.set_ylabel("Pareto k", fontsize=34, labelpad=12)
# ax.tick_params(axis="y", pad=32)
ax.tick_params(axis="both", labelsize=32)
# ax.set_title(f"$n = {n_points}$", fontsize=26)
ax.legend(markerscale=2.0)

plt.tight_layout()

# %%
fig.savefig(
    os.path.join(
        os.path.dirname(__file__),
        "figures",
        "diagnostics-aizawa",
        "aizawa_process_loo_pareto_k.svg",
    ),
    dpi=300,
)

# %%
