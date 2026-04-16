import glob
import os
import re

import dill
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.ticker import FuncFormatter, LogLocator
from statsmodels.stats.outliers_influence import variance_inflation_factor


def compute_vif(
    model=None,
    dill_path="../../data/n-analysis/aizawa_model_n_10_50.dill",
    equation_name="equation_3",
):
    """
    Compute VIF DataFrame for a given model object or a dill file path.
    Returns a pandas DataFrame with columns ['Variable', 'VIF'].
    """
    # load model if not provided
    if model is None:
        with open(dill_path, "rb") as f:
            model = dill.load(f)

    design_matrix = model.results[equation_name]["design_matrix"]
    del model
    design_matrix_name = design_matrix["sorted_feature_names"]
    design_matrix_value = design_matrix["sorted_theta"]

    # build a DataFrame from the design matrix
    X = pd.DataFrame(design_matrix_value, columns=design_matrix_name)

    # add an intercept if one isn't present
    if not any(
        c.lower() in ("const", "constant", "intercept") for c in X.columns
    ):
        X = sm.add_constant(X, has_constant="add")

    # compute VIF for each column
    vif_df = pd.DataFrame(
        {
            "Variable": X.columns,
            "VIF": [
                variance_inflation_factor(X.values, i)
                for i in range(X.shape[1])
            ],
        }
    )
    return vif_df


def collect_vif_results(
    data_dir="/home/yuzhengv/projects/pyargos_with_sindy_shred/pyargos/data/n-analysis",
    pattern="aizawa_model_n_10",
    equation_name="equation_3",
    out_csv=None,
):
    """
    Search recursively under data_dir for .dill files whose path or filename
    contains `pattern`, compute VIF for each using compute_vif, and return a
    concatenated DataFrame with columns: ['source_file', 'n', 'Variable', 'VIF'].
    If out_csv is provided, save the concatenated DataFrame to that path.
    """

    dill_files = [
        p
        for p in glob.glob(
            os.path.join(data_dir, "**", "*.dill"), recursive=True
        )
        if pattern in os.path.basename(p) or pattern in p
    ]

    results = []
    metadata = {}
    for p in sorted(dill_files):
        try:
            vif_df = compute_vif(dill_path=p, equation_name=equation_name)
        except Exception as e:
            # skip unreadable/invalid files but record the error
            metadata[p] = {"error": str(e)}
            continue

        # try to extract numeric n.
        # First, look for the number immediately following the given pattern
        # e.g. pattern="aizawa_model_n_10" -> "aizawa_model_n_10_105" yields 105
        base = os.path.basename(p)
        m = re.search(rf"{re.escape(pattern)}[_\-]?(\d+)", base)
        n_val = int(m.group(1))

        vif_df = vif_df.copy()
        vif_df["source_file"] = p
        vif_df["n"] = n_val
        results.append(vif_df)
        metadata[p] = {"n": n_val, "rows": len(vif_df)}

    if results:
        combined = pd.concat(results, ignore_index=True)
    else:
        combined = pd.DataFrame(columns=["Variable", "VIF", "source_file", "n"])

    cols = ["source_file", "n", "Variable", "VIF"]
    combined = combined[cols]

    if out_csv:
        combined.to_csv(out_csv, index=False)

    return combined, metadata


def get_chosen_var_vif(
    data_dir="/home/yuzhengv/projects/pyargos_with_sindy_shred/pyargos/n-analysis",
    pattern="aizawa_model_n_10",
    equation_name="equation_3",
    out_csv=None,
    n_divisor=None,
    manual_common_vars=None,  # new optional parameter: list/set of vars to force
):
    """
    Return (chosen_df, variables)
    - chosen_df: DataFrame filtered to variables that appear for every distinct n
                 with 'n' normalized by n_divisor (or those in manual_common_vars if provided).
    - variables: sorted list of variable names present in chosen_df (empty list if none).

    If manual_common_vars is provided (list or set), those variable names are used
    instead of computing the intersection across n. Any names in manual_common_vars
    that are not present in the collected VIF results are ignored.
    """
    if n_divisor is None:
        n_divisor = 1 if "snr" in pattern else 10

    vif_results_combined, _ = collect_vif_results(
        data_dir=data_dir,
        pattern=pattern,
        equation_name=equation_name,
        out_csv=out_csv,
    )

    # if no results, return empty
    if vif_results_combined.empty:
        return pd.DataFrame(columns=vif_results_combined.columns), []

    # determine variable set to use
    if manual_common_vars is not None:
        # coerce to set and only keep variables that actually exist in data
        manual_set = set(manual_common_vars)
        available = set(vif_results_combined["Variable"].unique())
        chosen_vars = sorted(manual_set & available)
    else:
        sets_per_n = (
            vif_results_combined.groupby("n")["Variable"].apply(set).tolist()
        )
        chosen_vars = (
            sorted(set.intersection(*sets_per_n)) if sets_per_n else []
        )

    # filter dataframe to chosen variables
    chosen_df = vif_results_combined[
        vif_results_combined["Variable"].isin(chosen_vars)
    ].copy()

    if chosen_df.empty:
        variables = []
    else:
        chosen_df["n"] = pd.to_numeric(chosen_df["n"]) / n_divisor
        variables = sorted(chosen_df["Variable"].unique())

    return chosen_df, variables


def postprocess_chosen_df_n(chosen_df):
    """
    Post-process chosen_df:
    - Replace Variable values equal to "const" (case-insensitive, trimmed) with "Intercept".
    Returns a new DataFrame (copy) with the replacement applied.
    """
    if chosen_df is None or chosen_df.empty:
        return chosen_df
    df = chosen_df.copy()
    df["Variable"] = (
        df["Variable"]
        .astype(str)
        .apply(lambda v: "Intercept" if v.strip().lower() == "const" else v)
    )
    df["log10_VIF"] = np.log10(df["VIF"])
    return df


def postprocess_chosen_df_snr(chosen_df):
    """
    Post-process chosen_df:
    - Rename column 'n' to 'snr' if present.
    - Replace Variable values equal to "const" (case-insensitive, trimmed) with "Intercept".
    Returns a new DataFrame (copy) with the changes applied.
    """
    if chosen_df is None or chosen_df.empty:
        return chosen_df
    df = chosen_df.copy()

    if "n" in df.columns:
        df = df.rename(columns={"n": "snr"})

    df["Variable"] = (
        df["Variable"]
        .astype(str)
        .apply(lambda v: "Intercept" if v.strip().lower() == "const" else v)
    )
    df["log10_VIF"] = np.log10(df["VIF"])
    return df


def postprocess_for_plotting(df, n_col="n"):
    """
    Return a copy of df augmented for plotting:
    - Adds 'Variable_tex' with TeX-formatted variable names, e.g.
        "x1"        -> "$x_{1}$"
        "x1^2"      -> "$x_{1}^{2}$"
        "x2^2 x3"   -> "$x_{2}^{2} x_{3}$"
        "Intercept" -> "$\\mathrm{Intercept}$"
      Tokens are split on whitespace. Handles tokens like 'x', 'x1', 'x1^2', 'x1x2' (best-effort).
    - If df contains column named by n_col (default "n"), adds:
        - 'n_pow10'        : numeric 10**n (NaN if n not numeric)
        - 'n_pow10_label'  : TeX label string like r"$10^{3.5}$" (for tick labels)
    Returns a new DataFrame (does not modify input).
    """
    if df is None:
        return df
    if df.empty:
        return df.copy()

    out = df.copy()

    def token_to_tex(tok: str) -> str:
        tok = tok.strip()
        if not tok:
            return tok
        lower = tok.lower()
        if lower in ("const", "constant") or lower == "intercept":
            return r"$\mathrm{Intercept}$"

        # split combined tokens like x1x2 into possible variable+index groups
        # first try splitting on non-alnum (rare); fallback to parsing contiguous groups
        parts = (
            re.split(r"[\*\:\s]+", tok)
            if re.search(r"[\*\:\s]", tok)
            else [tok]
        )

        tex_parts = []
        for part in parts:
            # match var letters, optional index digits, optional ^exponent
            m = re.match(r"^([A-Za-z]+)(\d*)(?:\^(\d+))?$", part)
            if m:
                var, idx, exp = m.group(1), m.group(2), m.group(3)
                tex = var
                if idx:
                    tex += (
                        "{" + "_" + idx + "}"
                        if False
                        else "_" + "{" + idx + "}"
                    )  # keep readable below
                    # produce x_{1} style
                    tex = f"{var}_{{{idx}}}"
                else:
                    tex = var
                if exp and exp != "1":
                    tex = f"{tex}^{{{exp}}}"
                tex_parts.append(tex)
            else:
                # fallback: try to split sequences of letter+digits, e.g. x1x2 -> x1,x2
                seq = re.findall(r"([A-Za-z]+\d*(?:\^\d+)?)", part)
                if seq:
                    fallback = []
                    for s in seq:
                        mm = re.match(r"^([A-Za-z]+)(\d*)(?:\^(\d+))?$", s)
                        if mm:
                            v, i, e = mm.group(1), mm.group(2), mm.group(3)
                            t = v + (f"_{{{i}}}" if i else "")
                            if e and e != "1":
                                t = f"{t}^{{{e}}}"
                            fallback.append(t)
                    tex_parts.extend(fallback)
                else:
                    # give up and escape unsafe chars minimally
                    safe = re.sub(r"([^0-9A-Za-z_\^])", r"\\\1", part)
                    tex_parts.append(safe)

        # join factors with a small space (LaTeX math)
        joined = " ".join(tex_parts)
        return f"${joined}$"

    # Generate Variable_tex column
    out["Variable_tex"] = out["Variable"].astype(str).apply(token_to_tex)

    # If n_col exists, add numeric and label columns for 10**n
    if n_col == "n" and n_col in out.columns:
        # coerce to numeric (safe)
        out[n_col] = pd.to_numeric(out[n_col], errors="coerce")
        out["n_pow10"] = np.power(10.0, out[n_col])

        # create label strings suitable for LaTeX ticks: "$10^{3.5}$"
        def make_label(val):
            if pd.isna(val):
                return ""
            # show integer without decimal if close to int
            if abs(val - round(val)) < 1e-8:
                txt = str(int(round(val)))
            else:
                txt = str(val)
            return rf"$10^{{{txt}}}$"

        out["n_pow10_label"] = out[n_col].apply(make_label)

    return out


def plot_vif_results(
    df,
    variables,
    out_dir,
    system_name="aizawa",
    use_log_scale=True,
    display_log_n=False,
    n_col="n",
    figsize=(14, 10),
    font_scale=1.0,
    plot_title=False,
    color_map=None,  # new: string name or iterable of colors
    marker_list=None,  # new: iterable of marker symbols
    line_styles=None,  # new: iterable of linestyles to combine with markers
    marker_every=None,  # new: force marker frequency (int) or None to auto
    highlight_below_indicator=True,  # new: if True, shade region below indicator_vif in green
):
    """
    Plot VIF results for given dataframe `df` and variable list `variables`.
    - df: DataFrame produced by postprocess_for_plotting (expects n_col, VIF, Variable_tex)
    - variables: iterable of variable names present in df['Variable']
    - out_dir: directory to save the output image (created if missing)
    - system_name: used to name the output file
    - use_log_scale: if True set y-axis to log scale (filters non-positive VIFs)
    - n_col: column name containing the x-axis values (default "n")

    New params:
      - color_map: string colormap name or iterable of color specs. If None,
                   a large palette is constructed automatically.
      - marker_list: iterable of matplotlib marker strings to use.
      - line_styles: iterable of linestyles (e.g. ['-','--',':','-.']) to cycle.
      - marker_every: if int, passed to matplotlib's markevery to reduce
                      marker density; if None an automatic value is chosen.
    """
    os.makedirs(out_dir, exist_ok=True)

    if df is None or df.empty:
        print("plot_df is empty — nothing to plot.")
        return

    rc = mpl.rcParams
    rc["font.family"] = "serif"
    rc["font.size"] = 20 * font_scale
    rc["axes.titlesize"] = 24 * font_scale
    rc["axes.labelsize"] = 20 * font_scale
    rc["xtick.labelsize"] = 20 * font_scale
    rc["ytick.labelsize"] = 20 * font_scale
    rc["axes.linewidth"] = 1
    rc["legend.frameon"] = True

    fig, ax = plt.subplots(figsize=figsize)

    # Build a color list large enough for many variables
    n_vars = len(variables) if variables else 0

    if color_map is None:
        # start with tab20 (20 easily distinguishable colors)
        try:
            base_cmap = plt.cm.get_cmap("tab20")
            base_colors = list(base_cmap.colors)
        except Exception:
            # fallback to tab10 if tab20 unavailable
            base_cmap = plt.cm.get_cmap("tab10")
            base_colors = list(base_cmap.colors)

        if n_vars > len(base_colors):
            n_extra = n_vars - len(base_colors)
            # add evenly spaced colors from HSV to extend palette
            extra = list(plt.cm.hsv(np.linspace(0, 1, n_extra, endpoint=False)))
            colors = base_colors + extra
        else:
            colors = base_colors
    else:
        if isinstance(color_map, str):
            cmap_obj = plt.cm.get_cmap(color_map)
            # sample evenly from the colormap
            colors = [cmap_obj(i / max(1, n_vars - 1)) for i in range(n_vars)]
        else:
            # assume iterable of color specs
            colors = list(color_map)

    # Extended marker pool
    if marker_list is None:
        markers = [
            "o",
            "s",
            "D",
            "^",
            "v",
            "P",
            "X",
            "*",
            "h",
            "d",
            "<",
            ">",
            "1",
            "2",
            "3",
            "4",
            "8",
            "p",
            "H",
            "+",
            "x",
            ".",
        ]
    else:
        markers = list(marker_list)

    # Line styles to increase distinctiveness when markers repeat
    if line_styles is None:
        line_styles = ["-", "--", "-.", ":"]
    else:
        line_styles = list(line_styles)

    for i, var in enumerate(variables):
        var = "Intercept" if var.strip().lower() == "const" else var
        dfv = df[df["Variable"] == var].copy()
        if dfv.empty:
            continue

        # If using log scale, remove non-positive VIF values to avoid plotting errors
        if use_log_scale:
            dfv = dfv[dfv["VIF"] > 0]
            if dfv.empty:
                # nothing to plot for this variable on log scale
                continue

        dfv = dfv.sort_values(n_col)
        x = dfv[n_col].values
        y = dfv["VIF"].values
        color = colors[i % len(colors)]
        mk = markers[i % len(markers)]
        ls = line_styles[i % len(line_styles)]

        # Match marker-point density behavior from plot_vif_n_and_snr_using_two_grids
        if marker_every is not None:
            mevery = max(1, int(len(x) / 8)) if len(x) > 8 else 1
        else:
            mevery = None

        ax.plot(
            x,
            y,
            label=dfv["Variable_tex"].iloc[0]
            if "Variable_tex" in dfv.columns
            else dfv["Variable"].iloc[0],
            marker=mk,
            linestyle=ls,
            color=color,
            linewidth=1.2,
            markersize=6,
            markeredgecolor="k",
            markeredgewidth=0.5,
            alpha=0.9,
            markevery=mevery,
        )

    # set y-scale
    if use_log_scale:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10.0))
    else:
        ax.set_yscale("linear")

    indicator_vif = 10

    if highlight_below_indicator:
        ax.axhline(
            y=indicator_vif,
            color="tab:green",
            linewidth=1.6,
            linestyle="--",
            alpha=0.9,
            zorder=6,
        )
    else:
        ax.axhline(
            y=indicator_vif,
            color="tab:red",
            linewidth=1.6,
            linestyle="--",
            alpha=0.9,
            zorder=6,
        )

    # set x-ticks to the unique n values and use the LaTeX labels provided (if present)
    unique_ns = (
        sorted(df[n_col].dropna().unique()) if n_col in df.columns else []
    )
    if unique_ns:
        if n_col == "n":
            x_min = float(np.min(unique_ns))
            x_max = float(np.max(unique_ns))

            k_start = int(np.floor(x_min))
            k_end = int(np.ceil(x_max))

            ticks = []
            for k in np.arange(k_start, k_end + 1, 0.4):
                if x_min <= k <= x_max:
                    ticks.append(float(k))
                    ticks.append(x_max)

            ticks = sorted(set(ticks))
            ax.set_xticks(ticks)
            ax.xaxis.set_major_formatter(
                FuncFormatter(lambda v, pos: rf"$10^{{{v:g}}}$")
            )
        else:
            if n_col == "snr" and 62 in unique_ns:
                ax.set_xticks(unique_ns)
                xtick_labels = [
                    "\\infty" if v == 62 else str(int(v)) for v in unique_ns
                ]
                ax.set_xticklabels(xtick_labels)
            else:
                ax.set_xticks(unique_ns)

    if use_log_scale:

        def _y_formatter(y, pos):
            if y <= 0 or np.isnan(y):
                return ""
            log10y = np.log10(y)
            return f"$10^{log10y:.0f}$"
    else:

        def _y_formatter(y, pos):
            return f"{y:.0f}"

    ax.yaxis.set_major_formatter(FuncFormatter(_y_formatter))

    ax.grid(True, linestyle="--", alpha=0.4)

    if n_col == "n":
        if display_log_n:
            ax.set_xlabel("\\log_{10}(n)")
        else:
            ax.set_xlabel("n")
    else:
        ax.set_xlabel("SNR(dB)")

    ax.set_ylabel("VIF")

    if plot_title:
        if n_col == "snr":
            ax.set_title("VIF across SNR")
        else:
            ax.set_title("VIF across $n$")

    # Adjust legend sizing/layout to be more compact and fit under the wider figure
    n_items = len(variables) if variables else 1
    # show legend labels in three columns (or fewer if there are fewer items)
    ncol = min(3, n_items)
    legend_fontsize = max(8, 20 * font_scale)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=ncol,
        frameon=False,
        fontsize=legend_fontsize,
        markerscale=3.0,
    )

    # --- FINAL Y-LIMITS + SHADING (fixed so shading always shows) ---
    fig.canvas.draw()

    ymin, ymax = ax.get_ylim()

    if ax.get_yscale() == "linear":
        ax.set_ylim(bottom=0)
        ymin, ymax = ax.get_ylim()

    if not (np.isnan(ymin) or np.isnan(ymax)):
        if ymax <= indicator_vif:
            ymax = indicator_vif + 1.0
            ax.set_ylim(ymin, ymax)
            ymin, ymax = ax.get_ylim()

        if highlight_below_indicator:
            span_low, span_high = ymin, indicator_vif
            span_color = "tab:green"
            text_y = indicator_vif - 1.5
        else:
            span_low, span_high = indicator_vif, ymax
            span_color = "tab:red"
            text_y = indicator_vif + 1.5

        ax.axhspan(
            span_low,
            span_high,
            facecolor=span_color,
            alpha=0.08,
            zorder=1.5,
        )

        ax.set_ylim(ymin, ymax)

        try:
            label_text = str(int(indicator_vif))
            ax.text(
                0.055 if use_log_scale else 0.04,
                text_y - 0.35,
                label_text,
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=max(8, 16 * font_scale),
                color=span_color,
            )
        except Exception:
            pass

    plt.tight_layout()

    if n_col == "snr":
        out_path = os.path.join(out_dir, f"{system_name}_vif_snr.svg")
    else:
        out_path = os.path.join(out_dir, f"{system_name}_vif_n.svg")

    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved combined plot to {out_path}")


def plot_vif_n_and_snr_using_two_grids(
    df_n,
    df_snr,
    variables,
    out_dir,
    system_name="aizawa",
    use_log_scale_n=True,
    use_log_scale_snr=True,
    display_log_n=False,
    figsize=(28, 20),
    figsize_legend_left=(20, 5.6),
    font_scale=1.0,
    plot_title=False,
    legend_left=True,
    color_map=None,
    marker_list=None,
    line_styles=None,
    marker_every=None,
    highlight_below_indicator=True,
):
    """Plot VIF vs ``n`` and VIF vs ``snr`` side by side.

    The two panels share the same colour/marker/style mapping for variables
    and a **single** centered legend spanning both subplots.
    """
    if df_n is None or df_n.empty or df_snr is None or df_snr.empty:
        print("One of the input dataframes is empty — nothing to plot.")
        return

    os.makedirs(out_dir, exist_ok=True)

    # Basic rc params (reuse from single-plot function)
    rc = mpl.rcParams
    rc["font.family"] = "serif"
    rc["font.size"] = 20 * font_scale
    rc["axes.titlesize"] = 24 * font_scale
    rc["axes.labelsize"] = 20 * font_scale
    rc["xtick.labelsize"] = 20 * font_scale
    rc["ytick.labelsize"] = 20 * font_scale
    rc["axes.linewidth"] = 1
    rc["legend.frameon"] = True

    # Prepare colours / markers shared across both axes
    n_vars = len(variables) if variables else 0

    if color_map is None:
        try:
            base_cmap = mpl.cm.get_cmap("tab20")
            base_colors = list(base_cmap.colors)
        except Exception:
            base_cmap = mpl.cm.get_cmap("tab10")
            base_colors = list(base_cmap.colors)

        if n_vars > len(base_colors):
            n_extra = n_vars - len(base_colors)
            extra = list(
                mpl.cm.get_cmap("hsv")(
                    np.linspace(0, 1, n_extra, endpoint=False)
                )
            )
            colors = base_colors + extra
        else:
            colors = base_colors
    else:
        if isinstance(color_map, str):
            cmap_obj = mpl.cm.get_cmap(color_map)
            colors = [cmap_obj(i / max(1, n_vars - 1)) for i in range(n_vars)]
        else:
            colors = list(color_map)

    if marker_list is None:
        markers = [
            "o",
            "s",
            "D",
            "^",
            "v",
            "P",
            "X",
            "*",
            "h",
            "d",
            "<",
            ">",
            "1",
            "2",
            "3",
            "4",
            "8",
            "p",
            "H",
            "+",
            "x",
            ".",
        ]
    else:
        markers = list(marker_list)

    if line_styles is None:
        line_styles = ["-", "--", "-.", ":"]
    else:
        line_styles = list(line_styles)

    # Adjust figure size if we place legend on the left
    if legend_left:
        used_figsize = figsize_legend_left
    else:
        used_figsize = figsize

    # --- Axes layout ---
    snr_unique = sorted(df_snr["snr"].unique())
    use_snr_break = 62 in snr_unique and len(snr_unique) >= 2

    if use_snr_break:
        # --- Nested GridSpec: outer (n | SNR block) and inner (SNR left | SNR right) ---
        wspace_outer = 0.15  # spacing between n and the whole SNR block
        wspace_snr = 0.1  # spacing between SNR left and SNR right (the "gap")

        fig = plt.figure(figsize=used_figsize)

        # Outer GridSpec: [ ax_n | SNR_block ]
        outer_gs = gridspec.GridSpec(
            1,
            2,
            figure=fig,
            width_ratios=[3.415, 3.275],
            wspace=wspace_outer,
        )

        # Inner GridSpec inside the SNR block: [ ax_snr_left | ax_snr_right ]
        snr_gs = gridspec.GridSpecFromSubplotSpec(
            1,
            2,
            subplot_spec=outer_gs[1],
            width_ratios=[0.9, 0.1],
            wspace=wspace_snr,
        )

        ax_n = fig.add_subplot(outer_gs[0])
        ax_snr_left = fig.add_subplot(snr_gs[0])
        ax_snr_right = fig.add_subplot(snr_gs[1])
        ax_snr = None  # not used in this mode
    else:
        fig, axes = plt.subplots(1, 2, figsize=used_figsize, sharey=False)
        ax_n, ax_snr = axes
        ax_snr_left = None
        ax_snr_right = None

    # Helper to draw on a given axis
    def _plot_on_axis(
        ax,
        df,
        n_col,
        use_log,
        display_log_n=display_log_n,
        highlight_below_indicator=highlight_below_indicator,
        xlim_override=None,
        draw_xticks=True,
        xlabel_override=None,
        show_threshold=False,
        draw_shading=True,
    ):
        for i, var in enumerate(variables):
            var_plot = "Intercept" if var.strip().lower() == "const" else var
            dfv = df[df["Variable"] == var_plot].copy()
            if dfv.empty:
                continue

            if use_log:
                dfv = dfv[dfv["VIF"] > 0]
                if dfv.empty:
                    continue

            dfv = dfv.sort_values(n_col)
            x = dfv[n_col].values
            y = dfv["VIF"].values
            color = colors[i % len(colors)]
            mk = markers[i % len(markers)]
            ls = line_styles[i % len(line_styles)]

            if marker_every is not None:
                mevery = max(1, int(len(x) / 8)) if len(x) > 8 else 1
            else:
                mevery = None

            ax.plot(
                x,
                y,
                label=dfv["Variable_tex"].iloc[0]
                if "Variable_tex" in dfv.columns
                else dfv["Variable"].iloc[0],
                marker=mk,
                linestyle=ls,
                color=color,
                linewidth=1.2,
                markersize=6,
                markeredgecolor="k",
                markeredgewidth=0.5,
                alpha=0.9,
                markevery=mevery,
            )

        if use_log:
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(LogLocator(base=10.0))
        else:
            ax.set_yscale("linear")

        indicator_vif = 10
        if highlight_below_indicator:
            ax.axhline(
                y=indicator_vif,
                color="tab:green",
                linewidth=1.6,
                linestyle="--",
                alpha=0.9,
                zorder=6,
            )
        else:
            ax.axhline(
                y=indicator_vif,
                color="tab:red",
                linewidth=1.6,
                linestyle="--",
                alpha=0.9,
                zorder=6,
            )

        unique_ns = (
            sorted(df[n_col].dropna().unique()) if n_col in df.columns else []
        )

        if draw_xticks and unique_ns:
            if n_col == "n":
                # Build ticks at integers and half-steps: 10^k and 10^{k+0.5}
                x_min = float(np.min(unique_ns))
                x_max = float(np.max(unique_ns))

                k_start = int(np.floor(x_min))
                k_end = int(np.ceil(x_max))

                ticks = []
                for k in np.arange(k_start, k_end + 1, 0.4):
                    if x_min <= k <= x_max:
                        ticks.append(float(k))
                        ticks.append(x_max)

                ticks = sorted(set(ticks))
                ax.set_xticks(ticks)

                # Labels: show as 10^{value}, where value is 2, 2.5, ...
                ax.xaxis.set_major_formatter(
                    FuncFormatter(lambda v, pos: rf"$10^{{{v:g}}}$")
                )
            else:
                if n_col == "snr" and 62 in unique_ns:
                    ax.set_xticks(unique_ns)
                    xtick_labels = [
                        "\\infty" if v == 62 else str(int(v)) for v in unique_ns
                    ]
                    ax.set_xticklabels(xtick_labels)
                else:
                    ax.set_xticks(unique_ns)

        # If requested, override x-limits after plotting & ticks
        if xlim_override is not None:
            ax.set_xlim(*xlim_override)

        if use_log:

            def _y_formatter(y, pos):
                if y <= 0 or np.isnan(y):
                    return ""
                log10y = np.log10(y)
                return f"$10^{log10y:.0f}$"
        else:

            def _y_formatter(y, pos):
                return f"{y:.0f}"

        ax.yaxis.set_major_formatter(FuncFormatter(_y_formatter))
        ax.grid(True, linestyle="--", alpha=0.4)

        if xlabel_override is not None:
            ax.set_xlabel(xlabel_override)
        else:
            if n_col == "n":
                if display_log_n:
                    ax.set_xlabel("\\log_{10}(n)")
                else:
                    ax.set_xlabel("n")
            else:
                ax.set_xlabel("SNR(dB)")

        ax.set_ylabel("VIF")

        if plot_title:
            if n_col == "snr":
                ax.set_title("VIF across SNR")
            else:
                ax.set_title("VIF across n")

        if not draw_shading:
            return

        # --- FINAL Y-LIMITS + SHADING (fixed so shading always shows) ---
        fig.canvas.draw()

        ymin, ymax = ax.get_ylim()

        if ax.get_yscale() == "linear":
            ax.set_ylim(bottom=0)
            ymin, ymax = ax.get_ylim()

        if not (np.isnan(ymin) or np.isnan(ymax)):
            if ymax <= indicator_vif:
                ymax = indicator_vif + 1.0
                ax.set_ylim(ymin, ymax)
                ymin, ymax = ax.get_ylim()

            # Always draw shading, independent of show_threshold
            if highlight_below_indicator:
                span_low, span_high = ymin, indicator_vif
                span_color = "tab:green"
                text_y = indicator_vif - 1.5
            else:
                span_low, span_high = indicator_vif, ymax
                span_color = "tab:red"
                text_y = indicator_vif + 1.5

            ax.axhspan(
                span_low,
                span_high,
                facecolor=span_color,
                alpha=0.08,
                zorder=1.5,  # slightly above background
            )

            ax.set_ylim(ymin, ymax)

            if use_log:
                # label_text = f"$\\log_{{10}}({int(indicator_vif)})$"
                label_text = str(int(indicator_vif))
            else:
                label_text = str(int(indicator_vif))

            # Only the *text* depends on show_threshold now
            if show_threshold:
                ax.text(
                    0.055 if use_log else 0.04,
                    text_y - 0.35,
                    label_text,
                    transform=ax.get_yaxis_transform(),
                    ha="right",
                    va="center",
                    fontsize=max(8, 16 * font_scale),
                    color=span_color,
                )

    # ---- Draw left n-panel ----
    _plot_on_axis(ax_n, df_n, "n", use_log_scale_n)

    # ---- Handle SNR axis (single or broken) ----
    if not use_snr_break:
        _plot_on_axis(ax_snr, df_snr, "snr", use_log_scale_snr)
    else:
        # Determine where to break: last real value before 62
        last_real = snr_unique[-2]

        # Left SNR axis: everything up to and including last_real
        _plot_on_axis(
            ax_snr_left,
            df_snr[df_snr["snr"] <= last_real],
            "snr",
            use_log_scale_snr,
        )

        # Right SNR axis: only the 62 point(s)
        df_snr_inf = df_snr[df_snr["snr"] >= 62]
        right_xlim = (61.5, 62.5)
        _plot_on_axis(
            ax_snr_right,
            df_snr_inf,
            "snr",
            use_log_scale_snr,
            xlim_override=right_xlim,
            draw_xticks=False,  # we'll add a custom ∞ tick
            xlabel_override="",
            show_threshold=False,  # now hides only the label, NOT shading
            draw_shading=False,
        )

        # Make the right SNR axis share the same y-limits as the left
        ymin_left, ymax_left = ax_snr_left.get_ylim()
        ax_snr_right.set_ylim(ymin_left, ymax_left)

        indicator_vif = 10
        if highlight_below_indicator:
            span_color = "tab:green"
            ax_snr_right.axhspan(
                ymin_left,
                indicator_vif,
                facecolor=span_color,
                alpha=0.08,
                zorder=1.5,
            )
        else:
            span_color = "tab:red"
            ax_snr_right.axhspan(
                indicator_vif,
                ymax_left,
                facecolor=span_color,
                alpha=0.08,
                zorder=1.5,
            )

        # Custom ticks on the right SNR axis
        ax_snr_right.set_xticks([62])
        ax_snr_right.set_xticklabels(["$\\infty$"])

        # Hide the spines between the two SNR axes and tweak ticks
        ax_snr_left.spines["right"].set_visible(False)
        ax_snr_right.spines["left"].set_visible(False)
        ax_snr_left.yaxis.tick_left()
        ax_snr_right.yaxis.tick_right()

        ax_snr_right.set_ylabel("")
        ax_snr_right.set_yticklabels([])

        # Slight rightward shift of the "SNR(dB)" label on the left SNR axis
        try:
            label = ax_snr_left.get_xaxis().get_label()
            label.set_position((0.65, label.get_position()[1]))
        except Exception:
            pass

        # Diagonal "break" marks between the two SNR axes.
        d_left = 0.0175
        d_right = 0.15

        slash_kwargs_left = dict(
            transform=ax_snr_left.transAxes,
            color="k",
            clip_on=False,
            linewidth=1.2,
        )
        slash_kwargs_right = dict(
            transform=ax_snr_right.transAxes,
            color="k",
            clip_on=False,
            linewidth=1.2,
        )

        tan_left = np.tan(np.deg2rad(60))
        tan_right = np.tan(np.deg2rad(12))
        ax_snr_left.plot(
            (1 - d_left, 1 + d_left),
            (-d_left * tan_left, d_left * tan_left),
            **slash_kwargs_left,
        )
        ax_snr_left.plot(
            (1 - d_left, 1 + d_left),
            (1 - d_left * tan_left, 1 + d_left * tan_left),
            **slash_kwargs_left,
        )

        ax_snr_right.plot(
            (-d_right, d_right),
            (-d_right * tan_right, d_right * tan_right),
            **slash_kwargs_right,
        )
        ax_snr_right.plot(
            (-d_right, d_right),
            (1 - d_right * tan_right, 1 + d_right * tan_right),
            **slash_kwargs_right,
        )

    handles, labels = ax_n.get_legend_handles_labels()
    n_items = len(handles) if handles else 1
    legend_left_min_number = 2 if n_items < 5 else 3
    bbcox_location = (
        (-0.20, 0.5) if legend_left_min_number == 2 else (-0.25, 0.5)
    )
    ncol = (
        min(7, n_items)
        if not legend_left
        else min(legend_left_min_number, n_items)
    )
    legend_fontsize = max(8, 20 * font_scale)

    if legend_left:
        fig.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=bbcox_location,
            ncol=ncol,
            frameon=False,
            fontsize=legend_fontsize,
        )
        plt.tight_layout()
    else:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.35),
            ncol=ncol,
            frameon=False,
            fontsize=legend_fontsize,
        )
        plt.tight_layout()

    out_path = os.path.join(out_dir, f"{system_name}_vif_n_and_snr.pdf")
    plt.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved side-by-side n & snr plot to {out_path}")
