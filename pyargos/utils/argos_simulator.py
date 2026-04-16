# from scipy.integrate import odeint
import os
import re
import sys

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import gridspec as mgrid
from matplotlib.colors import LinearSegmentedColormap


def _maybe_use_agg_backend():
    """Use a non-interactive backend only for truly headless runs.

    VS Code Interactive / Jupyter needs an interactive/inline backend, so we
    must not force Agg there.
    """

    if os.environ.get("MPLBACKEND"):
        return

    # Jupyter/VS Code Interactive (ipykernel) should decide the backend.
    if "ipykernel" in sys.modules:
        return

    # Common headless case: Linux without DISPLAY/WAYLAND.
    if sys.platform.startswith("linux") and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        matplotlib.use("Agg", force=True)


_maybe_use_agg_backend()

import matplotlib.pyplot as plt

# from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint, solve_ivp


## split string 'sin(cos(x))tan(x)' to ['sin(cos(x))', 'tan(x)']
def split_funcs(string):
    return re.split(r"(?<=\))(?=[a-z])", string)


# split_funcs('sin(cos(xy))tan(x)')


def split_term_func(string):
    # string = 'sin(xz)'
    if len(re.findall("\\^", string)) == 0:
        terms = re.findall(r"(\w+)", string)
    else:
        terms = re.findall(r"(\w+\^\d+|\w+)", string)
        # terms = re.findall(r'(\w+\^?\d*|\w+\^?\d*)', string)
        # terms = re.findall(r'(\w+(?:\^\d+)?|\w+(?:\^\d+)?)', string)
        if len(terms) == 1 and len(string) > 3:
            terms1 = list(re.findall(r"(\w\d*)?(\w\d*\^\d*)|w*", terms[0])[0])
            term1_tmp = [re.findall(r"^\d\^\d", q) != [] for q in terms1]
            if any(term1_tmp) == False:
                terms = terms1
    return terms


def split_term_func2(string):
    if len(re.findall("\\^", string)) == 0:
        # Use a regular expression to capture sin, cos, etc., numbers, and variables
        terms = re.findall(r"([a-zA-Z]+|\d*\.\d+|\d+|\w)", string)
    else:
        # If there are exponents, modify the regular expression accordingly
        terms = re.findall(r"([a-zA-Z]+|\d*\.\d+|\d+|\w+\^\d+|\w+)", string)

    return terms


def split_term(string):
    # string = 'sin(xz)'
    if len(re.findall("\\^", string)) == 0:
        terms = re.findall(r"(\w\d|\w\d)", string)
        if len(terms) == 0:
            terms = re.findall(r"(\w)", string)
    else:
        terms = re.findall(r"(\w+\^\d+|\w)", string)
        # terms = re.findall(r'(.+[\^\d+]*|.+[\^\d+]*)', string)
        # terms = list(re.findall(r'(\w\d*)?(\w\d*\^\d*)', string)[0])
    return terms


# split_term('xz')
# split_term('x2^5x3^3')


## find function orders for a term e.g. find [sin, cos] from sin(cos(x))
def find_functions(string):
    # string = 'sin(cos(x))'
    # string = 'sin(cos(xy))'
    # string = 'x1^2'
    # string = 'x1^2x2'
    # string = 'x1x2^2'
    # string = 'x1^2x2^2'
    # string = 'x'
    symbols = split_term_func(string)
    basic_funs = ["sin", "cos", "tan", "log", "exp"]
    funs_order = []
    for i in basic_funs:
        try:
            index = symbols.index(i)
            funs_order.append(index)
        except ValueError:
            pass
    find_basic_funs = [symbols[i] for i in funs_order]
    var_index = [x for x in list(range(len(symbols))) if x not in funs_order]
    variables0 = [symbols[i] for i in var_index]
    if len(variables0) == 1:
        variables1 = [split_term(variables0[j]) for j in range(len(var_index))][
            0
        ]
    else:
        variables1 = variables0
    # need to consider the operation order
    values1 = np.repeat(1.0, len(variables1)).tolist()
    num_index = []
    var_index = []
    for i in range(len(variables1)):
        try:
            value = eval(variables1[i])  # float(variables1[i])
            num_index.append(i)
        except:
            value = 1.0
            var_index.append(i)
        values1[i] = value
    variables = [variables1[i] for i in var_index]
    values = [values1[i] for i in num_index]
    if len(num_index) == 0:
        values = np.repeat(1.0, len(variables)).tolist()
    if len(variables) < len(values):
        variables.append("1")
    # if len(re.findall('\\^', string)) > 0:

    return (find_basic_funs, variables, values)


# find_functions('sin(cos(xy))')


def poly_order(string):
    # term = re.findall('[a-zA-Z0-9]*\\^', string)[0][:-1]
    if len(re.findall("\\^", string)) == 0:
        poly_order = 1
    else:
        poly_order = re.findall("\\^\d", string)[0][1]
    return int(poly_order)


# poly_order(split_term('x2^5x3^3')[0])


def find_which_term(string):
    return re.sub(r"\^\d+", "", string)


# find_which_term(split_term('x1^5x2^3')[0])


def basic_fun_np(string):
    if string == "sin":
        return np.sin
    if string == "cos":
        return np.cos
    if string == "tan":
        return np.tan
    if string == "log":
        return np.log
    if string == "exp":
        return np.exp


def term_comb(
    terms, term_names
):  # input one terms, terms = [x1,x2,x3] or [x,y,z]
    # term_names = 'sin(cos(xz))y'
    # term_names = 'x1^2x2';terms=[1,2]
    # terms = [2,3,1]; term_names='x1^2'
    if term_names == "":
        return float(1)
    else:
        out_f = []
        split_terms = split_funcs(term_names)
        for k in range(len(split_terms)):
            # if len(re.findall('\\^', split_terms[k])) == 0:
            used_funcs, term_names_split, coeffs = find_functions(
                split_terms[k]
            )
            poly = [
                poly_order(term_names_split[i])
                for i in range(len(term_names_split))
            ]
            base_terms = [
                find_which_term(term_names_split[i])
                for i in range(len(term_names_split))
            ]
            # else:
            # poly = [int(re.search('\d',re.search('\\^\d', term_names_split[i]).group()).group()) for i in range(len(term_names_split))]

            if len(re.findall(r"\d", base_terms[0])) != 0:
                terms_index = [f"x{n + 1}" for n in range(len(terms) - 1)]
                terms_index.append("t")
            else:
                if len(terms) == 3:
                    terms_index = ["x", "y", "t"]
                elif len(terms) == 4:
                    terms_index = ["x", "y", "z", "t"]
            terms_dict = {terms_index[i]: terms[i] for i in range(len(terms))}
            out = 1
            for i in range(len(term_names_split)):
                base_terms2 = base_terms[i]
                if base_terms2 in terms_dict:
                    out *= terms_dict[base_terms2] ** poly[i] * coeffs[i]
                else:
                    temp = eval(base_terms2)
                if base_terms2 == "1":  # consider 2.2*t+3.3
                    out += temp * coeffs[i]
            if len(used_funcs) > 0:
                used_funcs.reverse()
                for i in range(len(used_funcs)):
                    out = basic_fun_np(used_funcs[i])(out)
            out_f.append(out)
        return np.prod(out_f)


# term_comb([1,2,3], 'sin(cos(xz))y')

# use odeint


def ode_eq_3d_odeint(y, t, paras, terms_name):
    n = len(y)
    y2 = np.concatenate((y, [t]))  # add t to y
    out = [[] for _ in range(n)]
    for i in range(len(terms_name)):  # i for each equation
        out[i] = 0
        for j in range(
            len(terms_name[i])
        ):  # j for each term in the ith equation
            out[i] += paras[i][j] * term_comb(y2, terms_name[i][j])
    return out


def solve_ode_odeint(true_matrix, terms_name, initial, t_span):
    return odeint(
        ode_eq_3d_odeint, initial, t_span, args=(true_matrix, terms_name)
    )


# new use solve_ivp (import moved to top with odeint)


def ode_eq_3d_ivp(t, y, paras, terms_name):
    n = len(y)
    y2 = np.concatenate((y, [t]))  # add t to y
    out = np.zeros(n)
    for i in range(len(terms_name)):  # i for each equation
        for j in range(
            len(terms_name[i])
        ):  # j for each term in the ith equation
            out[i] += paras[i][j] * term_comb(y2, terms_name[i][j])
    return out


def solve_ode_ivp(true_matrix, terms_name, initial, t_span, method="RK45"):
    sol = solve_ivp(
        ode_eq_3d_ivp,
        [t_span[0], t_span[-1]],
        initial,
        args=(true_matrix, terms_name),
        t_eval=t_span,
        method=method,
    )
    return sol.y.T


def generate_noisy_dynamical_systems(
    variable_coeff: list,
    variable_names: list,
    n: int,
    dt: float,
    init_conditions: list,
    snr: int,
) -> any:
    """_summary_

    Args:
        variable_coeff (list): The coefficeints of the variables of the governing equations
        variable_names (list): The names of the variables of the governing equations
        n (int): The number of observations to generate
        dt (float): The time step of the generated observations
        init_conditions (list): Initial conditions of the dynamical system
        snr (int): snr of the generated observations

    Returns:
        any: A dataframe of the noisy dynamical system
    """
    t = np.arange(0, float(n) * dt, dt)  # np.arange(0, (n - 1) * dt, dt)
    x_t = solve_ode_odeint(variable_coeff, variable_names, init_conditions, t)
    # Convert snr (dB) to voltage
    snr_volt = 10 ** -(snr / 20)
    # Add noise (dB)
    if snr_volt != 0:
        x_init = x_t.copy()
        for i in range(int(x_t.shape[1])):
            x_t[:, i] = x_t[:, i] + snr_volt * np.random.normal(
                scale=np.std(x_init[:, i]), size=x_init[:, i].shape
            )
    return x_t


def generate_white_noise(
    n: float,
    dt: float,
    init_conditions: list,
    snr: int,
    std_scale: float = 1.0,
) -> np.ndarray:
    """Generate white noise in the same format as x_t from dynamical systems.

    Args:
        n (int): The number of observations to generate
        dt (float): The time step (for consistency, though not used in noise generation)
        init_conditions (list): Initial conditions to determine the number of variables
        snr (int): SNR of the generated noise
        std_scale (float): Standard deviation scaling factor for the noise

    Returns:
        np.ndarray: White noise array with shape (n, len(init_conditions))
    """
    # Convert n to nearest integer if it has decimal places
    n = int(round(n))

    num_variables = len(init_conditions)

    # Convert snr (dB) to voltage
    snr_volt = 10 ** -(snr / 20)

    # Generate white noise with same shape as x_t would have
    noise = np.random.normal(
        loc=0.0, scale=snr_volt * std_scale, size=(n, num_variables)
    )

    return noise


def generate_initial_value_df(
    seed,
    num_init_samples,
    x1_range,
    x2_range,
    x3_range=None,
    x4_range=None,
    num_columns=3,
):
    if num_columns not in [2, 3, 4]:
        raise ValueError("num_columns must be 2, 3, or 4")
    if num_columns == 3 and x3_range is None:
        raise ValueError("x3_range must be provided when num_columns is 3")
    if num_columns == 4 and (x3_range is None or x4_range is None):
        raise ValueError(
            "x3_range and x4_range must be provided when num_columns is 4"
        )

    np.random.seed(seed)

    ranges = [x1_range, x2_range]
    if num_columns == 3:
        ranges.append(x3_range)
    if num_columns == 4:
        ranges.extend([x3_range, x4_range])
    initial_values = [
        np.random.uniform(r[0], r[1], num_init_samples) for r in ranges
    ]

    initial_value_df = pd.DataFrame(
        np.vstack(initial_values).T,
        columns=[f"x{i + 1}" for i in range(num_columns)],
    )

    return initial_value_df


def plot_3d_trajectory(
    trajectory_data,
    system_name="System",
    show_colorbar=True,
    save_figure=True,
    figure_size=(8, 6),
    viewing_angle=(25, -80),
    color_bar_ticks=2500,
    colorbar_height=0.15,
    colorbar_gap=1.5,
    colorbar_left=0.15,
    colorbar_width=0.7,
    show_axes=True,
    tight_bbox=True,
    add_dummy_on_hide=True,  # NEW: ensure at least one artist for tight bbox
    return_fig=True,  # NEW: allow suppressing automatic inline re-render
    keep_safe_bbox=True,  # NEW: keep minimal invisible labels to avoid empty bbox crash
    output_format="pdf",
    time_step_indicator="blue-to-red",
    use_academic_style=True,
):
    """
    Plot a 3D trajectory with academic styling.

    Parameters:
    -----------
    trajectory_data : numpy.ndarray
        3D trajectory data with shape (n_points, 3)
    system_name : str
        Name of the dynamical system
    show_colorbar : bool
        Whether to show the colorbar
    save_figure : bool
        Whether to save the figure as PDF
    figure_size : tuple
        Base figure size (width, height) for display
    viewing_angle : tuple
        Elevation and azimuth angles for 3D view
    top_margin : float
        Top margin of the plot (0-1, where 1 means no top margin)
    show_axes : bool
        Whether to show axis frames, ticks and labels (default True). Setting this to
        False triggers several safeguards so that inline backends (e.g. Jupyter/IPython)
        do not raise the Matplotlib ValueError: "'bboxes' cannot be empty" that occurs
        when a 3D Axis has all of its tick/label artists removed and tight bounding box
        computation is requested (either explicitly or via rcParams like savefig.bbox='tight').
    tight_bbox : bool
        Whether to use tight bounding box for saving figures (default True)
    add_dummy_on_hide (bool): If True, when show_axes=False add a transparent dummy point
        so tight bbox passes still find at least one artist.
    keep_safe_bbox (bool): When hiding axes, retain a single-space label (invisible) and
        keep axis line (alpha=0) instead of fully removing it; this ensures Matplotlib's
        Axis3D.get_tightbbox still has at least one artist and avoids the empty union.
    return_fig (bool): If False, do not return (fig, ax) to avoid a second implicit render.

    Returns:
        any: A dataframe of the noisy dynamical system
    """
    # Auto-disable tight bbox if axes hidden or too few points to avoid empty bboxes error
    if not show_axes or trajectory_data.shape[0] < 2:
        tight_bbox = False

    # Helper: avoid global rcParams tight bbox triggering IPython render errors
    def _safe_save(fig, filename, want_tight):
        if not want_tight:
            fig.savefig(filename, dpi=150, facecolor="none")
            return
        try:
            fig.savefig(
                filename, dpi=150, facecolor="none", bbox_inches="tight"
            )
        except ValueError:
            # Fallback if tight bbox fails (e.g. empty 3D axis artists)
            fig.savefig(filename, dpi=150, facecolor="none", bbox_inches=None)

    plt.rcParams.update(
        {
            "font.size": 20,
            "font.family": "serif",
            # "font.serif": ["Times New Roman", "Computer Modern Roman"],
            "font.serif": ["sans-serif", "DejaVu Serif", "serif"],
            "text.usetex": False,
            "axes.linewidth": 1.2,
            "axes.labelweight": "normal",
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.minor.width": 0.8,
            "ytick.minor.width": 0.8,
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "legend.fancybox": False,
            "legend.edgecolor": "black",
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "savefig.format": "pdf",
        }
    )
    if output_format.lower() == "svg":
        plt.rcParams.update(
            {
                "savefig.format": "svg",
                "savefig.transparent": True,
            }
        )
    else:
        pass

    base_width, base_height = figure_size
    if show_colorbar:
        zlabel_labelpad = 23
        fig_height = base_height + colorbar_gap + colorbar_height
    else:
        zlabel_labelpad = 18
        fig_height = base_height

    fig = plt.figure(figsize=(base_width, fig_height))

    if show_colorbar:
        # Use GridSpec: two rows (3D plot + colorbar)
        # Height ratios proportional to actual inches to preserve 3D axis size
        gs = mgrid.GridSpec(
            2,
            1,
            height_ratios=[base_height, colorbar_height],
            hspace=colorbar_gap / max(base_height, 1e-6),
        )
        ax = fig.add_subplot(gs[0], projection="3d")
        # Make the 3D axis fill the entire figure canvas as requested.
        # ax.set_position([0, 0, 1, 1])
        cax = fig.add_subplot(gs[1])

    else:
        ax = fig.add_subplot(111, projection="3d")
        # Fill whole figure when no colorbar
        ax.set_position([0.01, 0.01, 0.99, 0.99])
        cax = None

    # Extract coordinates
    x = trajectory_data[:, 0]
    y = trajectory_data[:, 1]
    z = trajectory_data[:, 2]

    time_steps = np.arange(trajectory_data.shape[0])

    if time_step_indicator == "red-to-blue":
        favor_colors = [
            "#de2d26",
            "#fb6a4a",
            "#fc9272",
            "#fcbba1",
            "#fee5d9",
            "#d0d1e6",
            "#a6bddb",
            "#74a9cf",
            "#2b8cbe",
            "#045a8d",
        ]
    else:
        favor_colors = [
            "#045a8d",
            "#2b8cbe",
            "#74a9cf",
            "#a6bddb",
            "#d0d1e6",
            "#fee5d9",
            "#fcbba1",
            "#fc9272",
            "#fb6a4a",
            "#de2d26",
        ]

    custom_cmap = LinearSegmentedColormap.from_list(
        "custom", favor_colors, N=256
    )
    colors = custom_cmap(time_steps / float(time_steps.max()))

    # Plot trajectory with improved styling
    line_segments = []
    for i in range(1, len(time_steps)):
        line = ax.plot(
            x[i - 1 : i + 1],
            y[i - 1 : i + 1],
            z[i - 1 : i + 1],
            color=colors[i],
            alpha=0.8,
            linewidth=1.0,
            rasterized=True,
        )
        line_segments.extend(line)

    # Academic-style axis labels
    ax.set_xlabel(r"$x_1$", fontsize=20, labelpad=8)
    ax.set_ylabel(r"$x_2$", fontsize=20, labelpad=12)
    ax.set_zlabel(r"$x_3$", fontsize=20, labelpad=zlabel_labelpad)

    # Improve tick formatting
    ax.xaxis.set_major_locator(plt.MaxNLocator(4, prune="both"))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4, prune="both"))
    ax.zaxis.set_major_locator(plt.MaxNLocator(4, prune="both"))
    ax.zaxis.set_rotate_label(False)
    # ax.zaxis.set_label_coords(0.95, 0.5)

    ax.tick_params(axis="x", labelsize=18, pad=3)
    ax.tick_params(axis="y", labelsize=18, pad=6)
    ax.tick_params(axis="z", labelsize=18, pad=12)

    # Add colorbar with academic styling (only if enabled)
    if show_colorbar:
        sm = plt.cm.ScalarMappable(
            cmap=custom_cmap,
            norm=plt.Normalize(0, len(time_steps)),
        )
        sm.set_array([])
        # Draw horizontal colorbar in dedicated axis
        cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
        cbar.set_label(
            r"Time steps",
            fontsize=30,
            labelpad=20,
            fontfamily="sans-serif",
            fontweight=800,
        )
        ticks = np.arange(0, len(time_steps) + 1, color_bar_ticks)
        cbar.set_ticks(ticks)

        #  ---------------------------------------------------------------------
        if use_academic_style:
            labels = []
            for t in ticks:
                if t == 0:
                    labels.append("0")
                    continue
                exp = int(np.floor(np.log10(abs(t))))
                mant = t / (10**exp)
                if np.isclose(mant, 1.0):
                    labels.append(rf"$10^{{{exp}}}$")
                else:
                    labels.append(rf"${mant:.1f}\times10^{{{exp}}}$")

            cbar.set_ticklabels(labels)
        else:
            pass

        #  ---------------------------------------------------------------------
        cbar.ax.tick_params(labelsize=22)
        cbar.ax.xaxis.set_label_position("top")

    # Set viewing angle
    ax.view_init(elev=viewing_angle[0], azim=viewing_angle[1])

    # Clean styling
    ax.grid(False)
    ax.xaxis.pane.fill = True
    ax.yaxis.pane.fill = True
    ax.zaxis.pane.fill = True
    ax.xaxis.pane.set_facecolor("#9de0e6")
    ax.yaxis.pane.set_facecolor("#9de0e6")
    ax.zaxis.pane.set_facecolor("#9de0e6")
    ax.xaxis.pane.set_edgecolor("gray")
    ax.yaxis.pane.set_edgecolor("gray")
    ax.zaxis.pane.set_edgecolor("gray")
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)

    # NEW: optionally hide axes entirely
    if not show_axes:
        # HIDE visible decorations but keep minimal, invisible objects so bbox code has content.
        # Using a single space instead of an empty string forces Matplotlib to create a Text artist.
        if keep_safe_bbox:
            ax.set_xlabel(" ", labelpad=0)
            ax.set_ylabel(" ", labelpad=0)
            ax.set_zlabel(" ", labelpad=0)
        else:
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_zlabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        # Instead of removing axis line completely, set alpha to 0 so an artist remains.
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            try:
                axis.line.set_alpha(0.0)
            except Exception:
                axis.line.set_visible(False)
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.set_visible(False)
        # Some environments / external code may accidentally shadow set_frame_on attribute.
        _set_frame = getattr(ax, "set_frame_on", None)
        if callable(_set_frame):
            _set_frame(False)
        if add_dummy_on_hide:
            # Transparent dummy artist to further guarantee a bbox exists
            ax.plot([0], [0], [0], alpha=0)

    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)

    # Explicitly control colorbar size/position in figure coordinates
    if show_colorbar and cax is not None:
        # Convert inches to figure-fraction units
        cb_h = float(colorbar_height) / float(fig_height)
        cb_gap = float(colorbar_gap) / float(fig_height)

        # Place colorbar just below the 3D axes, honoring the requested gap/height
        bbox = ax.get_position()
        y0 = bbox.y0 - cb_gap - cb_h
        # Prevent going off-canvas
        y0 = max(0.01, y0)
        cax.set_position(
            [float(colorbar_left), y0, float(colorbar_width), cb_h]
        )

    if save_figure:
        filename = f"{system_name}_trajectory.{plt.rcParams['savefig.format']}"
        fig.set_size_inches((base_width, fig_height))
        _safe_save(fig, filename, tight_bbox)

    plt.show()

    if not return_fig:
        return None
    return fig, ax


def plot_trajectory_colorbar_only(
    n_steps,
    system_name="System",
    figure_size=(8, 2),
    only_show_main_ticks=False,
    color_bar_ticks=25000,
    colorbar_height=0.125,
    colorbar_left=0.15,
    colorbar_width=0.7,
    output_format="pdf",
    time_step_indicator="blue-to-red",
    save_figure=True,
    return_fig=True,
):
    """Generate a standalone horizontal colorbar matching `plot_3d_trajectory`.

    Parameters
    ----------
    n_steps : int
        Number of time steps (length of trajectory) to represent.
    system_name : str
        Used for naming the saved figure.
    figure_size : tuple
        (width, height) in inches.
    color_bar_ticks : int
        Spacing between tick labels along the colorbar.
    colorbar_height : float
        Height of the colorbar axes as a fraction of figure height.
    colorbar_left : float
        Left position of the colorbar axes (0-1, figure fraction).
    colorbar_width : float
        Width of the colorbar axes (0-1, figure fraction).
    output_format : {"pdf", "svg"}
        Output file format if saving.
    time_step_indicator : {"blue-to-red", "red-to-blue"}
        Direction of the colormap, consistent with `plot_3d_trajectory`.
    save_figure : bool
        If True, save the colorbar figure to disk.
    return_fig : bool
        If True, return (fig, cax).
    """

    if n_steps <= 0:
        raise ValueError("n_steps must be a positive integer")

    plt.rcParams.update(
        {
            "font.size": 20,
            "font.family": "serif",
            "font.serif": ["sans-serif", "DejaVu Serif", "serif"],
            "text.usetex": False,
            "figure.dpi": 600,
            "savefig.dpi": 600,
            "savefig.format": "pdf",
        }
    )
    if output_format.lower() == "svg":
        plt.rcParams.update(
            {
                "savefig.format": "svg",
                "savefig.transparent": True,
            }
        )

    if time_step_indicator == "red-to-blue":
        favor_colors = [
            "#de2d26",
            "#fb6a4a",
            "#fc9272",
            "#fcbba1",
            "#fee5d9",
            "#d0d1e6",
            "#a6bddb",
            "#74a9cf",
            "#2b8cbe",
            "#045a8d",
        ]
    else:
        favor_colors = [
            "#045a8d",
            "#2b8cbe",
            "#74a9cf",
            "#a6bddb",
            "#d0d1e6",
            "#fee5d9",
            "#fcbba1",
            "#fc9272",
            "#fb6a4a",
            "#de2d26",
        ]

    custom_cmap = LinearSegmentedColormap.from_list(
        "custom", favor_colors, N=256
    )

    fig = plt.figure(figsize=figure_size)
    cax = fig.add_axes(
        [
            float(colorbar_left),
            (1.0 - float(colorbar_height)) / 2.0,
            float(colorbar_width),
            float(colorbar_height),
        ]
    )

    sm = plt.cm.ScalarMappable(
        cmap=custom_cmap,
        norm=plt.Normalize(0, n_steps),
    )
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label(
        r"Time steps",
        fontsize=30,
        labelpad=20,
        fontfamily="sans-serif",
        fontweight=600,
    )
    ticks = np.arange(0, n_steps + 1, color_bar_ticks)
    cbar.set_ticks(ticks)

    if only_show_main_ticks:
        # Keep only main ticks
        # Keep only main ticks at 10^2, 10^2.5, 10^3, ... , 10^5 and format as 10^{*}
        exps = np.arange(2.0, 5.0 + 1e-9, 0.5)  # 2.0, 2.5, 3.0, ..., 5.0
        main_ticks = 10.0**exps
        # keep ticks within valid range [0, n_steps]
        main_ticks = main_ticks[main_ticks <= n_steps]
        ticks = main_ticks.tolist()
        cbar.set_ticks(ticks)

        # prepare LaTeX-style labels like 10^{2.5}, 10^{3}, ...
        custom_labels = []
        for e in exps:
            v = 10.0**e
            if v <= n_steps:
                if float(e).is_integer():
                    custom_labels.append(rf"$10^{{{int(e)}}}$")
                else:
                    # use compact decimal for fractional exponents (e.g. 2.5)
                    custom_labels.append(rf"$10^{{{e}}}$")

        # Ensure our custom labels survive the subsequent automatic formatting by
        # overriding ticklabels at draw time (savefig/draw will trigger this).
        def _override_labels(evt):
            try:
                cbar.set_ticklabels(custom_labels)
            except Exception:
                pass
            try:
                fig.canvas.mpl_disconnect(_cid)
            except Exception:
                pass

        _cid = fig.canvas.mpl_connect("draw_event", _override_labels)

    else:
        # Format tick labels in 10^* style (use LaTeX). 0 remains "0".
        labels = []
        for t in ticks:
            if t == 0:
                labels.append("0")
                continue
            exp = int(np.floor(np.log10(abs(t))))
            mant = t / (10**exp)
            if np.isclose(mant, 1.0):
                labels.append(rf"$10^{{{exp}}}$")
            else:
                labels.append(rf"${mant:.1f}\times10^{{{exp}}}$")

        cbar.set_ticklabels(labels)
        cbar.ax.tick_params(labelsize=20)
        cbar.ax.xaxis.set_label_position("top")

    if save_figure:
        filename = f"{system_name}_trajectory_colorbar.{plt.rcParams['savefig.format']}"
        fig.savefig(filename, dpi=600, facecolor="none", bbox_inches="tight")

    plt.show()

    if not return_fig:
        return None
    return fig, cax
