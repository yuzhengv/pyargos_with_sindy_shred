"""Band-aware versions of the Fig. 8b and Fig. S9 plotting blocks, plus an
RMSE-versus-lead-time panel. Same colours, fonts and layout as the originals in
`../rebuttal-sst-forecast-figs/plot_latent.py` and `plot_sensors.py`; the only
addition is a shaded posterior credible band around the forecast."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import MaxNLocator, ScalarFormatter  # noqa: E402

GRU_COLOR = "#6a51a3"
BA_COLOR = "#505ed5"
BAND_ALPHA = 0.25


def plot_latent_bands(gru, sim_point, q_lo, q_hi, save_path, ci_label="90 % credible band", panel="b"):
    """Fig. 8b with a posterior band: GRU latent trajectory, posterior-mean simulation, credible band."""
    d = gru.shape[1]
    n = min(len(gru), len(sim_point))
    fig, axes = plt.subplots(d, 1, figsize=(12, d * 3.5), dpi=600, sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    x = np.arange(n)
    for i in range(d):
        axes[i].fill_between(x, q_lo[:n, i], q_hi[:n, i], color=BA_COLOR, alpha=BAND_ALPHA, linewidth=0)
        axes[i].plot(gru[:n, i], linewidth=2, color=GRU_COLOR, alpha=0.8)
        axes[i].plot(sim_point[:n, i], linestyle="--", linewidth=2, color=BA_COLOR, alpha=0.8)
        axes[i].set_ylabel(f"$z_{{{i + 1}}}$", fontsize=30, fontweight="bold")
        axes[i].grid(True, alpha=0.3, linestyle="--")
        axes[i].tick_params(axis="both", which="major", labelsize=24)
    axes[-1].set_xlabel("Time Index (Weeks)", fontsize=30)
    if panel:
        fig.text(-0.01, 1.03, panel, fontsize=44, fontweight="bold", ha="left", va="top")
    handles = [Line2D([0], [0], color=GRU_COLOR, lw=2, label="GRU Output"),
               Line2D([0], [0], color=BA_COLOR, lw=2, linestyle="--", label="Bayesian-ARGOS Simulation"),
               Patch(facecolor=BA_COLOR, alpha=BAND_ALPHA, label=ci_label)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.54, 0), ncol=3, fontsize=26)
    fig.tight_layout()
    fig.savefig(save_path, dpi=600, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)


def plot_sensor_bands(obs, fore_point, q_lo, q_hi, save_path, num_train=52, num_pred=250, rows=3, cols=6,
                      ci_label="90 % credible band", font_scale=1.0):
    """Fig. S9 with a posterior band. obs/fore/q_* are (T, n_sensors) in the network's normalised units.
    Returns per-panel RMSE of the point forecast and per-panel coverage of the band over the forecast window."""
    rc = matplotlib.rcParams
    rc["font.size"] = 16 * font_scale; rc["axes.titlesize"] = 18 * font_scale
    rc["axes.labelsize"] = 16 * font_scale; rc["xtick.labelsize"] = 16 * font_scale
    rc["ytick.labelsize"] = 16 * font_scale; rc["axes.linewidth"] = 1
    n_panels = min(obs.shape[1], rows * cols)
    T = num_train + num_pred
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), dpi=600, sharex=True, sharey=True)
    axes = np.atleast_1d(axes).reshape(rows, cols)
    fmt = ScalarFormatter(useMathText=True); fmt.set_powerlimits((-2, 3))
    stats = []
    x_hist = np.arange(num_train); x_fore = np.arange(num_train, T)
    for k in range(n_panels):
        ax = axes[divmod(k, cols)]
        o, f = obs[:T, k], fore_point[:T, k]
        lo, hi = q_lo[:T, k], q_hi[:T, k]
        ax.fill_between(x_fore, lo[num_train:], hi[num_train:], color=BA_COLOR, alpha=BAND_ALPHA, linewidth=0)
        ax.plot(x_hist, o[:num_train], color=GRU_COLOR, lw=1.6, alpha=0.9)
        ax.plot(x_fore, o[num_train:], color=GRU_COLOR, lw=1.6, alpha=0.35)
        ax.plot(x_fore, f[num_train:], color=BA_COLOR, lw=1.6, linestyle="--", alpha=0.95)
        ax.axvline(x=num_train, color="gray", linestyle=":", linewidth=1.0)
        ax.set_title(f"Sensor {k}", pad=3.5)
        rmse = float(np.sqrt(np.mean((o[num_train:] - f[num_train:]) ** 2)))
        cov = float(np.mean((o[num_train:] >= lo[num_train:]) & (o[num_train:] <= hi[num_train:])))
        stats.append({"panel": k, "rmse_normalised": rmse, "coverage_90": cov})
        ax.text(0.98, 0.06, f"RMSE {rmse:.3g}\ncover {100 * cov:.0f} %", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=13 * font_scale, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.85", lw=0.6))
        ax.grid(True, alpha=0.15, linestyle="--")
        ax.xaxis.set_major_locator(MaxNLocator(5)); ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_formatter(fmt)
    for k in range(n_panels, rows * cols):
        axes[divmod(k, cols)].axis("off")
    for r in range(rows):
        axes[r, 0].set_ylabel("Value")
    for c in range(cols):
        axes[rows - 1, c].set_xlabel("Time (weeks)")
    handles = [Line2D([0], [0], color=GRU_COLOR, lw=1.6, label="Observation"),
               Line2D([0], [0], color=BA_COLOR, lw=1.6, linestyle="--", label="Forecast (posterior mean)"),
               Patch(facecolor=BA_COLOR, alpha=BAND_ALPHA, label=ci_label)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=3, frameon=True)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(save_path, dpi=600, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    return stats


def plot_rmse_lead_time(weeks, rmse_point, q_lo, q_med, q_hi, save_path, ci_label="90 % band over posterior draws",
                        marks=None):
    """Global decoded RMSE (degC) against lead time: point-estimate forecast, posterior median and band."""
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=600, constrained_layout=True)
    ax.fill_between(weeks, q_lo, q_hi, color=BA_COLOR, alpha=BAND_ALPHA, linewidth=0, label=ci_label)
    ax.plot(weeks, q_med, color=BA_COLOR, lw=1.5, label="posterior median of draws")
    ax.plot(weeks, rmse_point, color=GRU_COLOR, lw=2, linestyle="--", label="posterior-mean model (Fig. 8c)")
    if marks is not None:
        for t in marks:
            ax.axvline(t, color="gray", linestyle=":", lw=0.8)
    ax.set_xlabel("Lead time (weeks)", fontsize=16); ax.set_ylabel("Global RMSE (°C)", fontsize=16)
    ax.tick_params(labelsize=14); ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=13, loc="upper left")
    fig.savefig(save_path, dpi=600, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
