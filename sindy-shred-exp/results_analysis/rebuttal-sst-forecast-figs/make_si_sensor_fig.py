"""Supplementary Fig. S16: observed and forecast series at the 18 held-out
sensors of experiment 91 (eval mode), in the house style (plotting standard,
block S-F). A styled copy of plot_sensors.plot_multiple_sensor_predictions:
the same series (results/sensor_trajectories_91.npz, normalised units), the
same 52-week history window, the same per-panel RMSE over the 250 forecast
weeks (checked against results/sensor_rmse_91.csv); no grid, outward ticks,
sans-serif, one vertical scale for all panels (full data range, not the
2 to 98 % robust range of the original, so that no peak is clipped), the
training/forecast boundary as a dotted grey line, legend below.

The SI shows the figure as a sidewaysfigure, so it is authored for a final
width of 247 mm: canvas 247 / 25.4 * 2.5 = 24.31 in wide, aspect 0.5
(24.31 x 12.16 in), included at scale 0.4 by the manuscript's
\\includegraphics[width=247mm] (the PDF is saved at canvas size, so the
manuscript's width option gives the same 0.4 scale).

Run:  python make_si_sensor_fig.py          (writes figures/house/multi_sensor_forecast_91.pdf)
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "plot-style"))
import house_style as hs  # noqa: E402

FINAL_WIDTH_MM = 247.0
NUM_TRAIN, NUM_PRED = 52, 250


def sensor_grid(obs, fore, out, rmse_csv=None, rows=3, cols=6, aspect=0.5, lw=1.6):
    T = NUM_TRAIN + NUM_PRED
    n = min(obs.shape[1], rows * cols)
    fig, axes = plt.subplots(rows, cols, figsize=hs.canvas(FINAL_WIDTH_MM, aspect), sharex=True, sharey=True,
                             layout="constrained")
    axes = np.atleast_1d(axes).reshape(rows, cols)
    x_hist, x_fore = np.arange(NUM_TRAIN), np.arange(NUM_TRAIN, T)
    rmse = []
    for k in range(n):
        ax = axes[divmod(k, cols)]
        o, f = obs[:T, k].astype(float), fore[:T, k].astype(float)
        ax.plot(x_hist, o[:NUM_TRAIN], color=hs.GRU, lw=lw, alpha=0.9)
        ax.plot(x_fore, o[NUM_TRAIN:], color=hs.GRU, lw=lw, alpha=0.35)
        ax.plot(x_fore, f[NUM_TRAIN:], color=hs.PRIMARY, lw=lw, linestyle="--", alpha=0.95)
        ax.axvline(NUM_TRAIN, color="grey", linestyle=":", linewidth=1.0)
        ax.set_title(f"sensor {k}")
        valid = np.isfinite(o[NUM_TRAIN:]) & np.isfinite(f[NUM_TRAIN:])
        r = float(np.sqrt(np.mean((f[NUM_TRAIN:][valid] - o[NUM_TRAIN:][valid]) ** 2)))
        rmse.append(r)
        ax.text(0.97, 0.05, f"RMSE {r:.3f}", transform=ax.transAxes, fontsize=16, ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="0.5", linewidth=0.6))
        ax.xaxis.set_major_locator(MaxNLocator(5, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        hs.style_axes(ax)
    for k in range(n, rows * cols):
        axes[divmod(k, cols)].axis("off")
    allv = np.concatenate([obs[:T, :n].ravel(), fore[:T, :n].ravel()]).astype(float)
    lo, hi = np.nanmin(allv), np.nanmax(allv)
    pad = 0.04 * (hi - lo)
    axes[0, 0].set_ylim(lo - pad, hi + pad)
    axes[0, 0].set_xlim(0, T)
    for r in range(rows):
        axes[r, 0].set_ylabel("SST (normalised)")
    for c in range(cols):
        axes[rows - 1, c].set_xlabel("time (weeks)")
    handles = [Line2D([0], [0], color=hs.GRU, lw=lw, label="observation"),
               Line2D([0], [0], color=hs.PRIMARY, lw=lw, linestyle="--", label="forecast (posterior mean)")]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, title=None, handlelength=2.2)
    hs.save(fig, out, pad_inches=0.02)
    plt.close(fig)
    if rmse_csv is not None:
        ref = pd.read_csv(rmse_csv).rmse_normalised.to_numpy()
        assert np.allclose(rmse, ref[:n], atol=1e-6), "per-panel RMSE differs from the stored table"
    print("saved", out, "| mean per-panel RMSE (normalised units):", round(float(np.mean(rmse)), 4))
    return rmse


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--iteration", type=int, default=91)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    hs.use()
    out = a.out or os.path.join(HERE, "figures", "house", f"multi_sensor_forecast_{a.iteration}.pdf")
    d = np.load(os.path.join(HERE, "results", f"sensor_trajectories_{a.iteration}.npz"))
    sensor_grid(d["obs"], d["fore"], out, rmse_csv=os.path.join(HERE, "results", f"sensor_rmse_{a.iteration}.csv"))


if __name__ == "__main__":
    main()
