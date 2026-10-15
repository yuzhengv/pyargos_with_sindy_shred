"""Panels of the revised Fig. 9 (sea-surface-temperature application) in the
house style (plotting standard, blocks M9 and D3, D4, D1, D8).

Reads stored arrays and tables only; nothing is retrained, re-identified or
re-decoded:

  a  violins of the decoded-field forecast RMSE for three forecast intervals,
     Bayesian-ARGOS against SINDy, from ../rebuttal-latent-models/tables/
     experiments.csv selected as make_fig8a.py does (degree-1 models, stage1
     and valid: 87 and 67 experiments);
  b  experiment 91, training period: GRU latent trajectory, Bayesian-ARGOS
     posterior-mean simulation and the 90 % band over 1,000 coefficient draws
     (../rebuttal-posterior-forecast/results/latent_bands_91.npz);
  c  global maps of observations, predictions and differences at t = 50 ...
     250 weeks (results/fields_91.npz); the RMSE on each difference map is
     computed from the stored fields as plot_fields.plot_zoomed_comparison
     does and checked against results/rmse_91.csv (global, eval mode).

Canvases (k = 2.5 route, D1): a 8.3 x 9.4 in, b 9.7 x 9.4 in (together the
18 in canvas of a 183 mm figure), c 18 x 6.4 in. Each panel is saved with a
0.02 in pad and included at scale 0.4 by compose_fig9.sh, so the text sizes
set in plot-style/house.mplstyle come out at 8 pt (tags), 7 pt (titles, axis
titles, legends) and 6.5 pt (tick labels) at the final width of 183 mm.

Run:  python make_fig9_panels.py            (writes figures/house/fig9{a,b,c}.pdf)
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.colors import Normalize, TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RA = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(RA, "plot-style"))
import house_style as hs  # noqa: E402

TABLE = os.path.join(RA, "rebuttal-latent-models", "tables", "experiments.csv")
BANDS = os.path.join(RA, "rebuttal-posterior-forecast", "results", "latent_bands_91.npz")
RANGES = [(0, 100), (100, 200), (200, 275)]
RANGE_LABELS = ["0 to 100", "100 to 200", "200 to 275"]
PAD = 0.02           # tight-bbox pad (in) of every composed panel
LW = 1.8             # line width on the canvas (0.72 pt final) for the dense latent series


def save_panel(fig, path):
    hs.save(fig, path, pad_inches=PAD)
    plt.close(fig)
    print("saved", path)


# ---------------------------------------------------------------- panel a
def violin_table(table=TABLE):
    """Long table of decoded-field RMSE per experiment and interval, selected as make_fig8a.py does."""
    df = pd.read_csv(table)
    rows = []
    for tag, name in [("ba_deg1", "Bayesian-ARGOS"), ("sindy_deg1", "SINDy")]:
        g = df[(df.model == tag) & df.stage1 & df.valid]
        for _, r in g.iterrows():
            for (s, e), lab in zip(RANGES, RANGE_LABELS):
                rows.append({"interval": lab, "RMSE": r[f"RMSE_{s}_{e}"], "method": name, "iteration": int(r.iteration)})
    long = pd.DataFrame(rows)
    counts = long.groupby("method").iteration.nunique().to_dict()
    return long, counts


def panel_a(out, width_in=8.3, height_in=9.4, print_means=True):
    long, counts = violin_table()
    fig, ax = plt.subplots(figsize=(width_in, height_in), layout="constrained")
    palette = {m: hs.COLOURS[m] for m in ("Bayesian-ARGOS", "SINDy")}
    sns.violinplot(data=long, x="interval", y="RMSE", hue="method", hue_order=list(palette),
                   order=RANGE_LABELS, palette=palette, alpha=0.6, linewidth=1.0, cut=0, ax=ax, legend=False)
    hs.style_axes(ax)
    means = long.groupby(["interval", "method"]).RMSE.mean()
    for method, offset in (("Bayesian-ARGOS", -0.2), ("SINDy", 0.2)):
        for i, lab in enumerate(RANGE_LABELS):
            m = means[(lab, method)]
            ax.scatter(i + offset, m, marker="D", s=70, color=palette[method], edgecolors="black",
                       linewidths=1.0, zorder=4)
            if print_means:
                ax.annotate(f"{m:.3f}", (i + offset, m), textcoords="offset points", xytext=(0, 14),
                            ha="center", va="bottom", fontsize=16, color="black",
                            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85,
                                      edgecolor=palette[method], linewidth=0.8))
    ax.set_xlabel("forecast interval (weeks)")
    ax.set_ylabel("RMSE (°C)")
    ax.set_title("all valid experiments")
    ax.set_ylim(bottom=0)
    handles = [Patch(facecolor=palette[m], alpha=0.6, edgecolor="black", linewidth=0.8,
                     label=f"{m} (n = {counts[m]} valid experiments)") for m in palette]
    fig.legend(handles=handles, loc="outside lower center", ncol=1, title=None, handlelength=1.6)
    hs.tag(fig, "a", 0.0, 1.0, va="top")
    save_panel(fig, out)
    table = means.unstack().reindex(RANGE_LABELS)
    print(table.round(4).to_string())
    print("valid experiments:", counts)
    return table, counts


# ---------------------------------------------------------------- panel b (and the SI overlays)
def latent_overlay(gru, sim, out, q_lo=None, q_hi=None, title=None, tag="b",
                   width_in=9.7, height_in=9.4):
    """Three stacked sub-panels z1, z2, z3 against time (weeks): GRU latent trajectory (solid),
    Bayesian-ARGOS posterior-mean simulation (dashed) and, if given, the 5 to 95 % band."""
    d = gru.shape[1]
    n = min(len(gru), len(sim))
    x = np.arange(n)
    fig, axes = plt.subplots(d, 1, figsize=(width_in, height_in), sharex=True, layout="constrained")
    axes = np.atleast_1d(axes)
    for i, ax in enumerate(axes):
        if q_lo is not None:
            ax.fill_between(x, q_lo[:n, i], q_hi[:n, i], color=hs.PRIMARY, alpha=hs.BAND_ALPHA, linewidth=0)
        ax.plot(x, gru[:n, i], color=hs.GRU, linewidth=LW, alpha=0.9)
        ax.plot(x, sim[:n, i], color=hs.PRIMARY, linewidth=LW, linestyle="--", alpha=0.9)
        ax.set_ylabel(f"$z_{i + 1}$")
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 5, 10]))
        hs.style_axes(ax)
    if q_lo is not None:
        lo = min(np.nanmin(q_lo[:n]), np.nanmin(gru[:n])); hi = max(np.nanmax(q_hi[:n]), np.nanmax(gru[:n]))
        for ax in axes:                       # the three coordinates share one vertical scale
            ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
    axes[-1].set_xlabel("time (weeks)")
    axes[-1].set_xlim(0, int(np.ceil(n / 100.0)) * 100)
    axes[-1].xaxis.set_major_locator(MaxNLocator(6, steps=[1, 2, 5, 10]))
    if title:
        axes[0].set_title(title)
    handles = [Line2D([0], [0], color=hs.GRU, lw=LW, label="GRU latent trajectory"),
               Line2D([0], [0], color=hs.PRIMARY, lw=LW, linestyle="--",
                      label="Bayesian-ARGOS simulation (posterior mean)")]
    if q_lo is not None:
        handles.append(Patch(facecolor=hs.PRIMARY, alpha=hs.BAND_ALPHA,
                             label="90 % posterior band (1,000 coefficient draws)"))
    ncol = 2 if width_in >= 15 else 1
    fig.legend(handles=handles, loc="outside lower center", ncol=ncol, title=None, handlelength=2.2)
    if tag:
        hs.tag(fig, tag, 0.0, 1.0, va="top")
    save_panel(fig, out)


def panel_b(out, bands=BANDS):
    d = np.load(bands)
    latent_overlay(d["gru_train"], d["point_train"], out, q_lo=d["q05_train"], q_hi=d["q95_train"],
                   title="experiment 91, training period", tag="b")
    hw = (d["q95_train"] - d["q05_train"]) / 2
    for w in (52, 1040):
        print(f"band half-width at week {w}: " + " / ".join(f"{v:.2f}" for v in hw[w - 1]))


# ---------------------------------------------------------------- panel c (and the SI maps)
def field_maps(fields_npz, rmse_csv, out, title=None, tag="c", width_in=18.0, height_in=6.4,
               lat_range=(0, 180), lon_range=(0, 360), robust_pct=(1, 99)):
    """Observations, predictions and differences on the globe at the stored horizons; a styled
    copy of plot_fields.plot_zoomed_comparison (same field reconstruction, robust colour limits,
    symmetric difference limits and per-map RMSE), with the house colour maps and frames."""
    d = np.load(fields_npz)
    ts = [int(t) for t in d["timesteps"]]
    sst_locs = d["sst_locs"]

    def build(vec):
        f = np.full(180 * 360, np.nan); f[sst_locs] = vec
        return f.reshape(180, 360)[lat_range[0]:lat_range[1], lon_range[0]:lon_range[1]]

    real = np.stack([build(d["real"][j]) for j in range(len(ts))])
    pred = np.stack([build(d["pred"][j]) for j in range(len(ts))])
    diff = real - pred
    allv = np.concatenate([real.ravel(), pred.ravel()])
    vmin, vmax = np.nanpercentile(allv, robust_pct[0]), np.nanpercentile(allv, robust_pct[1])
    dlim = float(np.nanpercentile(np.abs(diff), robust_pct[1]))
    rmse = [float(np.sqrt(np.nanmean(diff[j] ** 2))) for j in range(len(ts))]
    ref = pd.read_csv(rmse_csv).set_index("t").rmse_global
    for t, r in zip(ts, rmse):
        assert abs(r - ref[t]) < 1e-6, (t, r, ref[t])
    print("map RMSE (°C):", " / ".join(f"t = {t}: {r:.2f}" for t, r in zip(ts, rmse)))

    n = len(ts)
    fig, axes = plt.subplots(3, n, figsize=(width_in, height_in), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.03, h_pad=0.03, wspace=0.0, hspace=0.0)
    for j, t in enumerate(ts):
        im0 = axes[0, j].imshow(real[j], cmap=hs.FIELD_CMAP, norm=Normalize(vmin, vmax), aspect="equal",
                                interpolation="nearest")
        axes[1, j].imshow(pred[j], cmap=hs.FIELD_CMAP, norm=Normalize(vmin, vmax), aspect="equal",
                          interpolation="nearest")
        im2 = axes[2, j].imshow(diff[j], cmap=hs.DIFF_CMAP, norm=TwoSlopeNorm(vmin=-dlim, vcenter=0.0, vmax=dlim),
                                aspect="equal", interpolation="nearest")
        axes[0, j].set_title(f"t = {t} weeks")
        axes[2, j].text(0.02, 0.04, f"RMSE {rmse[j]:.2f} °C", transform=axes[2, j].transAxes, fontsize=16,
                        ha="left", va="bottom",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85, edgecolor="0.5",
                                  linewidth=0.6))
        for r in range(3):
            hs.image_axes(axes[r, j])
    for r, lab in enumerate(["observations", "predictions", "differences"]):
        axes[r, 0].set_ylabel(lab)
    cb0 = fig.colorbar(im0, ax=list(axes[:2].ravel()), shrink=0.92, pad=0.01, aspect=28)
    cb0.set_label("SST (°C)")
    cb1 = fig.colorbar(im2, ax=list(axes[2].ravel()), shrink=0.92, pad=0.01, aspect=14)
    cb1.set_label("difference (°C)")
    for cb in (cb0, cb1):
        cb.outline.set_linewidth(0.8); cb.outline.set_edgecolor("0.5")
        cb.ax.tick_params(length=4, width=1.0)
    if title:
        fig.suptitle(title)
    if tag:
        hs.tag(fig, tag, 0.0, 1.0, va="top")
    save_panel(fig, out)
    return dict(zip(ts, rmse))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", default=os.path.join(HERE, "figures", "house"))
    p.add_argument("--no-means", action="store_true", help="omit the printed means in panel a")
    a = p.parse_args()
    hs.use()
    os.makedirs(a.out_dir, exist_ok=True)
    panel_a(os.path.join(a.out_dir, "fig9a.pdf"), print_means=not a.no_means)
    panel_b(os.path.join(a.out_dir, "fig9b.pdf"))
    field_maps(os.path.join(HERE, "results", "fields_91.npz"), os.path.join(HERE, "results", "rmse_91.csv"),
               os.path.join(a.out_dir, "fig9c.pdf"), title="experiment 91, test period", tag="c")


if __name__ == "__main__":
    main()
