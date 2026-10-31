"""Main-text candidate figure: the eigenvalues of the 87 valid Bayesian-ARGOS latent models with their posteriors.

Question (second-opinion review of the Section 2.5 redraft, 2026-10-31): the ensemble
finding of Section 2.5, a shared annual period and a slow mode sorted into three
classes by its posterior, is supported only by Supplementary Fig. S16 and Table S3.
This script draws the two quantities side by side for every valid model, in the
same model order, so that the annual cycle (the control) and the third mode (the
finding) can be read from one figure.

    a   period of the annual mode, posterior mean with its 90 % credible interval
        (5 to 95 % quantiles over the 1000 posterior draws of run_experiment.py),
        with the 1.00 +/- 0.05 year band of the main text
    b   decay rate of the real mode with its 90 % credible interval, the panel of
        Supplementary Fig. S16 (classify_third_mode.py), unchanged

Both panels sort the models by the decay rate of the real mode and colour them by
the third-mode class, so panel a also shows that the annual period does not differ
between the classes.  Reads tables/experiments.csv through classify_third_mode.load()
and writes figures/ensemble_eigenvalues.pdf.  Never modifies package code.

Run:  python ensemble_eigenvalues_figure.py          (~3 s)
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "plot-style"))
sys.path.insert(0, HERE)
import house_style as hs  # noqa: E402
import classify_third_mode as ctm  # noqa: E402

OUT_FIG = os.path.join(HERE, "figures", "ensemble_eigenvalues.pdf")
PERIOD_TOL = 0.05       # the main text counts periods within 1.00 +/- 0.05 years


def load():
    d = ctm.load()
    df = pd.read_csv(ctm.TABLE)
    v = df[(df.model == "ba_deg1") & df.stage1 & df.valid].set_index("iteration")
    d["period_q05"] = v.loc[d.iteration, "osc_period_yr_q05"].values
    d["period_q95"] = v.loc[d.iteration, "osc_period_yr_q95"].values
    return d.sort_values(["rate", "iteration"]).reset_index(drop=True)


def panel_period(ax, s, x):
    ax.axhspan(1 - PERIOD_TOL, 1 + PERIOD_TOL, color=hs.GREY, alpha=0.35, linewidth=0, zorder=0)
    ax.axhline(1.0, color="black", linewidth=1.0, zorder=1)
    margin = len(s) + 2
    n_in = int(s.period_annual.sum())
    ax.annotate(f"1.00 ± {PERIOD_TOL} years,\nn = {n_in} of {len(s)}", (margin, 1.0), textcoords="offset points",
                xytext=(4, 0), ha="left", va="center", fontsize=hs.PT["tick"], color="black")
    for cls in ctm.CLASSES:
        color, marker = ctm.STYLE[cls]
        m = (s["class"] == cls).values
        ax.vlines(x[m], s.period_q05[m], s.period_q95[m], color=color, linewidth=2.0, alpha=0.85, zorder=2)
        ax.scatter(x[m], s.period_yr[m], s=60, color=color, marker=marker, edgecolors="white", linewidths=1.0,
                   zorder=3)
    hs.style_axes(ax)
    ax.set_ylabel("period of the annual mode (years)")
    ax.set_ylim(0.94, 1.12)
    ax.set_yticks([0.95, 1.00, 1.05, 1.10])


def panel_third_mode(ax, s, x):
    """The panel of Supplementary Fig. S16 (classify_third_mode.figure), drawn into a given axis."""
    ax.axhspan(-ctm.NEUTRAL, ctm.NEUTRAL, color=hs.GREY, alpha=0.35, linewidth=0, zorder=0)
    ax.axhline(0, color="black", linewidth=1.0, zorder=1)
    ax.axhline(1 / ctm.FAST_EFOLD, color="black", linewidth=1.0, linestyle=":", zorder=1)
    margin = len(s) + 2
    ax.annotate("e-folding time\nof 2 years", (margin, 1 / ctm.FAST_EFOLD), textcoords="offset points",
                xytext=(4, 0), ha="left", va="center", fontsize=hs.PT["tick"], color="black")
    ax.annotate(f"|decay rate| < {ctm.NEUTRAL} /yr,\ne-folding time of\n20 years or more", (margin, 0.0),
                textcoords="offset points", xytext=(4, 0), ha="left", va="center", fontsize=hs.PT["tick"],
                color="black")
    for cls in ctm.CLASSES:
        color, marker = ctm.STYLE[cls]
        m = (s["class"] == cls).values
        ax.vlines(x[m], s.q05[m], s.q95[m], color=color, linewidth=2.0, alpha=0.85, zorder=2)
        ax.scatter(x[m], s.rate[m], s=60, color=color, marker=marker, edgecolors="white", linewidths=1.0,
                   zorder=3, label=f"{ctm.LABELS[cls]}, n = {int(m.sum())}")
    z = s.structural_zero.values
    ax.scatter(x[z], s.rate[z], s=140, facecolors="none", edgecolors="black", linewidths=1.4, zorder=4,
               label=f"eigenvalue zero by structure, n = {int(z.sum())}")
    hs.style_axes(ax)
    ax.set_xlim(0, len(s) + 22)
    ax.set_xticks(np.arange(0, len(s) + 1, 20))
    ax.set_xlabel(f"valid Bayesian-ARGOS latent models (n = {len(s)}), sorted by the decay rate of the real mode")
    ax.set_ylabel("decay rate of the real mode (yr$^{-1}$)")
    ax.legend(loc="upper left", frameon=False, fontsize=hs.PT["tick"], handlelength=1.2, borderaxespad=0.2)


def figure(s, out):
    hs.use()
    x = np.arange(len(s)) + 1
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=hs.canvas(183, aspect=0.78), sharex=True,
                                     height_ratios=[1.0, 1.7], layout="constrained")
    panel_period(ax_a, s, x)
    panel_third_mode(ax_b, s, x)
    for ax, letter in ((ax_a, "a"), (ax_b, "b")):
        hs.tag(ax, letter, x=-0.07, y=1.0)
    hs.save(fig, out)
    plt.close(fig)
    return out


def main():
    s = load()
    n_in = int(s.period_annual.sum())
    print(f"valid models {len(s)}, period within 1.00 +/- {PERIOD_TOL}: {n_in}, "
          f"classes: " + ", ".join(f"{c} {int((s['class'] == c).sum())}" for c in ctm.CLASSES))
    print("period range of the point estimates: %.3f to %.3f years" % (s.period_yr.min(), s.period_yr.max()))
    print("saved", figure(s, OUT_FIG))


if __name__ == "__main__":
    main()
