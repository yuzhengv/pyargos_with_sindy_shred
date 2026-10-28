"""Classify the third (real) mode of the valid Bayesian-ARGOS latent models by its posterior.

Question (Yuzheng, 2026-10-27, Referee 1 Point 8): beyond the typical slowly varying
level, what do the other valid models find?  Every valid degree-1 Bayesian-ARGOS
model is sorted by the 90 % credible interval of the decay rate of its real
eigenvalue (5 to 95 % quantiles over the 1000 posterior draws that
run_experiment.py records):

    decays   interval entirely above zero   (the mode relaxes within the record)
    grows    interval entirely below zero   (slow exponential rise)
    level    interval contains zero, including the [0, 0] interval of an
             eigenvalue that is zero by structure (a coordinate on no right-hand side)

The decay rate is -Re(lambda), positive for decay, and the e-folding time is its
reciprocal.  The classification uses the posterior, not the point estimate; the
point-estimate count of fast modes (e-folding time below two years) is listed for
comparison with the earlier summary.

Reads only tables/experiments.csv and tables/decoder_ablation.csv.  Writes
tables/third_mode_classes.csv (one row per valid model), tables/third_mode_summary.md
(the per-class table) and figures/third_mode_classes.pdf (every model's decay rate
with its 90 % credible interval, sorted, coloured by class).  Never modifies package code.

Run:  python classify_third_mode.py            (~3 s)
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
import house_style as hs  # noqa: E402

TABLE = os.path.join(HERE, "tables", "experiments.csv")
ABLATION = os.path.join(HERE, "tables", "decoder_ablation.csv")
OUT_CSV = os.path.join(HERE, "tables", "third_mode_classes.csv")
OUT_MD = os.path.join(HERE, "tables", "third_mode_summary.md")
OUT_FIG = os.path.join(HERE, "figures", "third_mode_classes.pdf")

NEUTRAL = 0.05          # |decay rate| below this: e-folding time of 20 years or more
FAST_EFOLD = 2.0        # point-estimate criterion of the earlier summary (years)
CLASSES = ["level", "decays", "grows"]
LABELS = {"level": "level (interval contains zero)",
          "decays": "decays (interval above zero)",
          "grows": "grows (interval below zero)"}
STYLE = {"level": (hs.PRIMARY, "o"), "decays": (hs.HIGHLIGHT, "s"), "grows": (hs.EXTRA[2], "^")}


def classify(lo, hi):
    if lo > 0:
        return "decays"
    if hi < 0:
        return "grows"
    return "level"


def load():
    df = pd.read_csv(TABLE)
    v = df[(df.model == "ba_deg1") & df.stage1 & df.valid].copy()
    abl = pd.read_csv(ABLATION).set_index("iteration")
    out = pd.DataFrame({
        "iteration": v.iteration.astype(int).values,
        "rate": v.fast_decay_rate.values,                 # decay rate of the real mode, point estimate (/yr)
        "q05": v.real_decay_rate_q05.values,
        "q95": v.real_decay_rate_q95.values,
        "e_folding_yr": v.fast_e_folding_yr.values,
        "period_yr": v.period_yr.values,
        "annual_q05": v.osc_decay_rate_q05.values,
        "annual_q95": v.osc_decay_rate_q95.values,
        "rmse_0_100": v.RMSE_0_100.values,
        "rmse_100_200": v.RMSE_100_200.values,
        "rmse_200_275": v.RMSE_200_275.values,
        "rmse_0_275": v.RMSE_0_275.values,
        "latent_mse_train": v.latent_mse_train.values,
        "recon_rrmse": v.recon_rrmse_eval.values,
        "n_terms": v.n_terms.astype(int).values,
    })
    out["class"] = [classify(lo, hi) for lo, hi in zip(out.q05, out.q95)]
    out["structural_zero"] = (out.q05 == 0) & (out.q95 == 0)
    out["period_annual"] = (out.period_yr - 1.0).abs() <= 0.05
    out["annual_sign_undetermined"] = (out.annual_q05 <= 0) & (out.annual_q95 >= 0)
    out["fast_point_estimate"] = out.e_folding_yr < FAST_EFOLD
    out = out.merge(abl[["min_ratio", "max_ratio", "least_needed_coord"]], left_on="iteration",
                    right_index=True, how="left")
    return out.sort_values("iteration").reset_index(drop=True)


def summary(d):
    lines = ["# Third-mode classes of the valid Bayesian-ARGOS latent models (degree 1, stage1 & valid)", "",
             "Class = position of the 90 % credible interval (5 to 95 % quantiles over 1000 posterior draws) of the",
             "decay rate of the real eigenvalue relative to zero. The interval [0, 0] of a structurally zero eigenvalue",
             "counts as containing zero. Medians unless stated. Decoder ratios: decoded-field rRMSE with one latent",
             "coordinate frozen at its training mean divided by the full decode (`decoder_ablation.csv`).", "",
             "| quantity | " + " | ".join(CLASSES) + " |", "|---|" + "---|" * len(CLASSES)]

    def row(name, fn):
        lines.append(f"| {name} | " + " | ".join(fn(d[d["class"] == c]) for c in CLASSES) + " |")

    row("models", lambda g: f"{len(g)}")
    row("eigenvalue zero by structure", lambda g: f"{int(g.structural_zero.sum())}")
    row("decay rate, median (/yr)", lambda g: f"{g.rate.median():+.3f}")
    row("decay rate, range (/yr)", lambda g: f"{g.rate.min():+.3f} to {g.rate.max():+.3f}")
    row(f"|decay rate| < {NEUTRAL} /yr", lambda g: f"{int((g.rate.abs() < NEUTRAL).sum())}")
    row("e-folding time >= 20 yr", lambda g: f"{int((g.e_folding_yr >= 20).sum())}")
    row("e-folding time < 2 yr (point estimate)", lambda g: f"{int(g.fast_point_estimate.sum())}")
    row("e-folding times (yr), decays/grows classes",
        lambda g: ("—" if g["class"].iloc[0] == "level" else ", ".join(f"{x:.1f}" for x in sorted(g.e_folding_yr))))
    row("period within 1.00 +/- 0.05 yr", lambda g: f"{int(g.period_annual.sum())}")
    row("annual decay sign undetermined", lambda g: f"{int(g.annual_sign_undetermined.sum())}")
    for k, lab in [("rmse_0_100", "0-100"), ("rmse_100_200", "100-200"), ("rmse_200_275", "200-275"), ("rmse_0_275", "0-275")]:
        row(f"forecast RMSE {lab} wk, median (degC)", lambda g, k=k: f"{g[k].median():.2f}")
    row("training latent MSE, median", lambda g: f"{g.latent_mse_train.median():.3f}")
    row("reconstruction rRMSE, median", lambda g: f"{g.recon_rrmse.median():.4f}")
    row("least-needed coordinate ratio, median", lambda g: f"{g.min_ratio.median():.2f}")
    row("least-needed coordinate ratio < 1.05", lambda g: f"{int((g.min_ratio < 1.05).sum())}")
    row("most-needed coordinate ratio, median", lambda g: f"{g.max_ratio.median():.2f}")
    lines += ["", "## Fast real modes by the point estimate (e-folding time < 2 yr)", "",
              "| experiment | decay rate (/yr) | e-folding (yr) | 90 % interval | class |", "|---|---|---|---|---|"]
    for _, r in d[d.fast_point_estimate].sort_values("e_folding_yr").iterrows():
        lines.append(f"| {int(r.iteration)} | {r.rate:+.3f} | {r.e_folding_yr:.2f} | [{r.q05:+.3f}, {r.q95:+.3f}] | {r['class']} |")
    lines += ["", "## Structurally zero eigenvalue", "",
              "experiments " + ", ".join(str(int(i)) for i in d[d.structural_zero].iteration)
              + " (one latent coordinate appears on no right-hand side, so the real eigenvalue is zero in every draw).", ""]
    lines += ["## Representatives shown in the Supplementary Information", ""]
    for e in (91, 123, 18, 80):
        r = d[d.iteration == e]
        if len(r):
            r = r.iloc[0]
            lines.append(f"- experiment {e}: {r['class']}, decay rate {r.rate:+.3f} /yr, 90 % interval [{r.q05:+.3f}, {r.q95:+.3f}]")
    return "\n".join(lines) + "\n"


def figure(d, out):
    hs.use()
    s = d.sort_values(["rate", "iteration"]).reset_index(drop=True)
    x = np.arange(len(s)) + 1
    fig, ax = plt.subplots(figsize=hs.canvas(183, aspect=0.48), layout="constrained")
    ax.axhspan(-NEUTRAL, NEUTRAL, color=hs.GREY, alpha=0.35, linewidth=0, zorder=0)
    ax.axhline(0, color="black", linewidth=1.0, zorder=1)
    ax.axhline(1 / FAST_EFOLD, color="black", linewidth=1.0, linestyle=":", zorder=1)
    margin = len(s) + 2                      # labels in the margin to the right of the last model
    ax.annotate("e-folding time\nof 2 years", (margin, 1 / FAST_EFOLD), textcoords="offset points", xytext=(4, 0),
                ha="left", va="center", fontsize=hs.PT["tick"], color="black")
    ax.annotate(f"|decay rate| < {NEUTRAL} /yr,\ne-folding time of\n20 years or more", (margin, 0.0),
                textcoords="offset points", xytext=(4, 0), ha="left", va="center", fontsize=hs.PT["tick"], color="black")
    for cls in CLASSES:
        color, marker = STYLE[cls]
        m = (s["class"] == cls).values
        ax.vlines(x[m], s.q05[m], s.q95[m], color=color, linewidth=2.0, alpha=0.85, zorder=2)
        ax.scatter(x[m], s.rate[m], s=60, color=color, marker=marker, edgecolors="white", linewidths=1.0,
                   zorder=3, label=f"{LABELS[cls]}, n = {int(m.sum())}")
    z = s.structural_zero.values
    ax.scatter(x[z], s.rate[z], s=140, facecolors="none", edgecolors="black", linewidths=1.4, zorder=4,
               label=f"eigenvalue zero by structure, n = {int(z.sum())}")
    hs.style_axes(ax)
    ax.set_xlim(0, len(s) + 22)
    ax.set_xticks(np.arange(0, len(s) + 1, 20))
    ax.set_xlabel(f"valid Bayesian-ARGOS latent models (n = {len(s)}), sorted by the decay rate of the real mode")
    ax.set_ylabel("decay rate of the real mode (yr$^{-1}$)")
    ax.legend(loc="upper left", frameon=False, fontsize=hs.PT["tick"], handlelength=1.2, borderaxespad=0.2)
    hs.save(fig, out)
    plt.close(fig)
    return out


def main():
    d = load()
    d.to_csv(OUT_CSV, index=False)
    md = summary(d)
    with open(OUT_MD, "w") as f:
        f.write(md)
    print(md)
    print("saved", OUT_CSV, OUT_MD, figure(d, OUT_FIG))


if __name__ == "__main__":
    main()
