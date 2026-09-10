"""Figures and summary tables for the rebuttal diagnostics.

Reads results/exp1_*.csv, exp2_*.csv/npz, exp3_*.csv/npz and writes to
figures/:

  fig_vif_redesign.pdf        proposed replacement for Fig. 3b: ground-truth
                              terms coloured and labelled, all other library
                              terms in grey, median over trials
  fig_collinearity_source.pdf condition number / max VIF of the four nested
                              designs vs n and vs SNR, plus the on-attractor
                              vs state-space vs multi-trajectory comparison
  fig_heteroscedasticity.pdf  quantitative residual diagnostics vs SNR and the
                              proposed replacement for Fig. 3d (binned
                              mean +/- 2 sd instead of the cubic fit) plus the
                              residual / derivative-error overlay
and tables/*.md with the numbers quoted in the reply.
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import argparse  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
_ap = argparse.ArgumentParser(description="Figures/tables from a results folder (Python or R run).")
_ap.add_argument("--res", default=os.path.join(HERE, "results"), help="folder with exp*_*.csv")
_ap.add_argument("--fig", default=None, help="output folder for figures (default <res>/../figures)")
_ap.add_argument("--tab", default=None, help="output folder for tables (default <res>/../tables)")
_ap.add_argument("--tables-only", action="store_true", help="write tables only (the R run draws its figures with make_figures.R in the house style)")
_args, _ = _ap.parse_known_args()
RES = os.path.abspath(_args.res)
FIG = _args.fig or os.path.join(os.path.dirname(RES), "figures")
TAB = _args.tab or os.path.join(os.path.dirname(RES), "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)


def load_examples(res):
    """exp2 example arrays: .npz from the Python run, or the long-format CSV from the R run."""
    npz = os.path.join(res, "exp2_examples.npz")
    if os.path.exists(npz):
        return dict(np.load(npz))
    df = pd.read_csv(os.path.join(res, "exp2_examples.csv"), keep_default_na=False)
    out = {}
    for (system, snr, key), g in df.groupby(["system", "snr", "key"]):
        try:
            lab = "inf" if np.isinf(float(snr)) else str(int(float(snr)))
        except ValueError:
            lab = str(snr).lower()
        out[f"{system}_{lab}_{key}"] = pd.to_numeric(g.sort_values("index")["value"], errors="coerce").to_numpy(dtype=float)
    return out

# fixed categorical order (validated palette); grey for de-emphasised series
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
GREY = "#b8b6b0"
INK = "#0b0b0b"
INK2 = "#52514e"
DESIGN_ORDER = ["full", "trimmed", "selected", "truth"]
DESIGN_LABEL = {"full": "full library (55 terms)", "trimmed": "degree-trimmed after pass 1 (Fig. 3b)",
                "selected": "selected support after pass 2", "truth": "ground-truth terms only"}
TRUTH_ORDER = ["x3", "x1^2", "x2^2", "x1^2 x3", "x2^2 x3", "x1^3 x3", "x3^3"]

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "grid.color": "#e6e5e1", "grid.linewidth": 0.6,
    "lines.linewidth": 1.6, "pdf.fonttype": 42,
})


def tex(term: str) -> str:
    out = []
    for tok in term.split():
        if "^" in tok:
            v, p = tok.split("^")
            out.append(f"{v[0]}_{v[1:]}^{p}")
        else:
            out.append(f"{tok[0]}_{tok[1:]}")
    return "$" + "".join(out) + "$"


INF_POS = 66


def snr_axis(ax, snrs, ticks=(10, 20, 30, 40, 50, 60)):
    """Finite SNRs on a linear axis; infinity drawn at a fixed position."""
    pos = {s: (INF_POS if np.isinf(s) else s) for s in snrs}
    ax.set_xticks(list(ticks) + [INF_POS])
    ax.set_xticklabels([str(t) for t in ticks] + [r"$\infty$"])
    ax.axvline(INF_POS - 3, color="#e6e5e1", lw=0.8)
    return pos


def med_iqr(df, by, col):
    g = df.groupby(by)[col]
    return g.median(), g.quantile(0.25), g.quantile(0.75)


# ---------------------------------------------------------------- Fig. 3b redesign
def fig_vif_redesign(vif):
    d = vif[vif.design == "trimmed"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw=dict(wspace=0.35))
    for ax, sweep, xcol, xlabel in [(axes[0], "n", "log10n", r"$\log_{10} n$"), (axes[1], "snr", "snr", "SNR (dB)")]:
        s = d[d.sweep == sweep]
        med = s.groupby(["term", xcol])["vif"].median().reset_index()
        pos = snr_axis(ax, s.snr.unique(), ticks=(1, 10, 20, 30, 40, 50, 60)) if sweep == "snr" else None
        xs = (lambda v: [pos[x] for x in v]) if sweep == "snr" else (lambda v: v)
        others = [t for t in med.term.unique() if t not in TRUTH_ORDER]
        for t in others:
            m = med[med.term == t]
            ax.plot(xs(m[xcol]), m.vif, color=GREY, lw=0.8, zorder=1)
        for i, t in enumerate(TRUTH_ORDER):
            m = med[med.term == t].sort_values(xcol)
            if m.empty:
                continue
            ax.plot(xs(m[xcol]), m.vif, color=PAL[i], zorder=3, label=tex(t))
        ax.axhline(10, color=INK2, ls="--", lw=0.8)
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.grid(True, axis="y")
        ax.set_title("varying $n$ (SNR = 49 dB)" if sweep == "n" else "varying SNR ($n = 5000$)")
    axes[0].set_ylabel(r"VIF ($\dot{x}_3$, degree-trimmed library)")
    axes[0].text(axes[0].get_xlim()[0] + 0.05, 13, "VIF = 10", color=INK2, fontsize=7, va="bottom")
    axes[0].plot([], [], color=GREY, lw=0.8, label="other library terms")
    axes[0].legend(ncol=2, frameon=False, loc="lower right", handlelength=1.5)
    fig.savefig(os.path.join(FIG, "fig_vif_redesign.pdf"), bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------- collinearity source
def fig_collinearity_source(kappa, geo):
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.8), gridspec_kw=dict(wspace=0.42, width_ratios=[1, 1, 1.35]))
    for ax, sweep, xcol, xlabel in [(axes[0], "n", "log10n", r"$\log_{10} n$"), (axes[1], "snr", "snr", "SNR (dB)")]:
        s = kappa[kappa.sweep == sweep]
        pos = snr_axis(ax, s.snr.unique(), ticks=(1, 10, 20, 30, 40, 50, 60)) if sweep == "snr" else None
        for i, design in enumerate(DESIGN_ORDER):
            m, lo, hi = med_iqr(s[s.design == design], xcol, "kappa")
            x = [pos[v] for v in m.index] if sweep == "snr" else m.index
            ax.fill_between(x, lo, hi, color=PAL[i], alpha=0.15, lw=0)
            ax.plot(x, m, color=PAL[i], label=DESIGN_LABEL[design])
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.grid(True, axis="y")
        ax.set_title("varying $n$ (SNR = 49 dB)" if sweep == "n" else "varying SNR ($n = 5000$)")
    axes[0].set_ylabel(r"condition number $\kappa$ (Belsley scaling)")
    axes[0].legend(frameon=False, loc="upper left", fontsize=6.5)

    ax = axes[2]
    order = ["trajectory_sg", "trajectory_clean", "multi_transient_sg", "multi_transient", "box_uniform"]
    lab = {"trajectory_sg": "one trajectory\n(noisy, SG)", "trajectory_clean": "one trajectory\n(clean)",
           "multi_transient_sg": "50 transients\n(noisy, SG)", "multi_transient": "50 transients\n(clean)",
           "box_uniform": "uniform box\nsamples"}
    w = 0.38
    for j, design in enumerate(["full", "truth"]):
        g = geo[geo.design == design]
        m, lo, hi = med_iqr(g, "sampling", "kappa")
        y = [m[o] for o in order]
        err = [[m[o] - lo[o] for o in order], [hi[o] - m[o] for o in order]]
        ax.bar(np.arange(len(order)) + (j - 0.5) * w, y, w * 0.92, color=PAL[[0, 3][j]],
               yerr=err, error_kw=dict(lw=0.8, capsize=2, ecolor=INK2), label=DESIGN_LABEL[design])
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([lab[o].replace("\n", " ") for o in order], fontsize=6.5, rotation=25, ha="right")
    ax.set_ylabel(r"condition number $\kappa$ ($n = 5000$)")
    ax.set_title("same monomials, different sampling")
    ax.grid(True, axis="y")
    ax.legend(frameon=False, fontsize=6.5, loc="upper right")
    for k, lbl in enumerate("abc"):
        axes[k].text(-0.22, 1.1, lbl, transform=axes[k].transAxes, fontweight="bold", fontsize=10)
    fig.savefig(os.path.join(FIG, "fig_collinearity_source.pdf"), bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------ heteroscedasticity
def binned(fitted, e, nb=12):
    order = np.argsort(fitted)
    fs, es = fitted[order], e[order]
    xb, mb, sb = [], [], []
    for f, r in zip(np.array_split(fs, nb), np.array_split(es, nb)):
        xb.append(f.mean()); mb.append(r.mean()); sb.append(r.std(ddof=1))
    return np.array(xb), np.array(mb), np.array(sb)


def fig_heteroscedasticity(stats, ex):
    fig = plt.figure(figsize=(9.2, 8.0))
    gs = fig.add_gridspec(3, 3, hspace=0.6, wspace=0.4)
    axA, axB, axC = [fig.add_subplot(gs[0, k]) for k in range(3)]
    axD = [fig.add_subplot(gs[1, k]) for k in range(3)]
    axE = [fig.add_subplot(gs[2, k]) for k in range(3)]

    for i, (system, name) in enumerate([("aizawa", r"Aizawa $\dot{x}_3$"), ("rossler", r"Rössler $\dot{x}_1$")]):
        s = stats[stats.system == system]
        pos = snr_axis(axA, s.snr.unique())
        snr_axis(axB, s.snr.unique())
        for ax, col in [(axA, "rho1"), (axB, "var_ratio")]:
            m, lo, hi = med_iqr(s, "snr", col)
            x = [pos[v] for v in m.index]
            ax.fill_between(x, lo, hi, color=PAL[i], alpha=0.15, lw=0)
            ax.plot(x, m, color=PAL[i], marker="o", ms=3, label=name)
    # the same statistics on the support the pipeline actually selects (Aizawa)
    s = stats[stats.system == "aizawa"]
    pos = snr_axis(axA, s.snr.unique())
    for ax, col in [(axA, "sel_rho1"), (axB, "sel_var_ratio")]:
        m, lo, hi = med_iqr(s, "snr", col)
        x = [pos[v] for v in m.index]
        ax.fill_between(x, lo, hi, color=PAL[0], alpha=0.12, lw=0)
        ax.plot(x, m, color=PAL[0], marker="s", ms=3, ls="--", label=r"Aizawa $\dot{x}_3$, selected support")
    axA.set_ylabel(r"lag-1 residual autocorrelation $\rho_1$")
    axA.set_ylim(-0.05, 1.02)
    axA.axhline(0, color=INK2, lw=0.6, ls=":")
    axB.set_yscale("log")
    axB.set_ylabel("residual-variance ratio\n(max / min over fitted-value deciles)")
    axB.axhline(1, color=INK2, lw=0.6, ls=":")
    for ax in (axA, axB):
        ax.set_xlabel("SNR (dB)")
        ax.grid(True, axis="y")
    axA.legend(frameon=False, loc="center left", fontsize=6.5)

    s = stats[stats.system == "aizawa"]
    pos = snr_axis(axC, s.snr.unique())
    for j, (col, lbl) in enumerate([("var_d", r"$d$: derivative estimation"), ("var_g", r"$g$: state smoothing"),
                                    ("var_h", r"$h$: coefficient error")]):
        m = (s[col] / s["var_e"]).groupby(s.snr).median()
        axC.plot([pos[v] for v in m.index], m, color=PAL[[2, 3, 6][j]], marker="o", ms=3, label=lbl)
    axC.set_yscale("log")
    axC.set_ylabel(r"variance share of residual, Var($\cdot$)/Var($e$)")
    axC.set_xlabel("SNR (dB)")
    axC.grid(True, axis="y")
    axC.legend(frameon=False, loc="lower left", fontsize=6.5)
    axC.set_title(r"Aizawa $\dot{x}_3$: residual composition", loc="right")

    def residual_panel(ax, f, e, title, note):
        ax.scatter(f, e, s=1.2, color=PAL[0], alpha=0.35, lw=0, rasterized=True)
        xb, mb, sb = binned(f, e)
        ax.fill_between(xb, mb - 2 * sb, mb + 2 * sb, color=PAL[1], alpha=0.18, lw=0)
        ax.plot(xb, mb, color=PAL[1], lw=1.4)
        ax.axhline(0, color=INK2, lw=0.6, ls=":")
        ax.set_title(title, fontsize=8)
        ax.text(0.03, 0.96, note, transform=ax.transAxes, fontsize=6.5, va="top", color=INK2)
        ax.set_xlabel("fitted value (posterior-predictive mean)")

    for k, snr in enumerate(["50", "60", "inf"]):
        st = stats[(stats.system == "aizawa") & (stats.trial == 0) & (stats.snr.map(lambda v: "inf" if np.isinf(v) else str(int(v))) == snr)].iloc[0]
        lab = "SNR = " + (r"$\infty$" if snr == "inf" else f"{snr} dB")
        residual_panel(axD[k], ex[f"aizawa_{snr}_fitted"], ex[f"aizawa_{snr}_residuals"], lab + ", ground-truth support",
                       rf"$\rho_1$ = {st.rho1:.2f}   BP $R^2$ = {st.bp_r2:.2f}   var. ratio = {st.var_ratio:.3g}")
        residual_panel(axE[k], ex[f"aizawa_{snr}_sel_fitted"], ex[f"aizawa_{snr}_sel_residuals"], lab + ", pipeline-selected support",
                       rf"$\rho_1$ = {st.sel_rho1:.2f}   BP $R^2$ = {st.sel_bp_r2:.2f}   var. ratio = {st.sel_var_ratio:.3g}")
    axD[0].set_ylabel("residual (ground-truth support)")
    axE[0].set_ylabel("residual (pipeline-selected support)")
    for ax, lbl in zip([axA, axB, axC] + axD + axE, "abcdefghi"):
        ax.text(-0.22, 1.1, lbl, transform=ax.transAxes, fontweight="bold", fontsize=10)
    fig.savefig(os.path.join(FIG, "fig_heteroscedasticity.pdf"), bbox_inches="tight", dpi=300)
    plt.close(fig)

    # residual vs derivative-estimation error, time window
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3), gridspec_kw=dict(wspace=0.3))
    for ax, snr in zip(axes, ["50", "inf"]):
        e = ex[f"aizawa_{snr}_residuals"]; d = ex[f"aizawa_{snr}_d"]
        t = np.arange(len(e)) * 0.01
        w = slice(1000, 1600)
        ax.plot(t[w], e[w], color=PAL[0], lw=1.0, label="residual $e$")
        ax.plot(t[w], d[w], color=PAL[1], lw=1.0, ls="--", label=r"$d = \dot{x}_{3,\mathrm{SG}} - f_3(x_{\mathrm{clean}})$")
        ax.set_title("SNR = " + (r"$\infty$" if snr == "inf" else f"{snr} dB") +
                     f"   corr($e$, $d$) = {np.corrcoef(e, d)[0, 1]:.2f}", fontsize=8)
        ax.set_xlabel("time")
        ax.grid(True, axis="y")
    axes[0].set_ylabel("value")
    axes[0].legend(frameon=False, loc="lower left", fontsize=6.5, ncol=2)
    fig.savefig(os.path.join(FIG, "fig_residual_vs_derivative_error.pdf"), bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------- tables
def tables(vif, kappa, geo, stats, meta, ndep, hmc, cvn):
    lines = ["# Summary tables for the rebuttal (medians over trials, IQR in brackets)", ""]

    def fmt(m, lo, hi, f="{:.3g}"):
        return f"{f.format(m)} [{f.format(lo)}, {f.format(hi)}]"

    lines += ["## VIF of the ground-truth x3-dot terms, n = 5000, SNR = 49 dB, by design", "",
              "| term | " + " | ".join(DESIGN_ORDER) + " |", "|---|" + "---|" * len(DESIGN_ORDER)]
    v0 = vif[(vif.sweep == "n") & (np.isclose(vif.log10n, 3.8))]
    for t in TRUTH_ORDER:
        cells = []
        for dsg in DESIGN_ORDER:
            x = v0[(v0.design == dsg) & (v0.term == t)].vif
            cells.append(fmt(x.median(), x.quantile(0.25), x.quantile(0.75)) if len(x) else "–")
        lines.append(f"| {t} | " + " | ".join(cells) + " |")
    lines.append("")

    lines += ["## Condition number kappa by design (n sweep, SNR = 49 dB)", "",
              "| log10 n | " + " | ".join(DESIGN_ORDER) + " |", "|---|" + "---|" * len(DESIGN_ORDER)]
    kn = kappa[kappa.sweep == "n"]
    for k in sorted(kn.log10n.unique()):
        cells = []
        for dsg in DESIGN_ORDER:
            x = kn[(kn.design == dsg) & (kn.log10n == k)].kappa
            cells.append(fmt(x.median(), x.quantile(0.25), x.quantile(0.75), "{:.2e}"))
        lines.append(f"| {k:.1f} | " + " | ".join(cells) + " |")
    lines.append("")

    lines += ["## Geometry: same monomials, different sampling (n = 5000)", "",
              "| sampling | design | kappa | max VIF | median VIF |", "|---|---|---|---|---|"]
    for smp in ["trajectory_sg", "trajectory_clean", "multi_transient_sg", "multi_transient", "box_uniform"]:
        for dsg in ["full", "truth"]:
            g = geo[(geo.sampling == smp) & (geo.design == dsg)]
            lines.append(f"| {smp} | {dsg} | {fmt(g.kappa.median(), g.kappa.quantile(.25), g.kappa.quantile(.75), '{:.2e}')} | "
                         f"{fmt(g.vif_max.median(), g.vif_max.quantile(.25), g.vif_max.quantile(.75), '{:.2e}')} | "
                         f"{fmt(g.vif_median.median(), g.vif_median.quantile(.25), g.vif_median.quantile(.75), '{:.2e}')} |")
    lines.append("")

    lines += ["## Near-dependency groups (Belsley variance-decomposition, trial 0, n = 5000, SNR = 49 dB)", ""]
    for dsg, rep in ndep.items():
        lines.append(f"- **{dsg}** (p = {rep['p']}, kappa = {rep['condition_number']:.2e})")
        for gdep in rep["near_dependencies"]:
            lines.append(f"  - condition index {gdep['condition_index']:.2e}: " + ", ".join(gdep["members"]))
    lines.append("")

    lines += ["## Screening outcome for x3-dot (pass-2 support, before the HMC stage prunes it)", "",
              "| sweep | value | trials | trimmed p (median) | max degree kept (median) | selected p (median) | exact truth | truth subset of selected |",
              "|---|---|---|---|---|---|---|---|"]
    for sweep, xcol in [("n", "log10n"), ("snr", "snr")]:
        mm = meta[meta.sweep == sweep]
        for v in sorted(mm[xcol].unique()):
            r = mm[mm[xcol] == v]
            psel = r.selected.fillna("").map(lambda t: len([u for u in t.split(";") if u]))
            lines.append(f"| {sweep} | {v} | {len(r)} | {r.trimmed_p.median():.0f} | {r.trimmed_max_degree.median():.0f} | {psel.median():.0f} | "
                         f"{r.screen_exact_truth.mean():.2f} | {r.truth_in_selected.mean():.2f} |")
    lines.append("")

    lines += ["## Clean trajectory (no noise, no smoothing) over the n grid", "",
              "| log10 n | kappa full | max VIF full | kappa truth | max VIF truth |", "|---|---|---|---|---|"]
    for k in sorted(cvn.log10n.unique()):
        f = cvn[(cvn.log10n == k) & (cvn.design == "full")]; t = cvn[(cvn.log10n == k) & (cvn.design == "truth")]
        lines.append(f"| {k:.1f} | {f.kappa.median():.2e} | {f.vif_max.median():.2e} | {t.kappa.median():.3g} | {t.vif_max.median():.3g} |")
    lines.append("")

    lines += ["## Residual diagnostics vs SNR (n = 5000; ground-truth support fitted by the pipeline's Bayesian stage in the R run, OLS in the Python prototype)", "",
              "| system | SNR | sigma_e | rho1 | Durbin-Watson | BP R^2 | BP p (median) | var ratio | Var(d)/Var(e) | Var(d_trunc)/Var(e) | corr(e,d) | corr(e,d_trunc) |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for system in ["aizawa", "rossler"]:
        s = stats[stats.system == system]
        for snr in sorted(s.snr.unique()):
            r = s[s.snr == snr]
            lines.append(f"| {system} | {snr} | {r.sigma.median():.3g} | {fmt(r.rho1.median(), r.rho1.quantile(.25), r.rho1.quantile(.75), '{:.2f}')} | "
                         f"{r.durbin_watson.median():.2f} | {r.bp_r2.median():.3f} | {r.bp_p.median():.2g} | "
                         f"{fmt(r.var_ratio.median(), r.var_ratio.quantile(.25), r.var_ratio.quantile(.75), '{:.2f}')} | "
                         f"{(r.var_d / r.var_e).median():.2f} | {(r.var_dtrunc / r.var_e).median():.3g} | {r.corr_e_d.median():.2f} | {r.corr_e_dtrunc.median():.2f} |")
    lines.append("")

    lines += ["## Residual diagnostics vs SNR on the support the pipeline actually selects (screening pass 2)", "",
              "| system | SNR | selected p (median) | exact truth | truth subset | rho1 | BP R^2 | BP p (median) | var ratio |",
              "|---|---|---|---|---|---|---|---|---|"]
    for system in ["aizawa", "rossler"]:
        s = stats[stats.system == system]
        for snr in sorted(s.snr.unique()):
            r = s[s.snr == snr]
            psel = r.selected.fillna("").map(lambda t: len([u for u in t.split(";") if u]))
            lines.append(f"| {system} | {snr} | {psel.median():.0f} | {r.screen_exact_truth.mean():.2f} | {r.truth_in_selected.mean():.2f} | "
                         f"{fmt(r.sel_rho1.median(), r.sel_rho1.quantile(.25), r.sel_rho1.quantile(.75), '{:.2f}')} | {r.sel_bp_r2.median():.3f} | {r.sel_bp_p.median():.2g} | "
                         f"{fmt(r.sel_var_ratio.median(), r.sel_var_ratio.quantile(.25), r.sel_var_ratio.quantile(.75), '{:.2f}')} |")
    lines.append("")

    if hmc is not None:
        lines += ["## Full HMC runs (trial 0): HMC residuals vs OLS residuals on the same support", "",
                  "| SNR | identified | truth | corr(e_HMC, e_OLS) | sd(diff)/sd(e) | rho1 (HMC) | BP p (HMC) | var ratio (HMC) | #k-hat>0.7 |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for _, r in hmc.iterrows():
            lines.append(f"| {r.snr} | {r.identified} | {r.truth_recovered} | {r.corr_hmc_vs_ols_selected:.6f} | "
                         f"{r.rel_diff_hmc_vs_ols_selected:.2e} | {r.hmc_rho1:.2f} | {r.hmc_bp_p:.2g} | {r.hmc_var_ratio:.2f} | {r.n_khat_gt_0p7} |")
    with open(os.path.join(TAB, "summary_tables.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    vif = pd.read_csv(os.path.join(RES, "exp1_vif_long.csv"))
    kappa = pd.read_csv(os.path.join(RES, "exp1_kappa.csv"))
    meta = pd.read_csv(os.path.join(RES, "exp1_screening_meta.csv"))
    geo = pd.read_csv(os.path.join(RES, "exp1_geometry.csv"))
    with open(os.path.join(RES, "exp1_near_dependencies.json")) as f:
        ndep = json.load(f)
    stats = pd.read_csv(os.path.join(RES, "exp2_stats.csv"))
    ex = load_examples(RES)
    hmc_path = os.path.join(RES, "exp3_hmc_check.csv")
    hmc = pd.read_csv(hmc_path) if os.path.exists(hmc_path) else None
    if not _args.tables_only:
        fig_vif_redesign(vif)
        fig_collinearity_source(kappa, geo)
        fig_heteroscedasticity(stats, ex)
    cvn = pd.read_csv(os.path.join(RES, "exp1_clean_vs_n.csv"))
    tables(vif, kappa, geo, stats, meta, ndep, hmc, cvn)
    print("figures ->", FIG, "\ntables  ->", TAB)


if __name__ == "__main__":
    main()
