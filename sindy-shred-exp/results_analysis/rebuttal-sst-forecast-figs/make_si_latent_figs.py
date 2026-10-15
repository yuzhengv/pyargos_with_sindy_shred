"""Supplementary latent-space figures for experiments 123, 18 and 80 in the
house style, from the stored arrays of this folder (nothing is re-identified
or re-decoded). Reuses the panel functions of make_fig9_panels.py:

  latent_space_123, latent_space_18:
     a  GRU latent trajectory against the Bayesian-ARGOS posterior-mean
        simulation over the training period (as Fig. 9b, without a band;
        results/latent_train_<it>.npz), 18 in canvas;
     b  global maps of observations, predictions and differences at
        t = 50 ... 250 weeks (as Fig. 9c; results/fields_<it>.npz, RMSE
        checked against results/rmse_<it>.csv), 18 in canvas;
  latent_space_80:
     the overlay only, with no tag: experiment 80 illustrates a quasi-static
     level in its third coordinate.

Panels go to figures/house/si_latent_<it>_{a,b}.pdf and are composed at
183 mm by compose_si_latent.sh (scale 0.4 = 1/2.5 of the canvas).

Run:  python make_si_latent_figs.py
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "plot-style"))
import house_style as hs  # noqa: E402
from make_fig9_panels import field_maps, latent_overlay  # noqa: E402

OVERLAY_W, OVERLAY_H = 18.0, 7.2   # canvas (in) of the full-width overlay: 183 x 73 mm final


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--iterations", default="123,18,80")
    p.add_argument("--out-dir", default=os.path.join(HERE, "figures", "house"))
    a = p.parse_args()
    hs.use()
    os.makedirs(a.out_dir, exist_ok=True)
    for it in [int(v) for v in a.iterations.split(",")]:
        with_maps = it != 80
        d = np.load(os.path.join(HERE, "results", f"latent_train_{it}.npz"))
        latent_overlay(d["gru"], d["sim"], os.path.join(a.out_dir, f"si_latent_{it}_a.pdf"),
                       title=f"experiment {it}, training period", tag="a" if with_maps else None,
                       width_in=OVERLAY_W, height_in=OVERLAY_H)
        if with_maps:
            field_maps(os.path.join(HERE, "results", f"fields_{it}.npz"),
                       os.path.join(HERE, "results", f"rmse_{it}.csv"),
                       os.path.join(a.out_dir, f"si_latent_{it}_b.pdf"),
                       title=f"experiment {it}, test period", tag="b")


if __name__ == "__main__":
    main()
