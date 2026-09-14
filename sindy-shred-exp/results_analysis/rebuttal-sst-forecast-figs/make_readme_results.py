"""Append the '## Results' section of README.md from the files in results/.

Run after `run_all.sh`. Replaces everything after the '## Results' heading.
The submitted-figure values are hard-coded here (read off the submitted PDFs).
"""
import argparse

import pandas as pd

SUBMITTED_FIG8C_CROP = {50: 0.866, 100: 0.779, 150: 0.833, 200: 0.756, 250: 0.804}
# per-panel RMSE printed on the submitted si-plots/multi_sensor_forecast_18.pdf (panels 0..17)
SUBMITTED_S9 = [0.0123, 0.161, 0.135, 0.135, 0.102, 0.116, 0.0927, 0.0821, 0.0877,
                0.186, 0.199, 0.092, 0.109, 0.0669, 0.092, 0.13, 0.0975, 0.0863]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--iteration", type=int, default=18)
    p.add_argument("--readme", default="README.md")
    a = p.parse_args()
    it = a.iteration
    r = pd.read_csv(f"results/rmse_{it}.csv")
    rd = pd.read_csv(f"results/rmse_{it}_dropout.csv")
    m = pd.read_csv(f"results/identified_model_{it}.csv")
    e = pd.read_csv(f"results/sensor_rmse_{it}.csv")
    d = pd.read_csv(f"results/sensor_rmse_{it}_dropout.csv")

    eqs = []
    for eq, g in m.groupby("equation", sort=False):
        eqs.append(f"{eq} = " + " ".join(
            f"{c:+.3f}{'' if t == '1' else ' ' + t.replace('x', 'z')}" for t, c in zip(g.term, g.coefficient)))

    tbl = "| t (weeks) | " + " | ".join(str(t) for t in r.t) + " |\n|---|" + "---|" * len(r) + "\n"
    tbl += "| submitted panel (crop 0–180°E, dropout active) | " + " | ".join(f"{SUBMITTED_FIG8C_CROP[t]:.3f}" for t in r.t) + " |\n"
    tbl += "| eval mode, crop 0–180°E | " + " | ".join(f"{v:.3f}" for v in r.rmse_crop_0_180E) + " |\n"
    tbl += "| **eval mode, global (revised Fig. 8c)** | " + " | ".join(f"**{v:.3f}**" for v in r.rmse_global) + " |\n"
    tbl += "| dropout active, global (one draw) | " + " | ".join(f"{v:.3f}" for v in rd.rmse_global) + " |\n"

    s9 = "| panel | sensor column | submitted (dropout active) | dropout active, this rerun | **eval mode (revised S9)** |\n|---|---|---|---|---|\n"
    for i in range(len(e)):
        s9 += f"| {i} | {e.sensor_column[i]} | {SUBMITTED_S9[i]:.4f} | {d.rmse_normalised[i]:.4f} | **{e.rmse_normalised[i]:.4f}** |\n"
    s9 += f"| mean | | {sum(SUBMITTED_S9) / len(SUBMITTED_S9):.4f} | {d.rmse_normalised.mean():.4f} | **{e.rmse_normalised.mean():.4f}** |\n"

    sec = f"""
**Identified latent model** (experiment {it}, eval mode). The manuscript's Eq. 6
rounds the same values to two decimals; repeated HMC runs move the third decimal.

```
{chr(10).join(eqs)}
```

**Fig. 8b.** Unchanged in substance: the GRU latent trajectory is identical in
train and eval mode and the identified model matches, so
`figures/comparison_plot_ba_{it}.pdf` differs from the submitted panel only by
HMC noise in the third decimal of the coefficients.

**Fig. 8c.** RMSE (°C) of the decoded forecast against the observed field:

{tbl}
Cropped and global values differ by up to 0.12 °C at a given horizon, in either
direction, so the submitted annotations were not representative of the global
error. Dropout at inference moves a single decode by up to ~0.07 °C.

**Fig. S9.** Per-sensor RMSE in the network's normalised units. The sensor
sample is the submitted one (see the `--sensor-offset` note above); the
dropout-active rerun matches the submitted values to within one draw's noise,
and the eval-mode values are the revised figure's.

{s9}
Files: `results/rmse_{it}.csv`, `results/sensor_rmse_{it}.csv`, `results/identified_model_{it}.csv`;
`_dropout` variants for the comparison run; fields and trajectories in the `.npz` files (~2 MB each).
"""
    t = open(a.readme).read()
    i = t.index("## Results\n")
    open(a.readme, "w").write(t[:i] + "## Results\n" + sec)
    print("README results section written")


if __name__ == "__main__":
    main()
