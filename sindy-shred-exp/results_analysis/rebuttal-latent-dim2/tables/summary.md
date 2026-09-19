# Latent dimension 2 versus 3: paired comparison on 20 sensor sets

- test reconstruction rRMSE (eval mode): median r=2 0.053 vs r=3 0.046; r=2 better in 5 of 20 pairs; median paired difference +0.004; Wilcoxon p = 0.027
- stage-1 pass (rRMSE <= 0.05): r=2 9 of 20, r=3 14 of 20; both 7, r=2 only 2, r=3 only 7
- Bayesian-ARGOS valid model: r=2 17, r=3 16 of 20
- Bayesian-ARGOS decoded-forecast RMSE 0-275 wk (degC), pairs valid for both: median r=2 1.351 vs r=3 0.932; r=2 better in 3 of 13 pairs; median paired difference +0.592; Wilcoxon p = 0.013
- Bayesian-ARGOS decoded-forecast RMSE 0-275 wk (degC), all pairs with a finite forecast: median r=2 1.583 vs r=3 1.079; r=2 better in 5 of 20 pairs; median paired difference +0.463; Wilcoxon p = 0.019
- Bayesian-ARGOS training latent MSE: median r=2 0.423 vs r=3 0.245; r=2 better in 8 of 20 pairs; median paired difference +0.016; Wilcoxon p = 1.000
- Bayesian-ARGOS identified period (yr): median r=2 1.033 vs r=3 1.017; r=2 larger in 12 of 20 pairs; median paired difference +0.023; Wilcoxon p = 0.030
- SINDy valid model: r=2 16, r=3 16 of 20
- SINDy decoded-forecast RMSE 0-275 wk (degC), pairs valid for both: median r=2 1.742 vs r=3 1.435; r=2 better in 5 of 13 pairs; median paired difference +0.570; Wilcoxon p = 0.146
- SINDy decoded-forecast RMSE 0-275 wk (degC), all pairs with a finite forecast: median r=2 1.903 vs r=3 1.527; r=2 better in 8 of 20 pairs; median paired difference +0.463; Wilcoxon p = 0.245
- SINDy training latent MSE: median r=2 0.388 vs r=3 0.419; r=2 better in 12 of 20 pairs; median paired difference -0.086; Wilcoxon p = 0.498
- SINDy identified period (yr): median r=2 1.037 vs r=3 1.034; r=2 larger in 10 of 20 pairs; median paired difference +0.005; Wilcoxon p = 0.165
