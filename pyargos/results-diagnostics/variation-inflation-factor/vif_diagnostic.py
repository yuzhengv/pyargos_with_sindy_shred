# %%
import importlib
import os
import sys

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

import vif_process_fun

# %%
importlib.reload(vif_process_fun)

# %%
# data_dir_n = "/Users/yuzhengzhang/Documents-Local/Code-Local/pyargos_with_sindy_shred/pyargos/data/n-analysis-new"
# data_dir_snr = "/Users/yuzhengzhang/Documents-Local/Code-Local/pyargos_with_sindy_shred/pyargos/data/snr-analysis-new"
# out_dir_path = "/Users/yuzhengzhang/Documents-Local/Code-Local/pyargos_with_sindy_shred/pyargos/results-diagnostics/variation-inflation-factor/results"

# %%
# data_dir_n = (
#     "/home/yuzheng/Projects/pyargos_with_sindy_shred/pyargos/data/n-analysis"
# )
# data_dir_snr = (
#     "/home/yuzheng/Projects/pyargos_with_sindy_shred/pyargos/data/snr-analysis"
# )
# out_dir_path = "/home/yuzheng/Projects/pyargos_with_sindy_shred/pyargos/results-diagnostics/variation-inflation-factor/results"

# %%
data_dir_n = (
    "/nobackup/qtzk83/Projects/pyargos_with_sindy_shred/pyargos/data/n-analysis"
)
data_dir_snr = "/nobackup/qtzk83/Projects/pyargos_with_sindy_shred/pyargos/data/snr-analysis"
out_dir_path = "/nobackup/qtzk83/Projects/pyargos_with_sindy_shred/pyargos/results-diagnostics/variation-inflation-factor/results"

# ! -------------------- For Different N --------------------
# %%
# -Aizawa System
system_name = "aizawa"
equation_name = "equation_3"

# %%
chosen_df, variables = vif_process_fun.get_chosen_var_vif(
    data_dir=data_dir_n,
    pattern=f"{system_name}_model_n_10",
    equation_name=equation_name,
)

# %%
chosen_df, variables = vif_process_fun.get_chosen_var_vif(
    data_dir=data_dir_n,
    pattern=f"{system_name}_model_n_10",
    equation_name=equation_name,
    manual_common_vars=[
        "const",
        "x3",
        "x1^2",
        "x2^2",
        "x3^3",
        "x1^2 x3",
        "x2^2 x3",
        "x1^3 x3",
        "x2^2 x3^2",
    ],
)

chosen_df = vif_process_fun.postprocess_chosen_df_n(chosen_df)
plot_df = vif_process_fun.postprocess_for_plotting(chosen_df)

# %%
vif_process_fun.plot_vif_results(
    df=plot_df,
    variables=variables,
    out_dir=out_dir_path,
    system_name=system_name,
    figsize=(14, 14),
    font_scale=2,
    use_log_scale=True,
    n_col="n",
)

# ! -------------------- For Different SNR --------------------
# %%
# -Aizawa System
system_name = "aizawa"
equation_name = "equation_3"

# %%
chosen_df, variables = vif_process_fun.get_chosen_var_vif(
    data_dir=data_dir_snr,
    pattern=f"{system_name}_model_snr",
    equation_name=equation_name,
)

# %%
chosen_df = vif_process_fun.postprocess_chosen_df_snr(chosen_df)
plot_df = vif_process_fun.postprocess_for_plotting(chosen_df)

vif_process_fun.plot_vif_results(
    df=plot_df,
    variables=variables,
    out_dir=out_dir_path,
    system_name=system_name,
    use_log_scale=True,
    n_col="snr",
)

# %%
importlib.reload(vif_process_fun)

# %%
# ! -------------------- Organized N and SNR --------------------
# - Aizawa System
# --- for n ---
system_name = "aizawa"
equation_name = "equation_3"

chosen_df_n, variables = vif_process_fun.get_chosen_var_vif(
    data_dir=data_dir_n,
    pattern=f"{system_name}_model_n_10",
    equation_name=equation_name,
    manual_common_vars=[
        "const",
        "x1",
        "x2",
        "x3",
        "x1^2",
        "x2^2",
        "x3^2",
        "x1 x2",
        "x1 x3",
        "x2 x3",
        "x3^3",
        "x1^2 x3",
        "x2^2 x3",
        "x1^3 x3",
        "x1^2 x2^2",
        "x1^2 x3^2",
        "x2^2 x3^2",
        "x1^4",
        "x2^4",
        "x3^4",
    ],
)
chosen_df_n = vif_process_fun.postprocess_chosen_df_n(chosen_df_n)
plot_df_n = vif_process_fun.postprocess_for_plotting(chosen_df_n, n_col="n")

# --- for snr ---
chosen_df_snr, _ = vif_process_fun.get_chosen_var_vif(
    data_dir=data_dir_snr,
    pattern=f"{system_name}_model_snr",
    equation_name=equation_name,
    manual_common_vars=[
        "const",
        "x1",
        "x2",
        "x3",
        "x1^2",
        "x2^2",
        "x3^2",
        "x1 x2",
        "x1 x3",
        "x2 x3",
        "x3^3",
        "x1^2 x3",
        "x2^2 x3",
        "x1^3 x3",
        "x1^2 x2^2",
        "x1^2 x3^2",
        "x2^2 x3^2",
        "x1^4",
        "x2^4",
        "x3^4",
    ],
)
chosen_df_snr = vif_process_fun.postprocess_chosen_df_snr(chosen_df_snr)
plot_df_snr = vif_process_fun.postprocess_for_plotting(
    chosen_df_snr, n_col="snr"
)
# --- side-by-side plot ---
vif_process_fun.plot_vif_n_and_snr_using_two_grids(
    df_n=plot_df_n,
    df_snr=plot_df_snr,
    variables=variables,
    out_dir=out_dir_path,
    system_name=system_name,
    use_log_scale_n=True,  # match your n-plot choice
    use_log_scale_snr=True,  # match your snr-plot choice
    display_log_n=False,
)

# %%
# - Chenlee System
# --- for n ---
system_name = "chenlee"
equation_name = "equation_1"

chosen_df_n, variables = vif_process_fun.get_chosen_var_vif(
    data_dir=data_dir_n,
    pattern=f"{system_name}_model_n_10",
    equation_name=equation_name,
    manual_common_vars=["const", "x1", "x2", "x3", "x2 x3"],
)
chosen_df_n = vif_process_fun.postprocess_chosen_df_n(chosen_df_n)
plot_df_n = vif_process_fun.postprocess_for_plotting(chosen_df_n, n_col="n")

# --- for snr ---
chosen_df_snr, _ = vif_process_fun.get_chosen_var_vif(
    data_dir=data_dir_snr,
    pattern=f"{system_name}_model_snr",
    equation_name=equation_name,
    manual_common_vars=["const", "x1", "x2", "x3", "x2 x3"],
)
chosen_df_snr = vif_process_fun.postprocess_chosen_df_snr(chosen_df_snr)
plot_df_snr = vif_process_fun.postprocess_for_plotting(
    chosen_df_snr, n_col="snr"
)
# --- side-by-side plot ---
vif_process_fun.plot_vif_n_and_snr_using_two_grids(
    df_n=plot_df_n,
    df_snr=plot_df_snr,
    variables=variables,
    out_dir=out_dir_path,
    system_name=system_name,
    use_log_scale_n=False,  # match your n-plot choice
    use_log_scale_snr=False,  # match your snr-plot choice
)

# %%
# - Rossler System
# --- for n ---
system_name = "rossler"
equation_name = "equation_3"

chosen_df_n, variables = vif_process_fun.get_chosen_var_vif(
    data_dir=data_dir_n,
    pattern=f"{system_name}_model_n_10",
    equation_name=equation_name,
)
chosen_df_n = vif_process_fun.postprocess_chosen_df_n(chosen_df_n)
plot_df_n = vif_process_fun.postprocess_for_plotting(chosen_df_n, n_col="n")

# --- for snr ---
chosen_df_snr, _ = vif_process_fun.get_chosen_var_vif(
    data_dir=data_dir_snr,
    pattern=f"{system_name}_model_snr",
    equation_name=equation_name,
)
chosen_df_snr = vif_process_fun.postprocess_chosen_df_snr(chosen_df_snr)
plot_df_snr = vif_process_fun.postprocess_for_plotting(
    chosen_df_snr, n_col="snr"
)
# --- side-by-side plot ---
vif_process_fun.plot_vif_n_and_snr_using_two_grids(
    df_n=plot_df_n,
    df_snr=plot_df_snr,
    variables=variables,
    out_dir=out_dir_path,
    system_name=system_name,
    use_log_scale_n=False,
    use_log_scale_snr=False,
    display_log_n=False,
)

# %%
