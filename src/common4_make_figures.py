import os
from itertools import product
import pandas as pd
import numpy as np
import tabulate
import matplotlib.pyplot as plt
import matplotlib.ticker
import seaborn as sns


def edit_results_df(results):

    # a = results.loc[(results["method"] == "sindy") & (results["data_size"] == "large") & \
    #                   (results["snr"] == "30"), "test_TE_nanmean"]

    results['data_size'] = results['data_size'].replace({'small': 'Small', 'large': 'Large'})
    results['snr'] = results['snr'].replace({'None': 'clean', '30': '30 dB', '13': '13 dB'})
    results['method'] = results['method'].replace({'proged': 'ProGED', 'sindy': 'SINDy', 'dso': 'DSO'})

    for icol in ['method', 'exp_version', 'data_size', 'system', 'snr']:
        results[icol] = results[icol].astype('category')

    results['method'] = results['method'].cat.reorder_categories(['ProGED', 'DSO', 'SINDy'])
    results['snr'] = results['snr'].cat.reorder_categories(['clean', '30 dB', '13 dB'])

    # rename test_TE_nanmean to test_TE
    results.rename(columns={'test_TE_nanmean': 'TE',
                            'complexity_norm': 'Complexity_norm'}, inplace=True)

    # transform all infs to nans
    results.replace([np.inf, -np.inf], np.nan, inplace=True)

    return results


def make_summary_of_results(results, path_to_save_results=None):

    results_edited = results.groupby(['method', 'system', 'snr', 'data_size']).apply(custom_aggregation).reset_index(drop=True)

    # Then average test_TE_nanmean, TD_manual, TD_auto, complexity_norm over all systems, still grouped by method, snr, data_size
    summary_over_systems = results_edited.groupby(['method', 'snr', 'data_size']).agg(
                            {'TE': [np.nanmean, np.nanstd], 'TD_manual': [np.nanmean, np.nanstd], 'TD_auto': [np.nanmean, np.nanstd],
                             'Complexity_norm': [np.nanmean, np.nanstd], 'TD_manual_norm': [np.nanmean, np.nanstd],
                             'TD_auto_norm': [np.nanmean, np.nanstd]}).reset_index(drop=False)

    # save
    if path_to_save_results is not None:
        summary_over_systems.to_csv(f"{path_to_save_results}best_results_{exp_version}_{exp_type}_summary_over_systems.csv", sep=',', index=True)
        content = tabulate.tabulate(summary_over_systems, headers="keys", tablefmt="plain", showindex="always")
        with open(f"{path_to_save_results}best_results_{exp_version}_{exp_type}_summary_over_systems_pretty.txt", "w") as f:
            f.write(content)
    return results_edited, summary_over_systems


def custom_aggregation(group):
    # Define your custom aggregation logic here
    te_sum = group['TE'].sum(skipna=False)
    if not np.isnan(te_sum):
        td_manual_sum = group['TD_manual'].sum()
        td_auto_sum = group['TD_auto'].sum()
        complexity_norm_sum = group['Complexity_norm'].sum()
        td_manual_norm_sum = group['TD_manual_norm'].sum()
        td_auto_norm_sum = group['TD_auto_norm'].sum()
        expr_list = group['expr'].tolist()  # Assuming 'expr' is a list column
    else:
        td_manual_sum = np.nan
        td_auto_sum = np.nan
        complexity_norm_sum = np.nan
        td_manual_norm_sum = np.nan
        td_auto_norm_sum = np.nan
        expr_list = np.nan

    # Create a new DataFrame with the aggregated values
    result = pd.DataFrame({
        'exp_version': [group['exp_version'].iloc[0]],  # Assuming 'exp_version' is a non-list column
        'method': [group['method'].iloc[0]],
        'system': [group['system'].iloc[0]],
        'snr': [group['snr'].iloc[0]],
        'data_size': [group['data_size'].iloc[0]],
        'TE': [te_sum],
        'TD_manual': [td_manual_sum],
        'TD_auto': [td_auto_sum],
        'Complexity_norm': [complexity_norm_sum],
        'TD_manual_norm': [td_manual_norm_sum],
        'TD_auto_norm': [td_auto_norm_sum],
        'expr': [expr_list],
    })

    return result

def set_plot_settings():
    plt.rc('font', size=8)  # controls default text sizes
    plt.rc('axes', titlesize=8)  # fontsize of the axes title
    plt.rc('axes', labelsize=8)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=8)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=8)  # fontsize of the tick labels
    plt.rc('legend', fontsize=8)  # legend fontsize
    plt.rc('xtick', direction='in')
    plt.rc('ytick', direction='in')


def plot_boxplot_over_systems(results, exp_version, exp_specific_settings, base_name, path_to_save_figures=None):

    columns_to_check = ["TE", "TD_manual_norm", "Complexity_norm"]
    nan_counts = results_edited.groupby(["method", "data_size", "snr"])[columns_to_check].apply(
        lambda x: x.isna().sum())

    capsize = 0.02
    hue = "method"
    hue_order = ['ProGED', 'DSO', 'SINDy']
    palette = sns.color_palette('deep')[:len(hue_order)]
    estimator = 'nanmedian'

    results_small = results[results.data_size == 'Small']
    results_large = results[results.data_size == 'Large']

    fig, axes = plt.subplots(3, 2)
    plt.subplots_adjust(wspace=0, hspace=0)
    [ax.spines[spine].set_linewidth(0.5) for ax in axes.flatten() for spine in ['left', 'right', 'top', 'bottom']]

    g = sns.boxplot(ax=axes[0, 0], x="snr", y="TE", data=results_small,
                    hue=hue, hue_order=hue_order, palette=palette, width=0.75, linewidth=0.5,
                    flierprops={'markersize': 2})
    # get count for sindy, large, 13 db
    if exp_version == "e1":
        g.text(1.3, 10, f'{10 - nan_counts.loc["SINDy", "Small", "30 dB"]["TE"]}', fontsize=5, color='red')  # for e2
        g.text(2.3, 10, f'{10 - nan_counts.loc["SINDy", "Small", "13 dB"]["TE"]}', fontsize=5, color='red')# for e2
    elif exp_version == "e2":
        g.text(1.3, 10, f'{10 - nan_counts.loc["SINDy", "Small", "30 dB"]["TE"]}', fontsize=5, color='red')  # for e2
        g.text(2.3, 10, f'{10 - nan_counts.loc["SINDy", "Small", "13 dB"]["TE"]}', fontsize=5, color='red')  # for e2

    g.axhline(1, color='black', linestyle='--', linewidth=0.5)
    g.set(xlabel="", ylim=exp_specific_settings["ylim_TE"],
          yscale="log")  # , ylim=[0, 20], ylabel=np.arange(0, 19, 5)) #np.arange(0, 20, 5)
    g.set_ylabel("Trajectory Error", labelpad=5)
    g.set_title("Small", pad=5)
    g.xaxis.set_tick_params(length=0)
    g.legend_.remove()

    g = sns.boxplot(ax=axes[0, 1], x="snr", y="TE", data=results_large,
                    hue=hue, hue_order=hue_order, palette=palette, width=0.75, linewidth=0.5,
                    flierprops={'markersize': 2})
    g.axhline(1, color='black', linestyle='--', linewidth=0.5)

    if exp_version == "e1":
        g.text(1.3, 10, f'{10 - nan_counts.loc["SINDy", "Small", "30 dB"]["TE"]}', fontsize=5, color='red')  # for e2
        g.text(2.3, 10, f'{10 - nan_counts.loc["SINDy", "Small", "13 dB"]["TE"]}', fontsize=5, color='red')# for e2
        g.text(0.28, 10 ** -4.8, r'$5.19 \cdot 10^{-10}$', fontsize=5)  # for e2
    elif exp_version == "e2":
        g.text(1.3, 10, f'{10 - nan_counts.loc["SINDy", "Small", "30 dB"]["TE"]}', fontsize=5, color='red')  # for e2
        g.text(2.3, 10, f'{10 - nan_counts.loc["SINDy", "Small", "13 dB"]["TE"]}', fontsize=5, color='red')  # for e2
        g.text(0.02, 10 ** -4.8, r'$5.19 \cdot 10^{-10}$', fontsize=5)  # for e2

    g.set(ylabel="", xlabel="", ylim=exp_specific_settings["ylim_TE"], yscale="log", yticklabels="")
    g.set_title("Large", pad=5)
    g.xaxis.set_tick_params(length=0)
    g.legend_.remove()

    g = sns.boxplot(ax=axes[1, 0], x="snr", y="TD_manual_norm", data=results_small,
                    hue=hue, hue_order=hue_order, palette=palette, width=0.75, linewidth=0.5,
                    flierprops={'markersize': 2})

    g.set(xlabel="", ylim=exp_specific_settings["ylim_TD"], yticks=exp_specific_settings["yticks_TD"])  # , yticklabels=['0', '', '2', '', '4', '', '6', '', '8', '', '']) #np.arange(0, 20, 5)
    g.set_ylabel("Normalized\nTerm Difference", labelpad=5)
    g.xaxis.set_tick_params(length=0)
    g.legend_.remove()

    g = sns.boxplot(ax=axes[1, 1], x="snr", y="TD_manual_norm", data=results_large,
                    hue=hue, hue_order=hue_order, palette=palette, width=0.75, linewidth=0.5,
                    flierprops={'markersize': 2})
    g.set(ylabel="", xlabel="", ylim=exp_specific_settings["ylim_TD"], yticks=exp_specific_settings["yticks_TD"], yticklabels="")
    # g.text(2.2, 26, r'$53$', fontsize=5)
    g.xaxis.set_tick_params(length=0)
    g.legend_.remove()

    g = sns.boxplot(ax=axes[2, 0], x="snr", y="Complexity_norm", data=results_small,
                    hue=hue, hue_order=hue_order, palette=palette, width=0.75, linewidth=0.5,
                    flierprops={'markersize': 2})
    g.axhline(1, color='black', linestyle='--', linewidth=0.5)
    g.set(xlabel="", yticks=exp_specific_settings["yticks_NC"], ylim=exp_specific_settings["ylim_NC"],
          yticklabels=exp_specific_settings["yticklabels_NC"])  # np.arange(0, 20, 5)
    g.set_ylabel("Normalized\nComplexity", labelpad=5)
    g.xaxis.set_tick_params(length=0)
    g.legend_.remove()

    g = sns.boxplot(ax=axes[2, 1], x="snr", y="Complexity_norm", data=results_large,
                    hue=hue, hue_order=hue_order, palette=palette, width=0.75, linewidth=0.5,
                    flierprops={'markersize': 2})
    g.axhline(1, color='black', linestyle='--', linewidth=0.5)
    if exp_version == "e2":
        g.text(1.26, 12, r'$61$', fontsize=5)  # for e2
        g.text(2.26, 12, r'$70$', fontsize=5)  # for e2
    g.set(ylabel="", xlabel="",  yticks=exp_specific_settings["yticks_NC"], ylim=exp_specific_settings["ylim_NC"], yticklabels="")
    g.xaxis.set_tick_params(length=0)
    plt.legend(title="", loc="center", bbox_to_anchor=(0, -0.3), frameon=True, ncol=3, fontsize=8)

    plt.show()

    if path_to_save_figures is not None:
        os.makedirs(path_to_save_figures, exist_ok=True)
        plt.savefig(f"{path_to_save_figures}fig_boxplot_over_systems_{base_name}.png", dpi=300, bbox_inches='tight')
        plt.close()

    return True


##
# MAIN
##
exp_version = "e1"
plot_version = "p1"
exp_type = 'sysident_num_full'
methods = ["proged", "sindy", "dso"]
data_sizes = ["small", "large"]
snrs = ['None', '30', '13']
obs = "full"
metric_val = "mean"

root_dir = "D:\\Experiments\\symreg_methods_comparison"
path_results = f"{root_dir}{os.sep}analysis{os.sep}{exp_type}{os.sep}{exp_version}{os.sep}"
path_figures = f"{path_results}figures{os.sep}"
fig_base_name = f"{exp_version}_{exp_type}_{plot_version}"
fname_results_in = f"best_results_{exp_version}_withValTE{metric_val}_withTestTE_withTD_compl.csv"
results = pd.read_csv(f"{path_results}{fname_results_in}", sep=",")

# edit names & values of categorical columns
results = edit_results_df(results)

# make summary of results ( also return results joined by equations of each model)
results_edited, summary = make_summary_of_results(results, path_to_save_results=path_results)

## plotting
set_plot_settings()

# for each method, data size and SNR, find number of invalid results (NaN, -inf, inf) for each metric (TE, TD, complexity)
columns_to_check = ["TE", "TD_manual_norm", "Complexity_norm"]
nan_counts = results_edited.groupby(["method", "data_size", "snr"])[columns_to_check].apply(
    lambda x: x.isna().sum())

if exp_version == "e1":
    exp_specific_settings = {"ylim_TE": [10 ** -5, 10 ** 4],
                             "ylim_TD": [-0.5, 15],
                             "yticks_TD": np.arange(0, 15, 5),
                             "yticks_NC": np.arange(0, 16, 2),
                             "ylim_NC": [0, 16],
                             "yticklabels_NC": np.array(['0', '', '4', '', '8', '', '12', ''])
                             }
else:
    exp_specific_settings = {"ylim_TE": [10 ** -5, 10 ** 4],
                             "ylim_TD": [-0.5, 23],
                             "yticks_TD": np.arange(0, 25, 5),
                             "yticks_NC": np.arange(0, 13, 2),
                             "ylim_NC": [0, 13],
                             "yticklabels_NC": np.array(['0', '', '4', '', '8', '', '12'])
                             }

plot_boxplot_over_systems(results_edited, exp_version, exp_specific_settings, base_name=fig_base_name, path_to_save_figures=path_figures)

## miscelanoeus -- TABLE 3 -- NOT YET EDITED
# # group results by method, data size and SNR, and check how many there is TD == 0
# grouped = results.groupby(["method", "data_size", "snr"])
# num_TD_is_zero = grouped.apply(lambda x: x[x["TD_manual_norm"] == 0].shape[0])
# num_TE_below_one = grouped.apply(lambda x: x[x["test_TE_nanmean"] < 0.1].shape[0])
#
# ## get best models for each data size and snr based on test_NE_nanmean
# grouped = results_edited.groupby(["data_size", "snr", "system"])
# best_models = grouped.apply(lambda x: x[x["TE"] == x["TE"].min()]).reset_index(drop=True)
# # remove snr = 13 from best_models
# best_models = best_models[best_models["snr"] != "13 dB"]
# best_models = best_models[best_models["snr"] != "30 dB"]
# # remove data_size = small
# best_models = best_models[best_models["data_size"] != "Small"]
#
# # order hierarchically by system then by snr
# best_models = best_models.sort_values(by=["system", "snr"], ascending=[True, False]).reset_index(drop=True)
#
# # only keep columns method, system, snr, TE, TD_manual_norm, Complexity_norm and expression
# best_models = best_models[["method", "system", "snr", "TE", "TD_manual_norm", "Complexity_norm", "expr"]].reset_index(drop=True)
#
# def round_constants(expr, n=3):
#     """takes sympy expression or expression string and rounds all numerical constants to n decimal spaces"""
#     if isinstance(expr, str):
#         if len(expr) == 0:
#             return expr
#         else:
#             expr = sp.sympify(expr)
#
#     for a in sp.preorder_traversal(expr):
#         if isinstance(a, sp.Float):
#             expr = expr.subs(a, round(a, n))
#     return expr
#
# # sympy each expression
# import sympy as sp
# for imodel in range(best_models.shape[0]):
#     edited_model = []
#     for iexpr in range(len(best_models.loc[imodel, 'expr'])):
#         expr_sym = best_models.loc[imodel, 'expr'][iexpr]
#         expr_sym2 = round_constants(sp.sympify(expr_sym), n=3)
#         edited_model.append(expr_sym2)
#     best_models.loc[imodel, 'expr'] = str(edited_model)
#
# # save best models to csv pretty using tabulate
# from tabulate import tabulate
# tabulate_content = tabulate(best_models, headers='keys', tablefmt='psql')
# with open(f"{path_results}best_models_{exp_version}_pretty.txt", "w") as f:
#     f.write(tabulate_content)
#
#
# ## TIME DURATION
#
#
#
# import pandas as pd
# import numpy as np
# import matplotlib.patheffects as path_effects
#
#
# def make_df(etype, methods, data_len, keys):
#     # Create an empty DataFrame
#     df = pd.DataFrame()
#
#     # Populate the DataFrame with the desired columns and values
#     for method in methods:
#         for length in data_len:
#             # Get the appropriate dictionary based on the method
#             dictionary = eval(f"{etype}_{method}")
#
#             # Create a new row for each key in the dictionary
#             for key in keys:
#                 row = {
#                     'methods': method,
#                     'exp_type': etype,
#                     'keys': key,
#                     'data_len': length,
#                     'duration': dictionary[key]
#                 }
#
#                 # Append the row to the DataFrame
#                 df = df.append(row, ignore_index=True)
#     return df
#
# def add_median_labels(ax, fmt='.1f'):
#     lines = ax.get_lines()
#     boxes = [c for c in ax.get_children() if type(c).__name__ == 'PathPatch']
#     lines_per_box = int(len(lines) / len(boxes))
#     for median in lines[4:len(lines):lines_per_box]:
#         x, y = (data.mean() for data in median.get_data())
#         y_final = y*2 if x != 1.0 else y*2.5
#         y_final = y_final if y_final < 10**6 else y_final*0.75
#         print(f"x: {x}, y_final: {y_final}")
#         # choose value depending on horizontal or vertical plot orientation
#         value = x if (median.get_xdata()[1] - median.get_xdata()[0]) == 0 else y
#         text = ax.text(x, y_final, f'{round(value):{fmt}}', ha='center',
#                        fontweight='normal', color='black', size=7)
#
#
#         # create median-colored border around white text for contrast
#         # text.set_path_effects([
#         #   #  path_effects.Stroke(linewidth=2, foreground=median.get_color()),
#         #     path_effects.Normal(),
#         # ])
# ##
# exp_version = "e1"
# proged_to_use = "proged_parallel"
#
# fullobs_sindy = {'inf-100': [510.787101, 1274.296720],
#                  'inf-2000': [1018.450821, 2543.801843],
#                  '30-100': [1347.343412, 2904.045797],
#                  '30-2000': [1632.476004, 2567.257607],
#                  '13-100': [494.229669, 765.183265],
#                  '13-2000': [1965.618718, 3040.601036]}
#
# fullobs_proged = {'inf-100': [840606.994904, 305092.495095],
#                   'inf-2000': [762397.087820, 296794.918797],
#                   '30-100': [771256.740116, 283178.490009],
#                   '30-2000': [672926.421836, 291248.540851],
#                   '13-100': [703258.948137, 308367.747996],
#                   '13-2000': [688299.564252, 353075.972680]}
# fullobs_proged_parallel = {}
# for key, value in fullobs_proged.items():
#     fullobs_proged_parallel[key] = [x / 100 for x in value]
# fullobs_dso = {'inf-100': [8037.408680, 9315.128108],
#                'inf-2000': [19350.115435, 11185.515194],
#                '30-100': [6744.577407, 4434.849635],
#                '30-2000': [16035.705884, 6672.054803],
#                '13-100': [3125.029587, 1889.641980],
#                '13-2000': [9691.157406, 2895.588048]}
#
# methods = [proged_to_use, 'dso', 'sindy']
# data_len = ['100', '2000']
# keys = list(fullobs_proged.keys())
# fullobs_df = make_df('fullobs', methods, data_len, keys)
#
# # Split "duration" column into "duration_median" and "duration_mad" columns
# fullobs_df[['duration_median', 'duration_mad']] = fullobs_df['duration'].apply(lambda x: pd.Series([np.median(x), np.nan] if len(x) == 1 else x))
#
#
#
# ## par obs
#
# parobs_proged = {'inf-s': [6262701.715528965, 193587.62830495834], 'inf-xy': [2712753.7693083286, 127817.55700707436], '30-s': [6113485.122621059, 112254.73761558533], '30-xy': [3120976.8472611904, 291282.56157040596], '13-s': [6237037.079036236, 120908.10507535934], '13-xy': [3139433.6070120335, 917674.5857298374]}
# parobs_gpomo = {'inf-s': [16.4185, 3.8935000000000013], 'inf-xy': [286.63874999999996, 79.46824999999998], '30-s': [21.582, 3.6792499999999997], '30-xy': [281.18925, 78.39874999999998], '13-s': [26.07925, 8.197500000000002], '13-xy': [265.3825, 53.394000000000005]}
# parobs_lodefind = {'inf-s': [3.9436825], 'inf-xy': [6.576113], '30-s': [3.471425], '30-xy': [6.284127], '13-s': [3.151224], '13-xy': [6.031139]}
# parobs_proged_parallel = {}
# for key, value in parobs_proged.items():
#     parobs_proged_parallel[key] = [x / 100 for x in value]
# methods = [proged_to_use, 'gpomo', 'lodefind']
# data_len = ['merged']
# keys = list(parobs_proged.keys())
# parobs_df = make_df('parobs', methods, data_len, keys)
#
# # Split "duration" column into "duration_median" and "duration_mad" columns
# parobs_df[['duration_median', 'duration_mad']] = parobs_df['duration'].apply(lambda x: pd.Series([np.median(x), np.nan] if len(x) == 1 else x))
#
# fullobs_df['methods'] = fullobs_df['methods'].astype('category')
# fullobs_df['methods'] = fullobs_df['methods'].cat.reorder_categories([proged_to_use, 'dso', 'sindy'])
# fullobs_df['methods'] = fullobs_df['methods'].cat.rename_categories({proged_to_use: 'ProGED',
#                                                                      'dso': 'DSO',
#                                                                      'sindy': 'SINDy'})
#
# parobs_df['methods'] = parobs_df['methods'].astype('category')
# parobs_df['methods'] = parobs_df['methods'].cat.reorder_categories([proged_to_use, 'gpomo', 'lodefind'])
# parobs_df['methods'] = parobs_df['methods'].cat.rename_categories({ proged_to_use: 'ProGED',
#                                                                     'gpomo': 'GPoM',
#                                                                     'lodefind': 'L-ODEfind'})
# ## PLOT
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# plt.rc('font', size=10)          # controls default text sizes
# plt.rc('axes', titlesize=10)     # fontsize of the axes title
# plt.rc('axes', labelsize=10)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=10)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=10)    # fontsize of the tick labels
# plt.rc('legend', fontsize=10)    # legend fontsize
# plt.rc('xtick', direction='in')
# plt.rc('ytick', direction='in')
#
# if proged_to_use == 'proged':
#     palette_fullobs = sns.color_palette('deep')[:3]
#     palette_parobs = [sns.color_palette('deep')[i] for i in [0, 3, 4]]
#
# else:
#     palette_fullobs = [sns.color_palette('pastel')[0]] + sns.color_palette('deep')[1:3]
#     palette_parobs = [sns.color_palette('pastel')[0]] + [sns.color_palette('deep')[i] for i in [3, 4]]
#
#
# fig, axes = plt.subplots(1, 2, sharey=True)
# plt.subplots_adjust(wspace=0, hspace=0)
#
# g = sns.boxplot(ax=axes[0], x="methods", y="duration_median", data=fullobs_df, palette=palette_fullobs,
#                 width=0.75, linewidth=0.5, flierprops={'markersize': 2})
# #g = sns.stripplot(data=fullobs_df, x="methods", y="duration_median", dodge=False, palette=palette_fullobs,
# #                  alpha=0.5, edgecolor='black', linewidth=0.1, ax=axes[0])
#
# g.set(xlabel="", ylabel="Duration / seconds", yscale="log", ylim=[1, 1e7], title="Full observability")
# add_median_labels(axes[0])
# g.set_ylim(top=1e7)
#
# g = sns.boxplot(ax=axes[1], x="methods", y="duration_median", data=parobs_df, palette=palette_parobs,
#                 width=0.75, linewidth=0.5, flierprops={'markersize': 2})
# #g = sns.stripplot(data=parobs_df, x="methods", y="duration_median", dodge=False, palette=palette_parobs,
# #                  alpha=0.5, edgecolor='black', linewidth=0.1, ax=axes[1])
# g.set(xlabel="", ylabel="", yscale="log",  ylim=[1, 1e7], title="Partial observability")
# add_median_labels(axes[1])
#
# path_out_plots = f"D:\\Experiments\\symreg_methods_comparison\\analysis\\{exp_type}\\{exp_version}\\"
# plt.savefig(f"{path_out_plots}fig_computational_time_{proged_to_use}.png", dpi=300, bbox_inches='tight')
# plt.close()
#
#
# ## overlay both images
#
#
# ##
#
# # calculate mean of TD_manual_norm for large and clean dataset
# results_TD_large_clean = results[(results['snr'] == '13') & (results['data_size'] == 'large')][['method', 'TD_manual_norm']]
# results_TD_large_clean.groupby('method').mean()
# results_TD_large_clean.groupby('method').median()
#
# # calculate correlation between TD_manual_norm and complexity_nrm
# results_TD_large_clean = results[['TD_manual_norm', 'complexity_norm']]
# results_TD_large_clean.corr(method='pearson')