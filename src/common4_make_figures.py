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
