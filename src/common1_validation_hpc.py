import os
import sys
import itertools
import pandas as pd
import numpy as np
import sympy as sp
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from src.utils.systems_collection import systems_collection

def load_and_join_results(methods, path_in_gathered_results):
    fnames_all = os.listdir(path_in_gathered_results)
    results = pd.DataFrame()
    for method in methods:
        # get files for this method that include the name "validation_gathered_results"
        fnames = [file for file in fnames_all if method in file and "validation_gathered_results" in file]
        for fname in fnames:
            if fname.endswith(".csv"):
                gathered_results = pd.read_csv(f"{path_in_gathered_results}{fname}", sep='\t')
                results = pd.concat([results, gathered_results], axis=0)
                print(f"Results for {fname} loaded.")
    results.reset_index(drop=True, inplace=True)

    # if snr is numeric column, transform it so it will be a categorical with 3 levels, None, 30 and 13 (before np.nan, 30.0 and 13.0)
    if isinstance(results.snr[0], np.float):
        # replace nans with 0
        results['snr'] = results['snr'].fillna(0)
        # transform to categorical
        results['snr'] = results['snr'].astype('object')
        results['snr'] = results['snr'].replace([0.0, 30.0, 13.0], ['None', '30', '13'])

    return results

def round_constants(expr, n=3):
    """takes sympy expression or expression string and rounds all numerical constants to n decimal spaces"""
    if isinstance(expr, str):
        if len(expr) == 0:
            return expr
        else:
            expr = sp.sympify(expr)

    for a in sp.preorder_traversal(expr):
        if isinstance(a, sp.Float):
            expr = expr.subs(a, round(a, n))
    return expr

def do_validation(results_subset, validation_data, eqsym):
    system_name = results_subset.system[0]
    times = validation_data['t']
    true_trajectory = np.array(validation_data[eqsym])
    initial_val = true_trajectory[0]
    ieq = systems_collection[system_name].state_vars.index(eqsym)

    # true trajectories extrapolated are added to simulation so that all variables are observed while only one is simulated
    true_trajectories = validation_data[systems_collection[system_name].state_vars]
    true_trajectories_extrapolated = interp1d(times, np.array(true_trajectories), axis=0, kind='cubic',
                                              fill_value="extrapolate")

    for ires in range(len(results_subset)):
        expr_string = results_subset.expr[ires]
        # sympify modeled_expr and round all numerical constants to 3 decimal spaces
        try:
            modeled_expr = round_constants(sp.sympify(expr_string), n=3)
        except Exception as e:
            modeled_expr = sp.sympify("0")
        # make modeled_expr a function of t and the state variables
        try:
            modeled_func = sp.lambdify(sp.symbols(['t'] + systems_collection[system_name].state_vars), modeled_expr, "numpy")
            modeled_trajectory = simulate_modeled_func(modeled_func, initial_val, times, true_trajectories_extrapolated, ieq)
        except Exception as e:
            modeled_trajectory = np.nan
            print(f"Exception {e} occurred for {system_name} and {expr_string}. Simulation not possible.")

        if modeled_trajectory is not np.nan:
            trajectory_error = calculate_trajectory_error(true_trajectory, modeled_trajectory)
            results_subset.loc[ires, 'val_TE'] = trajectory_error
        else:
            results_subset.loc[ires, 'val_TE'] = np.nan

    return results_subset


def simulate_modeled_func(modeled_func, initial_val, times, true_trajectories_extrapolated, ieq):
    def rhs(t, x):
        b = true_trajectories_extrapolated(t)
        b[ieq] = x
        return modeled_func(t, *b)

    simulation, odeint_output = odeint(rhs, initial_val, times, rtol=1e-12, atol=1e-12, tfirst=True, full_output=True)

    if 'successful' not in odeint_output['message']:
        return np.nan
    else:
        return simulation.flatten()


def calculate_trajectory_error(true_trajectory, modeled_trajectory):
    return np.sqrt((np.mean((modeled_trajectory - true_trajectory) ** 2))) / np.std(true_trajectory)


def get_job_configuration(iinput, systems_collection, data_sizes, snrs, n_train_data, n_val_data):

    train_data_indices = list(range(n_train_data))
    val_data_indices = list(range(n_val_data))
    # check if lorenz is in the keys of the systems_collection and adjust the combinations of state vars accordingly
    if 'lorenz' in systems_collection.keys():
        # remove lorenz from system_collection
        new_systems_collection = systems_collection.copy()
        lorenz = {'lorenz': new_systems_collection.pop('lorenz')}
        combinations_other = list(itertools.product(new_systems_collection, data_sizes, snrs, ['x', 'y'],
                                                    train_data_indices, val_data_indices))
        combinations_lorenz = list(itertools.product(lorenz, data_sizes, snrs, ['x', 'y', 'z'],
                                                     train_data_indices, val_data_indices))
        combinations = combinations_other + combinations_lorenz
    else:
        combinations = list(itertools.product(systems_collection, data_sizes, snrs, ['x', 'y'],
                                              train_data_indices, val_data_indices))

    return combinations[iinput]


def transform_input_to_only_unsuccessful(iinput, path_to_unsuccessful):
    if path_to_unsuccessful == "":
        return iinput
    else:
        unsuccessful_runs = pd.read_csv(path_to_unsuccessful)
        return unsuccessful_runs['0'][iinput]



##
if __name__ == "__main__":


    iinput = 435  # int(sys.argv[1])
    exp_version = "e1"
    exp_type = 'sysident_num_full'
    methods = ["proged", "sindy", "dso"]
    obs = "full"

    data_sizes = ["small", "large"]
    snrs = ['None', '30', '13']
    n_train_data = 4
    n_val_data = 4

    root_dir = "D:\\Experiments\\symreg_methods_comparison"
    sys.path.append(root_dir)

    path_in_gathered_results = f"{root_dir}{os.sep}analysis{os.sep}{exp_type}{os.sep}{exp_version}{os.sep}"
    results = load_and_join_results(methods, path_in_gathered_results)
    #path_to_unsuccessful = f"{path_in_gathered_results}unsucessful_validation_subsets_e2.csv"
    #iinput = transform_input_to_only_unsuccessful(iinput, path_to_unsuccessful)

    system, data_size, snr, eqsym, iinit_train, iinit_val = get_job_configuration(iinput, systems_collection, data_sizes, snrs,
                                                                 n_train_data, n_val_data)

    time_end = 20 if data_size == "large" else 10
    time_step = 0.01 if data_size == "large" else 0.1

    # get correct rows from results, that correspond to appropriate 'system', 'snr, 'data_size'
    # The column of old index values should be named orig_index

    results_subset = results[(results["system"] == system) &
                             (results["snr"] == snr) &
                             (results["data_size"] == data_size) &
                             (results["eq"] == eqsym) &
                             (results["iinit"] == iinit_train)].reset_index(drop=False)

    # rename the column of old index values to orig_index
    results_subset.rename(columns={'index': 'orig_index'}, inplace=True)
    # create a column for the trajectory error
    results_subset['val_TE'] = [[] for _ in range(len(results_subset))]
    # remove results to save memory
    del results

    data_type = "validation"
    validation_data_path = f"{root_dir}{os.sep}data{os.sep}{data_type}{os.sep}{data_size}{os.sep}{system}{os.sep}"
    validation_data_fname = f"data_{system}_len{time_end}_rate{str(time_step).replace('.', '')}_snr{snr}_init{iinit_val}.csv"
    validation_data = pd.read_csv(f"{validation_data_path}{validation_data_fname}", sep=',')

    # do validation
    results_subset = do_validation(results_subset, validation_data, eqsym)

    # calculate nanmean of TE for each row
    # results_subset['val_TE_nanmean'] = results_subset.val_TE.apply(lambda x: np.nanmean(x))

    # save results with additional validation column
    results_subset.to_csv(f"{path_in_gathered_results}validation_gathered_results_{exp_version}_withValTE_subset{iinput}.csv",
                          sep='\t', index=False)

##


## for proged, fpr all data size, snr, groups, sum up duration (THE TRAINING DURATION)
# for each group, calculate mean and std of duration
proged_grouped = results[results['method'] == 'proged'].groupby(['data_size', 'snr', 'system', 'iinit', 'eq'])
proged_duration_504 = proged_grouped['duration'].agg([np.sum]).reset_index()
proged_duration_6 = proged_duration_504.groupby(['data_size', 'snr']).agg([np.mean, np.std]).reset_index()

# sindy
sindy_grouped = results[(results['method'] == 'sindy') & (results['eq'] == 'x')].groupby(['data_size', 'snr', 'system', 'iinit', 'eq'])
sindy_duration_240 = sindy_grouped['duration'].agg([np.sum]).reset_index()
sindy_duration_6 = sindy_duration_240.groupby(['data_size', 'snr']).agg([np.mean, np.std]).reset_index()

# dso
dso_grouped = results[(results['method'] == 'dso')].groupby(['data_size', 'snr', 'system', 'iinit', 'eq'])
dso_duration_504 = dso_grouped['duration'].agg([np.mean, np.std]).reset_index()
dso_duration_6 = dso_duration_504.groupby(['data_size', 'snr']).agg([np.mean, np.std]).reset_index()


# plot a box plots
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")
ax = sns.boxplot(x="data_size", y="duration", hue="snr", data=a, palette="Set3")
ax.set_yscale("log")
plt.show()

