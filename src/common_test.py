import os
import sys
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
    return results


def do_test(modeled_expr, data, eqsym, system_name):
    times = data['t']
    true_trajectory = np.array(data[eqsym])
    initial_val = true_trajectory[0]
    ieq = systems_collection[system_name].state_vars.index(eqsym)

    # true trajectories extrapolated are added to simulation so that all variables are observed while only one is simulated
    true_trajectories = data[systems_collection[system_name].state_vars]
    true_trajectories_extrapolated = interp1d(times, np.array(true_trajectories), axis=0, kind='cubic', fill_value="extrapolate")

    modeled_func = sp.lambdify(sp.symbols(['t'] + systems_collection[system_name].state_vars), modeled_expr, "numpy")
    modeled_trajectory = simulate_modeled_func(modeled_func, initial_val, times, true_trajectories_extrapolated, ieq)

    if modeled_trajectory is not np.nan:
        trajectory_error = calculate_trajectory_error(true_trajectory, modeled_trajectory)
        return trajectory_error
    else:
        return np.nan


def simulate_modeled_func(modeled_func, initial_val, times, true_trajectories_extrapolated, ieq):
    def rhs(t, x):
        b = true_trajectories_extrapolated(t)
        b[ieq] = x
        return modeled_func(t, *b)

    simulation, odeint_output = odeint(rhs, initial_val, times, rtol=1e-12, atol=1e-12, tfirst=True, full_output=True)

    if 'successful' not in odeint_output['message']:
        return np.nan
    else:
        return simulation[:, ieq]


def calculate_trajectory_error(true_trajectory, modeled_trajectory):
    return np.sqrt((np.mean((modeled_trajectory - true_trajectory) ** 2))) / np.std(true_trajectory)

##
if __name__ == "__main__":

    exp_version = "e1"
    exp_type = 'sysident_num_full'
    methods = ["proged", "sindy", "dso"]
    obs = "full"

    data_sizes = ["small", "large"]
    snrs = [None, 30, 13]  # [None, 30, 13]
    n_data = 4

    root_dir = "D:\\Experiments\\symreg_methods_comparison"
    sys.path.append(root_dir)

    path_in_gathered_results = f"{root_dir}{os.sep}analysis{os.sep}{exp_type}{os.sep}"
    results = pd.read_csv(f"{path_in_gathered_results}validation_gathered_results_{exp_version}_BEST.csv", sep='\t')

    for ibest in range(len(results)):
        system_name = results.system[ibest]
        snr = results.snr[ibest]
        data_size = results.data_size[ibest]
        iinit = results.iinit[ibest]
        eqsym = results.eq[ibest]
        modeled_expr = results.expr[ibest]

        time_end = 20 if data_size == "large" else 10
        time_step = 0.01 if data_size == "large" else 0.1
        test_errors = []
        for iinit in range(n_data):

            data_type = "test"
            data_path = f"{root_dir}{os.sep}data{os.sep}{data_type}{os.sep}{data_size}{os.sep}{system_name}{os.sep}"
            data_fname = f"data_{system_name}_len{time_end}_rate{str(time_step).replace('.', '')}_snr{snr}_init{iinit}.csv"
            data = pd.read_csv(f"{data_path}{data_fname}", sep=',')

            # do tests on test data
            test_error = do_test(modeled_expr, data, eqsym, system_name)
            test_errors.append(test_error)

        # append new column values "TE" to the results dataframe based on orig_index
        results.loc[ibest, 'test_TE'] = test_errors
        # calculate nanmean of TE for each row
        results.loc[ibest, 'test_TE_nanmean'] = results.test_TE.apply(lambda x: np.nanmean(x))


    # save results with additional validation column
    results.to_csv(f"{path_in_gathered_results}validation_gathered_results_{exp_version}_BEST_tested.csv", sep='\t', index=False)







