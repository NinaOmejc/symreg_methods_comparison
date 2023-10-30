import os
import sys
import pandas as pd
import numpy as np
import sympy as sp
from scipy.integrate import odeint
from scipy.interpolate import interp1d
from src.utils.systems_collection import systems_collection

def test_results(results, root_dir, verbose=True):
    # create a column with empty lists for the test_TE
    results['test_TE'] = results['val_TE'].apply(lambda x: [])

    for ibest in range(len(results)):
        print(f"{ibest} / {len(results)}") if verbose else None
        system_name = results.system[ibest]
        snr = results.snr[ibest]
        data_size = results.data_size[ibest]
        eqsym = results["eq"][ibest]
        string_expr = results.expr[ibest]
        modeled_expr = round_constants(sp.sympify(string_expr), n=3)

        time_end = 20 if data_size == "large" else 10
        time_step = 0.01 if data_size == "large" else 0.1
        for iinit_test in range(n_test_data):

            data_type = "test"
            data_path = f"{root_dir}{os.sep}data{os.sep}{data_type}{os.sep}{data_size}{os.sep}{system_name}{os.sep}"
            data_fname = f"data_{system_name}_len{time_end}_rate{str(time_step).replace('.', '')}_snr{snr}_init{iinit_test}.csv"
            data = pd.read_csv(f"{data_path}{data_fname}", sep=',')

            # do tests on test data
            test_error = do_one_test(modeled_expr, data, eqsym, system_name)
            results.loc[ibest, 'test_TE'].append(test_error)

    # calculate nanmean of TE based on test_TE
    results['test_TE_nanmean'] = results['test_TE'].apply(lambda x: np.nanmean(x))
    results['test_TE_nanmean'] = results['test_TE_nanmean'].replace(np.nan, np.inf).astype(float)
    return results

def do_one_test(modeled_expr, data, eqsym, system_name):
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

def simulate_modeled_func(modeled_func, initial_val, times, true_trajectories_extrapolated, ieq):
    def rhs(t, x):
        b = true_trajectories_extrapolated(t)
        b[ieq] = x
        return modeled_func(t, *b)

    simulation, odeint_output = odeint(rhs, initial_val, times, rtol=1e-12, atol=1e-12, tfirst=True, full_output=True)

    if 'successful' not in odeint_output['message']:
        return np.nan
    else:
        return simulation


def calculate_trajectory_error(true_trajectory, modeled_trajectory):
    return np.sqrt((np.mean((modeled_trajectory.flatten() - true_trajectory.flatten()) ** 2))) / np.std(true_trajectory)


def load_and_join_results(systems, data_sizes, snrs, n_train_data, n_val_data, path_in_gathered_results, merge_func_val="nanmean", save_best_results=True):

    # check if f"{path_in_gathered_results}best_results_{exp_version}_withValTE.csv" exists
    path_to_save = os.path.dirname(os.path.dirname(path_in_gathered_results)) + os.sep
    if os.path.isfile(f"{path_to_save}best_results_{exp_version}_withValTE{merge_func_val}.csv"):
        results_best = pd.read_csv(f"{path_to_save}best_results_{exp_version}_withValTE{merge_func_val}.csv", sep='\t')
        print(f"Loading best results from {path_to_save}best_results_{exp_version}_withValTE{merge_func_val}.csv")
    else:

        # create a dataframe to store the results
        results_best = pd.DataFrame()

        if "lorenz" in systems.keys():
            n_combinations = (len(systems)-1) * len(data_sizes) * len(snrs) * n_train_data * n_val_data * 2 + \
                                            1 * len(data_sizes) * len(snrs) * n_train_data * n_val_data * 3
        else:
            n_combinations = len(systems) * len(data_sizes) * len(snrs) * n_train_data * n_val_data * 2

        # have a loop from 0 to n_combinations, but take only every fourth
        # i, iinit_train, iinit_val = 0, 0, 0
        for i in range(0, n_combinations, n_train_data * n_val_data):
            results_iconfig = pd.DataFrame()
            for iinit_train in range(n_train_data):
                results_part = pd.DataFrame()
                for iinit_val in range(n_val_data):
                    try:
                        iresult = pd.read_csv(f"{path_in_gathered_results}validation_gathered_results_{exp_version}"
                                              f"_withValTE_subset{i + n_train_data*iinit_train + iinit_val}.csv", sep='\t')
                        iresult['val_TE'] = iresult['val_TE'].apply(lambda x: [x])
                        print(f"Loading Combination i: {i + n_train_data*iinit_train + iinit_val}/{n_combinations} | init_train: {iinit_train} | init_val: {iinit_val} | snr: {iresult['snr'].unique()[0]} | data_size: {iresult['data_size'].unique()[0]} | system: {iresult['system'].unique()[0]} | eq: {iresult['eq'].unique()[0]}")
                    except:
                        print(f"Combination i: {i + n_train_data*iinit_train + iinit_val}/{n_combinations} | init_train: {iinit_train} | init_val: {iinit_val} not found.")
                        continue

                    # join iresults with the results_iconfig if iinit == 0
                    if iinit_val == 0:
                        results_part = pd.concat([results_part, iresult], axis=0)
                        # transform the val_TE column from float to a list
                    else:
                        # append the val_TE elements of each row from the iresult to the results_iconfig
                        results_part['val_TE'] = results_part['val_TE'] + iresult['val_TE']
                # append the results_part to the results_iconfig
                results_iconfig = pd.concat([results_iconfig, results_part], axis=0)

            # check if results_iconfig is empty
            if not results_iconfig.empty:
                results_iconfig.reset_index(drop=True, inplace=True)
                # calculate nanmean of val_TE, and save in the column 'val_TE_nanmean'
                if merge_func_val == "nanmean":
                    results_iconfig[f'val_TE_{merge_func_val}'] = results_iconfig['val_TE'].apply(lambda x: np.nanmean(x))
                elif merge_func_val == "mean":
                    results_iconfig[f'val_TE_{merge_func_val}'] = results_iconfig['val_TE'].apply(lambda x: np.mean(x))
                else:
                    raise ValueError(f"merge_func_val {merge_func_val} not implemented! Choose between 'nanmean' and 'mean'")

                # replace all nan values with np.inf
                results_iconfig[f'val_TE_{merge_func_val}'] = results_iconfig[f'val_TE_{merge_func_val}'].replace(np.nan, np.inf).astype(float)
                # get the best results (minimum val_TE_nanmean) for the current configuration, but for each method separately, regardless of init values
                best_iconfig = results_iconfig.loc[results_iconfig.groupby(['method'])[f'val_TE_{merge_func_val}'].idxmin()]
                # check if any is nan or inf
                if any(best_iconfig[f'val_TE_{merge_func_val}'] == np.inf):
                    print(f"best_iconfig[f'val_TE_{merge_func_val}'].isinf().any() is True!")
                    print(best_iconfig)
                    a = results_iconfig[results_iconfig['method'] == 'sindy']
                results_best = pd.concat([results_best, best_iconfig], axis=0)
            else:
                print(f"\n\n---------------\n\nCombinations {i} to {i + n_train_data * n_val_data} of {n_combinations} is empty!\n\n---------------\n\n")

        # remove columns orig_index and Unnamed: 0
        results_best = results_best.drop(columns=['orig_index', 'Unnamed: 0']).reset_index(drop=True)
        # save the results_best to a csv file
        if save_best_results:
            # get path to save the results that is one folder up from the path_in_gathered_results
            results_best.to_csv(f"{path_to_save}best_results_{exp_version}_withValTE{merge_func_val}.csv", sep='\t', index=False)

    return results_best



##
if __name__ == "__main__":

    exp_version = "e1"  # "e1" (constrained model search space) or "e2" (unconstrained model search space)
    observability = "full"  # "full" or "partial"
    simulation_type = "num" if observability == "full" else "sym"  # numerical derivation (num) or simulation by solving system of odes using initial values (sym)
    exp_type = f"sysident_{simulation_type}_{observability}"

    methods = ["proged", "sindy", "dso"]

    systems = systems_collection
    data_sizes = ["small", "large"]
    snrs = [None, 30, 13]  # [None, 30, 13]
    n_train_data = 4
    n_val_data = 4
    n_test_data = 4
    merge_func_val = "mean"

    root_dir = "D:\\Experiments\\symreg_methods_comparison"  # adjust to your root directory
    sys.path.append(root_dir)

    path_in_gathered_results = f"{root_dir}{os.sep}analysis{os.sep}{exp_type}{os.sep}{exp_version}{os.sep}val{os.sep}"

    results = load_and_join_results(systems, data_sizes, snrs, n_train_data, n_val_data,
                                    path_in_gathered_results, merge_func_val=merge_func_val, save_best_results=True)

    results_tested = test_results(results, root_dir)

    # save results with additional validation column
    path_to_save = os.path.dirname(os.path.dirname(path_in_gathered_results)) + os.sep
    results_tested.to_csv(f"{path_to_save}best_results_{exp_version}_withValTE{merge_func_val}_withTestTE.csv", sep=',', index=False)



