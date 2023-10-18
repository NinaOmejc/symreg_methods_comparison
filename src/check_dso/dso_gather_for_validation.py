import os
import sys
import pandas as pd
import numpy as np
from src.utils.systems_collection import systems_collection


def gather_for_validation(systems, data_sizes, snrs, n_data, path_results_in, path_out, save_results=True):

    # data_size, sys_name, snr, iinit, ieq, eqsym = 'large', 'bacres', None, 0, 0, 'x'
    all_models = []

    for data_size in data_sizes:

        time_end = 20 if data_size == "large" else 10
        time_step = 0.01 if data_size == "large" else 0.1

        for sys_name in systems_collection:

            # get results(models) for validation
            directories = os.listdir(path_results_in + sys_name + os.sep)
            # take only the directories that do not contain the word ".txt"
            directories = [i for i in directories if ".txt" not in i]

            for snr in snrs:
                for iinit in range(n_data):
                    for ieq, eqsym in enumerate(systems[sys_name].state_vars):

                        # find the correct folder
                        ifolder_fname = f"{sys_name}_len{time_end}_rate{str(time_step).replace('.', '')}_snr{snr}_init{iinit}_{eqsym}_"
                        ifolder_mask = [ifolder_fname in i for i in directories]
                        ifolder_idx = np.where(ifolder_mask)[0][0]

                        # check if ifolder_idx is not empty
                        if not isinstance(ifolder_idx, np.integer):
                            print(f"Folder not found: {ifolder_fname}")
                            stop_this_system = True
                            continue
                        else:
                            path_results_in_final = f"{path_results_in}{sys_name}{os.sep}{directories[ifolder_idx]}{os.sep}"

                        # try to load the results
                        try:
                            fname_hof = f"{path_results_in_final}dso_._data_train_{data_size}_for_dso_{sys_name}_data_{sys_name}_" \
                                        f"len{time_end}_rate{str(time_step).replace('.', '')}_snr{snr}_init{iinit}_{eqsym}_0_hof.csv"
                            fname_pf = f"{path_results_in_final}dso_._data_train_{data_size}_for_dso_{sys_name}_data_{sys_name}_" \
                                        f"len{time_end}_rate{str(time_step).replace('.', '')}_snr{snr}_init{iinit}_{eqsym}_0_pf.csv"

                            results_hof = pd.read_csv(fname_hof)
                            results_pf = pd.read_csv(fname_pf)
                            print(f"Results for {ifolder_fname} loaded.")
                        except:
                            print(f"Folder doesn't have hof or pf file: {ifolder_fname}. Skipping...")
                            continue

                        # get duration of the experiment
                        fname_duration = f"{path_results_in}{sys_name}{os.sep}{method}_{sys_name}_snr{snr}_init{iinit}_{eqsym}_duration.txt"
                        durations = np.loadtxt(fname_duration)
                        duration = np.min(durations) if data_size == "small" else np.max(durations)

                        for results in [results_hof, results_pf]:
                            for ie, expr in enumerate(results.expression):
                                expr = check_expression(expr)
                                all_models.append([exp_version, method, data_size, sys_name,
                                                   "".join(systems[sys_name].state_vars), snr,
                                                   iinit, eqsym, duration, expr])

    # convert to dataframe
    results = pd.DataFrame(all_models,
                           columns=["exp_version", "method", "data_size", "system", "obs", "snr", "iinit", "eq", "duration", "expr"])

    # save all results as a dataframe
    if save_results:
        results.to_csv(f"{path_out}validation_gathered_results_{method}_{exp_version}_{exp_type}.csv", sep='\t')

    print("finished gathering results.")
    return results


def check_expression(expr):

    if 'Add' not in str(expr):
        if 'x1' in expr:
            expr = expr.replace('x1', 'x')
        if 'x2' in expr:
            expr = expr.replace('x2', 'y')
        if 'x3' in expr:
            expr = expr.replace('x3', 'z')

    return expr

##
if __name__ == "__main__":
    method = "dso"
    exp_type = 'sysident_num_full'
    exp_version = "e1"

    data_sizes = ["small", "large"]
    snrs = [None, 30, 13]  # [None, 30, 13]
    n_data = 4

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    root_dir = "D:\\Experiments\\symreg_methods_comparison"
    # add root dir to the path
    sys.path.append(root_dir)

    path_results_in = f"{root_dir}{os.sep}results{os.sep}{exp_type}{os.sep}{method}{os.sep}{exp_version}{os.sep}"
    path_out = f"{root_dir}{os.sep}analysis{os.sep}{exp_type}{os.sep}"
    os.makedirs(path_out, exist_ok=True)

    results = gather_for_validation(systems=systems_collection,
                             data_sizes=data_sizes,
                             snrs=snrs,
                             n_data=n_data,
                             path_results_in=path_results_in,
                             path_out=path_out,
                             save_results=True)





