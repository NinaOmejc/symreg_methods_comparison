import os
import sys
import pandas as pd
import numpy as np
from src.utils.systems_collection import systems_collection
import ProGED as pg

def gather_for_validation(systems, data_sizes, snrs, n_data, n_batches, path_results_in, fname_results_in, path_out, save_results=True):

    # data_size, sys_name, snr, iinit, ieq, eqsym, ib, imodel = 'large', 'bacres', None, 0, 0, 'x', 0, 0
    all_models = []

    for data_size in data_sizes:
        for sys_name in systems_collection:
            for snr in snrs:
                for iinit in range(n_data):
                    for ieq, eqsym in enumerate(systems[sys_name].state_vars):

                        # find the correct folder
                        for ib in range(n_batches):
                            fname_results = fname_results_in.format(sys_name, data_size, snr, iinit, ib, eqsym)

                            try:
                                models = pg.ModelBox()
                                models.load(f"{path_results_in}{sys_name}{os.sep}{fname_results}")
                                print(f"Results for {fname_results} loaded.")
                            except:
                                print(f"Results not loaded correctly: {fname_results}")


                            for imodel in range(len(models)):
                                duration = get_duration(models, imodel)
                                expr = models[imodel].get_full_expr()
                                all_models.append([exp_version, method, data_size, sys_name,
                                                   "".join(systems[sys_name].state_vars), snr, iinit, eqsym, duration, expr[0]])

                            del models

    # convert to dataframe
    results = pd.DataFrame(all_models,
                           columns=["exp_version", "method", "data_size", "system", "obs", "snr", "iinit", "eq", "duration", "expr"])

    # save all results as a dataframe
    if save_results:
        results.to_csv(f"{path_out}validation_gathered_results_{method}_{exp_version}_{exp_type}.csv", sep='\t')

    print("finished gathering results")
    return results

def get_duration(models, imodel):
    if 'duration' in list(models[imodel].estimated.keys()):
        duration = models[imodel].estimated['duration']
    else:
        duration = np.nan
    return duration


##
if __name__ == "__main__":
    method = "proged"
    exp_type = 'sysident_num_full'
    exp_version = "e2"
    obs = "full"

    data_sizes = ["small", "large"]
    snrs = [None, 30, 13]  # [None, 30, 13]
    n_data = 4
    n_batches = 100

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    root_dir = "D:\\Experiments\\symreg_methods_comparison"
    # add root dir to the path
    sys.path.append(root_dir)

    path_results_in = f"{root_dir}{os.sep}results{os.sep}{exp_type}{os.sep}{method}{os.sep}{exp_version}{os.sep}"
    fname_results_in = f"{method}_{exp_type}_{exp_version}_{{}}_train_{{}}_snr{{}}_init{{}}_obs{obs}_b{{}}_{{}}_fitted.pg"
    path_out = f"{root_dir}{os.sep}analysis{os.sep}{exp_type}{os.sep}"
    os.makedirs(path_out, exist_ok=True)
    start_time = pd.Timestamp.now()
    results = gather_for_validation(systems=systems_collection,
                                     data_sizes=data_sizes,
                                     snrs=snrs,
                                     n_data=n_data,
                                     n_batches=n_batches,
                                     path_results_in=path_results_in,
                                     fname_results_in=fname_results_in,
                                     path_out=path_out,
                                     save_results=True)

    print(f"Duration: {pd.Timestamp.now() - start_time}")



