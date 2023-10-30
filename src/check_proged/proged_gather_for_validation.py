# Results from the validation experiments that were parallelized on cluster are here gathered in a single dataframe.
# Besides successful runs, also unsuccessful runs are gathered in a separate dataframe.

import os
import sys
import pandas as pd
import numpy as np
from src.utils.systems_collection import systems_collection
import ProGED as pg

def gather_for_validation(systems, data_sizes, snrs, n_data, n_batches, observability,
                          path_results_in, fname_results_in, path_out, save_results=True):

    # data_size, sys_name, snr, iinit, obs, ieq, eqsym, ib, imodel = 'large', 'bacres', None, 0, 0, ['x'], 'x', 0, 0
    successful_models = []
    nonsuccessful_models = []

    for data_size in data_sizes:
        for sys_name in systems_collection:
            obss = get_observability_vars(observability, sys_name)
            eqsyms = systems[sys_name].state_vars if observability == "full" else "".join(systems[sys_name].state_vars)
            for snr in snrs:
                for iinit in range(n_data):
                    for obs in obss:    # if full observability, only one element (no looping)
                        for eqsym in eqsyms:  # if partial observability, only one element (no looping)
                            for ib in range(n_batches):
                                obs_txt = "".join(obs) if isinstance(obs, list) else obs
                                fname_results = fname_results_in.format(sys_name, data_size, snr, iinit, obs_txt, ib, eqsym)

                                try:
                                    size_in_bytes = os.path.getsize(f"{path_results_in}{sys_name}{os.sep}{fname_results}")
                                    if size_in_bytes == 0:
                                        print("File is empty, marked as unsuccessful.")
                                        nonsuccessful_models.append([exp_version, method, data_size, sys_name,
                                                                         obs_txt, snr, iinit, eqsym, ib])
                                    else:
                                        models = pg.ModelBox()
                                        models.load(f"{path_results_in}{sys_name}{os.sep}{fname_results}")
                                        print(f"Results for {fname_results} loaded.")

                                        for imodel in range(len(models)):
                                            duration = get_duration(models, imodel)
                                            expr = models[imodel].get_full_expr()
                                            successful_models.append([exp_version, method, data_size, sys_name,
                                                                      obs_txt, snr, iinit, eqsym, duration, expr[0]])

                                        del models

                                except:
                                    print(f"Results not loaded correctly: {fname_results}")
                                    nonsuccessful_models.append([exp_version, method, data_size, sys_name,
                                                                 obs_txt, snr, iinit, eqsym, ib])


    # convert to dataframe
    results_succ = pd.DataFrame(successful_models,
                           columns=["exp_version", "method", "data_size", "system", "obs", "snr", "iinit", "eq", "duration", "expr"])

    # convert to dataframe
    results_unsucc = pd.DataFrame(nonsuccessful_models,
                           columns=["exp_version", "method", "data_size", "system", "obs", "snr", "iinit", "eq", "ib"])

    # save all results as a dataframe
    if save_results:
        results_succ.to_csv(f"{path_out}validation_gathered_results_{method}_{exp_version}_{exp_type}.csv", sep='\t')
        results_unsucc.to_csv(f"{path_out}unsuccessfully_ran_models_{method}_{exp_version}_{exp_type}.csv", sep='\t')

    print("finished gathering results")
    return results

def get_observability_vars(observability, sys_name):
    if observability == "full":
        obss = [['x', 'y']] if sys_name != "lorenz" else [['x', 'y', 'z']]
    else:
        obss = [['x'], ['y'], ['x', 'y']] if sys_name != "lorenz" else [['x', 'y'], ['x', 'z'], ['y', 'z'], ['x', 'y', 'z']]
    return obss

def get_duration(models, imodel):
    if 'duration' in list(models[imodel].estimated.keys()):
        duration = models[imodel].estimated['duration']
    else:
        duration = np.nan
    return duration


##
if __name__ == "__main__":

    # experiment settings
    # root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    # root_dir = "."  # for HPC cluster (uncomment if running on HPC cluster)
    root_dir = "D:\\Experiments\\symreg_methods_comparison"
    sys.path.append(root_dir)

    method = "proged"
    exp_version = "e1"  # "e1" (constrained model search space) or "e2" (unconstrained model search space)
    observability = "full"  # "full" or "partial"
    simulation_type = "num" if observability == "full" else "sym"  # numerical derivation (num) or simulation by solving system of odes using initial values (sym)
    exp_type = f"sysident_{simulation_type}_{observability}"

    # data settings
    data_type = "train"
    data_sizes = ["small", "large"]
    snrs = [None, 30, 13]   # signal-to-noise ratio
    n_data = 4  # number of different data sets (have different initial values)
    n_batches = 100

    path_results_in = f"{root_dir}{os.sep}results{os.sep}{exp_type}{os.sep}{method}{os.sep}{exp_version}{os.sep}"
    fname_results_in = f"{method}_{exp_type}_{exp_version}_{{}}_train_{{}}_snr{{}}_init{{}}_obs{{}}_b{{}}_{{}}_fitted.pg"
    path_out = f"{root_dir}{os.sep}analysis{os.sep}{exp_type}{os.sep}"
    os.makedirs(path_out, exist_ok=True)
    start_time = pd.Timestamp.now()
    results = gather_for_validation(systems=systems_collection,
                                     data_sizes=data_sizes,
                                     snrs=snrs,
                                     n_data=n_data,
                                     n_batches=n_batches,
                                     observability=observability,
                                     path_results_in=path_results_in,
                                     fname_results_in=fname_results_in,
                                     path_out=path_out,
                                     save_results=True)




