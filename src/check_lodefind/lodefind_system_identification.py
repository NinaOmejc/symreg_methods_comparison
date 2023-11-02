
# !!! Important note !!!
# This code should be run from the root directory of the project L-ODEfind: https://github.com/agussomacal/L-ODEfind
# Please move this script to ./L-ODEfind-master/. Alternatively, add the L-ODEfind-master folder to your PYTHONPATH and
# adjust the import statements accordingly.
# !!! Important note !!!

import os
import numpy as np
import pandas as pd
import time
from typing import List
from src.lib.skodefind import SkODEFind

np.random.seed(1)


def fit_and_save_coeffs(model: str, targets: List[int], maxpolys: List[int], obs_vars: List[str],
                        testsize=50, settings={}):

    os.makedirs(settings['path_results_out'], exist_ok=True)

    # target, maxpoly = targets[0], maxpolys[0]
    for target in targets:
        maxd = target - 1
        for maxpoly in maxpolys:
            data_df = pd.read_csv(f"{settings['path_data']}{os.sep}{settings['fname_data']}", index_col=0)[obs_vars]
            data_train = data_df.iloc[:-testsize, :]

            t0 = time.time()

            # configure fitting
            odefind = SkODEFind(
                target_derivative_order=target,
                max_derivative_order=maxd,
                max_polynomial_order=maxpoly,
                rational=False,
                with_mean=True, with_std=True, alphas=150, max_iter=10000, cv=20,
                use_lasso=True,
            )

            # main function
            odefind.fit(data_train)

            # save duration
            fname_out = f"{settings['method']}_{settings['exp_type']}_{settings['exp_version']}_{model}_" \
                        f"{settings['data_type']}_{settings['data_size']}_snr{snr}_" \
                        f"init{iinit}_obs{''.join(obs_vars)}_{{}}.csv"

            duration_pd = pd.DataFrame([time.time() - t0], columns=["duration"])
            duration_pd.to_csv(f"{path_out}{os.sep}{fname_out.format('duration')}", index=False)

            coeffs = odefind.coefs_.transpose()
            coeffs.to_csv(f"{path_out}{os.sep}{fname_out.format('coeffs')}")

method = "lodefind"
exp_type = "sysident_partial"
exp_version = "e1"  # only constrained search space
data_sizes = ["small", "large"]
observability = ["partial", "full"]
snrs = [None, 30, 13]
n_data = 4
data_type = "train"
sys_name = 'vdp'
observed_vars = [['x'], ['y'], ['x', 'y']]

path_out = f"D:\\Experiments\\symreg_methods_comparison\\results\\{exp_type}\\{method}\\{exp_version}\\{sys_name}"  # adjust accordingly
path_data = f"D:\\Experiments\\symreg_methods_comparison\\data\\{data_type}"

for data_size in data_sizes:
    for snr in snrs:
        for iinit in range(n_data):
            for obs in observed_vars:

                obs_text = ''.join(obs)
                data_length = 2000 if data_size == "large" else 100
                test_size = 200 if data_size == "large" else 20
                time_end = 20 if data_size == "large" else 10
                time_step = 0.01 if data_size == "large" else 0.1
                fname_data = f"data_{sys_name}_len{time_end}_rate{str(time_step).replace('.', '')}_snr{snr}_init{iinit}.csv"

                settings = {
                    'method': method,
                    'exp_type': exp_type,
                    'exp_version': exp_version,
                    'data_size': data_size,
                    'data_length': data_length,
                    'data_type': data_type,
                    'snr': snr,
                    'iinit': iinit,
                    'path_data': f"{path_data}\\{data_size}_for_lodefind_gpom\\{sys_name}",
                    'fname_data': fname_data,
                    'path_results_out': path_out
                }

                targets = [1] if obs == ['x', 'y'] else [2]   # the order of the derivative I'm searching
                maxpolys = [3]  # the maximum degree of the polynomials

                fit_and_save_coeffs(sys_name, targets=targets, maxpolys=maxpolys, obs_vars=obs,
                                    testsize=test_size, settings=settings)
