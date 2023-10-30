
import os
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tabulate import tabulate
from scipy.integrate import odeint
from src.generate_data.systems_collection import strogatz, mysystems
from plotting.parestim_sim_plots import plot_trajectories
import sympy as sp
from scipy.integrate import odeint
from scipy.interpolate import interp1d

def calculate_TE(traj_true, traj_model):  # trajectory error
    TEx = np.sqrt((np.mean((traj_model[:, 0] - traj_true[:, 0]) ** 2))) / np.std(traj_true[:, 0])
    TEy = np.sqrt((np.mean((traj_model[:, 1] - traj_true[:, 1]) ** 2))) / np.std(traj_true[:, 1])
    TE = TEx + TEy
    if sys_name == 'lorenz':
        TEz = np.sqrt((np.mean((traj_model[:, 2] - traj_true[:, 2]) ** 2))) / np.std(
            traj_true[:, 3])
        TE = TE + TEz
    return TE

def get_errors_per_expr(data_sets, data_inits, expr):
    val_errors_per_expr = []
    expr_func = [sp.lambdify(['t'] + systems[sys_name].lhs_vars, iexpr) for iexpr in expr]
    time = np.arange(0, sim_time, sim_step)

    for ival in list(data_inits.keys()):

        # X_extras = interp1d(estimator.data['t'], estimator.data[model.extra_vars], axis=0, kind='cubic',
        #                     fill_value="extrapolate") if model.extra_vars != [] else ( lambda t: np.array([]))
        # X_extras = interp1d(time, np.array(data_sets[ival].iloc[:, 1:]), axis=0, kind='cubic', fill_value="extrapolate")
        def rhs(t, x):
            #b = np.concatenate((x, X_extras(t).reshape(1,)))
            # b = X_extras(t)
            # b[ieq] = x
            return [iexpr_func(t, *x) for iexpr_func in expr_func]

        simulation, odeint_output = odeint(rhs, data_inits[ival], time, rtol=1e-12, atol=1e-12, tfirst=True, full_output=True)

        if 'successful' not in odeint_output['message']:
            val_errors_per_expr.append(np.nan)
        else:
            val_errors_per_expr.append(calculate_TE(np.array(data_sets[ival])[:, 1:], simulation))
    return val_errors_per_expr

##
method = "lodefind"
exp_type = 'parobs'

systems = {**strogatz, **mysystems}
data_versions = ['all', 'allonger']
path_main = f"D:{os.sep}Experiments{os.sep}MLJ23"
path_base_out = f"{path_main}{os.sep}results{os.sep}validation{os.sep}validation_results{os.sep}parobs_sim{os.sep}{method}{os.sep}"
os.makedirs(path_base_out, exist_ok=True)
structure_version = 's0'
sys_names = ['myvdp']
snrs = ['inf', 30, 13]
set_obs = "full"
n_inits = 4

plot_trajectories_bool = False

# data_version, sys_name, snr, iinit, iobs_name = 'allonger', 'myvdp', 'inf', 3, 'xy'
test_results = []

for data_version in data_versions:

    exp_version = 'e2'
    path_base_in = f"{path_main}{os.sep}results{os.sep}{method}{os.sep}sysident_num{os.sep}{exp_version}{os.sep}"
    path_test_data = f"{path_main}{os.sep}data{os.sep}test{os.sep}{data_version}{os.sep}"
    validation_set_path = f"{path_main}{os.sep}data{os.sep}validation{os.sep}{data_version}{os.sep}"
    sim_step = 0.01 if data_version == 'allonger' else 0.1
    sim_time = 20 if data_version == 'allonger' else 10
    data_length = int(sim_time / sim_step)

    for sys_name in sys_names:
        obss = systems[sys_name].get_obs("all")
        path_in = f"{path_base_in}{sys_name}{os.sep}"
        for snr in snrs:
            for iobs in obss:
                iobs_name = ''.join(iobs)
                dmax = 1 if iobs == 'xy' else 2

                # load validation data
                validation_sets = {'0': pd.read_csv(f"{validation_set_path}{sys_name}{os.sep}data_{sys_name}_{data_version}_len{data_length}_snr{snr}_init0.csv", sep=','),
                                   '1': pd.read_csv(f"{validation_set_path}{sys_name}{os.sep}data_{sys_name}_{data_version}_len{data_length}_snr{snr}_init1.csv", sep=','),
                                   '2': pd.read_csv(f"{validation_set_path}{sys_name}{os.sep}data_{sys_name}_{data_version}_len{data_length}_snr{snr}_init2.csv", sep=','),
                                   '3': pd.read_csv(f"{validation_set_path}{sys_name}{os.sep}data_{sys_name}_{data_version}_len{data_length}_snr{snr}_init3.csv", sep=','),
                                   }

                validation_inits = {'0': np.array(validation_sets['0'].iloc[0, 1:]),
                                    '1': np.array(validation_sets['1'].iloc[0, 1:]),
                                    '2': np.array(validation_sets['2'].iloc[0, 1:]),
                                    '3': np.array(validation_sets['3'].iloc[0, 1:]),
                                    }

                validation_errors = {}

                for iinit in range(n_inits):
                    filename_base = f"{sys_name}_{data_version}_{structure_version}_{exp_version}_len{data_length}_snr{snr}_init{iinit}_obs{iobs_name}"
                    coeffs = pd.read_csv(f"{path_base_in}{os.sep}{sys_name}{os.sep}{filename_base}_coeffs.csv")
                    print("Doing: " + filename_base)

                    if iobs_name == 'xy':
                        model_params = np.hstack((np.array(coeffs.iloc[:, 1]), np.array(coeffs.iloc[:, 2])))
                        terms = ['ct', 'X', 'Y', 'X^2', 'XY', 'Y^2', 'X^3', 'X^2 Y', 'X Y^2', 'Y^3']
                        def dxdt(t, x, p):
                            return [
                                p[0] * 1 + p[1] * x[0] + p[2] * x[1] + p[3] * x[0] ** 2 + p[4] * x[0] * x[1] + p[5] * x[1] ** 2 +
                                p[6] * x[0] ** 3 + p[7] * x[0] ** 2 * x[1] + p[8] * x[0] * x[1] ** 2 + p[9] * x[1] ** 3,
                                p[10] * 1 + p[11] * x[0] + p[12] * x[1] + p[13] * x[0] ** 2 + p[14] * x[0] * x[1] + p[15] * x[1] ** 2 +
                                p[16] * x[0] ** 3 + p[17] * x[0] ** 2 * x[1] + p[18] * x[0] * x[1] ** 2 + p[19] * x[1] ** 3]

                    else:
                        model_params = np.array(coeffs.iloc[:, 1])
                        terms = ['ct', 'X', 'Xdot', 'X^2', 'X Xdot', 'Xdot^2', 'X^3', 'X^2 Xdot', 'X Xdot^2', 'Xdot^3']
                        def dxdt(t, x, p):
                            return [x[1],
                                    p[0] * 1 + p[1] * x[0] + p[2] * x[1] + p[3] * x[0] ** 2 + p[4] * x[0] * x[1] + p[
                                        5] * x[1] ** 2 + p[6] * x[0] ** 3 +
                                    p[7] * x[0] ** 2 * x[1] + p[8] * x[0] * x[1] ** 2 + p[9] * x[1] ** 3]

                    # Define symbolic variables
                    t, x, y, x0, x1 = sp.symbols('t x y x0 x1')
                    p = sp.symbols('p0:20') if iobs_name == 'xy' else sp.symbols('p0:10') # p0, p1, ..., p19

                    # Convert the function to a symbolic expression
                    dxdt_expr = sp.sympify(sp.Matrix(dxdt(t, [x0, x1], p)))
                    dxdt_expr_subs = dxdt_expr.subs([(x0, x), (x1, y)])
                    model_params_dict = {p[i]: value for i, value in enumerate(model_params.flatten(order='F'))}
                    dxdt_specific = dxdt_expr_subs.subs(model_params_dict)

                    val_errors = get_errors_per_expr(validation_sets, validation_inits, dxdt_specific)
                    val_errors_mean = np.nanmean(val_errors)
                    if np.isnan(val_errors_mean):
                        val_errors_mean = 10 ** 10
                    validation_errors[dxdt_specific] = [val_errors_mean] + val_errors

                validation_errors_sorted = {k: v for k, v in sorted(validation_errors.items(), key=lambda item: item[1][0])}
                results_partial_fname = f"{path_main}{os.sep}results{os.sep}{method}{os.sep}sysident_num{os.sep}{exp_version}{os.sep}{sys_name}{os.sep}validation{os.sep}" \
                                        f"errors_sorted_{data_version}_{exp_version}_{sys_name}_snr{snr}_obs{iobs_name}.txt"

                with open(results_partial_fname, 'w') as file:
                    count = 0  # Counter for tracking the number of pairs written
                    for key, value in validation_errors_sorted.items():
                        file.write(f"{['{:.4f}'.format(round(v, 4)) for v in value]}: {key}\n")
                        count += 1
                        if count == 100:
                            break

                best_expr, best_validation_errors = next(iter(validation_errors_sorted.items()))

                # load test data
                test_sets = {'0': pd.read_csv( f"{path_test_data}{sys_name}{os.sep}data_{sys_name}_{data_version}_len{data_length}_snrinf_init0.csv", sep=','),
                             '1': pd.read_csv( f"{path_test_data}{sys_name}{os.sep}data_{sys_name}_{data_version}_len{data_length}_snrinf_init1.csv", sep=','),
                             '2': pd.read_csv( f"{path_test_data}{sys_name}{os.sep}data_{sys_name}_{data_version}_len{data_length}_snrinf_init2.csv", sep=','),
                             '3': pd.read_csv( f"{path_test_data}{sys_name}{os.sep}data_{sys_name}_{data_version}_len{data_length}_snrinf_init3.csv", sep=',')}

                test_inits = {'0': np.array(test_sets['0'].iloc[0, 1:]),
                              '1': np.array(test_sets['1'].iloc[0, 1:]),
                              '2': np.array(test_sets['2'].iloc[0, 1:]),
                              '3': np.array(test_sets['3'].iloc[0, 1:])}

                test_errors = get_errors_per_expr(test_sets, test_inits, best_expr)

                ## save results
                test_results.append([method, data_version, exp_version, sys_name, iobs_name, snr, best_expr,
                     np.nanmean(best_validation_errors), best_validation_errors, np.nanmean(test_errors), test_errors])

                del validation_errors, validation_errors_sorted


results = pd.DataFrame(test_results,
                       columns=["method", "data_version", "exp_version", "system", "obs_type", "snr", "expr",
                                "val_error", "val_errors", "test_error", "test_errors"])

results.to_csv(
    f"{path_base_out}validation_test_bestof_{method}.csv")
