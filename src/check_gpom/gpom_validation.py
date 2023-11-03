
import os
import pandas as pd
import numpy as np
import sympy as sp
from scipy.integrate import odeint

def get_expr_from_coeffs(results):

    iparams = np.hstack((np.array(results.iloc[:, 1]), np.array(results.iloc[:, 2])))

    def rhs(t, x, p):
        return [
            p[0] * 1 + p[1] * x[1] + p[2] * x[1] ** 2 + p[3] * x[1] ** 3 + p[4] * x[0] + p[5] * x[0] *
            x[1] + p[6] * x[0] * x[1] ** 2 + p[7] * x[0] ** 2 + p[8] * x[0] ** 2 * x[1] + p[9] * x[0] ** 3,
            p[10] * 1 + p[11] * x[1] + p[12] * x[1] ** 2 + p[13] * x[1] ** 3 + p[14] * x[0] + p[15] * x[0] * x[1] +
            p[16] * x[0] * x[1] ** 2 + p[17] * x[0] ** 2 + p[18] * x[0] ** 2 * x[1] + p[19] * x[0] ** 3]

    # Define symbolic variables
    p = sp.symbols('p0:20')
    t, x, y, x0, x1 = sp.symbols('t x y x0 x1')

    # Convert the function to a symbolic expression
    rhs_expr = sp.sympify(sp.Matrix(rhs(t, [x0, x1], p)))
    rhs_expr_subs = rhs_expr.subs([(x0, x), (x1, y)])
    model_params_dict = {p[i]: value for i, value in enumerate(iparams.flatten(order='F'))}
    rhs_expr_specific = rhs_expr_subs.subs(model_params_dict)

    return rhs_expr_specific

def get_errors(n_data, path_data, sys_name, time_end, time_step, snr, expr):

    errors = []
    for iinit in range(n_data):

        # load test data
        data_set = pd.read_csv(f"{path_data}{os.sep}data_{sys_name}_len{time_end}_rate{str(time_step).replace('.', '')}_snr{snr}_init{iinit}.csv", sep=',')
        data_set = data_set[['t', 'x', 'y']]
        data_inits = np.array(data_set.iloc[0, 1:])
        errors.append(get_error_per_expr(data_set, data_inits, expr))

    # replace nans with inf
    errors = np.array(errors)
    errors[np.isnan(errors)] = np.inf
    return errors


def get_error_per_expr(data_set, data_inits, expr):
    expr_func = [sp.lambdify(['t', 'x', 'y'], iexpr) for iexpr in expr]
    time = np.arange(0, time_end, time_step)

    def rhs(t, x):
        return [iexpr_func(t, *x) for iexpr_func in expr_func]

    simulation, odeint_output = odeint(rhs, data_inits, time, rtol=1e-12, atol=1e-12, tfirst=True, full_output=True)

    # calculate trajectory error if simulation was successful
    if 'successful' not in odeint_output['message']:
        return np.nan
    else:
        return calculate_TE(np.array(data_set)[:, 1:], simulation)


def calculate_TE(traj_true, traj_model):  # trajectory error
    TEx = np.sqrt((np.mean((traj_model[:, 0] - traj_true[:, 0]) ** 2))) / np.std(traj_true[:, 0])
    TEy = np.sqrt((np.mean((traj_model[:, 1] - traj_true[:, 1]) ** 2))) / np.std(traj_true[:, 1])
    TE = TEx + TEy
    if sys_name == 'lorenz':
        TEz = np.sqrt((np.mean((traj_model[:, 2] - traj_true[:, 2]) ** 2))) / np.std(
            traj_true[:, 3])
        TE = TE + TEz
    return TE


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


##
method = "gpom"
exp_version = 'e1'
observability = 'partial'
exp_type = f"sysident_{observability}"
data_sizes = ["small", "large"]
sys_name = 'vdp'
snrs = ['None', '30', '13']
n_train_data = 4
n_val_data = 4
n_test_data = 4
obss = [['x'], ['y'], ['x', 'y']]

path_main = f"D:{os.sep}Experiments{os.sep}symreg_methods_comparison"
path_base_in = f"{path_main}{os.sep}results{os.sep}{exp_type}{os.sep}{method}{os.sep}{exp_version}{os.sep}{sys_name}"
path_results_out = f"{path_main}{os.sep}analysis{os.sep}{exp_type}{os.sep}{method}"
os.makedirs(path_results_out, exist_ok=True)

##

# sys_name, snr, iinit_train, iobs, data_size, imodel = 'vdp', 'None', 0, obss[2], 'small', 1
validation_results = []

for data_size in data_sizes:

    time_end = 20 if data_size == "large" else 10
    time_step = 0.01 if data_size == "large" else 0.1
    path_validation_data = f"{path_main}{os.sep}data{os.sep}validation{os.sep}{data_size}{os.sep}{sys_name}"

    for snr in snrs:
        for iinit_train in range(n_train_data):
            for iobs in obss:

                iobs_name = "".join(iobs)
                dmax = 1 if len(iobs_name)==2 else 2

                filename_base = f"{method}_{sys_name}_{data_size}_dmax{dmax}_poly3_steps5120_obs{iobs_name}_snr{snr}_init{iinit_train}"
                duration_df = pd.read_csv(f"{path_base_in}{os.sep}{filename_base}{os.sep}duration_{filename_base}.csv")

                # find number of models in the folder path
                results_path = f"{path_base_in}{os.sep}{filename_base}"
                n_models = len([name for name in os.listdir(results_path) if os.path.isfile(os.path.join(results_path, name)) and 'model' in name])

                # load data
                for imodel in range(1, n_models):

                    # load results
                    try:
                        results = pd.read_csv(f"{results_path}{os.sep}params_{filename_base}_model{imodel}.csv")
                        print(f"data_size: {data_size} | snr: {snr} | iinit: {iinit_train} | obs: {iobs} | Model # {imodel} successfully loaded.")
                    except:
                        print(f"data_size: {data_size} | snr: {snr} | iinit: {iinit_train} | obs: {iobs} | Model # {imodel} could not be loaded.")
                        validation_results.append( [exp_version, method, data_size, sys_name, iobs_name, str(snr), iinit_train, 'xy', np.nan, np.inf, ""])
                        continue

                    # check if the first entry contains "Error"
                    if 'Error' in str(results.iloc[0].values):
                        validation_results.append( [exp_version, method, data_size, sys_name, iobs_name, str(snr), iinit_train, 'xy', np.nan, np.inf, ""])
                        continue

                    # get expression from coefficients
                    expr = round_constants(get_expr_from_coeffs(results), n=3)

                    # get validation error on 4 validation data sets
                    validation_errors = get_errors(n_val_data, path_validation_data, sys_name, time_end, time_step, snr, expr)
                    validation_errors_mean = np.mean(validation_errors)
                    validation_results.append([exp_version, method, data_size, sys_name, iobs_name, str(snr), iinit_train, 'xy', np.nan, validation_errors_mean, str(expr)])


# save validation results as dataframe
validation_results = pd.DataFrame(validation_results, columns=['exp_version', 'method', 'data_size', 'system', 'obs', 'snr', 'iinit', 'eq', 'duration', 'val_TE_mean', 'expr'])
validation_results.to_csv(f"{path_results_out}{os.sep}validation_gathered_results_{method}_{exp_version}_{exp_type}.csv", sep=',', index=False)

# get the models with the lowest validation error, grouped by data_size, sys_name, snr, obs_name
results_best = validation_results.groupby(['system', 'data_size', 'snr', 'obs']).agg({'val_TE_mean': 'min', 'expr': 'first'}).reset_index()



###################################################################################################
## START TEST
###################################################################################################
print("Doing testing...")

# load results_best
# validation_results = pd.read_csv(f"{path_results_out}{os.sep}validation_gathered_results_{method}_{exp_version}_{exp_type}.csv", sep=',')
# results_best = validation_results.groupby(['system', 'data_size', 'snr', 'obs']).agg({'val_TE_mean': 'min', 'expr': 'first'}).reset_index()

for ibest in range(len(results_best)):
    data_size = results_best.iloc[ibest]['data_size']
    sys_name = results_best.iloc[ibest]['system']
    snr = results_best.iloc[ibest]['snr']
    path_test_data = f"{path_main}{os.sep}data{os.sep}test{os.sep}{data_size}{os.sep}{sys_name}"
    expr = sp.sympify(results_best.iloc[ibest]['expr'])
    time_end = 20 if data_size == "large" else 10
    time_step = 0.01 if data_size == "large" else 0.1

    test_errors = get_errors(n_test_data, path_test_data, sys_name, time_end, time_step, snr, expr)

    results_best.loc[ibest, 'test_TE_mean'] = np.mean(test_errors)


## save results
results_best.to_csv(f"{path_results_out}{os.sep}best_results_{exp_version}_{method}_withTestTE.csv")
print(f"Finished. Best results saved into {path_results_out}")
