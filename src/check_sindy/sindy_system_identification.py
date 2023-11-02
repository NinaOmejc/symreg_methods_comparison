import os
import numpy as np
import pandas as pd
import time
import itertools
import pysindy as ps
from src.utils.systems_collection import systems_collection
# from src.check_sindy.re_auto3 import x_for_dM


def run_sindy(systems, path_data, fname_data, observability="full", snrs=[None, 30, 13],
              iinit=0, regulation_params=None, path_out=".\\check_sindy\\results\\", use_default_library=False,
              print_equations=False):

    """
    Run SINDy on the data generated by the function generate_data().

    Arguments:
        - systems            (list)                    list of <System> objects (see file system.py)
        - path_data          (string)                  path to the folder where the data is saved
        - fname_data         (string)                  name of the csv file with the data
        - observability      (string)                  either 'full' or 'partial' (default = 'full')
        - snrs               (list)                    list of desired Signal-to-Noise Ratios (in dB). If None (default),
                                                       no noise will be added, otherwise float or int values should be provided.
        - iinit             (int)                      the index of data to be loaded
        - regulation_params  (list of tuples)          list of tuples (thr, nu), where thr is the threshold for the
                                                         regularization and nu is the regularization parameter
        - path_out           (string)                  path to the folder where the results should be saved
        - use_default_library (bool)                   if True, the default library of functions is used for SINDy
                                                         (see pysindy documentation for details)

    Returns:
        Nothing. Results are saved in the folder specified by path_out.

    Example:
        Example run is shown at the end of the script.

    """

    # system_name, snr, iinit, thr, nu = 'barmag', None, 0, regulation_params[0][0], regulation_params[0][1]
    for system_name in systems:
        results_list_all = []
        for snr in snrs:

            # Get true noisy data
            data_true_fname = f"{path_data}{system_name}{os.sep}{fname_data}".format(system_name, snr, iinit)
            data_true = pd.read_csv(data_true_fname)
            data = np.array(data_true[['t'] + systems[system_name].state_vars])

            # Get true noisy data derivatives
            try:
                data_der = np.array(data_true[['d' + systems[system_name].state_vars[i] for i in
                                                range(len(systems[system_name].state_vars))]])
            except:
                raise ValueError("No derivatives in the data. Please, generate data with derivatives. Derivatives"
                                 "should be named as d<state_variable_name> in the csv file.")

            t = data[:, 0]
            X = data[:, 1:]

            for thr, nu in regulation_params:
                print(f"{method} | {system_name} | snr: {snr} | init: {iinit} | thr: {thr} | nu: {nu}")

                # -------------------------------------------------------------------------------
                # ------------------------ START SINDy -----------------------------------------
                # -------------------------------------------------------------------------------

                try:
                    startTime = time.time()
                    model = sindy_fit(system_name, X, t, x_dot=data_der, threshold=thr,
                                      nu=nu, use_default_library=use_default_library)
                    duration = time.time() - startTime



                    # speed up the process - stop if the expression is just 0
                    if ("['0.000'," in str([model.equations()])) | (", '0.000']" in str([model.equations()])):
                        break

                    expression = str([model.equations()])
                    # print equations if requested
                    if print_equations:
                        print(expression)

                    ires = [method, exp_version, data_type, data_size, system_name, observability, snr,
                            iinit, thr, nu, duration, expression]
                    results_list_all.append(ires)
                except Exception as e:
                    print(f"Error in model.fit. Continue. An error that occurred: {e}")
                    duration = np.nan
                    ires = [method, exp_version, data_type, data_size, system_name, observability, snr,
                            iinit, thr, nu, duration, np.nan]
                    results_list_all.append(ires)
                    continue


            # list -> dataframe, all results
            os.makedirs(path_out, exist_ok=True)
            results_all = pd.DataFrame(results_list_all, columns=["method", "exp_version", "data_type", "data_size",
                                      "system", "obs_type", "snr", "iinit", "thr", "nu", "duration", "expression"])
            results_all.to_csv(f"{path_out}{method}_{exp_type}_{exp_version}_{data_type}_{data_size}_"
                               f"_snr{snr}_init{iinit}_obs{observability}_fitted.csv", sep='\t', index=False)


def sindy_fit(system_name, X, t, x_dot=[], threshold=0.1, nu=1, use_default_library=False, error_with_log_occured=False):

    # library
    if use_default_library:
        lib = create_library_default(error_with_log_occured=error_with_log_occured)
    else:
        lib = get_library_functions(system_name)

    # variable names (system state variables)
    var_names = ["x", "y"] if system_name != 'lorenz' else ["x", "y", "z"]

    # optimizer settings
    opt = ps.SR3(
        threshold=threshold,
        nu=nu,
        thresholder="l1",
        max_iter=100000,
        normalize_columns=True,
        tol=1e-5)

    # run check_sindy
    model = ps.SINDy(feature_names=var_names,
                     optimizer=opt,
                     feature_library=lib)
    try:
        model.fit(X, t=t)
    except Exception as e:
        # if e includes "log", try to fit again with the default library but with the error_with_log_occured=True
        if use_default_library == True and "Input contains NaN, infinity or a value too large" in str(e):
            if not error_with_log_occured:
                print("Error with log occured. Try to fit again with the default library.")
                return sindy_fit(system_name, X, t, x_dot, threshold, nu, use_default_library=True, error_with_log_occured=True)
            else:
                raise ValueError("Error with log occured again. Skip this model fitting.")

    return model


def create_library_default(error_with_log_occured):
    lib_poly = ps.PolynomialLibrary(degree=3, include_interaction=True, include_bias=True)
    lib_trig = ps.FourierLibrary()

    if not error_with_log_occured:
        custom_functs = [lambda x, y: x / y,
                         lambda x, y: y / x,
                         lambda x: np.tan(x),
                         lambda x: 1 / np.tan(x),
                         lambda x: np.exp(x),
                         lambda x: np.log(x)]
        custom_functs_names = [lambda x, y: x + '/' + y,
                               lambda x, y: y + '/' + x,
                               lambda x: 'tan(' + x + ')',
                               lambda x: '1/tan(' + x + ')',
                               lambda x: 'exp(' + x + ')',
                               lambda x: 'log(' + x + ')']

    else:
        print("Error with log occured. Remove log from the library.")
        # remove log from the library
        custom_functs = [lambda x, y: x / y,
                         lambda x, y: y / x,
                         lambda x: np.tan(x),
                         lambda x: 1 / np.tan(x),
                         lambda x: np.exp(x)]

        custom_functs_names = [lambda x, y: x + '/' + y,
                               lambda x, y: y + '/' + x,
                               lambda x: 'tan(' + x + ')',
                               lambda x: '1/tan(' + x + ')',
                               lambda x: 'exp(' + x + ')']

    lib_custom = ps.CustomLibrary(library_functions=custom_functs,
                                  function_names=custom_functs_names,
                                  include_bias=True)

    lib = ps.GeneralizedLibrary([lib_poly, lib_trig, lib_custom])
    return lib




def get_library_functions(sys_name):

    if sys_name in ['lv', 'myvdp', 'vdp', 'stl', 'lorenz', 'predprey', 'bacres']:
        library_functions = [lambda x: x,
                             lambda x: x ** 2,
                             lambda x, y: x * y,
                             lambda x: x ** 3,
                             lambda x, y: x ** 2 * y,
                             lambda x, y: y ** 2 * x]

        library_function_names = [lambda x: x,
                                  lambda x: x + '^2',
                                  lambda x, y: x + '*' + y,
                                  lambda x: x + '^3',
                                  lambda x, y: x + '^2' + '*' + y,
                                  lambda x, y: y + '^2' + '*' + x]

    elif sys_name in ['glider', 'shearflow']:

        library_functions = [lambda x: x,

                             lambda x: x ** 2,
                             lambda x: x ** 3,
                             lambda x, y: x * y,
                             lambda x, y: x / y,
                             lambda x, y: y / x,
                             lambda x, y: x ** 2 * y,
                             lambda x, y: y ** 2 * x,

                             lambda x: np.sin(x),
                             lambda x: np.cos(x),
                             lambda x: np.tan(x),
                             lambda x: 1 / np.tan(x),

                             lambda x: np.cos(x) * np.cos(x),
                             lambda x: np.sin(x) * np.sin(x),

                             lambda x, y: np.cos(y) / x,
                             lambda x, y: np.sin(y) / x,
                             lambda x, y: np.cos(x) / y,
                             lambda x, y: np.sin(x) / y,

                             lambda x, y: (np.sin(x) ** 2) * np.sin(y),
                             lambda x, y: (np.sin(y) ** 2) * np.sin(x),
                             lambda x, y: (np.cos(x) ** 2) * np.sin(y),
                             lambda x, y: (np.cos(y) ** 2) * np.sin(x),
                             lambda x, y: np.sin(x) * (1 / np.tan(y)),
                             lambda x, y: np.sin(y) * (1 / np.tan(x)),
                             lambda x, y: np.cos(x) * (1 / np.tan(y)),
                             lambda x, y: np.cos(y) * (1 / np.tan(x))]

        library_function_names = [lambda x: x,
                                  lambda x: x + '^2',
                                  lambda x: x + '^3',
                                  lambda x, y: x + '*' + y,
                                  lambda x, y: x + '/' + y,
                                  lambda x, y: y + '/' + x,
                                  lambda x, y: x + '^2' + '*' + y,
                                  lambda x, y: y + '^2' + '*' + x,

                                  lambda x: 'sin(' + x + ')',
                                  lambda x: 'cos(' + x + ')',
                                  lambda x: 'tan(' + x + ')',
                                  lambda x: 'cot(' + x + ')',

                                  lambda x: 'sin^2(' + x + ')',
                                  lambda x: 'cos^2(' + x + ')',

                                  lambda x, y: 'cos(' + y + ')/(' + x + ')',
                                  lambda x, y: 'sin(' + y + ')/(' + x + ')',
                                  lambda x, y: 'cos(' + x + ')/(' + y + ')',
                                  lambda x, y: 'sin(' + x + ')/(' + y + ')',

                                  lambda x, y: 'sin^2(' + y + ') * sin(' + x + ')',
                                  lambda x, y: 'sin^2(' + x + ') * sin(' + y + ')',
                                  lambda x, y: 'cos^2(' + x + ') * sin(' + y + ')',
                                  lambda x, y: 'cos^2(' + y + ') * sin(' + x + ')',
                                  lambda x, y: 'sin(' + x + ') * cot(' + y + ')',
                                  lambda x, y: 'sin(' + y + ') * cot(' + x + ')',
                                  lambda x, y: 'cos(' + x + ') * cot(' + y + ')',
                                  lambda x, y: 'cos(' + y + ') * cot(' + x + ')']

        # feature names: ['1', 'x', 'y', 'x^2', 'y^2', 'sin(x)', 'sin(y)', 'cos(x)', 'cos(y)', 'xy', 'x/y', 'y/x', 'cos(y)/(x)', 'x^3', 'y^3', 'x^2y', 'y^2x']

    elif sys_name in ['barmag', 'cphase']:

        library_functions = [lambda x: np.sin(x),
                             lambda x: np.cos(x),
                             lambda x, y: np.sin(x + y),
                             lambda x, y: np.cos(x + y),
                             lambda x, y: np.sin(x - y),
                             lambda x, y: np.cos(x - y),
                             lambda x, y: np.sin(y - x),
                             lambda x, y: np.cos(y - x)]

        library_function_names = [lambda x: 'sin(' + x + ')',
                                  lambda x: 'cos(' + x + ')',
                                  lambda x, y: 'sin(' + x + '+' + y + ')',
                                  lambda x, y: 'cos(' + x + '+' + y + ')',
                                  lambda x, y: 'sin(' + x + '-' + y + ')',
                                  lambda x, y: 'cos(' + x + '-' + y + ')',
                                  lambda x, y: 'sin(' + y + '-' + x + ')',
                                  lambda x, y: 'cos(' + y + '-' + x + ')']

    else:
        raise ValueError("Error. No library could be chosen based on the system name (sys_name). Recheck inputs.")

    lib = ps.CustomLibrary(
        library_functions=library_functions,
        function_names=library_function_names,
        include_bias=True)

    return lib

def get_configuration(iinput, systems_collection, data_sizes, snrs, n_data):
    combinations = list(itertools.product(systems_collection, data_sizes, snrs, np.arange(n_data)))
    return combinations[iinput]


##

if __name__ == "__main__":

    # modifyable settings
    iinput = 210  # int(sys.argv[1])
    data_sizes = ["small", "large"]
    snrs = [None, 30, 13]   # [None, 30, 13]
    n_data = 4

    system_name, data_size, snr, iinit = get_configuration(iinput, systems_collection, data_sizes, snrs, n_data)

    # fixed settings
    # get a system dict that has only one item (key, value pair) of the system name from systems_collection
    systems = {system_name: systems_collection[system_name]}
    method = "sindy"
    exp_version = "e2"
    observability = "full"
    exp_type = f"sysident_num_{observability}"
    data_type = "train"
    time_end = 20 if data_size == "large" else 10
    time_step = 0.01 if data_size == "large" else 0.1

    thresholds = np.arange(0.1, 10, 0.05)  ## np.arange(0.1, 10, 0.05)
    nus = [0.5, 1]
    regulation_params = list(itertools.product(thresholds, nus))

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    root_dir = "D:\\Experiments\\symreg_methods_comparison\\"
    path_data = f"{root_dir}{os.sep}data{os.sep}{data_type}{os.sep}{data_size}{os.sep}"
    fname_data = f"data_{{}}_len{time_end}_rate{str(time_step).replace('.', '')}_snr{{}}_init{{}}.csv"
    path_out = f"{root_dir}{os.sep}results{os.sep}{exp_type}{os.sep}{method}{os.sep}{exp_version}{os.sep}"

    run_sindy(systems=systems,
              snrs=[snr],
              iinit=iinit,
              path_data=path_data,
              fname_data=fname_data,
              observability=observability,
              regulation_params=regulation_params,
              use_default_library=True,
              path_out=path_out,
              print_equations=True)

