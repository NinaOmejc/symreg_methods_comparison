import os
import sys
import numpy as np
import pandas as pd
import time
import itertools
from src.utils.systems_collection import systems_collection
from dso import DeepSymbolicOptimizer


def run_dso(systems, path_data, fname_data, iinit, eq_sym, snrs=[None, 30, 13], observability="full",
            use_default_library=False, path_out=".\\results\\check_dso\\", path_config=".\\src\\config.json"):

    """
    Run DSO on the data generated by the function generate_data().

    Arguments:
        - systems             (list)                    list of <System> objects (see file system.py)
        - path_data           (string)                  path to the folder where the data is saved
        - fname_data          (string)                  name of the csv file with the data
        - iinit               (int)                     index of the initial condition to be used
        - eq_sym              (string)                  symbol of the equation to be identified
        - snrs                (list)                    list of desired Signal-to-Noise Ratios (in dB). If None (default),
                                                        no noise will be added, otherwise float or int values should be provided.
        - observability       (string)                  either 'full' or 'partial' (default = 'full')
        - path_out            (string)                  path to the folder where the results should be saved
        - use_default_library (bool)                    if True, the default library of functions is used for SINDy
                                                         (see pysindy documentation for details)

    Returns:
        Nothing. Results are saved in the folder specified by path_out.

    Example:
        Example run is shown at the end of the script.

    """

    for system_name in systems:
        for snr in snrs:

            print(f"{method} | {system_name} | snr: {snr} | init: {iinit} | eq: {eq_sym}")

            # Configure the model
            model = DeepSymbolicOptimizer(path_config)
            model.config_experiment['logdir'] = f"{path_out}{system_name}{os.sep}"

            model = set_function_library(model, system_name=system_name, use_default_library=use_default_library)
            model.config_task["threshold"] = 10 ** -5
            model.config_task["dataset"] = f"{path_data}{system_name}{os.sep}{fname_data}".format(system_name, snr, iinit, eq_sym)

            # Set the system
            t1 = time.time()

            model.train()

            # save duration in a text file
            os.makedirs(f"{path_out}{system_name}", exist_ok=True)
            dur = time.time() - t1
            with open(f"{path_out}{system_name}{os.sep}dso_{system_name}_snr{snr}_init{iinit}_{eq_sym}_duration.txt", "a") as f:
                f.write(f"{dur}\n")


def set_function_library(model, system_name, use_default_library=False):
    if use_default_library:
        model.config_task["function_set"] = model.config_task["function_set"] + ["div", "sin", "cos", "tan"]
    else:
        if system_name in ['bacres', 'shearflow', 'glider']:
            model.config_task["function_set"] = model.config_task["function_set"] + ["div", "sin", "cos"]
        elif system_name in ['barmag', 'cphase']:
            model.config_task["function_set"] = model.config_task["function_set"] + ["sin", "cos"]
        elif system_name in ['lv', 'predprey', 'vdp', 'stl', 'lorenz']:
            model.config_task["function_set"] = model.config_task["function_set"] + ["div"]
    return model

def get_data(data_true_fname, systems, system_name, eq_idx, eq_sym, observability='full'):
    data_true = pd.read_csv(data_true_fname)
    data = np.array(data_true[['t'] + systems[system_name].state_vars])

    # Get true noisy data derivatives
    try:
        data_der = np.array(data_true[['d' + systems[system_name].state_vars[i] for i in
                                       range(len(systems[system_name].state_vars))]])
    except:
        raise ValueError("No derivatives in the data. Please, generate data with derivatives. Derivatives"
                         "should be named as d<state_variable_name> in the csv file.")

    return data, data_der

def get_configuration(iinput, systems_collection, data_sizes, snrs, n_data):

    # check if lorenz is in the keys of the systems_collection and adjust the combinations of state vars accordingly
    if 'lorenz' in systems_collection.keys():
        # remove lorenz from system_collection
        new_systems_collection = systems_collection.copy()
        lorenz = {'lorenz': new_systems_collection.pop('lorenz')}
        combinations_other = list(itertools.product(new_systems_collection, data_sizes, snrs, np.arange(n_data), ['x', 'y']))
        combinations_lorenz = list(itertools.product(lorenz, data_sizes, snrs, np.arange(n_data), ['x', 'y', 'z']))
        combinations = combinations_other + combinations_lorenz
    else:
        combinations = list(itertools.product(systems_collection, data_sizes, snrs, np.arange(n_data), ['x', 'y']))

    return combinations[iinput]


##
if __name__ == '__main__':

    # modifyable settings
    iinput = 0  # int(sys.argv[1])
    data_sizes = ["small", "large"]
    snrs = [None, 30, 13]   # [None, 30, 13]
    n_data = 4

    system_name, data_size, snr, iinit, eq_sym = get_configuration(iinput, systems_collection, data_sizes, snrs, n_data)

    # fixed settings
    systems = {system_name: systems_collection[system_name]}
    method = "dso"
    exp_version = "e1"
    observability = "full"
    exp_type = f"sysident_num_{observability}"
    data_type = "train"
    time_end = 20 if data_size == "large" else 10
    time_step = 0.01 if data_size == "large" else 0.1

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    root_dir = "D:\\Experiments\\symreg_methods_comparison"
    # add root dir to the path
    sys.path.append(root_dir)

    path_data = f".{os.sep}data{os.sep}{data_type}{os.sep}{data_size}_for_dso{os.sep}"
    fname_data = f"data_{{}}_len{time_end}_rate{str(time_step).replace('.', '')}_snr{{}}_init{{}}_{{}}.csv"
    path_out = f".{os.sep}results{os.sep}{exp_type}{os.sep}{method}{os.sep}{exp_version}{os.sep}"

    run_dso(systems=systems,
            path_data=path_data,
            fname_data=fname_data,
            iinit=iinit,
            eq_sym=eq_sym,
            snrs=[snr],
            observability=observability,
            use_default_library=False,
            path_out=path_out,
            path_config=f"{root_dir}{os.sep}src{os.sep}check_dso{os.sep}config_regression.json")

##
