import numpy as np


def get_fit_settings(obs, task_type="algebraic"):

    experiment = {
        "seed": 0,
        "verbosity": 0,
    }

    parameter_estimation = {
        "dataset": None,
        "observed_vars": None,
        "optimizer": 'DE',  # default is pymoo's DE
        "simulate_separately": False,
        "max_constants": 10, # 15
        "param_bounds": ((-5, 5),),
        "default_error": 10 ** 9,
        "timeout": np.inf,
    }

    optimizer_DE = {
        "strategy": 'DE/best/1/bin',
        "max_iter": 2000,  # 1000
        "pop_size": 60,
        "mutation": 0.5,
        "cr": 0.5,
        "tol": 0.001,
        "atol": 0.001,
        "termination_threshold_error": 10 ** (-4),
        "termination_after_nochange_iters": 200,
        "termination_after_nochange_tolerance": 10 ** (-6),
        "verbose": False,
        "save_history": False,
    }

    optimizer_hyperopt = {
        "a": 1,
    }

    objective_function = {
        "use_jacobian": False,
        "teacher_forcing": False,
        "atol": 10 ** (-6),
        "rtol": 10 ** (-4),
        "max_step": 10 ** 3,
        "default_error": 10 ** 9,
        "persistent_homology": False,
    }

    settings = {
        "task_type": task_type,
        "experiment": experiment,
        "parameter_estimation": parameter_estimation,
        "optimizer_DE": optimizer_DE,
        "optimizer_hyperopt": optimizer_hyperopt,
        "objective_function": objective_function,
    }

    return settings


def get_grammar_type(system_name, universal=False):
    if system_name in ['cphase']:
        if universal:
            return "universal_xyt"
        else:
            return "phase_osc_xyt"

    elif system_name in ['lorenz']:
        if universal:
            return "universal_xyz"
        else:
            return "state_osc_xyz"
    elif system_name in ['barmag']:
        if universal:
            return "universal_xy"
        else:
            return "phase_osc_xy"
    elif system_name in ['lv', 'predprey', 'vdp', 'stl']:
        if universal:
            return "universal_xy"
        else:
            return "state_osc_xy"
    else:  # ['bacres', 'glider', 'shearflow']
        if universal:
            return "universal_xy"
        else:
            return "general"