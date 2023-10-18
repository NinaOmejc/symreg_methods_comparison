
## validation of models calculated by numerical differentiation separately for each system variable - full observability

import os
import sys
import re
import pandas as pd
import numpy as np
import sympy as sym
import itertools
from generate_data.systems_collection import strogatz, mysystems
from scipy.integrate import odeint
from scipy.interpolate import interp1d

def get_settings(iinput, methods, data_versions, systems, snrs):
    sys_names = list(systems.keys())
    combinations = []
    for sys_name in sys_names:
        lhs_vars = systems[sys_name].lhs_vars
        combinations.append(list(itertools.product(methods, data_versions, [sys_name], snrs, lhs_vars)))
    combinations = [item for sublist in combinations for item in sublist]
    return combinations[iinput-1]

def calculate_TE(traj_true, traj_model):  # trajectory error
    TEx = np.sqrt((np.mean((traj_model[:, 0] - traj_true[:, 0]) ** 2))) / np.std(traj_true[:, 0])
    TEy = np.sqrt((np.mean((traj_model[:, 1] - traj_true[:, 1]) ** 2))) / np.std(traj_true[:, 1])
    TE = TEx + TEy
    if sys_name == 'lorenz':
        TEz = np.sqrt((np.mean((traj_model[:, 2] - traj_true[:, 2]) ** 2))) / np.std(traj_true[:, 2])
        TE = TE + TEz
    return TE


def replace_trig_function(match):
    function = match.group(1)
    variable = match.group(2)

    if function == 'sin':
        return f'sin({variable})**2'
    elif function == 'cos':
        return f'cos({variable})**2'
    elif function == 'cot':
        return f'cot({variable})**2'

def add_multiplication_operator(match):
    return match.group(1) + ' * ' + match.group(3)

def correct_expression(expr):
    if len(expr) == 1:
        expr = expr[0]
    for ie in range(len(expr)):
        if "+ -" in expr[ie]:
            expr[ie] = expr[ie].replace("+ -", "- ")
        if "^" in expr[ie]:
            expr[ie] = expr[ie].replace("^", " ** ")

        expr[ie] = re.sub(r'(sin|cos|cot)\s*\*\*\s*2\(([xyz])\)', replace_trig_function, expr[ie])
        expr[ie] = re.sub(r'(?<!\*\s)(sin|cos|cot)\([xyz]', r'* \g<0>', expr[ie])
        expr[ie] = re.sub(r'(?<![\w*])(\d+(\.\d+)?)\s*([xyz])', add_multiplication_operator, expr[ie])
        expr[ie] = re.sub(r'\bcot\b', '1/tan', expr[ie])
        expr[ie] = re.sub(r'(\d+)\s+tan', r'\1 * tan', expr[ie])
        # expr[ie] = re.sub(r'(?<!\*\s)cos\([xyz]\)', r'* \g<0>', expr[ie])
        # expr[ie] = re.sub(r'(?<!\*\s)cot\([xyz]\)', r'* \g<0>', expr[ie])
    return expr

def get_errors_per_expr(data_sets, data_inits, expr):
    val_errors_per_expr = []
    #print(expr)
    expr_editted = correct_expression(expr)
    #print(expr_editted)
    expr_func = [sym.lambdify(['t'] + systems[sys_name].lhs_vars, iexpr) for iexpr in expr_editted]
    time = np.arange(0, sim_time, sim_step)

    for ival in list(validation_inits.keys()):

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