import os
import sys
import pandas as pd
import numpy as np
from src.utils.systems_collection import systems_collection
import re
import sympy as sp

def gather_for_validation(systems, data_sizes, snrs, n_data, path_results_in, fname_results_in, path_out,
                          print_expressions=False, save_results=True):

    all_models = []

    for data_size in data_sizes:
        for sys_name in systems_collection:
            for snr in snrs:
                for iinit in range(n_data):

                    # find the correct file
                    path_results = path_results_in.format(sys_name)
                    fname_results = fname_results_in.format(sys_name, data_size, snr, iinit)

                    try:
                        results = pd.read_csv(f"{path_results}{fname_results}", sep='\t')
                        print(f"Results for {fname_results} loaded.")
                    except:
                        print(f"Results not loaded correctly: {fname_results}")

                    for imodel in range(len(results)):
                        # check if expression is not nan
                        if results['expression'][imodel] == results['expression'][imodel]:
                            exprs = results['expression'][imodel].split(',')
                            for ieq, eqsym in enumerate(systems[sys_name].state_vars):
                                expr = check_expression(exprs[ieq])
                                expr_sympy = sp.sympify(expr)
                                if print_expressions:
                                    print("Corrected expression:" + expr)
                                all_models.append([exp_version, method, data_size, sys_name,
                                                   "".join(systems[sys_name].state_vars), snr,
                                                   iinit, eqsym, results['duration'][imodel], expr])
                        else:
                            print(f"Expression not found for {fname_results}.")
                            for ieq, eqsym in enumerate(systems[sys_name].state_vars):
                                all_models.append([exp_version, method, data_size, sys_name,
                                                      "".join(systems[sys_name].state_vars), snr,
                                                      iinit, eqsym, np.nan, np.nan])


    # convert to dataframe
    results = pd.DataFrame(all_models,
                           columns=["exp_version", "method", "data_size", "system", "obs",
                                    "snr", "iinit", "eq", "duration", "expr"])

    # save all results as a dataframe
    if save_results:
        results.to_csv(f"{path_out}validation_gathered_results_{method}_{exp_version}_{exp_type}.csv", sep='\t')

    print("Finished gathering results")
    return results


def check_expression(expr_orig):
    # remove parenthesis, brackets and quotes
    expr = expr_orig.replace('[', '').replace(']', '').replace("'", '')
    # replace + - with -  ;  ^ with ** ; replace " 1 " with ""
    expr = expr.replace("+ -", "- ").replace("^", "**").replace(" 1 ", " ")

    # check if there is any number followed by empty space and then number 1 and replace the number 1 with nothing
    if re.search(r'\d+\s+1', expr):
        expr = re.sub(r'(\d+)\s+1', r'\1', expr)

    expr = re.sub(r'(sin|cos|cot)\s*\*\*\s*2\(([xyz])\)', replace_trig_function, expr)
    expr = re.sub(r'(?<!\*\s)(sin|cos|cot|log|exp)\([xyz]', r'* \g<0>', expr)
    expr = re.sub(r'(\d+)\s*(sin|cos|tan|log|exp)\(', r'\1 * \2(', expr)
    expr = re.sub(r'(?<![\w*])(\d+(\.\d+)?)\s*([xyz])', add_multiplication_operator, expr)
    expr = re.sub(r'\bcot\b', '1/tan', expr)
    expr = re.sub(r'(\d+)\s+tan', r'\1 * tan', expr)

    # check if a variable (x, y or z) is followed by a space and another variable and add a multiplication operator between them
    expr = re.sub(r'([xyz])\s+([xyz])', r'\1 * \2', expr)
    # check if a variable (x, y or z) is preceded by a space and a number and add a multiplication operator between them
    expr = re.sub(r'(\d+)\s+([xyz])', r'\1 * \2', expr)

    # check if y z appears in expr
    if re.search(r'y\s+z', expr):
        expr = re.sub(r'y\s+z', r'y*z', expr)
    return expr

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


##
if __name__ == "__main__":
    method = "sindy"
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

    path_results_in = f"{root_dir}{os.sep}results{os.sep}{exp_type}{os.sep}{method}{os.sep}{exp_version}{os.sep}{{}}{os.sep}"
    fname_results_in = f"{method}_{exp_type}_{exp_version}_{{}}_train_{{}}_snr{{}}_init{{}}_obs{obs}_fitted.csv"
    path_out = f"{root_dir}{os.sep}analysis{os.sep}{exp_type}{os.sep}{exp_version}{os.sep}"
    os.makedirs(path_out, exist_ok=True)

    results = gather_for_validation(systems=systems_collection,
                                    data_sizes=data_sizes,
                                    snrs=snrs,
                                    n_data=n_data,
                                    path_results_in=path_results_in,
                                    fname_results_in=fname_results_in,
                                    path_out=path_out,
                                    print_expressions=True,
                                    save_results=True)





