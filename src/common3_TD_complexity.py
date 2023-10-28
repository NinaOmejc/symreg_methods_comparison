import pandas as pd
import sys
import os
import regex as re
import sympy as sp
import tabulate
import numpy as np
sp.init_printing()
from copy import deepcopy
from src.utils.systems_collection import systems_collection


def calculate_metrics(row_idx, row, systems, to_display_results=False):

    true_expression = get_true_expression(systems, row)
    true_expression_final = modify_expression(true_expression)

    discovered_expression = round_constants(row["expr"], n=3)
    discovered_expression_final = modify_expression(discovered_expression)

    # calculate normalized complexity
    true_complexity = complexity(true_expression_final)
    discovered_complexity = complexity(discovered_expression_final)
    complexity_norm = discovered_complexity / true_complexity

    # calculate term difference
    TD_auto, missing_terms, extra_terms, num_true_terms = calculate_term_difference(true_expression_final, discovered_expression_final)

    if to_display_results:
        display_results(row_idx, row, true_expression, true_expression_final, discovered_expression, discovered_expression_final, TD_auto, missing_terms, extra_terms)

    return TD_auto, complexity_norm, num_true_terms


def get_true_expression(systems_collection, row):

    system = systems_collection[row["system"]]
    # get index of the state variable based on the system
    ieq = system.state_vars.index(row["eq"])
    raw_orig_expression = system.model[ieq]
    edited_orig_expression, system = edit_true_expr(raw_orig_expression, system)
    orig_expression = sp.sympify(edited_orig_expression).subs(system.model_params)
    return orig_expression

def edit_true_expr(expr, system):
    # Use a regular expression to replace "S" only when it's a standalone word
    if re.search(r'\bS\b', expr):
        expr = re.sub(r'\bS\b', 's', expr)
        system.model_params['s'] = system.model_params['S']
        print("Warning: 'S' was replaced with 's' in the expression. It was also added in the model_params dict.")
    return expr, system


def modify_expression(expr):

    try:
        expr_expanded = round_constants(sp.expand(sp.trigsimp(sp.expand(expr))), n=3)
    except:
        try:
            expr_expanded = round_constants(sp.expand(expr), n=3)
        except Exception as e:
            print(f"Error {e}:\nExpression could not be modified. Check the expression. Defaulting to the expression '0'.")
            expr_expanded = sp.sympify("0")

    # check if 0.e-3 is in the expression
    expr_expanded = check_expressions_for_e_zeros(expr_expanded)
    expr_expanded_c = replace_numbers_with_c(expr_expanded)
    # check that every function has a 'C' in front and replace all higher exponents of 'C' with an exponent of 1
    expr_expanded_c = recheck_c_in_expression(expr_expanded_c)
    return expr_expanded_c


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


def check_expressions_for_e_zeros(expr):
    if re.search(r'0\.\d*e-\d', str(expr)):
        # replace 0.e-3 with 0
        expr = re.sub(r'0\.\d*e-\d', '0', str(expr))
        # transform back to sympy
        expr = sp.sympify(str(expr))
    return expr

def replace_numbers_with_c(expr):

    # Replace numbers with 'C', except in exponents or '-1' in rational terms
    if expr.is_Pow:
        base = expr.base
        exponent = expr.exp
        new_base = replace_numbers_with_c(base)
        return sp.Symbol('C') * sp.Pow(new_base, exponent)
    elif expr.is_Rational and expr.denominator == -1:
        return expr
    elif expr.is_number:
        return sp.Symbol('C')
    elif expr.is_Symbol:
        return sp.Symbol('C') * expr
    else:
        return expr.xreplace({atom: replace_numbers_with_c(atom) for atom in expr.args})


def recheck_c_in_expression(expr):

    # add to every function a 'C' in front (just in case is missing)
    for a in sp.preorder_traversal(expr):
        if isinstance(a, sp.Function):
            expr = expr.subs(a, sp.Symbol('C') * a)

    # repeat until no more exponents of 'C' are greater than 1 and not just 2
    while any([a for a in sp.preorder_traversal(expr) if isinstance(a, sp.Pow) and a.base == sp.Symbol('C') and a.exp > 1]):
        for a in sp.preorder_traversal(expr):
            if isinstance(a, sp.Pow):
                base = a.base
                exponent = a.exp
                if base == sp.Symbol('C'):
                    if exponent > 1:
                        expr = expr.subs(a, sp.Pow(base, 1))

    return expr

def complexity(expr):
    if isinstance(expr, str):
        if len(expr) == 0:
            return 0
        else:
            expr = sp.sympify(expr)
    c=0
    for arg in sp.preorder_traversal(expr):
        c += 1
    return c

def count_ops(expr):
    if isinstance(expr, str):
        if len(expr) == 0:
            return 0
        else:
            expr = sp.sympify(expr)
    return sp.count_ops(expr)


# true_expr, model_expr = true_expression_final, discovered_expression_final
def calculate_term_difference(true_expr, model_expr):

    # Count the number of terms in each expression
    n_true_terms = count_terms(true_expr)
    n_model_terms = count_terms(model_expr)

    # Find the common terms
    common_terms = find_common_terms(true_expr, model_expr)

    # Calculate the missing and extra terms
    n_missing_terms = n_true_terms - len(common_terms)
    n_extra_terms = n_model_terms - len(common_terms)

    # Calculate the term difference metric
    term_difference = n_missing_terms + n_extra_terms

    return term_difference, n_missing_terms, n_extra_terms, n_true_terms


def count_terms(expression):
    # Count the number of terms in the expression
    return len(sp.Add.make_args(expression))


def find_common_terms(true_expr, model_expr):

    # Find the common terms between two expressions
    terms_true = list(sp.Add.make_args(true_expr))
    terms_model = list(sp.Add.make_args(model_expr))

    terms_true_modified = set(modify_terms(terms_true))
    terms_model_modified = set(modify_terms(terms_model))

    common_terms = terms_true_modified.intersection(terms_model_modified)
    return common_terms


def modify_terms(terms):
    output_terms = list(terms)  # Create a copy of the original terms
    for term in terms:
        output_terms.append(replace_trig(term))

    if sp.sympify("C*sin(C*y + C)") in output_terms:
        output_terms.append(sp.sympify("C*cos(C*y)"))
        output_terms.append(sp.sympify("C*sin(C*y)"))
    if sp.sympify("C*cos(C*y + C)") in output_terms:
        output_terms.append(sp.sympify("C*sin(C*y)"))
        output_terms.append(sp.sympify("C*cos(C*y)"))
    if sp.sympify("C*sin(C*x + C)") in output_terms:
        output_terms.append(sp.sympify("C*cos(C*x)"))
        output_terms.append(sp.sympify("C*sin(C*x)"))
    if sp.sympify("C*cos(C*x + C)") in output_terms:
        output_terms.append(sp.sympify("C*sin(C*x)"))
        output_terms.append(sp.sympify("C*cos(C*x)"))

    return output_terms


def replace_trig(term):
    # Replace trig functions with their equivalent, cotangens with 1/tan and tan with sin/cos
    new_args = []
    if len(term.args) != 0:
        for arg in term.args:
            if isinstance(arg, sp.cot):
                new_args.append(1/sp.tan(arg.args[0]))
            elif isinstance(arg, sp.tan):
                new_args.append(sp.sin(arg.args[0])/sp.cos(arg.args[0]))
            else:
                new_args.append(arg)
        return term.func(*new_args)
    return term


def display_results(row_idx, row, orig_expanded, orig_expanded_consts, model_expanded, model_expanded_consts, iTD, missing_terms, extra_terms):
    print("\n------------------------------------------")
    print(f"Idx: {row_idx} | Method: {row.method} | Data len: {row.data_size} | System: {row.system} | SNR: {row.snr}")
    print("------------------------------------------\n")
    print(f"Orig expanded eq:\n{orig_expanded}\n")
    print(f"Expanded Modeled Eq:\n{model_expanded}\n")
    print(f"Orig expanded eq with only constants:\n{orig_expanded_consts}\n")
    print(f"Expanded Modeled Eq with only constants:\n{model_expanded_consts}\n")
    print(f"Term Difference Metric per Eq: {iTD}")
    print(f"Missing Terms per Eq: {missing_terms}")
    print(f"Extra Terms per Eq: {extra_terms}\n")
    print("------------------------------------------\n")


##

if __name__ == "__main__":


    exp_version = "e2"
    exp_type = 'sysident_num_full'
    methods = ["proged", "sindy", "dso"]
    obs = "full"
    merge_func_val = "mean"

    data_sizes = ["small", "large"]
    snrs = [None, 30, 13]  # [None, 30, 13]
    n_train_data = 4
    n_val_data = 4
    n_test_data = 4
    check_manually = False
    print_pretty_csv_table_of_results = True  # module tabulate needed

    root_dir = "D:\\Experiments\\symreg_methods_comparison"
    sys.path.append(root_dir)
    path_results_in = f"{root_dir}{os.sep}analysis{os.sep}{exp_type}{os.sep}{exp_version}{os.sep}"
    fname_results_in = f"best_results_{exp_version}_withValTE{merge_func_val}_withTestTE.csv"
    fname_results_out = f"best_results_{exp_version}_withValTE{merge_func_val}_withTestTE_withTD_compl.csv"

    # check if fname_results_out exists and if, load it
    if os.path.isfile(f"{path_results_in}{fname_results_out}"):
        print(f"Loading results from {path_results_in}{fname_results_out}")
        results = pd.read_csv(f"{path_results_in}{fname_results_out}", sep=",")
    else:
        print(f"Loading results from {path_results_in}{fname_results_in}")
        results = pd.read_csv(f"{path_results_in}{fname_results_in}", sep=",")
    # if the TD_auto column is not present, create it
    if "TD_auto" not in results.columns:
        results["TD_auto"] = np.nan
        results["TD_manual"] = np.nan
        results["TD_auto_norm"] = np.nan
        results["TD_manual_norm"] = np.nan

    for row_idx, row in results.iterrows():
        # check if row has already been processed (if not empty)
        if not pd.isna(row["TD_auto"]):
            continue
        TD_auto, normalized_complexity_score, n_true_terms = calculate_metrics(row_idx, row, systems_collection, to_display_results=check_manually)
        results.loc[row_idx, "TD_auto"] = TD_auto
        results.loc[row_idx, "complexity_norm"] = normalized_complexity_score
        results.loc[row_idx, "TD_auto_norm"] = TD_auto / n_true_terms

        # check manually if needed
        while pd.isna(results.loc[row_idx, "TD_manual"]):
            manual_TD_input = input("manual TD: ") if check_manually else TD_auto
            # check if empty input string
            if manual_TD_input == "" or int(manual_TD_input) == 99:
                results.loc[row_idx, "TD_manual"] = TD_auto
                results.loc[row_idx, "TD_manual_norm"] = TD_auto / n_true_terms
            else:
                try:
                    results.loc[row_idx, "TD_manual"] = int(manual_TD_input)
                    results.loc[row_idx, "TD_manual_norm"] = int(manual_TD_input) / n_true_terms
                except ValueError as e:
                    print(f"Error {e}, repeat input...")

        # save intermediate results (useful as a backup due to manual checking and typing)
        results.to_csv(f"{path_results_in}{fname_results_out}", index=False, sep=",")

    results = results.sort_values(['data_size', 'snr', 'method'])
    results.to_csv(f"{path_results_in}{fname_results_out}", index=False, sep=",")

    if print_pretty_csv_table_of_results:
        import tabulate
        content_all = tabulate.tabulate(results.values.tolist(), list(results.columns), tablefmt="plain")
        open(f"{path_results_in}{fname_results_out[:-4]}_pretty.txt", "w").write(content_all)

