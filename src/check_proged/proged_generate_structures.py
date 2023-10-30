# Description: Generates structures from ProGED grammars and saves them in batches. The structures are saved in the
# folder specified by path_structures. The grammars are specified by the grammar_type and the variables. The
# batch_settings specify the number of structures to be generated, the number of variables, the probabilities of
# variables, and the seed. The batch_settings are saved in the folder specified by path_out.
#

import os
import pickle
import pandas as pd
import numpy as np
import ProGED as pg
from ProGED.generate import generate_models
from ProGED.generators.grammar_construction import construct_production, grammar_from_template
from src.utils.systems_collection import systems_collection
from src.utils.proged_utils import get_grammar_type


def create_batches(**batch_settings):
    np.random.seed(batch_settings['seed'])

    # get grammar
    if "phase_osc" in batch_settings["grammar"]:
        grammarstr = construct_production(left="S", items=["S '+' A", "A"], probs=[0.6, 0.4])
        grammarstr += construct_production(left="A", items=["'C' '*' 'sin''(' D ')'", "'C' '*' 'cos''(' D ')'"], probs=[0.5, 0.5])
        grammarstr += construct_production(left="D", items=["D '+' B", "B"], probs=[0.10, 0.90])
        grammarstr += construct_production(left="B", items=["'C' '*' V", "'C'"], probs=[0.20, 0.80])
        grammarstr += construct_production(left="V", items=batch_settings["variables"], probs=batch_settings["p_vars"])
        grammar = pg.GeneratorGrammar(grammarstr)

    elif "state_osc" in batch_settings["grammar"]:
        monods = []
        for ivar in batch_settings["variables"]:
            monod = "{} '/' '(' {} '+' 'C' ')'".format(ivar, ivar)
            monods.append(monod)

        grammarstr = construct_production(left="S", items=["S '+' T", "T"], probs=[0.6, 0.4])
        grammarstr += construct_production(left="T", items=["T '*' V", "T '*' M", "'C'"], probs=[0.35, 0.1, 0.55])
        grammarstr += construct_production(left="M", items=monods, probs=batch_settings["p_vars"])
        grammarstr += construct_production(left="V", items=batch_settings["variables"], probs=batch_settings["p_vars"])
        grammar = pg.GeneratorGrammar(grammarstr)

    elif batch_settings["grammar"] == "general":
        grammarstr = construct_production(left="S", items=["S '+' T", "T '/' '(' D ')'", "T"], probs=[0.60, 0.15, 0.25])
        grammarstr += construct_production(left="D", items=["D '+' T", "T"], probs=[0.50, 0.50])
        grammarstr += construct_production(left="T", items=["T '*' V", "T '*' A", "'C'"], probs=[0.3, 0.1, 0.6])
        grammarstr += construct_production(left="A", items=["'sin''(' B ')'", "'cos''(' B ')'"], probs=[0.50, 0.50])
        grammarstr += construct_production(left="B", items=["B '*' V", "'C'"], probs=[0.50, 0.50])
        grammarstr += construct_production(left="V", items=batch_settings["variables"], probs=batch_settings["p_vars"])
        grammar = pg.GeneratorGrammar(grammarstr)

    elif "universal" in batch_settings["grammar"]:
        grammarstr = construct_production(left="S", items=["S '+' F", "F"], probs=[0.6, 0.4])
        grammarstr += construct_production(left="F", items=["F '*' T", "F '/' T", "T"], probs=[0.2, 0.2, 0.6])
        grammarstr += construct_production(left="T", items=["R", "'C'", "V"], probs=[0.2, 0.4, 0.4])
        grammarstr += construct_production(left="R", items=["'(' S ')'"] + ["'" + f + "(' S ')'" for f in ["sin", "cos", "log", "exp"]], probs=[0.60, 0.10, 0.10, 0.10, 0.10])
        grammarstr += construct_production(left="V", items=batch_settings["variables"], probs=batch_settings["p_vars"])
        grammar = pg.GeneratorGrammar(grammarstr)
    else:
        print("Error: no such grammar.")

    # decide on dimension of generated models (warning: if additional vars are included (e.g. 't') it's not
    # gonna work properly.
    if batch_settings["whole_system"]:
        if "'t'" not in batch_settings["variables"]:
            sys_dimension = len(batch_settings["variables"])
        else:
            sys_dimension = len(batch_settings["variables"]) -1
    else:
        sys_dimension = 1

    # generate models from grammar
    sym_vars = [batch_settings["variables"][i][1] for i in range(len(batch_settings["variables"]))]
    symbols = {"x": sym_vars, "const": "C"}
    models = generate_models(grammar, symbols, system_size=sys_dimension,
                             strategy_settings={"N": batch_settings["n_samples"],
                                                "max_repeat": 50})

    model_batches = models.split(n_batches=batch_settings["n_batches"])

    # save batches
    os.makedirs(batch_settings["path_out"], exist_ok=True)

    for ib in range(batch_settings["n_batches"]):
        filename = "structs_{}_nsamp{}_nbatch{}_b{}.pg".format(batch_settings["grammar"],
                                                                batch_settings["n_samples"],
                                                                batch_settings["n_batches"],
                                                                str(ib))
        with open(batch_settings["path_out"] + filename, "wb") as file:
            pickle.dump(model_batches[ib], file)

    del model_batches, models
    # save info about grammar:
    grammar_info = batch_settings
    grammar_info['grammar_structure'] = grammarstr
    grammar_info_filename = "structs_{}_nsamp{}_nbatch{}_grammar_info.txt".format(
                                                                batch_settings["grammar"],
                                                                batch_settings["n_samples"],
                                                                batch_settings["n_batches"])
    fo = open(batch_settings["path_out"] + grammar_info_filename, "w")
    for k, v in grammar_info.items():
        fo.write(str(k) + ' >>> ' + str(v) + '\n\n')
    fo.close()

##

if __name__ == '__main__':

    method = "proged"
    exp_version = "e1"  # "e1" (constrained model search space) or "e2" (unconstrained model search space)
    observability = "partial"  # "full" or "partial"
    simulation_type = "sym" if observability == "partial" else "num"  # numerical derivation (num) or simulation by solving system of odes using initial values (sym)
    exp_type = f"sysident_{simulation_type}_{observability}"  # "sysident_sym_partial" or "sysident_num_full

    # root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    root_dir = "D:\\Experiments\\symreg_methods_comparison\\"
    path_out = f"{root_dir}results\\{exp_type}\\proged\\structures\\"

    systems = systems_collection
    if exp_type == "sysident_sym_partial":
        # take only vdp from systems
        systems = {k: v for k, v in systems.items() if k == "vdp"}

    n_samples = 2500
    n_batches = 100
    universal_grammar = False
    whole_system = True if "partial" in exp_type else False   # true for parobs experiments and False for full observability experiments

    # generate grammars based on system grammar
    for system_name in list(systems.keys()):
        grammar_type = get_grammar_type(system_name, universal=universal_grammar)
        # put every state vars inside state_vars in additional double quotes
        variables = [f"'{v}'" for v in systems[system_name].state_vars]
        p_vars = [1/len(variables)]*len(variables)

        # check if the grammar is already done
        if os.path.exists(f"{path_out}structs_{grammar_type}_nsamp{n_samples}_nbatch{n_batches}\\"):
            print(f"Grammar: {grammar_type} for system: {system_name} already done. Skipping...")
            continue
        print(f"Starting grammar: {grammar_type} for system: {system_name}")

        batch_settings = {
            "grammar": grammar_type,
            "variables": variables,
            "p_vars": p_vars,
            "whole_system": whole_system,  # if False, each equation in the model separately
            "n_samples": n_samples, #systems[sys_name].num_samples,
            "n_batches": n_batches,
            "path_out": f"{path_out}structs_{grammar_type}_nsamp{n_samples}_nbatch{n_batches}\\",
            "manual": False,
            "seed": 0,
        }

        create_batches(**batch_settings)
        print(f"Finished grammar: {grammar_type} for system: {system_name}")


## open and check created grammar
# import pickle
#filename = "structs_{}_nsamp{}_nbatch{}_b{}.pg".format(batch_settings["grammar"],
#                                                       batch_settings["n_samples"],
#                                                       batch_settings["n_batches"],
#                                                       str(0))
# with open(batch_settings["path_out"] + filename, "rb") as file:
#     models = pickle.load(file)
