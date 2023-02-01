# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 09:23:20 2022

@author: fege9
"""
import importlib
import json
import pathlib
import pandas as pd
import time
import numpy as np
#from . import helper_func as fun
import helper_func as fun
import sys

#use new seed and save used seed
p = pathlib.Path(__file__)
try:
    np.loadtxt(f"{p.parents[1]}\Results\seeds.txt")
except IOError:
    open(f"{p.parents[1]}\Results\seeds.txt", 'x')
seeds = np.loadtxt(f"{p.parents[1]}\Results\seeds.txt")
seeds = list(np.atleast_1d(seeds))
GLOBAL_RNG = None
seed = np.random.randint(1000000, size=1)
while seed in seeds:
    seed = np.random.randint(1000000, size=1)
seeds.append(seed[0])
np.array(seeds)
np.savetxt(f"{p.parents[1]}\Results\seeds.txt", seeds)
fun.seeded_rng(seed[0])
print("Seed used: ", seed[0])


def main(circuit_file, settings_file):
    # import circuit file (somewhat hacky...)        
    circ_file = pathlib.Path(circuit_file)
    assert circ_file.exists()
    print("Importing QCircuit from ", circ_file)
    sys.path.append(str(circ_file.parent.absolute()))
    circ = __import__(circ_file.stem)

    # read settings from file
    assert pathlib.Path(settings_file).exists()
    with open(settings_file, 'r') as settingsf:
        settings = json.load(settingsf)
        settings = adjust_settings(circ, settings)
    folder=circ_file.parent
    #sys.stdout = open("{}\Results_GP_{}".format(folder,settings["filename"]), "w")
    toolbox = fun.deap_init(settings, circ)
    print("Initialization done! Doing GP now...")

    start = time.time()
    pop, pareto, log, time_stamps, pareto_fitness_vals, evals = fun.nsga3(toolbox, settings)
    #print(time_stamps)
    end_gp=time.time()
    
    if settings.get("reduce_pareto") != None:
        pareto=fun.reduce_pareto(pareto, settings)
    res=fun.get_pareto_final_qc(settings, circ, pareto)
    end=time.time()
    gp_time=end_gp-start
    sel_time=end-end_gp
    overall_time=end-start

    print("The found oracle is/are: ", res)
    print("GP, Selection, Overall duration: ", gp_time, sel_time, overall_time)

    for i in range(len(pareto)):
        print(pareto[i], pareto[i].fitness.values)

    if settings["sel_scheme"] != "Manual":
        with open(f"{p.parents[1]}\Results\{settings['filename']}_{seed[0]}", "w") as CIRC_file:
            CIRC_file.write(res[0].qasm())
        #with open("{}\{}".format(folder,settings["filename"]), "w") as CIRC_file:
        #print(res[0].draw())
        print("Settings for this run are:")
        print(settings)

    if settings["sel_scheme"] == "Manual":
        for count,ind in enumerate(res):
            with open(f"{p.parents[1]}\Results\{settings['filename']}_{count}", "w") as CIRC_file:
                CIRC_file.write(ind[0].qasm())
                #print(ind[0].draw())
        print("Settings for this run are:")
        print(settings)

    #write log to CSV
    df_log = pd.DataFrame(log)
    df_log.to_csv(f"{p.parents[1]}\Results\Logbook_CSV_{seed[0]}_{settings['filename']}.csv", index=False)  # Writing to a CSV file

    #write number of evaluations per generation to json
    with open(f"{p.parents[1]}\Results\Evals_GENs_{seed[0]}_{settings['filename']}.json", 'w') as outfile:
        json.dump(evals, outfile)

    #write timestamps to json
    with open(f"{p.parents[1]}\Results\Timestamps_GENs_{seed[0]}_{settings['filename']}.json", 'w') as outfile:
        json.dump(time_stamps, outfile)

    #write fitness values of Pareto-Front individuals of each generation to json
    fitness_json = json.dumps(pareto_fitness_vals)
    with open(f"{p.parents[1]}\Results\Fitness_Pareto_GENs_{seed[0]}_{settings['filename']}.json", 'w') as outfile:
        json.dump(fitness_json,outfile)



def adjust_settings(circ, update_settings):
    # automatically generated

    #update default settings
    settings = {"N": 20, "use_numerical_optimizer": "yes", "NGEN": 10, "CXPB": 1.0, "MUTPB": 1.0, "gateset": "variable", "numerical_optimizer": "Nelder-Mead", "Sorted_order": [-1,2,3,4,5]}
    settings.update(update_settings)
    settings["prob"]=settings.get("prob")
    settings["weights_mut"]=settings.get("weights_mut")
    settings["weights_cx"]=settings.get("weights_cx")
    settings["opt_within"] = settings.get("opt_within")
    settings["opt_final"] = settings.get("opt_final")
    settings["opt_select"] = settings.get("opt_select")
    settings["sel_scheme"] = settings.get("sel_scheme")
   
    if settings["gateset"] == "variable":
        import Gates_var as Gates
    elif settings["gateset"] == "fixed":
        import Gates_fix as Gates
    else:
        print("Couldn't identify gateset '", settings["gateset"], "', select 'variable' or 'fixed'")

    settings["vals"] = Gates.gateArity
    settings["par_nums"] = Gates.gateParams
    settings["names"] = Gates.gateName
    #default for weights_gates
    if settings.get("weights_gates") == None:
        settings["weights_gates"]=[1 for i in range(len(settings["names"]))]
    settings["Non_func_gates"] = Gates.Non_func_gates
    settings["e_gate"] = Gates.ElementaryGate()
    settings["overall_QC"] = circ.overall_QC
    settings["overall_qubits"] = circ.overall_QC.num_qubits
    settings["qubits"] = len(circ.target_qubits_oracles[0])
    settings["weights"] = (1., -1., -1., -1., -1.)
    settings["M"] = len(settings["weights"])
    settings["target"] = np.array(settings["target"])
    settings["target"] = settings["target"] * (1. / (np.linalg.norm(settings["target"])))
    # init= init*(1./np.linalg.norm(init))

    if settings.get("weights_gates", None) is not None:
        settings["weights_gates"] = np.array(settings["weights_gates"])
        settings["weights_gates"] = settings["weights_gates"] / settings["weights_gates"].sum()
    if settings.get("weights_mut", None) is not None:
        settings["weights_mut"] = np.array(settings["weights_mut"])
        settings["weights_mut"] = settings["weights_mut"] / settings["weights_mut"].sum()
    if settings.get("weights_cx", None) is not None:
        settings["weights_cx"] = np.array(settings["weights_cx"])
        settings["weights_cx"] = settings["weights_cx"] / settings["weights_cx"].sum()
    settings["operand"] = range(settings["qubits"])
    settings["operator"] = list(Gates.gateName.keys())
    settings["basis"] = []
    for i in settings["operator"]:
        name = Gates.gateName.get(i)
        settings["basis"].append(name)

    return settings



#uncomment/comment for using "main" as function rather than right here for developing
#uncomment for use in console
if __name__ == "__main__":
    #circuit = sys.argv[1]
    #settings = sys.argv[2]

    circuit = f"{p.parents[1]}\examples\Grover\QC.py"
    settings = f"{p.parents[1]}\examples\Grover\settings.json"
    assert pathlib.Path(settings).exists()
    main(circuit, settings) #uncomment




