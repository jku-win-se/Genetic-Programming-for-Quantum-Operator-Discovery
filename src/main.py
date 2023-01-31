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


GLOBAL_RNG = None
seed = np.random.randint(1000000, size=1)
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
        adjust_settings(circ, settings)
    folder=circ_file.parent
    #sys.stdout = open("{}\Results_GP_{}".format(folder,settings["filename"]), "w")
    toolbox = fun.deap_init(settings, circ)
    print("Initialization done! Doing GP now...")

    start = time.time()
    pop, pareto, log, time_stamps, pareto_fitness_vals = fun.nsga3(toolbox, settings)
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
        with open("{}\{}".format(folder,settings["filename"]), "w") as CIRC_file:
            CIRC_file.write(res[0].qasm())
        #print(res[0].draw())
        print("Settings for this run are:")
        print(settings)

    if settings["sel_scheme"] == "Manual":
        for count,ind in enumerate(res):
            with open("{}\{}_Ind{}".format(folder,settings["filename"],count), "w") as CIRC_file:
                CIRC_file.write(ind[0].qasm())
                #print(ind[0].draw())
        print("Settings for this run are:")
        print(settings)

    #write log to CSV
    df_log = pd.DataFrame(log)
    df_log.to_csv(f"{folder}/Logbook_CSV_{settings['filename']}.csv", index=False)  # Writing to a CSV file

    # write log to json
    #log_json = json.dumps(log)
    #with open(f"{folder}/Logbook_JSON_{settings['filename']}.json", 'w') as outfile:
    #    json.dump(log_json, outfile)

    #write timestamps to json
    with open(f"{folder}/Timestamps_GENs_{settings['filename']}.json", 'w') as outfile:
        json.dump(time_stamps, outfile)

    #write fitness values of Pareto-Front individuals of each generation to json
    fitness_json = json.dumps(pareto_fitness_vals)
    with open(f"{folder}/Fitness_Pareto_GENs_{settings['filename']}.json", 'w') as outfile:
        json.dump(fitness_json,outfile)



def adjust_settings(circ, settings):
    # automatically generated
    
    #add default values:
    if settings.get("N") == None:
        settings["N"] = 20
    if settings.get("use_numerical_optimizer") == None:
        settings["use_numerical_optimizer"] = "yes"
    if settings.get("NGEN") == None:
        settings["NGEN"] = 10
    if settings.get("CXPB") == None:
        settings["CXPB"] = 1.0
    if settings.get("MUTPB") == None:
        settings["MUTPB"] = 1.0
    if settings.get("gateset") == None:
        settings["gateset"] = "variable"
    if settings.get("numerical_optimizer") == None:
        settings["numerical_optimizer"] = "Nelder-Mead"
    if settings.get("Sorted_order") == None:
        settings["Sorted_order"]=[-1,2,3,4,5]
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
    circuit="C:/Users/fege9/anaconda3/Model-based-QC/GP/Python_Scrips/GECCO2023_Artifact/examples/Grover/QC.py"
    settings="C:/Users/fege9/anaconda3/Model-based-QC/GP/Python_Scrips/GECCO2023_Artifact/examples/Grover/settings.json"
    assert pathlib.Path(settings).exists()
    main(circuit, settings) #uncomment




