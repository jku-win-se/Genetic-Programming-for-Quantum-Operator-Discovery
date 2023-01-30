# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 10:14:11 2023

@author: fege9
"""

import numpy as np
import json
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit import Gate
#dicke-state use case from GM-QAOA

dicke_42=np.zeros(16)
for i in range(2**4):
    a=(bin(i)[2:])
    count=a.count("1")
    if count==2:
        dicke_42[i]=1
"""dicke_53=np.zeros(32)
for i in range(2**5):
    a=(bin(i)[2:])
    count=a.count("1")
    if count==3:
        dicke_53[i]=1"""


settings={}
settings["target"]=list(dicke_42)
settings["target_pos"]=1
settings["filename"]="U_S.txt"
settings["max_gates"]=30
settings["N"]=30
settings["NGEN"]=15


with open ("settings.json","w") as f:
    json.dump(settings,f)

#GM-QAOAs
"""
target=dicke_42
target_pos=1
gateset="variable" #"fixed", "variable"
filename="U_S.txt"
max_gates = 30
N=30 #population size, assuming H~N in NSGA3
NGEN = 15
CXPB = 1.0
MUTPB = 1.0
weights_gates=None  #weights for gate probabilities used in tuple construct
prob=None #probability for geometric distribution
numerical_optimizer='Nelder-Mead' #choose one of Nelder-Mead, COBYLA, BFGS,...(see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)
weights_mut=None#np.array([1,4,6,4,3,5,6,7,3]) #enter 9 weights reflecting probability for mutator operator
weights_cx=None#np.array([3,4,5,2])
opt_within = None#["Q1"] #encoding for optimization possibilities: CC: CommutativeCancelling, OQ1:Optimize1qGatesDecomposition, Q1:Qiskit Level1 Optimization, Q2:Qiskit Level2 Optimization, ZX1: basic_optimization
#attention when using ZX: Circuit is translated into new basis --> circuit in new basis is optimized --> can be larger than original circuit!!
opt_final = None#["Q1"]#optimization for final circuit; do optimization according to every element in list
opt_select=None#["Q1"] #circuit optimization for Pareto-front
sel_scheme=None #Alternatives: None-->first QC when Pareto sorted according to fitness values, "Manual"->full Pareto front
weights2 = (1000,-1.,-1.,-1.,-1.) #weights for selecting best circuit from final population;

#framework input
oracle_index = []
target_qubits = [[0,1,2,3]]
overall_QC=QuantumCircuit(4)
dummy=Gate(name="oracle", num_qubits=len(target_qubits[0]), params=[])
overall_QC.append(dummy, [0,1,2,3]) #create dummy oracle and figure out how to replace this oracle in GP, or do it via opaque gates
oracle_index.append(len(overall_QC)-1)
overall_QC.barrier()

print(target)"""