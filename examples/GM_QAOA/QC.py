# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 10:13:40 2023

@author: fege9
"""

import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit import Gate


oracle_index = []
target_qubits_oracles = [[0,1,2,3]]
overall_QC=QuantumCircuit(4)
dummy=Gate(name="oracle", num_qubits=len(target_qubits_oracles[0]), params=[])
overall_QC.append(dummy, [0,1,2,3]) #create dummy oracle and figure out how to replace this oracle in GP, or do it via opaque gates
oracle_index.append(len(overall_QC)-1)
overall_QC.barrier()
