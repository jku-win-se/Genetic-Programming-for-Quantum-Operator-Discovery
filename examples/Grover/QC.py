# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 16:23:43 2022

@author: fege9
"""

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit import Gate
import numpy as np

oracle_index = []
target_qubits_oracles = [[0,1,2]]
overall_QC=QuantumCircuit(3)
for i in range(3):
    overall_QC.h(i)
overall_QC.barrier()
dummy=Gate(name="oracle", num_qubits=len(target_qubits_oracles[0]), params=[])
overall_QC.append(dummy, [0,1,2]) #create dummy oracle and figure out how to replace this oracle in GP, or do it via opaque gates
oracle_index.append(len(overall_QC)-1)
overall_QC.barrier()
for qubit in range(3):
    overall_QC.h(qubit)
for qubit in range(3):
    overall_QC.x(qubit)
overall_QC.h(2)
overall_QC.mct(list(range(2)), 2)
overall_QC.h(2)
for qubit in range(3):
    overall_QC.x(qubit)
for qubit in range(3):
    overall_QC.h(qubit)
overall_QC.barrier()


#print(overall_QC.draw())
#print(target_qubits_oracles)
#print(oracle_index)



