# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 09:28:25 2022

@author: fege9
"""

#elementary Quantum Gates
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit import Gate
import numpy as np
from numpy.random import choice, random as rand, randint

"RX, RY, and RZ rotations with fixed discrete angles 2πk/8 with k between 1-7"

Non_func_gates=["x","y","z","cx","cy","cz","swap","h"]


H='Hadamard'
X='Not'
Y='Pauli Y'
Z='Pauli Z'
#I='Identity'

#CH='Controlled Hadamard'
CX='Controlled Not'
CY='Controlled Y'
CZ='Controlled Z'
RZ1='Rotate Z 1/8'
RZ2='Rotate Z 2/8'
RZ3='Rotate Z 3/8'
RZ4='Rotate Z 4/8'
RZ5='Rotate Z 5/8'
RZ6='Rotate Z 6/8'
RZ7='Rotate Z 7/8'
RX1='Rotate X 1/8'
RX2='Rotate X 2/8'
RX3='Rotate X 3/8'
RX4='Rotate X 4/8'
RX5='Rotate X 5/8'
RX6='Rotate X 6/8'
RX7='Rotate X 7/8'
RY1='Rotate Y 1/8'
RY2='Rotate Y 2/8'
RY3='Rotate Y 3/8'
RY4='Rotate Y 4/8'
RY5='Rotate Y 5/8'
RY6='Rotate Y 6/8'
RY7='Rotate Y 7/8'
#CU='Controlled U'


#S='Clifford S'
#SC='Clifford S Conjugate'

#T='SquareRoot S'
#TC='T Conjugate'

SWAP='Swap'


gateArity={
        H:1,
        X:1,
        Y:1,
        Z:1,
#        I:1,
#        
        #CH:2,
        CX:2,
        CY:2,
        CZ:2,
#        
        #S:1,
        #SC:1,
#        
        #T:1,
        #TC:1,
#        
        SWAP:2,
        RZ1:1,
        RZ2:1,
        RZ3:1,
         RZ4:1,
         RZ5:1,
         RZ6:1,
         RZ7:1,
       RX1:1,
    RX2:1,
    RX3:1,
    RX4:1,
    RX5:1,
    RX6:1,
    RX7:1,
    RY1:1,
    RY2:1,
    RY3:1,
    RY4:1,
    RY5:1,
    RY6:1,
    RY7:1,
    
#        CU:2
        }

gateName={
        H:'h',
        X:'x',
        Y:'y',
        Z:'z',
#        I:'i',
        
        #CH:'ch',
        CX:'cx',
        CY:'cy',
        CZ:'cz',
        
        #S:'s',
        #SC:'sdg',
        
        #T:'t',
        #TC:'tdg',
        
        SWAP:'swap',
        RZ1:'rz1',
    RZ2:'rz2',
    RZ3:'rz3',
    RZ4:'rz4',
    RZ5:'rz5',
    RZ6:'rz6',
    RZ7:'rz7',
        RX1:'rx1',
    RX2:'rx2',
    RX3:'rx3',
    RX4:'rx4',
    RX5:'rx5',
    RX6:'rx6',
    RX7:'rx7',
       RY1:'ry1',
    RY2:'ry2',
    RY3:'ry3',
    RY4:'ry4',
    RY5:'ry5',
    RY6:'ry6',
    RY7:'ry7',
 #       CU:'cu'
        }

#give number of parameters for each gate
gateParams={
     H:0,
     X:0,
     Y:0,
     Z:0,
#     I:0,
#        
        #CH:2,
     CX:0,
     CY:0,
     CZ:0,
#        
        #S:1,
        #SC:1,
#        
        #T:1,
        #TC:1,
#        
     SWAP:0,
     RZ1:0,
         RZ2:0,
         RZ3:0,
     RZ4:0,
         RZ5:0,
         RZ6:0,
         RZ7:0,
     RX1:0,
    RX2:0,
    RX3:0,
    RX4:0,
    RX5:0,
    RX6:0,
    RX7:0,
     RY1:0,
    RY2:0,
    RY3:0,
    RY4:0,
    RY5:0,
    RY6:0,
    RY7:0,
   #  CU:0
    
}

class ElementaryGate:
    def __init__(self):
        self.type="ElementaryGate"
    def rz1(self,target_qubits=None, control_qubits=None, inverse=False):
            qc=QuantumCircuit(1)
            qc.rz((1/8)*2*np.pi,0)
            rz=qc.to_gate()
            rz.label="RZ1"
            return rz
    def rz2(self,target_qubits=None, control_qubits=None, inverse=False):
            qc=QuantumCircuit(1)
            qc.rz((2/8)*2*np.pi,0)
            rz=qc.to_gate()
            rz.label="RZ2"
            return rz
    def rz3(self,target_qubits=None, control_qubits=None, inverse=False):
            qc=QuantumCircuit(1)
            qc.rz((3/8)*2*np.pi,0)
            rz=qc.to_gate()
            rz.label="RZ3"
            return rz
    def rz4(self,target_qubits=None, control_qubits=None, inverse=False):
            qc=QuantumCircuit(1)
            qc.rz((4/8)*2*np.pi,0)
            rz=qc.to_gate()
            rz.label="RZ4"
            return rz
    def rz5(self,target_qubits=None, control_qubits=None, inverse=False):
            qc=QuantumCircuit(1)
            qc.rz((5/8)*2*np.pi,0)
            rz=qc.to_gate()
            rz.label="RZ5"
            return rz
    def rz6(self,target_qubits=None, control_qubits=None, inverse=False):
            qc=QuantumCircuit(1)
            qc.rz((6/8)*2*np.pi,0)
            rz=qc.to_gate()
            rz.label="RZ6"
            return rz
    def rz7(self,target_qubits=None, control_qubits=None, inverse=False):
            qc=QuantumCircuit(1)
            qc.rz((7/8)*2*np.pi,0)
            rz=qc.to_gate()
            rz.label="RZ7"
            return rz
    def rx1(self,target_qubits=None, control_qubits=None, inverse=False):
            qc=QuantumCircuit(1)
            qc.rx((1/8)*2*np.pi,0)
            rx=qc.to_gate()
            rx.label="RX1"
            return rx
    def rx2(self,target_qubits=None, control_qubits=None, inverse=False):
            qc=QuantumCircuit(1)
            qc.rx((2/8)*2*np.pi,0)
            rx=qc.to_gate()
            rx.label="RX2"
            return rx
    def rx3(self,target_qubits=None, control_qubits=None, inverse=False):
            qc=QuantumCircuit(1)
            qc.rx((3/8)*2*np.pi,0)
            rx=qc.to_gate()
            rx.label="RX3"
            return rx
    def rx4(self,target_qubits=None, control_qubits=None, inverse=False):
            qc=QuantumCircuit(1)
            qc.rx((4/8)*2*np.pi,0)
            rx=qc.to_gate()
            rx.label="RX4"
            return rx
    def rx5(self,target_qubits=None, control_qubits=None, inverse=False):
            qc=QuantumCircuit(1)
            qc.rx((5/8)*2*np.pi,0)
            rx=qc.to_gate()
            rx.label="RX5"
            return rx
    def rx6(self,target_qubits=None, control_qubits=None, inverse=False):
            qc=QuantumCircuit(1)
            qc.rx((6/8)*2*np.pi,0)
            rx=qc.to_gate()
            rx.label="RX6"
            return rx
    def rx7(self,target_qubits=None, control_qubits=None, inverse=False):
            qc=QuantumCircuit(1)
            qc.rx((7/8)*2*np.pi,0)
            rx=qc.to_gate()
            rx.label="RX7"
            return rx
    def ry1(self,target_qubits=None, control_qubits=None, inverse=False):
            qc=QuantumCircuit(1)
            qc.ry((1/8)*2*np.pi,0)
            ry=qc.to_gate()
            ry.label="RY1"
            return ry
    def ry2(self,target_qubits=None, control_qubits=None, inverse=False):
            qc=QuantumCircuit(1)
            qc.ry((2/8)*2*np.pi,0)
            ry=qc.to_gate()
            ry.label="RY2"
            return ry
    def ry3(self,target_qubits=None, control_qubits=None, inverse=False):
            qc=QuantumCircuit(1)
            qc.ry((3/8)*2*np.pi,0)
            ry=qc.to_gate()
            ry.label="RY3"
            return ry
    def ry4(self,target_qubits=None, control_qubits=None, inverse=False):
            qc=QuantumCircuit(1)
            qc.ry((4/8)*2*np.pi,0)
            ry=qc.to_gate()
            ry.label="RY4"
            return ry
    def ry5(self,target_qubits=None, control_qubits=None, inverse=False):
            qc=QuantumCircuit(1)
            qc.ry((5/8)*2*np.pi,0)
            ry=qc.to_gate()
            ry.label="RY5"
            return ry
    def ry6(self,target_qubits=None, control_qubits=None, inverse=False):
            qc=QuantumCircuit(1)
            qc.ry((6/8)*2*np.pi,0)
            ry=qc.to_gate()
            ry.label="RY6"
            return ry
    def ry7(self,target_qubits=None, control_qubits=None, inverse=False):
            qc=QuantumCircuit(1)
            qc.ry((7/8)*2*np.pi,0)
            ry=qc.to_gate()
            ry.label="RY7"
            return ry