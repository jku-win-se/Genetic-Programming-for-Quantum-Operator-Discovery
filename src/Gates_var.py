# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 09:25:28 2022

@author: fege9
"""

# elementary Quantum Gates

Non_func_gates = ["x", "y", "z", "cx", "cy", "cz", "swap", "h", "rz", "rx", "ry", "cu", "rzz", "ryy", "rxx"]

H = "Hadamard"
X = "Not"
Y = "Pauli Y"
Z = "Pauli Z"
CX = "Controlled Not"
CY = "Controlled Y"
CZ = "Controlled Z"
RZ = "Rotate Z"
RX = "Rotate X"
RY = "Rotate Y"
CU = "Controlled U"
RZZ = "Rotate ZZ"
RXX = "Rotate XX"
RYY = "Rotate YY"
SWAP = "Swap"


gateArity = {
    H: 1,
    X: 1,
    Y: 1,
    Z: 1,
    CX: 2,
    CY: 2,
    CZ: 2,
    SWAP: 2,
    RZ: 1,
    RX: 1,
    RY: 1,
    CU: 2,
    RZZ: 2,
    RXX: 2,
    RYY: 2,
}

gateName = {
    H: "h",
    X: "x",
    Y: "y",
    Z: "z",
    CX: "cx",
    CY: "cy",
    CZ: "cz",
    SWAP: "swap",
    RZ: "rz",
    RX: "rx",
    RY: "ry",
    CU: "cu",
    RZZ: "rzz",
    RXX: "rxx",
    RYY: "ryy",
}

# give number of parameters for each gate
gateParams = {H: 0, X: 0, Y: 0, Z: 0, CX: 0, CY: 0, CZ: 0, SWAP: 0, RZ: 1, RX: 1, RY: 1, CU: 4, RZZ: 1, RXX: 1, RYY: 1}


class ElementaryGate:
    def __init__(self):
        self.type = "ElementaryGate"
