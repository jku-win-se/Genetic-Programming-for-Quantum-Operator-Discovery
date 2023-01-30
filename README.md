# Genetic-Programming-for-Quantum-Operator-Discovery
## Overall Description:
This repository comprises a multi-objective genetic programming approach for automated quantum operator synthesis of parameterized as well as non-parameterized quantum circuits. 
The tool is based on quantum state information and, therefore, relies on quantum simulation. The genetic algorithm is hybridized with a numerical parameter optimizer in order to determine the fitness of parameterized quantum operators, and further allows the application of quantum circuit optimization procedures at various phases of the search process.
Therefore, it serves quantum algorithm developers to explore patterns for quantum operators on the small scale. The user can choose from a set of possible solutions, which allows to consider besides the quality of the found operator also its associated computing costs. The latter is particularly important in the current NISQ-era of quantum computing.
The routine takes as input a Quantum Circuit specified by the Qiskit SDK, and a settings-file in JSON-format, which specifies the parameters of the search. The output are the discovered quantum operator in OpenQASM 2.0 format as well as an additional output file, which comprises data on the search process as well as the Pareto-front.

## Folder Description
The /Evaluation comprises the computational results for the two use cases. The generated output files are provided only for the computationally more expensive GM-QAOA use case, whereas for the Grover use case the found alternatives to the analytical solution are provided.

The /src comprises the static files of the genetic programming approach.

The /examples include two explore use-cases: the Grover search algorithm, as well as the GM-QAOA algorithm.

The execution is done in the console via:

python src/main.py examples/.../QC.py examples/.../settings.json

where ... means "Grover" or "QM-QAOA", respectively.

## Requirements
Python Qiskit SDK
Python Deap Library

## Specification of settings.json (description, datatype, default value, constraints)
**target:** target quantum state, list of complex numbers, no default

**target_pos:** position of target quantum states, int, no default

**filename:** Name of File where results of the search are stored to, str, no default

**max_gates:** maximum length of an individual, int, no default, >2

**N:** population size, int, default=20

**NGEN:** number of generations, int, default=10

**CXPB:** probability for applying a crossover operation, float, default=1.0., 0<x<=1.0

**MUTPB:** probability for applying a mutation operation, float, default=1.0, 0<x<=1.0

**weights_cx:** probability for individual crossover operations, list of floats, default=None, lenght of list = 4

**weights_mut:** probability for individual mutation operations, list of floats, default=None, lenght of list = 9

**weights_gates:** probability for sampling individual elementary gates, list of floats, default=None, length of list = #gates in gateset

**prob:** mean and standarddeviation of Gaussian-distribution for individual length sampling, default=None

**numerical_optimizer:** classical parameter optimizer, str, default="Nelder-Mead", choose from {Nelder-Mead, COBYLA, BFGS, Powell, CG, Newton-CG, L- BFGS-B, TNC, SLSQP, trustconstr, dogleg, trust-ncg, trust-exact, trust-krylov}

**opt_within:** quantum circuit optimization procedures applied within fitness evaluation, list of str, default=None, choose from {CC, OQ1, Q1, Q2} where CC: CommutativeCancelling, OQ1:Optimize1qGatesDecomposition, Q1:Qiskit Level1 Optimization, Q2:Qiskit Level2 Optimization

**opt_select:** quantum circuit optimization procedures applied for selection of quantum operation from Pareto-front, list of str, default=None, choose from {CC, OQ1, Q1, Q2} where CC: CommutativeCancelling, OQ1:Optimize1qGatesDecomposition, Q1:Qiskit Level1 Optimization, Q2:Qiskit Level2 Optimization

**opt_final:** quantum circuit optimization procedures applied for final quantum operator, list of str, default=None, choose from {CC, OQ1, Q1, Q2} where CC: CommutativeCancelling, OQ1:Optimize1qGatesDecomposition, Q1:Qiskit Level1 Optimization, Q2:Qiskit Level2 Optimization

**sel_scheme:** method to choose final quantum operator from Pareto-front, str, default=None, choose from {Sorted, Weighted, Manual}

**weights2:** weights for selecting final quantum operator when using sel_scheme="Weighted", list of floats, default=None, length of list must match number of fitness values (i.e., 5)

**gateset:** the set of elementary gates which are used as genes, str, default="variable", choose from {variable, fixed}

## Specification of QC.py
The file containing the overall quantum circuit has to comprise 

i) the overall quantum circuit as a QuantumCircuit-Object as defined in the Qiskit SDK;

ii) target_qubits_oracle: a list of lists of integers defining the qubits to which the oracle is applied;

iii) oracle_index: a list of integers defining the positions of the oracle in the quantum circuit
