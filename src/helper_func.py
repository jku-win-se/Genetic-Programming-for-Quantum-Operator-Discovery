# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 09:28:56 2022

@author: fege9
"""
from deap import creator, base, tools, algorithms
import numpy as np
from numpy.random import choice, random as rand, randint
#import import_ipynb
#import Gates
import cmath
import scipy.special as scisp
#import pyzx as zx
import time
import logging
import logging
from typing import Optional

from qiskit import quantum_info
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.circuit import Gate
from qiskit import Aer,transpile
from qiskit.compiler import transpile
from qiskit.transpiler import PassManager, passes
from qiskit.transpiler.passes import CommutativeCancellation, Optimize1qGatesDecomposition
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize
from qiskit.circuit import ParameterVector, Parameter
from qiskit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag


# target = inp.target
# target_pos=inp.target_pos
# gateset=inp.gateset
# max_gates = inp.max_gates
# N=inp.N
# NGEN = inp.NGEN
# CXPB = inp.CXPB
# MUTPB = inp.MUTPB
# weights_gates=inp.weights_gates
# prob=inp.prob
# numerical_optimizer=inp.numerical_optimizer
# weights_mut=inp.weights_mut
# weights_cx=inp.weights_cx
# opt_within =inp.opt_within
# opt_final = inp.opt_final
# opt_select=inp.opt_select
# sel_scheme=inp.sel_scheme
# weights2 = inp.weights2
#
# target_qubits = inp.target_qubits
# overall_QC=inp.overall_QC
# oracle_index=inp.oracle_index
#
#
# vals=inp.vals
# par_nums=inp.par_nums
# names=inp.names
# e_gate=inp.e_gate
# overall_qubits=inp.overall_qubits
# qubits=inp.qubits
# weights =inp.weights
# M=inp.M
# operand=inp.operand
# operator=inp.operator
# basis=inp.basis


def deap_init(settings, circ):
    #create classes and toolbox initialization
    creator.create("FitnessMulti", base.Fitness, weights=settings["weights"])
    creator.create("Individual", list, fitness=creator.FitnessMulti, max_gates=None, operand=None, operator=None)
    toolbox = base.Toolbox()
    toolbox.register("circuit", singleInd, settings)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.circuit)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mutate", mutator, settings, circ)
    toolbox.register("mate", crossover, weights_cx=settings["weights_cx"])
    toolbox.register("evaluate", fitness_function, settings, circ)
    
    toolbox.decorate("mate", checkLength(settings))
    toolbox.decorate("mutate", checkLength(settings))
    
    P = get_P(settings["M"], settings["N"])
    ref_points = tools.uniform_reference_points(settings["M"], P)  #parameters: number of objective, P as defined in NSGA3-paper
    toolbox.register("select", tools.selNSGA3, ref_points=ref_points)
    return toolbox
    

def tupleConstruct(operand, operator, weights_gates, settings):
    '''
    This function generates the tuple (gene) of the genetic programming
    chromosomes. Which are later appended in a list to form an individual
    '''
    #Select an operator i.e. quantum gate
    optr=choice(operator, p=weights_gates)
    val=settings["vals"].get(optr)
    params=settings["par_nums"].get(optr)
    if params==1:
        p=list(rand(1))
    else:
        p=list(rand(params))

    if val==2:
        if params==0:
            tpl=(optr, choice(operand,2,False))
        else:
            tpl=(optr, choice(operand,2,False),p)

    elif val==1:
        if params==0:
            tpl=(optr,[choice(operand)])
        else:
            tpl=(optr,[choice(operand)], p)

    return tpl

def singleInd(settings):
    '''
    This function generates the single individual of genetic program
    '''
    max_gates = settings["max_gates"]
    prob = settings["prob"]

    if prob is None:
        N_CYCLES = randint(2,max_gates+1)
    #N_CYCLES = randint(2, max_gates+1)
    else:
        N_CYCLES=round(seeded_rng().normal(prob[0],prob[1]))
        if N_CYCLES <= 2:
            N_CYCLES=2
        if N_CYCLES >= max_gates:
            N_CYCLES=max_gates

    l = []
    for i in range(N_CYCLES):
            tpl = tupleConstruct(settings["operand"], settings["operator"], settings["weights_gates"], settings)
            l.append(tpl)
    return l


def remove_duplicates(pop):
    def not_within_list(res, i):
        li=len(i)
        matches=[]
        match=True
        for j in res:
            if li!=len(j):
                match=False
                matches.append(match)
                continue
            elif li==len(j):
                match=True
                #(i==j).all()
                for k in range(li):
                    if len(i[k])!=len(j[k]):
                        match=False
                        matches.append(match)
                        break
                    else:
                        if (i[k][0]!=j[k][0] or (np.array(i[k][1])!=np.array(j[k][1])).any()):
                            match=False
                            matches.append(match)
                            break
                matches.append(match)
        return(np.array(matches).any())
    
    res=[]
    for i in pop:
        if not not_within_list(res, i):
            res.append(i)
        
    return res


def is_similar(ind1, ind2):
    l1=len(ind1)
    l2=len(ind2)
    match=True
    if l1!=l2:
        match=False
        return match
    elif l1==l2:
        match=True
        for k in range(l1):
            if len(ind1[k])!=len(ind2[k]):
                match=False
                break
            else:
                if (ind1[k][0]!=ind2[k][0] or (np.array(ind1[k][1])!=np.array(ind2[k][1])).any()):
                    match=False
                    break
    return match



def var_form(settings, circ, ind, parameters):
    qc = circ.overall_QC
    for i in range(len(circ.oracle_index)):
        qc.append(ind_to_gate(settings, ind, parameters)[1], circ.target_qubits_oracles[i])
        qc.data[circ.oracle_index[i]]=qc.data[-1]
        del qc.data[-1]
    
    position=[]
    for i in range(len(qc.data)):
        if "barrier" in str(qc.data[i]):
                position.append(i)
    qc2=qc.copy()
    for j in range(len(qc2.data)-position[settings["target_pos"]-1]-1):
        del qc2.data[-1]
    out_state = Statevector.from_instruction(qc2)
    #out_state = Statevector.from_instruction(qc)
    
    return out_state, qc


def get_difference(settings, ind):

    def execute_circ(theta, settings, circ, target=None):
        target = target or settings["target"]
        state = var_form(settings, circ, ind, theta)[0]
        target = np.asarray(target)
        s = np.abs(np.vdot(state.data, target))  #taking real part ensures phase correct comparison
        diff=1-s
        return diff
        
    
    return execute_circ


def ind_to_gate(settings, ind, parameters, opt=None, qubits=None):
    opt = opt or settings["opt_within"]
    qubits = qubits or settings["qubits"]

    qc=QuantumCircuit(qubits)
    k=0
    for i in ind:
        name=settings["names"].get(i[0])
        if name in settings["Non_func_gates"]:
            if len(i)==2:
                getattr(qc,name)(*i[1])

            elif len(i)==3:
                getattr(qc,name)(*parameters[k:k+len(i[2])],*i[1]) #*i[2] is parameters[k:k+len(i[2])]
                k+=len(i[2])
        else:
            if len(i)==2:
                qc.append(getattr(settings["e_gate"],name)(),i[1])
            else:
                print("Parameters inside! ERROR!!")
           
    #simple circuit optimization
    new_qc=circuit_optimization(qc, settings["opt_within"])
    
    gate=new_qc.to_instruction()
    return new_qc, gate

#define fitness function

def quantum_state(settings, circ, ind, qubits=None, method=None, opt_within=None):
    qubits = qubits or settings["qubits"]
    method = method or settings["numerical_optimizer"]
    opt_within = opt_within or settings["opt_within"]
    var=0
    for i in ind:
        if len(i)==3:
            var=1
    #numerical parameter optimization 
    if var==1:
        init_parameters = []
        for i in ind:
            if len(i)==3:
                for j in range(len(i[2])):
                    init_parameters.append(i[2][j])
        if settings["use_numerical_optimizer"]=="yes":
            difference = get_difference(settings, ind)
            res = minimize(difference, 
                          init_parameters,
                          args=(settings,circ),
                          method=method)
            final_param = list(res.x)
        elif settings["use_numerical_optimizer"]=="no":
            final_param=init_parameters
    else:
        final_param=None
       
      
    #create QC
    qc = circ.overall_QC
    oracle=ind_to_gate(settings, ind, final_param)
    pos=len(qc.data)
    for i in range(len(circ.oracle_index)):
        qc.append(oracle[1], circ.target_qubits_oracles[i])
        qc.data[circ.oracle_index[i]]=qc.data[pos]
        del qc.data[pos]
    
    position=[]
    for i in range(len(qc.data)):
        if "barrier" in str(qc.data[i]):
                position.append(i)
    qc2=qc.copy()
    for j in range(len(qc2.data)-position[settings["target_pos"]-1]-1):
        del qc2.data[len(qc2.data)-1]
    
    state = Statevector.from_instruction(qc2)
    #state = Statevector.from_instruction(qc)       
    
    #qc.data.pop(0) #get rid of initialization of circuit
    
    return state, oracle, final_param, qc


#fitness function
def fitness_function(settings, circ, ind, target=None):
    target = target or settings["target"]
    
    out_state,oracle_qc,params,overall_qc = quantum_state(settings, circ, ind)
    
    #overlap
    target = np.asarray(target)
    s = round(np.abs(np.vdot(out_state.data, target)),10) #real part instead of abs?
    
    #number of gates
    num_gates=len(oracle_qc[0].data)
    for i in ind:
        if i[0]=='Identity':
            num_gates-=1
    
    #depth
    d=oracle_qc[0].depth()
    for i in ind:
        if i[0]=='Identity':
            d-=1
    """d=0
    for q in range(qubits):
        dt=0
        for i in ind:
            if (q in i[1] and i[0]!="Identity"):
                dt+=1
        d=max(d,dt)"""
        
    #number of non-local gates
    nl=oracle_qc[0].num_nonlocal_gates()
    """nl=0
    for i in ind:
        val=Gates.gateArity.get(i[0])
        if val!=1:
            nl+=1"""
            
    #number of parameters
    if params==None:
        p=0
    else:
        p=len(params)
    
    return s,num_gates,d,nl,p



def crossover(circ1, circ2, weights_cx):
    
    cx=[0,1,2,3]
    cxtype=choice(cx, p=weights_cx)
    l1=len(circ1)
    l2=len(circ2)
    
    while(1):
        if (cxtype==2 or cxtype==3) and (l1<=2 or l2<=2):
            cxtype=choice(cx, p=weights_cx)
        else:
            break
    
    #One point crossover where length of individuals may change
    if cxtype==0:
        cxpoint1 = randint(1,l1)
        cxpoint2 = randint(1,l2)
        circ1[cxpoint1:], circ2[cxpoint2:] = circ2[cxpoint2:], circ1[cxpoint1:]
        
    #One point crossover where length of individuals remains same
    elif cxtype==1:
        size = min(l1,l2)
        cxpoint = randint(1,size)
        circ1[cxpoint:], circ2[cxpoint:] = circ2[cxpoint:], circ1[cxpoint:]
    
    #two point crossover where length of individuals remains same
    elif cxtype==2:
        size = min(l1,l2)
        cxpoint1 = randint(1, size)
        cxpoint2 = randint(1, size - 1)
        if cxpoint2 >= cxpoint1:
            cxpoint2 += 1
        else:
            cxpoint1, cxpoint2 = cxpoint2, cxpoint1

        circ1[cxpoint1:cxpoint2], circ2[cxpoint1:cxpoint2] = circ2[cxpoint1:cxpoint2], circ1[cxpoint1:cxpoint2]
        
    #two point crossover where of individuals may change
    elif cxtype==3:
        cxp11 = randint(1, l1)
        cxp12 = randint(1, l1-1)
        if cxp12 >= cxp11:
            cxp12 += 1
        else:
            cxp11, cxp12 = cxp12, cxp11
        
        cxp21 = randint(1, l2)
        cxp22 = randint(1, l2-1)
        if cxp22 >= cxp21:
            cxp22 += 1
        else:
            cxp21, cxp22 = cxp22, cxp21
        
        temp1 = circ1[cxp11:cxp12]
        for n in range(cxp12-cxp11):
            del circ1[cxp11]
        for n in range(cxp22-cxp21):
            circ1.insert(cxp11+n, circ2[cxp21+n])
        
        for n in range(cxp22-cxp21):
            del circ2[cxp21]
        for n in range(cxp12-cxp11):
            circ2.insert(cxp21+n, temp1[n])
            
    return circ1, circ2


def mutator(settings, circ, ind):
    operand = settings["operand"]
    operator = settings["operator"]
    max_gates = settings["max_gates"]
    weights_gates = settings["weights_gates"]
    weights_mut = settings["weights_mut"]

    l = len(ind)
    mutators=[0,1,2,3,4,5,6,7,8]
    if settings["use_numerical_optimizer"]=="no":
        mutators=[0,1,2,3,4,5,6,7,8,9]
    muttype = choice(mutators, p=weights_mut)
    
    while(1):
        if muttype==6 and l<=2:
            muttype = choice(mutators, p=weights_mut)
        elif muttype==7 and l<=4:
            muttype = choice(mutators, p=weights_mut)
        elif muttype==8 and l<=2:
            muttype = choice(mutators, p=weights_mut)
        else:
            break
    
    #insert
    if muttype==0:
        i = randint(0,l)
        tpl=tupleConstruct(operand, operator,weights_gates, settings)
        ind.insert(i, tpl)

    #delete
    elif muttype==1:
        i = randint(0,l)
        del ind[i]

    #swap
    elif muttype==2:
        i,j = choice(range(l), 2, False)
        ind[i], ind[j]= ind[j], ind[i]

    #change whole gate
    elif muttype==3:
        i = randint(0,l)
        ind[i] = tupleConstruct(operand, operator, weights_gates, settings)
    
   
   #change only target/control-qubits as in Multi-objective (2018) paper
    elif muttype==4:
        i = randint(0,l)
        temp=list(ind[i])
        if len(temp[1])==1:
            qb=[choice(operand)]
        else:
            qb=choice(operand, len(ind[i][1]), False)
        temp[1]=qb
        ind[i]=tuple(temp)
    
    #move gate to different position in circuit
    elif muttype==5:
        i = randint(0,l)
        a = ind[i]
        del ind[i]
        j = randint(0,l-1)
        ind.insert(j, a)
        
    #replace sequence with random sequence of different size
    elif muttype==6 and l>2:
        m,n = choice(range(l), 2, False)
        i = min(m,n)
        j = max(m,n)

        for k in range(j-i+1):
            del ind[i]

        max_len = max_gates - len(ind)
        for n in range(randint(max_len+1)):
            tpl = tupleConstruct(operand, operator,weights_gates, settings)
            ind.insert(i + n, tpl)
    
    #swap two random sequences in chromosome
    elif muttype==7 and l>4:
        i,j,k,l1 = choice(range(l),4,False)
        lis = list([i,j,k,l1])
        lis.sort()
        a,b,c,d = lis[0], lis[1], lis[2], lis[3]
        
        temp1 = ind[a:b + 1]
        temp2 = ind[c:d + 1]
        for n in range(d+1-c):
            del ind[c]
        for n in range(d+1-c):
            ind.insert(b + 1 + n, temp2[n])
        for n in range(b+1-a):
            ind.insert(c + b + 1 - a + n, temp1[n])
        for n in range(a,b+1):
            del ind[a]
    
    #random permutation of gates within a sequence
    elif muttype==8 and l>2:
        m,n = choice(range(l), 2, False)
        i=min(m,n)
        j=max(m,n)
        
        temp= ind[i:j + 1]
        seeded_rng().shuffle(temp)
        temp=list(temp)
        
        for k in range(i,j+1):
            del ind[i]
        for n in range(j+1-i):
            ind.insert(i + n, temp[n])
            
    #use the inverted of the gate, gate has to be added in "Gates.ipybn"
    #elif muttype==9:
     #   i = randint(0,l)
      #  circ[i][0]=circ[i][0]+'_inverted'
      
    #additional mutator for comparison with [42]
    elif muttype==9:
        i = randint(0,l)
        temp=list(ind[i])
        if len(temp)==3:
            len_param=len(temp[2])
            p = seeded_rng().uniform(0,np.pi,len_param)
            temp[2]=p
        ind[i]=tuple(temp)
    
    return ind,


def checkLength(settings):
    min = 2
    max=settings["max_gates"]
    def decorator(func):
        def wrapper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
                    if(len(child) < min):
                        for n in range(min-len(child)):
                            gate = tupleConstruct(settings["operand"], settings["operator"], settings["weights_gates"], settings)
                            child.insert(len(child)+n, gate)
                    if(len(child)>max):
                        for n in range(len(child) - settings["max_gates"]):
                            del child[0]
            return offspring
        return wrapper
    return decorator



def nsga3(toolbox, settings, seed=None):
    #rand.seed(seed)

    # Initialize statistics object
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    pop = toolbox.population(n=settings["N"])
    pareto=tools.ParetoFront(similar=is_similar)
    
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    
    pareto.update(pop)
    
    # Compile statistics about the population
    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    print(logbook.stream)

    # Begin the generational process
    time_stamps = []
    start_time = time.time()
    fitness_values_generations = []

    for gen in range(1, settings["NGEN"]):
        cxpb=settings["CXPB"] or 1.0
        mutpb=settings["MUTPB"] or 1.0
        offspring = algorithms.varAnd(pop, toolbox, cxpb, mutpb)

        currenttime = time.time() - start_time
        time_stamps.append(currenttime)
        if (settings.get("time_limit") is not None) and (currenttime > settings["time_limit"]):
                break

        #remove duplicates and fill with new individuals
        offspring = remove_duplicates(offspring)
        x = settings["N"] - len(offspring)
        for i in range(x):
            ind_i = toolbox.individual()
            ind_i.fitness.values = toolbox.evaluate(ind_i)
            offspring.append(ind_i)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
    
        pareto.update(offspring)

        fitness_values_curr_gen = []
        for i in range(len(pareto)):
            fitness_values_curr_gen.append(pareto[i].fitness.values)
        fitness_values_generations.append(fitness_values_curr_gen)

        # Select the next generation population from parents and offspring
        pop = toolbox.select(pop + offspring, settings["N"])
        
        # Compile statistics about the new population
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)

    return pop, pareto, logbook, time_stamps, fitness_values_generations


def circuit_optimization(qc, opt):
    if opt is not None:
        new_qc=0
        count=0
        for i in opt:
            if count>0:
                qc=new_qc
            count+=1
            if i=='CC':
                pass_ = CommutativeCancellation()
                pm = PassManager(pass_)
                new_qc = pm.run(qc)
            elif i=='OQ1':
                pass_ = Optimize1qGatesDecomposition()
                pm = PassManager(pass_)
                new_qc = pm.run(qc)
            elif i=="Q1":
                new_qc = transpile(qc, optimization_level=1)
            elif i=="Q2":
                new_qc = transpile(qc, optimization_level=2)
            #elif i=="ZX":
             #   new_qc = zx_optimize(qc)
    if opt is None:
        new_qc=qc
    return new_qc


"""def zx_optimize(circ):

    basis=['x','z','rz','u1','u2','h','cx','cz','ccx','ccz']
    new_circ=transpile(circ, basis_gates=basis, optimization_level=0)

    circ2=new_circ.qasm()
    #print(circ2)
    circ3=zx.qasm(circ2)
    circ4=zx.basic_optimization(circ3)
    circ5=circ4.to_qasm()
    #print(circ5)
    qc=QuantumCircuit.from_qasm_str(circ5)

    return qc
"""

def fitness_from_qc(qc2, settings, circ, target=None):
    """
    #based on initial state:
    qc1=QuantumCircuit(qubits)
    qc1.initialize(init, qc1.qubits)
    qc=qc1+qc2
    out_state = Statevector.from_instruction(qc)"""
    target = target or settings["target"]

    qc2_gate=qc2.to_gate()
    qc = circ.overall_QC
    pos=len(qc.data)
    for i in range(len(circ.oracle_index)):
        qc.append(qc2_gate, circ.target_qubits_oracles[i])
        qc.data[circ.oracle_index[i]]=qc.data[pos]
        del qc.data[pos]
    
    position=[]
    for i in range(len(qc.data)):
        if "barrier" in str(qc.data[i]):
                position.append(i)
    qc3=qc.copy()
    for j in range(len(qc3.data)-position[settings["target_pos"]-1]-1):
        del qc3.data[len(qc3.data)-1]
    
    out_state = Statevector.from_instruction(qc3)
    #out_state = Statevector.from_instruction(qc)    
    
    #overlap
    target = np.asarray(target)
    s = round(np.abs(np.vdot(out_state.data, target)),10) #real part ensures phase correct comparison

    
    #number of gates
    dag_circuit = circuit_to_dag(qc2)
    num_gates=dag_circuit.size()
    
    #depth
    d=qc2.depth()
    
    #number of non-local gates
    nl=qc2.num_nonlocal_gates()
    
    #number of parameters given in get_best_circ
    
    return s,num_gates,d,nl


def get_P(M, N):
    dis1=N-1
    for i in range(1,N):
        a=scisp.binom(i+M-1,i)
        dis2=abs(N-a)
        if dis2<dis1:
            dis1=dis2
        else:
            break

    return (i-1)


#Möglich: o:overlap, g:number of gates, d: depth, nl: number of nl-gates, p: number of params
def reduce_pareto(pareto, settings):
    reduce=settings["reduce_pareto"]
    for element in reduce:
        s=element.strip().split()
        if s[0]=="o":
            f_val=0
        elif s[0]=="g":
            f_val=1
        elif s[0]=="d":
            f_val=2
        elif s[0]=="nl":
            f_val=3
        elif s[0]=="p":
            f_val=4
        else:
            print("Warning: Fitness values must be one of 'o','g', 'd', 'nl', 'p'!")
        ineq=s[1]
        constr=float(s[2])
        
        for i in reversed(range(len(pareto))):
            if ineq == "<":    
                if pareto[i].fitness.values[f_val] > constr:
                    pareto.remove(i)
            elif ineq == ">":
                if pareto[i].fitness.values[f_val] < constr:
                    pareto.remove(i)
            else:
                print("Warning: inequality operator must be either '<' or '>'!")
    return pareto

def get_pareto_final_qc(settings, circ, popu):
    
    """#delete identity gates
    for i in popu:
        l=len(i)-1
        j=0
        while j<=l:
            if i[j][0]=='Identity':
                i.pop(j)
                l=len(i)-1
                continue
            j+=1"""
    
    if settings["sel_scheme"] == 'Manual':
        pareto_front=[]
        for indi in popu:
            out_state,oracle,params,_=quantum_state(settings, circ, indi)
            qc=oracle[0]
            if settings["opt_select"] is not None:
                qc=circuit_optimization(qc, settings["opt_select"])
            fitness=list(fitness_from_qc(qc, settings, circ))
            if params==None:
                p=0
            else:
                p=len(params)
            fitness.append(p)
            ind=[qc, fitness, out_state]
            pareto_front.append(ind)
            
        return pareto_front
    
    if settings["sel_scheme"]==None or settings["sel_scheme"]=="Sorted":
        if settings["opt_select"] is None:
            order=settings["Sorted_order"]
            np.array(order)
            order_sign=list(np.sign(np.array(order)))
            order=list(np.abs(order))
            liste=sorted(range(len(popu)), key=lambda i: 
              (order_sign[0]*popu[i].fitness.values[order[0]-1],order_sign[1]*popu[i].fitness.values[order[1]-1],order_sign[2]*popu[i].fitness.values[order[2]-1],order_sign[3]*popu[i].fitness.values[order[3]-1],order_sign[4]*popu[i].fitness.values[order[4]-1]))
            
            out_state, qc_gate, params,_ = quantum_state(settings, circ, popu[liste[0]])
            qc1=qc_gate[0]
            if params==None:
                p1=0
            else:
                p1=len(params)
            
        else:
            pareto_front=[]
            f1=[]
            f2=[]
            f3=[]
            f4=[]
            f5=[]
            for indi in popu:
                state, oracle, params, _ = quantum_state(settings, circ, indi)
                qctmp = oracle[0]
                qc=circuit_optimization(qctmp, settings["opt_select"])
                fitness=list(fitness_from_qc(qc, settings, circ))
                if params==None:
                    p=0
                else:
                    p=len(params)
                fitness.append(p)
                ind=[qc, fitness, state]
                pareto_front.append(ind)
                f1.append(fitness[0])
                f2.append(fitness[1])
                f3.append(fitness[2])
                f4.append(fitness[3])
                f5.append(fitness[4])
            fitness_vals=[]
            fitness_vals.append(f1)
            fitness_vals.append(f2)
            fitness_vals.append(f3)
            fitness_vals.append(f4)
            fitness_vals.append(f5)
            order=settings["Sorted_order"]
            np.array(order)
            #print("Order: ", order)
            order_sign=list(np.sign(np.array(order)))
            #order=list(np.abs(order))
            print(order_sign)
            liste=sorted(range(len(pareto_front)), key=lambda i: 
                (order_sign[0]*fitness_vals[order[0]-1][i],order_sign[1]*fitness_vals[order[1]-1][i],order_sign[2]*fitness_vals[order[2]-1][i],order_sign[3]*fitness_vals[order[3]-1][i],order_sign[4]*fitness_vals[order[4]-1][i]))
            print("Liste: ",liste)
            out_state=pareto_front[liste[0]][2]
            qc1=pareto_front[liste[0]][0]
            p1=pareto_front[liste[0]][1][4]
            
    if settings["sel_scheme"]=='Weighted':
        if(settings["opt_select"]==None):
            m1=-1000
            for i in popu:
                m2 = np.dot(np.array(settings["weights2"]), np.array(i.fitness.values))
                if m2>m1:
                    m1=m2
                    ind=i
                    p1=i.fitness.values[4]

            #f=np.array(ind.fitness.values)
            out_state, qc_gate, params,_ = quantum_state(settings, circ, ind)
            qc1=qc_gate[0]
        else:
            m1=-1000
            for indi in popu:
                state, qc_gate, params,_ = quantum_state(settings, circ, indi)
                qc=qc_gate[0]

                new_qc=circuit_optimization(qc,settings["opt_select"])
                fitness=list(fitness_from_qc(new_qc, settings, circ))
                if params is None:
                    p=0
                else:
                    p=len(params)
                fitness.append(p)
                m2 = np.dot(np.array(settings["weights2"]), np.array(fitness))
                if m2>m1:
                    m1=m2
                    qc1=new_qc
                    ind=indi
                    p1=p
                    out_state=state
            #f=list(fitness_from_qc(qc1))
            #f.append(p1)

    #optimization of final QC
    final_qc=circuit_optimization(qc1, settings["opt_final"])
    f=list(fitness_from_qc(final_qc, settings, circ))
    f.append(p1)
    
    return final_qc, f, out_state


GLOBAL_RNG = None
def seeded_rng(seed: int = None) -> np.random.Generator:
    """
    Once the seed is set, it is permanent, until you use `reset_rng` to reset.
    Thus, to set it, manually call it before any other function accesses.

    Args:
        seed (int): Which seed to use. If not, will use random seed.

    Returns:
        (np.random.Generator): The random number generator
    """
    global GLOBAL_RNG
    if GLOBAL_RNG is None:
        #logger.info(f"Setting random seed to {seed}")
        GLOBAL_RNG = np.random.default_rng(seed)
    return GLOBAL_RNG