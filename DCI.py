import pandas as pd
import numpy as np
import itertools
try:
    # try importing the C version
    from deap.tools._hypervolume import hv
except ImportError:
    # fallback on python version
    from deap.tools._hypervolume import pyhv as hv

#functions: read fitness from CSV, Reduce PF to relevant region, Quality indicators: DCI, Hypervolume

#read fitness values of each generation for specified run and approach
#ToDo: make relative path as in main.py
def read_fitness(seed, approach, filename):
    file = f"Results_{approach}_appr/Run_{seed}_{filename}.csv"
    df = pd.read_csv(file)

    fitness_vals = []

    for gen in range(1,10):
        fitness_gen = []
        for i in range(len(df.index)):
            if(df.loc[i]["ngen"] == gen):
                fitness = [df.loc[i]["overlap"], df.loc[i]["num_gates"], df.loc[i]["depth"], df.loc[i]["num_nonloc_gates"],
                            df.loc[i]["num_parameters"]]
                fitness_gen.append(fitness)
        fitness_vals.append(fitness_gen)
    return fitness_vals


def reduce_pareto(pareto, reduce): #reduce: e.g., ["o > 0.9"]
    for element in reduce:
        s = element.strip().split()
        if s[0] == "o":
            f_val = 0
        elif s[0] == "g":
            f_val = 1
        elif s[0] == "d":
            f_val = 2
        elif s[0] == "nl":
            f_val = 3
        elif s[0] == "p":
            f_val = 4
        else:
            print("Warning: Fitness values must be one of 'o','g', 'd', 'nl', 'p'!")
        ineq = s[1]
        constr = float(s[2])

        for i in reversed(range(len(pareto))):
            if ineq == "<":
                if pareto[i][f_val] > constr:
                    del pareto[i]
            elif ineq == ">":
                if pareto[i][f_val] < constr:
                    del pareto[i]
            else:
                print("Warning: inequality operator must be either '<' or '>'!")
    return pareto


#calculate DCI according to Li & Young: Diversity Comparison of Pareto Front Approximations in Many-Objective Optimization (2014)
def DCI(pareto, div, low, up):

    d = [(up[i]-low[i])/div for i in range(5)] #hyperbox sizes

    #assign fitness values to grid coordinates
    f_gen_grid = []
    for ind in pareto:
        fitness = [int((ind[i]-low[i])/d[i]) for i in range(5)]
        f_gen_grid.append(fitness)
    gen = f_gen_grid

    # calculate DCI
    DCI = sum([contr(gen,np.array([o,g,d,nl,p])) for o,g,d,nl,p in
               itertools.product(range(div),range(div),range(div),range(div),range(div))])
    DCI *= 1. / (div ** 5)
    return DCI

#distance between Pareto-front and hyperbox
def distance(gen, h):
    dis = [np.linalg.norm(np.array(ind)-np.array(h)) for ind in gen]
    res = np.min(np.array(dis))
    return res

#contribution degree
def contr(gen, h):
    d = distance(gen,h)
    if d < np.sqrt(5+1):
        c = 1 - d**2/(5+1)
    else:
        c = 0
    return c

#hypervolume
def hypervolume(front, ref=None):
    # Must use wvalues * -1 since hypervolume use implicit minimization
    for ind in front:
        ind[0]*=-1
    wobj = np.array(front)
    if ref is None:
        ref = np.max(wobj, axis=0) + 1
    return hv.hypervolume(wobj, ref)

seed = 77
approach = "comp"
filename = "Grover_Oracle"
f = read_fitness(seed, approach, filename)
#print(f[8])

div = 10
low=[0.9,1,1,0,0]
up=[1.01,5,5,5,10]
#print([DCI(reduce_pareto(f[i], ["o > 0.9"]), div, low, up) for i in range(9)])