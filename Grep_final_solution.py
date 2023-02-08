import re
import numpy as np

# get fitness values for final quantum operator


def grep_fitness(seed, approach, filename):
    file = f"Results_{approach}_appr/Results_{filename}.txt_{seed}.out"
    with open(file, "r") as r_file:
        pattern = "The found oracle is/are:"
        for line in r_file:
            if re.search(pattern, line):
                s = line.strip().split()

    overlap = float(s[8].translate({ord(i): None for i in '[,'}))
    gates = int(s[9].replace(',' , ''))
    depth = int(s[10].replace(',' , ''))
    nl = int(s[11].replace(',', ''))
    params = int(s[12].translate({ord(i): None for i in '],'}))

    fitness_vals = [overlap, gates, depth, nl, params]
    return fitness_vals


seeds = np.loadtxt("C:/Users/fege9/anaconda3/Model-based-QC/GP/Python_Scrips/GECCO2023_Artifact/seeds.txt")
for seed in seeds:
    approach = "comp"
    filename = "U_S"
    f = grep_fitness(int(seed), approach, filename)
    print(f)
