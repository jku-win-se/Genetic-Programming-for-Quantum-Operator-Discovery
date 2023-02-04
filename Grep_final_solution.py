import re

#get fitness values for final quantum operator

#ToDo: make relative path as in main.py
def grep_fitness(seed, approach, filename):
    file = f"Results_{approach}_appr/Results_{filename}_{seed}.out"
    with open(file, "r") as r_file:
        pattern = "The found oracle is/are:"
        for line in r_file:
            if re.search(pattern, line):
                #print(line)
                s = line.strip().split()

    overlap = float(s[8].translate({ord(i): None for i in '[,'}))
    gates = int(s[9].replace(',' ,''))
    depth = int(s[10].replace(',' ,''))
    nl = int(s[11].replace(',', ''))
    params = int(s[12].translate({ord(i): None for i in '],'}))

    fitness_vals = [overlap, gates, depth, nl, params]
    return fitness_vals

seed = 55
approach = "comp"
filename = "Grover_Oracle"
f = grep_fitness(seed, approach, filename)
print(f)