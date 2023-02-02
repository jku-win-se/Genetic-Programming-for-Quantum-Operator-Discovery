import pathlib
import numpy as np

# use new seed and save used seed
p = pathlib.Path(__file__)
for i in range(30):
    try:
        np.loadtxt(f"{p.parents[0]}\seeds.txt")
    except IOError:
        open(f"{p.parents[0]}\seeds.txt", "x")
    seeds = np.loadtxt(f"{p.parents[0]}\seeds.txt")
    seeds = list(np.atleast_1d(seeds))
    seed = np.random.randint(1000, size=1)  # aus dem args
    while seed in seeds:
        seed = np.random.randint(1000, size=1)
    seeds.append(seed[0])
    np.array(seeds)
    np.savetxt(f"{p.parents[0]}\seeds.txt", seeds)