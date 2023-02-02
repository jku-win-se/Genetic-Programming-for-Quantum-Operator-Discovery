from src import main
from src import helper_func as fun
import sys

if __name__ == "__main__":
    circuit = sys.argv[1]
    settings = sys.argv[2]
    seed = int(sys.argv[3])
    fun.seeded_rng(seed=seed)
    main.main(circuit, settings, seed)
