import numpy as np
np.savetxt("seeds.txt", np.random.randint(0, 999, 30), fmt='%d')
