import numpy as np
import os
import re
import matplotlib.pyplot as plt



if __name__ == '__main__':
    dataset_path = "./datasets/expert_random"
    (_, _, filenames) = next(os.walk(dataset_path))
    value_matches = [tuple(re.findall('-?[0-9]+\.[0-9]*', name)) for name in filenames]

    vels = [float(d[0]) for d in value_matches]
    omegas = [float(d[1]) for d in value_matches]
    tiles = [int(float(d[2])) for d in value_matches]

    # print("count: " + str(sum([1 for item in dists if item > 100.])))

    plt.figure("velocities")
    plt.hist(vels, bins=100, range=[0, 1])
    
    plt.figure("angular velocities")
    plt.hist(omegas, bins=100, range=[-3, 3])

    plt.figure("tiles")
    plt.hist(tiles, bins=[0, 1, 2, 3, 4])

    plt.show()

