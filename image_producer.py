import numpy as np
import os
from gym_duckietown.envs import DuckietownImager
from matplotlib import pyplot as plt
from numpy import save
from numpy import load

if __name__ == '__main__':

    path = "./generated/"

    num_imgs = 1000
    sets = 30

    env = DuckietownImager(num_imgs)

    for i in range(sets):

        env.produce_images()
        images, labels = env.images, env.labels

        try:
            os.mkdir(path)
        except OSError:
            pass

        save(path + "data" + str(i) + ".npy", images)
        save(path + "labels" + str(i) + ".npy", labels)

    # loaded_i = load(path + "data.npy")
    # loaded_l = load(path + "labels.npy")
    #
    # for j in range(10):
    #     plt.figure(str(loaded_l[j]))
    #     plt.imshow(loaded_i[j])
    #
    # plt.show()

