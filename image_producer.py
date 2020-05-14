import numpy as np
import os
from gym_duckietown.envs import DuckietownImager
from matplotlib import pyplot as plt
from numpy import save
from numpy import load


def save_images(images):
    path = "./generated/"

    try:
        os.mkdir(path)
    except OSError:
        print(path + " may already be there..")
        return

    save("data.npy", images)


if __name__ == '__main__':

    num_imgs = 100

    env = DuckietownImager()
    labled_imgs = env.get_images(num_imgs)

    save_images(labled_imgs)

