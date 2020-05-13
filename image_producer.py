import numpy as np
import os
from gym_duckietown.envs import DuckietownImager
from matplotlib import pyplot as plt


def save_images(images):
    path = "./pictures/"

    try:
        os.mkdir(path)
    except OSError:
        print("./pictures may already be there..")
        return

    for img, label in images:
        plt.imsave(path + str(label) + ".png", img)


if __name__ == '__main__':

    num_imgs = 100

    env = DuckietownImager()
    imgs = env.get_images(num_imgs)

    save_images(imgs)

