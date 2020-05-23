import numpy as np
from numpy import load
import os
import fnmatch
from matplotlib import pyplot as plt


class DataLoader:
    def __init__(self, config):
        self.config = config
        self.set_size = len(fnmatch.filter(os.listdir(config.data_dir), "data*"))
        self.imgs = []
        self.labels = []

        for set_no in range(self.set_size):
            set_imgs = load(config.data_dir + "/data" + str(set_no) + ".npy")
            set_labels = load(config.data_dir + "/labels" + str(set_no) + ".npy")
            self.imgs.append(set_imgs)
            self.labels.append(set_labels)

    def next_batch(self, batch_size):
        batch_imgs = []
        batch_labels = []

        for n in range(batch_size):
            set_idx = np.random.choice(self.set_size)
            set_imgs = self.imgs[set_idx]
            set_labels = self.labels[set_idx]
            sample_idx = np.random.choice(len(set_imgs))

            batch_imgs.append(set_imgs[sample_idx])
            batch_labels.append(set_labels[sample_idx])

        return np.asarray(batch_imgs), np.asarray(batch_labels)

    def show_batch(self, batch_imgs, batch_labels):
        plt.figure(figsize=(50, 50))

        for n in range(len(batch_imgs)):
            plt.subplot(5, 5, n + 1)
            plt.imshow(batch_imgs[n])
            plt.title(batch_labels[n], fontdict={'fontsize': 50})
            plt.axis('off')

        plt.show()
