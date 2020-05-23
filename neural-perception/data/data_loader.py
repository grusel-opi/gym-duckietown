import numpy as np
from numpy import load
import os
import fnmatch
from matplotlib import pyplot as plt


class DataLoader:
    def __init__(self, config):
        self.config = config
        self.set_size = len(fnmatch.filter(os.listdir(config.data_dir), "data*"))
        self.data = []

        for set_no in range(self.set_size):
            set = load(config.data_dir + "/data" + str(set_no) + ".npy")
            labels = load(config.data_dir + "/labels" + str(set_no) + ".npy")
            self.data.append((set, labels))

    def next_batch(self, batch_size):
        batch = []

        for n in range(batch_size):
            set_idx = np.random.choice(self.set_size)
            set_imgs = self.data[set_idx][0]
            set_labels = self.data[set_idx][1]
            sample_idx = np.random.choice(len(set_imgs))

            batch.append((set_imgs[sample_idx], set_labels[sample_idx]))

        return batch

    def show_batch(self, batch):
        plt.figure(figsize=(50, 50))

        for n in range(len(batch)):
            ax = plt.subplot(5, 5, n + 1)
            plt.imshow(batch[n][0])
            plt.title(batch[n][1], fontdict = {'fontsize' : 50})
            plt.axis('off')

        plt.show()
