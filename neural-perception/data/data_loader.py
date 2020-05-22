import numpy as np
from numpy import load
import os
import fnmatch
from matplotlib import pyplot as plt


class DataLoader:
    def __init__(self, config):
        self.config = config
        set_size = len(fnmatch.filter(os.listdir(config.data_dir), "data*"))
        self.data = []

        for set_no in range(0, set_size):
            set = load(config.data_dir + "/data" + str(set_no) + ".npy")
            labels = load(config.data_dir + "/labels" + str(set_no) + ".npy")
            self.data.append((set, labels))

    def next_batch(self, batch_size):
        idx = np.random.choice(500, batch_size)
        yield self.input[idx], self.y[idx]
