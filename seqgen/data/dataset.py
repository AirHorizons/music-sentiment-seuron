import os    as os
import numpy as np
from abc import ABC, abstractmethod

class Dataset(ABC):

    # Set path for saving data sampled from trained models.
    OUTPUT_PATH = "output/samples/"

    def __init__(self, datapath):
        self.data, self.encoding_size, self.data_size = self.load(datapath)

    @abstractmethod
    def load(self, datapath):
        pass

    @abstractmethod
    def encode(self, datapoint):
        pass

    @abstractmethod
    def decode(self, datapoint):
        pass

    @abstractmethod
    def slice(self, i, length):
        pass

    @abstractmethod
    def labels(self, i, length):
        pass

    @abstractmethod
    def random_example(self):
        pass

    @abstractmethod
    def sample(self, ps):
        pass

    def write(self, data, path):
        if not os.path.isdir(self.OUTPUT_PATH):
            os.mkdir(self.OUTPUT_PATH)

    def onehot(self, ix):
        onehot = np.zeros(self.encoding_size)
        onehot[ix] = 1
        return onehot
