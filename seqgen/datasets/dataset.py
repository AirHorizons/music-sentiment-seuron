import os    as os
import numpy as np
from abc import ABC, abstractmethod

class Dataset(ABC):

    # Set path for saving data sampled from trained models.
    OUTPUT_PATH = "output/samples/"

    def __init__(self, datapath):
        # Load data and save size
        self.data = self.load(datapath)
        self.data_size = len(self.data)

        # Create vocabulary from data and save size
        vocab = list(set(self.data))
        self.encoding_size = len(vocab)
        
        # Create dictionaries to support char to index conversion and vice-versa
        self.symbol_to_ix = { ch:i for i,ch in enumerate(vocab) }
        self.ix_to_symbol = { i:ch for i,ch in enumerate(vocab) }

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

    def write(self, data, path):
        if not os.path.isdir(self.OUTPUT_PATH):
            os.mkdir(self.OUTPUT_PATH)

    def onehot(self, ix):
        onehot = np.zeros(self.encoding_size)
        onehot[ix] = 1
        return onehot
