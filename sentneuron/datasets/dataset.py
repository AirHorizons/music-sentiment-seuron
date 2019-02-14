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
        vocab.sort()

        self.encoding_size = len(vocab)

        # Create dictionaries to support symbol to index conversion and vice-versa
        self.symbol_to_ix = { symb:i for i,symb in enumerate(vocab) }
        self.ix_to_symbol = { i:symb for i,symb in enumerate(vocab) }

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

    def write(self, data, path):
        if not os.path.isdir(self.OUTPUT_PATH):
            os.mkdir(self.OUTPUT_PATH)
