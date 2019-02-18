import os    as os
import numpy as np
from abc import ABC, abstractmethod

class Encoder(ABC):

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
    def decode(self, datapoint):
        pass

    @abstractmethod
    def write(self, data, path):
        pass

    def slice(self, i, length):
        return [self.encode(ts) for ts in self.data[i:i+length]]

    def encode(self, symb):
        return self.symbol_to_ix[symb]
