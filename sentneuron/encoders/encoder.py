import json
import numpy as np

from abc import ABC, abstractmethod

class Encoder(ABC):
    def __init__(self, datapath, pre_loaded=False):
        if not pre_loaded:
            # Load data and vocabulary from files
            self.data, self.vocab = self.load(datapath)
            self.vocab = list(self.vocab)
            self.vocab.sort()
        else:
            # Load vocabulary
            with open(datapath, 'r') as fp:
                self.vocab = fp.read().split(" ")
                self.vocab = list(self.vocab)
                self.vocab.sort()
                fp.close()

            # Remove empty element "" from the first position
            self.vocab = self.vocab[1:]

        self.encoding_size = len(self.vocab)

        # Create dictionaries to support symbol to index conversion and vice-versa
        self.symbol_to_ix = { symb:i for i,symb in enumerate(self.vocab) }
        self.ix_to_symbol = { i:symb for i,symb in enumerate(self.vocab) }

    @abstractmethod
    def load(self, datapath):
        pass

    @abstractmethod
    def decode(self, datapoint):
        pass

    @abstractmethod
    def read(self, file):
        pass

    @abstractmethod
    def write(self, data, path):
        pass

    def slice(self, data, i, length):
        return [self.encode(ts) for ts in data[i:i+length]]

    def encode(self, symb):
        return self.symbol_to_ix[symb]
