import torch
import numpy as np

from .dataset import Dataset

class TextData(Dataset):
    def load(self, textpath):
        text = open(textpath, "r").read()

        vocab = list(set(text))

        # Create dictionaries to support char to index conversion and vice-versa
        self.char_to_ix = { ch:i for i,ch in enumerate(vocab) }
        self.ix_to_char = { i:ch for i,ch in enumerate(vocab) }

        return text, len(vocab), len(text)

    def encode(self, ch):
        return self.onehot(self.char_to_ix[ch])

    def decode(self, ixs):
        return ''.join(self.ix_to_char[ix] for ix in ixs)

    def slice(self, i, length):
        return [self.encode(ch) for ch in self.data[i:i + length]]

    def labels(self, i, length):
        return [self.char_to_ix[ch] for ch in self.data[i+1:i+1 + length]]

    def write(self, text, path):
        f = open("output/" + path + ".txt", "w")
        f.write(text)
        f.close()

    def random_example(self):
        rp = np.random.randint(self.data_size)
        return self.encode(self.data[rp])

    def sample(self, ps):
        return torch.multinomial(torch.Tensor(ps), 1).item()
