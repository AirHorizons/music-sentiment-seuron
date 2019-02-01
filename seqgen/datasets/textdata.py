from .dataset import Dataset

class TextData(Dataset):
    def load(self, textpath):
        return open(textpath, "r").read()

    def encode(self, ch):
        return self.onehot(self.symbol_to_ix[ch])

    def decode(self, ixs):
        return ''.join(self.ix_to_symbol[ix] for ix in ixs)

    def slice(self, i, length):
        return [self.encode(ch) for ch in self.data[i:i + length]]

    def labels(self, i, length):
        return [self.symbol_to_ix[ch] for ch in self.data[i+1:i+1 + length]]

    def write(self, text, path):
        # Base class checks if output path exists
        super().write(text, path)

        f = open(Dataset.OUTPUT_PATH + path + ".txt", "w")
        f.write(text)
        f.close()
