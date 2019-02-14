from .dataset import Dataset

class TextData(Dataset):
    def load(self, textpath):
        return open(textpath, "r").read()

    def encode(self, ch):
        return self.symbol_to_ix[ch]

    def decode(self, ixs):
        return ''.join(self.ix_to_symbol[ix] for ix in ixs)

    def slice(self, i, length):
        return [self.encode(ch) for ch in self.data[i:i + length]]

    def write(self, text, path):
        # Base class checks if output path exists
        super().write(text, path)

        f = open(Dataset.OUTPUT_PATH + path + ".txt", "a")
        f.write(text)
        f.close()
