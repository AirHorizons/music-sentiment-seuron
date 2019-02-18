from .encoder import Encoder

class EncoderText(Encoder):
    def load(self, textpath):
        return open(textpath, "r").read()

    def decode(self, ixs):
        return ''.join(self.ix_to_symbol[ix] for ix in ixs)

    def write(self, text, path):
        f = open(path + ".txt", "a")
        f.write(text)
        f.close()
