import os
import music21 as m21
import numpy   as np

from abc import ABC, abstractmethod
from ..dataset import Dataset

class MidiData(Dataset):
    def load(self, datapath):
        encoded_midi = []

        # Read every file in the given directory
        for file in os.listdir(datapath):
            midipath = os.path.join(datapath, file)

            # Check if it is not a directory and if it has either .midi or .mid extentions
            if os.path.isfile(midipath) and (midipath[-5:] == ".midi" or midipath[-4:] == ".mid"):
                print("Parsing midi file:", midipath)

                # Create a music21 stream and open the midi file
                midi = m21.midi.MidiFile()

                try:
                    midi.open(midipath)
                    midi.read()
                    midi.close()
                except:
                    print("Skipping file: Midi file has bad formatting")
                    continue

                # Translate midi to stream of notes and chords
                encoded_midi += self.midi_to_note_encoding(midi)

        return encoded_midi

    @abstractmethod
    def midi_to_note_encoding(self, midi):
        pass

    @abstractmethod
    def note_encoding_to_midi(self, encoded_midi):
        pass

    def encode(self, ts):
        return self.symbol_to_ix[ts]

    def decode(self, ixs):
        # Create piano roll and return it
        return np.array([self.ix_to_symbol[ix] for ix in ixs])

    def slice(self, i, length):
        return [self.encode(ts) for ts in self.data[i:i+length]]

    def labels(self, i, length):
        return [self.symbol_to_ix[ts] for ts in self.data[i+1:i+1+length]]

    def discretize_tempo(self, tempo):
        pass

    def discretize_velocity(self, velocity):
        pass

    def write(self, encoded_midi, path):
        # Base class checks if output path exists
        super().write(encoded_midi, path)

        midi = self.note_encoding_to_midi(encoded_midi)
        midi.open(Dataset.OUTPUT_PATH + path + ".mid", "wb")
        midi.write()
        midi.close()
