import os
import music21 as m21
import numpy   as np

from abc import ABC, abstractmethod
from ..encoder import Encoder

class EncoderMidi(Encoder):
    def load(self, datapath):
        encoded_midi = []

        vocab = set()

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
                midi_content = self.midi_to_note_encoding(midi)
                midi_name = midipath.split("/")[-1]

                vocab = vocab | set(midi_content)
                encoded_midi.append((midi, midi_name))

        return encoded_midi, vocab

    @abstractmethod
    def midi_to_note_encoding(self, midi):
        pass

    @abstractmethod
    def note_encoding_to_midi(self, encoded_midi):
        pass

    def decode(self, ixs):
        # Create piano roll and return it
        return np.array([self.ix_to_symbol[ix] for ix in ixs])

    def discretize_tempo(self, tempo):
        pass

    def discretize_velocity(self, velocity):
        pass

    def read(self, midi):
        return self.midi_to_note_encoding(midi)

    def write(self, encoded_midi, path):
        # Base class checks if output path exists
        midi = self.note_encoding_to_midi(encoded_midi)
        midi.open(path + ".mid", "wb")
        midi.write()
        midi.close()
