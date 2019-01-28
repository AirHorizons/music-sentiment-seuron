import os
import torch
import math    as ma
import music21 as m21
import numpy   as np

from .dataset import Dataset

class MidiData(Dataset):
    def load(self, datapath):
        data = self.__midi_to_piano_roll(m21.midi.MidiFile())

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
                piano_roll = self.__midi_to_piano_roll(midi)

                # Modulate the piano_roll for every possible key
                for i in range(0, 12):
                    modulated_piano_roll = self.__modulate_piano_roll(piano_roll, i)
                    data = np.concatenate((data, modulated_piano_roll), axis=0)

        vocab = list(set([self.__ts2str(ts) for ts in data]))

        # Create dictionaries to support piano time-step (ts) to index conversion and vice-versa
        self.ts_to_ix = { ts:i for i,ts in enumerate(vocab) }
        self.ix_to_ts = { i:ts for i,ts in enumerate(vocab) }

        return data, len(vocab), len(data)

    def encode(self, ts):
        ix = self.ts_to_ix[self.__ts2str(ts)]
        return self.onehot(ix)

    def decode(self, ixs):
        # Create piano roll and return it
        return np.array([self.__str2ts(self.ix_to_ts[ix]) for ix in ixs])

    def slice(self, i, length):
        return [self.encode(ts) for ts in self.data[i:i+length]]

    def labels(self, i, length):
        return [self.ts_to_ix[self.__ts2str(ts)] for ts in self.data[i+1:i+1+length]]

    def write(self, piano_roll, path):
        # Base class checks if output path exists
        super().write(piano_roll,path)

        midi = self.__piano_roll_to_midi(piano_roll)
        midi.open(Dataset.OUTPUT_PATH + path + ".mid", "wb")
        midi.write()
        midi.close()

    def random_example(self):
        rp = np.random.randint(self.data_size)
        return self.encode(self.data[rp])

    def sample(self, ps, top_ps=10, random_prob=1.0):
        if np.random.rand() <= random_prob:
            ps = self.__truncate_probabilities(ps.squeeze(), top_ps)
            return torch.multinomial(ps, 1).item()

        return torch.argmax(ps).item()

    def __midi_to_piano_roll(self, midi, sample_freq = 4, piano_range = 88):
        try:
            midi_stream = m21.midi.translate.midiFileToStream(midi)
        except:
            return np.empty((0, piano_range))

        note_filter = m21.stream.filters.ClassFilter('Note')
        chord_filter = m21.stream.filters.ClassFilter('Chord')

        # Parse the midi file into a list of notes (pitch, offset)
        notes = []

        # Append (pitch, offset) from individual notes
        for note in midi_stream.recurse().addFilter(note_filter):
            notes.append((note.pitch.midi, ma.floor(note.offset * sample_freq)))

        # Append (pitch, offset) from chords
        for chord in midi_stream.recurse().addFilter(chord_filter):
            pitches_in_chord = chord.pitches
            for pitch in pitches_in_chord:
                notes.append((pitch.midi, ma.floor(chord.offset * sample_freq)))

        # Create piano roll from the list of notes
        time_steps = ma.floor(midi_stream.duration.quarterLength * sample_freq) + 1
        piano_roll = np.zeros((time_steps, piano_range))

        for n in notes:
            pitch, offset = n

            # Force notes to be inside the range 0--88
            while pitch < 0:
                pitch += 12
            while pitch >= piano_range:
                pitch -= 12

            piano_roll[offset, pitch] = 1

        return piano_roll

    def __piano_roll_to_midi(self, piano_roll, sample_freq = 4, duration = 2):
        # Set the volume of the notes to 100
        speed = 1./sample_freq
        notes = []

        for i in range(len(piano_roll)):
            if np.count_nonzero(piano_roll[i]) == 0:
                continue

            for j in range(len(piano_roll[i])):
                if piano_roll[i][j] == 1:
                    note = m21.note.Note(j)
                    note.duration = m21.duration.Duration(duration * speed)
                    note.offset = i * duration * speed
                    notes.append(note)

        piano = m21.instrument.fromString("Piano")
        notes.insert(0, piano)

        piano_stream = m21.stream.Stream(notes)
        main_stream  = m21.stream.Stream([piano_stream])

        return m21.midi.translate.streamToMidiFile(main_stream)

    def __modulate_piano_roll(self, piano_roll, key):
        modulated = []
        note_range = len(piano_roll[0]) - 1

        for ts in piano_roll:
                ts_str = self.__ts2str(ts[1:])
                padded = '000000' + ts_str[1:] + '000000'
                modulated.append(self.__str2ts(ts_str[0] + padded[key:key + note_range]))

        return modulated

    def __ts2str(self, ts):
        return "".join(str(int(t)) for t in ts)

    def __str2ts(self, s):
        return [int(ch) for ch in s]

    def __truncate_probabilities(self, ps, top_ps=1):
        higher_ps = ps.topk(top_ps)[1]

        for i in set(range(len(ps))) - set(higher_ps):
            ps[i] = 0.

        sum_ps = min(1., sum(ps))
        for i in higher_ps:
            ps[i] += (1. - sum_ps)/len(higher_ps)

        return ps
