# External imports
import os
import math    as ma
import numpy   as np
import music21 as m21

from abc import ABC, abstractmethod

# Local imports
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
                midi_content = self.midi2encoding(midi)
                midi_name = midipath.split("/")[-1]

                vocab = vocab | set(midi_content)
                encoded_midi.append((midi, midi_name))

        return encoded_midi, vocab

    @abstractmethod
    def midi2encoding(self, midi):
        pass

    @abstractmethod
    def encoding2midi(self, encoded_midi):
        pass

    def decode(self, ixs):
        # Create piano roll and return it
        return np.array([self.ix_to_symbol[ix] for ix in ixs])

    def read(self, midi):
        return self.midi2encoding(midi)

    def write(self, encoded_midi, path):
        # Base class checks if output path exists
        midi = self.encoding2midi(encoded_midi)
        midi.open(path + ".mid", "wb")
        midi.write()
        midi.close()

    def midi_parse_notes(self, midi_stream, sample_freq):
        note_filter = m21.stream.filters.ClassFilter('Note')

        note_events = []
        for note in midi_stream.recurse().addFilter(note_filter):
            note_events.append((note.pitch.midi, ma.floor(note.offset * sample_freq)))

        return note_events

    def midi_parse_chords(self, midi_stream, sample_freq):
        chord_filter = m21.stream.filters.ClassFilter('Chord')

        note_events = []
        for chord in midi_stream.recurse().addFilter(chord_filter):
            pitches_in_chord = chord.pitches
            for pitch in pitches_in_chord:
                note_events.append((pitch.midi, ma.floor(chord.offset * sample_freq)))

        return note_events

    def midi_parse_metronome(self, midi_stream, sample_freq):
        metronome_filter = m21.stream.filters.ClassFilter('MetronomeMark')

        time_events = []
        for metro in midi_stream.recurse().addFilter(metronome_filter):
            time_events.append((int(metro.number), ma.floor(chord.offset * sample_freq)))

        return time_events

    def midi2piano_roll(self, midi_stream, sample_freq, piano_range, modulate_range):
        # Parse the midi file into a list of notes (pitch, offset)
        notes = []
        notes += self.midi_parse_notes(midi_stream, sample_freq)
        notes += self.midi_parse_chords(midi_stream, sample_freq)

        # Create piano roll from the list of notes
        time_steps = ma.floor(midi_stream.duration.quarterLength * sample_freq) + 1
        piano_roll = self.notes2piano_roll(notes, time_steps, piano_range)

        # Modulate the piano_roll for other keys
        for i in range(0, modulate_range):
            modulated_piano_roll = self.modulate_piano_roll(piano_roll, i)
            piano_roll = np.concatenate((piano_roll, modulated_piano_roll), axis=0)

        return piano_roll

    def notes2piano_roll(self, notes, time_steps, piano_range):
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

    def modulate_piano_roll(self, piano_roll, key):
        modulated = []
        note_range = len(piano_roll[0]) - 1

        for ts in piano_roll:
                ts_str = self.ts2str(ts[1:])
                padded = '000000' + ts_str[1:] + '000000'
                modulated.append(self.str2ts(ts_str[0] + padded[key:key + note_range]))

        return modulated

    def ts2str(self, ts):
        return "".join(str(int(t)) for t in ts)

    def str2ts(self, s):
        return [int(ch) for ch in s]
