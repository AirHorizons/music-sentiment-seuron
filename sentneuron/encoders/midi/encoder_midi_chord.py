# External imports
import math    as ma
import numpy   as np
import music21 as m21

# Local imports
from .encoder_midi import EncoderMidi

class EncoderMidiChord(EncoderMidi):
    def midi2encoding(self, midi, sample_freq=4, piano_range=128, modulate_range=1):
        try:
            midi_stream = m21.midi.translate.midiFileToStream(midi)
        except:
            return []

        # Get piano roll from midi stream
        piano_roll = self.midi2piano_roll(midi_stream, sample_freq, piano_range, modulate_range)

        # Transform piano roll into a list of chords in string format
        chord_encoding = [self.ts2str(ts) for ts in piano_roll]

        return chord_encoding

    def encoding2midi(self, chord_encoding, sample_freq = 4, duration = 2):
        speed = 1./sample_freq
        notes = []

        for i in range(len(chord_encoding)):
            ts = self.__str2ts(chord_encoding[i])

            if np.count_nonzero(ts) == 0:
                continue

            for j in range(len(ts)):
                if ts[j] == 1:
                    note = m21.note.Note(j)
                    note.duration = m21.duration.Duration(duration * speed)
                    note.offset = i * duration * speed
                    notes.append(note)

        piano = m21.instrument.fromString("Piano")
        notes.insert(0, piano)

        piano_stream = m21.stream.Stream(notes)
        main_stream  = m21.stream.Stream([piano_stream])

        return m21.midi.translate.streamToMidiFile(main_stream)

    def ts2str(self, ts):
        return "".join(str(int(t)) for t in ts)

    def str2ts(self, s):
        return [int(ch) for ch in s]

    def type(self):
        return "midi_chord"
