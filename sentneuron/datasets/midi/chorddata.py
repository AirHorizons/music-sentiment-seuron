import math    as ma
import music21 as m21
import numpy   as np

from .mididata import MidiData

class ChordData(MidiData):
    def midi_to_note_encoding(self, midi, sample_freq = 4, piano_range = 88, modulate_range=1):
        try:
            midi_stream = m21.midi.translate.midiFileToStream(midi)
        except:
            return []

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

        # Modulate the piano_roll for other keys
        for i in range(0, modulate_range):
            modulated_piano_roll = self.__modulate_piano_roll(piano_roll, i)
            piano_roll = np.concatenate((piano_roll, modulated_piano_roll), axis=0)

        # Transform piano roll into a list of chords in string format
        piano_roll_str = [self.__ts2str(ts) for ts in piano_roll]

        return piano_roll_str

    def note_encoding_to_midi(self, piano_roll, sample_freq = 4, duration = 2):
        # Set the volume of the notes to 100
        speed = 1./sample_freq
        notes = []

        for i in range(len(piano_roll)):
            ts = self.__str2ts(piano_roll[i])

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
