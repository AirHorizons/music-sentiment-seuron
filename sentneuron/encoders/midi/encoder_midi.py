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
            pitch    = note.pitch.midi
            duration = note.duration.quarterLength
            velocity = note.volume.velocity
            offset   = ma.floor(note.offset * sample_freq)

            note_events.append((pitch, duration, velocity, offset))

        return note_events

    def midi_parse_chords(self, midi_stream, sample_freq):
        chord_filter = m21.stream.filters.ClassFilter('Chord')

        note_events = []
        for chord in midi_stream.recurse().addFilter(chord_filter):
            pitches_in_chord = chord.pitches
            for pitch in pitches_in_chord:
                pitch    = pitch.midi
                duration = chord.duration.quarterLength
                velocity = chord.volume.velocity
                offset   = ma.floor(chord.offset * sample_freq)

                note_events.append((pitch, duration, velocity, offset))

        return note_events

    def midi_parse_metronome(self, midi_stream, sample_freq):
        metronome_filter = m21.stream.filters.ClassFilter('MetronomeMark')

        time_events = []
        for metro in midi_stream.recurse().addFilter(metronome_filter):
            time = int(metro.number)
            offset = ma.floor(metro.offset * sample_freq)
            time_events.append((time, offset))

        return time_events

    def midi_parse_events(self, midi):
        events = []

        track_lens = [len(track.events) for track in midi.tracks]
        n_events = max(track_lens)

        for i in range(n_events):
            for track in midi.tracks:
                if i < len(track.events):
                    ev = track.events[i]
                    if ev.isNoteOn() or ev.isNoteOn() or ev.isDeltaTime():
                        events.append(ev)

        return events

    def midi2piano_roll(self, midi_stream, sample_freq, piano_range, modulate_range, add_perform=False):
        # Parse the midi file into a list of notes (pitch, duration, velocity, offset)
        notes = []
        notes += self.midi_parse_notes(midi_stream, sample_freq)
        notes += self.midi_parse_chords(midi_stream, sample_freq)

        # Parse the midi file into a list of metronome events (time, offset)
        time_events = None
        if add_perform:
            time_events = self.midi_parse_metronome(midi_stream, sample_freq)

        # Create piano roll from the list of notes
        time_steps = ma.floor(midi_stream.duration.quarterLength * sample_freq) + 1
        piano_roll = self.notes2piano_roll(notes, time_steps, piano_range, time_events)

        return piano_roll

    def notes2piano_roll(self, notes, time_steps, piano_range, time_events=None):
        if time_events is None:
            piano_roll = np.zeros((time_steps, piano_range))
        else:
            # Increment range by one to store time events
            piano_range += 1
            piano_roll = np.zeros((time_steps, piano_range, 2))

        for n in notes:
            pitch, duration, velocity, offset = n

            # Force notes to be inside the specified piano_range
            while pitch < 0:
                pitch += 12
            while pitch >= piano_range:
                pitch -= 12

            if time_events is None:
                piano_roll[offset, pitch] = 1
            else:
                piano_roll[offset, pitch][0] = duration
                piano_roll[offset, pitch][1] = self.discretize_value(velocity, bins=32, range=(0, 128))

                for t in time_events:
                    time, offset = t
                    piano_roll[offset, -1][0] = self.discretize_value(time, bins=100, range=(0, 200))

        return piano_roll

    def modulate_piano_roll(self, piano_roll, modulate_range=1):
        modulated_piano_rolls = []

        # Modulate the piano_roll for other keys
        for key in range(0, modulate_range):
            modulated = []
            note_range = len(piano_roll[0]) - 1

            for ts in piano_roll:
                ts_str = self.ts2str(ts[1:])
                padded = '000000' + ts_str[1:] + '000000'
                modulated.append(self.str2ts(ts_str[0] + padded[key:key + note_range]))

            modulated_piano_rolls.append(modulated)

        return modulated_piano_rolls

    def ts2str(self, ts):
        return "".join(str(t) for t in ts)

    def str2ts(self, s):
        return [int(ch) for ch in s]

    def discretize_value(self, val, bins, range):
        min_val, max_val = range

        velocity = int(max(min_val, val))
        velocity = int(min(val, max_val))

        bin_size = (max_val/bins)

        return ma.floor(velocity/bin_size) * bin_size
