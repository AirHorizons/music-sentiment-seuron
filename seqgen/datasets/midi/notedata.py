import math    as ma
import music21 as m21
import numpy   as np

from .mididata import MidiData

class NoteData(MidiData):
    def midi_to_note_encoding(self, midi):
        try:
            midi_stream = m21.midi.translate.midiFileToStream(midi)
        except:
            return []

        # Parse the midi file into a list of notes (pitch, offset)
        events = {}

        # Parse midi events: notes, chords and metronome marks
        self.midi_parse_notes(midi_stream, events)
        self.midi_parse_chords(midi_stream, events)
        self.midi_parse_metronome(midi_stream, events)

        note_encoding = []
        for ev in sorted(events.keys()):
            for note in events[ev]:
                pitch, duration, velocity = note
                if pitch >= 0:
                    note_encoding.append("n" + str(pitch) + "_" + duration + "_" + str(velocity))
                else:
                    note_encoding.append("t" + str(velocity))

            note_encoding.append(".")

        self.write(note_encoding, "encoded")

        return note_encoding

    def midi_parse_notes(self, midi_stream, events):
        note_filter = m21.stream.filters.ClassFilter('Note')

        for note in midi_stream.recurse().addFilter(note_filter):
            offset = float(note.offset)
            if offset not in events:
                events[offset] = []

            events[offset].append((note.pitch.midi, note.duration.type, int(note.volume.velocity)))

    def midi_parse_chords(self, midi_stream, events):
        chord_filter = m21.stream.filters.ClassFilter('Chord')

        for chord in midi_stream.recurse().addFilter(chord_filter):
            pitches_in_chord = chord.pitches
            for pitch in pitches_in_chord:
                offset = float(chord.offset)
                if offset not in events:
                    events[offset] = []

                events[offset].append((pitch.midi, chord.duration.type, int(chord.volume.velocity)))

    def midi_parse_metronome(self, midi_stream, events):
        metronome_filter = m21.stream.filters.ClassFilter('MetronomeMark')

        for metro in midi_stream.recurse().addFilter(metronome_filter):
            offset = float(metro.offset)
            if offset not in events:
                events[offset] = []

            events[offset].append((-1, None, int(metro.number)))


    def note_encoding_to_midi(self, note_encoding, sample_freq=4):
        # Set the volume of the notes to 100
        notes = []

        time_signature = m21.meter.TimeSignature("4/4")

        ts = 0

        for note in note_encoding:
            if note == ".":
                ts += 1

            elif note[0] == "t":
                metro = m21.tempo.MetronomeMark(number=int(note[1:]))
                metro.offset = ts * 0.25
                notes.append(metro)

            elif note[0] == "n":
                pitch = int(note.split("_")[0][1:])
                duration = note.split("_")[1]
                velocity = int(note.split("_")[2])

                if duration == "complex":
                    continue

                note = m21.note.Note(pitch)
                note.duration = m21.duration.Duration(type=duration)
                note.offset = ts * 0.25
                note.volume.velocity = velocity
                notes.append(note)

        piano = m21.instrument.fromString("Piano")
        notes.insert(0, piano)

        piano_stream = m21.stream.Stream(notes)
        main_stream  = m21.stream.Stream([piano_stream])

        return m21.midi.translate.streamToMidiFile(main_stream)
