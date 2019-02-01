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

        note_filter = m21.stream.filters.ClassFilter('Note')
        chord_filter = m21.stream.filters.ClassFilter('Chord')

        # Parse the midi file into a list of notes (pitch, offset)
        chords = {}

        # Append (pitch, offset) from individual notes
        for note in midi_stream.recurse().addFilter(note_filter):
            if note.offset not in chords:
                chords[note.offset] = []

            chords[note.offset].append((note.pitch.midi, note.duration.type))

        # Append (pitch, offset) from chords
        for chord in midi_stream.recurse().addFilter(chord_filter):
            pitches_in_chord = chord.pitches
            for pitch in pitches_in_chord:
                if chord.offset not in chords:
                    chords[chord.offset] = []

                chords[chord.offset].append((pitch.midi, chord.duration.type))

        note_encoding = []
        for chord in sorted(chords.keys()):
            for note in chords[chord]:
                pitch, duration = note
                note_encoding.append(str(pitch))
            note_encoding.append(".")

        self.write(note_encoding, "notenec")
        return note_encoding

    def note_encoding_to_midi(self, note_encoding, sample_freq=4, duration=2):
        # Set the volume of the notes to 100
        speed = 1./sample_freq
        notes = []

        ts = 0
        for note in note_encoding:
            if note == ".":
                ts += 1
                continue

            note = m21.note.Note(int(note))
            note.duration = m21.duration.Duration(duration * speed)
            note.offset = ts * duration * speed
            notes.append(note)

        piano = m21.instrument.fromString("Piano")
        notes.insert(0, piano)

        piano_stream = m21.stream.Stream(notes)
        main_stream  = m21.stream.Stream([piano_stream])

        return m21.midi.translate.streamToMidiFile(main_stream)
