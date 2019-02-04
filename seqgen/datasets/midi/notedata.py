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

            chords[note.offset].append((note.pitch.midi, note.duration.type, int(note.volume.velocity)))

        # Append (pitch, offset) from chords
        for chord in midi_stream.recurse().addFilter(chord_filter):
            pitches_in_chord = chord.pitches
            for pitch in pitches_in_chord:
                if chord.offset not in chords:
                    chords[chord.offset] = []

                chords[chord.offset].append((pitch.midi, chord.duration.type, int(chord.volume.velocity)))

        # Get initial time signature
        time_signature = midi_stream.getTimeSignatures()[0].ratioString
        tempo = "t_" + str(ma.ceil(midi_stream.metronomeMarkBoundaries()[0][2].number))

        note_encoding = [time_signature, tempo]
        for chord in sorted(chords.keys()):
            for note in chords[chord]:
                pitch, duration, velocity = note
                note_encoding.append(str(pitch) + "_" + duration + "_" + str(velocity))
            note_encoding.append(".")

        self.write(note_encoding, "enc_test")

        return note_encoding

    def note_encoding_to_midi(self, note_encoding, sample_freq=4):
        # Set the volume of the notes to 100
        notes = []

        ts = 0
        time_signature = m21.meter.TimeSignature("4/4")
        tempo = 120

        for note in note_encoding:
            if note == ".":
                ts += 1
                continue

            if note[1] == "/":
                time_signature = m21.meter.TimeSignature(note)
                notes.append(time_signature)
                continue

            if note[0] == "t":
                tempo = m21.tempo.MetronomeMark(number=int(note.split("_")[1]))
                notes.append(tempo)
                continue

            pitch = note.split("_")[0]
            duration = note.split("_")[1]
            velocity = note.split("_")[2]

            if duration == "zero" or duration == "complex":
                continue

            note = m21.note.Note(int(pitch))
            note.duration = m21.duration.Duration(type=duration)
            note.offset = ts * 0.5
            note.volume.velocity = int(velocity)
            notes.append(note)

        piano = m21.instrument.fromString("Piano")
        notes.insert(0, piano)

        piano_stream = m21.stream.Stream(notes)
        main_stream  = m21.stream.Stream([piano_stream])

        return m21.midi.translate.streamToMidiFile(main_stream)
