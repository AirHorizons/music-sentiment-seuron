# External imports
import math    as ma
import numpy   as np
import music21 as m21

# Local imports
from .encoder_midi import EncoderMidi

class EncoderMidiNote(EncoderMidi):
    def midi2encoding(self, midi, sample_freq=4, piano_range=88, modulate_range=1):
        try:
            midi_stream = m21.midi.translate.midiFileToStream(midi)
        except:
            return []

        # Get piano roll from midi stream
        piano_roll = self.midi2piano_roll(midi_stream, sample_freq, piano_range, modulate_range)

        # Transform piano roll into a list of notes in string format
        note_encoding = []
        for i in range(len(piano_roll)):
            for j in range(len(piano_roll[i])):
                if piano_roll[i,j] == 1:
                    note_encoding.append("n" + str(j))
            note_encoding.append(".")

        self.write(note_encoding, "note_test")

        return note_encoding

    def encoding2midi(self, note_encoding, sample_freq=4, duration=2):
        speed = 1./sample_freq
        notes = []

        ts = 0
        for note in note_encoding:
            if note == ".":
                ts += 1

            elif note[0] == "n":
                pitch = int(note.split("_")[0][1:])

                note = m21.note.Note(pitch)
                note.duration = m21.duration.Duration(duration * speed)
                note.offset = ts * duration * speed
                notes.append(note)

        piano = m21.instrument.fromString("Piano")
        notes.insert(0, piano)

        piano_stream = m21.stream.Stream(notes)
        main_stream  = m21.stream.Stream([piano_stream])

        return m21.midi.translate.streamToMidiFile(main_stream)
