# External imports
import math    as ma
import numpy   as np
import music21 as m21

# Local imports
from .encoder_midi import EncoderMidi

class EncoderMidiPerform(EncoderMidi):
    def midi2encoding(self, midi, sample_freq=4, piano_range=128, modulate_range=12):
        try:
            midi_stream = m21.midi.translate.midiFileToStream(midi)
        except:
            return []

        # Get piano roll from midi stream
        piano_roll = self.midi2piano_roll(midi_stream, sample_freq, piano_range, modulate_range, add_perform=True)

        # Transform piano roll into a list of notes in string format
        lastVelocity = -1
        lastDuration = -1.0

        note_encoding = []
        for i in range(len(piano_roll)):
            for j in range(len(piano_roll[i]) - 1):
                duration = piano_roll[i,j][0]
                velocity = int(piano_roll[i,j][1])

                if velocity != 0 and velocity != lastVelocity:
                    note_encoding.append("v_" + str(velocity))

                if duration != 0 and duration != lastDuration:
                    try:
                        duration_type, _ = m21.duration.quarterLengthToClosestType(duration)
                    except:
                        duration_type = "16th"

                    note_encoding.append("d_" + duration_type)

                if duration != 0 and velocity != 0:
                    note_encoding.append("n_" + str(j))

                lastVelocity = velocity
                lastDuration = duration

            # Time events are stored at the last row
            tempo_change = piano_roll[i,-1][0]
            if tempo_change != 0:
                note_encoding.append("t_" + str(int(tempo_change)))

            note_encoding.append(".")

        return " ".join(note_encoding)

    def encoding2midi(self, note_encoding,  ts_duration=0.25):
        notes = []

        velocity = 100
        duration = "quarter"

        ts = 0
        for note in note_encoding.split(" "):
            if note == ".":
                ts += 1

            elif note[0] == "n":
                pitch    = int(note.split("_")[1])
                note = m21.note.Note(pitch)
                note.duration = m21.duration.Duration(type=duration)
                note.offset = ts * ts_duration
                note.volume.velocity = velocity
                notes.append(note)

            elif note[0] == "d":
                duration = note.split("_")[1]

            elif note[0] == "v":
                velocity = int(note.split("_")[1])

            elif note[0] == "t":
                tempo = int(note.split("_")[1])

                if tempo > 0:
                    mark = m21.tempo.MetronomeMark(number=tempo)
                    mark.offset = ts * ts_duration
                    notes.append(mark)

        piano = m21.instrument.fromString("Piano")
        notes.insert(0, piano)

        piano_stream = m21.stream.Stream(notes)
        main_stream  = m21.stream.Stream([piano_stream])

        return m21.midi.translate.streamToMidiFile(main_stream)

    def type(self):
        return "midi_perform"
