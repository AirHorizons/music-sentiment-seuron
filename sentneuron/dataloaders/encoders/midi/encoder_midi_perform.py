# External imports
import math    as ma
import numpy   as np
import music21 as m21

# Local imports
from .encoder_midi import EncoderMidi

class EncoderMidiPerform(EncoderMidi):
    def midi2encoding(self, midi, sample_freq, piano_range, modulate_range, stretching_range, invert, retrograde):
        try:
            midi_stream = m21.midi.translate.midiFileToStream(midi)
        except:
            return []

        # Get piano roll from midi stream
        performances = self.midi2piano_roll_with_performance(midi_stream, sample_freq, piano_range, modulate_range, stretching_range, invert, retrograde)

        return " ".join(self.performances2encoding(performances))

    def performances2encoding(self, performances):
        # Transform piano roll into a list of notes in string format
        lastVelocity = -1
        lastDuration = -1.0

        final_encoding = {}

        perform_i = 0
        for piano_roll in performances:
            current_tempo = "t_120"
            perform_encoding = []
            for i in range(len(piano_roll)):
                # Time events are stored at the last row
                tempo_change = piano_roll[i,-1][0]
                if tempo_change != 0:
                    current_tempo = "t_" + str(int(tempo_change))
                    perform_encoding.append(current_tempo)

                for j in range(len(piano_roll[i]) - 1):
                    duration = piano_roll[i,j][0]
                    velocity = int(piano_roll[i,j][1])

                    if velocity != 0 and velocity != lastVelocity:
                        perform_encoding.append("v_" + str(velocity))

                    if duration != 0 and duration != lastDuration:
                        duration_tuple = m21.duration.durationTupleFromQuarterLength(duration)
                        perform_encoding.append("d_" + duration_tuple.type + "_" + str(duration_tuple.dots))

                    if duration != 0 and velocity != 0:
                        perform_encoding.append("n_" + str(j))

                    lastVelocity = velocity
                    lastDuration = duration

                # After every 4-bar phrase (64 time spets),
                # add current time and mark end of phrase with period.
                if i > 0 and i % 64 == 0:
                    perform_encoding.append(".")
                    perform_encoding.append(current_tempo)
                else:
                    perform_encoding.append(",")

            perform_encoding.append(".")
            perform_encoding.append("\n")

            # Check if this version of the MIDI is already added
            perform_encoding_str = " ".join(perform_encoding)
            if perform_encoding_str not in final_encoding:
                final_encoding[perform_encoding_str] = perform_i

            perform_i += 1

        return final_encoding.keys()

    def encoding2midi(self, note_encoding, ts_duration=0.25):
        notes = []

        velocity = 100
        duration = "16th"
        dots = 0

        ts = 0
        for note in note_encoding.split(" "):
            if len(note) == 0:
                continue

            if note == ",":
                ts += 1

            if note == ".":
                ts += 1

            elif note[0] == "n":
                pitch = int(note.split("_")[1])
                note = m21.note.Note(pitch)
                note.duration = m21.duration.Duration(type=duration, dots=dots)
                note.offset = ts * ts_duration
                note.volume.velocity = velocity
                notes.append(note)

            elif note[0] == "d":
                duration = note.split("_")[1]
                dots = int(note.split("_")[2])

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
