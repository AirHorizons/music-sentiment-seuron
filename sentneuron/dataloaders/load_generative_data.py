from .generative import *

def load_generative_data_with_type(data_type, data_path, vocab=None, data=None):
    gen_data = None

    if data_type == "txt":
        gen_data = EncoderText(data_path, vocab, data)
    elif data_type == "midi_note":
        gen_data = EncoderMidiNote(data_path, vocab, data)
    elif data_type == "midi_chord":
        gen_data = EncoderMidiChord(data_path, vocab, data)
    elif data_type == "midi_perform":
        gen_data = EncoderMidiPerform(data_path, vocab, data)

    return gen_data
