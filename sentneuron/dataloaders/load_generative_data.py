import os
from .generative import *

def load_generative_data_with_type(data_type, data_path, vocab=None, data=None):
    gen_data = None

    name = "sentneuron"
    if data_path != None:
        name = os.path.basename(data_path)

    if data_type == "txt":
        gen_data = EncoderText(data_path, vocab, data, name)
    elif data_type == "midi_note":
        gen_data = EncoderMidiNote(data_path, vocab, data, name)
    elif data_type == "midi_chord":
        gen_data = EncoderMidiChord(data_path, vocab, data, name)
    elif data_type == "midi_perform":
        gen_data = EncoderMidiPerform(data_path, vocab, data, name)

    return gen_data
