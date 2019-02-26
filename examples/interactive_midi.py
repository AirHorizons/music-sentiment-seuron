# External imports
import mido
import sentneuron as sn

# Local imports
from interactive       import *
from utils             import *

# =============================
# Sampling callback function
# =============================
def sample_music(notes):
    sample = neuron.sample(seq_data, sample_init=notes, sample_len=200)
    seq_data.write(sample, "interactive/sentneuron")

    midi_data = mido.MidiFile("interactive/sentneuron.mid")

    for msg in midi_data.play():
        piano.port.send(msg)

    piano.start()

# =============================
# Load data and model
# =============================

seq_data = sn.encoders.midi.EncoderMidiPerform("../input/generative/midi/beethoven/")

# Model layer sizes
model_path = "../output/generative/models/seqgen_midi_beethoven.pth"
neuron = load_generative_model(model_path, seq_data, embed_size=64, hidden_size=4096, n_layers=1, dropout=0)

# =============================
# Create interactive piano app
# =============================
TITLE = "Interactive Sampling"
SIZE  = (800,600)

app = InteractiveApp(TITLE, SIZE)

piano = InteractivePiano(sample_music)
app.add_interactive_obj(piano)

app.start()
