import sentneuron as sn

# Local imports
from utils import train_generative_model

# Load midi data
seq_data = sn.encoders.midi.EncoderMidiPerform("../input/generative/midi/beethoven_mond/")

# Model layer sizes
neuron = train_generative_model(seq_data, embed_size=64, hidden_size=128, n_layers=1, dropout=0, epochs=100, seq_length=256, lr=5e-4)
neuron.save(seq_data, "trainned_models/beethoven_mond")

# Sampling
notes = ["n_60_quarter_80", "n_62_quarter_80", "."]
sample = neuron.sample(seq_data, sample_init=notes, sample_len=200)
seq_data.write(sample, "samples/beethoven_mond")
