# External imports
import sentneuron as sn

# Local imports
from utils import load_generative_model

# Load pre-calculated vocabulary
seq_data = sn.encoders.midi.EncoderMidiPerform("trainned_models/beethoven_mond_vocab.txt", pre_loaded=True)

# Model layer sizes
model_path = "trainned_models/beethoven_mond_model.pth"
neuron = load_generative_model(model_path, seq_data, embed_size=64, hidden_size=128, n_layers=1, dropout=0)

# Sampling
notes = ["n_60_quarter_80", "n_62_quarter_80", "."]
sample = neuron.sample(seq_data, sample_init=notes, sample_len=200)
seq_data.write(sample, "samples/beethoven_mond")
