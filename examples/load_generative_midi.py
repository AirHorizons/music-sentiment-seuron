# External imports
import sentneuron as sn

# Load pre-calculated vocabulary
seq_data = sn.encoders.midi.EncoderMidiPerform("../trained_models/beethoven_vocab.txt", pre_loaded=True)

# Model layer sizes
model_path = "../trained_models/beethoven_model.pth"
neuron = sn.utils.load_generative_model(model_path, seq_data, embed_size=64, hidden_size=4096, n_layers=1, dropout=0)

# Sampling
notes = ["n_60_quarter_80", ".", ".", ".", ".", "n_62_quarter_80"]
sample = neuron.sample(seq_data, sample_init=notes, sample_len=200)
seq_data.write(sample, "../samples/beethoven")
