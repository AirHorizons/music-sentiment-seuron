import sentneuron as sn

# Local imports
from utils import train_generative_model

# Load text data
seq_data = sn.encoders.EncoderText("../input/generative/txt/")

# Model layer sizes
neuron = train_generative_model(seq_data, embed_size=64, hidden_size=4096, n_layers=1, dropout=0, epochs=100, seq_length=256, lr=5e-4)
neuron.save(seq_data, "../trained_models/shakespeare")

# Sampling
sample = neuron.sample(seq_data, sample_init="I don't know ", sample_len=200)
seq_data.write(sample, "../samples/shakespeare")
