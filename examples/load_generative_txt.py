import sentneuron as sn

# Local imports
from utils import load_generative_model

# Load text data
seq_data = sn.encoders.EncoderText("../trained_models/shakespeare_vocab.txt", pre_loaded=True)

# Model layer sizes
model_path = "../trained_models/shakespeare_model.pth"
neuron = load_generative_model(model_path, embed_size=64, hidden_size=4096, n_layers=1, dropout=0)

# Sampling
sample = neuron.sample(seq_data, sample_init="I don't know ", sample_len=200)
seq_data.write(sample, "../samples/shakespeare")
