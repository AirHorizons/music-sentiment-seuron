import seqgen as sg
import numpy as np

data = sg.data.TextData("input/txt/shakespeare.txt")
# data = sg.data.MidiData("input/midi/")

# Model parameters
input_size = data.encoding_size
hidden_size = 256
output_size = data.encoding_size

# Model hyper parameters
lstm_layers  = 2
lstm_dropout = 0

neuron = sg.SequenceGenerator(input_size, hidden_size, output_size, lstm_layers, lstm_dropout)

# Training parameters
epochs        = 100000
seq_length    = 100
learning_rate = 1e-3
weight_decay  = 0

neuron.train(data, epochs, seq_length, learning_rate, weight_decay)

# Sampling paramenters
sample_size = 1000
trunc_size = seq_dataset.encoding_size
random_prob = 1.

neuron(data, log_size, seq_dataset.encoding_size, random_prob)
