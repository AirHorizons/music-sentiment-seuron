import seqgen as sg
import numpy  as np

# data = sg.data.TextData("input/txt/shakespeare.txt")
data = sg.data.MidiData("input/midi/beethoven")

# Model layer parameters
input_size = data.encoding_size
hidden_size = 256
output_size = data.encoding_size

# Model hyper parameters
lstm_layers  = 2
lstm_dropout = 0

# Model device parameters: 'cpu' or 'cuda'
enable_cuda = True

neuron = sg.SequenceGenerator(input_size, hidden_size, output_size, lstm_layers, lstm_dropout, enable_cuda)

# Training parameters
epochs        = 100000
seq_length    = 100
learning_rate = 1e-3
weight_decay  = 0

# Sampling paramenters
sample_size =  2 * seq_length
save_samples = True

# neuron.load("seqgen_2019-01-24_18-40.pth")
neuron.train(data, epochs, seq_length, learning_rate, weight_decay)
sample = neuron.sample(data, sample_size)

print(sample)
