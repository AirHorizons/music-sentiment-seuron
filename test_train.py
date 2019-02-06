import seqgen as sg

# data = sg.datasets.TextData("input/txt/shakespeare.txt")
data = sg.datasets.midi.NoteData("input/midi/")

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
epochs        = 10000
seq_length    = 100
learning_rate = 1e-3
weight_decay  = 0

# Sampling paramenters
sample_size =  2 * seq_length
save_samples = True

neuron.fit(data, epochs, seq_length, learning_rate, weight_decay)
data.write(neuron.sample(data, sample_size), "final")
