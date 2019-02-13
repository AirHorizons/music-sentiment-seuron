import seqgen as sg

# data = sg.datasets.TextData("input/txt/shakespeare.txt")
data = sg.datasets.midi.NoteData("input/midi/")

# Model layer parameters
embed_size = data.encoding_size
input_size = 64
hidden_size = 4096
output_size = 256

# Model hyper parameters
lstm_layers  = 1
lstm_dropout = 0

# Model device parameters: 'cpu' or 'cuda'
enable_cuda = True

neuron = sg.SentimentNeuron(embed_size, input_size, hidden_size, output_size, lstm_layers, lstm_dropout, enable_cuda)

# Training parameters
epochs        = 10000
seq_length    = 256
learning_rate = 5e-4

# Sampling paramenters
sample_size =  2 * seq_length
save_samples = True

neuron.fit(data, epochs, seq_length, learning_rate)
data.write(neuron.sample(data, sample_size), "final")
