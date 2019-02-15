import sentneuron as sn

data = sn.datasets.TextData("input/txt/shakespeare.txt")
# data = sn.datasets.midi.NoteData("input/midi/")

# Model layer sizes
input_size  = data.encoding_size
embed_size  = 64
hidden_size = 128
output_size = data.encoding_size

# Model hyper parameters
lstm_layers  = 1
lstm_dropout = 0

neuron = sn.SentimentNeuron(input_size, embed_size, hidden_size, output_size, lstm_layers, lstm_dropout)

# Training parameters
epochs        = 10
seq_length    = 256
learning_rate = 5e-4

neuron.fit(data, epochs, seq_length, learning_rate)

# Sampling paramenters
sample_size =  200
sample_init = "I don't know "

sample = neuron.sample(data, sample_init, sample_size)
data.write(sample, "final_cpu")
