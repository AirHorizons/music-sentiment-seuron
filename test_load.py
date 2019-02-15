import sentneuron as sn

data = sn.datasets.TextData("input/txt/shakespeare.txt")
# data = sg.datasets.midi.NoteData("input/midi/")

# Model layer sizes
input_size  = data.encoding_size
embed_size  = 64
hidden_size = 4096
output_size = data.encoding_size

# Model hyper parameters
lstm_layers  = 1
lstm_dropout = 0

neuron = sn.SentimentNeuron(input_size, embed_size, hidden_size, output_size, lstm_layers, lstm_dropout)
neuron.load("output/models/seqgen_2019-02-14_13-13.pth")

# Sampling paramenters
sample_size =  200
sample_init = "I don't know "

sample = neuron.sample(data, sample_init, sample_size)
data.write(sample, "sample_loaded_model")
