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
neuron.load("output/models/seqgen_2019-02-06_13-55.pth")

sample_size = 100
top_ps      = 0
sample_prob = 0

midi_sample = neuron.sample(data, sample_size, top_ps, sample_prob)
data.write(midi_sample, "final_cpu")
