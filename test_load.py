import sentneuron as sn

seq_data = sn.encoders.EncoderText("input/generative/txt/shakespeare.txt")
sen_data = sn.encoders.SentimentData("input/classifier/sst", "sentence", "label")

# data = sg.datasets.midi.NoteData("input/midi/")

# Model layer sizes
input_size  = seq_data.encoding_size
embed_size  = 64
hidden_size = 4096
output_size = seq_data.encoding_size

# Loading model
neuron = sn.SentimentNeuron(input_size, embed_size, hidden_size, output_size, n_layers=1, dropout=0)
neuron.load("output/generative/models/seqgen_2019-02-14_13-13.pth")

neuron.fit_sentiment(seq_data, sen_data)

# Sampling
sample = neuron.sample(seq_data, sample_init="I don't know ", sample_len=200)
data.write(sample, "output/generative/samples/sample_loaded_model")
