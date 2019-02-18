import sentneuron as sn

seq_data = sn.encoders.EncoderText("input/generative/txt/shakespeare.txt")
sen_data = sn.encoders.SentimentData("input/classifier/sst", "sentence", "label")
# data = sn.datasets.midi.NoteData("input/midi/")

# Model layer sizes
input_size  = seq_data.encoding_size
embed_size  = 64
hidden_size = 128
output_size = seq_data.encoding_size

neuron = sn.SentimentNeuron(input_size, embed_size, hidden_size, output_size, n_layers=1, dropout=0)

neuron.fit_sequence(seq_data, epochs=10, seq_length=256, lr=5e-4)
neuron.fit_sentiment(sen_data, sen_data)

# Sampling
sample = neuron.sample(data, sample_init="I don't know ", sample_len=200)
seq_data.write(sample, "output/generative/samples/sample_trained_model")
