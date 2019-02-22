import sentneuron as sn

# seq_data = sn.encoders.EncoderText("input/generative/txt/")
seq_data = sn.encoders.midi.EncoderMidiPerform("input/generative/midi/beethoven_mond/")

# Model layer sizes
input_size  = seq_data.encoding_size
embed_size  = 64
hidden_size = 128
output_size = seq_data.encoding_size

neuron = sn.SentimentNeuron(input_size, embed_size, hidden_size, output_size, n_layers=1, dropout=0)

neuron.fit_sequence(seq_data, epochs=1000, seq_length=256, lr=5e-4)

# Sampling
sample = neuron.sample(seq_data, sample_init="I don't know ", sample_len=200)
seq_data.write(sample, "output/generative/samples/sample_trained_model")
