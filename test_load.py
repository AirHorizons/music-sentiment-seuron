import sentneuron as sn

seq_data = sn.encoders.EncoderText("input/generative/txt/shakespeare.txt")
sen_data = sn.encoders.SentimentData("input/classifier/sst", "sentence", "label", slice=(110,120))
# data = sg.encoders.midi.NoteData("input/generative/midi/")

# Model layer sizes
input_size  = seq_data.encoding_size
embed_size  = 64
hidden_size = 4096
output_size = seq_data.encoding_size

# Loading trainned model for predicting elements in a sequence.
neuron = sn.SentimentNeuron(input_size, embed_size, hidden_size, output_size, n_layers=1, dropout=0)
neuron.load("output/generative/models/seqgen_2019-02-14_13-13.pth")

# Running sentiment analysis
full_rep_acc, c, n_not_zero = neuron.fit_sentiment(seq_data, sen_data)
print('%05.3f Test accuracy' % full_rep_acc)
print('%05.3f Regularization coef' % c)
print('%05d Features used' % n_not_zero)

# Sampling
sample = neuron.sample(seq_data, sample_init="I don't know ", sample_len=200)
seq_data.write(sample, "output/generative/samples/sample_loaded_model")
