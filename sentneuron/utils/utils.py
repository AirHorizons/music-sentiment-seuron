import json
import sentneuron as sn

def load_generative_model(model_path):
    with open(model_path + "_meta.json", 'r') as fp:
        meta = json.loads(fp.read())
        fp.close()

    # Load pre-calculated vocabulary
    seq_data = None
    if meta["data_type"] == "txt":
        seq_data = sn.encoders.EncoderText(meta["vocab"], pre_loaded=True)
    elif meta["data_type"] == "midi_note":
        seq_data = sn.encoders.midi.EncoderMidiNote(meta["vocab"], pre_loaded=True)
    elif meta["data_type"]== "midi_chord":
        seq_data = sn.encoders.midi.EncoderMidiChord(meta["vocab"], pre_loaded=True)
    elif meta["data_type"] == "midi_perform":
        seq_data = sn.encoders.midi.EncoderMidiPerform(meta["vocab"], pre_loaded=True)

    # Model layer sizes
    input_size  = seq_data.encoding_size
    output_size = seq_data.encoding_size

    # Loading trainned model for predicting elements in a sequence.
    neuron = sn.SentimentNeuron(meta["input_size"], meta["embed_size"], meta["hidden_size"], meta["output_size"], meta["n_layers"], meta["dropout"])
    neuron.load(model_path + "_model.pth")

    return neuron, seq_data

def train_generative_model(seq_data, embed_size, hidden_size, n_layers=1, dropout=0, epochs=1000, seq_length=256, lr=5e-4):
    # Model layer sizes
    input_size  = seq_data.encoding_size
    output_size = seq_data.encoding_size

    # Training model for predicting elements in a sequence.
    neuron = sn.SentimentNeuron(input_size, embed_size, hidden_size, output_size, n_layers=n_layers, dropout=dropout)
    neuron.fit_sequence(seq_data, epochs=epochs, seq_length=seq_length, lr=lr)

    return neuron

def train_sentiment_analysis(neuron, seq_data, sen_data):
    # Running sentiment analysis
    full_rep_acc, c, n_not_zero = neuron.fit_sentiment(seq_data, sen_data)
    print('%05.3f Test accuracy' % full_rep_acc)
    print('%05.3f Regularization coef' % c)
    print('%05d Features used' % n_not_zero)
