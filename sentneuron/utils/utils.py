import json
import sentneuron as sn

def create_data_with_type(data, data_type, pre_loaded):
    seq_data = None

    if data_type == "txt":
        seq_data = sn.encoders.EncoderText(data, pre_loaded)
    elif data_type == "midi_note":
        seq_data = sn.encoders.midi.EncoderMidiNote(data, pre_loaded)
    elif data_type == "midi_chord":
        seq_data = sn.encoders.midi.EncoderMidiChord(data, pre_loaded)
    elif data_type == "midi_perform":
        seq_data = sn.encoders.midi.EncoderMidiPerform(data, pre_loaded)

    return seq_data

def load_generative_model(model_path):
    with open(model_path + "_meta.json", 'r') as fp:
        meta = json.loads(fp.read())
        fp.close()

    # Load pre-calculated vocabulary
    seq_data = create_data_with_type(meta["vocab"], meta["data_type"], pre_loaded=True)

    # Model layer sizes
    input_size  = seq_data.encoding_size
    output_size = seq_data.encoding_size

    # Loading trainned model for predicting elements in a sequence.
    neuron = sn.SentimentNeuron(meta["input_size"], meta["embed_size"], meta["hidden_size"], meta["output_size"], meta["n_layers"], meta["dropout"])
    neuron.load(model_path + "_model.pth")

    return neuron, seq_data

def train_generative_model(data, data_type, embed_size, hidden_size, n_layers=1, dropout=0, epochs=1000, seq_length=256, lr=5e-4):
    seq_data = create_data_with_type(data, data_type, pre_loaded=False)

    input_size  = seq_data.encoding_size
    output_size = seq_data.encoding_size

    # Training model for predicting elements in a sequence.
    neuron = sn.SentimentNeuron(input_size, embed_size, hidden_size, output_size, n_layers=n_layers, dropout=dropout)
    neuron.fit_sequence(seq_data, epochs=epochs, seq_length=seq_length, lr=lr)

    return neuron, seq_data

def train_sentiment_analysis(neuron, seq_data, sen_data):
    print("Embedding Trainning Sequences.")
    trX, trY = sen_data.train
    for i in range(len(trX)):
        trX[i] = neuron.transform_sequence(seq_data, trX[i])

    print("Embedding Validation Sequences.")
    vaX, vaY = sen_data.validation
    for i in range(len(validation_xs)):
        vaX[i] = neuron.transform_sequence(seq_data, vaX[i])

    print("Embedding Test Sequences.")
    teX, teY = sen_data.test
    for i in range(len(test_xs)):
        teX[i] = neuron.transform_sequence(seq_data, teX[i])

    # Running sentiment analysis
    full_rep_acc, c, n_not_zero, logreg_model = neuron.fit_sentiment(trX, trY, vaX, vaY, teX, teY)
    print('%05.3f Test accuracy' % full_rep_acc)
    print('%05.3f Regularization coef' % c)
    print('%05d Features used' % n_not_zero)

    k_indices = neuron.get_top_k_neuron_weights(logreg_model)
    print(k_indices)

def plot_logits(save_root, X, Y_pred, top_neurons):
    """plot logits and save to appropriate experiment directory"""
    save_root = os.path.join(save_root,'logit_vis')
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    print('plotting_logits at', save_root)

    for i, n in enumerate(top_neurons):
        plot_logit_and_save(trXt, trY, n, os.path.join(save_root, str(i)+'_'+str(n)))


def plot_logit_and_save(logits, labels, logit_index, name):
    """
    Plots histogram (wrt to what label it is) of logit corresponding to logit_index.
    Saves plotted histogram to name.
    Args:
        logits:
        labels:
        logit_index:
        name:
"""
    logit = logits[:,logit_index]
    plt.title('Distribution of Logit Values')
    plt.ylabel('# of logits per bin')
    plt.xlabel('Logit Value')
    plt.hist(logit[labels < .5], bins=25, alpha=0.5, label='neg')
    plt.hist(logit[labels >= .5], bins=25, alpha=0.5, label='pos')
    plt.legend()
    plt.savefig(name+'.png')
    plt.clf()
