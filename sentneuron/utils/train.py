import os
import json

import numpy      as np
import sentneuron as sn

# Local imports
from .plot import *

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

def train_generative_model(data, data_type, embed_size, hidden_size, n_layers=1, dropout=0, epochs=100, seq_length=256, lr=5e-4, lr_decay=0.7, grad_clip=5, batch_size=128):
    seq_data = create_data_with_type(data, data_type, pre_loaded=False)

    input_size  = seq_data.encoding_size
    output_size = seq_data.encoding_size

    # Training model for predicting elements in a sequence.
    neuron = sn.SentimentNeuron(input_size, embed_size, hidden_size, output_size, n_layers, dropout)
    neuron.fit_sequence(seq_data, epochs, seq_length, lr, lr_decay, grad_clip, batch_size)

    return neuron, seq_data

def tranform_sentiment_data(neuron, seq_data, xs, xs_filename):
    if(os.path.isfile(xs_filename)):
        xs = np.squeeze(np.load(xs_filename))
    else:
        for i in range(len(xs)):
            xs[i], _ = neuron.transform_sequence(seq_data, xs[i])
        np.save(xs_filename, xs)

    return xs

def train_sentiment_analysis(neuron, seq_data, sent_data_path, results_path):
    # Load sentiment data from given path
    sent_data = sn.encoders.SentimentData(sent_data_path, "sentence", "label")

    print("Transforming Trainning Sequences.")
    trX, trY = sent_data.train
    trXt = tranform_sentiment_data(neuron, seq_data, trX, os.path.join(sent_data_path, 'trX.npy'))

    print("Transforming Validation Sequences.")
    vaX, vaY = sent_data.validation
    vaXt = tranform_sentiment_data(neuron, seq_data, vaX, os.path.join(sent_data_path, 'vaX.npy'))

    print("Transforming Test Sequences.")
    teX, teY = sent_data.test
    teXt = tranform_sentiment_data(neuron, seq_data, teX, os.path.join(sent_data_path, 'teX.npy'))

    # Running sentiment analysis
    print("Trainning sentiment classifier with transformed sequences.")
    full_rep_acc, c, n_not_zero, logreg_model = neuron.fit_sentiment(trXt, trY, vaXt, vaY, teXt, teY)

    print('%05.3f Test accuracy' % full_rep_acc)
    print('%05.3f Regularization coef' % c)
    print('%05d Features used' % n_not_zero)

    sentneuron_ixs = get_top_k_neuron_weights(logreg_model)

    plot_logits(results_path, trXt, np.array(trY), sentneuron_ixs)
    plot_weight_contribs_and_save(results_path, logreg_model.coef_)

    return sentneuron_ixs[0], logreg_model

def get_top_k_neuron_weights(logreg_model, k=5):
    weights = logreg_model.coef_.T
    weight_penalties = np.squeeze(np.linalg.norm(weights, ord=1, axis=1))

    if k == 1:
        k_indices = np.array([np.argmax(weight_penalties)])
    elif k >= np.log(len(weight_penalties)):
        k_indices = np.argsort(weight_penalties)[-k:][::-1]
    else:
        k_indices = np.argpartition(weight_penalties, -k)[-k:]
        k_indices = (k_indices[np.argsort(weight_penalties[k_indices])])[::-1]

    return k_indices

def get_neuron_values_for_a_sequence(neuron, seq_data, sequence, track_indices):
    _ ,tracked_indices_values = neuron.transform_sequence(seq_data, sequence, track_indices)
    return np.array([np.array(vals).flatten() for vals in tracked_indices_values])
