import os
import json

import numpy      as np
import sentneuron as sn

# Local imports
from .plot import *
from .ga   import GeneticAlgorithm

def create_data_with_type(seq_data_path, data_type, pre_loaded):
    seq_data = None

    if data_type == "txt":
        seq_data = sn.encoders.EncoderText(seq_data_path, pre_loaded)
    elif data_type == "midi_note":
        seq_data = sn.encoders.midi.EncoderMidiNote(seq_data_path, pre_loaded)
    elif data_type == "midi_chord":
        seq_data = sn.encoders.midi.EncoderMidiChord(seq_data_path, pre_loaded)
    elif data_type == "midi_perform":
        seq_data = sn.encoders.midi.EncoderMidiPerform(seq_data_path, pre_loaded)

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

def train_generative_model(seq_data_path, data_type, embed_size, hidden_size, n_layers=1, dropout=0, epochs=100, seq_length=256, lr=5e-4, lr_decay=0.7, grad_clip=5, batch_size=128):
    seq_data = create_data_with_type(seq_data_path, data_type, pre_loaded=False)

    input_size  = seq_data.encoding_size
    output_size = seq_data.encoding_size

    # Training model for predicting elements in a sequence.
    neuron = sn.SentimentNeuron(input_size, embed_size, hidden_size, output_size, n_layers, dropout)
    neuron.fit_sequence(seq_data, epochs, seq_length, lr, lr_decay, grad_clip, batch_size)

    return neuron, seq_data

def train_supervised_classification_model(seq_data_path, data_type, sent_data, embed_size, hidden_size, n_layers=1, dropout=0, epochs=100, lr=5e-4, lr_decay=0.7, batch_size=128):
    seq_data = create_data_with_type(seq_data_path, data_type, pre_loaded=False)

    input_size  = seq_data.encoding_size
    output_size = 1

    for train, test in sent_data.split:
        trX, trY = sent_data.unpack_fold(train)
        teX, teY = sent_data.unpack_fold(test)

        print("Trainning sentiment classifier.")
        neuron = sn.SentimentLSTM(input_size, embed_size, hidden_size, output_size, n_layers, dropout)
        score = neuron.fit_sentiment(seq_data, trX, trY, teX, teY, epochs, lr, lr_decay, batch_size)

        print('%05.3f Test accuracy' % score)

def train_unsupervised_classification_model(neuron, seq_data, sent_data, results_path):
    test_ix = 0

    accs = []

    for train, test in sent_data.split:
        print("-> Test", test_ix)
        trX, trY = sent_data.unpack_fold(train)
        teX, teY = sent_data.unpack_fold(test)

        sent_data_dir = "/".join(sent_data.data_path.split("/")[:-1])

        print("Transforming Trainning Sequences.")
        trXt = tranform_sentiment_data(neuron, seq_data, trX, os.path.join(sent_data_dir, 'trX_' + str(test_ix) + '.npy'))

        print("Transforming Test Sequences.")
        teXt = tranform_sentiment_data(neuron, seq_data, teX, os.path.join(sent_data_dir, 'teX_' + str(test_ix) + '.npy'))

        # Running sentiment analysis
        print("Trainning sentiment classifier with transformed sequences.")
        acc, c, n_not_zero, logreg_model = neuron.fit_sentiment(trXt, trY, teXt, teY)

        accs.append(acc)
        test_ix += 1

    test_ix = 0
    for train, test in sent_data.split:
        if test_ix == np.argmax(accs):
            trX, trY = sent_data.unpack_fold(train)
            teX, teY = sent_data.unpack_fold(test)

            sent_data_dir = "/".join(sent_data.data_path.split("/")[:-1])

            trXt = tranform_sentiment_data(neuron, seq_data, trX, os.path.join(sent_data_dir, 'trX_' + str(test_ix) + '.npy'))
            teXt = tranform_sentiment_data(neuron, seq_data, teX, os.path.join(sent_data_dir, 'teX_' + str(test_ix) + '.npy'))
            acc, c, n_not_zero, logreg_model = neuron.fit_sentiment(trXt, trY, teXt, teY)

            print('Test accuracy', acc)
            print('Regularization coef', c)
            print('Features used', len(n_not_zero))

            break

        test_ix += 1

    sentneuron_ixs = get_top_k_neuron_weights(logreg_model)
    print(sentneuron_ixs)

    plot_logits(results_path, trXt, np.array(trY), sentneuron_ixs, fold="fold_")
    plot_weight_contribs_and_save(results_path, logreg_model.coef_, fold="fold_")

    genAlg = GeneticAlgorithm(neuron, sentneuron_ixs[0], seq_data, logreg_model)
    genAlg.evolve()

def tranform_sentiment_data(neuron, seq_data, xs, xs_filename):
    if(os.path.isfile(xs_filename)):
        xs = np.squeeze(np.load(xs_filename))
    else:
        for i in range(len(xs)):
            print(i, xs[i])
            xs[i], _ = neuron.transform_sequence(seq_data, xs[i].split(" "))
            print(i, len(xs[i]))
        np.save(xs_filename, xs)

    return xs

def get_top_k_neuron_weights(logreg_model, k=1):
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
