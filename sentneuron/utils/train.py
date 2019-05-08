import os
import json

import numpy      as np
import sentneuron as sn

# Local imports
from .plot import *
from .ga   import GeneticAlgorithm

from sklearn.metrics import confusion_matrix

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

def train_unsupervised_classification_model(neuron, seq_data, sent_data):
    test_ix = 0

    accuracy = []

    data_split = list(sent_data.split)
    for train, test in data_split:
        print("-> Test", test_ix)
        trX, trY, trNam = sent_data.unpack_fold(train)
        teX, teY, teNam = sent_data.unpack_fold(test)

        sent_data_dir = "/".join(sent_data.data_path.split("/")[:-1])

        print("Transforming Trainning Sequences.")
        trXt = tranform_sentiment_data(neuron, seq_data, trX, os.path.join(sent_data_dir, 'trX_' + str(test_ix) + '.npy'))

        print("Transforming Test Sequences.")
        teXt = tranform_sentiment_data(neuron, seq_data, teX, os.path.join(sent_data_dir, 'teX_' + str(test_ix) + '.npy'))

        posLabelsTr = len(np.where(np.array(trY) == 1)[0])
        negLabelsTr = len(np.where(np.array(trY) == 0)[0])
        posLabelsTe = len(np.where(np.array(teY) == 1)[0])
        negLabelsTe = len(np.where(np.array(teY) == 0)[0])

        print("Total positive examples training/test:", posLabelsTr, posLabelsTe)
        print("Total negative examples training/test:", negLabelsTr, negLabelsTe)

        # Running sentiment analysis
        print("Trainning sentiment classifier with transformed sequences.")
        acc = neuron.fit_sentiment(trXt, trY, teXt, teY)

        y_true = teY
        y_pred = neuron.predict_sentiment(seq_data, teXt, transformed=True)
        print("Confusion Matrix")
        print(confusion_matrix(y_true, y_pred))

        print('Test accuracy', acc)
        accuracy.append(acc)
        test_ix += 1

    best_test_ix = np.argmax(accuracy)
    print("---> Best Test:", best_test_ix)

    train, test = data_split[best_test_ix]
    trX, trY, trNam = sent_data.unpack_fold(train)
    teX, teY, teNam = sent_data.unpack_fold(test)

    sent_data_dir = "/".join(sent_data.data_path.split("/")[:-1])

    trXt = tranform_sentiment_data(neuron, seq_data, trX, os.path.join(sent_data_dir, 'trX_' + str(best_test_ix) + '.npy'))
    teXt = tranform_sentiment_data(neuron, seq_data, teX, os.path.join(sent_data_dir, 'teX_' + str(best_test_ix) + '.npy'))
    acc = neuron.fit_sentiment(trXt, trY, teXt, teY)
    print("---> Best Test accuracy:", acc)

    return acc

def tranform_sentiment_data(neuron, seq_data, xs, xs_filename):
    if(os.path.isfile(xs_filename)):
        xs = np.squeeze(np.load(xs_filename))
    else:
        for i in range(len(xs)):
            xs[i], _ = neuron.transform_sequence(seq_data, xs[i].split(" "))
        np.save(xs_filename, xs)

    return xs


def evolve_weights(neuron, seq_data, results_path):
    n_not_zero = len(np.argwhere(neuron.sent_classfier.coef_))
    sentneuron_ixs = neuron.get_top_k_neuron_weights(k=n_not_zero)
    print(sentneuron_ixs)

    # plot_logits(results_path, trXt, np.array(trY), sentneuron_ixs, fold="fold_")
    plot_weight_contribs_and_save(results_path, neuron.sent_classfier.coef_, fold="fold_")

    genAlg = GeneticAlgorithm(neuron, sentneuron_ixs, seq_data, ofInterest=0)
    best_ind, best_fit = genAlg.evolve()

    override = {}
    for i in range(len(sentneuron_ixs)):
        override[int(sentneuron_ixs[i])] = best_ind[i]

    print(override)
    with open('../output/ga_best.json', 'w') as fp:
        json.dump(override, fp)
