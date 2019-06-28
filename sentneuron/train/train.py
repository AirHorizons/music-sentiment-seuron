import os
import json

import numpy      as np
import sentneuron as sn

from sklearn.metrics import confusion_matrix

# Local imports
from ..dataloaders import *

def load_generative_model(model_path):
    with open(model_path + "_meta.json", 'r') as fp:
        meta = json.loads(fp.read())
        fp.close()

    with open(model_path + "_train.json", 'r') as fp:
        checkpoint = json.loads(fp.read())
        fp.close()

    # Load pre-calculated vocabulary
    seq_data = load_generative_data_with_type(meta["data_type"], None, meta["vocab"], meta["train_data"])

    # Model layer sizes
    input_size  = seq_data.encoding_size
    output_size = seq_data.encoding_size

    # Loading trainned model for predicting elements in a sequence.
    neuron = sn.SentimentNeuron(meta["input_size"], meta["embed_size"], meta["hidden_size"], meta["output_size"], meta["n_layers"], meta["dropout"])
    checkpoint["optimizer_state_dict"] = neuron.load(model_path + "_model.pth")

    return neuron, seq_data, meta["test_data"], checkpoint

def resume_generative_training(model_path, epochs=100, seq_length=256, lr=5e-4, grad_clip=5, batch_size=128):
    neuron, seq_data, test_data, checkpoint = load_generative_model(model_path)

    loss = neuron.fit_sequence(seq_data, test_data, epochs, seq_length, lr, grad_clip, batch_size, checkpoint)
    print("Testing loss:", loss)

    return neuron, seq_data

def train_generative_model(train_data, test_data, data_type, embed_size, hidden_size, n_layers=1, dropout=0, epochs=100, seq_length=256, lr=5e-4, grad_clip=5, batch_size=128):
    seq_data = load_generative_data_with_type(data_type, train_data)

    input_size  = seq_data.encoding_size
    output_size = seq_data.encoding_size

    # Training model for predicting elements in a sequence.
    neuron = sn.SentimentNeuron(input_size, embed_size, hidden_size, output_size, n_layers, dropout)

    loss = neuron.fit_sequence(seq_data, test_data, epochs, seq_length, lr, grad_clip, batch_size)
    print("Testing loss:", loss)

    return neuron, seq_data

def train_supervised_classification_model(seq_data_path, data_type, sent_data, embed_size, hidden_size, n_layers=1, dropout=0, epochs=100, lr=5e-4, batch_size=128):
    seq_data = load_generative_data_with_type(data_type, seq_data_path)

    input_size  = seq_data.encoding_size
    output_size = 1

    for train, test in sent_data.split:
        trX, trY, trNam = sent_data.unpack_fold(train)
        teX, teY, teNam = sent_data.unpack_fold(test)

        print("Trainning sentiment classifier.")
        neuron = sn.SentimentNeuron(input_size, embed_size, hidden_size, output_size, n_layers, dropout)
        score = neuron.fit_sentiment(seq_data, trX, trY, teX, teY, epochs, lr, batch_size)

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
        # print(trNam)
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

    y_pred = neuron.predict_sentiment(seq_data, teXt, transformed=True)

    for i in range(len(y_pred)):
        print(teNam[i], teY[i], y_pred[i])

    return acc

def tranform_sentiment_data(neuron, seq_data, xs, xs_filename):
    if(os.path.isfile(xs_filename)):
        xs = np.squeeze(np.load(xs_filename))
    else:
        for i in range(len(xs)):
            xs[i], _ = neuron.transform_sequence(seq_data, xs[i].split(" "))
        np.save(xs_filename, xs)

    return xs
