import argparse
import sentneuron as sn

parser = argparse.ArgumentParser(description='train_classifier.py')

parser.add_argument('-model_path'      , type=str, required=True,  help="Model metadata path.")
parser.add_argument('-sent_data_path'  , type=str, required=True,  help="Sentiment dataset path.")
parser.add_argument('-results_path'    , type=str, required=True,  help="Path to save plots.")

parser.add_argument('-pad', dest='pad', action='store_true')
parser.add_argument('-no-pad', dest='pad', action='store_false')
parser.set_defaults(pad=False)

parser.add_argument('-balance', dest='balance', action='store_true')
parser.add_argument('-no-balance', dest='balance', action='store_false')
parser.set_defaults(balance=False)

parser.add_argument('-separate', dest='separate', action='store_true')
parser.add_argument('-no-separate', dest='separate', action='store_false')
parser.set_defaults(separate=False)

opt = parser.parse_args()

neuron, seq_data, _ , _ = sn.train.load_generative_model(opt.model_path)

# Load sentiment data from given path
sent_data = sn.dataloaders.SentimentMidi(opt.sent_data_path, "sentence", "label", "id", "filepath", opt.pad, opt.balance, opt.separate)
logreg_model = sn.train.train_unsupervised_classification_model(neuron, seq_data, sent_data)

# evolve weights to generate either positive or negative pieces
sn.evolve.evolve_weights(neuron, seq_data, opt.results_path)

dataset_name = opt.model_path.split("/")[-1]

gen_pieces = []
piece_init = "t_59 v_108 d_16th_0 n_33 v_108 d_16th_0 n_45 v_108 d_16th_0 n_61 v_108 d_16th_0 n_71 , v_108 d_16th_0 n_33 v_108 d_16th_0 n_45 v_108 d_16th_0 n_61 v_108 d_16th_0 n_71 , , v_108 d_16th_0 n_33 v_108 d_16th_0 n_45 v_108 d_16th_0 n_61 v_108 d_16th_0 n_71 , , v_108 d_16th_0 n_33 v_108 d_16th_0 n_45 v_108 d_16th_0 n_61 v_108 d_16th_0 n_67 , v_108 d_16th_0 n_33 v_108 d_16th_0 n_45 v_108 d_16th_0 n_61 v_108 d_16th_0 n_71 , , v_108 d_16th_0 n_38 v_108 d_16th_0 n_50 v_108 d_16th_0 n_62 v_108 d_16th_0 n_66 v_108 d_16th_0 n_74 , , , ,"

for i in range(30):
    ini_seq = seq_data.str2symbols(piece_init)
    gen_seq, final_cell = neuron.generate_sequence(seq_data, ini_seq, 128, 1.0)
    gen_pieces.append(final_cell)

    # Writing sampled sequence
    seq_data.write(gen_seq, opt.model_path + dataset_name + "_" + str(i))

guesses = neuron.predict_sentiment(seq_data, gen_pieces, transformed=True)
for i in range(len(guesses)):
    print("Gen piece", i, "sentiment: ", guesses[i])
