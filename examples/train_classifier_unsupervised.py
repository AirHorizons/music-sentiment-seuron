import argparse
import sentneuron as sn

parser = argparse.ArgumentParser(description='train_classifier.py')

parser.add_argument('-model_path'    , type=str, required=True, help="Model metadata path.")
parser.add_argument('-sent_data_path', type=str, required=True, help="Sentiment dataset path.")
parser.add_argument('-results_path'  , type=str, required=True, help="Path to save plots.")
opt = parser.parse_args()

neuron, seq_data = sn.utils.load_generative_model(opt.model_path)

# Load sentiment data from given path
sent_data = sn.dataloaders.SentimentMidi(opt.sent_data_path, "sentence", "label", "id", pad=False, balance=True)
logreg_model = sn.utils.train_unsupervised_classification_model(neuron, seq_data, sent_data)

# sn.utils.evolve_weights(neuron, seq_data, opt.results_path)

dataset_name = opt.model_path.split("/")[-1]
for i in range(30):
    ini_seq = seq_data.str2symbols(".")
    gen_seq = neuron.generate_sequence(seq_data, ini_seq, 128, 1.0)
    guess = neuron.predict_sentiment(seq_data, gen_seq)
    print("Gen piece", i, "sentiment: ", guess)


    # Writing sampled sequence
    seq_data.write(gen_seq, "../output/" + dataset_name + "_" + str(i))
