import argparse
import sentneuron as sn

parser = argparse.ArgumentParser(description='train_classifier.py')

parser.add_argument('-model_path'    , type=str, required=True, help="Model metadata path.")
parser.add_argument('-sent_data_path', type=str, required=True, help="Sentiment dataset path.")
parser.add_argument('-results_path'  , type=str, required=True, help="Path to save plots.")
opt = parser.parse_args()

neuron, seq_data = sn.utils.load_generative_model(opt.model_path)

# Load sentiment data from given path
sent_data = sn.dataloaders.SentimentMidi(opt.sent_data_path, "sentence", "label", "id")
sn.utils.train_unsupervised_classification_model(neuron, seq_data, sent_data, opt.results_path)

# Sampling
# sample_pos = neuron.generate_sequence(seq_data, "This is ", 200, 0.8, override={neuron_ix : 4.0})
# sample_neg = neuron.generate_sequence(seq_data, "This is ", 200, 0.8, override={neuron_ix : -4.0})
#
# neuron_values = sn.utils.get_neuron_values_for_a_sequence(neuron, seq_data, sample_pos, [neuron_ix])[0]
# sn.utils.plot_heatmap(opt.results_path, sample_pos, neuron_values)
