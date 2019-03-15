import argparse
import sentneuron as sn

parser = argparse.ArgumentParser(description='train_classifier.py')

parser.add_argument('-model_path'    , type=str, required=True, help="Model metadata path.")
parser.add_argument('-sent_data_path', type=str, required=True, help="Sentiment dataset path.")
parser.add_argument('-results_path'  , type=str, required=True, help="Path to save plots.")
opt = parser.parse_args()

neuron, seq_data = sn.utils.load_generative_model(opt.model_path)
neuron_ix, logreg_model = sn.utils.train_sentiment_analysis(neuron, seq_data, opt.sent_data_path, opt.results_path)

# Sampling
sample = neuron.generate_sequence(seq_data, "This is ", 200, 0.4, override={neuron_ix : 1.0})
neuron_values = sn.utils.get_neuron_values_for_a_sequence(neuron, seq_data, sample, neuron_ix)

print(sample)
print(neuron_values)

sn.utils.plot_heatmap(opt.results_path, sample, neuron_values)
