import argparse
import sentneuron as sn

parser = argparse.ArgumentParser(description='train_classifier.py')

parser.add_argument('-model_path'    , type=str, required=True, help="Model metadata path.")
parser.add_argument('-sent_data_path', type=str, required=True, help="Sentiment dataset path.")
parser.add_argument('-results_path'  , type=str, required=True, help="Path to save plots.")
parser.add_argument('-kfold'         , type=int,   default=None,  help="Use k-fold cross validation." )
opt = parser.parse_args()

neuron, seq_data = sn.utils.load_generative_model(opt.model_path)

if opt.kfold == None:
    neuron_ix, logreg_model = sn.utils.train_sentiment_analysis(neuron, seq_data, opt.sent_data_path, opt.results_path)
else:
    sn.utils.train_sentiment_analysis_k_fold(neuron, seq_data, opt.sent_data_path, opt.results_path, opt.kfold)

# # Sampling
# sample_pos = neuron.generate_sequence(seq_data, "This is ", 200, 0.8, override={neuron_ix : 4.0})
# sample_neg = neuron.generate_sequence(seq_data, "This is ", 200, 0.8, override={neuron_ix : -4.0})
#
# print(sample_pos)
# print(sample_neg)
#
# neuron_values = sn.utils.get_neuron_values_for_a_sequence(neuron, seq_data, sample_pos, [neuron_ix])[0]
# sn.utils.plot_heatmap(opt.results_path, sample_pos, neuron_values)
