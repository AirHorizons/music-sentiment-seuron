import argparse
import sentneuron as sn

parser = argparse.ArgumentParser(description='train_classifier.py')

parser.add_argument('-model_path'    , type=str, required=True, help="Model metadata path.")
parser.add_argument('-sent_data_path', type=str, required=True, help="Sentiment dataset path.")
opt = parser.parse_args()

neuron, seq_data = sn.utils.load_generative_model(opt.model_path)
sn.utils.train_sentiment_analysis(neuron, seq_data, opt.sent_data_path)
