import argparse
import sentneuron as sn

parser = argparse.ArgumentParser(description='train_generative_midi.py')

dIn = '../trained_models/beethoven_mond'
parser.add_argument('-input'      , type=str,   default=dIn,  help="Model metadata path."                  )
parser.add_argument('-seq_init'   , type=str,   default=".",  help="Init of the sequence to be generated." )
parser.add_argument('-seq_length' , type=int,   default=256,  help="Size of the sequence to be generated." )
opt = parser.parse_args()

# Model layer sizes
neuron, seq_data = sn.utils.load_generative_model(opt.input)
dataset_name = opt.input.split("/")[-1]

# Sampling
sample = neuron.sample(seq_data, sample_init=opt.seq_init, sample_len=opt.seq_length)
seq_data.write(sample, "../samples/" + dataset_name)
