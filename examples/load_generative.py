import argparse
import sentneuron as sn

parser = argparse.ArgumentParser(description='train_generative_midi.py')

parser.add_argument('-model_path' , type=str,   required=True, help="Model metadata path."                  )
parser.add_argument('-seq_init'   , type=str,   default=".",   help="Init of the sequence to be generated." )
parser.add_argument('-seq_length' , type=int,   default=256,   help="Size of the sequence to be generated." )
parser.add_argument('-temp'       , type=float, default=0.4,   help="Temperature for sampling." )
opt = parser.parse_args()

# Model layer sizes
neuron, seq_data = sn.utils.load_generative_model(opt.model_path)

# Sampling
sample = neuron.sample(seq_data, opt.seq_init.split(" "), opt.seq_length, opt.temp)

# Writing sampled sequence
dataset_name = opt.model_path.split("/")[-1]
seq_data.write(sample, "../samples/" + dataset_name)
