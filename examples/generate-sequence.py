import json
import argparse
import sentneuron as sn

parser = argparse.ArgumentParser(description='generate_sequence.py')

parser.add_argument('-model_path' , type=str,   required=True, help="Model metadata path."                  )
parser.add_argument('-seq_init'   , type=str,   default=".",   help="Init of the sequence to be generated." )
parser.add_argument('-seq_length' , type=int,   default=256,   help="Size of the sequence to be generated." )
parser.add_argument('-temp'       , type=float, default=1.0,   help="Temperature for sampling." )
parser.add_argument('-override'   , type=str,   default="" ,   help="Numpy array file path to override neurons." )
opt = parser.parse_args()

# Model layer sizes
neuron, seq_data = sn.utils.load_generative_model(opt.model_path)

# Sampling
init = seq_data.str2symbols(opt.seq_init)

override = {}
if opt.override != "":
    override = json.loads(open(opt.override).read())
    override = {int(k):v for k,v in override.items()}
    print(override)

sample = neuron.generate_sequence(seq_data, init, opt.seq_length, opt.temp, override=override)

# Writing sampled sequence
dataset_name = opt.model_path.split("/")[-1]
seq_data.write(sample, "../output/" + dataset_name)
