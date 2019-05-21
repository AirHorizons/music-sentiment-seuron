import json
import argparse
import random
import sentneuron as sn
import os

parser = argparse.ArgumentParser(description='generate_sequence.py')

parser.add_argument('-model_path' , type=str,   required=True, help="Model metadata path."                  )
parser.add_argument('-seq_init'   , type=str,   default="\n",   help="Init of the sequence to be generated." )
parser.add_argument('-seq_length' , type=int,   default=256,   help="Size of the sequence to be generated." )
parser.add_argument('-temp'       , type=float, default=1.0,   help="Temperature for sampling." )
parser.add_argument('-override'   , type=str,   default="" ,   help="Numpy array file path to override neurons." )
parser.add_argument('-n'          , type=int,   default=1 ,    help="Amount of sequences to generate." )
opt = parser.parse_args()

# Load generative model
neuron, seq_data, _ , _ = sn.utils.load_generative_model(opt.model_path)

# Set initial sequence
init = seq_data.str2symbols(opt.seq_init)

override = {}
if opt.override != "":
    override = json.loads(open(opt.override).read())
    override = {int(k):v for k,v in override.items()}

for i in range(opt.n):
    sample, _ = neuron.generate_sequence(seq_data, init, opt.seq_length, opt.temp, override=override)
    seq_data.write(sample, "../output/" + os.path.basename(opt.model_path) + "_" + str(i))
