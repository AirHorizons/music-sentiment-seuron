import argparse
import sentneuron as sn

# Parse arguments
parser = argparse.ArgumentParser(description='generate_shards.py')
parser.add_argument('-datadir'  , type=str,   required=True, help="Directory with the data.")
parser.add_argument('-trainp'   , type=float, default=0.9,   help="Percentage of training data.")
parser.add_argument('-shards'   , type=int,   default=10,    help="Amount of shards.")
parser.add_argument('-data_type', type=str,   required=True,  help="Type of the training dataset: 'txt', 'midi_note', 'midi_chord' or 'midi_perform'." )

parser.add_argument('-valid', dest='valid', action='store_true')
parser.add_argument('-no-valid', dest='valid', action='store_false')
parser.set_defaults(valid=False)
opt = parser.parse_args()

# Load each line of each file as a sentence
data = sn.utils.load_data(opt.data_type, opt.datadir)

# Split these sentences into train and test sets
train, valid, test = sn.utils.split_data(data, opt.trainp, opt.valid)

# Generate train shards
sn.utils.generate_shards(train, shards_amount=opt.shards, shard_prefix="train", data_type="midi")

# Generate validation shards
if opt.valid:
    sn.utils.generate_shards(valid, shards_amount=1, shard_prefix="valid", data_type="midi")

# Generate test shards
sn.utils.generate_shards(test, shards_amount=1, shard_prefix="test", data_type="midi")
