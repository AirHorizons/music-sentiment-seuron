import argparse
import sentneuron as sn

# Parse arguments
parser = argparse.ArgumentParser(description='generate_shards.py')
parser.add_argument('-datadir' , type=str, required=True, help="Directory with the data.")
opt = parser.parse_args()

# Load each line of each file as a sentence
data = sn.utils.load_pieces(opt.datadir)

# Split these sentences into train and test sets
train, test = sn.utils.split_data(data)

# Generate train and test shards
sn.utils.generate_shards(train, shards_amount=10, shard_prefix="train", data_type="midi")
sn.utils.generate_shards(test, shards_amount=1, shard_prefix="test", data_type="midi")
