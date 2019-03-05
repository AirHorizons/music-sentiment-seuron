import argparse
import sentneuron as sn

parser = argparse.ArgumentParser(description='train_generative_txt.py')

dIn = '../input/generative/txt/amazon_reviews'
parser.add_argument('-input'      , type=str,   default=dIn,  help="Training dataset."     )
parser.add_argument('-embed_size' , type=int,   default=64 ,  help="Embedding layer size." )
parser.add_argument('-hidden_size', type=int,   default=128,  help="Hidden layer size."    )
parser.add_argument('-n_layers'   , type=int,   default=1  ,  help="Number of LSTM layers.")
parser.add_argument('-dropout'    , type=int,   default=0  ,  help="Dropout probability."  )
parser.add_argument('-epochs'     , type=int,   default=100,  help="Training Epochs."      )
parser.add_argument('-seq_length' , type=int,   default=256,  help="Training Batch size."  )
parser.add_argument('-lr'         , type=float, default=5e-4, help="Learning Rate."        )
opt = parser.parse_args()

# Load text data
seq_data = sn.encoders.EncoderText(opt.input)
dataset_name = opt.input.split("/")[-1]

# Model layer sizes
neuron = sn.utils.train_generative_model(seq_data, opt.embed_size, opt.hidden_size, opt.n_layers, opt.dropout, opt.epochs, opt.seq_length, opt.lr)
neuron.save(seq_data, "../trained_models/" + dataset_name)

# Sampling
chars = "I don't know "
sample = neuron.sample(seq_data, sample_init=chars, sample_len=opt.seq_length)
seq_data.write(sample, "../samples/" + dataset_name)
