import argparse
import sentneuron as sn

parser = argparse.ArgumentParser(description='train_generative_midi.py')

dIn = '../input/generative/midi/beethoven_mond'
parser.add_argument('-input'      , type=str,   default=dIn,  help="Training dataset."     )
parser.add_argument('-embed_size' , type=int,   default=64 ,  help="Embedding layer size." )
parser.add_argument('-hidden_size', type=int,   default=128,  help="Hidden layer size."    )
parser.add_argument('-n_layers'   , type=int,   default=1  ,  help="Number of LSTM layers.")
parser.add_argument('-dropout'    , type=int,   default=0  ,  help="Dropout probability."  )
parser.add_argument('-epochs'     , type=int,   default=100,  help="Training Epochs."      )
parser.add_argument('-seq_length' , type=int,   default=256,  help="Training Batch size."  )
parser.add_argument('-lr'         , type=float, default=5e-4, help="Learning Rate."        )
opt = parser.parse_args()

# Load midi data
seq_data = sn.encoders.midi.EncoderMidiPerform(opt.input)
dataset_name = opt.input.split("/")[-1]

# Model layer sizes
neuron = sn.utils.train_generative_model(seq_data, opt.embed_size, opt.hidden_size, opt.n_layers, opt.dropout, opt.epochs, opt.seq_length, opt.lr)
neuron.save(seq_data, "../trained_models/" + dataset_name)

# Sampling
notes = ["v_80", "t_40", "d_quarter", "n_60", ".", "n_62", "."]
sample = neuron.sample(seq_data, sample_init=notes, sample_len=opt.seq_length)
seq_data.write(sample, "../samples/" + dataset_name)
