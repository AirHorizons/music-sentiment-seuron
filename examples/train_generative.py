import os
import argparse
import sentneuron as sn

parser = argparse.ArgumentParser(description='train_generative.py')

parser.add_argument('-train_data' , type=str,   required=True,  help="Training dataset."        )
parser.add_argument('-test_data'  , type=str,   required=True,  help="Test dataset."            )
parser.add_argument('-data_type'  , type=str,   required=True,  help="Type of the training dataset: 'txt', 'midi_note', 'midi_chord' or 'midi_perform'." )
parser.add_argument('-embed_size' , type=int,   default=64   ,  help="Embedding layer size."    )
parser.add_argument('-hidden_size', type=int,   default=4096 ,  help="Hidden layer size."       )
parser.add_argument('-n_layers'   , type=int,   default=1    ,  help="Number of LSTM layers."   )
parser.add_argument('-dropout'    , type=float, default=0    ,  help="Dropout probability."     )
parser.add_argument('-epochs'     , type=int,   default=1    ,  help="Training epochs."         )
parser.add_argument('-seq_length' , type=int,   default=256  ,  help="Training batch size."     )
parser.add_argument('-lr'         , type=float, default=5e-6 ,  help="Learning rate."           )
parser.add_argument('-grad_clip'  , type=int,   default=5    ,  help="Gradiant clipping value." )
parser.add_argument('-batch_size' , type=int,   default=32   ,  help="Batch size."              )
parser.add_argument('-model_path' , type=str,   default=""   ,  help="Model to resume training.")
opt = parser.parse_args()

# Train a generative model to predict characters in a sequence
if opt.model_path == "":
    neuron, seq_data = sn.train.train_generative_model(opt.train_data, opt.test_data, opt.data_type,    \
                                                       opt.embed_size, opt.hidden_size, opt.n_layers,   \
                                                       opt.dropout, opt.epochs, opt.seq_length, opt.lr, \
                                                       opt.grad_clip, opt.batch_size)
else:
    neuron, seq_data = sn.train.resume_generative_training(opt.model_path, opt.epochs, opt.seq_length, \
                                                           opt.lr, opt.grad_clip, opt.batch_size)
