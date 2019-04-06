import argparse
import sentneuron as sn

parser = argparse.ArgumentParser(description='train_classifier.py')

parser.add_argument('-seq_data_path'  , type=str,   required=True,  help="Training dataset."            )
parser.add_argument('-seq_data_type'  , type=str,   required=True,  help="Type of the training dataset: 'txt', 'midi_note', 'midi_chord' or 'midi_perform'." )
parser.add_argument('-sent_data_path' , type=str,   required=True,  help="Sentiment dataset path."      )
parser.add_argument('-embed_size'     , type=int,   default=64   ,  help="Embedding layer size."        )
parser.add_argument('-hidden_size'    , type=int,   default=128  ,  help="Hidden layer size."           )
parser.add_argument('-n_layers'       , type=int,   default=1    ,  help="Number of LSTM layers."       )
parser.add_argument('-dropout'        , type=int,   default=0    ,  help="Dropout probability."         )
parser.add_argument('-epochs'         , type=int,   default=100  ,  help="Training epochs."            )
parser.add_argument('-kfold'          , type=int,   default=10   ,  help="Use k-fold cross validation." )
parser.add_argument('-lr'             , type=float, default=5e-4 ,  help="Learning rate."          )
parser.add_argument('-lr_decay'       , type=float, default=1    ,  help="Learning rate detay."    )
parser.add_argument('-batch_size'     , type=int,   default=128  ,  help="Batch size."             )
opt = parser.parse_args()

# Load sentiment data from given path
sent_data = sn.dataloaders.SentimentMidi(opt.sent_data_path, "sentence", "label", "id")

sn.utils.train_supervised_classification_model(opt.seq_data_path, opt.seq_data_type, sent_data, \
                                              opt.embed_size, opt.hidden_size, opt.n_layers,   \
                                              opt.dropout, opt.epochs, opt.lr, opt.lr_decay,   \
                                              opt.batch_size)

dataset_name = opt.sent_data_path.split("/")[-1]
neuron.save(seq_data, "../trained_models/" + dataset_name)
