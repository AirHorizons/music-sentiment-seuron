import os
import argparse
import sentneuron as sn

parser = argparse.ArgumentParser(description='evaluate_generative_models.py'                 )
parser.add_argument('-models_path' , type=str,   required=True,  help="Models path."         )
parser.add_argument('-test_data'   , type=str,   required=True,  help="Test shard."          )
parser.add_argument('-seq_length'  , type=int,   default=256  ,  help="Training batch size." )
parser.add_argument('-batch_size'  , type=int,   default=128  ,  help="Batch size."          )
opt = parser.parse_args()

# Read every file in the given directory
for file in os.listdir(opt.models_path):
    modelfile = os.path.join(opt.models_path, file)

    if os.path.isfile(modelfile) and modelfile[-4:] == ".pth":
        # Load model meta data
        metafile = modelfile.split("_model.pth")[0] + "_meta.json"

        if os.path.isfile(metafile):
            model_path = metafile.split("_meta.json")[0]
            neuron, seq_data, _ , _ = sn.train.load_generative_model(model_path)
            loss = neuron.evaluate_sequence_fit(seq_data, opt.batch_size, opt.seq_length, opt.test_data)
            print(model_path, loss)
