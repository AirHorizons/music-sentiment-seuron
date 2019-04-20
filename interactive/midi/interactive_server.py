import json
import argparse
import flask      as fl
import sentneuron as sn
import music21    as m21

from flask_socketio import SocketIO

NOTE_SEQUENCE_LENGTH = 256

app = fl.Flask(__name__)
socketio = SocketIO(app)

hidden_cell = None

def transform_durations(sample):
    sample = sample.split(" ")

    for i in range(len(sample)):
        if sample[i] != "" and sample[i][0] == "d":
            # Replace duration type to number to easily play with tone.js
            duration = sample[i].split("_")[1]
            d_number = int(m21.duration.convertTypeToNumber(duration))
            sample[i] = sample[i].replace("d_" + duration, "d_" + str(d_number))

    return " ".join(sample)

@app.route('/')
def index():
    return fl.render_template('index.html')

@socketio.on('generate')
def handle_my_custom_event(data):
    global hidden_cell

    # Parse init sequence
    init = seq_data.str2symbols(data["init"])
    init = list(filter(('').__ne__, init))

    override = {}
    if data["sentiment"] == "positive":
        override = pos_weights
    elif data["sentiment"] == "negative":
        override = neg_weights

    print(data["sentiment"])
    print(override)

    # Initialize LSTM hidden states
    sample = neuron.generate_sequence(seq_data, init, NOTE_SEQUENCE_LENGTH, opt.temp, override=override, append_init=False)

    if data["first"]:
        sample = init + sample.split(" ")
        sample = " ".join(sample)

    # Emit notes back to client
    socketio.emit('notes', (sample, transform_durations(sample), data["sentiment"]))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='generate_sequence.py')

    parser.add_argument('-model_path' , type=str,   required=True, help="Model metadata path."      )
    parser.add_argument('-temp'       , type=float, default=1.0,   help="Temperature for sampling." )
    parser.add_argument('-neg_weights', type=str,   default="" ,   help="Numpy array file path to override neurons and generate negative content." )
    parser.add_argument('-pos_weights', type=str,   default="" ,   help="Numpy array file path to override neurons and generate positive content." )
    opt = parser.parse_args()

    # Load generative LSTM
    neuron, seq_data = sn.utils.load_generative_model(opt.model_path)

    # Override given neurons
    neg_weights = {}
    if opt.neg_weights != "":
        neg_weights = json.loads(open(opt.neg_weights).read())
        neg_weights = {int(k):v for k,v in neg_weights.items()}

    pos_weights = {}
    if opt.pos_weights != "":
        pos_weights = json.loads(open(opt.pos_weights).read())
        pos_weights = {int(k):v for k,v in pos_weights.items()}

    # Start app
    socketio.run(app)
