import flask      as fl
import sentneuron as sn
import music21    as m21

app = fl.Flask(__name__)

@app.route('/')
def index():
    return fl.render_template('index.html')

@app.route('/generate', methods=['POST', 'GET'])
def upload():
    if fl.request.method == 'POST':
        noteSequence = fl.request.form['noteSequence'].split(",")
        genSequenceLen = int(fl.request.form['genSequenceLen'])
        return generate(noteSequence, genSequenceLen)

    return fl.render_template('index.html')

def binarySearch(value, alist):
    first = 0
    last = len(alist)-1

    foundPos = -1;

    while first <= last and foundPos == -1:
        midpoint = (first + last)//2
        if value == "_".join(alist[midpoint].split("_")[:-1]):
            foundPos = midpoint;
        else:
            if value < "_".join(alist[midpoint].split("_")[:-1]):
                last = midpoint - 1
            else:
                if value > "_".join(alist[midpoint].split("_")[:-1]):
                    first = midpoint+1
    return foundPos

def findClosestNoteInVocab(chord, vocab):
    for note in chord.split(" "):
        noteWithoutVel = "_".join(note.split("_")[:-1])
        foundPos = binarySearch(noteWithoutVel, vocab)

        if noteWithoutVel != -1:
            print("found:", vocab[foundPos])
            return vocab[foundPos]

    return note

def generate(sample_init, sample_len):
    print(sample_init)

    # Replace duration type to number to easily play with tone.js
    for i in range(len(sample_init)):
        if sample_init[i][0] == "n":
            # Replace duration type to number to easily play with tone.js
            duration = sample_init[i].split("_")[2]
            try:
                d_type = m21.duration.convertQuarterLengthToType(float(duration)/4.)
            except:
                print("Can't convert duration to type:" + str(duration) + ". Assuming quarter note.")
                d_type = "quarter"

            sample_init[i] = sample_init[i].replace(duration, d_type, 1)
            sample_init[i] = findClosestNoteInVocab(sample_init[i], seq_data.vocab)

    print(sample_init)

    sample = neuron.sample(seq_data, sample_init=sample_init, sample_len=sample_len)
    seq_data.write(sample, "../../samples/beethoven_server")

    for i in range(len(sample)):
        if sample[i][0] == "n":
            # Replace duration type to number to easily play with tone.js
            duration = sample[i].split("_")[2]
            d_number = int(m21.duration.convertTypeToNumber(duration))
            sample[i] = sample[i].replace(duration, str(d_number), 1)

    return " ".join(sample)

# Load pre-calculated vocabulary
seq_data = sn.encoders.midi.EncoderMidiPerform("../../trained_models/beethoven_vocab.txt", pre_loaded=True)

# Model layer sizes
model_path = "../../trained_models/beethoven_model.pth"
neuron = sn.utils.load_generative_model(model_path, seq_data, embed_size=64, hidden_size=4096, n_layers=1, dropout=0)

app.run()
