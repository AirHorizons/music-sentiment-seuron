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
        noteSequence = fl.request.form['noteSequence'].split(" ")
        genSequenceLen = int(fl.request.form['genSequenceLen'])
        return generate(noteSequence, genSequenceLen)

    return fl.render_template('index.html')

def binarySearch(target, alist):
    first = 0
    last = len(alist)-1

    foundPos = -1;

    while first <= last and foundPos == -1:
        midpoint = (first + last)//2

        value = "_".join(alist[midpoint].split("_")[:-1])
        if target == value:
            foundPos = midpoint;
        else:
            if target < value:
                last = midpoint - 1
            else:
                if target > value:
                    first = midpoint+1
    return foundPos

def findClosestSymbolInVocab(note, vocab):
    velocity = int(note.split("_")[-1])
    noteWithoutVel = "_".join(note.split("_")[:-1])

    foundPos = binarySearch(noteWithoutVel, vocab)
    if foundPos != -1:
        i = foundPos

        possibleVels = []
        while "_".join(vocab[i].split("_")[:-1]) == noteWithoutVel:
            possibleVels.append(int(vocab[i].split("_")[-1]))
            i += 1

        clossestVel = min(possibleVels, key=lambda x:abs(x - velocity))
        return noteWithoutVel + "_" + str(clossestVel)

    return note

def generate(sample_init, sample_len):
    print(sample_init)

    # Replace duration type to number to easily play with tone.js
    for i in range(len(sample_init)):
        if sample_init[i][0] == "d":
            # Replace duration type to number to easily play with tone.js
            duration = sample_init[i].split("_")[1]
            try:
                d_type = m21.duration.convertQuarterLengthToType(4./float(duration))
            except:
                print("Can't convert duration to type:" + str(duration) + ". Assuming quarter note.")
                d_type = "quarter"

            sample_init[i] = sample_init[i].replace("d_" + duration, "d_" + d_type)
            # sample_init[i] = findClosestSymbolInVocab(sample_init[i], seq_data.vocab)

    print(sample_init)

    sample = neuron.sample(seq_data, sample_init=sample_init, sample_len=sample_len)
    seq_data.write(sample, "../../samples/beethoven_mond")

    for i in range(len(sample)):
        if sample[i][0] == "d":
            # Replace duration type to number to easily play with tone.js
            duration = sample[i].split("_")[1]
            d_number = int(m21.duration.convertTypeToNumber(duration))
            sample[i] = sample[i].replace("d_" + duration, "d_" + str(d_number))

    return " ".join(sample)

# Load pre-trained model
neuron, seq_data = sn.utils.load_generative_model('../../trained_models/beethoven_mond')

app.run()
