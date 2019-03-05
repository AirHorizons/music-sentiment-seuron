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
        txtSequence = fl.request.form['txtSequence'].split(" ")
        genSequenceLen = int(fl.request.form['genSequenceLen'])
        return generate(txtSequence, genSequenceLen)

    return fl.render_template('index.html')

def generate(sample_init, sample_len):
    pass

# Load pre-trained model
neuron, seq_data = sn.utils.load_generative_model('../../trained_models/amazon_reviews')

app.run()
