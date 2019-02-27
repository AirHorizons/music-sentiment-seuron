from flask import *

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        noteSequence = request.form['noteSequence']
        return generate(noteSequence)

    return render_template('index.html')

def generate(sample_init):
    return sample_init

app.run()
