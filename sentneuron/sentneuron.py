# External imports
import json
import torch
import torch.nn       as nn
import torch.optim    as optim
import torch.autograd as ag
import numpy          as np

# Local imports
from .models import mLSTM
from sklearn.linear_model import LogisticRegression

class SentimentNeuron(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, output_size, n_layers=1, dropout=0):
        super(SentimentNeuron, self).__init__()
        # Save current training state to  resume it later if needed
        self.training_state = {
            "epoch": 0,
            "shard": 0,
            "batch": 0,
            "loss":  0,
            "optim": {}
         }

        # Set running device to "cpu" or "cuda" (if available)
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda")

        # Init layer sizes
        self.input_size  = input_size
        self.embed_size  = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Init number of LSTM layers
        self.n_layers = n_layers
        self.dropout  = dropout

        # Embedding layer
        self.i2h = nn.Embedding(input_size, embed_size)

        # Dropout layer
        self.drop = nn.Dropout(dropout)

        # Hidden to hidden layers
        self.h2h = []
        for i in range(n_layers):
            # Create a new mLSTM layer and add to the model
            h2h = mLSTM(embed_size, hidden_size)
            self.add_module('layer_%d' % i, h2h)
            self.h2h += [h2h]

            embed_size = hidden_size

        # Hidden to output layers
        self.h2y = nn.Linear(hidden_size, output_size)

        # Linear regression model that classifies sentiment
        self.sent_classfier = None

        # Set this model to run in the given device
        self.to(device=self.device)

    def init_hidden(self, batch_size=1):
        h = torch.zeros(self.n_layers, batch_size, self.hidden_size, device=self.device)
        c = torch.zeros(self.n_layers, batch_size, self.hidden_size, device=self.device)
        return (h, c)

    def forward(self, x, h):
        # First layer maps the input layer to the hidden layer
        emb_x = self.i2h(x)

        h_0, c_0 = h
        h_1, c_1 = [], []

        for i, h2h in enumerate(self.h2h):
            h_1_i, c_1_i = h2h(emb_x, (h_0[i], c_0[i]))

            if i == 0:
            	emb_x = h_1_i
            else:
            	emb_x = emb_x + h_1_i

            if i != len(self.h2h):
                emb_x = self.drop(emb_x)

            h_1 += [h_1_i]
            c_1 += [c_1_i]

        # Update hidden state
        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        y = self.h2y(emb_x)

        return (h_1, c_1), y

    def fit_sentiment(self, trX, trY, teX, teY, C=2**np.arange(-8, 1).astype(np.float), seed=42, penalty="l1"):
        scores = []

        # Hyper-parameter C optimization
        for i, c in enumerate(C):
            logreg_model = LogisticRegression(C=c, penalty=penalty, random_state=seed+i, solver="liblinear")
            logreg_model.fit(trX, trY)

            score = logreg_model.score(teX, teY)
            scores.append(score)

        c = C[np.argmax(scores)]

        self.sent_classfier = LogisticRegression(C=c, penalty=penalty, random_state=seed+len(C), solver="liblinear")
        self.sent_classfier.fit(trX, trY)

        score = self.sent_classfier.score(teX, teY) * 100.
        return score

    def predict_sentiment(self, seq_dataset, xs, transformed=False):
        with torch.no_grad():
            if self.sent_classfier == None:
                return None;

            sequences = []
            for x in xs:
                if not transformed:
                    x, _ = self.transform_sequence(seq_dataset, x.split(" "))
                sequences.append(x)

            return self.sent_classfier.predict(sequences)

    def evaluate(self, seq_dataset, batch_size, seq_length, test_shard_path):
        with torch.no_grad():
            # Loss function
            loss_function = nn.CrossEntropyLoss()

            h_init = self.init_hidden(batch_size)
            shard_content = seq_dataset.read(test_shard_path)

            sequence = seq_dataset.encode_sequence(shard_content)
            sequence = self.__batchify_sequence(torch.tensor(sequence, dtype=torch.uint8, device=self.device), batch_size)

            n_batches = sequence.size(0)//seq_length

            loss_avg = 0
            for batch_ix in range(n_batches - 1):
                batch = sequence.narrow(0, batch_ix * seq_length, seq_length + 1).long()

                h = h_init

                loss = 0
                for t in range(seq_length):
                    h, y = self(batch[t], h)
                    loss += loss_function(y, batch[t+1])

                h_init = (ag.Variable(h[0].data), ag.Variable(h[1].data))
                loss_avg += loss.item()/seq_length

            return loss_avg/n_batches

    def fit_sequence(self, seq_dataset, epochs=100, seq_length=100, lr=1e-3, lr_decay=1, grad_clip=5, batch_size=32, checkpoint=None):
        try:
            self.__fit_sequence(seq_dataset, epochs, seq_length, lr, lr_decay, grad_clip, batch_size, checkpoint)
        except KeyboardInterrupt:
            print('Exiting from training early.')

    def __fit_sequence(self, seq_dataset, epochs, seq_length, lr, lr_decay, grad_clip, batch_size, checkpoint):
        # Loss function
        loss_function = nn.CrossEntropyLoss()

        # Loss at epoch 0
        if checkpoint == None:
            epoch_lr    = lr
            epoch_in    = 0
            shard_in    = 0
            batch_in    = 0
            smooth_loss = -torch.log(torch.tensor(1.0/seq_dataset.encoding_size)).item() * seq_length
        else:
            epoch_in = checkpoint["epoch"]
            shard_in = checkpoint["shard"]
            batch_in = checkpoint["batch"]
            smooth_loss = checkpoint["loss"]
            epoch_lr = lr
            for shard in range(shard_in):
                epoch_lr *= lr_decay

        for epoch in range(epoch_in, epochs):
            # Iterate on each shard of the dataset
            for shard in range(shard_in, len(seq_dataset.data)):
                # Start optimizer with current learning rate
                optimizer = optim.Adam(self.parameters(), lr=epoch_lr)
                if checkpoint != None and shard_in == shard:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

                h_init = self.init_hidden(batch_size)

                # Use file pointer to read file content
                filepath, filename = seq_dataset.data[shard]
                shard_content = seq_dataset.read(filepath)

                # Batchify file content
                sequence = seq_dataset.encode_sequence(shard_content)
                sequence = self.__batchify_sequence(torch.tensor(sequence, dtype=torch.uint8, device=self.device), batch_size)

                n_batches = sequence.size(0)//seq_length

                # Each epoch consists of one entire pass over the dataset
                for batch_ix in range(batch_in, n_batches - 1):
                    # Reset optimizer grad
                    optimizer.zero_grad()

                    # Slice the dataset to create the current batch
                    batch = sequence.narrow(0, batch_ix * seq_length, seq_length + 1).long()

                    # Initialize hidden state with the hidden state from the previous batch
                    h = h_init

                    loss = 0
                    for t in range(seq_length):
                        # Run forward pass, get output y and calculate loss in respect to target batch[t+1]
                        h, y = self(batch[t], h)
                        loss += loss_function(y, batch[t+1])
                    loss.backward()

                    # Copy current hidden state to be next h_init
                    h_init = (ag.Variable(h[0].data), ag.Variable(h[1].data))

                    # Clip gradients
                    self.__clip_gradient(grad_clip)

                    # Run Stochastic Gradient Descent and Update weights
                    optimizer.step()
                    self.training_state["optim"] = optimizer.state_dict()

                    # Calculate average loss and log the results of this batch
                    smooth_loss = smooth_loss * 0.999 + loss.item() * 0.001
                    if batch_ix % 10 == 0:
                        self.__fit_sequence_log(epoch, (batch_ix, n_batches), smooth_loss, filename, seq_dataset, shard_content)

                    self.training_state["loss"] = smooth_loss
                    self.training_state["batch"] = batch_ix

                # Apply learning rate decay before the next shard
                epoch_lr *= lr_decay
                self.training_state["shard"] = shard

            self.training_state["epoch"] = epoch

    def __fit_sequence_log(self, epoch, batch_ix, loss, filename, seq_dataset, data, sample_init_range=(0, 20)):
        with torch.no_grad():
            i_init, i_end = sample_init_range
            sample_dat, _ = self.generate_sequence(seq_dataset, data[i_init:i_end], sample_len=200)

            print('epoch:', epoch)
            print('filename:', filename)
            print('batch: {}/{}'.format(batch_ix[0], batch_ix[1]))
            print('loss = ', loss)
            print('----\n' + str(sample_dat) + '\n----')

    def __batchify_sequence(self, sequence, batch_size=1):
        n_batch = sequence.size(0) // batch_size
        sequence = sequence.narrow(0, 0, n_batch * batch_size)
        sequence = sequence.view(batch_size, -1).t().contiguous()
        return sequence

    def __clip_gradient(self, clip):
        for p in self.parameters():
            p.grad.data = p.grad.data.clamp(-clip, clip)

    def generate_sequence(self, seq_dataset, sample_init, sample_len, temperature=1.0, override={}, append_init=True):
        with torch.no_grad():
            # Initialize the sequence
            seq = []

            # Create a new hidden state
            hidden_cell = self.init_hidden()

            xs = seq_dataset.encode_sequence(sample_init)
            batch = self.__batchify_sequence(torch.tensor(xs, dtype=torch.long, device=self.device))

            for t in range(batch.size(0)):
                hidden_cell, y = self.forward(batch[t], hidden_cell)
                x = batch[t].data[0].item()
                if append_init:
                    seq.append(x)

            for t in range(sample_len):
                # Override salient neurons
                hidden, cell = hidden_cell
                for neuron, value in override.items():
                    last = hidden.size(0) - 1
                    hidden[last,0,neuron] += value
                hidden_cell = (hidden, cell)

                x = torch.tensor([x], dtype=torch.long, device=self.device)
                hidden_cell, y = self.forward(x, hidden_cell)

                # Transform output into a probability distribution
                ps = torch.softmax(y[0].squeeze().div(temperature), dim=0)

                # Sample the next index according to the probability distribution ps
                x = torch.multinomial(ps, 1).item()

                # Append the index to the sequence
                seq.append(x)

            final_hidden, final_cell = hidden_cell
            trans_sequence = np.squeeze(cell.data.cpu().numpy())

            return seq_dataset.decode(seq), trans_sequence

    def transform_sequence(self, seq_dataset, sequence, track_indices=[]):
        with torch.no_grad():
            track_indices_values = [[] for i in range(len(track_indices))]

            # Create a new hidden state
            hidden_cell = self.init_hidden()

            xs = seq_dataset.encode_sequence(sequence)
            batch = self.__batchify_sequence(torch.tensor(xs, dtype=torch.long, device=self.device))

            for t in range(batch.size(0)):
                hidden_cell, y = self.forward(batch[t], hidden_cell)

                hidden, cell = hidden_cell
                trans_sequence = np.squeeze(cell.data.cpu().numpy())
                for i, index in enumerate(track_indices):
                    track_indices_values[i].append(trans_sequence[index])

            # Use cell state as feature vector fot the sentence
            final_hidden, final_cell = hidden_cell
            trans_sequence = np.squeeze(final_cell.data.cpu().numpy())

            return trans_sequence, track_indices_values

    def load(self, model_filename):
        print("Loading model:", model_filename)

        checkpoint = torch.load(model_filename, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.eval()

        return checkpoint['optimizer_state_dict']

    def save(self, seq_dataset, path=""):
        # Persist model on disk with current timestamp
        model_filename = path + "_model.pth"

        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.training_state["optim"],
        }, model_filename)

        self.training_state.pop("optim", None)

        train_filename = path + "_train.json"
        with open(train_filename, 'w') as fp:
            json.dump(self.training_state, fp)

        # Persist encoding vocab on disk
        meta_data = {}
        meta_filename = path + "_meta.json"
        with open(meta_filename, 'w') as fp:
            meta_data["data"] = seq_dataset.data
            meta_data["vocab"] = seq_dataset.vocab
            meta_data["data_type"]   = seq_dataset.type()
            meta_data["input_size"]  = self.input_size
            meta_data["embed_size"]  = self.embed_size
            meta_data["hidden_size"] = self.hidden_size
            meta_data["output_size"] = self.output_size
            meta_data["n_layers"]    = self.n_layers
            meta_data["dropout"]     = self.dropout
            json.dump(meta_data, fp)

        print("Saved model:", model_filename)

    def get_top_k_neuron_weights(self, k=1):
        weights = self.sent_classfier.coef_.T
        weight_penalties = np.squeeze(np.linalg.norm(weights, ord=1, axis=1))

        if k == 1:
            k_indices = np.array([np.argmax(weight_penalties)])
        elif k >= np.log(len(weight_penalties)):
            k_indices = np.argsort(weight_penalties)[-k:][::-1]
        else:
            k_indices = np.argpartition(weight_penalties, -k)[-k:]
            k_indices = (k_indices[np.argsort(weight_penalties[k_indices])])[::-1]

        return k_indices

    def get_neuron_values_for_a_sequence(self, seq_data, sequence, track_indices):
        _ ,tracked_indices_values = self.transform_sequence(seq_data, sequence, track_indices)
        return np.array([np.array(vals).flatten() for vals in tracked_indices_values])
