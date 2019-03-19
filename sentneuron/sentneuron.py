# External imports
import json
import torch
import torch.nn       as nn
import torch.optim    as optim
import torch.autograd as ag
import numpy          as np

from sklearn.linear_model import LogisticRegression

# Local imports
from .models import mLSTM

class SentimentNeuron(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, output_size, n_layers=1, dropout=0):
        super(SentimentNeuron, self).__init__()

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

        # Set this model to run in the given device
        self.to(device=self.device)

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

    def fit_sequence(self, seq_dataset, epochs=100, seq_length=100, lr=1e-3, lr_decay=1, grad_clip=5, batch_size=32):
        try:
            self.__fit_sequence(seq_dataset, epochs, seq_length, lr, lr_decay, grad_clip, batch_size)
        except KeyboardInterrupt:
            print('Exiting from training early.')

    def __fit_sequence(self, seq_dataset, epochs, seq_length, lr, lr_decay, grad_clip, batch_size):
        # Loss function
        loss_function = nn.CrossEntropyLoss()

        # Optimizer
        optimizer = optim.Adam(self.parameters(), lr=lr)

        # Loss at epoch 0
        smooth_loss = -torch.log(torch.tensor(1.0/seq_dataset.encoding_size)).item() * seq_length

        for epoch in range(epochs):
            # Iterate on each shard of the dataset
            for shard in seq_dataset.data:
                h_init = self.__init_hidden(batch_size)

                # Use file pointer to read file content
                fp, filename = shard
                shard_content = seq_dataset.read(fp)

                # Batchify file content
                sequence = seq_dataset.encode_sequence(shard_content)
                sequence = self.__batchify_sequence(torch.tensor(sequence, dtype=torch.uint8, device=self.device), batch_size)

                n_batches = sequence.size(0)//seq_length

                # Each epoch consists of one entire pass over the dataset
                for batch_ix in range(n_batches - 1):
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

                    # Calculate average loss and log the results of this batch
                    smooth_loss = smooth_loss * 0.999 + loss.item() * 0.001
                    if batch_ix % 10 == 0:
                        self.__fit_sequence_log(epoch, (batch_ix, n_batches), smooth_loss, filename, seq_dataset, shard_content)

            # Apply learning rate decay before the next epoch
            lr *= lr_decay

    def __fit_sequence_log(self, epoch, batch_ix, loss, filename, seq_dataset, data, sample_init_range=(0, 20)):
        with torch.no_grad():
            i_init, i_end = sample_init_range
            sample_dat = self.generate_sequence(seq_dataset, data[i_init:i_end], sample_len=200)

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

    def __init_hidden(self, batch_size=1):
        h = torch.zeros(self.n_layers, batch_size, self.hidden_size, device=self.device)
        c = torch.zeros(self.n_layers, batch_size, self.hidden_size, device=self.device)
        return (h, c)

    def __clip_gradient(self, clip):
        totalnorm = 0
        for p in self.parameters():
            p.grad.data = p.grad.data.clamp(-clip, clip)

    def fit_sentiment(self, trX, trY, vaX, vaY, teX, teY, C=2**np.arange(-8, 1).astype(np.float), seed=42, penalty="l1"):
        with torch.no_grad():
            scores = []
            for i, c in enumerate(C):
                logreg_model = LogisticRegression(C=c, penalty=penalty, random_state=seed+i, solver="liblinear")
                logreg_model.fit(trX, trY)

                score = logreg_model.score(vaX, vaY)
                scores.append(score)

            c = C[np.argmax(scores)]

            logreg_model = LogisticRegression(C=c, penalty=penalty, random_state=seed+len(C), solver="liblinear")
            logreg_model.fit(trX, trY)
            score = logreg_model.score(teX, teY) * 100.

            n_not_zero = np.sum(logreg_model.coef_ != 0.)
            return score, c, n_not_zero, logreg_model

    def generate_sequence(self, seq_dataset, sample_init, sample_len, temperature=0.4, override={}):
        with torch.no_grad():
            # Initialize the sequence
            seq = []

            # Create a new hidden state
            hidden_cell = self.__init_hidden()

            xs = seq_dataset.encode_sequence(sample_init)
            xs = self.__batchify_sequence(torch.LongTensor(xs))

            batch = ag.Variable(xs)

            for t in range(xs.size(0)):
                hidden_cell, y = self.forward(batch[t], hidden_cell)
                x = batch[t].data[0].item()
                seq.append(x)

            for t in range(sample_len):
                # Override salient neurons
                hidden, cell = hidden_cell
                for neuron, value in override.items():
                    hidden[:, :, neuron] = value
                hidden_cell = (hidden, cell)

                x = ag.Variable(torch.LongTensor([x]))
                hidden_cell, y = self.forward(x, hidden_cell)

                # Transform output into a probability distribution
                ps = torch.softmax(y[0].squeeze().div(temperature), dim=0)

                # Sample the next index according to the probability distribution ps
                x = torch.multinomial(ps, 1).item()

                # Append the index to the sequence
                seq.append(x)

            return seq_dataset.decode(seq)

    def transform_sequence(self, seq_dataset, sequence):
        with torch.no_grad():
            hidden_cell = self.__init_hidden()

            outputs = []
            for element in sequence:
                try:
                    x = seq_dataset.encode(element)
                    tensor_x = torch.tensor(x, dtype=torch.long, device=self.device)
                    hidden_cell, y = self.forward(tensor_x, hidden_cell)
                    outputs.append(y)
                except KeyError as e:
                    pass

            hidden, cell = hidden_cell

            # Use cell state as feature vector fot the sentence
            trans_sequence = np.squeeze(cell.data.cpu().numpy())

            return trans_sequence, outputs

    def get_top_k_neuron_weights(self, logreg_model, k=1):
        weights = logreg_model.coef_.T
        weight_penalties = np.squeeze(np.linalg.norm(weights, ord=1, axis=1))

        if k == 1:
            k_indices = np.array([np.argmax(weight_penalties)])
        elif k >= np.log(len(weight_penalties)):
            k_indices = np.argsort(weight_penalties)[-k:][::-1]
        else:
            k_indices = np.argpartition(weight_penalties, -k)[-k:]
            k_indices = (k_indices[np.argsort(weight_penalties[k_indices])])[::-1]

        return k_indices

    def load(self, model_filename):
        print("Loading model:", model_filename)

        model = torch.load(model_filename, map_location=self.device)
        self.load_state_dict(model)
        self.eval()

    def save(self, seq_dataset, path=""):
        # Persist model on disk with current timestamp
        model_filename = path + "_model.pth"
        torch.save(self.state_dict(), model_filename)

        # Persist encoding vocab on disk
        meta_data = {}
        meta_filename = path + "_meta.json"
        with open(meta_filename, 'w') as fp:
            meta_data["vocab"] = " ".join([symb for symb in seq_dataset.vocab])
            meta_data["data_type"]   = seq_dataset.type()
            meta_data["input_size"]  = self.input_size
            meta_data["embed_size"]  = self.embed_size
            meta_data["hidden_size"] = self.hidden_size
            meta_data["output_size"] = self.output_size
            meta_data["n_layers"]    = self.n_layers
            meta_data["dropout"]     = self.dropout
            json.dump(meta_data, fp)

        print("Saved model:", model_filename)
