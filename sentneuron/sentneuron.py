import torch
import torch.nn as nn
import torch.nn.functional as fc
import torch.optim as optim
import numpy as np

from copy import deepcopy

class SentimentNeuron(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, lstm_layers = 1, dropout = 0):
        super(SentimentNeuron, self).__init__()

        # Init layer sizes
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Init number of LSTM layers
        self.lstm_layers = lstm_layers

        # Init model layers
        self.ixh = nn.Linear(input_size, hidden_size)
        self.hxh = nn.LSTM(hidden_size, hidden_size, lstm_layers, dropout=dropout)
        self.hxy = nn.Linear(hidden_size, output_size)

        # Init hidden state with random weights
        self.h = self.__init_hidden()

    def __init_hidden(self):
        h = torch.randn(self.lstm_layers, 1, self.hidden_size)
        c = torch.randn(self.lstm_layers, 1, self.hidden_size)
        return (h, c)

    def forward(self, xs):
        # First layer maps the input layer to the hidden layer
        ixh = self.ixh(xs)

        # Second layer updates the LSTM hidden state
        hxh, self.h = self.hxh(ixh.view(len(xs), 1, -1), self.h)

        # Third layers maps the hidden state to the output
        y = self.hxy(hxh.view(len(xs), -1))

        return y

    def train(self, seq_dataset, epochs=100000, seq_length=100, lr=1e-3, wd=0):
        # Data pointer
        i = 0

        # Loss function is Negative Log-likelihood because this is a multi-class problem
        loss_function = nn.CrossEntropyLoss()

        # Optimizer is AdaGrad
        optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=wd)

        for n in range(epochs):
            # Clear out the hidden state of the LSTM, detaching it from
            # its history on the last instance.
            self.h = self.__init_hidden()
            self.zero_grad()

            # Check if we already reached the end of the piece
            if i + 1 + seq_length >= seq_dataset.data_size:
                i = 0

            # Get inputs ready for the network, that is, turn them into
            # Tensors of word indices.
            xs = seq_dataset.slice(i, seq_length)
            ts = seq_dataset.labels(i, seq_length)

            # Run forward pass.
            y = self(torch.tensor(xs, dtype=torch.float))

            # Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss = loss_function(y, torch.tensor(ts, dtype=torch.long))

            if n % 100 == 0:
                self.train_log(n, loss, seq_dataset, 1000, False)

            loss.backward()

            optimizer.step()

            # Move data pointer
            i += seq_length

    def train_log(self, n, loss, seq_dataset, log_size=100, write=False):
        with torch.no_grad():
            sample_seq = self.sample(seq_dataset, log_size, seq_dataset.encoding_size, 1.)
            sample_dat = seq_dataset.decode(sample_seq)

            print('n = ', n)
            print('loss = ', loss)
            print('----\n' + str(sample_dat) + '\n----')

            if write:
                seq_dataset.write(sample_dat, "sample_dat_" + str(n))

    def sample(self, seq_dataset, sample_len, top_ps=1, random_prob=0.5):
        with torch.no_grad():
            # Retrieve a random example from the dataset as the first element of the sequence
            x = seq_dataset.random_example()

            # Initialize the sequence
            seq = []

            for t in range(sample_len):
                y = self.forward(torch.tensor([x], dtype=torch.float))

                # Transform output into a probability distribution for a multi-class problem
                ps = fc.softmax(y, dim=1)

                # Sample the next index according to the probability distribution p
                if np.random.rand() <= random_prob:
                    # ps = self.__truncate_probabilities(np.exp(ps.numpy().ravel()), top_ps)
                    ix = torch.multinomial(torch.Tensor(ps), 1).item()
                else:
                    ix = torch.argmax(ps).item()

                # Append the index to the sequence
                seq.append(ix)

                # Encode x for the next step
                x = seq_dataset.encode(seq_dataset.decode([ix])[0])
            return seq

    def __truncate_probabilities(self, ps, top_ps=1):
        higher_ps = np.argpartition(ps, -top_ps)[-top_ps:]

        for i in set(range(len(ps))) - set(higher_ps):
            ps[i] = 0.

        sum_ps = min(1., sum(ps))
        for i in higher_ps:
            ps[i] += (1. - sum_ps)/len(higher_ps)

        return ps
