import torch
import torch.nn as nn
import torch.nn.functional as fc
import torch.optim as optim
import numpy as np

from copy import deepcopy

class SequenceGenerator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, lstm_layers = 1, dropout = 0, enable_cuda = False):
        super(SequenceGenerator, self).__init__()

        # Set running device to "cpu" or "cuda" (if available)
        self.device = torch.device("cpu")
        if enable_cuda:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                print("Cuda is not available. Model will run on the cpu.")

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
        h = torch.randn(self.lstm_layers, 1, self.hidden_size, device=self.device)
        c = torch.randn(self.lstm_layers, 1, self.hidden_size, device=self.device)
        return (h, c)

    def forward(self, xs):
        # First layer maps the input layer to the hidden layer
        ixh = self.ixh(xs)

        # Second layer updates the LSTM hidden state
        hxh, self.h = self.hxh(ixh.view(len(xs), 1, -1), self.h)

        # Third layers maps the hidden state to the output
        y = self.hxy(hxh.view(len(xs), -1))

        return y

    def train(self, seq_dataset, epochs=100000, seq_length=100, lr=1e-3, wd=0, sample_size=100, write_sample=False):
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
            y = self(torch.tensor(xs, dtype=torch.float, device=self.device))

            # Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            loss = loss_function(y, torch.tensor(ts, dtype=torch.long, device=self.device))

            if n % 100 == 0:
                self.train_log(n, loss, seq_dataset, sample_size, write_sample)

            loss.backward()

            optimizer.step()

            # Move data pointer
            i += seq_length

    def train_log(self, n, loss, seq_dataset, sample_size=100, write_sample=False):
        with torch.no_grad():
            sample_seq = self.sample(seq_dataset, sample_size)
            sample_dat = seq_dataset.decode(sample_seq)

            print('n = ', n)
            print('loss = ', loss)
            print('----\n' + str(sample_dat) + '\n----')

            if write_sample:
                seq_dataset.write(sample_dat, "sample_dat_" + str(n))

    def sample(self, seq_dataset, sample_len):
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
                ix = seq_dataset.sample(ps)

                # Append the index to the sequence
                seq.append(ix)

                # Encode x for the next step
                x = seq_dataset.encode(seq_dataset.decode([ix])[0])
            return seq
