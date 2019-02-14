# External imports
import os
import time
import datetime
import torch
import torch.nn       as nn
import torch.optim    as optim
import torch.autograd as ag

# Local imports
from .mlstm          import mLSTM

class SentimentNeuron(nn.Module):

    # Training Log constants
    LOG_PERSIST_PATH = "output/models/"
    LOG_SAMPLE_LEN   = 200
    LOG_SAVE_SAMPLES = True

    def __init__(self, input_size, embed_size, hidden_size, output_size, lstm_layers=1, dropout=0):
        super(SentimentNeuron, self).__init__()

        # Set running device to "cpu" or "cuda" (if available)
        self.device = torch.device("cpu")

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            print("Cuda is not available. Training/Sampling will run on the cpu.")

        # Init layer sizes
        self.input_size  = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Init number of LSTM layers
        self.lstm_layers = lstm_layers

        # Embedding layer
        self.i2h = nn.Embedding(input_size, embed_size)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Hidden to hidden layers
        self.h2h = []
        for i in range(lstm_layers):
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
                emb_x = self.dropout(emb_x)

            h_1 += [h_1_i]
            c_1 += [c_1_i]

        # Update hidden state
        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        y = self.h2y(emb_x)

        return (h_1, c_1), y

    def fit(self, seq_dataset, epochs=100000, seq_length=100, lr=1e-3, grad_clip=5):
        try:
            self.__fit(seq_dataset, epochs, seq_length, lr, grad_clip)
        except KeyboardInterrupt:
            print('Exiting from training early.')

        self.save()

    def __fit(self, seq_dataset, epochs=100000, seq_length=100, lr=1e-3, lr_decay=0.7, grad_clip=5):
        # Loss function
        loss_function = nn.CrossEntropyLoss()

        # Optimizer
        optimizer = optim.SGD(self.parameters(),  lr=lr)

        # Loss at epoch 0
        smooth_loss = -torch.log(torch.tensor(1.0/seq_dataset.encoding_size)).item() * seq_length

        # Calculate batch size
        batch_size = seq_dataset.data_size//seq_length

        for epoch in range(epochs):
            h_init = self.__init_hidden()

            # Each epoch consists of one entire pass over the dataset
            for batch_ix in range(batch_size):
                # Reset optimizer grad
                optimizer.zero_grad()

                # Slice the dataset to create the current batch
                batch = seq_dataset.slice(batch_ix * seq_length, seq_length + 1)

                # Initialize hidden state with the hidden state from the previous batch
                h = h_init

                loss = 0
                for t in range(seq_length):
                    # Run forward pass and get output y
                    h, y = self(torch.tensor(batch[t], dtype=torch.long, device=self.device), h)

                    # Calculate loss in respect to the target ts
                    loss += loss_function(y, torch.tensor([batch[t+1]], dtype=torch.long, device=self.device))

                loss.backward()

                # Copy current hidden state to be next h_init
                h_init = (ag.Variable(h[0].data), ag.Variable(h[1].data))

                # Clip gradients
                self.__clip_gradient(grad_clip)

                # Run Stochastic Gradient Descent and Update weights
                optimizer.step()

                # Calculate average loss and log the results of this batch
                smooth_loss = smooth_loss * 0.999 + loss.item() * 0.001
                self.train_log(epoch, (batch_ix, batch_size), smooth_loss, seq_dataset)

            # Apply learning rate decay before the next epoch
            lr *= lr_decay

        # Save trained model for sampling
        self.save()

    def train_log(self, epoch, batch_ix, loss, seq_dataset):
        with torch.no_grad():
            sample_dat = self.sample(seq_dataset, self.LOG_SAMPLE_LEN)

            print('epoch:', epoch)
            print('batch: {}/{}'.format(batch_ix[0], batch_ix[1]))
            print('loss = ', loss)
            print('----\n' + str(sample_dat) + '\n----')

            if self.LOG_SAVE_SAMPLES:
                seq_dataset.write(sample_dat, "sample_dat_" + str(epoch))

    def sample(self, seq_dataset, sample_len, temperature=0.4):
        with torch.no_grad():
            # Retrieve a random example from the dataset as the first element of the sequence
            xs = seq_dataset.slice(0, 20)

            # Initialize the sequence
            seq = []

            # Create a new hidden state
            h = self.__init_hidden()
            for x in xs:
                h, y = self.forward(torch.tensor(x, dtype=torch.long, device=self.device), h)
                seq.append(x)

            for t in range(sample_len):
                h, y = self.forward(torch.tensor(x, dtype=torch.long, device=self.device), h)

                # Transform output into a probability distribution
                ps = torch.softmax(y[0].squeeze().div(temperature), dim=0)

                # Sample the next index according to the probability distribution ps
                ix = torch.multinomial(ps, 1).item()

                # Append the index to the sequence
                seq.append(ix)

                # Encode x for the next step
                x = seq_dataset.encode(seq_dataset.decode([ix])[0])

            return seq_dataset.decode(seq)

    def load(self, model_filename):
        print("Loading model:", model_filename)

        model = torch.load(model_filename, map_location=self.device)
        self.load_state_dict(model)
        self.eval()

    def save(self):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
        model_filename = self.LOG_PERSIST_PATH + "seqgen_" + timestamp + ".pth"

        print("saving model:", model_filename)

        # If LOG_PERSIST_PATH does not exist, create it
        if not os.path.isdir(self.LOG_PERSIST_PATH):
            os.mkdir(self.LOG_PERSIST_PATH)

        # Persist model on disk with current timestamp
        torch.save(self.state_dict(), model_filename)

    def __init_hidden(self, batch_size=1):
        h = torch.zeros(self.lstm_layers, batch_size, self.hidden_size, device=self.device)
        c = torch.zeros(self.lstm_layers, batch_size, self.hidden_size, device=self.device)
        return (h, c)

    def __clip_gradient(self, clip):
        totalnorm = 0
        for p in self.parameters():
            p.grad.data = p.grad.data.clamp(-clip, clip)

    def __truncate_probabilities(self, ps, top_ps=1):
        higher_ps = ps.topk(top_ps)[1]

        for i in set(range(len(ps))) - set(higher_ps):
            ps[i] = 0.

        sum_ps = min(1., sum(ps))
        for i in higher_ps:
            ps[i] += (1. - sum_ps)/len(higher_ps)

        return ps
