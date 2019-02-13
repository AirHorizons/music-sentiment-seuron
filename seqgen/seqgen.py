import os
import time
import datetime
import torch
import torch.nn     as nn
import torch.optim  as optim
from torch.autograd import Variable

class mLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(mLSTM, self).__init__()

        self.input_size  = input_size
        self.hidden_size = hidden_size

        self.wx  = nn.Linear(input_size , 4*hidden_size, bias = False)
        self.wh  = nn.Linear(hidden_size, 4*hidden_size, bias = True)
        self.wmx = nn.Linear(input_size ,   hidden_size, bias = False)
        self.wmh = nn.Linear(hidden_size,   hidden_size, bias = False)

    def forward(self, data, last_hidden):
        hx, cx = last_hidden
        m = self.wmx(data) * self.wmh(hx)
        gates = self.wx(data) + self.wh(m)
        i, f, o, u = gates.chunk(4, 1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        u = torch.tanh(u)
        o = torch.sigmoid(o)

        cy = f * cx + i * u
        hy = o * torch.tanh(cy)

        return hy, cy

class SentimentNeuron(nn.Module):

    # Training Log constants
    LOG_PERSIST_PATH = "output/models/"
    LOG_FREQ         = 100
    LOG_SAMPLE_LEN   = 200
    LOG_SAMPLE_TOP_K = 3
    LOG_SAVE_SAMPLES = True

    def __init__(self, embed_size, input_size, hidden_size, output_size, lstm_layers = 1, dropout = 0, enable_cuda = False):
        super(SentimentNeuron, self).__init__()

        # Set running device to "cpu" or "cuda" (if available)
        self.device = torch.device("cpu")
        if enable_cuda:
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
        self.ixh = nn.Embedding(embed_size, input_size)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Hidden to hidden layers
        self.hxh = []
        for i in range(lstm_layers):
            # Create a new mLSTM layer and add to the model
            hxh = mLSTM(input_size, hidden_size)
            self.add_module('layer_%d' % i, hxh)
            self.hxh += [hxh]

            input_size = hidden_size

        # Hidden to output layers
        self.hxy = nn.Linear(hidden_size, output_size)

        # Set this model to run in the given device
        self.to(device=self.device)

    def __init_hidden(self):
        h = torch.zeros(self.lstm_layers, 1, self.hidden_size, device=self.device)
        c = torch.zeros(self.lstm_layers, 1, self.hidden_size, device=self.device)
        return (h, c)

    def forward(self, x, h):
		 # First layer maps the input layer to the hidden layer
        ixh = self.ixh(x)

        h_0, c_0 = h
        h_1, c_1 = [], []

        for i, hxh in enumerate(self.hxh):
            h_1_i, c_1_i = hxh(ixh, (h_0[i], c_0[i]))

            if i == 0:
            	ixh = h_1_i
            else:
            	ixh = ixh + h_1_i

            if i != len(self.hxh):
                ixh = self.dropout(ixh)

            h_1 += [h_1_i]
            c_1 += [c_1_i]

        # Update hidden state
        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        y = self.hxy(ixh)

        return (h_1, c_1), y

    def fit(self, seq_dataset, epochs=100000, seq_length=100, lr=1e-3):
        # Data pointer
        i = 0

        # Loss function is Negative Log-likelihood because this is a multi-class problem
        loss_function = nn.CrossEntropyLoss()

        # Optimizer
        optimizer = optim.SGD(self.parameters(),  lr=lr)

        # Save time before trainning
        t0 = time.time()

        # Loss at epoch 0
        smooth_loss = -torch.log(torch.tensor(1.0/seq_dataset.encoding_size)).item() * seq_length

        h_init = self.__init_hidden()
        for n in range(1, epochs):
            optimizer.zero_grad()

            # Check if we already reached the end of the piece
            if i + 1 + seq_length >= seq_dataset.data_size:
                h_init = self.__init_hidden()
                i = 0

            # Get inputs ready for the network, that is, turn them into
            # Tensors of word indices.
            xs = seq_dataset.slice(i, seq_length)
            ts = seq_dataset.labels(i, seq_length)

            # Run forward pass and get output y
            loss = 0

            h = h_init
            for x,t in zip(xs, ts):
                h, y = self(torch.tensor(x, dtype=torch.long, device=self.device), h)
                loss += loss_function(y, torch.tensor([t], dtype=torch.long, device=self.device))

            # Copy current hidden state to be next h_init
            h_init = (Variable(h[0].data), Variable(h[1].data))

            # Calculate loss in respect to the target ts
            smooth_loss = smooth_loss * 0.999 + loss.item() * 0.001

            if n % self.LOG_FREQ == 0:
                # Save time after every MODEL_LOG_FREQ epochs to calculate delta time
                t1 = time.time()

                self.train_log(n, smooth_loss, seq_dataset, t1 - t0)

                # Save time before restart trainning
                t0 = time.time()

            loss.backward()

            self.__clip_gradient(5)

            optimizer.step()

            # Move data pointer
            i += seq_length

        # Save trained model for sampling
        self.save()

    def train_log(self, n, loss, seq_dataset, dt):
        with torch.no_grad():
            sample_dat = self.sample(seq_dataset, self.LOG_SAMPLE_LEN, self.LOG_SAMPLE_TOP_K)

            print('epoch: n = ', n)
            print('delta time: = ', dt, "s")
            print('loss = ', loss)
            print('----\n' + str(sample_dat) + '\n----')

            if self.LOG_SAVE_SAMPLES:
                seq_dataset.write(sample_dat, "sample_dat_" + str(n))

    def sample(self, seq_dataset, sample_len, topk=1):
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
                ps = torch.softmax(y[0].squeeze(), dim=0)
                ps = self.__truncate_probabilities(ps, topk)

                # Sample the next index according to the probability distribution p
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
        # If LOG_PERSIST_PATH does not exist, create it
        if not os.path.isdir(self.LOG_PERSIST_PATH):
            os.mkdir(self.LOG_PERSIST_PATH)

        # Persist model on disk with current timestamp
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
        torch.save(self.state_dict(), self.LOG_PERSIST_PATH + "seqgen_" + timestamp + ".pth")

    def __truncate_probabilities(self, ps, top_ps=1):
        higher_ps = ps.topk(top_ps)[1]

        for i in set(range(len(ps))) - set(higher_ps):
            ps[i] = 0.

        sum_ps = min(1., sum(ps))
        for i in higher_ps:
            ps[i] += (1. - sum_ps)/len(higher_ps)

        return ps

    def __clip_gradient(self, clip):
        totalnorm = 0
        for p in self.parameters():
            p.grad.data = p.grad.data.clamp(-clip, clip)
