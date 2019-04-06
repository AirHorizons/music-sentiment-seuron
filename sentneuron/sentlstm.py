# External imports
import json
import torch
import torch.nn       as nn
import torch.optim    as optim
import torch.autograd as ag
import numpy          as np

class SentimentLSTM(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, output_size, n_layers=1, dropout=0):
        super(SentimentLSTM, self).__init__()

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
        self.h2h = nn.LSTM(embed_size, hidden_size)

        # Hidden to output layers
        self.h2y = nn.Linear(hidden_size, output_size)

        # Set this model to run in the given device
        self.to(device=self.device)

    def forward(self, x, h):
        emb_x = self.i2h(x)

        h_0, c_0 = h
        output, (hn, cn) = self.h2h(emb_x, (h_0, c_0))

        y = self.h2y(hn.squeeze(0))
        return (hn, cn), y

    def fit_sentiment(self, seq_dataset, trX, trY, teX, teY, epochs=100, lr=1e-3, lr_decay=1, batch_size=32):
        try:
            return self.__fit_sentiment(seq_dataset, trX, trY, teX, teY, epochs, lr, lr_decay, batch_size)
        except KeyboardInterrupt:
            print('Exiting from training early.')

    def __fit_sentiment(self, seq_dataset, trX, trY, teX, teY, epochs, lr, lr_decay, batch_size):
        # Loss function
        loss_function = nn.BCEWithLogitsLoss()

        # Start optimizer with initial learning rate
        epoch_lr = lr

        for epoch in range(epochs):
            h_init = self.__init_hidden(batch_size)

            # Start optimizer with initial learning rate every epoch
            optimizer = optim.Adam(self.parameters(), lr=epoch_lr)

            # Loss at epoch 0
            avg_loss = 0

            # Calculate number of batches
            n_batches = len(trX)//batch_size

            # Each epoch consists of one entire pass over the dataset
            for i in range(n_batches):
                # Reset optimizer grad
                optimizer.zero_grad()

                xs = trX[i*batch_size:(i+1)*batch_size]
                ls = trY[i*batch_size:(i+1)*batch_size]

                # Encode batch
                for j in range(len(xs)):
                    xs[j] = seq_dataset.encode_sequence(xs[j].split(" "))

                # Initialize hidden state with the hidden state from the previous batch
                h = h_init

                xs = torch.tensor(np.array(xs).T, dtype=torch.long, device=self.device)
                h, y = self(xs, h)

                loss = loss_function(y, torch.tensor(np.array([ls]).T, dtype=torch.float, device=self.device))
                loss.backward()

                # Copy current hidden state to be next h_init
                h_init = (ag.Variable(h[0].data), ag.Variable(h[1].data))

                optimizer.step()

                avg_loss += loss.item()

            avg_loss /= len(trX)
            self.__fit_sentiment_log(epoch, avg_loss)

            epoch_lr *= lr_decay

        return self.evaluate_sentiment(seq_dataset, teX, teY, batch_size)

    def __fit_sentiment_log(self, epoch, loss):
        with torch.no_grad():
            print('epoch:', epoch)
            print('loss = ', loss)

    def evaluate_sentiment(self, seq_dataset, teX, teY, batch_size):
        with torch.no_grad():
            enc_teX = []
            for i in range(len(teX)):
                enc_teX.append(seq_dataset.encode_sequence(teX[i].split(" ")))

            guesses = self.predict_sentiment(seq_dataset, enc_teX)
            labels  = torch.tensor(np.array(teY).T, dtype=torch.float, device=self.device)

            correct = (guesses == labels).float()
            return correct.sum() / len(correct)

    def predict_sentiment(self, seq_dataset, enc_teX):
        with torch.no_grad():
            h_init = self.__init_hidden(len(enc_teX))

            xs = torch.tensor(np.array(enc_teX).T, dtype=torch.long, device=self.device)
            h_init, y = self(xs, h_init)

            return torch.round(torch.sigmoid(y))

    def __init_hidden(self, batch_size=1):
        h = torch.zeros(self.n_layers, batch_size, self.hidden_size, device=self.device)
        c = torch.zeros(self.n_layers, batch_size, self.hidden_size, device=self.device)
        return (h, c)
