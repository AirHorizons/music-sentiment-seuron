import torch
import torch.nn     as nn
import torch.optim  as optim

class mLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(mLSTM, self).__init__()

        self.input_size  = input_size
        self.hidden_size = hidden_size

        self.wx  = nn.Linear(input_size , 4*hidden_size, bias = False)
        self.wh  = nn.Linear(hidden_size, 4*hidden_size, bias = True)
        self.wmx = nn.Linear(input_size ,   hidden_size, bias = False)
        self.wmh = nn.Linear(hidden_size,   hidden_size, bias = False)

    def forward(self, x, last_hidden):
        hx, cx = last_hidden
        m = self.wmx(x) * self.wmh(hx)
        gates = self.wx(x) + self.wh(m)
        i, f, o, u = gates.chunk(4, 1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        u = torch.tanh(u)
        o = torch.sigmoid(o)

        cy = f * cx + i * u
        hy = o * torch.tanh(cy)

        return hy, cy
