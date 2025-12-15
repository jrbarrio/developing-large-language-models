import torch
import torch.nn as nn

chars = ['d', 'a', '.', 'l', 'c', 'i', 'w', 'T', 'A', 's', 'n', 'r', 'p', 'v', 'k', 't', 'u', 'm', 'e', 'b', 'f', ',', 'h', 'o', ' ', 'g', '-', 'y']

# Include an RNN layer and linear layer in RNNmodel class
class RNNmodel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNmodel, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
      h0 = torch.zeros(1, x.size(0), self.hidden_size)
      out, _ = self.rnn(x, h0)  
      out = self.fc(out[:, -1, :])  
      return out

# Instantiate the RNN model
model = RNNmodel(1, 16, len(chars))