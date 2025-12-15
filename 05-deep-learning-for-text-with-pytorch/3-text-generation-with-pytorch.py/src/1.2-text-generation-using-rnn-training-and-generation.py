import torch
import torch.nn as nn

data = "The rabbit-hole went straight on like a tunnel for some way, and then dipped suddenly down, so suddenly that Alice had not a moment to think about stopping herself before she found herself falling down a very deep well."
chars = list(set(data))

char_to_ix = {char: i for i, char in enumerate(chars)}
ix_to_char = {i: char for i, char in enumerate(chars)}

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

# Instantiate the loss function
criterion = nn.CrossEntropyLoss()
# Instantiate the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

inputs = [char_to_ix[ch] for ch in data[:-1]]
targets = [char_to_ix[ch] for ch in data[1:]]
inputs = torch.tensor(inputs, dtype=torch.long).view(-1, 1)
inputs = nn.functional.one_hot(inputs, num_classes=len(chars)).float()
targets = torch.tensor(targets, dtype=torch.long)

# Train the model
for epoch in range(100):
    model.train()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}/100, Loss: {loss.item()}')

# Test the model
model.eval()
test_input = char_to_ix['r']
test_input = nn.functional.one_hot(torch.tensor(test_input).view(-1, 1), num_classes=len(chars)).float()
predicted_output = model(test_input)
predicted_char_ix = torch.argmax(predicted_output, 1).item()
print(f"Test Input: 'r', Predicted Output: '{ix_to_char[predicted_char_ix]}'")