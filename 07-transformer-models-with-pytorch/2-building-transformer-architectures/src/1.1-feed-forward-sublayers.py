class FeedForwardSubLayer(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        # Define the layers and activation
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Pass the input through the layers and activation
        return self.fc2(self.relu(self.fc1(x)))
    
# Instantiate the FeedForwardSubLayer and apply it to x
feed_forward = FeedForwardSubLayer(512, 2048)
output = feed_forward(x)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")