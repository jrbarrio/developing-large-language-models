import torch
import torch.nn as nn

# Define the transformer model
model = nn.Transformer(d_model=1536, nhead=8, num_encoder_layers=6, num_decoder_layers=6)

# Print the model object
print(model)