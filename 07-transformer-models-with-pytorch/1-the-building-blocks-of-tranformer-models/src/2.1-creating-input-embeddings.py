import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        # Set the model dimensionality and vocabulary size
        self.d_model = d_model
        self.vocab_size = vocab_size
        # Instantiate the embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # Return the embeddings multiplied by the square root of d_model
        return self.embedding(x) * math.sqrt(self.d_model)


token_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])

# Instantiate InputEmbeddings and apply it to token_ids
embedding_layer = InputEmbeddings(vocab_size=10000, d_model=512)
output = embedding_layer(token_ids)
print(output.shape)