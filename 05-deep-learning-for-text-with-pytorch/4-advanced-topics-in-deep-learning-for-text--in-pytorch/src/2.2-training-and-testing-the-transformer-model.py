import torch
import torch.nn as nn
import torch.optim as optim

class TransformerEncoder(nn.Module):
    def __init__(self, embed_size, heads, num_layers, dropout):
        super(TransformerEncoder, self).__init__()
        # Initialize the encoder 
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_size, nhead=heads),
            num_layers=num_layers)
        # Define the fully connected layer
        self.fc = nn.Linear(embed_size, 2)

    def forward(self, x):
        # Pass the input through the transformer encoder 
        x = self.encoder(x)
        x = x.mean(dim=1) 
        return self.fc(x)

model = TransformerEncoder(embed_size=512, heads=8, num_layers=3, dropout=0.5)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

train_sentences = ['I love this product', 'This is terrible', 'Could be better']
train_labels = [1, 0, 0]  # 1: positive, 0: negative

# Collect all unique tokens
all_tokens = set()
for sentence in train_sentences:
    all_tokens.update(sentence.split())

# Create token embeddings
token_embeddings = {token: torch.randn(1, 512) for token in all_tokens}

for epoch in range(5):  
    for sentence, label in zip(train_sentences, train_labels):
        # Split the sentences into tokens and stack the embeddings
        tokens = sentence.split()
        data = torch.stack([token_embeddings[token] for token in tokens], dim=1)
        output = model(data)
        loss = criterion(output, torch.tensor([label]))
        # Zero the gradients and perform a backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item()}")

def predict(sentence):
    model.eval()
    # Deactivate the gradient computations and get the sentiment prediction.
    with torch.no_grad():
        tokens = sentence.split()
        data = torch.stack([token_embeddings.get(token, torch.rand((1, 512))) for token in tokens], dim=1)
        output = model(data)
        predicted = torch.argmax(output, dim=1)
        return "Positive" if predicted.item() == 1 else "Negative"

sample_sentence = "This product can be better"
print(f"'{sample_sentence}' is {predict(sample_sentence)}")