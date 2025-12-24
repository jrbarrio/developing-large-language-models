seq_length= 3

# Create a Boolean matrix to mask future tokens
tgt_mask = (1 - torch.triu(
  torch.ones(1, seq_length, seq_length), diagonal=1)
).bool()

print(tgt_mask)