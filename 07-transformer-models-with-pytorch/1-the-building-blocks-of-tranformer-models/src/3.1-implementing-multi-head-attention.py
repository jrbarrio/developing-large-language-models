# Define attention parameters
d_model = 512
num_heads = 8

# Instantiate a MultiHeadAttention instance
multihead_attn = MultiHeadAttention(d_model=d_model, num_heads=num_heads)

# Pass the query, key, and value matrices through the mechanism
output = multihead_attn(query, key, value)
print(output.shape)