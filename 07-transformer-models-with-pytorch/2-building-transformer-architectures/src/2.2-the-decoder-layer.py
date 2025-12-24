class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ff_sublayer = FeedForwardSubLayer(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, tgt_mask):
        # Perform the attention calculation
        attn_output = self.self_attn(x, x, x, tgt_mask)
        # Apply dropout and the first layer normalization
        x = self.norm1(x + self.dropout(attn_output))
        # Pass through the feed-forward sublayer
        ff_output = self.ff_sublayer(x)
        # Apply dropout and the second layer normalization
        x = self.norm2(x + self.dropout(ff_output))
        return x