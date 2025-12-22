class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        # Calculate the dimensions each head will process
        self.num_heads = num_heads
        self.d_model = d_model
        self.head_dim = d_model // num_heads
        # Define the three input layers and one output layer
        self.query_linear = nn.Linear(d_model, d_model, bias=False)
        self.key_linear = nn.Linear(d_model, d_model, bias=False)
        self.value_linear = nn.Linear(d_model, d_model, bias=False)
        self.output_linear = nn.Linear(d_model, d_model)