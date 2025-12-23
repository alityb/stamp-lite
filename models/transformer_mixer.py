import torch
import torch.nn as nn

class TransformerMixer(nn.Module):
    """
    Full self-attention mixes everything with everything
    ignores axis-aligned structure, needs more data
    """
    def __init__(self, dim, num_layers, num_heads=4):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        # x: (B, S, T, D)
        B, S, T, D = x.shape
        # Flatten to tokens: (B, S*T, D)
        x = x.reshape(B, S * T, D)
        x = self.transformer(x)
        # Reshape back: (B, S, T, D)
        x = x.reshape(B, S, T, D)
        return self.norm(x)

