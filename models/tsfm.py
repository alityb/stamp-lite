import torch 
import torch.nn as nn 

class SimpleTSFM(nn.Module):
    """Frozen TSFM --- location-agnostic embeddings"""
    def __init__(self, token_length, embed_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 64, 7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.proj = nn.Linear(128, embed_dim)

    def forward(self, x):
        # x: (N, K) -> (N, embed_dim)
        x = x.unsqueeze(1)
        x = self.encoder(x).squeeze(-1)
        return self.proj(x)
    
    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

