import torch
import torch.nn as nn 

class PositionalEncoding(nn.Module):
    """Does PE make locations explicit?"""

    def __init__(self, S, T, D, use_token=False, use_spatial=False, use_temporal=False):
        super().__init__()
        self.use_token = use_token
        self.use_spatial = use_spatial
        self.use_temporal = use_temporal

        if use_token:
            self.token_pe = nn.Parameter(torch.randn(S, T, D) * 0.02)
        if use_spatial:
            self.spatial_pe = nn.Parameter(torch.randn(S, 1, D) * 0.02)
        if use_temporal:
            self.temporal_pe = nn.Parameter(torch.randn(1, T, D) * 0.02)

    def forward(self, x):
        # x: (B, S, T, D)
        if self.use_token:
            x = x + self.token_pe.unsqueeze(0)
        if self.use_spatial:
            x = x + self.spatial_pe.unsqueeze(0)
        if self.use_temporal:
            x = x + self.temporal_pe.unsqueeze(0)
        return x
    
    def config_str(self):
        parts = []
        if self.use_token: parts.append("token")
        if self.use_spatial: parts.append("spatial")
        if self.use_temporal: parts.append("temporal")
        return "+".join(parts) if parts else "none"


