import torch 
import torch.nn as nn 

class ccgmlpblock(nn.module):
    """hypothesis: Hard-coded inductive bias (spatial ≠ temporal)
    Gating = multiplicative interactions = parameter efficient"""
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.temporal_gate = nn.Linear(dim, dim)
        self.spatial_gate = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim * 2, dim)
        
    def forward(self, x):
        # x: (B, S, T, D)
        residual = x
        x = self.norm(x)
        x = torch.nn.functional.gelu(self.fc1(x))
        
        # Split for criss-cross gating
        x_t, x_s = x.chunk(2, dim=-1)
        
        # Temporal gating: operates along T
        B, S, T, D = x_t.shape
        x_t_flat = x_t.reshape(B * S, T, D)
        x_t_flat = x_t_flat * torch.sigmoid(self.temporal_gate(x_t_flat))
        x_t = x_t_flat.reshape(B, S, T, D)
        
        # Spatial gating: operates along S
        x_s_perm = x_s.permute(0, 2, 1, 3).reshape(B * T, S, D)
        x_s_perm = x_s_perm * torch.sigmoid(self.spatial_gate(x_s_perm))
        x_s = x_s_perm.reshape(B, T, S, D).permute(0, 2, 1, 3)

        x = torch.cat([x_t, x_s], dim=-1)
        x = self.fc2(x)
        return x + residual

class CCGMLP(nn.module):
    """stack of cc-gmlp blocks"""
    def __init__(self, dim, num_layers):
        super().__init__()
        self.blocks = nn.ModuleList([ccgmlpblock(dim) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.norm(x)
