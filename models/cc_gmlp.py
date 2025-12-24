import torch
import torch.nn as nn

class CCGMLPBlock(nn.Module):
    """
    Key hypothesis: Hard-coded inductive bias (spatial ≠ temporal)
    Gating = multiplicative interactions = parameter efficient

    Paper specs (ML4H 2025, Section 3.3):
    - Gate init: N(0, 10^-6) - CRITICAL for preventing saturation
    - InstanceNorm (per-sample normalization)
    - h=256 feedforward dimension
    """
    def __init__(self, dim, h=256):
        super().__init__()
        # Instance normalization: LayerNorm normalizes per-sample across features
        # This is the standard "instance norm" for transformers (per-position, per-sample)
        self.norm = nn.LayerNorm(dim)

        self.fc1 = nn.Linear(dim, h * 2)  # Paper: h=256
        self.temporal_gate = nn.Linear(h // 2, h // 2)
        self.spatial_gate = nn.Linear(h // 2, h // 2)
        self.fc2 = nn.Linear(h, dim)

        # CRITICAL: Initialize gate weights to N(0, 10^-6)
        # This keeps gates nearly open initially, preventing saturation
        nn.init.normal_(self.temporal_gate.weight, mean=0.0, std=1e-6)
        nn.init.zeros_(self.temporal_gate.bias)
        nn.init.normal_(self.spatial_gate.weight, mean=0.0, std=1e-6)
        nn.init.zeros_(self.spatial_gate.bias)

        # Track gate statistics for debugging
        self.register_buffer('gate_stats_temporal', torch.zeros(4))  # mean, std, min, max
        self.register_buffer('gate_stats_spatial', torch.zeros(4))
        
    def forward(self, x):
        # x: (B, S, T, D)
        B, S, T, D = x.shape
        residual = x

        # LayerNorm: normalizes across features (D) for each position independently
        x = self.norm(x)

        # GELU activation before gating (paper Eq. 2)
        x = torch.nn.functional.gelu(self.fc1(x))

        # Split for criss-cross gating
        x_t, x_s = x.chunk(2, dim=-1)  # Each has shape (B, S, T, h)

        # Temporal gating: implements g_T(Z) = Z1 ⊙ σ(W_T · Z2)
        x_t_val, x_t_gate_pre = x_t.chunk(2, dim=-1) # Each (B, S, T, h/2)
        h_half = x_t_val.shape[-1]

        x_t_gate_pre_flat = x_t_gate_pre.reshape(B * S, T, h_half)
        gate_t = torch.sigmoid(self.temporal_gate(x_t_gate_pre_flat))

        # Update gate statistics (only during training)
        if self.training:
            with torch.no_grad():
                self.gate_stats_temporal[0] = gate_t.mean()
                self.gate_stats_temporal[1] = gate_t.std()
                self.gate_stats_temporal[2] = gate_t.min()
                self.gate_stats_temporal[3] = gate_t.max()

        gated_t_flat = x_t_val.reshape(B * S, T, h_half) * gate_t
        gated_t = gated_t_flat.reshape(B, S, T, h_half)

        # Spatial gating: implements g_S(Z) = Z1 ⊙ σ(W_S · Z2)
        x_s_val, x_s_gate_pre = x_s.chunk(2, dim=-1) # Each (B, S, T, h/2)
        
        x_s_gate_pre_perm = x_s_gate_pre.permute(0, 2, 1, 3).reshape(B * T, S, h_half)
        gate_s = torch.sigmoid(self.spatial_gate(x_s_gate_pre_perm))

        # Update gate statistics (only during training)
        if self.training:
            with torch.no_grad():
                self.gate_stats_spatial[0] = gate_s.mean()
                self.gate_stats_spatial[1] = gate_s.std()
                self.gate_stats_spatial[2] = gate_s.min()
                self.gate_stats_spatial[3] = gate_s.max()

        gated_s_perm = x_s_val.permute(0, 2, 1, 3).reshape(B * T, S, h_half) * gate_s
        gated_s = gated_s_perm.reshape(B, T, S, h_half).permute(0, 2, 1, 3)

        # Combine and project back (Paper Eq. 5)
        x = torch.cat([gated_t, gated_s], dim=-1)
        x = self.fc2(x)

        # Residual connection: output = Ê + Ẽ
        return x + residual

    def get_gate_stats(self):
        """Return gate statistics for monitoring during training."""
        return {
            'temporal': {
                'mean': self.gate_stats_temporal[0].item(),
                'std': self.gate_stats_temporal[1].item(),
                'min': self.gate_stats_temporal[2].item(),
                'max': self.gate_stats_temporal[3].item(),
            },
            'spatial': {
                'mean': self.gate_stats_spatial[0].item(),
                'std': self.gate_stats_spatial[1].item(),
                'min': self.gate_stats_spatial[2].item(),
                'max': self.gate_stats_spatial[3].item(),
            }
        }

class CCGMLP(nn.Module):
    """
    Stack of CC-GMLP blocks (Paper: L=8 blocks)

    Args:
        dim: Model dimension
        num_layers: Number of CC-GMLP blocks (paper uses L=8)
        h: Feedforward hidden dimension (paper uses h=256)
    """
    def __init__(self, dim, num_layers, h=256):
        super().__init__()
        self.blocks = nn.ModuleList([CCGMLPBlock(dim, h) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: (B, S, T, D)
        for block in self.blocks:
            x = block(x)

        # Final LayerNorm
        return self.norm(x)

    def get_all_gate_stats(self):
        """Get gate statistics from all blocks for monitoring."""
        stats = {}
        for i, block in enumerate(self.blocks):
            stats[f'block_{i}'] = block.get_gate_stats()
        return stats
