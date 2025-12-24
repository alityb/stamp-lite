import torch
import torch.nn as nn
from .tsfm import SimpleTSFM
from .positional_encoding import PositionalEncoding
from .cc_gmlp import CCGMLP
from .transformer_mixer import TransformerMixer

class STAMPAdapter(nn.Module):
    def __init__(self, S, K, T, num_classes,
                 embed_dim=128, adapter_dim=256, num_layers=4,
                 pe_config="all", mixer_type="gmlp", pool_type="mean", debug=False):
        super().__init__()
        self.S, self.K, self.T = S, K, T
        self.mixer_type = mixer_type
        self.pool_type = pool_type
        self.debug = debug
        self.forward_count = 0
        
        # TSFM encoder (trainable - not frozen)
        # CRITICAL: If frozen without pre-training, embeddings are random noise!
        self.tsfm = SimpleTSFM(K, embed_dim)
        # self.tsfm.freeze()  # Commented out - train from scratch
        
        # Projection
        self.proj = nn.Linear(embed_dim, adapter_dim)
        
        # Positional encoding
        use_token = "token" in pe_config or pe_config == "all"
        use_spatial = "spatial" in pe_config or pe_config == "all"
        use_temporal = "temporal" in pe_config or pe_config == "all"
        self.pe = PositionalEncoding(S, T, adapter_dim, use_token, use_spatial, use_temporal)
        
        # Mixer: CC-GMLP, or Transformer
        if mixer_type == "gmlp":
            self.mixer = CCGMLP(adapter_dim, num_layers)
        else:
            self.mixer = TransformerMixer(adapter_dim, num_layers)
        
        # Pooling
        if pool_type == "attention":
            self.pool = nn.Sequential(
                nn.Linear(adapter_dim, adapter_dim // 2),
                nn.Tanh(),
                nn.Linear(adapter_dim // 2, 1)
            )
        
        # Classifier
        self.classifier = nn.Linear(adapter_dim, num_classes)
        
    def forward(self, x):
        # x: (B, S, T_raw)
        B, S, T_raw = x.shape

        # --- Paper Assumptions ---
        # Enforce dimensional assumptions from the paper
        assert S == self.S, f"Input spatial dimension {S} does not match model's expected {self.S}"
        assert T_raw == self.T * self.K, f"Input temporal dimension {T_raw} does not match model's expected T*K={self.T*self.K}"

        if self.debug and self.forward_count % 10 == 0:
            print(f"\n{'='*60}")
            print(f"FORWARD PASS DEBUG (batch {self.forward_count})")
            print(f"{'='*60}")
            print(f"Input shape: {x.shape}")
            print(f"Input stats: mean={x.mean():.4f}, std={x.std():.4f}, min={x.min():.4f}, max={x.max():.4f}")

        x = x.reshape(B, S, self.T, self.K)  # Tokenize

        # TSFM: (B*S*T, K) -> (B*S*T, embed_dim)
        x_flat = x.reshape(B * S * self.T, self.K)
        emb = self.tsfm(x_flat)

        if self.debug and self.forward_count % 10 == 0:
            print(f"\nAfter TSFM: mean={emb.mean():.4f}, std={emb.std():.4f}, min={emb.min():.4f}, max={emb.max():.4f}")

        # Grid: (B, S, T, D)
        x = self.proj(emb.reshape(B, S, self.T, -1))

        if self.debug and self.forward_count % 10 == 0:
            print(f"After proj: mean={x.mean():.4f}, std={x.std():.4f}, min={x.min():.4f}, max={x.max():.4f}")

        # Add PE
        x_before_pe = x.clone() if self.debug and self.forward_count % 10 == 0 else None
        x = self.pe(x)

        if self.debug and self.forward_count % 10 == 0:
            pe_diff = (x - x_before_pe).abs().mean()
            print(f"After PE: mean={x.mean():.4f}, std={x.std():.4f}")
            print(f"PE effect: |diff|={pe_diff:.6f} (should be > 0 if PE is active)")

        # Mix
        x = self.mixer(x)

        if self.debug and self.forward_count % 10 == 0:
            print(f"After mixer: mean={x.mean():.4f}, std={x.std():.4f}, min={x.min():.4f}, max={x.max():.4f}")

        # Pool
        if self.pool_type == "mean":
            x = x.mean(dim=(1, 2))
        else:
            # Attention pooling
            B, S, T, D = x.shape
            x_flat = x.reshape(B, S * T, D)
            attn = torch.softmax(self.pool(x_flat), dim=1)
            x = (x_flat * attn).sum(dim=1)

        if self.debug and self.forward_count % 10 == 0:
            print(f"After pool: mean={x.mean():.4f}, std={x.std():.4f}")

        logits = self.classifier(x)

        if self.debug and self.forward_count % 10 == 0:
            print(f"Logits: mean={logits.mean():.4f}, std={logits.std():.4f}")
            print(f"Logits per class: {logits[0].detach().cpu().numpy()}")
            print(f"Are all logits similar? std={logits.std(dim=1).mean():.4f} (< 0.1 is bad)")
            print(f"{'='*60}\n")

        self.forward_count += 1
        return logits
    
    def get_config_str(self):
        return f"{self.mixer_type}_pe-{self.pe.config_str()}_pool-{self.pool_type}"

    def check_gradient_flow(self):
        """Check gradient magnitudes for each module"""
        grad_stats = {}

        for name, param in self.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_stats[name] = grad_norm

        return grad_stats

    def print_gradient_flow(self, detailed=False):
        """Print gradient flow diagnostics"""
        grad_stats = self.check_gradient_flow()

        if not grad_stats:
            print("WARNING: No gradients found!")
            return

        print(f"\n{'='*60}")
        print("GRADIENT FLOW DIAGNOSTICS")
        print(f"{'='*60}")

        # Group by module
        groups = {
            'tsfm': [],
            'proj': [],
            'pe': [],
            'mixer': [],
            'pool': [],
            'classifier': []
        }

        for name, grad_norm in grad_stats.items():
            for group_name in groups.keys():
                if group_name in name:
                    groups[group_name].append((name, grad_norm))
                    break

        for group_name, params in groups.items():
            if params:
                avg_grad = sum(g for _, g in params) / len(params)
                max_grad = max(g for _, g in params)
                min_grad = min(g for _, g in params)
                print(f"{group_name:12s}: avg={avg_grad:.6f}, min={min_grad:.6f}, max={max_grad:.6f}")

                if detailed:
                    for pname, grad in params[:3]:  # Show first 3
                        print(f"  └─ {pname}: {grad:.6f}")

        # Check for dead layers
        dead_params = [name for name, grad in grad_stats.items() if grad < 1e-7]
        if dead_params:
            print(f"\n⚠️  WARNING: {len(dead_params)} parameters have near-zero gradients:")
            for name in dead_params[:5]:
                print(f"  - {name}")

        print(f"{'='*60}\n")
