import torch
import torch.nn as nn
from .tsfm import SimpleTSFM
from .positional_encoding import PositionalEncoding
from .cc_gmlp import CCGMLP
from .transformer_mixer import TransformerMixer

class STAMPAdapter(nn.Module):
    def __init__(self, S, K, T, num_classes, 
                 embed_dim=128, adapter_dim=256, num_layers=4,
                 pe_config="all", mixer_type="gmlp", pool_type="mean"):
        super().__init__()
        self.S, self.K, self.T = S, K, T
        self.mixer_type = mixer_type
        self.pool_type = pool_type
        
        # Frozen TSFM
        self.tsfm = SimpleTSFM(K, embed_dim)
        self.tsfm.freeze()
        
        # Projection
        self.proj = nn.Linear(embed_dim, adapter_dim)
        
        # Positional encoding
        use_token = "token" in pe_config or pe_config == "all"
        use_spatial = "spatial" in pe_config or pe_config == "all"
        use_temporal = "temporal" in pe_config or pe_config == "all"
        self.pe = PositionalEncoding(S, T, adapter_dim, use_token, use_spatial, use_temporal)
        
        # Mixer: CC-GMLP or Transformer
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
        x = x.reshape(B, S, self.T, self.K)  # Tokenize
        
        # TSFM: (B*S*T, K) -> (B*S*T, embed_dim)
        x_flat = x.reshape(B * S * self.T, self.K)
        with torch.no_grad():
            emb = self.tsfm(x_flat)
        
        # Grid: (B, S, T, D)
        x = self.proj(emb.reshape(B, S, self.T, -1))
        
        # Add PE
        x = self.pe(x)
        
        # Mix
        x = self.mixer(x)
        
        # Pool
        if self.pool_type == "mean":
            x = x.mean(dim=(1, 2))
        else:
            # Attention pooling
            B, S, T, D = x.shape
            x_flat = x.reshape(B, S * T, D)
            attn = torch.softmax(self.pool(x_flat), dim=1)
            x = (x_flat * attn).sum(dim=1)
        
        return self.classifier(x)
    
    def get_config_str(self):
        return f"{self.mixer_type}_pe-{self.pe.config_str()}_pool-{self.pool_type}"
