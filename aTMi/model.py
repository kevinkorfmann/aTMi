import torch
import torch.nn as nn
from x_transformers import Encoder

def num_params(model): return sum([p.numel() for p in model.parameters() if p.requires_grad])

class CoalescenceTransformer(nn.Module):
    def __init__(self, seq_len, enc_dim, enc_depth, enc_heads, in_dim, out_dim, sum_encoder_dim=True):
        super().__init__()
        self.sum_encoder_dim = sum_encoder_dim
        self.inp = nn.Linear(in_dim, enc_dim)
        self.encoder = Encoder(
            dim=enc_dim, depth=enc_depth,
            heads=enc_heads,ff_glu=True,
            residual_attn=True, flash=True,
            rotary_pos_emb=True, dropout=0.6)#0.6#0.2
        self.out = nn.Linear(enc_dim, out_dim)
    def forward(self, x):
        x = self.inp(x)
        x = self.encoder(x)
        if self.sum_encoder_dim: x = x.sum(dim=1)
        return self.out(x)
    

    
