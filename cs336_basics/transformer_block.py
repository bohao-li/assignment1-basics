import torch
import torch.nn as nn
from typing import Optional

from cs336_basics.multi_head_self_attention import CausalMultiHeadSelfAttention
from cs336_basics.rms_norm import RMSNorm
from cs336_basics.swiglu import SwiGLU

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: Optional[int] = None,
        theta: Optional[float] = None,
        device = None,
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = CausalMultiHeadSelfAttention(d_model, num_heads, max_seq_len, theta, device)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff // d_model)

    def forward(self, x):
        y = x + self.attn.forward(self.ln1(x))
        y = y + self.ffn.forward(self.ln2(y))
        return y
