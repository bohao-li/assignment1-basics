import torch
import torch.nn as nn
import torch.nn.functional as F


from cs336_basics.rope import RotaryPositionalEmbedding
from typing import Optional


class CausalMultiHeadSelfAttention(nn.Module):
    """
    Causal Multi-Head Self-Attention as described by Vaswani et al. (2017).

    This module implements a self-attention mechanism where each position
    in the output sequence can only attend to earlier positions, preventing
    it from "peeking" into future tokens.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: Optional[int] = None,
        theta: Optional[float] = None,
        device: torch.device | None = None,
    ):
        """
        Initializes the CausalMultiHeadSelfAttention module.

        Args:
            d_model (int): The dimensionality of the input and output.
            num_heads (int): The number of attention heads.
        """
        super(CausalMultiHeadSelfAttention, self).__init__()

        # Ensure d_model is divisible by num_heads
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads

        # d_k and d_v are set as d_model / num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        # Linear projections for query, key, and value
        self.q_proj = nn.Linear(d_model, d_model, device=device, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, device=device, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, device=device, bias=False)

        # Final output projection
        self.output_proj = nn.Linear(d_model, d_model, bias=False)

        if max_seq_len is not None and theta is not None:
            self.rope_module = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device)
        else:
            self.rope_module = None

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, token_positions: torch.Tensor = None) -> torch.Tensor:
        """
        Performs the forward pass of the attention mechanism.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len, d_model).
            mask (torch.Tensor, optional): An optional mask tensor. This is typically
                                           used for padding tokens but can also be
                                           used to enforce causal attention.

        Returns:
            torch.Tensor: The output tensor after self-attention, of shape
                          (batch_size, seq_len, d_model).
        """
        batch_size, seq_len, _ = x.shape

        # 1. Project Q, K, V using the linear layers
        #    and reshape for multi-head attention.
        #    The shape becomes (batch_size, num_heads, seq_len, d_k)

        Q = (
            self.q_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        K = (
            self.k_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        V = (
            self.v_proj(x)
            .view(batch_size, seq_len, self.num_heads, self.d_v)
            .transpose(1, 2)
        )
        
        if self.rope_module is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=self.rope_module.device)
            Q = self.rope_module.forward(Q, token_positions)
            K = self.rope_module.forward(K, token_positions)

        # 2. Compute the attention scores: Q * K^T
        #    The resulting shape is (batch_size, num_heads, seq_len, seq_len)

        attension_score = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k**0.5)

        # 3. Apply the causal mask.
        #    This is the key step for "causal" attention. It ensures that
        #    each token can only attend to previous and current tokens.
        #    Hint: You can use torch.tril or a similar masking method.

        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
        )
        attension_score = attension_score.masked_fill(causal_mask == 0, float("-inf"))

        if mask is not None:
            # A value of -1e9 is used to effectively zero out the probabilities of masked positions
            # after the softmax operation, since e^(-1e9) is very close to zero.
            attension_score = attension_score.masked_fill(mask == 0, float("-inf"))

        # 4. Apply the softmax function to get attention weights.

        attention_probabilities = torch.softmax(attension_score, dim=-1)

        # 5. Compute the weighted sum of values: Attention_Weights * V
        #    The shape becomes (batch_size, num_heads, seq_len, d_v)

        weighted_values = torch.matmul(attention_probabilities, V)

        # 6. Concatenate the heads back together.
        #    Reshape from (batch_size, num_heads, seq_len, d_v) to
        #    (batch_size, seq_len, d_model).

        weighted_values_transposed = weighted_values.transpose(1, 2).contiguous()
        concat_output = weighted_values_transposed.view(
            batch_size, seq_len, self.d_model
        )

        # 7. Apply the final output linear projection.

        return self.output_proj(concat_output)
