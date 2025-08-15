import torch

class RotaryPositionalEmbedding(torch.nn.Module):
    """
    Implements Rotary Positional Embedding (RoPE).
    """

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        """
        Constructs the RoPE module and creates buffers if needed.

        Args:
            theta: The theta value for the RoPE.
            d_k: The dimension of query and key vectors.
            max_seq_len: The maximum sequence length that will be inputted.
            device: The device to store the buffer on.
        """
        super().__init__()
        # TODO: Implement initialization logic
        # You may want to precompute the cos and sin tensors
        # and register them as buffers.

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Applies RoPE to an input tensor.

        Args:
            x: An input tensor of shape (..., seq_len, d_k).
            token_positions: A tensor of shape (..., seq_len) specifying the
                             token positions of x along the sequence dimension.

        Returns:
            A tensor of the same shape as x with RoPE applied.
        """
        # TODO: Implement the forward pass
        # The implementation should tolerate x with an arbitrary number of batch dimensions.
        # Use the token positions to slice the cos and sin tensors.
        raise NotImplementedError