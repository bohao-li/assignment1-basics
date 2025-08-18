import torch
from typing import Optional

def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Computes the scaled dot-product attention.

    Args:
        query: Queries, a tensor of shape (..., seq_len, d_k).
        key: Keys, a tensor of shape (..., seq_len, d_k).
        value: Values, a tensor of shape (..., seq_len, d_v).
        mask: Optional mask, a boolean tensor of shape (seq_len, seq_len).

    Returns:
        A tensor of shape (..., seq_len, d_v) representing the attention output.
    """
    # Get the dimension of the key vectors.
    d_k = query.size(-1)

    # Calculate the dot product of query and key.
    # The result has shape (..., seq_len, seq_len).
    # torch.matmul performs matrix multiplication.
    scores = torch.matmul(query, key.transpose(-2, -1))

    # Scale the dot product by the square root of d_k.
    scaled_scores = scores / (d_k ** 0.5)

    # Apply the optional mask.
    if mask is not None:
        # A value of -1e9 is used to effectively zero out the probabilities of masked positions
        # after the softmax operation, since e^(-1e9) is very close to zero.
        scaled_scores = scaled_scores.masked_fill(mask == 0, float('-inf'))

    # Compute the attention probabilities using a softmax.
    # The softmax is applied along the last dimension (the sequence length dimension).
    attention_probabilities = torch.softmax(scaled_scores, dim=-1)

    # Multiply the attention probabilities with the value tensor.
    # The result has shape (..., seq_len, d_v).
    output = torch.matmul(attention_probabilities, value)

    return output