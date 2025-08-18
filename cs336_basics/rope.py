import torch

class RotaryPositionalEmbedding(torch.nn.Module):
    """
    Implements Rotary Positional Embedding (RoPE).
    """

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        self.d_k = d_k
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.device = device

        assert d_k % 2 == 0, "d_k must be an even number."

        # Compute theta_k values for each pair
        theta_k = 1.0 / (self.theta ** (torch.arange(0, d_k, 2).float() / d_k)).to(device)

        # Precompute the full rotation matrices for all positions
        rotation_matrices = torch.zeros(max_seq_len, d_k, d_k, device=device)

        for i in range(max_seq_len):
            # Calculate the angular frequency for position i
            # This is equivalent to theta_i,k = i / (theta^(2k/d))
            theta_i_k = i * theta_k
            
            # Construct the block-diagonal rotation matrix for position i
            for k in range(d_k // 2):
                angle = theta_i_k[k]
                c, s = angle.cos(), angle.sin()
                
                # Fill the 2x2 rotation block
                rotation_matrices[i, 2*k, 2*k] = c
                rotation_matrices[i, 2*k, 2*k + 1] = -s
                rotation_matrices[i, 2*k + 1, 2*k] = s
                rotation_matrices[i, 2*k + 1, 2*k + 1] = c
                
        self.register_buffer('rotation_matrices', rotation_matrices)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Applies RoPE to an input tensor using precomputed rotation matrices.
        """
        # We need to map the token positions from the input to the precomputed
        # rotation matrices.
        # token_positions shape: (..., seq_len)
        # rotation_matrices shape: (max_seq_len, d_k, d_k)
        
        # Get the appropriate rotation matrices using the token positions as indices.
        # The result will have shape (..., seq_len, d_k, d_k).
        rotations = self.rotation_matrices[token_positions]
        
        # We can now perform batch matrix multiplication.
        # The operation is R @ x.
        # x shape: (..., seq_len, d_k)
        # rotations shape: (..., seq_len, d_k, d_k)
        
        # To make the matrix multiplication compatible, we can treat x as a
        # batch of vectors of shape (..., seq_len, d_k, 1).
        # torch.matmul handles the batch dimensions automatically.
        x_rotated = torch.matmul(rotations, x.unsqueeze(-1)).squeeze(-1)
        
        return x_rotated