import torch
import torch.nn as nn
import math

class SwiGLU(nn.Module):
    """
    Implements the SwiGLU (Swish Gated Linear Unit) feed-forward network,
    using nn.Linear layers for all projections and NO bias terms.

    The SwiGLU network is composed of a SiLU activation and a GLU.
    In this specific implementation, one branch of the GLU uses SiLU,
    and the other uses a sigmoid activation for numerical stability.

    Args:
        d_model (int): The dimensionality of the input and output features
                       (often referred to as embedding dimension).
        d_ff_multiplier (float): Multiplier for d_model to determine the
                                 intermediate feed-forward dimension (d_ff).
                                 Defaults to 8/3 as specified.
        multiple_of (int): The d_ff dimension will be rounded up to the
                           nearest multiple of this value. Defaults to 64.
    """
    def __init__(self, d_model: int, d_ff_multiplier: float = 8/3, multiple_of: int = 64):
        super().__init__()
        self.d_model = d_model

        # Calculate d_ff ensuring it's a multiple of `multiple_of`
        # d_ff is approximately 8/3 * d_model, rounded up to the nearest multiple of 64.
        d_ff_unrounded = int(d_model * d_ff_multiplier)
        self.d_ff = (d_ff_unrounded + multiple_of - 1) // multiple_of * multiple_of

        # Define nn.Linear layers for all projections, with NO bias terms.
        # w1 for the SiLU branch (input d_model, output d_ff)
        self.w1 = nn.Linear(d_model, self.d_ff, bias=False)

        # w3 for the Sigmoid branch (input d_model, output d_ff)
        self.w3 = nn.Linear(d_model, self.d_ff, bias=False)

        # w2 for the output layer (input d_ff, output d_model)
        self.w2 = nn.Linear(self.d_ff, d_model, bias=False)

        # Activation functions
        self.silu = nn.SiLU()
        self.sigmoid = nn.Sigmoid()

        print(f"Initialized SwiGLU with d_model={d_model}, calculated d_ff={self.d_ff}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SwiGLU network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        """
        # Apply the first projection with SiLU activation
        # Using self.w1(x) instead of x @ self.w1.weight directly
        proj1_out = self.silu(self.w1(x))

        # Apply the second projection with Sigmoid activation
        # Using self.w3(x) instead of x @ self.w3.weight directly
        proj2_out = self.w3(x)

        # Perform element-wise multiplication (gating mechanism)
        gated_output = proj1_out * proj2_out

        # Apply the final output projection
        # Using self.w2(gated_output) instead of gated_output @ self.w2.weight directly
        output = self.w2(gated_output)

        return output