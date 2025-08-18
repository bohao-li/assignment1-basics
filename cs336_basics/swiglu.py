import torch
import torch.nn as nn
import math

class SwiGLU(nn.Module):
    """
    Implements the SwiGLU (Swish Gated Linear Unit) feed-forward network,
    with manual weight multiplication and NO bias terms.

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
        # dff is approximately 8/3 * d_model, rounded up to the nearest multiple of 64.
        d_ff_unrounded = int(d_model * d_ff_multiplier)
        self.d_ff = (d_ff_unrounded + multiple_of - 1) // multiple_of * multiple_of

        # Define weights as learnable parameters for the first projection (NO BIAS)
        # Weights for the SiLU branch (input d_model, output d_ff)
        self.W1 = nn.Parameter(torch.empty(d_model, self.d_ff))

        # Define weights as learnable parameters for the second projection (NO BIAS)
        # Weights for the Sigmoid branch (input d_model, output d_ff)
        self.W3 = nn.Parameter(torch.empty(d_model, self.d_ff))

        # Define weights as learnable parameters for the output projection (NO BIAS)
        # Weights for the output layer (input d_ff, output d_model)
        self.W2 = nn.Parameter(torch.empty(self.d_ff, d_model))

        # Initialize parameters using Kaiming uniform for weights
        nn.init.kaiming_uniform_(self.W1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W3, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W2, a=math.sqrt(5))

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
        # Apply the first projection with SiLU activation (NO BIAS)
        proj1_out = self.silu(x @ self.W1)

        # Apply the second projection with Sigmoid activation (NO BIAS)
        proj2_out = x @ self.W3

        # Perform element-wise multiplication (gating mechanism)
        gated_output = proj1_out * proj2_out

        # Apply the final output projection (NO BIAS)
        output = gated_output @ self.W2

        return output