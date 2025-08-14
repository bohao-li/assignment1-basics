import torch
import torch.nn as nn
import torch.nn.init as init

class Linear(nn.Module):
    """
    Implements a linear transformation layer similar to torch.nn.Linear,
    but without a bias term.
    """
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        """
        Constructs a linear transformation module.

        Args:
            in_features (int): The final dimension of the input.
            out_features (int): The final dimension of the output.
            device (torch.device | None): Device to store the parameters on (e.g., 'cuda', 'cpu').
            dtype (torch.dtype | None): Data type of the parameters (e.g., torch.float32).
        """
        # Call the superclass constructor for nn.Module
        super().__init__()

        # Store in_features and out_features for potential future reference
        self.in_features = in_features
        self.out_features = out_features

        # Create the weight parameter W. It's stored as (out_features, in_features)
        # because during matrix multiplication, input x W_transpose is common.
        # However, the prompt specifically asks to store it as W (not W^T) for memory
        # ordering, meaning a direct matrix multiplication (input @ W) would require
        # W to be (in_features, out_features).
        # We will create W as (out_features, in_features) and then perform (x @ W.T)
        # to get (batch_size, out_features).
        self.W = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )

        # Initialize the weights using trunc_normal_
        # According to the prompt, we use init.trunc_normal_ to initialize W.
        # trunc_normal_ takes (mean, std, a, b) for its distribution.
        # For typical linear layers, a common initialization is using kaiming uniform
        # or xavier uniform, which considers fan-in/fan-out.
        # For trunc_normal_, common practice often involves small mean and std.
        # Here, we'll use mean=0.0 and std=0.02, which are common for some models.
        # The 'a' and 'b' parameters define the truncation bounds.
        # A common practice is to truncate at 2 times the standard deviation.
        init.trunc_normal_(self.W, mean=0.0, std=0.02, a=-0.04, b=0.04)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the linear transformation to the input.

        Args:
            x (torch.Tensor): The input tensor. Its last dimension must match in_features.

        Returns:
            torch.Tensor: The output tensor after the linear transformation.
                          Its last dimension will match out_features.
        """
        # Perform the linear transformation: output = input @ W^T
        # Since we stored W as (out_features, in_features),
        # we need to transpose it to (in_features, out_features) for multiplication
        # with an input x of shape (..., in_features).
        # (..., in_features) @ (in_features, out_features) -> (..., out_features)
        return torch.matmul(x, self.W.T) # Or x @ self.W.T if you prefer the operator