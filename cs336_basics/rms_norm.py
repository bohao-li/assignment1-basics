import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        Construct the RMSNorm module.

        Args:
            d_model (int): Hidden dimension of the model.
            eps (float): Epsilon value for numerical stability.
            device (torch.device | None): Device to store the parameters on.
            dtype (torch.dtype | None): Data type of the parameters.
        """
        super().__init__()
        
        self.d_model = d_model
        self.eps = eps
        
        self.weight = nn.Parameter(
            torch.empty(d_model, device=device, dtype=dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process an input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: The normalized tensor.
        """
        
        in_dtype = x.dtype
        x = x.to(torch.float32)
        
        # Calculate the Root Mean Square (RMS) of the input tensor 'x'.
        # This involves squaring the tensor, taking the mean along the last dimension,
        # adding epsilon for numerical stability, and then taking the square root.
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        
        # The final result is the normalized tensor scaled by the learnable parameter 'G'.
        result = self.weight * x / rms
        
        return result.to(in_dtype)
        
        