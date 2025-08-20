import math
import torch
import torch.nn # Import torch.nn to access torch.nn.Parameter

def run_gradient_clipping(parameters: list[torch.nn.Parameter], max_norm: float, eps: float = 1e-6):
    """
    Implements gradient clipping for a list of PyTorch neural network parameters.

    This function calculates the L2-norm of the concatenated gradients from all
    given PyTorch `torch.nn.Parameter` objects. If this norm exceeds `max_norm`,
    it scales down each parameter's gradient in-place to ensure the total norm
    is at most `max_norm`.

    Args:
        parameters (list[torch.nn.Parameter]): A list of PyTorch `torch.nn.Parameter`
                                             objects, where each object is expected
                                             to have a '.grad' attribute which is
                                             a PyTorch tensor representing its gradient.
        max_norm (float): The maximum allowed L2-norm for the total gradient.
        eps (float): A small value added for numerical stability in division
                     to prevent division by zero if the norm is exactly zero.
                     Defaults to 1e-6, which is common in frameworks like PyTorch.
    """
    if not parameters:
        return # No parameters to clip

    # 1. Gather all gradients into a list of tensors.
    # We filter out parameters that don't have gradients attached (e.g., if backward
    # has not been called yet for a specific parameter).
    grads = [p.grad for p in parameters if p.grad is not None]

    if not grads:
        return # No gradients found to clip

    # 2. Compute the L2-norm of the combined gradient vector using PyTorch's native function.
    # torch.cat flattens and concatenates all gradient tensors into a single 1D tensor.
    # .norm(2) then computes the L2-norm of this combined tensor.
    grad_norm = torch.norm(torch.cat([g.view(-1) for g in grads]), 2)

    # 3. Apply clipping if the computed norm exceeds the maximum allowed norm.
    if grad_norm > max_norm:
        # Calculate the scaling factor.
        # This is a PyTorch tensor operation for consistency, though it could be float division.
        clip_factor = max_norm / (grad_norm + eps)

        # 4. Scale each parameter's gradient in place.
        # Iterate through the original parameters and modify their gradients directly.
        # For each gradient tensor, multiply it by the calculated clip_factor.
        for p in parameters:
            if p.grad is not None:
                p.grad.mul_(clip_factor) # In-place multiplication by the scalar factor