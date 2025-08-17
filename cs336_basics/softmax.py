import torch

def run_softmax(input_tensor: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Applies the softmax operation to a tensor along a specified dimension,
    incorporating a numerical stability trick.

    The numerical stability trick involves subtracting the maximum value
    in the specified dimension from all elements before exponentiation.
    This prevents potential overflow issues when dealing with large
    positive numbers in the exponent.

    Args:
        input_tensor (torch.Tensor): The input tensor to apply softmax to.
        dim (int): The dimension along which to apply the softmax.

    Returns:
        torch.Tensor: The output tensor with the softmax applied,
                      having the same shape as the input_tensor,
                      with probabilities summing to 1 along the specified dimension.
    """
    # 1. Subtract the maximum value for numerical stability
    # This step is crucial. If input values (logits) are very large,
    # exp(x) can lead to numerical overflow (infinity). Subtracting the
    # maximum shifts the values so that the largest value becomes 0,
    # and all others become negative. This ensures that exp(x_shifted)
    # remains within representable floating-point limits.
    max_values, _ = input_tensor.max(dim=dim, keepdim=True)
    shifted_input = input_tensor - max_values

    # 2. Apply the exponential function
    # All values in shifted_input are now non-positive. exp(non-positive)
    # will result in values between 0 (exclusive) and 1 (inclusive).
    exp_shifted_input = torch.exp(shifted_input)

    # 3. Calculate the sum of exponentials along the specified dimension
    # This sum will be used as the normalization factor.
    sum_exp_shifted_input = exp_shifted_input.sum(dim=dim, keepdim=True)

    # 4. Divide each exponential by the sum of exponentials
    # This normalizes the values into a probability distribution, where
    # each element is between 0 and 1, and the sum along the 'dim' is 1.
    softmax_output = exp_shifted_input / sum_exp_shifted_input

    return softmax_output