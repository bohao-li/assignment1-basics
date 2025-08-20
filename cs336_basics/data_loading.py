import numpy as np
import torch

def get_batch(x: np.ndarray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Loads a batch of data for training a language model.

    This function takes a NumPy array of token IDs, samples random starting
    positions, and extracts input sequences and their corresponding next-token
    targets. Both tensors are then moved to the specified PyTorch device.

    Args:
        x (np.ndarray): A 1D NumPy array (dtype=int) containing token IDs.
                        This represents the entire dataset of token IDs.
        batch_size (int): The number of independent sequences to sample for the batch.
        context_length (int): The length of each input sequence (and target sequence).
                              This is often referred to as block_size or sequence_length.
        device (str): The PyTorch device string (e.g., 'cpu', 'cuda:0').
                      The returned tensors will be placed on this device.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A pair of PyTorch tensors:
            - input_sequences (torch.Tensor): Tensor of shape (batch_size, context_length)
                                              containing the sampled input token IDs.
            - target_sequences (torch.Tensor): Tensor of shape (batch_size, context_length)
                                               containing the next-token targets for each
                                               input sequence.
            Both tensors are of type torch.long and are placed on the specified device.
    """
    # Ensure the input data is long enough to sample at least one full context length.
    # If not, it means we can't create valid input/target pairs of the specified length.
    if len(x) < context_length + 1:
        raise ValueError(
            f"Input array length ({len(x)}) is too short for the requested "
            f"context_length ({context_length}). Need at least {context_length + 1} tokens."
        )

    # Generate random starting indices for each sequence in the batch.
    # We sample from 0 up to len(x) - context_length - 1 to ensure that both
    # the input sequence (context_length tokens) and the target sequence
    # (the next context_length tokens) can be fully extracted.
    # For example, if context_length is 8, and x has 100 tokens, the last
    # possible start index for input is 100 - 8 = 92 (indices 92 to 99).
    # The last target token would then be x[92+8] = x[100], which is out of bounds.
    # So, we need to ensure there's enough room for x[start_index + context_length].
    # Therefore, the maximum start_index should be len(x) - context_length.
    # This ensures that the *last* token of the *target* sequence (x[idx + context_length])
    # is within the bounds of `x`.
    ix = torch.randint(low=0, high=len(x) - context_length, size=(batch_size,))

    # Extract input sequences and convert to PyTorch tensor.
    # For each sampled index 'i' in 'ix', we take 'x[i : i + context_length]' as the input.
    input_sequences = torch.stack([torch.from_numpy(x[i : i + context_length]) for i in ix])

    # Extract target sequences and convert to PyTorch tensor.
    # For each sampled index 'i' in 'ix', we take 'x[i + 1 : i + 1 + context_length]' as the target.
    # This means the target for input_sequences[k] is the token that immediately follows
    # each token in input_sequences[k].
    target_sequences = torch.stack([torch.from_numpy(x[i + 1 : i + 1 + context_length]) for i in ix])

    # Move both tensors to the specified device and ensure they are of type torch.long.
    input_sequences = input_sequences.to(device=device, dtype=torch.long)
    target_sequences = target_sequences.to(device=device, dtype=torch.long)

    return input_sequences, target_sequences