import os
import torch
import typing

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: typing.Union[str, os.PathLike, typing.BinaryIO, typing.IO[bytes]]
) -> None:
    """
    Dumps the state of the model, optimizer, and iteration to a file-like object or path.

    Args:
        model: The PyTorch model to save.
        optimizer: The optimizer to save.
        iteration: The current iteration number.
        out: The path or file-like object where the checkpoint will be saved.
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }
    torch.save(checkpoint, out)

def load_checkpoint(
    src: typing.Union[str, os.PathLike, typing.BinaryIO, typing.IO[bytes]],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer
) -> int:
    """
    Loads a checkpoint from a source and restores the model and optimizer states.

    Args:
        src: The path or file-like object from which to load the checkpoint.
        model: The model whose state will be restored.
        optimizer: The optimizer whose state will be restored.

    Returns:
        The iteration number saved in the checkpoint.
    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    iteration = checkpoint['iteration']
    return iteration