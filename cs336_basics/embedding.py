import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        """
        Construct an embedding module.

        Args:
            num_embeddings (int): Size of the vocabulary.
            embedding_dim (int): Dimension of the embedding vectors.
            device (torch.device | None): Device to store the parameters on.
            dtype (torch.dtype | None): Data type of the parameters.
        """
        super().__init__()
        # You'll likely want to use a torch.nn.Parameter to store the embedding matrix.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        self.W = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        # It's a good practice to initialize parameters, for example, using a uniform distribution.
        nn.init.uniform_(self.W, -1.0 / self.embedding_dim, 1.0 / self.embedding_dim)


    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Lookup the embedding vectors for the given token IDs.

        Args:
            token_ids (torch.Tensor): A tensor of token IDs.

        Returns:
            torch.Tensor: The corresponding embedding vectors.
        """
        return self.W[token_ids]