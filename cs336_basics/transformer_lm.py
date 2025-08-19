import torch
import torch.nn as nn

from cs336_basics.transformer_block import TransformerBlock
from cs336_basics.rms_norm import RMSNorm


class TransformerLanguageModel(nn.Module):
    """
    A Transformer-based language model.

    This class implements the core components of a Transformer model, including
    token and position embeddings, a stack of Transformer blocks, and a final
    linear projection for next-token prediction.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int,
        context_length: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: str = "cpu",
        dtype=torch.float32,
    ):
        """
        Initializes the TransformerLanguageModel.

        Args:
            vocab_size (int): The size of the vocabulary.
            d_model (int): The dimensionality of the token embeddings and the model's
                           internal representations.
            num_layers (int): The number of Transformer blocks to stack.
            context_length (int): The maximum sequence length.
            num_heads (int): The number of attention heads in the multi-head
                             attention mechanism.
            d_ff (int): The dimensionality of the hidden layer in the
                        feed-forward network.
            dropout (float): The dropout rate to apply for regularization.
            device (str): The device on which to run the model (e.g., 'cpu', 'cuda').
            dtype (torch.dtype): The data type to use for model parameters.
        """
        super().__init__()

        self.token_embeddings = nn.Embedding(
            vocab_size, d_model, device=device, dtype=dtype
        )

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    context_length,
                    rope_theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

        # The final normalization layer is now correctly defined as a module.
        # It's an RMSNorm, which is often implemented as a linear layer or a custom module.
        # To match the state dict, we'll use a Linear module without a bias, as is
        # common for normalization layers.
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)

        # The output projection or "language model head"
        self.lm_head = nn.Linear(
            d_model, vocab_size, bias=False, device=device, dtype=dtype
        )

    def forward(self, input_ids):
        """
        Defines the forward pass of the model.
        """
        x = self.token_embeddings(input_ids)
        for layer in self.layers:
            x = layer(x)

        x = self.ln_final(x)
        logits = self.lm_head(x)

        return logits
