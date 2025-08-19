import torch
import torch.nn.functional as F

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute cross entropy loss: ℓi = -log softmax(oi)[xi+1]
    
    Args:
        logits: Tensor of shape (..., vocab_size) - predicted logits oi
        targets: Tensor of shape (...,) - target token indices xi+1
    
    Returns:
        Tensor: Average cross entropy loss across batch dimensions (scalar)
    """
    # Step 1: Subtract max for numerical stability
    # This prevents overflow in exp() while keeping softmax unchanged
    max_logits = torch.max(logits, dim=-1, keepdim=True)[0]
    stable_logits = logits - max_logits
    
    # Step 2: Compute log-sum-exp for the denominator
    # log(sum(exp(stable_logits))) = log(sum(exp(logits - max))) 
    log_sum_exp = torch.logsumexp(stable_logits, dim=-1)
    
    # Step 3: Extract logits for target tokens
    # We need to gather the logit values at the target indices
    # Using advanced indexing to handle arbitrary batch dimensions
    target_logits = stable_logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    
    # Step 4: Compute cross entropy using log-softmax formula
    # -log softmax(oi)[xi+1] = -log(exp(oi[xi+1]) / sum(exp(oi)))
    # = -log(exp(oi[xi+1])) + log(sum(exp(oi)))
    # = -oi[xi+1] + log(sum(exp(oi)))
    # With stability: -(oi[xi+1] - max) + log(sum(exp(oi - max)))
    cross_entropy_loss = -target_logits + log_sum_exp
    
    # Step 5: Average across all batch dimensions
    return torch.mean(cross_entropy_loss)