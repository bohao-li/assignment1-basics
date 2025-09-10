import numpy as np
import torch
import torch.optim as optim
import os
import logging
from datetime import datetime
# import regex as re # Removed as it's not used in this script

# Import your actual Tokenizer and TransformerLanguageModel
# Ensure these are correctly set up in your cs336_basics package
from cs336_basics.transformer_lm import TransformerLanguageModel


# --- Configure logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def transformer_lm(vocab_size, d_model, num_layers, context_length, num_heads, d_ff, rope_theta, device, dtype):
    """
    Instantiates your TransformerLanguageModel.
    """
    logging.info(f"Instantiating TransformerLanguageModel with vocab_size={vocab_size}, d_model={d_model}, "
                 f"num_layers={num_layers}, context_length={context_length}, num_heads={num_heads}, "
                 f"d_ff={d_ff}, rope_theta={rope_theta}, device={device}, dtype={dtype}...")
    return TransformerLanguageModel(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        context_length=context_length,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
        device=device,
        dtype=dtype,
    )

# --- Data Loading with np.memmap ---
class MemoryMappedDataset(torch.utils.data.Dataset):
    def __init__(self, filepath, sequence_length, dtype=np.uint16):
        """
        Initializes the dataset using np.memmap for memory-efficient loading.
        Assumes the data file is a flat array of token IDs.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")

        logging.info(f"Loading data from {filepath} with np.memmap...")
        self.data = np.memmap(filepath, dtype=dtype, mode='r')
        self.sequence_length = sequence_length
        # The -1 is because we need a target for each input token
        self.num_sequences = (len(self.data) - 1) // sequence_length
        logging.info(f"Memory-mapped dataset loaded. Total tokens: {len(self.data)}, "
                     f"Total sequences available: {self.num_sequences}")

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        # Start and end indices for the input sequence
        start_idx = idx * self.sequence_length
        end_idx = start_idx + self.sequence_length
        # Input sequence (tokens 0 to L-1)
        input_seq = torch.from_numpy(self.data[start_idx:end_idx].astype(np.int64))
        # Target sequence (tokens 1 to L)
        target_seq = torch.from_numpy(self.data[start_idx+1:end_idx+1].astype(np.int64))
        return input_seq, target_seq

# --- Training Configuration (Hyperparameters) ---
class TrainingConfig:
    def __init__(self,
                 model_name="my_transformer_lm",
                 vocab_filepath="", # Not strictly used in this script but good to keep
                 merges_filepath="", # Not strictly used in this script but good to keep
                 special_tokens=["<|endoftext|>"], # Not strictly used in this script but good to keep
                 train_data_path="",
                 val_data_path="",
                 checkpoint_dir="./checkpoints",
                 log_interval_steps=10, # Reduced for mini run
                 eval_interval_steps=20, # Reduced for mini run
                 vocab_size=12000,
                 epochs=2, # Reduced for mini run
                 batch_size=16, # Reduced for mini run
                 sequence_length=64, # Reduced for mini run, context_length for TransformerLM
                 d_model=64, # Reduced for mini run
                 num_layers=2, # Reduced for mini run
                 num_heads=4, # Reduced for mini run
                 d_ff=128, # Reduced for mini run
                 rope_theta=10000.0,
                 learning_rate=3e-4,
                 weight_decay=0.01,
                 gradient_accumulation_steps=1,
                 max_grad_norm=1.0,
                 # Prioritize 'mps' for Mac, then 'cuda', then 'cpu'
                 device="mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"),
                 dtype=torch.float32):
        self.model_name = model_name
        self.vocab_filepath = vocab_filepath
        self.merges_filepath = merges_filepath
        self.special_tokens = special_tokens
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.checkpoint_dir = checkpoint_dir
        self.log_interval_steps = log_interval_steps
        self.eval_interval_steps = eval_interval_steps
        self.vocab_size = vocab_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.dtype = dtype

        os.makedirs(self.checkpoint_dir, exist_ok=True) # Ensure checkpoint directory exists

    def __str__(self):
        return "\n".join([f"{k}: {v}" for k, v in self.__dict__.items()])

# --- Training Function ---
def train(config: TrainingConfig):
    """
    Runs the main training loop.

    Args:
        config (TrainingConfig): Configuration object containing all hyperparameters.
    """
    logging.info(f"Starting training with the following configuration:\n{config}")

    # 2. Initialize Model Components
    model = transformer_lm(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        num_layers=config.num_layers,
        context_length=config.sequence_length, # context_length maps to sequence_length
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        rope_theta=config.rope_theta,
        device=config.device,
        dtype=config.dtype,
    ).to(config.device)

    # 3. Optimizer and Loss Function
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1) # -1 can be used for padding if needed

    # 4. Data Loaders
    # Using the same data file for both train and val for a quick test run
    train_dataset = MemoryMappedDataset(config.train_data_path, config.sequence_length)
    val_dataset = MemoryMappedDataset(config.val_data_path, config.sequence_length)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0, # Keep at 0 for simplicity in a mini test run on MPS
        pin_memory=True # For faster data transfer to GPU
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    logging.info(f"Number of training batches: {len(train_loader)}")
    logging.info(f"Number of validation batches: {len(val_loader)}")

    global_step = 0
    best_val_loss = float('inf')

    for epoch in range(config.epochs):
        model.train()
        total_train_loss = 0
        train_batch_count = 0
        optimizer.zero_grad() # Initialize gradients

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(config.device), targets.to(config.device)

            # Forward pass through the transformer LM
            # logits shape: (batch_size, sequence_length, vocab_size)
            logits = model(inputs)

            # Reshape logits to (batch_size * sequence_length, vocab_size)
            # Reshape targets to (batch_size * sequence_length)
            loss = loss_fn(logits.view(-1, config.vocab_size), targets.view(-1))

            # Normalize loss by gradient accumulation steps
            loss = loss / config.gradient_accumulation_steps
            loss.backward()

            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad() # Reset gradients after step

                global_step += 1
                total_train_loss += loss.item() * config.gradient_accumulation_steps # Revert normalization for logging
                train_batch_count += 1

                if global_step % config.log_interval_steps == 0:
                    avg_loss = total_train_loss / train_batch_count if train_batch_count > 0 else 0
                    logging.info(f"Epoch {epoch+1}, Step {global_step}, Train Loss: {avg_loss:.4f}")
                    total_train_loss = 0
                    train_batch_count = 0

                # if global_step % config.eval_interval_steps == 0:
                #     val_loss = evaluate(model, val_loader, loss_fn, config, config.vocab_size)
                #     logging.info(f"Epoch {epoch+1}, Step {global_step}, Validation Loss: {val_loss:.4f}")

                    # Save checkpoint if validation loss improves
                    # if val_loss < best_val_loss:
                    #     best_val_loss = val_loss
                    #     save_checkpoint(model, optimizer, config, global_step, best_val_loss)

    logging.info("Training complete.")


def evaluate(model, data_loader, loss_fn, config: TrainingConfig, vocab_size: int):
    """
    Evaluates the model on a given dataset.
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(config.device), targets.to(config.device)

            logits = model(inputs)

            loss = loss_fn(logits.view(-1, vocab_size), targets.view(-1))
            total_loss += loss.item() * inputs.size(0) # Multiply by batch size for correct average

    avg_loss = total_loss / len(data_loader.dataset)
    model.train() # Set model back to training mode
    return avg_loss

def save_checkpoint(model, optimizer, config: TrainingConfig, step, val_loss):
    """
    Serializes model and optimizer states to a user-provided path.
    """
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_name = f"{config.model_name}_step_{step}_loss_{val_loss:.4f}_{timestamp}.pt"
    checkpoint_path = os.path.join(config.checkpoint_dir, checkpoint_name)

    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'global_step': step,
        'val_loss': val_loss,
        'config': config.__dict__, # Save config for reproducibility
    }
    torch.save(state, checkpoint_path)
    logging.info(f"Checkpoint saved to: {checkpoint_path}")

def main():
    """
    Main entry point for the training script.
    """
    # Create a configuration object with mini hyperparameters and MPS device
    config = TrainingConfig(
        epochs=2, # Very short run
        batch_size=8, # Small batch size
        sequence_length=32, # Short sequences
        vocab_size=12000, # Assuming this is your actual vocab size
        d_model=32, # Tiny model dimension
        num_layers=2, # Very few layers
        num_heads=4, # Few attention heads
        d_ff=64, # Small feed-forward dimension
        rope_theta=10000.0,
        learning_rate=1e-4,
        # IMPORTANT: Adjust these paths to your actual small binary data files
        train_data_path="data/val.bin", # Using val.bin for both for a quick test
        val_data_path="data/val.bin",
        checkpoint_dir="./my_model_checkpoints",
        device="mps"
    )

    logging.info(f"Using device: {config.device}")
    if config.device == "mps":
        logging.info("MPS (Metal Performance Shaders) is available and will be used.")
    elif config.device == "cuda":
        logging.info("CUDA is available and will be used.")
    else:
        logging.info("No GPU detected, training on CPU.")


    # Run the training loop
    train(config)

if __name__ == "__main__":
    main()
