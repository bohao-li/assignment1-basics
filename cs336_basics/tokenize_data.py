import numpy as np
import os
import logging
from tqdm import tqdm # Import tqdm for progress bar
import json # Will use for mock vocab file generation for consistency in format, but will warn if it's not a standard json

# Import your actual Tokenizer
from cs336_basics.bpe_tokenizer import Tokenizer

# --- Configure logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_line_count(filepath):
    """
    Efficiently counts the number of lines in a file.
    """
    lines = 0
    # Use a buffer for faster reading of large files
    buf_size = 1024 * 1024 # 1MB buffer
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            read_f = f.read
            buf = read_f(buf_size)
            while buf:
                lines += buf.count('\n')
                buf = read_f(buf_size)
    except FileNotFoundError:
        logging.error(f"File not found for line count: {filepath}")
        return 0
    except Exception as e:
        logging.error(f"Error counting lines in {filepath}: {e}")
        return 0
    return lines

def tokenize_data(
    input_text_path: str,
    vocab_filepath: str,
    merges_filepath: str,
    output_bin_path: str,
    sequence_length: int,
    special_tokens: list[str] | None = None,
    dtype=np.uint16
):
    """
    Tokenizes a raw text corpus file into a single binary file.

    Args:
        input_text_path (str): Path to the raw text corpus file (e.g., train.txt or val.txt).
        vocab_filepath (str): Path to the vocabulary file for the tokenizer.
        merges_filepath (str): Path to the merges file for the tokenizer.
        output_bin_path (str): Path to save the tokenized data (binary).
        sequence_length (int): The sequence length used for model training. Used for data length checks.
        special_tokens (list[str] | None): List of special tokens for the tokenizer.
        dtype (np.dtype): Data type for saving token IDs (e.g., np.uint16).
    """
    logging.info(f"Starting tokenization for corpus: {input_text_path}")

    # Load Tokenizer
    tokenizer = Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens)
    vocab_size = tokenizer.vocab_size()
    logging.info(f"Tokenizer loaded. Vocabulary size: {vocab_size}")

    # Read and tokenize the corpus with a progress bar
    encoded_ids = []
    if not os.path.exists(input_text_path):
        raise FileNotFoundError(f"Corpus file not found: {input_text_path}")

    total_lines = get_line_count(input_text_path)
    if total_lines == 0:
        logging.warning(f"Input file {input_text_path} is empty or unreadable.")
        # If the file is empty, we might still need to create a dummy bin file for the loader
        # Or, ideally, the user ensures there's enough data. For now, exit.
        return

    with open(input_text_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in tqdm(f, total=total_lines, desc=f"Tokenizing {os.path.basename(input_text_path)}"):
            encoded_ids.extend(tokenizer.encode(line))

    if not encoded_ids:
        logging.warning(f"No tokens generated from {input_text_path}. Please check corpus content or tokenizer.")
        return # Exit if no tokens are produced

    all_tokens_np = np.array(encoded_ids, dtype=dtype)
    logging.info(f"Corpus tokenized. Total tokens: {len(all_tokens_np)}")

    # Ensure enough data to form at least one full sequence
    min_required_tokens = sequence_length + 1
    if len(all_tokens_np) < min_required_tokens:
        logging.warning(f"Data in {input_text_path} is too short ({len(all_tokens_np)} tokens) "
                        f"for sequence_length ({sequence_length}). "
                        f"Appending random tokens to reach minimum ({min_required_tokens}).")
        # Ensure that random tokens are within the valid vocab_size range
        all_tokens_np = np.concatenate((all_tokens_np, np.random.randint(0, vocab_size, size=(min_required_tokens - len(all_tokens_np),), dtype=dtype)))
        logging.info(f"Data padded to {len(all_tokens_np)} tokens.")


    # Save to binary file
    all_tokens_np.tofile(output_bin_path)
    logging.info(f"Tokenized data saved to: {output_bin_path} (Tokens: {len(all_tokens_np)})")

if __name__ == "__main__":
    data_dir = "../data"
    os.makedirs(data_dir, exist_ok=True)

    # Configuration for tokenization
    train_text_path = os.path.join(data_dir, "TinyStoriesV2-GPT4-train.txt")
    val_text_path = os.path.join(data_dir, "TinyStoriesV2-GPT4-valid.txt")
    vocab_filepath = os.path.join(data_dir, "bpe_vocab.json")
    merges_filepath = os.path.join(data_dir, "bpe_merges.txt")
    train_bin_path = os.path.join(data_dir, "train.bin")
    val_bin_path = os.path.join(data_dir, "val.bin")
    special_tokens = ["<|endoftext|>"]
    sequence_length = 128 # Should match your training config's sequence_length

    # Run the tokenization for training data
    # tokenize_data(
    #     train_text_path,
    #     vocab_filepath,
    #     merges_filepath,
    #     train_bin_path,
    #     sequence_length,
    #     special_tokens
    # )

    # Run the tokenization for validation data
    tokenize_data(
        val_text_path,
        vocab_filepath,
        merges_filepath,
        val_bin_path,
        sequence_length,
        special_tokens
    )
