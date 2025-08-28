import numpy as np
import os
import logging
from tqdm import tqdm  # Import tqdm for progress bar
import json  # Will use for mock vocab file generation for consistency in format, but will warn if it's not a standard json

# Import your actual Tokenizer
from cs336_basics.bpe_tokenizer import Tokenizer

# --- Configure logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
# Insert at the top of the file
import multiprocessing as mp
from functools import partial


# New function to tokenize a single chunk of text
def tokenize_chunk(chunk, tokenizer):
    """Tokenizes a single chunk of text and returns the encoded IDs."""
    encoded_ids = []
    for line in chunk:
        encoded_ids.extend(tokenizer.encode(line))
    return encoded_ids


def tokenize_data_parallel(
    input_text_path: str,
    vocab_filepath: str,
    merges_filepath: str,
    output_bin_path: str,
    sequence_length: int,
    special_tokens: list[str] | None = None,
    dtype=np.uint16,
    num_processes: int | None = None,
):
    """
    Tokenizes a raw text corpus file into a single binary file using parallel processing.

    Args:
        input_text_path (str): Path to the raw text corpus file.
        vocab_filepath (str): Path to the vocabulary file.
        merges_filepath (str): Path to the merges file.
        output_bin_path (str): Path to save the tokenized data (binary).
        sequence_length (int): The sequence length for data length checks.
        special_tokens (list[str] | None): List of special tokens.
        dtype (np.dtype): Data type for saving token IDs.
        num_processes (int | None): Number of processes to use. Defaults to os.cpu_count().
    """
    logging.info(f"Starting parallel tokenization for corpus: {input_text_path}")

    # Load Tokenizer once per process
    tokenizer = Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens)
    vocab_size = tokenizer.vocab_size()
    logging.info(f"Tokenizer loaded. Vocabulary size: {vocab_size}")

    if not os.path.exists(input_text_path):
        raise FileNotFoundError(f"Corpus file not found: {input_text_path}")

    # Determine number of processes to use
    if num_processes is None:
        num_processes = os.cpu_count()
    logging.info(f"Using {num_processes} processes for tokenization.")

    # Read the entire file and split it into chunks
    # Note: For very large files, a memory-efficient approach would be to chunk by byte position
    # and seek in each process, but reading all lines at once is simpler for many cases.
    with open(input_text_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    total_lines = len(lines)
    if total_lines == 0:
        logging.warning(f"Input file {input_text_path} is empty or unreadable.")
        return

    chunk_size = total_lines // num_processes
    chunks = [lines[i : i + chunk_size] for i in range(0, total_lines, chunk_size)]

    # Use a multiprocessing pool to parallelize tokenization
    with mp.Pool(processes=num_processes) as pool:
        # The partial function is used to pass the tokenizer object to the worker processes
        tokenize_func = partial(tokenize_chunk, tokenizer=tokenizer)
        results = list(
            tqdm(
                pool.imap(tokenize_func, chunks),
                total=len(chunks),
                desc=f"Parallel Tokenizing {os.path.basename(input_text_path)}",
            )
        )

    # Combine results from all processes
    encoded_ids = [token for sublist in results for token in sublist]

    if not encoded_ids:
        logging.warning(f"No tokens generated from {input_text_path}.")
        return

    all_tokens_np = np.array(encoded_ids, dtype=dtype)
    logging.info(f"Corpus tokenized. Total tokens: {len(all_tokens_np)}")

    # Ensure enough data to form at least one full sequence
    min_required_tokens = sequence_length + 1
    if len(all_tokens_np) < min_required_tokens:
        logging.warning(
            f"Data is too short ({len(all_tokens_np)} tokens) for sequence_length ({sequence_length})."
        )
        all_tokens_np = np.concatenate(
            (
                all_tokens_np,
                np.random.randint(
                    0,
                    vocab_size,
                    size=(min_required_tokens - len(all_tokens_np),),
                    dtype=dtype,
                ),
            )
        )
        logging.info(f"Data padded to {len(all_tokens_np)} tokens.")

    # Save to binary file
    all_tokens_np.tofile(output_bin_path)
    logging.info(
        f"Tokenized data saved to: {output_bin_path} (Tokens: {len(all_tokens_np)})"
    )


# --- Modify the __main__ block ---
if __name__ == "__main__":
    data_dir = "../data"
    os.makedirs(data_dir, exist_ok=True)

    train_text_path = os.path.join(data_dir, "TinyStoriesV2-GPT4-train.txt")
    val_text_path = os.path.join(data_dir, "TinyStoriesV2-GPT4-valid.txt")
    vocab_filepath = os.path.join(data_dir, "bpe_vocab.json")
    merges_filepath = os.path.join(data_dir, "bpe_merges.txt")
    train_bin_path = os.path.join(data_dir, "train.bin")
    val_bin_path = os.path.join(data_dir, "val.bin")
    special_tokens = ["<|endoftext|>"]
    sequence_length = 128

    # Use the parallelized function
    tokenize_data_parallel(
        train_text_path,
        vocab_filepath,
        merges_filepath,
        train_bin_path,
        sequence_length,
        special_tokens,
        num_processes=8
    )

    tokenize_data_parallel(
        val_text_path,
        vocab_filepath,
        merges_filepath,
        val_bin_path,
        sequence_length,
        special_tokens,
        num_processes=8
    )
