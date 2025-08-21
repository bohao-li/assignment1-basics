import os
import regex as re
import time
from collections import defaultdict
from typing import Dict, Tuple, Set, BinaryIO
from tqdm import tqdm


class BPETokenizerTrainer:
    """
    A class to train a Byte Pair Encoding (BPE) tokenizer with progress indicators.
    """

    # Regex pattern for initial pre-tokenization
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def __init__(
        self,
        initial_vocab_size: int = 256,
        special_tokens: Set[str] = {"<|endoftext|>"},
    ) -> None:
        """
        Initializes the BPE tokenizer trainer.

        Args:
            initial_vocab_size: The starting vocabulary size (typically 256 for all bytes).
            special_tokens: A set of special token strings to be included in the vocabulary.
        """
        self.merges: list[Tuple[bytes, bytes]] = (
            []
        )  # Stores the learned merge rules as bytes
        self.vocabulary: Dict[int, bytes] = {
            i: bytes([i]) for i in range(initial_vocab_size)
        }  # For decoding

        # New: Store special tokens and their byte representations
        self.special_tokens_set: Set[str] = special_tokens
        self.special_tokens_bytes: Dict[int, bytes] = {}

        # New: Add special tokens to the vocabulary and update the vocab count
        self.vocab_count = initial_vocab_size
        for token_str in sorted(list(self.special_tokens_set)):
            token_bytes = token_str.encode("utf-8")
            self.vocabulary[self.vocab_count] = token_bytes
            self.special_tokens_bytes[self.vocab_count] = token_bytes
            self.vocab_count += 1

        # The split token is the first one in the sorted set
        self.split_special_token: bytes = sorted(
            [t.encode("utf-8") for t in self.special_tokens_set]
        )[0]

    def _find_chunk_boundaries(
        self,
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_token: bytes,
    ) -> list[int]:
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """
        assert isinstance(
            split_special_token, bytes
        ), "Must represent special token as a bytestring"

        print("🔍 Finding chunk boundaries...")
        
        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks
        print(f"📊 File size: {file_size:,} bytes, Target chunks: {desired_num_chunks}, Chunk size: ~{chunk_size:,} bytes")

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        # Progress bar for boundary finding
        with tqdm(total=len(chunk_boundaries)-2, desc="Finding boundaries", unit="boundary") as pbar:
            for bi in range(1, len(chunk_boundaries) - 1):
                initial_position = chunk_boundaries[bi]
                file.seek(initial_position)  # Start at boundary guess
                while True:
                    mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                    # If EOF, this boundary should be at the end of the file
                    if mini_chunk == b"":
                        chunk_boundaries[bi] = file_size
                        break

                    # Find the special token in the mini chunk
                    found_at = mini_chunk.find(split_special_token)
                    if found_at != -1:
                        chunk_boundaries[bi] = initial_position + found_at
                        break
                    initial_position += mini_chunk_size
                pbar.update(1)

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        unique_boundaries = sorted(set(chunk_boundaries))
        print(f"✅ Found {len(unique_boundaries)-1} unique chunks")
        return unique_boundaries

    def _collect_token_pairs(
        self, token_sequences_with_counts: Dict[Tuple[int, ...], int]
    ) -> Dict[Tuple[int, int], int]:
        """
        Collects and counts all unique bigrams (pairs of adjacent integers representing tokens)
        from a dictionary where keys are tuples of integers and values are their counts.
        """
        bigram_counts: Dict[Tuple[int, int], int] = defaultdict(int)
        
        # Add progress bar for bigram collection
        with tqdm(total=len(token_sequences_with_counts), desc="Collecting bigrams", unit="seq", leave=False) as pbar:
            for token_sequence, token_frequency in token_sequences_with_counts.items():
                for i in range(len(token_sequence) - 1):
                    bigram = (token_sequence[i], token_sequence[i + 1])
                    bigram_counts[bigram] += token_frequency
                pbar.update(1)
        
        return dict(bigram_counts)

    def _get_most_used_bigram(
        self, bigrams: Dict[Tuple[int, int], int]
    ) -> Tuple[int, int]:
        """
        Finds the most frequently used bigram. When ties occur in frequency,
        it prefers the lexicographically greater pair.
        """
        if not bigrams:
            return (0, 0)

        # We first find the maximum frequency.
        max_frequency = max(bigrams.values())

        # Then we collect all bigrams that have this maximum frequency.
        tied_bigrams = [
            bigram for bigram, freq in bigrams.items() if freq == max_frequency
        ]

        # If there's more than one, we break the tie.
        if len(tied_bigrams) > 1:
            most_used_bigram = max(
                tied_bigrams,
                key=lambda bigram: (
                    self.vocabulary[bigram[0]],
                    self.vocabulary[bigram[1]],
                ),
            )
            return most_used_bigram
        else:
            # No tie, so there's only one bigram with the max frequency.
            return tied_bigrams[0]

    def _merge_pair_in_sequence(
        self,
        sequence: Tuple[int, ...],
        bigram_to_merge: Tuple[int, int],
        new_token_id: int,
    ) -> Tuple[int, ...]:
        """
        Replaces all occurrences of bigram_to_merge in a single sequence with new_token_id.
        """
        b1, b2 = bigram_to_merge
        new_sequence = []
        i = 0
        while i < len(sequence):
            if i + 1 < len(sequence) and sequence[i] == b1 and sequence[i + 1] == b2:
                new_sequence.append(new_token_id)
                i += 2
            else:
                new_sequence.append(sequence[i])
                i += 1
        return tuple(new_sequence)

    def _find_all_occurrences(
        self, sequence: Tuple[int, ...], bigram: Tuple[int, int]
    ) -> list[int]:
        """
        Helper function to find all occurrences of a bigram within a single sequence.
        Returns a list of starting indices of the bigram.
        """
        occurrences = []
        i = 0
        while i < len(sequence) - 1:
            if (sequence[i], sequence[i + 1]) == bigram:
                occurrences.append(i)
                i += 2  # Jump past the merged pair
            else:
                i += 1
        return occurrences
    
    def save_merges_and_vocab(self, merges_file, vocab_file):
        """
        Save the BPE merges and vocabulary to files with each item on a new line.
        
        Args:
            merges_file (str): Path to save the merges file
            vocab_file (str): Path to save the vocabulary file
        """
        print("💾 Saving merges and vocabulary...")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(merges_file), exist_ok=True)
        os.makedirs(os.path.dirname(vocab_file), exist_ok=True)
        
        # Save merges - each merge pair on a new line with progress
        with open(merges_file, 'w', encoding='utf-8') as f:
            with tqdm(total=len(self.merges), desc="Saving merges", unit="merge") as pbar:
                for merge in self.merges:
                    f.write(str(merge) + "\n")
                    pbar.update(1)
        
        # Save vocabulary - each key-value pair on a new line with progress
        with open(vocab_file, 'w', encoding='utf-8') as f:
            with tqdm(total=len(self.vocabulary), desc="Saving vocab", unit="token") as pbar:
                for key, value in self.vocabulary.items():
                    f.write(f"{key}: {repr(value)}\n")
                    pbar.update(1)
        
        print(f"✅ Merges saved to: {merges_file}")
        print(f"✅ Vocabulary saved to: {vocab_file}")

    def train(
        self, data_file_path: str, target_vocab_size: int = 3000, num_processes: int = 4
    ):
        """
        Trains the BPE tokenizer with progress indicators.

        Args:
            data_file_path: Path to the text file for training.
            target_vocab_size: The desired final vocabulary size.
            num_processes: Number of processes to use for initial chunking (serial in this example).
        """
        start_time = time.time()
        print(f"🚀 Starting BPE training...")
        print(f"📁 Input file: {data_file_path}")
        print(f"🎯 Target vocab size: {target_vocab_size:,}")
        print(f"🔢 Starting vocab size: {self.vocab_count:,}")
        print(f"📈 Merges needed: {target_vocab_size - self.vocab_count:,}")
        
        pretoken_counts: Dict[str, int] = defaultdict(int)

        # 1. Initial Pre-tokenization and Byte Conversion
        print("\n📖 Phase 1: Pre-tokenization and chunking")
        with open(data_file_path, "rb") as f:
            # New: Use the stored split token
            boundaries = self._find_chunk_boundaries(
                f, num_processes, self.split_special_token
            )
            
            special_token_pattern = "|".join(re.escape(token) for token in sorted(list(self.special_tokens_set), reverse=True))

            # Process chunks with progress bar
            print("🔤 Processing text chunks...")
            chunk_pairs = list(zip(boundaries[:-1], boundaries[1:]))
            
            with tqdm(total=len(chunk_pairs), desc="Processing chunks", unit="chunk") as chunk_pbar:
                for start, end in chunk_pairs:
                    f.seek(start)
                    chunk = f.read(end - start).decode("utf-8", errors="ignore")

                    # Split the chunk by the special token pattern
                    # This removes the special tokens from the text completely
                    parts = re.split(special_token_pattern, chunk)
                    
                    # Process each part independently
                    for part in parts:
                        pretokens = re.findall(self.PAT, part)
                        for pretoken in pretokens:
                            pretoken_counts[pretoken] += 1
                    
                    chunk_pbar.update(1)

        print(f"📊 Found {len(pretoken_counts):,} unique pretokens")

        # Convert string pretokens to tuples of byte integers
        print("🔄 Converting pretokens to byte sequences...")
        token_sequences_with_counts: Dict[Tuple[int, ...], int] = {}
        
        with tqdm(total=len(pretoken_counts), desc="Converting to bytes", unit="pretoken") as pbar:
            for pretoken_str, count in pretoken_counts.items():
                byte_sequence = tuple(pretoken_str.encode("utf-8"))
                token_sequences_with_counts[byte_sequence] = count
                pbar.update(1)

        # 2. BPE Training Loop with Progress
        print(f"\n🎯 Phase 2: BPE merge training")
        current_token_sequences = token_sequences_with_counts.copy()
        
        merges_needed = target_vocab_size - self.vocab_count
        merge_pbar = tqdm(total=merges_needed, desc="Training merges", unit="merge")

        iteration = 0
        while self.vocab_count < target_vocab_size:
            iteration += 1
            
            # Re-collect bigram counts in every loop
            bigram_counts = self._collect_token_pairs(current_token_sequences)
            most_used_bigram_ids = self._get_most_used_bigram(bigram_counts)

            # Check for termination conditions
            if not bigram_counts or bigram_counts[most_used_bigram_ids] <= 1:
                print(f"\n⚠️  Early termination: No more frequent bigrams (iteration {iteration})")
                break

            # Get the byte representations of the two tokens
            b1 = self.vocabulary[most_used_bigram_ids[0]]
            b2 = self.vocabulary[most_used_bigram_ids[1]]

            # Store the learned merge rule as a tuple of byte strings
            self.merges.append((b1, b2))

            # Update vocabulary with the new token
            self.vocabulary[self.vocab_count] = b1 + b2
            
            # Update progress bar with current merge info
            merge_frequency = bigram_counts[most_used_bigram_ids]
            merge_pbar.set_postfix({
                'freq': f"{merge_frequency:,}",
                'vocab': f"{self.vocab_count+1:,}",
                'bigrams': f"{len(bigram_counts):,}"
            })
            
            self.vocab_count += 1

            # Apply merge to all sequences
            new_token_sequences_for_iteration: Dict[Tuple[int, ...], int] = {}
            for sequence, count in current_token_sequences.items():
                # Perform the merge for this sequence
                merged_seq = self._merge_pair_in_sequence(
                    sequence, most_used_bigram_ids, self.vocab_count - 1
                )
                new_token_sequences_for_iteration[merged_seq] = count

            current_token_sequences = new_token_sequences_for_iteration
            merge_pbar.update(1)

        merge_pbar.close()
        
        # Training complete
        elapsed_time = time.time() - start_time
        print(f"\n✅ Training Complete!")
        print(f"⏱️  Total time: {elapsed_time:.2f} seconds")
        print(f"📊 Final vocabulary size: {self.vocab_count:,}")
        print(f"🔄 Total merges learned: {len(self.merges):,}")
        print(f"📈 Average time per merge: {elapsed_time/len(self.merges):.3f}s" if self.merges else "N/A")

    def _merge_bytes(self, b1: bytes, b2: bytes) -> bytes:
        """Helper to concatenate bytes for vocabulary creation."""
        return b1 + b2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer and save its merges and vocabulary.")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to the input text file for training.")
    parser.add_argument("--merges_file", type=str, default="output/bpe_merges.txt",
                        help="Path to save the BPE merges file.")
    parser.add_argument("--vocab_file", type=str, default="output/bpe_vocab.json",
                        help="Path to save the BPE vocabulary file.")
    parser.add_argument("--vocab_size", type=int, default=12000,
                        help="Target vocabulary size for the tokenizer.")
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' does not exist.")
        exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.merges_file), exist_ok=True)
    os.makedirs(os.path.dirname(args.vocab_file), exist_ok=True)
    
    # Example of using special tokens
    special_tokens = {"<|endoftext|>"}
    trainer = BPETokenizerTrainer(special_tokens=special_tokens)
    
    print(f"Training BPE tokenizer on: {args.input_file}")
    print(f"Target vocabulary size: {args.vocab_size}")
    
    trainer.train(args.input_file, target_vocab_size=args.vocab_size)
    
    trainer.save_merges_and_vocab(args.merges_file, args.vocab_file)
    
    print(f"\nMerges saved to: {args.merges_file}")
    print(f"Vocabulary saved to: {args.vocab_file}")
    print(f"Final vocabulary size: {len(trainer.vocabulary)}")
    
    # You would typically save `trainer.merges` and `trainer.vocabulary` to disk
    # for later use in an actual tokenizer (encoding/decoding).
    print("\nTraining completed successfully!")