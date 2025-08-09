import os
import regex as re
from typing import BinaryIO, Dict, Tuple
from collections import defaultdict


class BPETokenizerTrainer:
    """
    A class to train a Byte Pair Encoding (BPE) tokenizer.
    """

    # Regex pattern for initial pre-tokenization
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def __init__(self, initial_vocab_size: int = 256):
        """
        Initializes the BPE tokenizer trainer.

        Args:
            initial_vocab_size: The starting vocabulary size (typically 256 for all bytes).
        """
        self.vocab_count = initial_vocab_size
        self.merges: list[Tuple[int, int]] = []  # Stores the learned merge rules (bigrams)
        self.vocabulary: Dict[int, bytes] = {i: bytes([i]) for i in range(initial_vocab_size)} # For decoding

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
        assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

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

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))

    def _collect_token_pairs(
        self, token_sequences_with_counts: Dict[Tuple[int, ...], int]
    ) -> Dict[Tuple[int, int], int]:
        """
        Collects and counts all unique bigrams (pairs of adjacent integers representing tokens)
        from a dictionary where keys are tuples of integers and values are their counts.
        """
        bigram_counts: Dict[Tuple[int, int], int] = defaultdict(int)
        for token_sequence, token_frequency in token_sequences_with_counts.items():
            for i in range(len(token_sequence) - 1):
                bigram = (token_sequence[i], token_sequence[i + 1])
                bigram_counts[bigram] += token_frequency
        return dict(bigram_counts)

    def _get_most_used_bigram(self, bigrams: Dict[Tuple[int, int], int]) -> Tuple[int, int]:
        """
        Finds the most frequently used bigram. If tied, prefers the one with the larger left token.
        """
        if not bigrams:
            return (0, 0) # Return a default, non-meaningful bigram for empty case

        most_used_bigram = max(
            bigrams,
            key=lambda bigram: (bigrams.get(bigram), bigram[0])
        )
        return most_used_bigram

    def _merge_pair_in_sequence(
        self, sequence: Tuple[int, ...], bigram_to_merge: Tuple[int, int], new_token_id: int
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
    
    def _find_all_occurrences(self, sequence: Tuple[int, ...], bigram: Tuple[int, int]) -> list[int]:
        """
        Helper function to find all occurrences of a bigram within a single sequence.
        Returns a list of starting indices of the bigram.
        """
        occurrences = []
        i = 0
        while i < len(sequence) - 1:
            if (sequence[i], sequence[i+1]) == bigram:
                occurrences.append(i)
                i += 2 # Jump past the merged pair
            else:
                i += 1
        return occurrences

    def train(self, data_file_path: str, target_vocab_size: int = 3000, num_processes: int = 4):
        """
        Trains the BPE tokenizer.

        Args:
            data_file_path: Path to the text file for training.
            target_vocab_size: The desired final vocabulary size.
            num_processes: Number of processes to use for initial chunking (serial in this example).
        """
        pretoken_counts: Dict[str, int] = defaultdict(int)

        # 1. Initial Pre-tokenization and Byte Conversion
        with open(data_file_path, "rb") as f:
            boundaries = self._find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk = f.read(end - start).decode("utf-8", errors="ignore")
                chunk = chunk.replace("<|endoftext|>", "")

                pretokens = re.findall(self.PAT, chunk)
                for pretoken in pretokens:
                    pretoken_counts[pretoken] += 1

        # Convert string pretokens to tuples of byte integers
        token_sequences_with_counts: Dict[Tuple[int, ...], int] = {}
        for pretoken_str, count in pretoken_counts.items():
            byte_sequence = tuple(pretoken_str.encode("utf-8"))
            token_sequences_with_counts[byte_sequence] = count

        # 2. Optimized BPE Training Loop
        # `current_token_sequences` will be updated with merged tokens in each step
        current_token_sequences = token_sequences_with_counts.copy()
        
        # Initialize the bigram cache
        bigram_cache = self._collect_token_pairs(current_token_sequences)
        most_used_bigram = self._get_most_used_bigram(bigram_cache)

        while self.vocab_count < target_vocab_size and bigram_cache and bigram_cache[most_used_bigram] > 1:
            # Store the merge rule
            self.merges.append(most_used_bigram)
            self.vocab_count += 1
            
            # Update vocabulary with the new token
            self.vocabulary[self.vocab_count - 1] = self._merge_bytes(
                self.vocabulary[most_used_bigram[0]],
                self.vocabulary[most_used_bigram[1]]
            )

            new_token_sequences_for_iteration: Dict[Tuple[int, ...], int] = {}
            affected_bigram_changes: Dict[Tuple[int, int], int] = defaultdict(int)

            for sequence, count in current_token_sequences.items():
                occurrences = self._find_all_occurrences(sequence, most_used_bigram)
                
                if not occurrences:
                    new_token_sequences_for_iteration[sequence] = count
                    continue
                
                # Decrement counts of old bigrams affected by this sequence's merge
                for pos in occurrences:
                    if pos > 0: # Left context bigram
                        affected_bigram_changes[(sequence[pos-1], sequence[pos])] -= count
                    if pos + 2 < len(sequence): # Right context bigram
                        affected_bigram_changes[(sequence[pos+1], sequence[pos+2])] -= count

                # Perform the merge for this sequence
                merged_seq = self._merge_pair_in_sequence(sequence, most_used_bigram, self.vocab_count - 1)
                new_token_sequences_for_iteration[merged_seq] = count

                # Increment counts of new bigrams formed in this sequence
                # Need to re-find occurrences in the *newly merged* sequence
                # This part is a bit tricky: rebuild from scratch or track positions carefully.
                # For simplicity and correctness in this example, we'll re-scan the new sequence
                # which is generally faster than complex index tracking.
                temp_bigrams_in_new_seq = self._collect_token_pairs({merged_seq: count})
                for bg, bg_count in temp_bigrams_in_new_seq.items():
                     affected_bigram_changes[bg] += bg_count # Add to changes

            # Update the global bigram cache with all collected changes
            for bigram, change in affected_bigram_changes.items():
                bigram_cache[bigram] = bigram_cache.get(bigram, 0) + change
                if bigram_cache[bigram] <= 0:
                    del bigram_cache[bigram]
            
            # Remove the merged bigram from the cache (its count should be 0 or less now)
            if most_used_bigram in bigram_cache:
                del bigram_cache[most_used_bigram]

            current_token_sequences = new_token_sequences_for_iteration
            most_used_bigram = self._get_most_used_bigram(bigram_cache)
            
        print("\n--- Training Complete ---")
        print("Final merge rules (bigrams):", self.merges)
        print("Final vocabulary count:", self.vocab_count)
        # print("Final token sequences (after all merges):", current_token_sequences)
        # You could also store final_token_sequences as an attribute if needed
    
    def _merge_bytes(self, b1: bytes, b2: bytes) -> bytes:
        """Helper to concatenate bytes for vocabulary creation."""
        return b1 + b2


# --- Usage Example ---
if __name__ == "__main__":
    # Create a dummy data file for demonstration
    # In a real scenario, this would be your actual training data.
    dummy_data_dir = "data"
    os.makedirs(dummy_data_dir, exist_ok=True)
    dummy_file_path = os.path.join(dummy_data_dir, "test.txt")

    trainer = BPETokenizerTrainer()
    trainer.train(dummy_file_path, target_vocab_size=500)

    # You would typically save `trainer.merges` and `trainer.vocabulary` to disk
    # for later use in an actual tokenizer (encoding/decoding).
    print("\nTrainer merges:", trainer.merges)
    print("\nTrainer vocabulary:", trainer.vocabulary)