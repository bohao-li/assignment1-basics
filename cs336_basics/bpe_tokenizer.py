import re
import json
from collections.abc import Iterable, Iterator


class Tokenizer:
    """
    A tokenizer that encodes text into integer IDs and decodes integer IDs into text.
    It uses a vocabulary and a list of merges, and supports special tokens.
    """

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """
        Constructs a tokenizer from a given vocabulary, list of merges, and optional special tokens.

        Args:
            vocab: A dictionary mapping integer IDs to byte-encoded tokens.
            merges: A list of tuples, where each tuple represents a merge rule.
            special_tokens: An optional list of special tokens (as strings).
        """
        # Core data structures
        self._id_to_token: dict[int, bytes] = vocab  # Maps int ID to byte token
        self._merges_list: list[tuple[bytes, bytes]] = (
            merges  # Ordered list of merge rules (pair -> new_token)
        )

        # Derived data structures for efficiency
        self._token_to_id: dict[bytes, int] = {
            v: k for k, v in vocab.items()
        }  # Maps byte token to int ID
        self._merges_map: dict[tuple[bytes, bytes], bytes] = {
            (p1, p2): p1 + p2 for p1, p2 in merges
        }  # Maps (token1, token2) pair to their merged result

        # Define the regex pattern for initial splitting (pre-tokenization)
        # This is a common pattern inspired by GPT-2 tokenizers that separates:
        # - Common contractions like 's, 't, 're, etc.
        # - Words (unicode letters)
        # - Numbers (unicode digits)
        # - Non-alphanumeric characters (punctuation, symbols, etc.)
        # - Remaining whitespace
        self._PAT = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
            re.UNICODE,
        )

        # Special tokens handling
        self._special_tokens_set: set[bytes] = set()
        if special_tokens:
            # Find the next available ID, ensuring it's higher than existing vocab IDs
            next_id = max(self._id_to_token.keys()) + 1 if self._id_to_token else 0
            for token_str in special_tokens:
                token_bytes = token_str.encode("utf-8")
                if token_bytes not in self._token_to_id:
                    # If the special token is new, add it to both mappings
                    self._id_to_token[next_id] = token_bytes
                    self._token_to_id[token_bytes] = next_id
                    next_id += 1
                self._special_tokens_set.add(
                    token_bytes
                )  # Add to the set of special tokens

        # Pre-compile the regex pattern for splitting text by special tokens
        self._special_token_pattern = None
        if self._special_tokens_set:
            # Sort by length descending to ensure longer special tokens are matched first
            special_token_re_patterns = [
                re.escape(token.decode("utf-8"))  # Decode to string for regex
                for token in sorted(
                    list(self._special_tokens_set), key=len, reverse=True
                )
            ]
            # Use a capturing group for the pattern so `re.split` includes the delimiters
            self._special_token_pattern = re.compile(
                "(" + "|".join(special_token_re_patterns) + ")"
            )

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        """
        Class method to construct a Tokenizer from serialized files.

        Args:
            vocab_filepath: The path to the serialized vocabulary file (expected JSON string).
            merges_filepath: The path to the serialized merges file (expected space-separated pairs).
            special_tokens: An optional list of special tokens (as strings).

        Returns:
            A Tokenizer instance.
        """
        # Load the vocabulary from the JSON file
        vocab_int_to_bytes = {}
        try:
            with open(vocab_filepath, "r", encoding="utf-8") as f:
                parsed_vocab_str_to_int = json.load(f)  # Directly load JSON
                # Convert the parsed dictionary to the required int -> bytes format
                for token_str, token_id in parsed_vocab_str_to_int.items():
                    vocab_int_to_bytes[token_id] = token_str.encode("utf-8")

        except FileNotFoundError:
            raise FileNotFoundError(f"Vocabulary file not found at {vocab_filepath}")
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Error decoding JSON from vocabulary file: {vocab_filepath} - {e}"
            )
        except Exception as e:
            raise RuntimeError(
                f"An unexpected error occurred while loading vocabulary: {e}"
            )

        # Load the merges from the file
        merges_list_of_tuples_bytes = []
        try:
            with open(merges_filepath, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 2:
                        # Encode the merge parts to bytes before adding to the list
                        merges_list_of_tuples_bytes.append(
                            (parts[0].encode("utf-8"), parts[1].encode("utf-8"))
                        )
                    else:
                        print(
                            f"Warning: Skipping malformed line in merges file: {line.strip()}"
                        )
        except FileNotFoundError:
            raise FileNotFoundError(f"Merges file not found at {merges_filepath}")
        except Exception as e:
            raise RuntimeError(
                f"An unexpected error occurred while loading merges: {e}"
            )

        # Create and return a new Tokenizer instance, passing the correctly formatted data
        return cls(vocab_int_to_bytes, merges_list_of_tuples_bytes, special_tokens)

    # --- Core Encoding/Decoding Methods ---

    def encode(self, text: str) -> list[int]:
        """
        Encodes an input text into a sequence of token IDs.

        Args:
            text: The input text to encode.

        Returns:
            A list of integer IDs representing the encoded text.
        """
        token_ids: list[int] = []

        # Step 1: Split text by special tokens
        # `re.split` with a capturing group will include the delimiters in the result.
        # This allows us to re-insert special tokens correctly.
        # Example: "hello<SPECIAL>world" -> ["hello", "<SPECIAL>", "world"]
        parts = (
            re.split(self._special_token_pattern, text)
            if self._special_token_pattern
            else [text]
        )

        # Step 2: Process each part
        for part in parts:
            if not part:  # Handle empty strings that can result from re.split
                continue

            part_bytes = part.encode("utf-8")

            # Check if the part is a special token
            if part_bytes in self._special_tokens_set:
                token_ids.append(self._token_to_id[part_bytes])
            else:
                # This part is regular text, apply pre-tokenization and BPE merges
                # Initial pre-tokenization based on self._PAT
                pre_tokens_bytes: list[bytes] = [
                    t.encode("utf-8") for t in self._PAT.findall(part)
                ]

                # Apply BPE merges iteratively until no more merges can be found
                # This helper function will handle the entire merging process for a list of tokens
                final_bpe_tokens_bytes = self._apply_all_merges(pre_tokens_bytes)

                # Convert the final byte tokens to their integer IDs
                for bpe_token in final_bpe_tokens_bytes:
                    if bpe_token in self._token_to_id:
                        token_ids.append(self._token_to_id[bpe_token])
                    else:
                        # This scenario should ideally not happen if vocabulary is complete,
                        # but as a fallback, we could break it down to individual bytes or
                        # use an unknown token ID if the pre-token is not in vocab.
                        # For now, let's assume it's always in vocab.
                        # In a real tokenizer, you'd likely break it down to individual bytes
                        # or use a <unk> token. For this exercise, assume full coverage.
                        print(
                            f"Warning: Token '{bpe_token.decode('utf-8')}' not found in vocabulary during encoding."
                        )

        return token_ids

    def _apply_all_merges(self, tokens: list[bytes]) -> list[bytes]:
        """
        Applies all defined BPE merges to a list of byte tokens iteratively
        until no more merges are possible according to the defined merge rules.

        Args:
            tokens: A list of byte tokens.

        Returns:
            A list of byte tokens after all applicable merges have been performed.
        """
        # This function repeatedly applies the merge rules from self._merges_list.
        # The merge rules are applied in the order they appear in self._merges_list.
        # For each merge rule, it iterates through the current list of tokens and
        # performs all possible occurrences of that specific merge.
        # This loop continues until no more merges can be made based on the
        # _merges_list, meaning the tokens list has been fully compressed.

        # Create a mutable copy to work with
        current_tokens = list(tokens)

        # The merging process is iterative. We keep merging until no more merges can be applied.
        # A more robust implementation might track if any merge occurred in an iteration
        # and stop when a full pass yields no changes.
        # For a standard BPE encode, you usually iterate through the specific merges in order,
        # applying each one as many times as possible to the *current* list of tokens.

        for pair1, pair2 in self._merges_list:
            # Check if this merge pair exists in our pre-computed map
            if (pair1, pair2) not in self._merges_map:
                continue  # Skip if this merge rule isn't in our map (shouldn't happen with correct data)

            merged_token = self._merges_map[(pair1, pair2)]
            new_tokens = []
            i = 0
            while i < len(current_tokens):
                if (
                    i + 1 < len(current_tokens)
                    and current_tokens[i] == pair1
                    and current_tokens[i + 1] == pair2
                ):
                    # Found a pair to merge
                    new_tokens.append(merged_token)
                    i += 2  # Skip the next token as it was merged
                else:
                    new_tokens.append(current_tokens[i])
                    i += 1
            current_tokens = new_tokens  # Update tokens for the next merge rule

        return current_tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs.
        """
        for text_chunk in iterable:
            yield from self.encode(text_chunk)

    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        """
        decoded_bytes_list = []
        for token_id in ids:
            if token_id in self._id_to_token:
                decoded_bytes_list.append(self._id_to_token[token_id])
            else:
                # Handle unknown IDs, e.g., by skipping or inserting a replacement character.
                # The prompt mentions Unicode replacement character for unknown symbols.
                decoded_bytes_list.append(
                    b"\xef\xbf\xbd"
                )  # Unicode replacement character bytes

        # Concatenate all bytes and then decode to string
        return b"".join(decoded_bytes_list).decode(
            "utf-8", errors="replace"
        )  # Replace handles errors during decoding
