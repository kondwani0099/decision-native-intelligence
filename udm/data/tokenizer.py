"""
Simple Word-Level Tokenizer for UDM Canonical Decision Format.

Design choice: We use a word-level tokenizer with character fallback instead of
SentencePiece/BPE because:
  1. The canonical format is highly structured (keywords, numbers, field names)
  2. Token efficiency on natural language prose is not a priority
  3. Zero external dependencies
  4. Deterministic tokenization (important for decision reproducibility)

Special tokens:
  <PAD>       = 0   Padding
  <BOS>       = 1   Beginning of sequence
  <EOS>       = 2   End of sequence
  <UNK>       = 3   Unknown token
  <DECISION>  = 4   Decision block start
  </DECISION> = 5   Decision block end
  <TASK_CLS>  = 6   Classification task marker
  <TASK_REG>  = 7   Regression task marker
  <TASK_RANK> = 8   Ranking task marker
"""

import re
import json
import os
from typing import List, Dict, Optional
from collections import Counter


# Regex pattern to split on whitespace, punctuation, and special markers
# Keeps numbers (including decimals) as single tokens
TOKENIZE_PATTERN = re.compile(
    r"(<DECISION>|</DECISION>|<TASK_CLS>|<TASK_REG>|<TASK_RANK>)"  # Special markers
    r"|(\d+\.\d+)"                                                   # Decimal numbers
    r"|(\d+)"                                                        # Integers
    r"|([a-zA-Z_][a-zA-Z0-9_]*)"                                    # Words/identifiers
    r"|([^\s])"                                                      # Single non-space chars (punctuation)
)


class DecisionTokenizer:
    """
    Word-level tokenizer for UDM canonical decision format.

    Vocabulary is built from training data and persisted to disk.
    Unknown tokens fall back to character-level encoding.
    """

    SPECIAL_TOKENS = {
        "<PAD>": 0,
        "<BOS>": 1,
        "<EOS>": 2,
        "<UNK>": 3,
        "<DECISION>": 4,
        "</DECISION>": 5,
        "<TASK_CLS>": 6,
        "<TASK_REG>": 7,
        "<TASK_RANK>": 8,
    }

    def __init__(self, vocab_size: int = 32000):
        self.target_vocab_size = vocab_size
        self.token_to_id: Dict[str, int] = dict(self.SPECIAL_TOKENS)
        self.id_to_token: Dict[int, str] = {v: k for k, v in self.SPECIAL_TOKENS.items()}
        self._next_id = len(self.SPECIAL_TOKENS)

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_token_id(self) -> int:
        return self.SPECIAL_TOKENS["<PAD>"]

    @property
    def bos_token_id(self) -> int:
        return self.SPECIAL_TOKENS["<BOS>"]

    @property
    def eos_token_id(self) -> int:
        return self.SPECIAL_TOKENS["<EOS>"]

    @property
    def unk_token_id(self) -> int:
        return self.SPECIAL_TOKENS["<UNK>"]

    def _tokenize_text(self, text: str) -> List[str]:
        """Split text into tokens using regex pattern."""
        tokens = []
        for match in TOKENIZE_PATTERN.finditer(text):
            token = match.group()
            tokens.append(token)
        return tokens

    def build_vocab(self, texts: List[str], min_freq: int = 1):
        """
        Build vocabulary from a list of canonical decision texts.

        Args:
            texts: list of canonical decision format strings
            min_freq: minimum frequency for a token to be included
        """
        # Count all tokens
        counter = Counter()
        for text in texts:
            tokens = self._tokenize_text(text)
            counter.update(tokens)

        # Add all printable ASCII characters as fallback tokens
        for ch in range(32, 127):
            char = chr(ch)
            if char not in self.token_to_id:
                self.token_to_id[char] = self._next_id
                self.id_to_token[self._next_id] = char
                self._next_id += 1

        # Add frequent tokens
        for token, freq in counter.most_common():
            if freq < min_freq:
                continue
            if token not in self.token_to_id:
                self.token_to_id[token] = self._next_id
                self.id_to_token[self._next_id] = token
                self._next_id += 1

        # Pad vocabulary to target size with placeholder tokens
        while self._next_id < self.target_vocab_size:
            placeholder = f"<UNUSED_{self._next_id}>"
            self.token_to_id[placeholder] = self._next_id
            self.id_to_token[self._next_id] = placeholder
            self._next_id += 1

        print(f"[Tokenizer] Vocabulary built: {self.vocab_size} tokens "
              f"({len(counter)} unique word tokens + {len(self.SPECIAL_TOKENS)} special + characters)")

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
        task_type: str = "classification",
        max_length: Optional[int] = None,
    ) -> List[int]:
        """
        Encode text to token IDs.

        Args:
            text: canonical decision text
            add_bos: prepend BOS token
            add_eos: append EOS token
            task_type: "classification", "regression", "ranking" — prepends task marker
            max_length: truncate to this length (including special tokens)

        Returns:
            List of token IDs
        """
        ids = []

        # Add task type marker
        task_markers = {
            "classification": self.SPECIAL_TOKENS["<TASK_CLS>"],
            "regression": self.SPECIAL_TOKENS["<TASK_REG>"],
            "ranking": self.SPECIAL_TOKENS["<TASK_RANK>"],
        }
        if task_type in task_markers:
            ids.append(task_markers[task_type])

        # Add BOS
        if add_bos:
            ids.append(self.bos_token_id)

        # Tokenize text
        tokens = self._tokenize_text(text)
        for token in tokens:
            if token in self.token_to_id:
                ids.append(self.token_to_id[token])
            else:
                # Character-level fallback for unknown tokens
                for ch in token:
                    ids.append(self.token_to_id.get(ch, self.unk_token_id))

        # Add EOS
        if add_eos:
            ids.append(self.eos_token_id)

        # Truncate if needed
        if max_length is not None and len(ids) > max_length:
            ids = ids[:max_length - 1] + [self.eos_token_id]

        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        """
        Decode token IDs back to text.

        Args:
            ids: list of token IDs
            skip_special: if True, omit special tokens from output

        Returns:
            Decoded text string
        """
        tokens = []
        special_ids = set(self.SPECIAL_TOKENS.values()) if skip_special else set()

        for tid in ids:
            if tid in special_ids:
                continue
            token = self.id_to_token.get(tid, "<UNK>")
            tokens.append(token)

        # Simple space joining (works well for structured format)
        return " ".join(tokens)

    def save(self, path: str):
        """Save tokenizer vocabulary to disk."""
        data = {
            "token_to_id": self.token_to_id,
            "target_vocab_size": self.target_vocab_size,
        }
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[Tokenizer] Saved to {path}")

    @classmethod
    def load(cls, path: str) -> "DecisionTokenizer":
        """Load tokenizer vocabulary from disk."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tokenizer = cls(vocab_size=data.get("target_vocab_size", 32000))
        tokenizer.token_to_id = data["token_to_id"]
        tokenizer.id_to_token = {int(v): k for k, v in tokenizer.token_to_id.items()}
        tokenizer._next_id = max(tokenizer.id_to_token.keys()) + 1
        print(f"[Tokenizer] Loaded from {path} ({tokenizer.vocab_size} tokens)")
        return tokenizer

    def load_vocab(self, path: str) -> "DecisionTokenizer":
        """Load tokenizer vocabulary directly into this instance."""
        loaded = self.load(path)
        self.target_vocab_size = loaded.target_vocab_size
        self.token_to_id = loaded.token_to_id
        self.id_to_token = loaded.id_to_token
        self._next_id = loaded._next_id
        return self
