"""Character-level tokenizer: one token per distinct character."""

from __future__ import annotations

import numpy as np


class CharTokenizer:
    def __init__(self, chars):
        self.chars = list(chars)
        self.index = {ch: i for i, ch in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text):
        return cls(sorted(set(text)))

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, text):
        try:
            return np.array([self.index[ch] for ch in text], dtype=np.int64)
        except KeyError as err:
            raise ValueError(f"character {err.args[0]!r} is not in the vocab") from None

    def decode(self, ids):
        return "".join(self.chars[i] for i in ids)
