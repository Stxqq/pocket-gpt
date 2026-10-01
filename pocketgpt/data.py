"""Token streams and random training windows."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .tokenizer import CharTokenizer


def load_splits(data_dir):
    """Read the arrays written by scripts/prepare.py."""
    data_dir = Path(data_dir)
    vocab = json.loads((data_dir / "vocab.json").read_text())
    train = np.fromfile(data_dir / "train.bin", dtype=np.uint16)
    val = np.fromfile(data_dir / "val.bin", dtype=np.uint16)
    return CharTokenizer(vocab), train, val


def get_batch(tokens, batch_size, block_size, rng):
    """Sample `batch_size` windows; targets are the inputs shifted by one."""
    starts = rng.integers(0, len(tokens) - block_size, size=batch_size)
    windows = tokens[starts[:, None] + np.arange(block_size + 1)].astype(np.int64)
    return windows[:, :-1], windows[:, 1:]
