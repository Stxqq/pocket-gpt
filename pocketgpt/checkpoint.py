"""Checkpoints as plain .npz files: one array per parameter plus JSON metadata."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .nn import GPT, GPTConfig
from .tokenizer import CharTokenizer


def save_checkpoint(path, model, tokenizer, **meta):
    path = Path(path)
    meta = {"config": asdict(model.config), "vocab": tokenizer.chars, **meta}
    arrays = {f"param/{name}": p for name, p in model.state_dict().items()}
    # write next to the target and rename, so an interrupted run never leaves
    # a half-written checkpoint behind
    partial = path.with_name(path.name + ".partial.npz")
    np.savez(partial, meta=np.array(json.dumps(meta)), **arrays)
    os.replace(partial, path)


def load_checkpoint(path):
    """Return (model, tokenizer, meta) for a checkpoint written by train.py."""
    with np.load(path) as archive:
        meta = json.loads(archive["meta"].item())
        state = {
            key.removeprefix("param/"): archive[key]
            for key in archive.files
            if key.startswith("param/")
        }
    model = GPT(GPTConfig(**meta["config"]))
    model.load_state_dict(state)
    return model.eval(), CharTokenizer(meta["vocab"]), meta
