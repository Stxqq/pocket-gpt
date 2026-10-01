"""Export a model for the browser: raw float32 weights plus a JSON manifest.

weights.bin holds every parameter back to back as little-endian float32, in
the order of `named_parameters()`. model.json says where each one starts.
fixture.json has the logits for a fixed prompt, so a second implementation
(the JavaScript one in docs/) can prove it computes the same function.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .nn import GPT, GPTConfig
from .tensor import no_grad
from .tokenizer import CharTokenizer

FORMAT = "pocket-gpt/1"


def export_web(model, tokenizer, out_dir, prompt="ROMEO:", **stats):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tensors, offset = [], 0
    with open(out_dir / "weights.bin", "wb") as f:
        for name, p in model.named_parameters():
            raw = p.data.astype("<f4").tobytes()
            f.write(raw)
            tensors.append({"name": name, "shape": list(p.shape), "offset": offset})
            offset += len(raw)

    manifest = {
        "format": FORMAT,
        "config": asdict(model.config),
        "layer_norm_eps": 1e-5,
        "activation": "gelu_tanh",
        "vocab": tokenizer.chars,
        "params": model.num_params(),
        "bytes": offset,
        "tensors": tensors,
        **stats,
    }
    (out_dir / "model.json").write_text(json.dumps(manifest, indent=1))

    ids = tokenizer.encode(prompt)
    was_training = model.training
    model.eval()
    with no_grad():
        logits = model(ids[None]).data[0]
    model.train(was_training)
    fixture = {
        "prompt": prompt,
        "tokens": ids.tolist(),
        "logits": logits.astype(np.float32).tolist(),
    }
    (out_dir / "fixture.json").write_text(json.dumps(fixture))
    return manifest


def read_web(out_dir):
    """Load an exported model back into {name: array}, plus its manifest."""
    out_dir = Path(out_dir)
    manifest = json.loads((out_dir / "model.json").read_text())
    blob = (out_dir / "weights.bin").read_bytes()
    state = {}
    for t in manifest["tensors"]:
        count = int(np.prod(t["shape"]))
        flat = np.frombuffer(blob, dtype="<f4", count=count, offset=t["offset"])
        state[t["name"]] = flat.reshape(t["shape"])
    return state, manifest


def load_web(out_dir):
    """Return (model, tokenizer, manifest) for a directory written by export_web."""
    state, manifest = read_web(out_dir)
    model = GPT(GPTConfig(**manifest["config"]))
    model.load_state_dict(state)
    return model.eval(), CharTokenizer(manifest["vocab"]), manifest
