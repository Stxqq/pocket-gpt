"""Stream text from a trained checkpoint or an exported model directory."""

import argparse
import sys
from pathlib import Path

import numpy as np

from pocketgpt.checkpoint import load_checkpoint
from pocketgpt.export import load_web

TRAINED = Path("out/shakespeare/ckpt.npz")
SHIPPED = Path("docs/model")


def load_model(path):
    path = Path(path)
    model, tokenizer, _ = load_web(path) if path.is_dir() else load_checkpoint(path)
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=TRAINED if TRAINED.exists() else SHIPPED,
        help="a .npz from train.py or a directory from export.py "
        "(default: your own run if there is one, else the shipped docs/model)",
    )
    parser.add_argument("--prompt", default="\n")
    parser.add_argument("--tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    model, tokenizer = load_model(args.model)
    rng = np.random.default_rng(args.seed)
    # an empty prompt starts after a newline, like every speech in the corpus
    prompt = args.prompt or "\n"
    sys.stdout.write(prompt)
    tokens = model.generate(
        tokenizer.encode(prompt),
        args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        rng=rng,
    )
    for token in tokens:
        sys.stdout.write(tokenizer.decode([token]))
        sys.stdout.flush()
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
