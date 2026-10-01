"""Stream text from a trained checkpoint."""

import argparse
import sys

import numpy as np

from pocketgpt.checkpoint import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default="out/shakespeare/ckpt.npz")
    parser.add_argument("--prompt", default="\n")
    parser.add_argument("--tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    model, tokenizer, _ = load_checkpoint(args.ckpt)
    rng = np.random.default_rng(args.seed)
    sys.stdout.write(args.prompt)
    tokens = model.generate(
        tokenizer.encode(args.prompt),
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
