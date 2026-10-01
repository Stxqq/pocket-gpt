"""Download tiny shakespeare and encode it into train/val token files."""

import argparse
import json
import os
import urllib.request
from pathlib import Path

import numpy as np

from pocketgpt.tokenizer import CharTokenizer

URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/shakespeare")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    source = out / "input.txt"
    if not source.exists():
        print(f"downloading {URL}")
        # a cut-off download must not look finished to the next run
        partial = source.with_name("input.txt.part")
        urllib.request.urlretrieve(URL, partial)
        os.replace(partial, source)
    text = source.read_text(encoding="utf-8")

    tokenizer = CharTokenizer.from_text(text)
    # uint16 like nanoGPT's .bin files, so a BPE vocab would fit the same format
    ids = tokenizer.encode(text).astype(np.uint16)
    split = int(len(ids) * (1 - args.val_fraction))
    ids[:split].tofile(out / "train.bin")
    ids[split:].tofile(out / "val.bin")
    (out / "vocab.json").write_text(json.dumps(tokenizer.chars))

    print(f"{len(text):,} characters, vocab {tokenizer.vocab_size}")
    print(f"train {split:,} tokens, val {len(ids) - split:,} tokens -> {out}")


if __name__ == "__main__":
    main()
