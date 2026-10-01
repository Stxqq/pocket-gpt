"""Write a checkpoint out in the browser format used by the demo page."""

import argparse
import shutil
from pathlib import Path

import numpy as np

from pocketgpt.checkpoint import load_checkpoint
from pocketgpt.export import export_web


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", default="out/shakespeare/ckpt.npz")
    parser.add_argument("--out", default="docs/model")
    parser.add_argument("--data", default="data/shakespeare")
    parser.add_argument(
        "--prompt",
        help="text for fixture.json (default: the first block_size characters "
        "of the val split, so the check covers every position)",
    )
    args = parser.parse_args()

    model, tokenizer, meta = load_checkpoint(args.ckpt)
    prompt = args.prompt
    if prompt is None:
        val = np.fromfile(Path(args.data) / "val.bin", dtype=np.uint16)
        prompt = tokenizer.decode(val[: model.config.block_size].tolist())
    stats = {k: meta[k] for k in ("step", "train", "val") if k in meta}
    manifest = export_web(model, tokenizer, args.out, prompt, **stats)

    log = Path(args.ckpt).with_name("log.json")
    if log.exists():
        shutil.copy(log, Path(args.out) / "loss.json")
    print(
        f"{manifest['params']:,} parameters, {manifest['bytes'] / 1e6:.2f} MB "
        f"-> {args.out}"
    )


if __name__ == "__main__":
    main()
