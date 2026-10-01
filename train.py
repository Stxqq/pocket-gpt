"""Train a character-level GPT on the prepared tiny shakespeare split."""

import argparse
import dataclasses
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pocketgpt.checkpoint import save_checkpoint
from pocketgpt.data import get_batch, load_splits
from pocketgpt.nn import GPT, GPTConfig
from pocketgpt.optim import AdamW, clip_grad_norm, cosine_lr
from pocketgpt.tensor import cross_entropy, no_grad


@dataclass
class TrainConfig:
    data_dir: str = "data/shakespeare"
    out_dir: str = "out/shakespeare"
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    block_size: int = 128
    dropout: float = 0.0
    batch_size: int = 32
    max_steps: int = 6000
    lr: float = 2e-3
    min_lr: float = 2e-4
    warmup_steps: int = 200
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_every: int = 250
    eval_batches: int = 20
    log_every: int = 10
    seed: int = 1337


def parse_config():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in dataclasses.fields(TrainConfig):
        flag = "--" + field.name.replace("_", "-")
        parser.add_argument(flag, type=field.type, default=field.default)
    return TrainConfig(**vars(parser.parse_args()))


def estimate_loss(model, splits, cfg):
    # the same eval windows every time, so the curve shows the model changing
    # rather than the sampling noise of the batches
    model.eval()
    losses = {}
    with no_grad():
        for name, tokens in splits.items():
            rng = np.random.default_rng(0)
            batches = (
                get_batch(tokens, cfg.batch_size, cfg.block_size, rng)
                for _ in range(cfg.eval_batches)
            )
            losses[name] = float(
                np.mean([cross_entropy(model(x), y).item() for x, y in batches])
            )
    model.train()
    return losses


def main():
    cfg = parse_config()
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, train_tokens, val_tokens = load_splits(cfg.data_dir)
    model_config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=cfg.block_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        dropout=cfg.dropout,
    )
    model = GPT(model_config, seed=cfg.seed)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    rng = np.random.default_rng(cfg.seed)
    print(f"{model.num_params():,} parameters, vocab {tokenizer.vocab_size}")

    log = {
        "config": dataclasses.asdict(cfg),
        "params": model.num_params(),
        "train": [],
        "eval": [],
    }
    tokens_per_step = cfg.batch_size * cfg.block_size
    best_val = float("inf")
    train_seconds = 0.0

    for step in range(cfg.max_steps + 1):
        if step % cfg.eval_every == 0 or step == cfg.max_steps:
            losses = estimate_loss(
                model, {"train": train_tokens, "val": val_tokens}, cfg
            )
            log["eval"].append({"step": step, **losses})
            print(
                f"step {step:5d} | train {losses['train']:.4f} "
                f"| val {losses['val']:.4f}"
            )
            if losses["val"] < best_val:
                best_val = losses["val"]
                save_checkpoint(
                    out_dir / "ckpt.npz", model, tokenizer, step=step, **losses
                )
            log["tokens_per_sec"] = round(
                tokens_per_step * step / max(train_seconds, 1e-9)
            )
            log["train_seconds"] = round(train_seconds, 1)
            (out_dir / "log.json").write_text(json.dumps(log))
        if step == cfg.max_steps:
            break

        started = time.perf_counter()
        lr = cosine_lr(
            step,
            max_lr=cfg.lr,
            min_lr=cfg.min_lr,
            warmup=cfg.warmup_steps,
            total=cfg.max_steps,
        )
        x, y = get_batch(train_tokens, cfg.batch_size, cfg.block_size, rng)
        loss = cross_entropy(model(x), y)
        model.zero_grad()
        loss.backward()
        grad_norm = clip_grad_norm(model.parameters(), cfg.grad_clip)
        optimizer.step(lr)
        elapsed = time.perf_counter() - started
        train_seconds += elapsed

        if step % cfg.log_every == 0:
            log["train"].append({"step": step, "loss": round(loss.item(), 4)})
            print(
                f"step {step:5d} | loss {loss.item():.4f} | lr {lr:.2e} "
                f"| norm {grad_norm:.2f} | {tokens_per_step / elapsed:,.0f} tok/s"
            )

    print(f"best val {best_val:.4f}, {log['tokens_per_sec']:,} tok/s")


if __name__ == "__main__":
    main()
