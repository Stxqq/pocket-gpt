"""AdamW, the learning-rate schedule and gradient clipping."""

from __future__ import annotations

import math

import numpy as np


class AdamW:
    """Adam with decoupled weight decay (Loshchilov & Hutter, 2019).

    Decay only touches parameters flagged with `decay=True` - the matmul
    weights. Biases, LayerNorm gains and embeddings are left alone.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.1):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.steps = 0
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self, lr=None):
        lr = self.lr if lr is None else lr
        self.steps += 1
        b1, b2 = self.beta1, self.beta2
        correction1 = 1 - b1**self.steps
        correction2 = 1 - b2**self.steps
        for p, m, v in zip(self.params, self.m, self.v, strict=True):
            if p.grad is None:
                continue
            g = p.grad
            if getattr(p, "decay", False) and self.weight_decay:
                p.data *= 1 - lr * self.weight_decay
            m *= b1
            m += (1 - b1) * g
            v *= b2
            v += (1 - b2) * g * g
            p.data -= lr * (m / correction1) / (np.sqrt(v / correction2) + self.eps)

    def zero_grad(self):
        for p in self.params:
            p.grad = None


def cosine_lr(step, *, max_lr, min_lr, warmup, total):
    """Linear warmup to max_lr, then a cosine decay down to min_lr."""
    if step < warmup:
        return max_lr * (step + 1) / warmup
    if step >= total:
        return min_lr
    progress = (step - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (1 + math.cos(math.pi * progress)) * (max_lr - min_lr)


def clip_grad_norm(params, max_norm):
    """Rescale gradients so their global L2 norm is at most max_norm."""
    grads = [p.grad for p in params if p.grad is not None]
    norm = math.sqrt(sum(float((g.astype(np.float64) ** 2).sum()) for g in grads))
    if norm > max_norm:
        scale = max_norm / (norm + 1e-6)
        for p in params:
            if p.grad is not None:
                p.grad = p.grad * scale
    return norm
