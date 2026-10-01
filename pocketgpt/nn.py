"""Modules and the GPT itself."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .tensor import Tensor, dropout, layer_norm, no_grad, softmax


class Parameter(Tensor):
    """A trainable leaf. `decay` marks weights that AdamW should shrink."""

    def __init__(self, data, decay=False, dtype=np.float32):
        # copy: the optimizer updates parameters in place
        super().__init__(np.array(data, dtype=dtype), requires_grad=True)
        self.decay = decay


class Module:
    training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def children(self):
        for value in vars(self).values():
            if isinstance(value, Module):
                yield value
            elif isinstance(value, list):
                yield from (v for v in value if isinstance(v, Module))

    def named_parameters(self, prefix=""):
        for name, value in vars(self).items():
            if isinstance(value, Parameter):
                yield prefix + name, value
            elif isinstance(value, Module):
                yield from value.named_parameters(f"{prefix}{name}.")
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, Module):
                        yield from item.named_parameters(f"{prefix}{name}.{i}.")

    def parameters(self):
        return [p for _, p in self.named_parameters()]

    def num_params(self):
        return sum(p.data.size for p in self.parameters())

    def train(self, mode=True):
        self.training = mode
        for child in self.children():
            child.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def zero_grad(self):
        for p in self.parameters():
            p.grad = None

    def state_dict(self):
        return {name: p.data for name, p in self.named_parameters()}

    def load_state_dict(self, state):
        own = dict(self.named_parameters())
        if own.keys() != state.keys():
            missing = sorted(own.keys() - state.keys())
            unexpected = sorted(state.keys() - own.keys())
            raise KeyError(f"missing {missing}, unexpected {unexpected}")
        for name, p in own.items():
            value = np.asarray(state[name])
            if value.shape != p.shape:
                raise ValueError(f"{name}: expected {p.shape}, got {value.shape}")
            p.data = value.astype(p.dtype)

    def astype(self, dtype):
        for p in self.parameters():
            p.data = p.data.astype(dtype)
        return self


class Linear(Module):
    def __init__(self, n_in, n_out, rng, bias=True, std=0.02):
        self.weight = Parameter(rng.normal(0, std, (n_in, n_out)), decay=True)
        self.bias = Parameter(np.zeros(n_out)) if bias else None

    def forward(self, x):
        out = x @ self.weight
        return out + self.bias if self.bias is not None else out


class Embedding(Module):
    def __init__(self, n_rows, dim, rng):
        self.weight = Parameter(rng.normal(0, 0.02, (n_rows, dim)))

    def forward(self, ids):
        return self.weight[ids]


class LayerNorm(Module):
    def __init__(self, dim, eps=1e-5):
        self.weight = Parameter(np.ones(dim))
        self.bias = Parameter(np.zeros(dim))
        self.eps = eps

    def forward(self, x):
        return layer_norm(x, self.weight, self.bias, self.eps)


@dataclass
class GPTConfig:
    vocab_size: int = 65
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0


class CausalSelfAttention(Module):
    def __init__(self, config, rng):
        c = config.n_embd
        self.n_head = config.n_head
        self.p_drop = config.dropout
        self.rng = rng
        self.qkv = Linear(c, 3 * c, rng)
        # residual projections start smaller so the sum over layers keeps a
        # stable variance at init (GPT-2, section 2.3)
        self.proj = Linear(c, c, rng, std=0.02 / math.sqrt(2 * config.n_layer))
        self.future = ~np.tril(np.ones((config.block_size,) * 2, dtype=bool))

    def forward(self, x):
        batch, steps, width = x.shape
        head_dim = width // self.n_head
        qkv = self.qkv(x).reshape(batch, steps, 3, self.n_head, head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = (q @ k.transpose(0, 1, 3, 2)) * (1 / math.sqrt(head_dim))
        scores = scores.masked_fill(self.future[:steps, :steps], -np.inf)
        weights = dropout(softmax(scores), self.p_drop, self.rng, self.training)

        out = (weights @ v).transpose(0, 2, 1, 3).reshape(batch, steps, width)
        return dropout(self.proj(out), self.p_drop, self.rng, self.training)


class MLP(Module):
    def __init__(self, config, rng):
        c = config.n_embd
        self.p_drop = config.dropout
        self.rng = rng
        self.up = Linear(c, 4 * c, rng)
        self.down = Linear(4 * c, c, rng, std=0.02 / math.sqrt(2 * config.n_layer))

    def forward(self, x):
        out = self.down(self.up(x).gelu())
        return dropout(out, self.p_drop, self.rng, self.training)


class Block(Module):
    def __init__(self, config, rng):
        self.ln1 = LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config, rng)
        self.ln2 = LayerNorm(config.n_embd)
        self.mlp = MLP(config, rng)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(Module):
    """Decoder-only transformer: (batch, time) token ids -> next-token logits."""

    def __init__(self, config, seed=0):
        rng = np.random.default_rng(seed)
        self.config = config
        self.rng = rng
        self.wte = Embedding(config.vocab_size, config.n_embd, rng)
        self.wpe = Embedding(config.block_size, config.n_embd, rng)
        self.blocks = [Block(config, rng) for _ in range(config.n_layer)]
        self.ln_f = LayerNorm(config.n_embd)

    def forward(self, ids):
        ids = np.asarray(ids)
        steps = ids.shape[1]
        if steps > self.config.block_size:
            raise ValueError(f"sequence of {steps} exceeds block_size")
        x = self.wte(ids) + self.wpe.weight[:steps]
        x = dropout(x, self.config.dropout, self.rng, self.training)
        for block in self.blocks:
            x = block(x)
        # the output head reuses the token embedding (weight tying)
        return self.ln_f(x) @ self.wte.weight.T

    def generate(self, ids, max_new_tokens, temperature=1.0, top_k=None, rng=None):
        """Yield sampled token ids one at a time, continuing from `ids`."""
        rng = rng or np.random.default_rng()
        context = list(ids)
        was_training = self.training
        self.eval()
        try:
            with no_grad():
                for _ in range(max_new_tokens):
                    window = np.array([context[-self.config.block_size :]])
                    logits = self(window).data[0, -1].astype(np.float64)
                    token = sample_token(logits, temperature, top_k, rng)
                    context.append(token)
                    yield token
        finally:
            self.train(was_training)


def sample_token(logits, temperature, top_k, rng):
    if temperature == 0:
        return int(np.argmax(logits))
    logits = logits / temperature
    if top_k is not None and top_k < len(logits):
        cutoff = np.partition(logits, -top_k)[-top_k]
        logits = np.where(logits < cutoff, -np.inf, logits)
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    return int(rng.choice(len(probs), p=probs))
