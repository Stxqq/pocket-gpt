import math

import numpy as np
import pytest

from pocketgpt.nn import GPT, GPTConfig, Parameter
from pocketgpt.optim import AdamW, clip_grad_norm, cosine_lr
from pocketgpt.tensor import cross_entropy


def reference_adamw(p, grads, lr, b1, b2, eps, wd):
    """Algorithm 2 of the AdamW paper, written out longhand."""
    m = np.zeros_like(p)
    v = np.zeros_like(p)
    for t, g in enumerate(grads, start=1):
        p = p - lr * wd * p
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g**2
        m_hat = m / (1 - b1**t)
        v_hat = v / (1 - b2**t)
        p = p - lr * m_hat / (np.sqrt(v_hat) + eps)
    return p


@pytest.mark.parametrize("decay", [True, False])
def test_adamw_matches_reference(decay):
    rng = np.random.default_rng(0)
    start = rng.standard_normal((4, 3))
    grads = [rng.standard_normal((4, 3)) for _ in range(5)]
    p = Parameter(start, decay=decay, dtype=np.float64)
    opt = AdamW([p], lr=0.01, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    for g in grads:
        p.grad = g
        opt.step()
    wd = 0.1 if decay else 0.0
    expected = reference_adamw(start, grads, 0.01, 0.9, 0.95, 1e-8, wd)
    np.testing.assert_allclose(p.data, expected, rtol=1e-12)


def test_only_matmul_weights_decay():
    model = GPT(GPTConfig(vocab_size=5, block_size=4, n_layer=1, n_head=1, n_embd=4))
    decayed = sorted(name for name, p in model.named_parameters() if p.decay)
    assert decayed == [
        "blocks.0.attn.proj.weight",
        "blocks.0.attn.qkv.weight",
        "blocks.0.mlp.down.weight",
        "blocks.0.mlp.up.weight",
    ]


def test_cosine_schedule():
    kw = dict(max_lr=1.0, min_lr=0.1, warmup=10, total=110)
    assert cosine_lr(0, **kw) == pytest.approx(0.1)
    assert cosine_lr(9, **kw) == pytest.approx(1.0)
    assert cosine_lr(10, **kw) == pytest.approx(1.0)
    assert cosine_lr(60, **kw) == pytest.approx(0.55)
    assert cosine_lr(110, **kw) == pytest.approx(0.1)
    assert cosine_lr(500, **kw) == pytest.approx(0.1)
    values = [cosine_lr(s, **kw) for s in range(10, 111)]
    assert all(a >= b for a, b in zip(values, values[1:]))


def test_clip_grad_norm():
    a = Parameter(np.zeros(2))
    b = Parameter(np.zeros(1))
    a.grad = np.array([3.0, 0.0], dtype=np.float32)
    b.grad = np.array([4.0], dtype=np.float32)
    assert clip_grad_norm([a, b], 1.0) == pytest.approx(5.0)
    total = math.sqrt((a.grad**2).sum() + (b.grad**2).sum())
    assert total == pytest.approx(1.0, rel=1e-5)
    assert clip_grad_norm([a, b], 10.0) == pytest.approx(1.0, rel=1e-5)


def test_tiny_gpt_overfits_one_batch():
    config = GPTConfig(vocab_size=12, block_size=16, n_layer=2, n_head=2, n_embd=32)
    model = GPT(config, seed=0)
    rng = np.random.default_rng(0)
    ids = rng.integers(0, config.vocab_size, size=(4, 17))
    x, y = ids[:, :-1], ids[:, 1:]
    opt = AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)

    first = None
    for _ in range(150):
        loss = cross_entropy(model(x), y)
        first = first or loss.item()
        model.zero_grad()
        loss.backward()
        opt.step()

    assert first == pytest.approx(math.log(config.vocab_size), abs=0.1)
    assert loss.item() < 0.05
