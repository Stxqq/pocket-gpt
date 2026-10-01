import numpy as np
import pytest
from numerical import numerical_grad

from pocketgpt.nn import GPT, GPTConfig, sample_token
from pocketgpt.tensor import cross_entropy

TINY = GPTConfig(vocab_size=11, block_size=8, n_layer=2, n_head=2, n_embd=12)


def tiny_batch(seed=0, batch=2, steps=6):
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, TINY.vocab_size, size=(batch, steps + 1))
    return ids[:, :-1], ids[:, 1:]


def test_whole_model_gradients_match_finite_differences():
    model = GPT(TINY, seed=3).astype(np.float64)
    # perturb LN params away from 1/0 so their gradients are not degenerate
    pick = np.random.default_rng(1)
    for name, p in model.named_parameters():
        if "ln" in name:
            p.data += pick.normal(0, 0.1, p.shape)
    ids, targets = tiny_batch()

    def loss():
        return cross_entropy(model(ids), targets)

    loss().backward()
    for name, p in model.named_parameters():
        assert p.dtype == np.float64
        # check a random handful of entries per tensor to keep the test fast
        flat = p.data.reshape(-1)
        entries = pick.choice(flat.size, size=min(6, flat.size), replace=False)
        for i in entries:
            view = flat[i : i + 1]
            expected = numerical_grad(lambda: loss().item(), view)[0]
            got = p.grad.reshape(-1)[i]
            assert np.isclose(got, expected, rtol=1e-4, atol=1e-8), name


def test_future_tokens_never_change_past_logits():
    model = GPT(TINY, seed=0).eval()
    ids, _ = tiny_batch(steps=8)
    base = model(ids).data
    for t in range(ids.shape[1]):
        changed = ids.copy()
        changed[:, t] = (changed[:, t] + 1) % TINY.vocab_size
        logits = model(changed).data
        np.testing.assert_array_equal(logits[:, :t], base[:, :t])
        assert not np.allclose(logits[:, t], base[:, t])


def test_output_head_is_tied_to_token_embedding():
    model = GPT(TINY)
    names = [name for name, _ in model.named_parameters()]
    assert not any("head" in name for name in names)
    assert len(names) == len(set(names))
    ids, targets = tiny_batch()
    cross_entropy(model(ids), targets).backward()
    assert model.wte.weight.grad is not None
    assert np.abs(model.wte.weight.grad).sum() > 0


def test_parameter_count():
    c, v, b, n = TINY.n_embd, TINY.vocab_size, TINY.block_size, TINY.n_layer
    ln, qkv, proj = 2 * 2 * c, 3 * c * c + 3 * c, c * c + c
    up, down = 4 * c * c + 4 * c, 4 * c * c + c
    per_block = ln + qkv + proj + up + down
    expected = v * c + b * c + n * per_block + 2 * c
    assert GPT(TINY).num_params() == expected


def test_state_dict_round_trip():
    a, b = GPT(TINY, seed=1), GPT(TINY, seed=2)
    ids, _ = tiny_batch()
    assert not np.allclose(a(ids).data, b(ids).data)
    b.load_state_dict(a.state_dict())
    np.testing.assert_array_equal(a(ids).data, b(ids).data)


def test_dropout_only_in_training():
    model = GPT(GPTConfig(**{**vars(TINY), "dropout": 0.5}), seed=0)
    ids, _ = tiny_batch()
    assert not np.allclose(model(ids).data, model(ids).data)
    model.eval()
    np.testing.assert_array_equal(model(ids).data, model(ids).data)


def test_generate_respects_block_size_and_restores_mode():
    model = GPT(TINY, seed=0)
    tokens = list(model.generate([1, 2], 20, rng=np.random.default_rng(0)))
    assert len(tokens) == 20
    assert all(0 <= t < TINY.vocab_size for t in tokens)
    assert model.training
    with pytest.raises(ValueError):
        next(model.generate([], 1))


def test_top_k_and_greedy_sampling():
    logits = np.array([0.0, 5.0, 4.0, -1.0])
    rng = np.random.default_rng(0)
    assert sample_token(logits, 0, None, rng) == 1
    draws = {sample_token(logits, 1.0, 2, rng) for _ in range(200)}
    assert draws == {1, 2}
