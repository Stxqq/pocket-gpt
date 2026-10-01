import numpy as np
import pytest
from numerical import assert_grads_match

from pocketgpt.tensor import (
    Tensor,
    cross_entropy,
    dropout,
    layer_norm,
    no_grad,
    softmax,
)

rng = np.random.default_rng(0)


def param(*shape, positive=False):
    values = rng.standard_normal(shape)
    if positive:
        values = np.abs(values) + 0.5
    return Tensor(values, requires_grad=True)


def check(op, *inputs):
    """Gradient-check op(*inputs) through a random linear readout."""
    out_shape = op(*inputs).shape
    readout = Tensor(rng.standard_normal(out_shape))
    assert_grads_match(lambda: (op(*inputs) * readout).sum(), *inputs)


BINARY = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b,
}


@pytest.mark.parametrize("name", BINARY)
@pytest.mark.parametrize(
    "shapes", [((3, 4), (3, 4)), ((2, 3, 4), (4,)), ((2, 1, 4), (3, 1))]
)
def test_binary_ops_with_broadcasting(name, shapes):
    a = param(*shapes[0])
    b = param(*shapes[1], positive=True)
    check(BINARY[name], a, b)


def test_scalar_operands_on_both_sides():
    x = param(3, 2, positive=True)
    check(lambda x: 2.0 - x * 3 + 1 / x - (x / 4.0) + (-x), x)


def test_pow():
    x = param(3, 4, positive=True)
    check(lambda x: x**3 + x**0.5 + x**-1.5, x)


@pytest.mark.parametrize(
    "shapes",
    [
        ((4, 3), (3, 5)),
        ((2, 4, 3), (3, 5)),
        ((2, 3, 4, 3), (2, 3, 3, 5)),
        ((2, 1, 4, 3), (3, 3, 5)),
    ],
)
def test_matmul(shapes):
    check(lambda a, b: a @ b, param(*shapes[0]), param(*shapes[1]))


@pytest.mark.parametrize("axis", [None, 0, -1, (0, 2)])
@pytest.mark.parametrize("keepdims", [False, True])
def test_sum_and_mean(axis, keepdims):
    x = param(2, 3, 4)
    check(lambda x: x.sum(axis=axis, keepdims=keepdims), x)
    check(lambda x: x.mean(axis=axis, keepdims=keepdims), x)


@pytest.mark.parametrize("fn", ["exp", "log", "tanh", "gelu", "relu"])
def test_pointwise(fn):
    x = param(3, 5, positive=fn == "log")
    if fn == "relu":
        # keep inputs away from the kink where the derivative is undefined
        x.data += np.sign(x.data) * 0.1
    check(lambda x: getattr(x, fn)(), x)


def test_gelu_matches_reference_values():
    x = Tensor(np.array([-3.0, -1.0, 0.0, 0.5, 2.0]))
    expected = [-0.0036373, -0.1588080, 0.0, 0.3457140, 1.9545977]
    np.testing.assert_allclose(x.gelu().data, expected, atol=1e-6)


def test_reshape_and_transpose():
    x = param(2, 3, 4)
    check(lambda x: x.reshape(4, 6).T.reshape(2, 12), x)
    check(lambda x: x.transpose(2, 0, 1), x)


def test_basic_indexing():
    x = param(4, 5)
    check(lambda x: x[1:3, ::2], x)
    check(lambda x: x[2], x)


def test_gather_accumulates_repeated_rows():
    table = param(6, 3)
    ids = np.array([[0, 2, 2], [5, 0, 2]])
    check(lambda t: t[ids], table)

    table.grad = None
    table[ids].sum().backward()
    np.testing.assert_array_equal(table.grad[:, 0], [2, 0, 3, 0, 0, 1])


def test_masked_fill():
    x = param(3, 3)
    mask = np.triu(np.ones((3, 3), dtype=bool), k=1)
    out = x.masked_fill(mask, -1e9)
    assert (out.data[mask] == -1e9).all()
    check(lambda x: x.masked_fill(mask, 0.0), x)


def test_softmax():
    x = param(2, 3, 5)
    check(lambda x: softmax(x), x)
    check(lambda x: softmax(x, axis=0), x)
    np.testing.assert_allclose(softmax(x).data.sum(-1), 1.0)


def test_softmax_ignores_masked_logits():
    x = Tensor(np.array([[1.0, 2.0, -np.inf]]))
    probs = softmax(x).data
    assert probs[0, 2] == 0
    np.testing.assert_allclose(probs[0, :2], [0.268941, 0.731059], atol=1e-6)


def test_cross_entropy():
    logits = param(2, 3, 7)
    targets = rng.integers(0, 7, size=(2, 3))
    assert_grads_match(lambda: cross_entropy(logits, targets), logits)

    flat = logits.data.reshape(-1, 7)
    log_probs = flat - np.log(np.exp(flat).sum(1, keepdims=True))
    expected = -log_probs[np.arange(6), targets.reshape(-1)].mean()
    assert cross_entropy(logits, targets).item() == pytest.approx(expected)


def test_cross_entropy_is_stable_for_large_logits():
    logits = Tensor(np.array([[1000.0, 0.0]], dtype=np.float32))
    assert cross_entropy(logits, [1]).item() == pytest.approx(1000.0)


def test_layer_norm():
    x = param(2, 3, 6)
    w = param(6)
    b = param(6)
    check(lambda x, w, b: layer_norm(x, w, b), x, w, b)

    out = layer_norm(x, Tensor(np.ones(6)), Tensor(np.zeros(6))).data
    np.testing.assert_allclose(out.mean(-1), 0, atol=1e-7)
    np.testing.assert_allclose(out.std(-1), 1, atol=1e-4)


def test_dropout():
    x = param(200, 50)
    gen = np.random.default_rng(1)
    out = dropout(x, 0.25, gen)
    zeros = (out.data == 0).mean()
    assert 0.22 < zeros < 0.28
    kept = out.data != 0
    np.testing.assert_allclose(out.data[kept], x.data[kept] / 0.75)

    out.sum().backward()
    np.testing.assert_allclose(x.grad, kept / 0.75)
    assert dropout(x, 0.25, gen, training=False) is x


def test_shared_subexpression_accumulates():
    x = param(3)
    check(lambda x: x * x + x.exp() * x, x)


def test_no_grad_skips_the_graph():
    x = param(3)
    with no_grad():
        y = (x * 2).sum()
    assert not y.requires_grad
    assert y._parents == ()


def test_float32_stays_float32():
    x = Tensor(np.ones((2, 2)), dtype=np.float32)
    y = (x * 0.5 + 1) / 3.0 - np.float64(1.0)
    assert y.dtype == np.float32
    assert Tensor([1, 2, 3]).dtype == np.float32
