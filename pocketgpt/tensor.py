"""A small reverse-mode autograd engine over numpy arrays.

Every op computes its forward value eagerly and, when gradients are being
tracked, remembers a closure that maps the output gradient to one gradient per
input. `Tensor.backward` walks the graph in reverse topological order.
"""

from __future__ import annotations

import contextlib
import math

import numpy as np

_grad_enabled = True


@contextlib.contextmanager
def no_grad():
    """Run a block without recording the graph (eval, sampling)."""
    global _grad_enabled
    previous, _grad_enabled = _grad_enabled, False
    try:
        yield
    finally:
        _grad_enabled = previous


def _unbroadcast(grad, shape):
    """Sum a gradient back down to the shape of the input that was broadcast."""
    if grad.shape == shape:
        return grad
    lead = grad.ndim - len(shape)
    if lead:
        grad = grad.sum(axis=tuple(range(lead)))
    axes = tuple(i for i, n in enumerate(shape) if n == 1 and grad.shape[i] != 1)
    if axes:
        grad = grad.sum(axis=axes, keepdims=True)
    return grad


def _is_basic_index(index):
    basic = (int, slice, type(None), type(Ellipsis))
    if isinstance(index, tuple):
        return all(isinstance(i, basic) for i in index)
    return isinstance(index, basic)


def _node(data, parents, backward):
    out = Tensor.__new__(Tensor)
    out.data = data
    out.grad = None
    out.requires_grad = _grad_enabled and any(p.requires_grad for p in parents)
    out._parents = parents if out.requires_grad else ()
    out._backward = backward if out.requires_grad else None
    return out


class Tensor:
    """An n-dimensional array that records how it was computed."""

    # makes `ndarray + Tensor` defer to Tensor.__radd__ instead of broadcasting
    __array_priority__ = 100

    def __init__(self, data, requires_grad=False, dtype=None):
        if isinstance(data, Tensor):
            data = data.data
        if dtype is None:
            is_float = isinstance(data, np.ndarray) and data.dtype.kind == "f"
            dtype = data.dtype if is_float else np.float32
        self.data = np.asarray(data, dtype=dtype)
        self.grad = None
        self.requires_grad = requires_grad
        self._parents = ()
        self._backward = None

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        flag = ", requires_grad=True" if self.requires_grad else ""
        return f"Tensor({self.data!r}{flag})"

    def item(self):
        return self.data.item()

    def numpy(self):
        return self.data

    def detach(self):
        return Tensor(self.data)

    def _lift(self, other):
        if isinstance(other, Tensor):
            return other
        return Tensor(np.asarray(other, dtype=self.data.dtype))

    def backward(self, grad=None):
        """Accumulate d(self)/d(leaf) into `.grad` of every leaf that needs it."""
        if grad is None:
            grad = np.ones_like(self.data)

        order, seen = [], set()
        stack = [(self, False)]
        while stack:
            node, done = stack.pop()
            if done:
                order.append(node)
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.append((node, True))
            stack.extend((p, False) for p in node._parents if p.requires_grad)

        # gradients of intermediate nodes live only here and are dropped as
        # soon as they have been pushed to the parents
        pending = {id(self): grad}
        for node in reversed(order):
            g = pending.pop(id(node), None)
            if g is None:
                continue
            if node._backward is None:
                node.grad = g if node.grad is None else node.grad + g
                continue
            for parent, pg in zip(node._parents, node._backward(g), strict=True):
                if pg is None or not parent.requires_grad:
                    continue
                key = id(parent)
                pending[key] = pg if key not in pending else pending[key] + pg

    def __add__(self, other):
        other = self._lift(other)

        def backward(g):
            return _unbroadcast(g, self.shape), _unbroadcast(g, other.shape)

        return _node(self.data + other.data, (self, other), backward)

    __radd__ = __add__

    def __sub__(self, other):
        other = self._lift(other)

        def backward(g):
            return _unbroadcast(g, self.shape), _unbroadcast(-g, other.shape)

        return _node(self.data - other.data, (self, other), backward)

    def __rsub__(self, other):
        return self._lift(other) - self

    def __neg__(self):
        return _node(-self.data, (self,), lambda g: (-g,))

    def __mul__(self, other):
        other = self._lift(other)

        def backward(g):
            return (
                _unbroadcast(g * other.data, self.shape),
                _unbroadcast(g * self.data, other.shape),
            )

        return _node(self.data * other.data, (self, other), backward)

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = self._lift(other)

        def backward(g):
            ga = g / other.data
            gb = -ga * self.data / other.data
            return _unbroadcast(ga, self.shape), _unbroadcast(gb, other.shape)

        return _node(self.data / other.data, (self, other), backward)

    def __rtruediv__(self, other):
        return self._lift(other) / self

    def __pow__(self, exponent):
        if isinstance(exponent, Tensor):
            raise TypeError("only constant exponents are supported")

        def backward(g):
            return (g * exponent * self.data ** (exponent - 1),)

        return _node(self.data**exponent, (self,), backward)

    def __matmul__(self, other):
        a, b = self.data, other.data

        def backward(g):
            ga = _unbroadcast(g @ np.swapaxes(b, -1, -2), a.shape)
            if b.ndim == 2 and a.ndim > 2:
                # a stack of activations times one weight matrix: fold the
                # batch into a single matmul instead of summing B products
                gb = a.reshape(-1, a.shape[-1]).T @ g.reshape(-1, g.shape[-1])
            else:
                gb = _unbroadcast(np.swapaxes(a, -1, -2) @ g, b.shape)
            return ga, gb

        return _node(a @ b, (self, other), backward)

    def sum(self, axis=None, keepdims=False):
        shape = self.shape

        def backward(g):
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            return (np.broadcast_to(g, shape),)

        return _node(self.data.sum(axis=axis, keepdims=keepdims), (self,), backward)

    def mean(self, axis=None, keepdims=False):
        axes = range(self.ndim) if axis is None else np.atleast_1d(axis)
        count = math.prod(self.shape[a] for a in axes)
        return self.sum(axis=axis, keepdims=keepdims) / count

    def exp(self):
        out = np.exp(self.data)
        return _node(out, (self,), lambda g: (g * out,))

    def log(self):
        return _node(np.log(self.data), (self,), lambda g: (g / self.data,))

    def tanh(self):
        out = np.tanh(self.data)
        return _node(out, (self,), lambda g: (g * (1 - out * out),))

    def relu(self):
        return _node(
            np.maximum(self.data, 0), (self,), lambda g: (g * (self.data > 0),)
        )

    def gelu(self):
        """GELU with the tanh approximation used by GPT-2."""
        x = self.data
        c = math.sqrt(2 / math.pi)
        # x * x * x rather than x**3: float32 power is ~35x slower in numpy
        t = np.tanh(c * (x + 0.044715 * x * x * x))

        def backward(g):
            dt = c * (1 + 3 * 0.044715 * x * x) * (1 - t * t)
            return (g * (0.5 * (1 + t) + 0.5 * x * dt),)

        return _node(0.5 * x * (1 + t), (self,), backward)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        original = self.shape
        return _node(
            self.data.reshape(shape), (self,), lambda g: (g.reshape(original),)
        )

    def transpose(self, *axes):
        axes = axes or tuple(reversed(range(self.ndim)))
        inverse = tuple(np.argsort(axes))
        return _node(
            self.data.transpose(axes), (self,), lambda g: (g.transpose(inverse),)
        )

    @property
    def T(self):
        return self.transpose()

    def __getitem__(self, index):
        if isinstance(index, Tensor):
            index = index.data.astype(np.intp)

        def backward(g):
            full = np.zeros_like(self.data)
            if _is_basic_index(index):
                full[index] = g
            else:
                # rows can repeat (the same token twice in a batch), so the
                # scatter has to accumulate rather than assign
                np.add.at(full, index, g)
            return (full,)

        return _node(self.data[index], (self,), backward)

    def masked_fill(self, mask, value):
        """Masked entries become `value` and pass no gradient back."""
        out = np.where(mask, np.asarray(value, dtype=self.dtype), self.data)
        return _node(out, (self,), lambda g: (np.where(mask, 0, g),))


def dropout(x, p, rng, training=True):
    """Inverted dropout: zero a fraction p of entries and rescale the rest."""
    if not training or p == 0:
        return x
    keep = (rng.random(x.shape) >= p).astype(x.dtype) / (1 - p)
    return _node(x.data * keep, (x,), lambda g: (g * keep,))


def softmax(x, axis=-1):
    z = np.exp(x.data - x.data.max(axis=axis, keepdims=True))
    probs = z / z.sum(axis=axis, keepdims=True)

    def backward(g):
        return (probs * (g - (g * probs).sum(axis=axis, keepdims=True)),)

    return _node(probs, (x,), backward)


def cross_entropy(logits, targets):
    """Mean negative log-likelihood of integer `targets` under `logits`."""
    vocab = logits.shape[-1]
    z = logits.data.reshape(-1, vocab)
    targets = np.asarray(targets).reshape(-1)
    rows = np.arange(len(targets))

    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    total = e.sum(axis=1, keepdims=True)
    loss = (np.log(total[:, 0]) - z[rows, targets]).mean()

    def backward(g):
        dz = e / total
        dz[rows, targets] -= 1
        return ((dz * (g / len(targets))).reshape(logits.shape),)

    return _node(np.asarray(loss, dtype=logits.dtype), (logits,), backward)


def layer_norm(x, weight, bias, eps=1e-5):
    """Layer norm as one node with the closed-form backward, not a dozen small ones."""
    mu = x.data.mean(axis=-1, keepdims=True)
    centered = x.data - mu
    rstd = 1 / np.sqrt((centered * centered).mean(axis=-1, keepdims=True) + eps)
    xhat = centered * rstd

    def backward(g):
        lead = tuple(range(g.ndim - 1))
        dxhat = g * weight.data
        dx = rstd * (
            dxhat
            - dxhat.mean(axis=-1, keepdims=True)
            - xhat * (dxhat * xhat).mean(axis=-1, keepdims=True)
        )
        return dx, (g * xhat).sum(axis=lead), g.sum(axis=lead)

    return _node(xhat * weight.data + bias.data, (x, weight, bias), backward)
