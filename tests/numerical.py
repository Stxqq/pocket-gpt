import numpy as np


def numerical_grad(loss_fn, array, eps=1e-6):
    """Central differences of a scalar function with respect to `array`."""
    grad = np.zeros_like(array)
    for i in np.ndindex(array.shape):
        original = array[i]
        array[i] = original + eps
        up = loss_fn()
        array[i] = original - eps
        down = loss_fn()
        array[i] = original
        grad[i] = (up - down) / (2 * eps)
    return grad


def assert_grads_match(build_loss, *inputs, rtol=1e-5, atol=1e-7):
    for t in inputs:
        t.grad = None
    build_loss().backward()
    for t in inputs:
        expected = numerical_grad(lambda: build_loss().item(), t.data)
        np.testing.assert_allclose(t.grad, expected, rtol=rtol, atol=atol)
