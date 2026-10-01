import numpy as np
import pytest

from pocketgpt.data import get_batch
from pocketgpt.tokenizer import CharTokenizer


def test_char_tokenizer_round_trip():
    tok = CharTokenizer.from_text("to be, or not to be")
    assert tok.chars == sorted(set("to be, or not to be"))
    ids = tok.encode("not to be")
    assert ids.dtype == np.int64
    assert tok.decode(ids) == "not to be"


def test_unknown_character_is_reported():
    with pytest.raises(ValueError, match="'z'"):
        CharTokenizer.from_text("abc").encode("abz")


def test_batch_targets_are_shifted_inputs():
    tokens = np.arange(100, dtype=np.uint16)
    x, y = get_batch(tokens, batch_size=8, block_size=10, rng=np.random.default_rng(0))
    assert x.shape == y.shape == (8, 10)
    np.testing.assert_array_equal(y, x + 1)
    np.testing.assert_array_equal(np.diff(x, axis=1), 1)
    assert y.max() <= 99
