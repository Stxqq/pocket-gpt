import json

import numpy as np

from pocketgpt.checkpoint import load_checkpoint, save_checkpoint
from pocketgpt.export import export_web, read_web
from pocketgpt.nn import GPT, GPTConfig
from pocketgpt.tokenizer import CharTokenizer

TOKENIZER = CharTokenizer.from_text("\n :ABEMOR,.abcdeilmnorstuw")
CONFIG = GPTConfig(
    vocab_size=TOKENIZER.vocab_size, block_size=16, n_layer=2, n_head=2, n_embd=16
)


def test_web_export_round_trip(tmp_path):
    model = GPT(CONFIG, seed=4)
    manifest = export_web(model, TOKENIZER, tmp_path, prompt="ROMEO:", step=7)

    state, loaded = read_web(tmp_path)
    assert loaded == json.loads(json.dumps(manifest))
    assert loaded["step"] == 7
    assert loaded["bytes"] == (tmp_path / "weights.bin").stat().st_size
    assert loaded["bytes"] == 4 * model.num_params()
    assert list(state) == [name for name, _ in model.named_parameters()]

    clone = GPT(GPTConfig(**loaded["config"]), seed=99)
    clone.load_state_dict(state)
    for name, p in model.named_parameters():
        np.testing.assert_array_equal(dict(clone.named_parameters())[name].data, p.data)

    fixture = json.loads((tmp_path / "fixture.json").read_text())
    assert TOKENIZER.decode(fixture["tokens"]) == "ROMEO:"
    expected = clone.eval()(np.array([fixture["tokens"]])).data[0]
    np.testing.assert_allclose(fixture["logits"], expected, rtol=1e-6)


def test_checkpoint_round_trip(tmp_path):
    model = GPT(CONFIG, seed=5)
    path = tmp_path / "ckpt.npz"
    save_checkpoint(path, model, TOKENIZER, step=3, val=1.5)
    loaded, tokenizer, meta = load_checkpoint(path)

    assert meta["step"] == 3 and meta["val"] == 1.5
    assert tokenizer.chars == TOKENIZER.chars
    assert not loaded.training
    ids = TOKENIZER.encode("ROMEO: be")[None]
    np.testing.assert_array_equal(loaded(ids).data, model.eval()(ids).data)
    assert [p.name for p in tmp_path.iterdir()] == ["ckpt.npz"]
