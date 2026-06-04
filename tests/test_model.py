import torch

from gpt2 import GPT, GPTConfig


def test_forward_shape_and_loss():
    config = GPTConfig(vocab_size=11, block_size=8, n_layer=2, n_head=2, n_embd=16, dropout=0.0)
    model = GPT(config)
    idx = torch.randint(0, config.vocab_size, (4, config.block_size))
    logits, loss = model(idx, idx)
    assert logits.shape == (4, config.block_size, config.vocab_size)
    assert loss is not None
    assert loss.ndim == 0


def test_generation_extends_sequence():
    config = GPTConfig(vocab_size=7, block_size=4, n_layer=1, n_head=1, n_embd=8, dropout=0.0)
    model = GPT(config)
    idx = torch.randint(0, config.vocab_size, (2, 3))
    out = model.generate(idx, max_new_tokens=5, top_k=3)
    assert out.shape == (2, 8)


def test_causal_mask_prevents_future_leakage():
    torch.manual_seed(1)
    config = GPTConfig(vocab_size=13, block_size=6, n_layer=2, n_head=2, n_embd=16, dropout=0.0)
    model = GPT(config)
    model.eval()

    idx_a = torch.tensor([[1, 2, 3, 4, 5, 6]])
    idx_b = torch.tensor([[1, 2, 3, 9, 9, 9]])
    logits_a, _ = model(idx_a)
    logits_b, _ = model(idx_b)

    assert torch.allclose(logits_a[:, :3], logits_b[:, :3], atol=1e-5)
