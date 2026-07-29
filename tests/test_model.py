"""Forward-pass + shape + pad-mask sanity. Needs torch -> run on Colab:
`python tests/test_model.py`. Exits clean (skip) if torch is absent."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import torch
except ImportError:
    print("torch not installed - skipping model tests (run on Colab)")
    sys.exit(0)

from model.config import CONFIG, SOS_ID                       # noqa: E402
from model.model import BahdanauAttention, Encoder, Seq2Seq   # noqa: E402


def test_encoder_returns_seq_and_dims():
    enc = Encoder(CONFIG["encoder_dim"], pretrained=False)
    feats, (fh, fw) = enc(torch.randn(2, 3, 128, 320))
    assert feats.shape == (2, fh * fw, CONFIG["encoder_dim"])


def test_attention_shapes_and_softmax():
    att = BahdanauAttention(256, 256, 128)
    ctx, alpha = att(torch.randn(2, 35, 256), torch.randn(2, 256))
    assert ctx.shape == (2, 256) and alpha.shape == (2, 35)
    assert torch.allclose(alpha.sum(1), torch.ones(2), atol=1e-4)


def test_attention_mask_zeros_padded_cells():
    att = BahdanauAttention(256, 256, 128)
    mask = torch.ones(1, 4, dtype=torch.bool)
    mask[0, 2:] = False                                   # last 2 cells are padding
    _, alpha = att(torch.randn(1, 4, 256), torch.randn(1, 256), mask)
    assert torch.allclose(alpha[0, 2:], torch.zeros(2), atol=1e-6)  # no weight on padding
    assert abs(alpha.sum().item() - 1.0) < 1e-4


def test_seq2seq_forward_logits_shape():
    V = 50
    model = Seq2Seq(V, CONFIG, pretrained=False)
    caps = torch.randint(4, V, (2, 12)); caps[:, 0] = SOS_ID
    valid_hw = torch.tensor([[128, 256], [128, 128]])     # 2nd image half-width padded
    logits, alphas = model(torch.randn(2, 3, 128, 256), caps, valid_hw, 1.0)
    assert logits.shape == (2, 11, V)                     # T-1 decode steps
    assert alphas.shape[:2] == (2, 11)


def test_greedy_returns_id_lists():
    model = Seq2Seq(50, CONFIG, pretrained=False)
    seqs = model.greedy(torch.randn(1, 3, 128, 256), max_len=10)
    assert isinstance(seqs, list) and len(seqs) == 1


def _run():
    fns = [f for n, f in sorted(globals().items()) if n.startswith("test_")]
    for f in fns:
        f()
        print(f"  ok: {f.__name__}")
    print(f"\n{len(fns)} model tests passed")


if __name__ == "__main__":
    _run()
