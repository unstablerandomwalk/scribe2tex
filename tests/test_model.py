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


LEGACY = {**CONFIG, "fine_grid": False, "coverage": False}


def test_fine_grid_doubles_feature_rows():
    img = torch.randn(1, 3, 128, 256)
    _, (h32, _) = Encoder(CONFIG["encoder_dim"], pretrained=False)(img)
    _, (h16, _) = Encoder(CONFIG["encoder_dim"], pretrained=False, fine_grid=True)(img)
    assert (h32, h16) == (4, 8)


def test_coverage_warm_start_matches_old_model():
    import tempfile
    from model.model import warm_start
    torch.manual_seed(0)
    old = Seq2Seq(50, LEGACY, pretrained=False).eval()
    path = Path(tempfile.mkdtemp()) / "old.pth"
    torch.save({"model": old.state_dict()}, path)
    new = Seq2Seq(50, {**LEGACY, "coverage": True}, pretrained=False).eval()
    missing = warm_start(new, path, "cpu")
    assert missing and all("coverage_att" in k for k in missing)
    caps = torch.randint(4, 50, (1, 8)); caps[:, 0] = SOS_ID
    img = torch.randn(1, 3, 128, 256)
    assert torch.allclose(old(img, caps)[0], new(img, caps)[0], atol=1e-6)


def test_load_model_rebuilds_legacy_checkpoint():
    import tempfile
    from model.model import load_model
    path = Path(tempfile.mkdtemp()) / "legacy.pth"
    torch.save({"model": Seq2Seq(50, LEGACY, pretrained=False).state_dict()}, path)
    model = load_model(path, 50, CONFIG, "cpu")             # no "arch" key -> legacy
    assert model.decoder.attention.coverage_att is None


def test_train_augmentation_keeps_shape():
    from PIL import Image
    from model.dataset import transform_image
    img = Image.new("L", (300, 80), 255)
    for _ in range(20):                                   # hit the random branches
        t, hw = transform_image(img, 128, 1024, train=True)
        assert t.shape[:2] == (3, 128) and hw[0] == 128


def _run():
    fns = [f for n, f in sorted(globals().items()) if n.startswith("test_")]
    for f in fns:
        f()
        print(f"  ok: {f.__name__}")
    print(f"\n{len(fns)} model tests passed")


if __name__ == "__main__":
    _run()
