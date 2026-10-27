"""The only module that imports upstream TAMER: path setup, its global dictionary, model build,
checkpoint loading (upstream Lightning .ckpt or ours), loss and beam-search decoding."""
import os
import sys
from pathlib import Path

import torch

from .data import SPECIALS, word2idx

ROOT = Path(__file__).resolve().parent.parent
TAMER_DIR = Path(os.environ.get("TAMER_DIR", ROOT / "third_party" / "TAMER"))
if not (TAMER_DIR / "tamer").is_dir():
    raise SystemExit(f"TAMER not found at {TAMER_DIR}; run scripts/fetch_tamer.sh first")
sys.path.insert(0, str(TAMER_DIR))

from tamer.datamodule import vocab                                       # noqa: E402
from tamer.datamodule.latex2gtd import to_struct                         # noqa: E402
from tamer.model.tamer import TAMER                                      # noqa: E402
from tamer.utils.utils import ce_loss, to_bi_tgt_out, to_struct_output   # noqa: E402

# Released HME100K weights, "w/ fusion" row of the TAMER README (69.50 ExpRate on the official test).
# version_3 ships no dictionary.txt; it shares version_1's 245-word HME100K one (vocab_size 248).
PRETRAINED = TAMER_DIR / "lightning_logs/version_3/checkpoints/epoch=55-step=175503-val_ExpRate=0.6954.ckpt"
PRETRAINED_DICT = TAMER_DIR / "lightning_logs/version_1/dictionary.txt"
ARCH_KEYS = ("d_model", "growth_rate", "num_layers", "nhead", "num_decoder_layers",
             "dim_feedforward", "dropout", "dc", "cross_coverage", "self_coverage")
BEAM = {"max_len": 200, "alpha": 1.0, "early_stopping": False, "temperature": 1.0}


def read_checkpoint(path):
    """Upstream Lightning ckpt or ours -> {"model", "words", "arch", "invert", ...}.
    `invert` is None for upstream weights: their image polarity is checked on Kaggle (cell 4)."""
    path = PRETRAINED if str(path) == "pretrained" else path
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if "state_dict" not in ck:
        return ck
    prefix = "tamer_model."
    return {
        "model": {k[len(prefix):]: v for k, v in ck["state_dict"].items() if k.startswith(prefix)},
        "words": [line.strip() for line in PRETRAINED_DICT.read_text().splitlines()],
        "arch": {k: ck["hyper_parameters"][k] for k in ARCH_KEYS},
        "invert": None,
    }


def set_vocab(words):
    """Point upstream's global dictionary (used inside its decoder and beam search) at `words`."""
    vocab.word2idx = word2idx(words)
    vocab.idx2word = {i: w for w, i in vocab.word2idx.items()}


def build(arch, words, state):
    """TAMER sized for `words`. Weights whose vocab dim grew (embedding, output projection)
    keep the pretrained rows; rows for newly added tokens stay randomly initialised."""
    model = TAMER(**arch, vocab_size=len(SPECIALS) + len(words))
    own = model.state_dict()
    for k, v in state.items():
        if own[k].shape != v.shape:
            grown = own[k].clone()
            grown[: v.shape[0]] = v
            v = grown
        own[k] = v
    model.load_state_dict(own)
    return model


def struct_parses(tokens):
    """Upstream's to_struct_output survives only AssertionError/IndexError from its tree parser;
    rows that raise anything else (unfamiliar MathWriting LaTeX) would crash training."""
    try:
        to_struct(tokens)
    except (AssertionError, IndexError):
        pass
    except Exception:
        return False
    return True


def loss_fn(model, imgs, mask, indices):
    """Upstream training_step: bidirectional token CE + tree-structure CE."""
    tgt, out = to_bi_tgt_out(indices, imgs.device)
    struct_out, _ = to_struct_output(indices, imgs.device)
    out_hat, sim = model(imgs, mask, tgt)
    return ce_loss(out_hat, out) + ce_loss(sim, struct_out, ignore_idx=-1)


@torch.no_grad()
def predict(model, imgs, mask, beam_size):
    """Bidirectional beam search (+ structure-score fusion) -> space-separated token strings."""
    hyps = model.beam_search(imgs, mask, beam_size=beam_size, **BEAM)
    return [vocab.indices2label(h.seq) for h in hyps]
