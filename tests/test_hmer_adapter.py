"""hmer adapter: one manifest row -> TAMER input tensors -> back to the same LaTeX.
No upstream TAMER needed. `python -m pytest tests/test_hmer_adapter.py`"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hmer.data import (SPECIALS, batches, collate, decode, encode, extend_dictionary,  # noqa: E402
                       load_image, to_tokens, word2idx)

BASE = ["x", "^", "{", "2", "}", "\\frac", "1", "+", "的"]      # stand-in for TAMER's dictionary


def test_manifest_row_round_trips_through_model_input(tmp_path):
    img_path = tmp_path / "a.png"
    Image.fromarray(np.full((40, 100), 255, np.uint8)).save(img_path)
    row = {"image": str(img_path), "latex": r"\frac{1}{x^{2}}+\alpha"}      # one JSONL line

    tokens = to_tokens(row["latex"], set(BASE))
    words = extend_dictionary(BASE, [tokens])
    ids = encode(tokens, word2idx(words))
    imgs, mask = collate([load_image(row["image"])])

    assert words[: len(BASE)] == BASE and words[len(BASE):] == ["\\alpha"]   # pretrained ids kept
    assert min(ids) >= len(SPECIALS)
    assert decode(ids, words) == r"\frac { 1 } { x ^ { 2 } } + \alpha"
    assert imgs.shape == (1, 1, 40, 100) and mask.shape == (1, 40, 100) and not mask.any()
    assert float(imgs.max()) == 1.0 and float(load_image(row["image"], invert=True).max()) == 0.0


def test_hme100k_label_kept_verbatim():
    # already-tokenized HME100K labels (incl. Chinese words) must hit TAMER's dictionary as-is
    assert to_tokens("x ^ { 2 } 的", set(BASE)) == ["x", "^", "{", "2", "}", "的"]


def test_unknown_token_is_rejected_not_mapped():
    assert encode(["\\beta"], word2idx(BASE)) is None


def test_padding_mask_marks_only_padding():
    imgs, mask = collate([torch.ones(2, 3), torch.ones(4, 5)])
    assert imgs.shape == (2, 1, 4, 5)
    assert not mask[0, :2, :3].any() and mask[0, 2:, :].all() and mask[0, :, 3:].all()
    assert not mask[1].any()


def test_batches_respect_area_and_count():
    sizes = [(100, 1000)] * 3 + [(10, 10)] * 20
    out = batches(sizes, max_area=250_000, max_batch=8)
    assert sorted(i for b in out for i in b) == list(range(23))
    for b in out:
        assert len(b) <= 8 and max(sizes[i][0] * sizes[i][1] for i in b) * len(b) <= 250_000

