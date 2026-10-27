"""Manifest rows -> TAMER inputs. No upstream import, so this (and its test) runs anywhere.

TAMER's input contract: grayscale image in [0, 1] as [b, 1, H, W], a bool pad mask [b, H, W]
(True = padding), and token-id lists over its dictionary, where ids 0/1/2 are <pad>/<sos>/<eos>."""
import math
from collections import Counter

import numpy as np
import torch
from PIL import Image

from model.tokenizer import tokenize_latex

SPECIALS = ["<pad>", "<sos>", "<eos>"]
MAX_AREA = 480_000      # TAMER's hme100k max_size: caps one image here and one batch in batches()
MAX_BATCH = 8           # TAMER's train_batch_size
MAX_LEN = 200           # TAMER's max_len


def to_tokens(latex, known):
    """HME100K labels are already space-tokenized in TAMER's dictionary, so known pieces are kept
    as-is; raw LaTeX pieces (MathWriting's `\\frac{a}{b}`) go through the project tokenizer."""
    out = []
    for piece in latex.split():
        out += [piece] if piece in known else tokenize_latex(piece)
    return out


def extend_dictionary(base_words, token_lists):
    """Pretrained words keep their ids; tokens new to TAMER are appended after them."""
    known = set(base_words)
    new = Counter(t for toks in token_lists for t in toks if t not in known)
    return list(base_words) + sorted(new)


def word2idx(words):
    return {w: i for i, w in enumerate(SPECIALS + list(words))}


def encode(tokens, w2i):
    """Token list -> ids, or None when a token is outside the dictionary (TAMER has no <unk>)."""
    if any(t not in w2i for t in tokens):
        return None
    return [w2i[t] for t in tokens]


def decode(ids, words):
    itos = SPECIALS + list(words)
    return " ".join(itos[i] for i in ids if i >= len(SPECIALS))


def fitted_hw(path):
    """(h, w) after load_image's resize; reads only the header."""
    w, h = Image.open(path).size
    scale = min(1.0, math.sqrt(MAX_AREA / (h * w)))
    return max(1, round(h * scale)), max(1, round(w * scale))


def load_image(path, invert=False):
    """-> float [H, W] in [0, 1]. Aspect kept; only images over MAX_AREA are shrunk."""
    img = Image.open(path).convert("L")
    h, w = fitted_hw(path)
    if (w, h) != img.size:
        img = img.resize((w, h), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(1.0 - arr if invert else arr)


def collate(images):
    """[H_i, W_i] list -> zero-padded [b, 1, H, W] + mask [b, H, W] (True = pad), as upstream."""
    h = max(im.shape[0] for im in images)
    w = max(im.shape[1] for im in images)
    x = torch.zeros(len(images), 1, h, w)
    mask = torch.ones(len(images), h, w, dtype=torch.bool)
    for k, im in enumerate(images):
        x[k, 0, : im.shape[0], : im.shape[1]] = im
        mask[k, : im.shape[0], : im.shape[1]] = False
    return x, mask


def batches(sizes, max_area=MAX_AREA, max_batch=MAX_BATCH):
    """Upstream's data_iterator: sort by area, then pack while biggest_area * count <= max_area."""
    order = sorted(range(len(sizes)), key=lambda i: sizes[i][0] * sizes[i][1])
    out, cur, biggest = [], [], 0
    for i in order:
        area = sizes[i][0] * sizes[i][1]
        if cur and (len(cur) == max_batch or max(biggest, area) * (len(cur) + 1) > max_area):
            out.append(cur)
            cur, biggest = [], 0
        cur.append(i)
        biggest = max(biggest, area)
    if cur:
        out.append(cur)
    return out


class BatchSet(torch.utils.data.Dataset):
    """One item = one prebuilt batch: (images [b,1,H,W], mask [b,H,W], id lists)."""

    def __init__(self, samples, batch_list, invert):
        self.samples, self.batch_list, self.invert = samples, batch_list, invert

    def __len__(self):
        return len(self.batch_list)

    def __getitem__(self, k):
        rows = [self.samples[i] for i in self.batch_list[k]]
        imgs, mask = collate([load_image(path, self.invert) for path, _ in rows])
        return imgs, mask, [ids for _, ids in rows]
