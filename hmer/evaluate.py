"""ExpRate, ExpRate<=1, ExpRate<=2 and BLEU for a TAMER checkpoint, scored like model/evaluate.py:
reference and prediction both go through model.tokenizer.tokenize_latex, so the numbers sit in
the same table as the ResNet-18 + LSTM models.

    python -m hmer.evaluate --manifest data/hme100k/test.jsonl --ckpt checkpoints/tamer_mix_best.pth --limit 500
    python -m hmer.evaluate --manifest data/hme100k/test.jsonl --ckpt pretrained --invert --limit 500
"""
import argparse

import torch
from tqdm.auto import tqdm

from model.tokenizer import tokenize_latex
from model.train import load_manifest

from .data import collate, load_image
from .upstream import build, predict, read_checkpoint, set_vocab

try:
    from model.metrics import score
except ImportError:  # model/metrics.py not on this branch yet: same formula, from model/evaluate.py
    from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

    from model.evaluate import edit_distance

    def score(refs, hyps):
        dists = [edit_distance(r, h) for r, h in zip(refs, hyps)]
        n = max(1, len(hyps))
        return {
            "n": len(hyps),
            "bleu": corpus_bleu([[r] for r in refs], hyps,
                                smoothing_function=SmoothingFunction().method4),
            "exprate": sum(d == 0 for d in dists) / n,
            "exprate_le1": sum(d <= 1 for d in dists) / n,
            "exprate_le2": sum(d <= 2 for d in dists) / n,
        }


def evaluate(ckpt, manifest, limit=None, beam=10, invert=None, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = read_checkpoint(ckpt)
    invert = ck["invert"] if invert is None else invert
    if invert is None:
        raise SystemExit("upstream weights: pass --invert or --no-invert (notebook cell 4 picks)")
    set_vocab(ck["words"])
    model = build(ck["arch"], ck["words"], ck["model"]).to(device).eval()
    refs, hyps = [], []
    for path, latex in tqdm(load_manifest(manifest)[:limit], desc="eval", unit="img"):
        imgs, mask = collate([load_image(path, invert)])
        pred = predict(model, imgs.to(device), mask.to(device), beam)[0]
        refs.append(tokenize_latex(latex))
        hyps.append(tokenize_latex(pred))
    return score(refs, hyps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ckpt", required=True, help="'pretrained' or a .pth from hmer.train")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--beam", type=int, default=10, help="upstream default")
    ap.add_argument("--invert", action=argparse.BooleanOptionalAction, default=None,
                    help="override the polarity stored in the checkpoint")
    args = ap.parse_args()
    print(evaluate(args.ckpt, args.manifest, args.limit, args.beam, args.invert))


if __name__ == "__main__":
    main()
