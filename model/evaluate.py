"""BLEU, ExpRate, ExpRate<=1/<=2 over a manifest. Same metrics for your model and
pix2tex -> the benchmark table.
    python -m model.evaluate --manifest data/crohme/test.jsonl --ckpt checkpoints/finetune_best.pth
"""
import argparse

import torch
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

from .config import CONFIG, VOCAB_PATH
from .dataset import transform_image
from .inference import beam_search
from .model import Seq2Seq
from .tokenizer import Vocab, tokenize_latex
from .train import load_manifest


def edit_distance(a, b):
    """Token-level Levenshtein, for ExpRate<=k near-miss buckets."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return dp[n]


def evaluate(model, samples, vocab, cfg, device, beam_width=5, limit=None):
    if limit:
        samples = samples[:limit]
    refs, hyps, dists = [], [], []
    for path, latex in samples:
        img, valid_hw = transform_image(path, cfg["image_height"], cfg["image_max_width"])
        img = img.unsqueeze(0).to(device)
        pred, _ = beam_search(model, img, vocab, valid_hw.unsqueeze(0).to(device),
                              beam_width, cfg["max_seq_length"], cfg["length_penalty"])
        ref, hyp = tokenize_latex(latex), pred.split()
        refs.append([ref]); hyps.append(hyp); dists.append(edit_distance(ref, hyp))
    n = max(1, len(hyps))
    return {
        "n": len(hyps),
        "bleu": corpus_bleu(refs, hyps, smoothing_function=SmoothingFunction().method4),
        "exprate": sum(d == 0 for d in dists) / n,
        "exprate_le1": sum(d <= 1 for d in dists) / n,
        "exprate_le2": sum(d <= 2 for d in dists) / n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--beam", type=int, default=CONFIG["beam_width"])
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab = Vocab.load(VOCAB_PATH)
    model = Seq2Seq(len(vocab), CONFIG, pretrained=False).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)["model"])
    print(evaluate(model, load_manifest(args.manifest), vocab, CONFIG, device,
                   args.beam, args.limit))


if __name__ == "__main__":
    main()
