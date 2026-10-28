"""BLEU, ExpRate, ExpRate<=1/<=2 over a manifest, plus selective accuracy (ExpRate on
the most-confident X% of predictions). pix2tex is scored by the same model/metrics.py.
    python -m model.evaluate --manifest data/crohme/test.jsonl --ckpt checkpoints/finetune_best.pth
"""
import argparse
import json

import torch
from tqdm.auto import tqdm

from .config import CONFIG, VOCAB_PATH
from .dataset import transform_image
from .inference import beam_search
from .metrics import score, selective
from .model import load_model
from .tokenizer import Vocab, tokenize_latex
from .train import load_manifest


def evaluate(model, samples, vocab, cfg, device, beam_width=5, limit=None, out=None):
    if limit:
        samples = samples[:limit]
    refs, hyps, confs = [], [], []
    for path, latex in tqdm(samples, desc="eval", unit="img"):
        img, valid_hw = transform_image(path, cfg["image_height"], cfg["image_max_width"])
        img = img.unsqueeze(0).to(device)
        pred, conf = beam_search(model, img, vocab, valid_hw.unsqueeze(0).to(device),
                                 beam_width, cfg["max_seq_length"], cfg["length_penalty"])
        refs.append(tokenize_latex(latex)); hyps.append(pred.split()); confs.append(conf)
    correct = [r == h for r, h in zip(refs, hyps)]
    if out:  # per-sample predictions, for error analysis
        with open(out, "w") as f:
            for (path, _), r, h, c in zip(samples, refs, hyps, confs):
                f.write(json.dumps({"image": path, "ref": r, "hyp": h, "conf": c}) + "\n")
    return {**score(refs, hyps), **selective(correct, confs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--beam", type=int, default=CONFIG["beam_width"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", help="write per-sample predictions (JSONL) here")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab = Vocab.load(VOCAB_PATH)
    model = load_model(args.ckpt, len(vocab), CONFIG, device)
    print(evaluate(model, load_manifest(args.manifest), vocab, CONFIG, device,
                   args.beam, args.limit, args.out))


if __name__ == "__main__":
    main()
