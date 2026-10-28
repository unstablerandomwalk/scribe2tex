"""Score pix2tex (ViT + transformer LaTeX-OCR) on the same manifest and metrics as ours.

    pip install pix2tex          # separate venv recommended: it pins its own timm/x-transformers
    python scripts/eval_pix2tex.py --manifest data/mathwriting/test.jsonl --limit 500

Both models' outputs and references go through the same tokenize_latex + model/metrics.py,
so the numbers sit in one table. pix2tex exposes no confidence, so no selective row.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image                                    # noqa: E402
from pix2tex.cli import LatexOCR                         # noqa: E402
from tqdm import tqdm                                    # noqa: E402

from model.metrics import score                          # noqa: E402
from model.tokenizer import tokenize_latex               # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    rows = [json.loads(l) for l in Path(args.manifest).read_text().splitlines() if l.strip()]
    rows = rows[:args.limit] if args.limit else rows
    ocr = LatexOCR()
    refs, hyps, failures = [], [], 0
    for row in tqdm(rows):
        try:
            pred = ocr(Image.open(row["image"]).convert("RGB"))
        except Exception as e:                           # count it as a wrong answer
            print(f"pix2tex failed on {row['image']}: {e}", file=sys.stderr)
            pred, failures = "", failures + 1
        refs.append(tokenize_latex(row["latex"])); hyps.append(tokenize_latex(pred))
    print({**score(refs, hyps), "failures": failures})


if __name__ == "__main__":
    main()
