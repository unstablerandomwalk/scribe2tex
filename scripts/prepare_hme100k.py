"""Prepare HME100K (real handwriting photos on paper) -> JSONL manifests {image, latex}.

HME100K ships as image folder(s) + a caption/label file whose lines are
`<image_name> <space-separated LaTeX tokens>` (tab- or space-delimited); the exact
folder layout varies by mirror. Attach the Kaggle dataset (Add Input -> cutedeadu/hme100k),
inspect it, then point this at the real paths:

    ls -R /kaggle/input/hme100k | head        # find the images dir + caption file
    python scripts/prepare_hme100k.py \
        --images-dir /kaggle/input/hme100k/train_images \
        --label-file /kaggle/input/hme100k/train_caption.txt \
        --out data/hme100k/train.jsonl --cap 40000

Reuses the stage-1 vocab and reports token coverage (photos are still handwriting, so
coverage should be high).
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tqdm import tqdm                                   # noqa: E402

from model.config import VOCAB_PATH                    # noqa: E402
from model.tokenizer import Vocab, tokenize_latex      # noqa: E402


def parse_label_line(line: str):
    """'img.jpg tok tok' or 'img.jpg\\ttok tok' -> ('img.jpg', 'tok tok')."""
    line = line.rstrip("\n")
    if "\t" in line:
        name, latex = line.split("\t", 1)
    else:
        parts = line.split(" ", 1)
        name, latex = parts[0], (parts[1] if len(parts) > 1 else "")
    return name.strip(), latex.strip()


def build(images_dir: str, label_file: str, out: str, cap: int) -> Path:
    images_dir = Path(images_dir)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    written = missing = 0
    with out.open("w") as f:
        lines = Path(label_file).read_text().splitlines()
        for line in tqdm(lines, desc="checking images", unit="img"):
            if not line.strip():
                continue
            name, latex = parse_label_line(line)
            if not latex:
                continue
            img = images_dir / name
            if not img.exists():
                missing += 1
                continue
            f.write(json.dumps({"image": str(img.resolve()), "latex": latex}) + "\n")
            written += 1
            if cap and written >= cap:
                break
    print(f"{out}: {written} rows written ({missing} images not found)")
    if missing and not written:
        print("ALL images missing -- check --images-dir matches the caption file's names.")
    return out


def coverage(manifest: Path) -> None:
    known = set(Vocab.load(VOCAB_PATH).stoi)
    counter: Counter = Counter()
    for line in Path(manifest).read_text().splitlines():
        if line.strip():
            counter.update(tokenize_latex(json.loads(line)["latex"]))
    total = sum(counter.values())
    oov = {t: n for t, n in counter.items() if t not in known}
    pct = 100 * (1 - sum(oov.values()) / max(1, total))
    print(f"vocab coverage: {pct:.2f}% ({len(oov)} distinct OOV types -> <UNK>)")
    top = sorted(oov.items(), key=lambda kv: -kv[1])[:20]
    if top:
        print("top OOV tokens:", top)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--label-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap", type=int, default=0, help="0 = all rows")
    args = ap.parse_args()
    coverage(build(args.images_dir, args.label_file, args.out, args.cap))


if __name__ == "__main__":
    main()
