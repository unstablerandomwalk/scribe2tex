"""Download MathWriting-human (already rendered images) -> PNGs + JSONL manifests.

The HF mirror ships pre-rendered RGB images + LaTeX (no InkML to rasterize). We stream
so a capped subset doesn't pull the full ~250k-image dataset, and cap the train split
per the project scope (default 40k; the 400k synthetic split is skipped by design).

    python scripts/download_mathwriting.py --train-cap 40000 --val-cap 4000

Reuses the stage-1 im2latex vocab (so the fine-tune warm-starts without resizing the
output layer) and reports how well that vocab covers MathWriting's tokens.
"""
import argparse
import json
import sys
from collections import Counter
from itertools import islice
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset                       # noqa: E402
from tqdm import tqdm                                   # noqa: E402

from model.config import DATA_DIR, VOCAB_PATH           # noqa: E402
from model.tokenizer import Vocab, tokenize_latex       # noqa: E402

OUT = DATA_DIR / "mathwriting"


def dump_split(name: str, cap: int) -> Path:
    """Stream `name` split, save up to `cap` grayscale PNGs + a JSONL manifest."""
    ds = load_dataset("deepcopy/MathWriting-human", split=name, streaming=True)
    rows = islice(ds, cap) if cap else ds
    splits = ds.info.splits or {}                         # streaming may not know sizes
    total = cap or (splits[name].num_examples if name in splits else None)
    img_dir = OUT / name
    img_dir.mkdir(parents=True, exist_ok=True)
    manifest = OUT / f"{name}.jsonl"
    n = 0
    with manifest.open("w") as f:
        for i, row in enumerate(tqdm(rows, total=total, desc=name, unit="img")):
            path = img_dir / f"{i:07d}.png"
            row["image"].convert("L").save(path)
            f.write(json.dumps({"image": str(path.resolve()), "latex": row["latex"]}) + "\n")
            n = i + 1
    print(f"{name}: {manifest} ({n} rows)")
    return manifest


def coverage_report(manifest: Path) -> None:
    """How many MathWriting tokens fall outside the reused im2latex vocab?"""
    known = set(Vocab.load(VOCAB_PATH).stoi)
    counter: Counter = Counter()
    for line in manifest.read_text().splitlines():
        if line.strip():
            counter.update(tokenize_latex(json.loads(line)["latex"]))
    total = sum(counter.values())
    oov = {t: c for t, c in counter.items() if t not in known}
    oov_count = sum(oov.values())
    pct = 100 * (1 - oov_count / max(1, total))
    print(f"\nvocab coverage: {pct:.2f}% of MathWriting tokens are in the stage-1 vocab "
          f"({len(oov)} distinct OOV types -> <UNK>)")
    top = sorted(oov.items(), key=lambda kv: -kv[1])[:20]
    if top:
        print("top OOV tokens:", top)
    if pct < 98:
        print("NOTE: coverage < 98% -- tell Claude; may be worth expanding the vocab.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-cap", type=int, default=40000)
    ap.add_argument("--val-cap", type=int, default=4000)
    ap.add_argument("--test-cap", type=int, default=0, help="0 = full test split")
    args = ap.parse_args()
    # ponytail: streaming takes the FIRST N rows (not random) -- fine for a fine-tune
    # subset; switch to .shuffle(buffer_size=...) if class balance ever looks off.
    train_manifest = dump_split("train", args.train_cap)
    dump_split("val", args.val_cap)
    dump_split("test", args.test_cap)
    coverage_report(train_manifest)


if __name__ == "__main__":
    main()
