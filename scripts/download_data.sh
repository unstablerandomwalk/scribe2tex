#!/usr/bin/env bash
# Download im2latex-100k from HuggingFace -> image files + JSONL manifests.
# Manifest line: {"image": "<abs path>.png", "latex": "<latex>"}
# MathWriting/CROHME are InkML (stroke data) -> rasterize separately (next increment:
# scripts/rasterize_inkml.py, following Google's MathWriting Colab).
set -euo pipefail
cd "$(dirname "$0")/.."
python - <<'PY'
import json
from pathlib import Path
from datasets import load_dataset

OUT = Path("data/im2latex"); OUT.mkdir(parents=True, exist_ok=True)
ds = load_dataset("yuntian-deng/im2latex-100k")   # splits: train / validation / test
for split in ds:
    img_dir = OUT / split; img_dir.mkdir(exist_ok=True)
    manifest = OUT / f"{split}.jsonl"
    with manifest.open("w") as f:
        for i, row in enumerate(ds[split]):
            img = row["image"]                        # PIL.Image
            latex = row.get("formula") or row.get("latex") or row.get("text")
            path = img_dir / f"{i:07d}.png"
            img.save(path)
            f.write(json.dumps({"image": str(path.resolve()), "latex": latex}) + "\n")
    print(f"{split}: {manifest} ({len(ds[split])} rows)")
PY
echo "im2latex done. Next: build vocab over the train manifest(s)."
