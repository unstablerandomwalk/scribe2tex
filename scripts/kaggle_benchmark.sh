#!/usr/bin/env bash
# Paper group B on Kaggle: pix2tex on im2latex + HME100K, our model on the same HME100K
# test slice, split sizes, HME100K length stats. Needs the cutedeadu/hme100k input.
# Results -> /kaggle/working/benchmark_results.txt
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=/kaggle/working/benchmark_results.txt
: > "$OUT"
log() { echo "$@" | tee -a "$OUT"; }

pip install -q datasets nltk gdown virtualenv
# Kaggle's python has no ensurepip, so `python -m venv` fails; virtualenv doesn't need it
# (a half-made venv from a failed `python -m venv` has python but no pip: rebuild it)
[ -x /kaggle/working/p2t/bin/pip ] || { rm -rf /kaggle/working/p2t; python -m virtualenv -q /kaggle/working/p2t; }
/kaggle/working/p2t/bin/python -c "import pix2tex" 2>/dev/null \
    || /kaggle/working/p2t/bin/pip install -q pix2tex nltk tqdm
/kaggle/working/p2t/bin/python -c "import pix2tex, PIL; print('pix2tex venv ok')" || { log "pix2tex venv broken"; exit 1; }
python scripts/kaggle_fetch_checkpoints.py

log "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
[ -s data/im2latex/test.jsonl ] || bash scripts/download_data.sh
# .complete marks a finished build; an interrupted earlier run leaves a short all.jsonl
if [ ! -f data/hme100k/.complete ]; then
rm -rf data/hme100k
python - <<'PY'
import random, subprocess
from pathlib import Path
hme = next(Path("/kaggle/input").rglob("train.txt")).parent
print("HME100K found at", hme)
subprocess.run(["python", "scripts/prepare_hme100k.py", "--images-dir", str(hme),
                "--label-file", str(hme / "train.txt"), "--out", "data/hme100k/all.jsonl"],
               check=True)
rows = Path("data/hme100k/all.jsonl").read_text().splitlines()
random.Random(0).shuffle(rows)  # same seed-0 split as stage 3
for name, part in {"test": rows[:2000], "val": rows[2000:4000], "train": rows[4000:]}.items():
    Path(f"data/hme100k/{name}.jsonl").write_text("\n".join(part) + "\n")
Path("data/hme100k/.complete").touch()
PY
fi
log "== split sizes"; wc -l data/im2latex/*.jsonl data/hme100k/*.jsonl | tee -a "$OUT"

log "== pix2tex, im2latex (first 500)"
/kaggle/working/p2t/bin/python scripts/eval_pix2tex.py --manifest data/im2latex/test.jsonl --limit 500 | tee -a "$OUT"
log "== pix2tex, HME100K test"
/kaggle/working/p2t/bin/python scripts/eval_pix2tex.py --manifest data/hme100k/test.jsonl | tee -a "$OUT"
log "== ours (hme100k_best), HME100K test"
python -m model.evaluate --manifest data/hme100k/test.jsonl --ckpt checkpoints/hme100k_best.pth | tee -a "$OUT"
log "== HME100K test length: n mean median min max"
python -c "import json,statistics as s;from model.tokenizer import tokenize_latex as t;L=[len(t(json.loads(l)['latex'])) for l in open('data/hme100k/test.jsonl')];print(len(L),s.mean(L),s.median(L),min(L),max(L))" | tee -a "$OUT"
log "== DONE"
