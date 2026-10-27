# scribe2tex — Handwritten Math → LaTeX

Image of a handwritten (or printed) math expression in, LaTeX string out.
CNN encoder (ResNet-18) + Bahdanau attention + LSTM decoder, trained from scratch and
**benchmarked against pix2tex** (a modern ViT + transformer-decoder model).

```
image ──► ResNet-18 encoder ──► (H'·W', 256) feature grid
                                      │
                        Bahdanau attention ◄─┐ per decode step
                                      │       │
                                 LSTM decoder ─┘ ──► \frac{x^2}{y} + z
```

## Why this architecture (the interview answer)
The classic CNN+attention+LSTM approach is built here *deliberately*: pix2tex already
occupies the "modern transformer" slot, so this repo owns the **classic-vs-modern
comparison**. The RNN version is the one you can derive on a whiteboard, and ResNet-18's
pretrained conv bias is the right call for a capped training set (a ViT would be
data-starved here).

## Two-stage training
- **Stage 1 — pretrain** on im2latex-100k (printed). From-scratch converges well here.
- **Stage 2 — fine-tune** on a **~40k subset** of MathWriting-human (handwritten). The HF
  mirror ships pre-rendered images, so no InkML rasterization is needed; the 400k
  synthetic split is skipped by design. Warm-starts from the stage-1 checkpoint (LR 1e-4)
  and reuses the stage-1 vocabulary so the output layer never resizes.
- **Eval** on each dataset's own test split (handwriting = MathWriting test). CROHME
  (InkML, needs rasterizing) is optional future work for cross-paper comparability.

Quality detail: images are **aspect-ratio-preserved** (scaled to a fixed height, padded —
never squished), and attention is **pad-masked** so it only ever attends to real ink.

## Results
| Model | Test set | BLEU | ExpRate | ExpRate ≤1 | ExpRate ≤2 |
|-------|----------|------|---------|-----------|-----------|
| Ours (ResNet18+Bahdanau+LSTM) | im2latex (printed)¹ | **91.5** | 0.47 | 0.55 | 0.70 |
| Ours (fine-tuned) | MathWriting (handwritten)¹ | **79.4** | 0.32 | 0.51 | 0.65 |
| pix2tex (baseline) | MathWriting (handwritten) | – | – | – | – |

¹ Estimated on 500-sample test subsets; run full splits for final figures.
The printed→handwritten gap is the expected story — handwriting is genuinely harder (BLEU 91.5→79.4, exact-match 47%→32%). Fine-tuning lifted handwritten val token-accuracy from 0.52 to **0.82**; the stage-1 vocab already covered **99.9%** of MathWriting tokens, so no retokenizing was needed. pix2tex comparison row is pending.

## Layout
```
model/    config, tokenizer, dataset, model (enc+attn+dec+seq2seq), train, inference (beam), evaluate
scripts/  download_data.sh (im2latex), download_mathwriting.py (handwritten), build_vocab.py
tests/    test_tokenizer.py (runs anywhere), test_model.py (needs torch)
api/      FastAPI (/predict,/render,/health) + Gradio demo over one predict() core
hmer/     TAMER adapter: manifests -> pinned upstream TAMER, train + evaluate
```

## Run
```bash
pip install -r requirements.txt

# --- Stage 1: printed ---
bash scripts/download_data.sh
python scripts/build_vocab.py data/im2latex/train.jsonl
python -m model.train --train data/im2latex/train.jsonl --val data/im2latex/val.jsonl
python -m model.evaluate --manifest data/im2latex/test.jsonl --ckpt checkpoints/pretrain_best.pth --limit 500

# --- Stage 2: handwritten (warm-start from stage 1) ---
python scripts/download_mathwriting.py --train-cap 40000 --val-cap 4000
python -m model.train --stage finetune --init-from checkpoints/pretrain_best.pth \
    --train data/mathwriting/train.jsonl --val data/mathwriting/val.jsonl --epochs 15
python -m model.evaluate --manifest data/mathwriting/test.jsonl --ckpt checkpoints/finetune_best.pth --limit 500

# tests
python tests/test_tokenizer.py   # pure python, runs now
python tests/test_model.py       # needs torch (GPU box / Colab / Kaggle)
```

## Stronger model: TAMER (DenseNet + tree-aware Transformer), `hmer/`

A second, near-state-of-the-art model built on an existing open-source implementation rather than
written from scratch. The ResNet-18 + LSTM model above stays as it is.

**Why TAMER.** Compared on 2026-10-27:

| Repo | Licence | HME100K config | HME100K ExpRate (paper) | Released weights | Notes |
|---|---|---|---|---|---|
| [BTTR](https://github.com/Green-Wood/BTTR) | MIT | no | 64.1 | CROHME only | oldest of the family |
| [CoMER](https://github.com/Green-Wood/CoMER) | none | no | 68.12 | CROHME only | last commit 2022 |
| [ICAL](https://github.com/qingzhenduyu/ICAL) (ICDAR 2024) | none | yes | 69.06 | HME100K + CROHME | same code base, one step older |
| [TAMER](https://github.com/qingzhenduyu/TAMER) (AAAI 2025) | none | **yes** | **69.50** | **HME100K + CROHME** | CoMER + tree-structure head |

TAMER has the best published HME100K score of the family, and like ICAL it releases HME100K
weights, so fine-tuning starts at about 69% on photos instead of from zero. Upstream trained it
with batches of 8 on 11 GB GPUs, so it fits a 16 GB T4. It has no licence file, so it is **not
vendored**: `scripts/fetch_tamer.sh` clones it into `third_party/TAMER` (gitignored) at a pinned
commit. Upstream targets PyTorch 1.8 + Lightning 1.4, so `hmer/` uses only its `nn.Module`s and
supplies its own training loop, data pipeline and scoring.

**What the adapter does.** `hmer/data.py` reads the same JSONL manifests (`{"image", "latex"}`).
HME100K labels are already tokenized in TAMER's dictionary and pass through unchanged. Raw LaTeX
from MathWriting goes through `model/tokenizer.py`. Tokens TAMER has never seen are added to the
end of its dictionary, so the pretrained ids stay where they are. TAMER has no `<unk>`, so its
own dictionary replaces `Vocab`. `hmer/evaluate.py` scores exactly like `model/evaluate.py`: both
sides are re-tokenized with `tokenize_latex`, and BLEU, ExpRate, ≤1 and ≤2 use the same formulas,
so the numbers can sit in the same table.

```bash
bash scripts/fetch_tamer.sh
pip install einops pytorch-lightning torchmetrics opencv-python-headless
# zero-shot: released HME100K weights (--invert / --no-invert: see Kaggle cell 4)
python -m hmer.evaluate --manifest data/hme100k/test.jsonl --ckpt pretrained --no-invert --limit 500
# fine-tune on photos + handwriting (resumable: --resume checkpoints/tamer_mix_last.pth)
python -m hmer.train --train data/mix_train.jsonl --val data/mix_val.jsonl --init pretrained --tag tamer_mix
python -m hmer.evaluate --manifest data/mathwriting/test.jsonl --ckpt checkpoints/tamer_mix_best.pth --limit 200
python -m pytest tests/test_hmer_adapter.py     # adapter only, no TAMER needed
```

## Demo & API
One `predict()` core (`api/predict.py`), two thin wrappers over it.

**Weights:** `checkpoints/vocab.json` is in the repo; the checkpoints live on the
[v0.1.0 release](https://github.com/unstablerandomwalk/scribe2tex/releases/tag/v0.1.0):
`mwfull20_best.pth` (full MathWriting, 20 epochs; demo default), `hme100k_best.pth`
(phone photos), `finetune_best.pth` (handwriting), `pretrain_best.pth` (printed).
```bash
gh release download v0.1.0 -D checkpoints
```

**Gradio demo** — upload an image → LaTeX + rendered preview + confidence:
```bash
export MODEL_PATH=checkpoints/finetune_best.pth   # handwriting model (or pretrain_best.pth for printed)
python -m api.app        # local; set share=True in app.py for a temporary public link on Colab
```

**FastAPI** — `/predict`, `/render`, `/health`:
```bash
uvicorn api.main:app --reload
curl -F file=@sample.png http://127.0.0.1:8000/predict
```

**Docker:**
```bash
docker build -t scribe2tex .
docker run -p 8000:8000 -v $PWD/checkpoints:/app/checkpoints scribe2tex
```

**Deploy to Hugging Face Spaces** (permanent public demo): create a Gradio Space, push this
repo, add `app_file: api/app.py` to the Space README frontmatter, and place
`finetune_best.pth` + `vocab.json` under `checkpoints/` (git-lfs).

## Known simplifications
- Best checkpoint is picked on val cross-entropy (cheap); BLEU/ExpRate reported on the
  saved best via `evaluate.py`.
- Stage-2 reuses the stage-1 vocab; MathWriting tokens outside it map to `<UNK>`
  (`download_mathwriting.py` reports the coverage).
- Backend (FastAPI + Gradio) and a brace-balancing post-process are the next increments.
