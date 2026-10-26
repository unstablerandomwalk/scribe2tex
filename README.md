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
