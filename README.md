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

## Scope (honest)
- **Stage 1 — pretrain** on im2latex-100k (printed). From-scratch converges well here → BLEU.
- **Stage 2 — fine-tune** on a **~40k subset** of MathWriting-human (rasterized from InkML;
  the 400k synthetic split is skipped by design).
- **Eval** on CROHME test → ExpRate, next to pix2tex on the same sets.
- Backend is a single **synchronous FastAPI** service + one Dockerfile. Celery/Redis/Postgres
  are intentionally omitted — under real concurrent load I'd move inference to a task queue;
  a single-user demo doesn't need it.

## Results (fill with real numbers)
| Model | Test set | BLEU | ExpRate | ExpRate ≤1 | ExpRate ≤2 |
|-------|----------|------|---------|-----------|-----------|
| Ours (ResNet18+Bahdanau+LSTM) | im2latex test | – | – | – | – |
| Ours | CROHME 2019 | – | – | – | – |
| pix2tex (baseline) | CROHME 2019 | – | – | – | – |

## Layout
```
model/   config, tokenizer, dataset, model (enc+attn+dec+seq2seq), train, inference (beam), evaluate
api/     FastAPI + Gradio over one predict() core   (next increment)
scripts/ download_data.sh, build_vocab.py, rasterize_inkml.py (next)
tests/   test_tokenizer.py (runs anywhere), test_model.py (needs torch)
```

## Run
```bash
pip install -r requirements.txt

# data + vocab
bash scripts/download_data.sh
python scripts/build_vocab.py data/im2latex/train.jsonl        # add mathwriting train when ready

# stage 1: pretrain on printed
python -m model.train --train data/im2latex/train.jsonl --val data/im2latex/validation.jsonl

# stage 2: fine-tune on handwritten (warm-start from stage 1)
python -m model.train --stage finetune --init-from checkpoints/pretrain_best.pth \
    --train data/mathwriting/train.jsonl --val data/mathwriting/val.jsonl --epochs 15

# evaluate (same command runs pix2tex-side comparison manually)
python -m model.evaluate --manifest data/crohme/test.jsonl --ckpt checkpoints/finetune_best.pth

# tests
python tests/test_tokenizer.py     # pure python, runs now
python tests/test_model.py         # needs torch (Colab)
```

## Known simplifications
- Best checkpoint is picked on val cross-entropy (cheap); BLEU/ExpRate reported on the
  saved best via `evaluate.py`.
- Batch images are zero-padded to a common width (no attention pad-mask yet).
- MathWriting/CROHME InkML rasterization is the next increment.
