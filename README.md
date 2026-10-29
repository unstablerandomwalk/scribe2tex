# scribe2tex — Handwritten Math → LaTeX

Image of a handwritten (or printed) math expression in, LaTeX string out.
CNN encoder (ResNet-18) + Bahdanau attention + LSTM decoder, trained in three stages
(printed → handwritten → phone photos) and **benchmarked against pix2tex** (a modern ViT + transformer-decoder model).

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

## Three-stage training
One 429-token vocabulary (built on im2latex) is shared by every stage, so each stage
warm-starts from the previous best checkpoint without resizing the output layer.
- **Stage 1: printed.** im2latex-100k (55,033 / 6,072 / 6,810 train/val/test), ImageNet-initialised ResNet-18, LR 1e-3.
- **Stage 2: handwritten.** A 40k subset of MathWriting-human (pre-rendered images from the HF mirror),
  LR 1e-4. The stage-1 vocab covers 99.9% of its tokens. A second run on the full human split
  (229,864 images, 20 epochs) gives `mwfull20_best.pth`, the demo default.
- **Stage 3: phone photos.** HME100K, self-split (seed 0) into train / 2,000 val / 2,000 test.
  The stage-1 vocab covers 98.27% of its tokens.

Images are **aspect-ratio-preserved** (scaled to a fixed height, padded, never squished), and
attention is **pad-masked** so it only ever attends to real ink. The best checkpoint per stage
is picked on val cross-entropy; BLEU/ExpRate come from `evaluate.py` on that checkpoint.

## Results
All rows are scored with one tokenizer and one metric implementation (`model/metrics.py`).
Test sets are the first 500 images of each split, except HME100K (2,000-image slice).

| Model | Test set | BLEU | ExpRate | ExpRate ≤1 | ExpRate ≤2 |
|-------|----------|------|---------|-----------|-----------|
| Ours | im2latex (printed) | **91.5** | **0.470** | 0.550 | 0.700 |
| pix2tex (off the shelf) | im2latex (printed) | 87.3 | 0.352 | 0.422 | 0.524 |
| Ours, 40k MathWriting | MathWriting (handwritten) | 79.4 | 0.320 | 0.510 | 0.650 |
| Ours, full MathWriting | MathWriting (handwritten) | **86.7** | **0.478** | 0.680 | 0.794 |
| pix2tex (off the shelf) | MathWriting (handwritten) | 0.7 | 0.000 | 0.000 | 0.000 |
| Ours, stage 3 | HME100K (photos) | **88.3** | **0.524** | 0.724 | 0.832 |
| pix2tex (off the shelf) | HME100K (photos)¹ | 4.0 | 0.000 | 0.0005 | 0.003 |

¹ Re-built slice from the 99,109-row label file; it may not match the slice behind our row.
Our model scores 92.0 / 0.618 on the re-built slice, read as an upper bound (likely train overlap).

**Ablation** (MathWriting, 10-epoch fine-tunes from stage 1, rows cumulative, same 500 test images):

| Variant | BLEU | ExpRate | ≤1 | ≤2 |
|---|---|---|---|---|
| Baseline (stride 32, no coverage, light aug) | 77.0 | 0.256 | 0.472 | 0.616 |
| + fine grid (stride 16) | 76.7 | 0.280 | 0.470 | 0.582 |
| + coverage attention | 76.5 | 0.256 | 0.444 | 0.574 |
| + photo-style augmentation | 78.5 | 0.270 | 0.486 | 0.628 |

Every gap is inside the ±4-point 95% interval of a 500-image ExpRate, single seed: none of the
three changes hurts, none is a clear win yet.

**Confidence.** Keeping only the most confident 20% of predictions lifts ExpRate from 0.478 to
0.870 (full-MathWriting model), so the confidence score is useful for flagging bad outputs.

## Paper
IEEE-format write-up of the comparison, staged transfer, ablation and error analysis:
[`docs/paper/main.pdf`](docs/paper/main.pdf) (source `docs/paper/main.tex`, build with `tectonic main.tex`).

## Layout
```
model/    config, tokenizer, dataset, model (enc+attn+dec+seq2seq), train, inference (beam), evaluate, metrics
scripts/  download_data.sh (im2latex), download_mathwriting.py, prepare_hme100k.py, build_vocab.py,
          eval_pix2tex.py, attention_map.py, kaggle_fetch_checkpoints.py,
          kaggle_benchmark.sh (pix2tex comparison), kaggle_ablation.sh (ablation)
tests/    test_tokenizer.py, test_metrics.py (run anywhere), test_model.py (needs torch)
api/      FastAPI (/predict,/render,/health) + Gradio demo over one predict() core
docs/     paper/ (IEEE paper + figures), research-context.html
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
python -m model.evaluate --manifest data/mathwriting/test.jsonl --ckpt checkpoints/finetune_best.pth \
    --limit 500 --out preds.jsonl      # --out writes per-sample {image, ref, hyp, conf}

# --- Stage 3: phone photos (see prepare_hme100k.py docstring for paths) ---
python scripts/prepare_hme100k.py --images-dir ... --label-file ... --out data/hme100k/train.jsonl
python -m model.train --stage finetune --init-from checkpoints/finetune_best.pth --tag hme100k \
    --train data/hme100k/train.jsonl --val data/hme100k/val.jsonl

# --- Ablation switches (turn off one improvement) ---
python -m model.train ... --no-fine-grid --no-coverage --no-photo-aug

# --- Paper runs on Kaggle (GPU, Internet on, HF_TOKEN secret) ---
bash scripts/kaggle_benchmark.sh   # pix2tex comparison; needs the cutedeadu/hme100k input
bash scripts/kaggle_ablation.sh    # 4 ablation fine-tunes, resumable

# --- Attention figure ---
python scripts/attention_map.py --ckpt checkpoints/mwfull20_best.pth \
    --image data/mathwriting/test/0000001.png --out docs/paper/figs/att.png

# tests
python tests/test_tokenizer.py
python tests/test_metrics.py
python tests/test_model.py       # needs torch
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
# optional: export MODEL_PATH=checkpoints/hme100k_best.pth  (default mwfull20_best.pth)
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
`mwfull20_best.pth` + `vocab.json` under `checkpoints/` (git-lfs).

## Known limitations
- Scores are on 500-image test subsets, not full splits; the HME100K test set is a self-carved slice,
  not the official one. No CROHME evaluation yet.
- Best checkpoint is picked on val cross-entropy, not ExpRate.
- Sequential fine-tuning forgets earlier domains; joint multi-domain training is future work.
- Tokens outside the stage-1 vocab map to `<UNK>` (the download/prep scripts report coverage).
- pix2tex is evaluated off the shelf, not fine-tuned on handwriting.
