# scribe2tex — Session Handoff / Project State

> Attach this file at the START of any new Claude Code session
> (`@/Users/hardikkhaneja/scribe2tex/HANDOFF.md`) so it starts with full context
> instead of cold-starting. Keep it updated as the project moves.

## What this is
Portfolio project (goal: impress ML interviewers). Image of handwritten/printed math → LaTeX.
From-scratch **ResNet-18 encoder + Bahdanau attention + LSTM decoder + beam search**,
deliberately the "classic" model so it can be **benchmarked vs pix2tex** (the modern transformer).

- Repo: https://github.com/unstablerandomwalk/scribe2tex  (public)
- Local: `~/scribe2tex`
- Owner runs GPU work on **Kaggle** (30h/week; use *Save & Run All → Commit* to avoid idle timeout).

## Where things live (the ML / NN / backend map)
- **Neural network** — `model/model.py`: `Encoder` (ResNet-18 CNN), `BahdanauAttention`,
  `Decoder` (LSTM+embedding+attention), `Seq2Seq`. Pad-masked attention.
- **ML pipeline** — `model/train.py` (two-stage transfer learning, teacher forcing, CE+attn-reg
  loss, grad clip, LR sched, checkpointing, `--tag` for checkpoint names), `dataset.py`
  (aspect-preserving resize to height 128 + pad, never squish; valid_hw pad masks), `tokenizer.py`
  (LaTeX regex tokenizer + Vocab), `inference.py` (beam search + confidence), `evaluate.py`
  (BLEU/ExpRate), `config.py` (hyperparams).
- **Backend** — `api/predict.py` (shared core: checkpoint→beam search), `api/main.py`
  (FastAPI /predict,/render,/health), `api/app.py` (Gradio demo), `Dockerfile`. `api/__init__.py`
  registers pillow-heif so iPhone .HEIC opens.
- **Data scripts** — `scripts/download_data.sh` (im2latex), `download_mathwriting.py`,
  `prepare_hme100k.py`, `build_vocab.py`.

## Key decisions (already settled — don't relitigate)
- Height **128**, aspect ratio ALWAYS preserved (pad, never squish); attention pad-masked. `config.py`.
- **One vocab** (429 tokens from im2latex) REUSED for all fine-tunes → warm-start never resizes the
  output layer. MathWriting coverage was 99.9%.
- Backend is **sync FastAPI + Gradio only** — Celery/Redis/Postgres deliberately cut (documented as
  "future work under load").
- Checkpoints named per stage via `--tag`: `pretrain_*`, `finetune_*` (MathWriting), `hme100k_*` (photos).

## Results (500-sample test subsets)
| Stage | Checkpoint | Test set | BLEU | ExpRate | ≤1 | ≤2 |
|---|---|---|---|---|---|---|
| 1 pretrain | `pretrain_best.pth` | im2latex (printed) | **91.5** | 0.47 | 0.55 | 0.70 |
| 2 finetune | `finetune_best.pth` | MathWriting (handwritten) | **79.4** | 0.32 | 0.51 | 0.65 |
| 3 hme100k | `hme100k_best.pth` | HME100K (photos) | **88.3** | 0.524 | 0.724 | 0.832 |

Stage-2 val token-accuracy peaked ~0.82. Stage-3 val token-acc peaked ~0.836 (ep13, val_loss 1.866).
NOTE: the `cutedeadu/hme100k` Kaggle mirror ships only `train.txt` (no test captions), so stage-3
train/val/test were carved 3-ways from it (~70k/2k/2k). The HME100K row is on that self-carved 2k
held-out slice (in-distribution), NOT the official 24,607-image HME100K test set — say so on the résumé.

## Checkpoints
- Google Drive folder (shared "anyone with link"): https://drive.google.com/drive/folders/1nSJGLWYajiinlNDYWbBibtLpdbL4tyr0
  Contains pretrain_best/last, finetune_best, vocab.json (~142MB each). NOTE: folder also holds
  a big `hf_cache/` — gdown pulls everything, so flatten and keep only *.pth + vocab.json.
- Also local at `~/scribe2tex/checkpoints/`.

## How to run
**Local demo (Mac):** Mac Python is 3.14 (NO torch). Use the prebuilt venv:
```
cd ~/scribe2tex && .venv/bin/python -m api.app        # http://127.0.0.1:7860
```
`.venv` = Python 3.12 with torch 2.13, torchvision, gradio, matplotlib, pillow-heif.
`MODEL_PATH` env picks the checkpoint (default `checkpoints/finetune_best.pth`).

**Kaggle training/eval (canonical flow):**
```
!git clone https://github.com/unstablerandomwalk/scribe2tex.git && cd scribe2tex
!pip install -q datasets nltk gdown
# pull checkpoints from Drive + flatten:
!gdown --folder "https://drive.google.com/drive/folders/1nSJGLWYajiinlNDYWbBibtLpdbL4tyr0" -O checkpoints
# (python: move any nested files up into checkpoints/)
!bash scripts/download_data.sh                 # im2latex (split is 'val', NOT 'validation')
!python scripts/download_mathwriting.py --train-cap 40000 --val-cap 4000   # pre-rendered, streamed
!python -m model.train --train data/im2latex/train.jsonl --val data/im2latex/val.jsonl        # stage 1
!python -m model.train --stage finetune --init-from checkpoints/pretrain_best.pth \
    --train data/mathwriting/train.jsonl --val data/mathwriting/val.jsonl --epochs 30          # stage 2
!python -m model.evaluate --manifest data/mathwriting/test.jsonl --ckpt checkpoints/finetune_best.pth --limit 500
```

## In progress / TODO (next session picks these up)
1. ~~**Stage-3 HME100K fine-tune**~~ ✅ DONE (BLEU 88.3 / ExpRate 0.524). Mirror layout was
   `/kaggle/input/datasets/cutedeadu/hme100k/{images/,train.txt}`; labels carry the `images/` prefix
   so `--images-dir` is the parent, not `images/`. Run was resumable via
   `$([ -f checkpoints/hme100k_last.pth ] && echo --resume checkpoints/hme100k_last.pth)`.
   TODO: download `hme100k_best.pth` from Kaggle output → Drive + local `checkpoints/` for the demo.
2. **pix2tex benchmark** — fill the comparison rows (pip install pix2tex, run on same 500 test images).
3. **Deploy Gradio to Hugging Face Spaces** — permanent public link (`app_file: api/app.py`).
4. **Résumé bullets** with the real numbers.
5. Optional: photo preprocessing front-end (crop-to-content + Otsu binarize), brace-balancing
   post-process, full-test-set eval for official numbers.

## Known limitations
- **Raw phone photos fail** (e.g. blue pen on grid paper, uncropped) — domain gap: trained on clean
  renders. Confidence score correctly drops (~0.49) on OOD input. Fix = stage-3 HME100K (+preprocessing).
- Best checkpoint chosen on val cross-entropy (cheap proxy); BLEU/ExpRate reported separately.

## Working conventions
- Ponytail mode: lazy/minimal, shortest working change, understand before editing.
- Commits: `type: description`; NO `Co-Authored-By` trailer.
- New files created via bash heredoc (the Write tool is gated in this environment and doubles cost).
- Data (`data/`) and `checkpoints/*.pth` are gitignored — never committed.
